"""HTTP client and CLIP proxy for the Remote CLIP Colab runtime (protocol v3).

The packing helpers mirror colab/packing.py (PACKING_VERSION must match on
both sides); floats travel as fp16 by fixed policy.
"""
import http.client
import json
import struct
import time
import urllib.error
import urllib.request

import torch

PACKING_VERSION = 3
LEN_FMT = ">Q"
LEN_SIZE = struct.calcsize(LEN_FMT)
MAX_META_BYTES = 8 * 1024 * 1024
MAX_BLOB_BYTES = 512 * 1024 * 1024
SOCKET_TIMEOUT = 120
ENCODE_TIMEOUT = 600
GENERATE_TIMEOUT = 3600
ENCODE_RETRIES = 4
POLL_MAX_FAILURES = 5
PACKED_CONTENT_TYPE = "application/x-rcp-v3"

ALLOWED_DTYPES = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def log(msg):
    print(f"[RemoteCLIPColab {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# packing (mirror of colab/packing.py)
# ---------------------------------------------------------------------------

def pack_tensors(tensors):
    manifest = {}
    blobs = []
    offset = 0
    for name, t in tensors.items():
        if t is None:
            raise ValueError(f"Tensor '{name}' is None")
        t = t.detach().cpu()
        orig_dtype = str(t.dtype)
        if t.is_floating_point() and t.dtype != torch.float16:
            t = t.to(torch.float16)
        t = t.contiguous()
        raw = t.numpy().tobytes()
        manifest[name] = {
            "dtype": str(t.dtype),
            "orig_dtype": orig_dtype,
            "shape": list(t.shape),
            "offset": offset,
            "size": len(raw),
        }
        offset += len(raw)
        blobs.append(raw)
    return manifest, b"".join(blobs)


def unpack_tensors(manifest, blob):
    out = {}
    view = memoryview(blob)
    for name, info in manifest.items():
        dtype_name = info["dtype"]
        if dtype_name not in ALLOWED_DTYPES:
            raise ValueError(f"Refusing to decode disallowed dtype: {dtype_name}")
        dtype = ALLOWED_DTYPES[dtype_name]
        start = info["offset"]
        raw = bytearray(view[start:start + info["size"]])
        t = torch.frombuffer(raw, dtype=dtype).reshape(info["shape"]).clone()
        orig = info.get("orig_dtype")
        if orig in ALLOWED_DTYPES and ALLOWED_DTYPES[orig] != dtype:
            t = t.to(ALLOWED_DTYPES[orig])
        out[name] = t
    return out


def extract_tensors(obj, prefix, out):
    if isinstance(obj, torch.Tensor):
        out[prefix] = obj
        return {"__tensor__": prefix}
    if isinstance(obj, dict):
        return {k: extract_tensors(v, f"{prefix}.{k}", out) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [extract_tensors(v, f"{prefix}.{i}", out) for i, v in enumerate(obj)]
    return obj


def restore_tensors(obj, tensors):
    if isinstance(obj, dict):
        if len(obj) == 1 and "__tensor__" in obj:
            return tensors[obj["__tensor__"]]
        return {k: restore_tensors(v, tensors) for k, v in obj.items()}
    if isinstance(obj, list):
        return [restore_tensors(v, tensors) for v in obj]
    return obj


def frame(meta):
    """Build a packed request body; tensor leaves inside meta are packed."""
    tensor_inputs = {}
    meta = extract_tensors(meta, "meta", tensor_inputs)
    manifest, blob = pack_tensors(tensor_inputs)
    meta["tensors"] = manifest
    meta["blob_size"] = len(blob)
    meta_bytes = json.dumps(meta).encode("utf-8")
    return struct.pack(LEN_FMT, len(meta_bytes)) + meta_bytes + blob


def parse(body):
    if len(body) < LEN_SIZE:
        raise ValueError("Body too short for length prefix")
    meta_len = struct.unpack(LEN_FMT, body[:LEN_SIZE])[0]
    if meta_len > MAX_META_BYTES:
        raise ValueError(f"Meta too large: {meta_len} bytes")
    meta = json.loads(body[LEN_SIZE:LEN_SIZE + meta_len].decode("utf-8"))
    blob_size = meta.get("blob_size", 0)
    if blob_size > MAX_BLOB_BYTES:
        raise ValueError(f"Blob too large: {blob_size} bytes")
    blob = body[LEN_SIZE + meta_len:LEN_SIZE + meta_len + blob_size]
    if len(blob) != blob_size:
        raise ValueError("Body truncated: blob shorter than declared")
    tensors = unpack_tensors(meta.get("tensors", {}), blob)
    return meta, tensors


# ---------------------------------------------------------------------------
# REST client
# ---------------------------------------------------------------------------

class WorkerError(RuntimeError):
    pass


class RemoteCLIPClient:
    def __init__(self, base_url, auth_token=""):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token

    def _headers(self, extra=None):
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        if extra:
            headers.update(extra)
        return headers

    def _request(self, method, path, body=None, headers=None, timeout=SOCKET_TIMEOUT):
        url = f"{self.base_url}{path}"
        data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method,
                                     headers=self._headers(headers))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            message = detail
            try:
                parsed = json.loads(detail)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                # FastAPI errors use "detail"; worker 503s use {"error": {...}}
                message = parsed.get("detail") or parsed.get("error") or detail
                if isinstance(message, dict):
                    message = message.get("message") or json.dumps(message)
            raise WorkerError(f"{method} {path} -> HTTP {e.code}: {message}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.HTTPException) as e:
            raise WorkerError(f"{method} {path} -> {e}") from None

    def get_status(self):
        _status, body, _headers = self._request("GET", "/v1/status")
        return json.loads(body)

    def check_protocol(self):
        status = self.get_status()
        proto = status.get("proto")
        if proto != PACKING_VERSION:
            raise WorkerError(
                f"protocol mismatch: worker speaks v{proto}, client speaks v{PACKING_VERSION}")
        return status

    def list_loras(self):
        _status, body, _headers = self._request("GET", "/v1/loras")
        return json.loads(body).get("loras", [])

    def encode(self, tokens):
        """tokens: dict from RemoteCLIPProxy.tokenize -> result dict with tensors."""
        header = {
            "proto": PACKING_VERSION,
            "text": tokens["text"],
            "kwargs": tokens.get("kwargs", {}),
            "lora_stack": tokens.get("lora_stack", []),
        }
        body = frame(header)
        last_error = None
        for attempt in range(1, ENCODE_RETRIES + 1):
            try:
                _status, resp, _headers = self._request(
                    "POST", "/v1/encode", body,
                    headers={"Content-Type": PACKED_CONTENT_TYPE},
                    timeout=ENCODE_TIMEOUT)
                meta, tensors = parse(resp)
                if meta.get("error"):
                    raise WorkerError(f"remote worker error: {meta['error']}")
                return restore_tensors(meta["struct"], tensors)
            except WorkerError as e:
                if "HTTP 503" in str(e) and attempt < ENCODE_RETRIES:
                    last_error = e
                    log(f"encode retryable timeout, retrying ({attempt}/{ENCODE_RETRIES})")
                    time.sleep(2 * attempt)
                    continue
                raise
        raise last_error

    def generate(self, tokens, gen_kwargs):
        header = {
            "proto": PACKING_VERSION,
            "text": tokens["text"],
            "kwargs": tokens.get("kwargs", {}),
            "gen_kwargs": gen_kwargs,
            "lora_stack": tokens.get("lora_stack", []),
        }
        body = frame(header)
        _status, resp, _headers = self._request(
            "POST", "/v1/generate", body,
            headers={"Content-Type": PACKED_CONTENT_TYPE})
        job = json.loads(resp)
        job_id = job["job_id"]
        t0 = time.time()
        poll_failures = 0
        while True:
            try:
                _status, body, _hdr = self._request("GET", f"/v1/jobs/{job_id}")
            except WorkerError as e:
                if "HTTP 404" in str(e):
                    raise WorkerError(
                        f"job {job_id} not found on worker (record expired or worker "
                        "was restarted); generated text is lost") from None
                poll_failures += 1
                if poll_failures >= POLL_MAX_FAILURES:
                    raise
                log(f"job poll failed ({poll_failures}/{POLL_MAX_FAILURES}): {e}")
                time.sleep(min(2 * poll_failures, 5))
                continue
            poll_failures = 0
            state = json.loads(body)
            if state["state"] == "done":
                return state.get("text", "")
            if state["state"] == "error":
                raise WorkerError(f"remote generation failed: {state.get('error')}")
            if time.time() - t0 > GENERATE_TIMEOUT:
                try:
                    self._request("DELETE", f"/v1/jobs/{job_id}")
                except WorkerError as cleanup_error:
                    log(f"job cleanup after timeout failed: {cleanup_error}")
                raise WorkerError("remote generation timed out")
            time.sleep(1.0)

    # -- control plane helpers (used by RemoteCLIPController) ----------------

    def control(self, action, params=None):
        params = params or {}
        if action == "status":
            return self.get_status()
        if action == "list_models":
            _s, body, _h = self._request("GET", "/v1/models")
            return json.loads(body)
        if action == "list_loras":
            return {"loras": self.list_loras()}
        if action == "tunnel":
            _s, body, _h = self._request("GET", "/v1/tunnel")
            return json.loads(body)
        if action == "clear_cache":
            _s, body, _h = self._request("POST", "/v1/cache/clear", {})
            return json.loads(body)
        if action == "load_model":
            spec = {"name": params.get("model_name") or "",
                    "kind": params.get("kind") or ""}
            if params.get("source"):
                spec["source"] = params["source"]
            if params.get("dtype"):
                spec["dtype"] = params["dtype"]
            if params.get("clip_type"):
                spec["clip_type"] = params["clip_type"]
            sources = params.get("sources")
            if sources:
                if isinstance(sources, str):
                    stripped = sources.strip()
                    if stripped.startswith("["):
                        try:
                            sources = json.loads(stripped)
                        except ValueError as e:
                            raise WorkerError(f"sources must be valid JSON: {e}")
                    else:
                        sources = [s.strip() for s in stripped.split("+") if s.strip()]
                if not isinstance(sources, list) or not sources:
                    raise WorkerError("sources must be a non-empty list")
                spec["sources"] = sources
            if params.get("components"):
                comps = params["components"]
                if isinstance(comps, str):
                    try:
                        comps = json.loads(comps)
                    except ValueError as e:
                        raise WorkerError(f"components must be valid JSON: {e}")
                if not isinstance(comps, dict) or not comps:
                    raise WorkerError("components must be a non-empty JSON object")
                spec["components"] = comps
            if not spec["name"] or not spec["kind"]:
                raise WorkerError("load_model requires model_name and kind")
            _s, body, _h = self._request("POST", "/v1/models/load", spec)
            return json.loads(body)
        if action == "unload_model":
            if not params.get("model_name"):
                raise WorkerError("unload_model requires model_name")
            _s, body, _h = self._request("POST", "/v1/models/unload",
                                         {"name": params["model_name"]})
            return json.loads(body)
        if action == "set_default":
            if not params.get("model_name"):
                raise WorkerError("set_default requires model_name")
            _s, body, _h = self._request("POST", "/v1/models/default",
                                         {"name": params["model_name"]})
            return json.loads(body)
        if action == "shutdown":
            if not params.get("shutdown_confirm"):
                raise WorkerError(
                    "shutdown aborted: enable 'shutdown_confirm' on the "
                    "Remote CLIP Controller node, or POST /v1/server/shutdown "
                    "with {\"confirm\": true} directly")
            _s, body, _h = self._request("POST", "/v1/server/shutdown",
                                         {"confirm": True})
            return json.loads(body)
        raise WorkerError(f"unknown action: {action}")


# ---------------------------------------------------------------------------
# duck-typed CLIP proxy
# ---------------------------------------------------------------------------

class RemoteCLIPProxy:
    """Drop-in CLIP object backed by the remote worker over HTTP."""

    def __init__(self, base_url, auth_token="", client=None, lora_stack=None):
        self.base_url = base_url
        self.client = client or RemoteCLIPClient(base_url, auth_token)
        self.lora_stack = list(lora_stack or [])

    def clone(self, disable_dynamic=False):
        return RemoteCLIPProxy(self.base_url, client=self.client,
                               lora_stack=self.lora_stack)

    def with_lora(self, lora_name, strength_clip):
        c = self.clone()
        c.lora_stack = self.lora_stack + [[lora_name, float(strength_clip)]]
        return c

    def add_patches(self, *_args, **_kwargs):
        raise NotImplementedError(
            "Apply LoRAs to a remote CLIP with the 'LoraLoaderCLIPOnly' node, "
            "which forwards them to the worker.")

    def tokenize(self, text, return_word_ids=False, **kwargs):
        if return_word_ids:
            kwargs["return_word_ids"] = True
        return {"text": text, "kwargs": kwargs, "lora_stack": self.lora_stack}

    def _encode(self, tokens):
        log("Sending encode request")
        out = self.client.encode(tokens)
        log("Received embeddings")
        return out

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        out = self._encode(tokens)
        if return_dict:
            return out
        cond = out["cond"]
        if return_pooled:
            return cond, out.get("pooled_output")
        return cond

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict: dict = None,
                                     show_pbar=True):
        return_pooled = "unprojected" if unprojected else True
        pooled_dict = self.encode_from_tokens(tokens, return_pooled=return_pooled,
                                              return_dict=True)
        cond = pooled_dict.pop("cond")
        if add_dict:
            pooled_dict.update(add_dict)
        return [[cond, pooled_dict]]

    def generate(self, tokens, **gen_kwargs):
        log("Sending generate request")
        text = self.client.generate(tokens, gen_kwargs)
        log("Received generated text")
        return text

    def decode(self, generated):
        return generated
