"""FastAPI server for the Remote CLIP Colab runtime (protocol v3)."""
import asyncio
import hmac
import json
import os
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse

import engines as eng
import engines_native
import packing

JOB_RETENTION_SECONDS = 30 * 60
SAFETENSORS_MAGIC_MAX_HEADER = 64 * 1024 * 1024


def log(msg):
    print(f"[server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class AppState:
    def __init__(self, registry, tunnel=None, token="", encode_timeout=90.0):
        self.registry = registry
        self.tunnel = tunnel
        self.token = token
        self.encode_timeout = encode_timeout
        self.started = time.time()
        self.jobs = OrderedDict()
        self.jobs_lock = threading.Lock()
        self.job_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rcp-gen")
        self.encode_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rcp-enc")


STATE = None  # set by create_app


def create_app(registry, tunnel=None, token="", encode_timeout=90.0):
    global STATE
    STATE = AppState(registry, tunnel, token, encode_timeout)
    app = FastAPI(title="Remote CLIP Colab Worker", version=f"proto {eng.PROTOCOL_VERSION}")

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        open_paths = ("/health", "/docs", "/openapi.json", "/redoc")
        if STATE.token and not any(path == p or path.startswith(p + "/") for p in open_paths):
            header = request.headers.get("authorization", "")
            expected = f"Bearer {STATE.token}"
            if not hmac.compare_digest(header, expected):
                return JSONResponse(
                    {"error": {"code": "unauthorized", "message": "missing or invalid bearer token"}},
                    status_code=401)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/v1/status")
    def status():
        info = STATE.registry.status()
        info["uptime_seconds"] = round(time.time() - STATE.started)
        if STATE.tunnel is not None:
            info["tunnel"] = STATE.tunnel.info()
        return info

    @app.get("/v1/models")
    def list_models():
        files = []
        models_dir = getattr(STATE.registry, "models_dir", None)
        if models_dir and os.path.isdir(models_dir):
            for root, _dirs, names in os.walk(models_dir):
                for n in names:
                    if n.endswith(".safetensors"):
                        rel = os.path.relpath(os.path.join(root, n), models_dir)
                        files.append(rel.replace("\\", "/")[:-len(".safetensors")])
        return {"loaded": [
            {"name": e["name"], "kind": e["kind"]} for e in STATE.registry.status()["models"]],
            "local_files": sorted(files),
            "default": STATE.registry.default_name}

    @app.post("/v1/models/load")
    async def load_model(request: Request):
        spec = await request.json()
        if "name" not in spec or "kind" not in spec:
            raise HTTPException(422, "spec requires 'name' and 'kind'")
        native = engines_native.dispatch_is_native(spec)
        if native:
            has_sources = (isinstance(spec.get("sources"), (list, tuple)) and spec["sources"]) \
                or bool(spec.get("source"))
            if not has_sources:
                raise HTTPException(422, "native specs require 'sources': [paths] "
                                         "(or a single 'source' path)")
            if spec.get("kind") == "native" and not spec.get("clip_type"):
                raise HTTPException(422, "kind 'native' requires 'clip_type'")
        elif spec["kind"] in ("sdxl", "sd3", "flux") and not spec.get("components"):
            raise HTTPException(422, f"{spec['kind']} requires 'components'")
        elif spec["kind"] not in ("sdxl", "sd3", "flux") and not spec.get("source"):
            raise HTTPException(422, "spec requires 'source'")
        try:
            def build_locked():
                with STATE.registry._infer_lock:
                    return STATE.registry.build_engine(spec)
            engine = await asyncio.get_running_loop().run_in_executor(None, build_locked)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except (ValueError, RuntimeError, OSError) as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # noqa: BLE001 - surface loader crashes as readable 400s
            import traceback
            raise HTTPException(
                400, f"{type(e).__name__}: {e} | "
                     f"{traceback.format_exc(limit=2).splitlines()[-2].strip()}")
        gpu = eng.gpu_summary()
        return {"loaded": engine.name, "kind": getattr(engine, "kind", spec["kind"]),
                "load_seconds": engine.load_seconds, "gpu": gpu}

    @app.post("/v1/models/unload")
    async def unload_model(request: Request):
        body = await request.json()
        try:
            with STATE.registry._infer_lock:
                STATE.registry.unload(body.get("name", ""))
        except KeyError as e:
            raise HTTPException(404, str(e))
        return {"unloaded": body.get("name"), "default": STATE.registry.default_name}

    @app.post("/v1/models/default")
    async def set_default(request: Request):
        body = await request.json()
        try:
            with STATE.registry._infer_lock:
                STATE.registry.set_default(body.get("name", ""))
        except KeyError as e:
            raise HTTPException(404, str(e))
        return {"default": body.get("name")}

    @app.get("/v1/loras")
    def list_loras():
        return {"loras": STATE.registry.list_loras()}

    @app.put("/v1/loras/{name}")
    async def upload_lora(name: str, request: Request):
        if not name or name != os.path.basename(name) or name.startswith("."):
            raise HTTPException(422, "invalid lora name")
        data = await request.body()
        if len(data) < 9:
            raise HTTPException(422, "not a safetensors file")
        import struct
        header_len = struct.unpack("<Q", data[:8])[0]
        if header_len <= 1 or header_len > SAFETENSORS_MAGIC_MAX_HEADER or header_len > len(data):
            raise HTTPException(422, "not a safetensors file")
        try:
            json.loads(data[8:8 + header_len])
        except ValueError:
            raise HTTPException(422, "not a safetensors file")
        loras_dir = STATE.registry.loras_dir
        if not loras_dir:
            raise HTTPException(400, "server started without a loras directory")
        os.makedirs(loras_dir, exist_ok=True)
        if not name.endswith(".safetensors"):
            name += ".safetensors"
        path = os.path.join(loras_dir, name)
        if os.path.abspath(path) != os.path.abspath(os.path.join(loras_dir, os.path.basename(name))):
            raise HTTPException(422, "invalid path")
        with open(path, "wb") as f:
            f.write(data)
        return {"saved": name, "bytes": len(data)}

    @app.delete("/v1/loras/{name}")
    def delete_lora(name: str):
        path = STATE.registry.resolve_lora(name)
        if path is None:
            raise HTTPException(404, f"lora not found: {name}")
        os.remove(path)
        return {"deleted": name}

    @app.post("/v1/cache/clear")
    def clear_cache():
        return STATE.registry.clear_cache()

    @app.get("/v1/tunnel")
    def tunnel_info():
        if STATE.tunnel is None:
            return {"provider": "none"}
        return STATE.tunnel.info()

    @app.post("/v1/server/shutdown")
    async def shutdown(request: Request):
        body = await request.json()
        if body.get("confirm") is not True:
            raise HTTPException(422, "pass {\"confirm\": true} to shut down")

        def _stop():
            time.sleep(0.5)
            if STATE.tunnel is not None:
                STATE.tunnel.stop()
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=_stop, daemon=True).start()
        return {"shutting_down": True}

    # ------------------------------------------------------------------
    # data plane
    # ------------------------------------------------------------------

    def _parse_packed(request_bytes):
        try:
            meta, tensors = packing.parse(request_bytes)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if meta.get("proto") != packing.PACKING_VERSION:
            raise HTTPException(409, f"protocol mismatch: worker speaks v{packing.PACKING_VERSION}, "
                                     f"client sent v{meta.get('proto')}")
        return meta, tensors

    def _pack_response(payload):
        body = packing.frame(payload)
        return Response(content=body, media_type="application/x-rcp-v3")

    @app.post("/v1/encode")
    async def encode(request: Request):
        meta, tensors = _parse_packed(await request.body())
        kwargs = packing.restore_tensors(meta.get("kwargs", {}), tensors)
        lora_stack = meta.get("lora_stack", [])
        future = STATE.encode_executor.submit(
            STATE.registry.encode, meta.get("text", ""), kwargs, lora_stack)
        try:
            result = await asyncio.wait_for(
                asyncio.wrap_future(future), timeout=STATE.encode_timeout)
        except asyncio.TimeoutError:
            return JSONResponse(
                {"error": {"code": "retryable_timeout",
                           "message": f"encode exceeded {STATE.encode_timeout}s; retry the request"}},
                status_code=503)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"{type(e).__name__}: {e}")
        return _pack_response({"struct": result})

    @app.post("/v1/generate")
    async def generate(request: Request):
        meta, tensors = _parse_packed(await request.body())
        kwargs = packing.restore_tensors(meta.get("kwargs", {}), tensors)
        lora_stack = meta.get("lora_stack", [])
        job_id = uuid.uuid4().hex[:16]
        job = {"id": job_id, "state": "queued", "created": time.time()}
        with STATE.jobs_lock:
            STATE.jobs[job_id] = job

        def run():
            job["state"] = "running"
            try:
                job["text"] = STATE.registry.generate(
                    meta.get("text", ""), kwargs, meta.get("gen_kwargs", {}), lora_stack)
                job["state"] = "done"
            except Exception as e:  # noqa: BLE001
                job["state"] = "error"
                job["error"] = f"{type(e).__name__}: {e}"

        STATE.job_executor.submit(run)
        return JSONResponse({"job_id": job_id, "state": "queued"}, status_code=202)

    @app.get("/v1/jobs/{job_id}")
    def job_status(job_id: str):
        with STATE.jobs_lock:
            job = STATE.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"job not found: {job_id}")
        out = {"job_id": job["id"], "state": job["state"]}
        if "text" in job:
            out["text"] = job["text"]
        if "error" in job:
            out["error"] = job["error"]
        return out

    @app.delete("/v1/jobs/{job_id}")
    def cancel_job(job_id: str):
        with STATE.jobs_lock:
            job = STATE.jobs.pop(job_id, None)
        if job is None:
            raise HTTPException(404, f"job not found: {job_id}")
        return {"deleted": job_id, "state": job["state"]}

    return app


def start_job_reaper():
    def reap():
        while True:
            time.sleep(60)
            with STATE.jobs_lock:
                stale = [jid for jid, j in STATE.jobs.items()
                         if j["state"] in ("done", "error")
                         and time.time() - j["created"] > JOB_RETENTION_SECONDS]
                for jid in stale:
                    STATE.jobs.pop(jid, None)
    threading.Thread(target=reap, daemon=True).start()
