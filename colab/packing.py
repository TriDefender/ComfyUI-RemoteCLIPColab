"""Wire packing for the Remote CLIP Colab runtime (protocol v3).

Body layout: [8-byte big-endian meta length][meta JSON bytes][tensor blob].
Floating point tensors are always transported as fp16 (fixed policy); the
original dtype is recorded so the client can restore it.
"""
import json
import struct
import torch

PACKING_VERSION = 3
LEN_FMT = ">Q"
LEN_SIZE = struct.calcsize(LEN_FMT)
MAX_META_BYTES = 8 * 1024 * 1024
MAX_BLOB_BYTES = 512 * 1024 * 1024

ALLOWED_DTYPES = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int64": torch.int64,
    "torch.int32": torch.int32,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def pack_tensors(tensors):
    """Serialize {name: tensor} into (manifest, blob). Floats -> fp16 on the wire."""
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
    """Replace tensor leaves with {"__tensor__": name} references."""
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
    """Build a full request/response body; tensor leaves inside meta are packed."""
    tensor_inputs = {}
    meta = extract_tensors(meta, "meta", tensor_inputs)
    manifest, blob = pack_tensors(tensor_inputs)
    meta["tensors"] = manifest
    meta["blob_size"] = len(blob)
    meta_bytes = json.dumps(meta).encode("utf-8")
    if len(meta_bytes) > MAX_META_BYTES:
        raise ValueError(f"Meta too large: {len(meta_bytes)} bytes")
    if len(blob) > MAX_BLOB_BYTES:
        raise ValueError(f"Blob too large: {len(blob)} bytes")
    return struct.pack(LEN_FMT, len(meta_bytes)) + meta_bytes + blob


def parse(body):
    """Parse a body produced by frame() -> (meta, {name: tensor})."""
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
