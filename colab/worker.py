"""CLI entry point for the Remote CLIP Colab runtime.

Example (Google Colab):
    !git clone <this repo> && cd ComfyUI-RemoteCLIPColab/colab
    !pip -q install -r requirements.txt
    !python worker.py --model clip_l:./clip_l.safetensors --tunnel cloudflare
"""
import argparse
import json
import os
import secrets
import sys
import time

# Attention mode must reach the vendored stack before comfy is first imported
# (its attention dispatch binds at import time), so pre-scan argv here.
def _prescan_attention():
    modes = ("auto", "sdpa", "sage", "flash")
    for i, a in enumerate(sys.argv):
        if a == "--attention" and i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
            if mode not in modes:
                raise SystemExit(f"--attention must be one of {modes}, got '{mode}'")
            os.environ["RCP_ATTENTION"] = mode
            return
        if a.startswith("--attention="):
            mode = a.split("=", 1)[1]
            if mode not in modes:
                raise SystemExit(f"--attention must be one of {modes}, got '{mode}'")
            os.environ["RCP_ATTENTION"] = mode
            return


_prescan_attention()

import uvicorn

import engines as eng
import engines_native
from server import create_app, start_job_reaper
from tunnel import TunnelManager


def probe_native():
    """Import-test the vendored stack in a subprocess (it may hard-crash on
    CUDA-less torch builds), so the worker process stays untouched."""
    import subprocess
    import sys
    try:
        r = subprocess.run([sys.executable, "-c", "import comfy.sd"],
                           capture_output=True, timeout=120,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def resolve_engine_mode(requested):
    if requested == "native":
        return "native"
    if requested == "hf":
        return "hf"
    return "native" if probe_native() else "hf"


def parse_model_spec(spec, args):
    """'kind:source' or 'kind:src1+src2' -> engine spec dict."""
    kind, _, source = spec.partition(":")
    kind = kind.strip()
    source = source.strip()
    if args.engine_mode == "native":
        sources = [s.strip() for s in source.split("+") if s.strip()]
        return {"name": kind, "kind": kind, "engine": "native", "sources": sources}
    known = ("clip_l", "sdxl_clip_l", "clip_g", "t5", "qwen_image", "causal_lm")
    if kind not in known:
        raise SystemExit(f"unknown model kind '{kind}' in '{spec}' (hf engine expects one of {known})")
    return {"name": kind, "kind": kind, "source": source,
            "dtype": args.dtype, "device": args.device}


def main():
    parser = argparse.ArgumentParser(description="Remote CLIP Colab worker")
    parser.add_argument("--model", action="append", default=[],
                        help="initial model as kind:source, e.g. clip_l:./clip_l.safetensors")
    parser.add_argument("--models-config", help="JSON file with a list of engine specs "
                        "(supports composite kinds sdxl/sd3/flux)")
    parser.add_argument("--host", default=None, help="bind host (default: 127.0.0.1, "
                        "0.0.0.0 when --tunnel direct)")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--token", help="auth token; auto-generated for tunnels")
    parser.add_argument("--tunnel", choices=["cloudflare", "ngrok", "direct"], default="cloudflare")
    parser.add_argument("--ngrok-token", help="ngrok authtoken (required for --tunnel ngrok)")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "tpu", "cpu"])
    parser.add_argument("--engine", default="auto", choices=["auto", "native", "hf"],
                        help="text-encoder backend: native (vendored comfy stack, full "
                             "format coverage) or hf (pure transformers). auto probes.")
    parser.add_argument("--attention", default=os.environ.get("RCP_ATTENTION", "auto"),
                        choices=["auto", "sdpa", "sage", "flash"],
                        help="native-backend attention kernel: auto (comfy default "
                             "sub-quad), sdpa (torch scaled_dot_product_attention), "
                             "sage (sageattention pkg), flash (flash-attn pkg)")
    parser.add_argument("--vram", default=os.environ.get("RCP_VRAM", "auto"),
                        choices=["auto", "stream", "cpu"],
                        help="native-backend VRAM strategy: auto (maximize VRAM "
                             "residency; on CUDA OOM degrade to partial-load then "
                             "per-layer streaming and retry), stream (never "
                             "full-load at construction; keep what fits in VRAM, "
                             "stream overflow layers), cpu (run text encoder in "
                             "RAM; last resort)")
    parser.add_argument("--attention-hf", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"],
                        help="hf-backend attention implementation passed to "
                             "from_pretrained (CUDA only; TPU/XLA picks its own kernel)")
    parser.add_argument("--xla-cache", default=None, metavar="DIR",
                        help="TPU only: persist XLA compiled executables to DIR so "
                             "recompiles survive worker restarts")
    parser.add_argument("--models-dir", default="./models", help="directory scanned by GET /v1/models")
    parser.add_argument("--loras-dir", default="./models/loras")
    parser.add_argument("--embeddings-dir", default="./models/embeddings")
    parser.add_argument("--tokenizer-clip", help="clip tokenizer source override")
    parser.add_argument("--tokenizer-t5", help="t5 tokenizer source override")
    parser.add_argument("--tokenizer-qwen", help="qwen tokenizer source override")
    parser.add_argument("--encode-timeout", type=float, default=90.0,
                        help="seconds before /v1/encode answers 503 retryable")
    parser.add_argument("--lifetime", type=float, default=0,
                        help="auto-shutdown after this many seconds (0 = run forever)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    if args.attention != "auto":
        os.environ["RCP_ATTENTION"] = args.attention
    os.environ["RCP_VRAM"] = args.vram
    if args.device == "tpu":
        args.engine = "hf" if args.engine == "native" else args.engine
    args.engine_mode = resolve_engine_mode(args.engine)
    os.environ.setdefault("RCP_ATTENTION_HF", args.attention_hf)
    if args.xla_cache:
        eng.enable_xla_persistent_cache(args.xla_cache)
        print(f"xla persistent cache: {args.xla_cache}")

    for d in (args.models_dir, args.loras_dir, args.embeddings_dir):
        os.makedirs(d, exist_ok=True)

    registry = eng.EngineRegistry(loras_dir=args.loras_dir, embeddings_dir=args.embeddings_dir)
    registry.models_dir = args.models_dir
    if args.tokenizer_clip:
        registry.tokenizer_overrides["clip_l"] = args.tokenizer_clip
        registry.tokenizer_overrides["clip_g"] = args.tokenizer_clip
    if args.tokenizer_t5:
        registry.tokenizer_overrides["t5"] = args.tokenizer_t5
    if args.tokenizer_qwen:
        registry.tokenizer_overrides["qwen"] = args.tokenizer_qwen

    specs = []
    for spec in args.model:
        specs.append(parse_model_spec(spec, args))
    if args.models_config:
        with open(args.models_config) as f:
            config = json.load(f)
        default = config.get("default")
        specs.extend(config.get("models", []))
        if default:
            registry._pending_default = default  # noqa: SLF001

    token = args.token
    if not token and args.tunnel != "direct":
        token = secrets.token_hex(16)

    host = args.host or ("0.0.0.0" if args.tunnel == "direct" else "127.0.0.1")

    print_banner = args.tunnel != "direct"
    tunnel = TunnelManager(args.tunnel, args.port, host if args.tunnel == "direct" else "127.0.0.1",
                           ngrok_token=args.ngrok_token)

    app = create_app(registry, tunnel=tunnel, token=token, encode_timeout=args.encode_timeout)

    print("\n" + "=" * 62)
    print("  Remote CLIP Colab Worker")
    print("=" * 62)
    gpu = eng.gpu_summary()
    print(f"  device : {gpu}")
    attention_desc = f" (attention: {os.environ.get('RCP_ATTENTION', 'auto')})" if args.engine_mode == "native" else ""
    print(f"  engine : {args.engine_mode}"
          + ("" if args.engine_mode != "native" else " (vendored comfy stack)" + attention_desc))
    print(f"  listen : http://{host}:{args.port}")
    print(f"  tunnel : {args.tunnel}")
    if specs:
        print(f"  models : {[s['name'] for s in specs]}")
    else:
        print("  models : (none at startup; load via POST /v1/models/load)")
    print("=" * 62, flush=True)

    for spec in specs:
        t0 = time.time()
        origin = spec.get("sources") or spec.get("source") or spec.get("components")
        print(f"loading {spec.get('name', spec.get('kind'))} from {origin} ...", flush=True)
        registry.build_engine(spec)
        print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
    pending_default = getattr(registry, "_pending_default", None)
    if pending_default:
        registry.set_default(pending_default)

    tunnel.start()
    start_job_reaper()

    if args.lifetime > 0:
        import threading

        def _expire():
            time.sleep(args.lifetime)
            print("lifetime reached; shutting down", flush=True)
            tunnel.stop()
            os._exit(0)

        threading.Thread(target=_expire, daemon=True).start()

    if print_banner:
        def show_url():
            shown = [False]
            t0 = time.time()
            while not shown[0] and time.time() - t0 < 120:
                if tunnel.public_url:
                    print("\n" + "#" * 62)
                    print("  CONNECT FROM THE LOCAL COMFYUI NODE WITH:")
                    print(f"    base_url   : {tunnel.public_url}")
                    print(f"    auth_token : {token}")
                    print("#" * 62 + "\n", flush=True)
                    shown[0] = True
                time.sleep(0.5)
        import threading
        threading.Thread(target=show_url, daemon=True).start()

    try:
        uvicorn.run(app, host=host, port=args.port, log_level=args.log_level, workers=1)
    finally:
        tunnel.stop()


if __name__ == "__main__":
    main()
