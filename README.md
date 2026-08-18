# Remote CLIP Colab for ComfyUI

Run CLIP/text-encoder inference on a remote machine (Google Colab out of the box)
and use it from your local ComfyUI over HTTPS. The remote side is a fully
independent runtime — it does **not** need ComfyUI — with a RESTful API and a
built-in Cloudflare tunnel, so no public IP, port forwarding, or account is
required.

Two roles:

- **Worker** — any Python 3.10+ machine with a GPU/TPU. Runs `colab/worker.py`,
  loads text encoders, answers encode/generate requests.
- **Loader** — your local ComfyUI. The `Load Remote CLIP (Colab)` node returns a
  CLIP object backed by the worker, usable in `CLIPTextEncode`, `Generate Text`,
  SDXL/SD3/Flux/Qwen-Image conditioning nodes, etc.

Transport is always fp16 to keep payloads small; embeddings are restored to
their original dtype on arrival.

## Inference acceleration

- **GPU / native backend** — `--attention auto|sdpa|sage|flash` selects the
  attention kernel through ComfyUI's own dispatch (same flags as upstream
  `--use-sage-attention` etc.; `sage` needs `pip install sageattention`,
  `flash` needs `flash-attn`). Note: text encoders run the small-input path
  where ComfyUI already defaults to fused SDPA on NVIDIA + torch ≥ 2, so the
  default (`auto`) is already fast for CLIP/T5; explicit sage/flash mainly
  matters for other vendored-model paths.
- **GPU / hf backend** — `--attention-hf sdpa|eager|flash_attention_2` is
  passed to `from_pretrained` (`sdpa` default).
- **TPU** — `--xla-cache DIR` persists compiled XLA executables so
  recompilation survives worker restarts (measured on a real v5e: first
  request after restart 11.3 s → 8.4 s, steady-state ~2 s unchanged; the
  remaining time is the ~10 GB weight transfer). The notebook's `XLA_CACHE`
  toggle enables it at `/content/rcp_xla_cache`. TPU attention kernels are
  chosen by the XLA compiler itself — no per-model flag applies.

## VRAM management (native backend)

The native backend uses ComfyUI's own model management, with the same
"keep-as-much-in-VRAM-as-fits" behavior as a local ComfyUI: at each encode,
weights that fit in free VRAM (minus a ~1.4 GB activation reserve) stay
resident on the GPU for the whole session; overflow layers keep their weights
in RAM, pinned, and upload to the GPU just-in-time per forward pass.

`--vram {auto,stream,cpu}` (default `auto`) controls the strategy:

- **auto** — try full GPU residency first; on CUDA OOM the worker frees VRAM
  and retries on the partial-load path (max residency, overflow layers
  stream), then on per-layer streaming, before failing with a clear error.
  Each retry is logged. Big cards running small encoders keep the fast
  full-residency path with zero overhead.
- **stream** — skip the construction-time full-load gamble: weights stage in
  RAM from the start and every encode uses the keep-what-fits budget. For
  GPUs you already know are tight (e.g. a 16 GB card with the MiniMax H3
  NVFP4 file), this avoids a doomed full-load attempt.
- **cpu** — run the text encoder in RAM (ComfyUI `--novram` semantics).
  Last resort; expect minutes per prompt on large encoders.

This is also why a model that "partially falls back to RAM" locally used to
OOM remotely: the vendored loader's construction-time heuristic could
full-load the encoder without the activation reserve, and a worker GPU is
often empty at load time — exactly the situation that misfires. The OOM
ladder now degrades gracefully instead.

## Engine backends

The worker has two interchangeable text-encoder backends (`--engine auto|native|hf`):

- **native** (default when available) — runs the *vendored ComfyUI text-encoder
  stack* (`colab/comfy/`, synced from the ComfyUI repo by `vendor_comfy.ps1`).
  This is ComfyUI's own loading path, so every text-encoder format ComfyUI
  supports works here by construction: checkpoint auto-detection, embedded
  tokenizers (spiece/tekken/tokenizer.json), composite encoders, Long-CLIP,
  the full Qwen3/VL family, and quantized checkpoints (NVFP4/AWQ — verified on
  an RTX 4060 Ti with the MiniMax H3 qwen3vl-32b NVFP4 file, 3 s load). LoRAs
  are applied through comfy's own patcher, matching local behavior.
- **hf** — the pure-transformers engines (zero vendored code). Used when the
  vendored stack cannot import (CPU-only torch builds) or on TPU (`--device tpu`
  forces hf; TPU is not supported by the vendored stack).

`auto` probes the vendored import in a subprocess and picks native on success.

## Quick start (Google Colab)

1. Runtime → Change runtime type → GPU (T4) or TPU v5e.
2. Open `colab/RemoteCLIP_Colab.ipynb` in Colab (File → Upload notebook, or keep it
   in your fork). It is a self-contained launcher:
   - **cell 1** clones the runtime (`REPO_URL` defaults to this repository; point
     it at your fork, or upload the `colab/` folder yourself),
   - **cell 2** is the configure & launch form (token / tunnel / engine /
     attention); it installs the matching dependencies and starts the worker,
     printing the `base_url` + `auth_token` once the tunnel is up,
   - **cell 3** downloads text-encoder / LoRA files into `models/`,
   - **cells 4–5** tail the worker log and stop the worker.
   All worker management (status / load / switch / unload models, list LoRAs,
   clear cache, tunnel info, shutdown) happens from ComfyUI through the
   **Remote CLIP Controller** node — no notebook UI needed.
3. Copy the printed `base_url` (`https://....trycloudflare.com`) and
   `auth_token` into the local node.

Manual equivalent, in one cell:

```python
%cd /content/ComfyUI-RemoteCLIPColab/colab
!pip -q install -r requirements.txt

# download a text encoder (user-initiated, one file):
!wget -q -P models https://huggingface.co/Comfy-Org/flux_text_encoders/resolve/main/clip_l.safetensors

!python worker.py --model clip_l:./models/clip_l.safetensors --tunnel cloudflare
```

For TPU runtimes the worker automatically uses the `hf` backend; install
`torch_xla` per Google's instructions first and the device auto-detects, uses
bf16, and pads dynamic-length encoders to shape buckets to limit XLA recompiles.

## Quick start (any machine, direct mode)

```bash
pip install -r colab/requirements.txt
python colab/worker.py --model sdxl_clip_l:./clip_l.safetensors --tunnel direct --host 0.0.0.0 --port 8188 --token mysecret
```

## Model kinds

With `--engine native` (default on GPU machines), `--model kind:src1+src2`
loads checkpoints through ComfyUI's own detection and construction — any
format ComfyUI's CLIPLoader accepts. Friendly kinds: `clip_l`, `clip_g`,
`sdxl`, `sd3`, `flux`, `flux2`, `wan`, `ltxv`, `pixart`, `chroma`, `cosmos`,
`mochi`, `aura_t5`, `qwen_image`, `qwen3vl`, `z_image`, `minimax_h3`,
`hunyuan_video`, `hydit`, `lumina2`, `anima`, `ovis`, `omnigen2`, `hidream`.
For anything else pass an explicit `clip_type` (`kind: native`, see
`comfy.sd.CLIPType` for all 35 names) via the API/controller node.

Example — MiniMax H3 conditioning from the NVFP4 file:

```bash
python worker.py --model minimax_h3:./models/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
```

With `--engine hf`, the pure-transformers kinds remain available
(`clip_l`, `sdxl_clip_l`, `clip_g`, `t5`, `qwen_image`, `causal_lm`, plus
composites `sdxl`/`sd3`/`flux` via `components`). Sources are HF repo ids,
HF-format dirs, or comfy-format safetensors.

## REST API (protocol v3)

All endpoints except `/health` need `Authorization: Bearer <token>`.
`/docs` on the worker renders the full OpenAPI UI.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/status` | device, models, default, cache stats, tunnel URL |
| GET | `/v1/models` | loaded + local model files |
| POST | `/v1/models/load` | `{name, kind, source?, components?, dtype?}` |
| POST | `/v1/models/unload` | `{name}` |
| POST | `/v1/models/default` | `{name}` |
| GET/PUT/DELETE | `/v1/loras[/{name}]` | list / upload / remove LoRA files |
| POST | `/v1/cache/clear` | clear embedding cache |
| GET | `/v1/tunnel` | current public URL + health |
| POST | `/v1/encode` | packed body, returns embeddings (packed) |
| POST | `/v1/generate` | packed body → `202 {job_id}` (async, survives CF timeout) |
| GET/DELETE | `/v1/jobs/{id}` | poll / drop a generation job |
| POST | `/v1/server/shutdown` | `{confirm: true}` |

Packed bodies use `application/x-rcp-v3`:
`[8-byte BE meta length][JSON meta][tensor blob]` — the same manifest format as
v2 (dtype/orig_dtype/shape/offset/size), with floats transported as fp16.

## Local nodes

- **Load Remote CLIP (Colab)** — `base_url` + `auth_token` → CLIP output.
- **LoraLoader CLIP Only (Colab)** — stack after the loader; forwards
  `(lora_name, strength)` to the worker. With native engines the worker applies
  them through comfy's own LoRA patcher (any format comfy supports); patched
  CLIPs are cached per stack and pristine weights are reused between stacks.
  The file must exist in the worker's `models/loras` (upload via API).
- **Remote CLIP Controller** — run any control-plane action from the canvas:
  status / list models / load / unload / set default / list loras / clear cache /
  tunnel info / shutdown. Native loads use `sources` (`a.safetensors +
  b.safetensors`), HF composites use `components`.

## Caching & retries

- Identical `(model, text, kwargs, lora_stack)` encode requests hit an LRU cache
  on the worker (requests carrying image/video tensors are never cached).
- Encode retries automatically on retryable timeouts; generation runs as a job
  and is never auto-restarted (avoid double sampling), matching the original
  plugin's semantics.
- If the tunnel URL changes (Colab restart), paste the new URL into the node —
  no workflow rebuild needed.

## Known limitations

- Cloudflare free tunnels cap single requests (~100 MB) and first-byte time
  (~100 s). Generation is already job-based; for very large image inputs use
  `--tunnel direct` on a VPS/LAN.
- TPU: generation (autoregressive sampling) is slower than GPU; encoding is
  unaffected. TPU runs force the `hf` backend (the vendored stack has no XLA
  path).
- `causal_lm` (hf backend): image and video inputs are supported for multimodal
  models; audio inputs are not. Model-specific tokenize kwargs such as
  `thinking` are forwarded but currently ignored by the hf engine (native
  engines handle them via comfy's own tokenizers).
- A hung generation (pathological sampling or a device hang) blocks later
  generate jobs on the worker's single inference thread; restart the worker
  if that happens. Encode requests are unaffected.
- Textual inversion embeddings (`embedding:` syntax) work for native engines
  via the worker's `models/embeddings` directory, and for the hf `clip_l`/`t5`
  engines.

## Vendored ComfyUI stack (license)

`colab/comfy/` and `colab/node_helpers.py` are synced from the ComfyUI repo
(see `colab/comfy/NOTICE`). ComfyUI is GPL-3.0, so distributing this project
requires GPL-3 compatibility; personal and Colab use is unaffected. Re-sync
after ComfyUI updates with:

```powershell
cd colab
powershell -File vendor_comfy.ps1 <path-to-ComfyUI-checkout>
```

## VRAM guide (bf16)

| Runtime | fits |
|---------|------|
| T4 16 GB | clip_l/g, sdxl, flux (t5 fp16), causal_lm 4B |
| L4 24 GB | sd3, qwen_image (7B), causal_lm 8B, MiniMax H3 (NVFP4) |
| A100 40 GB | everything incl. qwen_image + large causal LMs |
| RTX 4060 Ti 16 GB (tested) | MiniMax H3 NVFP4 (14.9 GB file) loads and encodes via the native backend |

Tight cards: if a model barely fits, `--vram stream` starts in the
partial-load mode directly instead of trying full residency first.

Node category: `Remote CLIP`
