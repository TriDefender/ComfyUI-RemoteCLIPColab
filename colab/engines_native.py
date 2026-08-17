"""Native engines: vendored ComfyUI text-encoder stack as worker engines.

Bridges comfy.sd.load_text_encoder_state_dicts (checkpoint detection, embedded
tokenizer data, composite construction) behind the same encode/generate
interface the HF engines implement. LoRA support rides comfy's own patcher
(clone + load_lora_for_models chain, LRU-cached per stack) — the same design
the original RemoteCLIPLoader worker used.

comfy is imported lazily: the attention-kernel flags on comfy.cli_args.args
must be set (from RCP_ATTENTION) BEFORE comfy.ldm.modules.attention binds its
dispatch at import time, so nothing may import comfy before configure time.
"""
import json
import os
import time
from collections import OrderedDict

PATCHED_CLIP_CACHE = 4

_comfy = None
_comfy_error = None
comfy = None  # set by ensure_comfy(); module-level so engine methods can use comfy.utils/sd

ATTENTION_MODES = ("auto", "sdpa", "sage", "flash")


def log(msg):
    print(f"[native {time.strftime('%H:%M:%S')}] {msg}", flush=True)


HF_KINDS = {"clip_l", "sdxl_clip_l", "clip_g", "t5", "qwen_image", "causal_lm",
            "sdxl", "sd3", "flux"}


def dispatch_is_native(spec):
    """Shared hf/native dispatch rule: an explicit engine field wins; packed
    'sources' means native; kinds that only exist natively go native even when
    the caller filled the singular 'source' field."""
    kind = spec.get("kind", "")
    if spec.get("engine") == "hf":
        return False
    if spec.get("engine") == "native":
        return True
    if kind == "native" or "sources" in spec:
        return True
    return kind in NATIVE_KINDS and kind not in HF_KINDS


def _probe_import(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def ensure_comfy():
    """Import the vendored stack with the attention mode applied. The comfy
    attention dispatch (sage/flash/pytorch-sdpa/sub-quad) binds at import time,
    so RCP_ATTENTION must be honored on first import only."""
    global comfy, _comfy, _comfy_error
    if _comfy is not None:
        return _comfy
    if _comfy_error is not None:
        raise RuntimeError(_comfy_error)
    try:
        mode = os.environ.get("RCP_ATTENTION", "auto")
        import comfy.cli_args
        if mode == "sage":
            if not _probe_import("sageattention"):
                raise ValueError(
                    "--attention sage requires the sageattention package: "
                    f"{os.sys.executable} -m pip install sageattention")
            comfy.cli_args.args.use_sage_attention = True
        elif mode == "flash":
            if not _probe_import("flash_attn"):
                raise ValueError(
                    "--attention flash requires the flash-attn package: "
                    f"{os.sys.executable} -m pip install flash-attn --no-build-isolation")
            comfy.cli_args.args.use_flash_attention = True
        elif mode == "sdpa":
            comfy.cli_args.args.use_pytorch_cross_attention = True
        import comfy.sd
        import comfy.utils
        import comfy.model_management
        _comfy = comfy
        return _comfy
    except ValueError:
        _comfy_error = "attention mode rejected"
        raise
    except Exception as e:  # noqa: BLE001 - fall back to HF engines
        _comfy_error = str(e)
        raise


def vendored_status():
    if _comfy is not None:
        return {"available": True, "attention": os.environ.get("RCP_ATTENTION", "auto")}
    return {"available": False, "error": _comfy_error}


# friendly kind -> (CLIPType name, allowed source counts); None clip_type keeps
# the loader default (STABLE_DIFFUSION) and lets checkpoint detection decide.
NATIVE_KINDS = {
    "clip_l": ("STABLE_DIFFUSION", (1,)),
    "clip_g": ("STABLE_DIFFUSION", (1,)),
    "clip_g_refiner": ("STABLE_CASCADE", (1,)),
    "sdxl": ("STABLE_DIFFUSION", (2,)),
    "sdxl_longclip": ("STABLE_DIFFUSION", (2,)),
    "sd3": ("SD3", (2, 3)),
    "flux": ("FLUX", (2,)),
    "flux2": ("FLUX2", (1,)),
    "t5": (None, (1,)),
    "t5_ltxv": ("LTXV", (1,)),
    "t5_pixart": ("PIXART", (1,)),
    "t5_chroma": ("CHROMA", (1,)),
    "t5_cogvideox": ("COGVIDEOX", (1,)),
    "wan": ("WAN", (1,)),
    "ltxv": ("LTXV", (1,)),
    "pixart": ("PIXART", (1,)),
    "chroma": ("CHROMA", (1,)),
    "cosmos": (None, (1,)),
    "mochi": (None, (1,)),
    "aura_t5": (None, (1,)),
    "sa_t5": (None, (1,)),
    "ace_t5": ("ACE", (1,)),
    "sa3_t5gemma": (None, (1,)),
    "qwen_image": (None, (1,)),
    "qwen3vl": (None, (1,)),
    "z_image": (None, (1,)),
    "minimax_h3": (None, (1,)),
    "hunyuan_video": ("HUNYUAN_VIDEO", (1, 2)),
    "hunyuan_video_15": ("HUNYUAN_VIDEO_15", (1, 2)),
    "hunyuan_image": (None, (1,)),
    "hydit": ("HUNYUAN_DIT", (2,)),
    "lumina2": (None, (1,)),
    "lumina2_n": (None, (1,)),
    "anima": (None, (1,)),
    "ovis": (None, (1,)),
    "omnigen2": (None, (1,)),
    "ernie": (None, (1,)),
    "hidream": ("HIDREAM", (2, 4)),
    "longcat_image": (None, (1,)),
    "ideogram4": (None, (1,)),
    "boogu": (None, (1,)),
    "krea2": (None, (1,)),
    "joyimage": (None, (1,)),
    "mage_flow": (None, (1,)),
    "gemma4": (None, (1,)),
    "ltxav_gemma4": (None, (1,)),
    "jina_clip_2": (None, (1,)),
}


class NativeEngine:
    handles_lora_internally = True

    def __init__(self, name, kind, clip_type, sources, embedding_directory=None):
        comfy = ensure_comfy()
        self.name = name
        self.kind = kind or "native"
        self.clip_type_name = clip_type
        self.sources = sources
        self.device = None
        self.dtype = None
        self.lora = None

        clip_data = []
        for src in sources:
            sd, _metadata = comfy.utils.load_torch_file(src, safe_load=True, return_metadata=True)
            clip_data.append(sd)
        ct = comfy.sd.CLIPType[clip_type] if clip_type else comfy.sd.CLIPType.STABLE_DIFFUSION
        t0 = time.time()
        self.clip = comfy.sd.load_text_encoder_state_dicts(
            clip_data, embedding_directory=embedding_directory, clip_type=ct)
        self.load_seconds = round(time.time() - t0, 1)
        try:
            self.device = self.clip.patcher.load_device
            self.dtype = self.clip.patcher.model_dtype()
        except Exception:  # noqa: BLE001 - status fields only
            pass
        self._patched = OrderedDict()

    def _get_clip(self, lora_stack, lora_resolver):
        if not lora_stack:
            return self.clip
        sig = json.dumps(lora_stack)
        cached = self._patched.get(sig)
        if cached is not None:
            self._patched.move_to_end(sig)
            return cached
        clip = self.clip
        for lora_name, strength in lora_stack:
            if strength == 0:
                continue
            path = lora_resolver(lora_name)
            if path is None:
                raise FileNotFoundError(f"LoRA not found on worker: {lora_name}")
            lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
            _model, clip = comfy.sd.load_lora_for_models(None, clip, lora_sd, 0, strength)
        self._patched[sig] = clip
        while len(self._patched) > PATCHED_CLIP_CACHE:
            self._patched.popitem(last=False)
        return clip

    def encode(self, text, kwargs, lora_stack, lora_resolver=None):
        clip = self._get_clip(lora_stack, lora_resolver)
        tokens = clip.tokenize(text, **kwargs)
        out = clip.encode_from_tokens(tokens, return_dict=True)
        return out

    def generate(self, text, kwargs, gen_kwargs, lora_stack, lora_resolver=None):
        clip = self._get_clip(lora_stack, lora_resolver)
        tokens = clip.tokenize(text, **kwargs)
        generated = clip.generate(tokens, **gen_kwargs)
        return clip.decode(generated)

    def unload(self):
        self._patched.clear()
        try:
            comfy.model_management.unload_all_models()
        except Exception:  # noqa: BLE001
            pass


def build_native_engine(spec, registry):
    kind = spec.get("kind")
    sources = spec.get("sources")
    if not sources and spec.get("source"):
        sources = [spec["source"]]
    if not sources or not isinstance(sources, (list, tuple)) or not all(isinstance(s, str) for s in sources):
        raise ValueError("native engines require 'sources': [checkpoint paths] "
                         "(or a single 'source' path)")
    sources = [s if os.path.isabs(s) else os.path.abspath(s) for s in sources]
    for s in sources:
        if _looks_like_path(s) and not os.path.exists(s):
            raise FileNotFoundError(f"model file not found on worker: {s}")
    clip_type = spec.get("clip_type")
    if kind in NATIVE_KINDS:
        ct_name, counts = NATIVE_KINDS[kind]
        clip_type = clip_type or ct_name
        if len(sources) not in counts:
            raise ValueError(f"kind '{kind}' expects {counts} source file(s), got {len(sources)}")
    elif kind == "native":
        if not clip_type:
            raise ValueError("kind 'native' requires 'clip_type' (a comfy.sd.CLIPType name)")
        try:
            comfy = ensure_comfy()
            comfy.sd.CLIPType[clip_type]
        except KeyError:
            raise ValueError(
                f"unknown clip_type '{clip_type}'; see comfy.sd.CLIPType for valid names") from None
    else:
        return None
    embedding_directory = registry.embeddings_dir if registry.embeddings_dir else None
    return NativeEngine(spec["name"], kind, clip_type, sources,
                        embedding_directory=embedding_directory)


def _looks_like_path(source):
    return (os.sep in source or "/" in source or "\\" in source
            or source.startswith(".") or source.endswith(".safetensors"))
