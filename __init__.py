import json

import folder_paths
import comfy.sd
import comfy.utils

from .client import RemoteCLIPClient, RemoteCLIPProxy, WorkerError

DEFAULT_PORT_HINT = "https://xxxx.trycloudflare.com"


class LoadRemoteCLIPColab:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": DEFAULT_PORT_HINT,
                    "tooltip": "Public URL of the remote worker. Copy the base_url printed "
                               "by the Colab worker at startup (https://....trycloudflare.com, "
                               "or http://ip:port in direct mode). Paste the new URL here after "
                               "a Colab restart — no workflow rebuild needed."}),
                "auth_token": ("STRING", {"default": "",
                    "tooltip": "Bearer token the worker printed next to its URL. "
                               "Required for tunnels (the URL is public); leave empty only "
                               "for a direct-mode worker started without --token."}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    OUTPUT_TOOLTIPS = ("A CLIP backed by the remote worker: connect it to CLIPTextEncode, "
                       "SDXL/SD3/Flux/Qwen-Image conditioning, or Generate Text nodes.",)
    FUNCTION = "load_remote"
    CATEGORY = "Remote CLIP"
    DESCRIPTION = ("Connect to a Remote CLIP Colab worker over HTTPS. The worker "
                   "prints its base_url and auth_token at startup. The returned CLIP "
                   "behaves like a local one; tokenization stays local, encoding runs "
                   "on the remote GPU/TPU.")

    def load_remote(self, base_url, auth_token):
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise WorkerError(
                f"base_url must start with http:// or https:// (got: {base_url!r}). "
                "Copy the URL printed by the worker.")
        client = RemoteCLIPClient(base_url, auth_token)
        status = client.check_protocol()
        print(f"[RemoteCLIPColab] connected to worker: device={status.get('device')} "
              f"default_model={status.get('default_model')}")
        return (RemoteCLIPProxy(base_url, client=client),)


class LoraLoaderCLIPOnlyColab:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "CLIP": ("CLIP", {"tooltip": "A local CLIP or the output of Load Remote "
                                             "CLIP (Colab). With a remote CLIP the LoRA is "
                                             "forwarded and applied on the worker."}),
                "lora_name": (folder_paths.get_filename_list("loras"),
                              {"tooltip": "Name shown to the worker. The file with this "
                                          "name must exist in the worker's models/loras "
                                          "(upload via the controller's list_loras / API); "
                                          "the worker resolves it there."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,
                                             "step": 0.01,
                                             "tooltip": "Effect strength on the text "
                                                        "encoder. 0 disables it."}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    OUTPUT_TOOLTIPS = ("CLIP with the LoRA stacked; the worker applies it at encode time.",)
    FUNCTION = "load_lora"
    CATEGORY = "Remote CLIP"
    DESCRIPTION = ("Stack a text-encoder LoRA onto a CLIP. With a remote (Colab) CLIP it "
                   "only forwards name + strength — the worker must have the file. Stack "
                   "multiple loaders to combine LoRAs. The standard LoRA loader does not "
                   "work on a remote CLIP; use this one instead.")

    def load_lora(self, CLIP, lora_name, strength_clip):
        if isinstance(CLIP, RemoteCLIPProxy):
            return (CLIP.with_lora(lora_name, strength_clip),)
        if strength_clip == 0:
            return (CLIP,)
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        _, clip_lora = comfy.sd.load_lora_for_models(None, CLIP, lora, 0, strength_clip)
        return (clip_lora,)


class RemoteCLIPController:
    ACTIONS = ["status", "list_models", "list_loras", "load_model", "unload_model",
               "set_default", "clear_cache", "tunnel", "shutdown"]
    KINDS = ["", "clip_l", "sdxl_clip_l", "clip_g", "t5", "qwen_image", "causal_lm",
             "sdxl", "sd3", "flux",
             "native", "flux2", "wan", "ltxv", "pixart", "chroma",
             "cosmos", "mochi", "aura_t5", "qwen3vl", "z_image", "minimax_h3",
             "hunyuan_video", "hydit", "lumina2", "anima", "ovis", "omnigen2", "hidream"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": DEFAULT_PORT_HINT,
                    "tooltip": "Public URL of the remote worker (same value as in "
                               "Load Remote CLIP (Colab))."}),
                "action": (cls.ACTIONS, {"default": "status",
                    "tooltip": "status: device/models/cache overview · list_models: loaded + "
                               "local files · list_loras: worker's models/loras · load_model: "
                               "load one (see kind/source fields) · unload_model/set_default: "
                               "by model_name · clear_cache: drop cached embeddings · tunnel: "
                               "current public URL · shutdown: stop the worker (requires "
                               "shutdown_confirm)."}),
            },
            "optional": {
                "auth_token": ("STRING", {"default": "",
                    "tooltip": "Bearer token printed by the worker. Required for tunnels."}),
                "model_name": ("STRING", {"default": "", "placeholder": "engine name",
                    "tooltip": "Your name for the engine. Used by load_model and by "
                               "unload_model / set_default to pick an existing engine."}),
                "kind": (cls.KINDS, {"default": "",
                    "tooltip": "load_model only. Text-encoder family. checkpoint formats "
                               "(clip_l, sdxl, flux, minimax_h3, wan, ...) load through "
                               "ComfyUI's own loader (worker GPU backend); clip_l/t5/qwen_image/"
                               "causal_lm also have pure-transformers forms (used on TPU). "
                               "sdxl/sd3/flux take 'components' instead of a path."}),
                "source": ("STRING", {"default": "",
                    "placeholder": "HF repo id or safetensors path",
                    "tooltip": "load_model, single-source kinds. Path on the worker "
                               "(e.g. /content/rcp/models/clip_l.safetensors) or an HF "
                               "repo id for the pure-transformers backend."}),
                "sources": ("STRING", {"default": "",
                    "placeholder": "path1.safetensors + path2.safetensors",
                    "tooltip": "load_model, checkpoint kinds. One or more worker-side "
                               "paths joined by '+'; composites like sdxl/flux take their "
                               "parts in order (e.g. flux: clip_l + t5xxl)."}),
                "components": ("STRING", {"default": "",
                    "placeholder": '{"clip_l": "...", "t5": "..."}',
                    "tooltip": "load_model for pure-transformers sdxl/sd3/flux only: JSON "
                               "mapping part -> worker-side path or HF repo id."}),
                "clip_type": ("STRING", {"default": "",
                    "placeholder": "comfy CLIPType name for kind=native",
                    "tooltip": "Only with kind 'native': a comfy.sd.CLIPType name "
                               "(e.g. QWEN_IMAGE, HUNYUAN_IMAGE) for formats without a "
                               "friendly kind yet."}),
                "dtype": (["auto", "bf16", "fp16", "fp32"], {"default": "auto",
                    "tooltip": "load_model compute dtype. auto = bf16 on GPU/TPU, fp32 on "
                               "CPU. Checkpoint kinds ignore this (comfy decides)."}),
                "shutdown_confirm": ("BOOLEAN", {"default": False,
                    "tooltip": "shutdown only: the worker is stopped exclusively when this "
                               "is enabled, so queueing the node can't kill it by accident."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "run_action"
    OUTPUT_NODE = True
    CATEGORY = "Remote CLIP"
    DESCRIPTION = ("Run one control-plane action on the remote worker and show the JSON "
                   "result on the node. Typical flow: load_model -> set_default -> switch "
                   "Load Remote CLIP inputs (or just re-queue) to use the new engine.")

    def run_action(self, base_url, action, auth_token="", model_name="", kind="",
                   source="", components="", sources="", clip_type="", dtype="auto",
                   shutdown_confirm=False):
        client = RemoteCLIPClient(base_url.strip().rstrip("/"), auth_token)
        params = {"model_name": model_name, "kind": kind, "source": source,
                  "components": components, "sources": sources, "clip_type": clip_type,
                  "dtype": dtype, "shutdown_confirm": shutdown_confirm}
        result = client.control(action, params)
        lines = [f"{action}:"]
        formatted = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        lines.extend(formatted.splitlines())
        print("\n".join(lines))
        return {"ui": {"text": lines}}


NODE_CLASS_MAPPINGS = {
    "LoadRemoteCLIPColab": LoadRemoteCLIPColab,
    "LoraLoaderCLIPOnlyColab": LoraLoaderCLIPOnlyColab,
    "RemoteCLIPController": RemoteCLIPController,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadRemoteCLIPColab": "Load Remote CLIP (Colab)",
    "LoraLoaderCLIPOnlyColab": "LoraLoader CLIP Only (Colab)",
    "RemoteCLIPController": "Remote CLIP Controller",
}
