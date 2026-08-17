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
                "base_url": ("STRING", {"default": DEFAULT_PORT_HINT}),
                "auth_token": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_remote"
    CATEGORY = "Remote CLIP"
    DESCRIPTION = ("Connect to a Remote CLIP Colab worker over HTTPS. The worker "
                   "prints its base_url and auth_token at startup.")

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
                "CLIP": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0,
                                            "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_lora"
    CATEGORY = "Remote CLIP"

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
                "base_url": ("STRING", {"default": DEFAULT_PORT_HINT}),
                "action": (cls.ACTIONS, {"default": "status"}),
            },
            "optional": {
                "auth_token": ("STRING", {"default": ""}),
                "model_name": ("STRING", {"default": "", "placeholder": "engine name"}),
                "kind": (cls.KINDS, {"default": ""}),
                "source": ("STRING", {"default": "", "placeholder": "HF repo id or safetensors path"}),
                "components": ("STRING", {"default": "",
                                          "placeholder": '{"clip_l": "...", "t5": "..."}'}),
                "sources": ("STRING", {"default": "",
                                       "placeholder": "path1.safetensors + path2.safetensors"}),
                "clip_type": ("STRING", {"default": "",
                                         "placeholder": "comfy CLIPType name for kind=native"}),
                "dtype": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "run_action"
    OUTPUT_NODE = True
    CATEGORY = "Remote CLIP"

    def run_action(self, base_url, action, auth_token="", model_name="", kind="",
                   source="", components="", sources="", clip_type="", dtype="auto"):
        client = RemoteCLIPClient(base_url.strip().rstrip("/"), auth_token)
        params = {"model_name": model_name, "kind": kind, "source": source,
                  "components": components, "sources": sources, "clip_type": clip_type,
                  "dtype": dtype}
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
