"""Text encoder engines for the Remote CLIP Colab runtime.

Pure HF transformers implementations that replicate ComfyUI's text encoding
behavior: emphasis weighted encoding, hidden-layer selection, EOS pooling,
composite encoders (SDXL / SD3 / Flux), Qwen-Image template stripping, kohya
style LoRA injection, embedding caches, and cuda/tpu/cpu device handling.
"""
import json
import os
import re
import threading
import time
import zipfile
from collections import OrderedDict

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

import engines_native

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

# TPU: persist compiled executables so XLA recompiles happen once per machine
# (across worker restarts), not once per process. Enabled via --xla-cache.
XLA_CACHE_DIR = None


def enable_xla_persistent_cache(path):
    global XLA_CACHE_DIR
    if xm is None:
        raise RuntimeError("--xla-cache requires torch_xla (TPU runtime)")
    os.makedirs(path, exist_ok=True)
    XLA_CACHE_DIR = path
    os.environ["XLA_PERSISTENT_CACHE_PATH"] = path


def _hf_attn_kwargs(device):
    """Attention implementation for HF from_pretrained loads. XLA compiles its
    own fused attention, so only CUDA gets an explicit kernel choice."""
    if device is not None and device.type == "cuda":
        return {"attn_implementation": os.environ.get("RCP_ATTENTION_HF", "sdpa")}
    return {}

from transformers import (
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    Qwen2Tokenizer,
    T5Config,
    T5EncoderModel,
    T5TokenizerFast,
)

PROTOCOL_VERSION = 3
EMBED_CACHE_SIZE = 64

TOKENIZER_DEFAULTS = {
    "clip_l": "openai/clip-vit-large-patch14",
    "clip_g": "openai/clip-vit-large-patch14",
    "t5": "google/t5-v1_1-base",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
}

# Length buckets for dynamic-length encoders on TPU (XLA recompiles per shape,
# so lengths are rounded up; padding is attention-mask protected).
TPU_BUCKETS = [77, 128, 256, 384, 512, 768, 1024, 1536, 2048]


def log(msg):
    print(f"[engines {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# device / dtype helpers
# ---------------------------------------------------------------------------

def detect_device(requested="auto"):
    if requested == "tpu" and xm is None:
        raise RuntimeError("torch_xla is not installed; cannot use the tpu device")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if xm is not None:
            try:
                return xm.xla_device()
            except Exception:
                pass
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    if requested == "tpu":
        return xm.xla_device()
    return torch.device(requested)


def is_xla(device):
    return device.type == "xla"


def pick_dtype(device):
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16


def mark_step(device):
    if device.type == "xla":
        xm.mark_step()


# ---------------------------------------------------------------------------
# emphasis parsing (port of comfy.sd1_clip)
# ---------------------------------------------------------------------------

def parse_parentheses(string):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def token_weights(string, current_weight):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ')' and x[0] == '(':
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx + 1:])
                    x = x[:xx]
                except ValueError:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out


def escape_important(text):
    return text.replace("\\)", "\0\1").replace("\\(", "\0\2")


def unescape_important(text):
    return text.replace("\0\1", ")").replace("\0\2", "(")


# ---------------------------------------------------------------------------
# textual inversion embeddings (port of comfy.sd1_clip.load_embed)
# ---------------------------------------------------------------------------

def _safe_load_embed_zip(embed_path):
    with zipfile.ZipFile(embed_path) as myzip:
        names = [n for n in myzip.namelist() if "data/" in n]
        names.reverse()
        for n in names:
            with myzip.open(n) as myfile:
                data = myfile.read()
                number = len(data) // 4
                if number < 768:
                    continue
                length_embed = 1024
                if number % 768 == 0:
                    length_embed = 768
                return torch.frombuffer(data, dtype=torch.float).reshape(
                    (number // length_embed, length_embed)).clone()
    return None


def load_embed(embedding_name, embedding_dirs, embedding_size, embed_key=None):
    for embed_dir in embedding_dirs:
        embed_dir = os.path.abspath(embed_dir)
        embed_path = os.path.abspath(os.path.join(embed_dir, embedding_name))
        if os.path.commonpath((embed_dir, embed_path)) != embed_dir:
            continue
        valid_file = None
        if os.path.isfile(embed_path):
            valid_file = embed_path
        else:
            for ext in (".safetensors", ".pt", ".bin"):
                if os.path.isfile(embed_path + ext):
                    valid_file = embed_path + ext
                    break
        if valid_file is None:
            continue
        if valid_file.lower().endswith(".safetensors"):
            embed = load_safetensors(valid_file, device="cpu")
        else:
            try:
                embed = torch.load(valid_file, weights_only=True, map_location="cpu")
            except Exception:
                embed_out = _safe_load_embed_zip(valid_file)
                if embed_out is not None:
                    return embed_out
                continue
        if "string_to_param" in embed:
            return next(iter(embed["string_to_param"].values()))
        if embed_key is not None and embed_key in embed:
            return embed[embed_key]
        return next(iter(embed.values()))
    return None


# ---------------------------------------------------------------------------
# weighted tokenizer (port of comfy.sd1_clip.SDTokenizer.tokenize_with_weights)
# ---------------------------------------------------------------------------

class WeightedTokenizer:
    """Splits text into sections of (token_id|tensor, weight) pairs following
    ComfyUI's SDTokenizer: emphasis parsing, 77-token chunking for CLIP,
    min-length padding for T5, textual inversion embedding tokens."""

    def __init__(self, tokenizer, max_length=77, pad_with_end=True,
                 pad_to_max_length=True, min_length=None, start_token=None,
                 end_token=None, has_start_token=True, has_end_token=True,
                 pad_token=None, disable_weights=False, embedding_size=768,
                 embedding_key="clip_l", embedding_dirs=None, max_word_length=8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        self.disable_weights = disable_weights
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key
        self.embedding_dirs = embedding_dirs or []
        self.max_word_length = max_word_length
        self.embedding_identifier = "embedding:"

        self.start_token = start_token
        self.end_token = end_token
        if self.start_token is None and has_start_token:
            specials = self.tokenizer("", add_special_tokens=True)["input_ids"]
            self.start_token = specials[0] if specials else None
        if self.end_token is None and has_end_token:
            specials = self.tokenizer("", add_special_tokens=True)["input_ids"]
            if len(specials) >= 2:
                self.end_token = specials[1]
            elif specials:
                self.end_token = specials[0]
        if pad_token is not None:
            self.pad_token = pad_token
        elif pad_with_end:
            self.pad_token = self.end_token
        else:
            self.pad_token = 0

    def _pad_tokens(self, tokens, amount):
        tokens.extend([(self.pad_token, 1.0)] * amount)

    def _try_get_embedding(self, embedding_name):
        split_embed = embedding_name.split()
        name = split_embed[0]
        leftover = " ".join(split_embed[1:])
        match = re.search(r"[<\[]", name)
        if match is not None:
            leftover = name[match.start():] + (" " + leftover if leftover else "")
            name = name[:match.start()]
        embed = load_embed(name, self.embedding_dirs, self.embedding_size, self.embedding_key)
        if embed is None:
            stripped = name.strip(",")
            if len(stripped) < len(name):
                embed = load_embed(stripped, self.embedding_dirs, self.embedding_size, self.embedding_key)
                return embed, name, "{} {}".format(name[len(stripped):], leftover)
        return embed, name, leftover

    def tokenize_with_weights(self, text, min_length=None, disable_weights=None, **kwargs):
        min_length = min_length if min_length is not None else self.min_length
        text = escape_important(text)
        use_disable = disable_weights if disable_weights is not None else self.disable_weights
        parsed_weights = [(text, 1.0)] if use_disable else token_weights(text, 1.0)

        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment)
            split = re.split(r"(?<=\s){}".format(re.escape(self.embedding_identifier)), to_tokenize)
            pieces = [split[0]]
            for i in range(1, len(split)):
                pieces.append("{}{}".format(self.embedding_identifier, split[i]))
            pieces = [x for x in pieces if x != ""]
            for word in pieces:
                if word.startswith(self.embedding_identifier) and self.embedding_dirs:
                    embedding_name = word[len(self.embedding_identifier):].strip("\n")
                    embed, embedding_name, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        log(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                ids = self.tokenizer(word, add_special_tokens=False)["input_ids"]
                tokens.append([(t, weight) for t in ids])

        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0))
        batched_tokens.append(batch)
        for t_group in tokens:
            is_large = len(t_group) >= self.max_word_length
            has_end = 1 if self.end_token is not None else 0
            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - has_end:
                    remaining = self.max_length - len(batch) - has_end
                    if is_large:
                        batch.extend(t_group[:remaining])
                        if self.end_token is not None:
                            batch.append((self.end_token, 1.0))
                        t_group = t_group[remaining:]
                    else:
                        if self.end_token is not None:
                            batch.append((self.end_token, 1.0))
                        if self.pad_to_max_length:
                            self._pad_tokens(batch, remaining)
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0))
                    batched_tokens.append(batch)
                else:
                    batch.extend(t_group)
                    t_group = []

        if self.end_token is not None:
            batch.append((self.end_token, 1.0))
        if self.pad_to_max_length and len(batch) < self.max_length:
            self._pad_tokens(batch, self.max_length - len(batch))
        if min_length is not None and len(batch) < min_length:
            self._pad_tokens(batch, min_length - len(batch))
        return batched_tokens

    def gen_empty_tokens(self, length):
        out = []
        if self.start_token is not None:
            out.append(self.start_token)
        if self.end_token is not None:
            out.append(self.end_token)
        out += [self.pad_token] * (length - len(out))
        return out


# ---------------------------------------------------------------------------
# weighted multi-pass encoding (port of comfy ClipTokenWeightEncoder)
# ---------------------------------------------------------------------------

def split_section(section):
    """Section -> (ids with -1 placeholders at tensor positions, inserts,
    weights)."""
    ids = []
    inserts = []
    weights = []
    for tok, weight in section:
        if isinstance(tok, torch.Tensor):
            inserts.append((len(ids), tok))
            ids.append(-1)
        else:
            ids.append(int(tok))
        weights.append(weight)
    return ids, inserts, weights


def encode_weighted(sections, forward_rows, tokenizer):
    to_encode = []
    all_inserts = []
    all_weights = []
    max_token_len = 0
    has_weights = False
    for section in sections:
        ids, inserts, weights = split_section(section)
        max_token_len = max(len(ids), max_token_len)
        has_weights = has_weights or any(w != 1.0 for w in weights)
        to_encode.append(ids)
        all_inserts.append(inserts)
        all_weights.append(weights)

    n_sections = len(to_encode)
    if has_weights or n_sections == 0:
        to_encode.append(tokenizer.gen_empty_tokens(max_token_len))
        all_inserts.append([])
        all_weights.append([])

    out, pooled = forward_rows(to_encode, all_inserts, max_token_len)

    if pooled is not None:
        first_pooled = pooled[0:1].float()
    else:
        first_pooled = None

    output = []
    for k in range(n_sections):
        z = out[k:k + 1]
        if has_weights:
            z_empty = out[-1]
            for i in range(len(z)):
                for j in range(len(z[i])):
                    if j < len(all_weights[k]):
                        weight = all_weights[k][j]
                        if weight != 1.0:
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
        output.append(z)

    if len(output) == 0:
        cond = out[-1:].float()
    else:
        cond = torch.cat(output, dim=-2).float()
    return cond, first_pooled


def pad_rows(rows, pad_token, lengths):
    return [row + [pad_token] * (length - len(row)) for row, length in zip(rows, lengths)]


# ---------------------------------------------------------------------------
# HF model loading (HF repo or comfy-format safetensors)
# ---------------------------------------------------------------------------

CLIP_L_CONFIG = dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                     intermediate_size=3072, max_position_embeddings=77,
                     vocab_size=49408, hidden_act="quick_gelu", eos_token_id=49407)
CLIP_G_CONFIG = dict(hidden_size=1280, num_hidden_layers=32, num_attention_heads=20,
                     intermediate_size=5120, max_position_embeddings=77,
                     vocab_size=49408, hidden_act="gelu", eos_token_id=49407)
T5_XXL_CONFIG = dict(d_model=4096, d_kv=64, num_heads=64, d_ff=10240,
                     num_layers=24, vocab_size=32128,
                     feed_forward_proj="gated-gelu", dense_act_fn="gelu_pytorch_tanh",
                     layer_norm_epsilon=1e-6, is_gated_act=True,
                     relative_attention_num_buckets=32)


def _is_hf_dir(source):
    return os.path.isdir(source) and os.path.isfile(os.path.join(source, "config.json"))


def _looks_like_path(source):
    return (os.sep in source or "/" in source or "\\" in source
            or source.startswith(".") or source.endswith(".safetensors"))


def _load_safetensors(source):
    if os.path.isdir(source):
        sd = {}
        for f in sorted(os.listdir(source)):
            if f.endswith(".safetensors"):
                sd.update(load_safetensors(os.path.join(source, f), device="cpu"))
        return sd
    return load_safetensors(source, device="cpu")


def load_clip_text_model(source, variant, dtype, device=None):
    """variant: clip_l / clip_g. Accepts HF repo, HF-format local dir, or a
    comfy-format safetensors file/dir (keys prefixed with 'transformer.')."""
    if _looks_like_path(source) and not os.path.exists(source):
        raise FileNotFoundError(f"model file not found on worker: {source}")
    cfg = CLIP_L_CONFIG if variant == "clip_l" else CLIP_G_CONFIG
    if os.path.exists(source) and not _is_hf_dir(source):
        sd = _load_safetensors(source)
        if not any(k.startswith("text_model.") for k in sd):
            sd = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
        config = CLIPTextConfig(**cfg, projection_dim=cfg["hidden_size"])
        model = CLIPTextModelWithProjection(config)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            log(f"clip {variant}: ignoring unexpected keys: {unexpected[:4]}")
        if any("text_model.encoder" in k for k in missing):
            raise RuntimeError(f"clip {variant}: checkpoint missing encoder weights: {missing[:4]}")
        model.has_projection = "text_projection.weight" not in missing
    else:
        model = CLIPTextModelWithProjection.from_pretrained(source, torch_dtype=dtype,
                                                            **_hf_attn_kwargs(device))
        model.has_projection = True
    model = model.to(dtype)
    model.eval()
    return model


def load_t5_model(source, dtype, device=None):
    if _looks_like_path(source) and not os.path.exists(source):
        raise FileNotFoundError(f"model file not found on worker: {source}")
    if os.path.exists(source) and not _is_hf_dir(source):
        sd = _load_safetensors(source)
        if not any(k.startswith("encoder.") or k.startswith("shared.") for k in sd):
            sd = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
        model = T5EncoderModel(T5Config(**T5_XXL_CONFIG))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            log(f"t5: ignoring unexpected keys: {unexpected[:4]}")
        if any("encoder.block" in k for k in missing):
            raise RuntimeError(f"t5: checkpoint missing encoder weights: {missing[:4]}")
    else:
        model = T5EncoderModel.from_pretrained(source, torch_dtype=dtype,
                                               **_hf_attn_kwargs(device))
    model = model.to(dtype)
    model.eval()
    return model


def get_tokenizer(kind, override=None):
    src = override or TOKENIZER_DEFAULTS[kind]
    if kind in ("clip_l", "clip_g"):
        from transformers import CLIPTokenizer
        return CLIPTokenizer.from_pretrained(src)
    if kind == "t5":
        return T5TokenizerFast.from_pretrained(src)
    if kind == "qwen":
        return Qwen2Tokenizer.from_pretrained(src)
    return AutoTokenizer.from_pretrained(src)


# ---------------------------------------------------------------------------
# manual CLIP forward (only used with textual-inversion embeds, because HF
# CLIPTextTransformer does not accept inputs_embeds)
# ---------------------------------------------------------------------------

def clip_forward_manual(text_model, embeds):
    tm = text_model.text_model
    bsz, seq_len, dim = embeds.shape
    minv = torch.finfo(embeds.dtype).min
    causal = torch.full((seq_len, seq_len), minv, dtype=embeds.dtype,
                        device=embeds.device).triu(1)[None, None, :, :]
    x = embeds
    pre_hidden = None
    n_layers = len(tm.encoder.layers)
    for li, layer in enumerate(tm.encoder.layers):
        sa = layer.self_attn
        heads = sa.num_heads
        head_dim = x.shape[-1] // heads

        def split_heads(t):
            return t.view(bsz, seq_len, heads, head_dim).transpose(1, 2)

        h = layer.layer_norm1(x)
        q, k, v = split_heads(sa.q_proj(h)), split_heads(sa.k_proj(h)), split_heads(sa.v_proj(h))
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=causal.expand(bsz, -1, -1, -1))
        o = o.transpose(1, 2).reshape(bsz, seq_len, -1)
        x = x + sa.out_proj(o)
        x = x + layer.mlp(layer.layer_norm2(x))
        if li == n_layers - 2:
            pre_hidden = x.clone()
    final = tm.final_layer_norm(x)
    return final, pre_hidden


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------

class ClipVariantEngine:
    """Single CLIP text encoder with a comfy-compatible variant config.

    variants:
      sd1_clip_l   last layer + final norm; pooled = unprojected EOS output
      sdxl_clip_l  hidden -2 without final norm; pooled = unprojected EOS output
      clip_g       hidden -2 without final norm; pooled = projected EOS output
    """

    VARIANTS = ("sd1_clip_l", "sdxl_clip_l", "clip_g")

    def __init__(self, name, variant, source, device, dtype, tokenizer_src=None, embedding_dirs=None):
        assert variant in self.VARIANTS
        self.name = name
        self.kind = "clip"
        self.variant = variant
        self.device = device
        self.dtype = dtype
        self.model = load_clip_text_model(source, "clip_l" if variant.endswith("clip_l") else "clip_g",
                                          dtype, device)
        self.model.to(device)
        tok_kind = "clip_l" if variant.endswith("clip_l") else "clip_g"
        self.tokenizer = WeightedTokenizer(
            get_tokenizer(tok_kind, tokenizer_src), max_length=77,
            pad_with_end=(tok_kind == "clip_l"), pad_to_max_length=True,
            start_token=49406, end_token=49407,
            pad_token=None if tok_kind == "clip_l" else 0,
            embedding_size=1280 if tok_kind == "clip_g" else 768,
            embedding_key=tok_kind, embedding_dirs=embedding_dirs)
        self.hidden_size = 768 if tok_kind == "clip_l" else 1280
        self.lora = LoraPatcher()

    def _forward_rows(self, rows, inserts, max_len):
        lengths = [max_len - len(ins) for ins in inserts]
        rows = pad_rows(rows, self.tokenizer.pad_token, lengths)
        for row, ins in zip(rows, inserts):
            for pos, _vec in ins:
                row[pos] = self.tokenizer.pad_token
        input_ids = torch.tensor(rows, device=self.device, dtype=torch.long)

        if not any(inserts):
            outputs = self.model.text_model(input_ids=input_ids, output_hidden_states=True)
            final = outputs.last_hidden_state
            hidden_states = outputs.hidden_states
            pooled = outputs.pooler_output
            hidden = final if self.variant == "sd1_clip_l" else hidden_states[-2]
        else:
            embeds = self.model.text_model.embeddings(input_ids=input_ids)
            for i, ins in enumerate(inserts):
                for pos, vec in ins:
                    embeds[i, pos] = vec.to(device=embeds.device, dtype=embeds.dtype)
            final, pre_hidden = clip_forward_manual(self.model, embeds)
            hidden = final if self.variant == "sd1_clip_l" else pre_hidden
            eos = self.model.config.eos_token_id or 49407
            eos_pos = (input_ids == eos).int().argmax(dim=-1)
            pooled = final[torch.arange(final.shape[0], device=final.device), eos_pos]

        if self.variant == "clip_g" and getattr(self.model, "has_projection", True):
            pooled = pooled @ self.model.text_projection.weight.t().to(pooled.dtype)
        return hidden.float(), pooled.float()

    def encode(self, text, kwargs, lora_stack):
        min_length = kwargs.get("min_length")
        disable_weights = kwargs.get("disable_weights")
        sections = self.tokenizer.tokenize_with_weights(text, min_length=min_length,
                                                        disable_weights=disable_weights)
        has_embeds = any(isinstance(t, torch.Tensor)
                         for section in sections for t, _w in section)
        if has_embeds and self.variant != "sd1_clip_l":
            raise RuntimeError(
                "textual inversion embeddings are only supported for the clip_l (last layer) variant")
        cond, pooled = encode_weighted(sections, self._forward_rows, self.tokenizer)
        return {"cond": cond, "pooled_output": pooled}

    def generate(self, text, kwargs, gen_kwargs):
        raise RuntimeError(
            f"engine '{self.name}' cannot generate text; load a causal_lm engine for that")


class T5Engine:
    def __init__(self, name, source, device, dtype, min_length=77, tokenizer_src=None, embedding_dirs=None):
        self.name = name
        self.kind = "t5"
        self.device = device
        self.dtype = dtype
        self.min_length = min_length
        self.model = load_t5_model(source, dtype, device)
        self.model.to(device)
        self.tokenizer = WeightedTokenizer(
            get_tokenizer("t5", tokenizer_src), max_length=99999999,
            pad_with_end=False, pad_to_max_length=False, min_length=min_length,
            has_start_token=False, start_token=None, end_token=1, pad_token=0,
            embedding_size=4096, embedding_key="t5xxl", embedding_dirs=embedding_dirs)
        self.lora = LoraPatcher()

    def _forward_rows(self, rows, inserts, max_len):
        length = max_len
        if is_xla(self.device):
            length = next((b for b in TPU_BUCKETS if b >= max_len), max_len)
        lengths = [length - len(ins) for ins in inserts]
        rows = pad_rows(rows, 0, lengths)
        for row, ins in zip(rows, inserts):
            for pos, _vec in ins:
                row[pos] = 0
        input_ids = torch.tensor(rows, device=self.device, dtype=torch.long)
        # comfy's T5 text encoders run with enable_attention_masks=False: pad
        # positions stay attended, so no attention mask is passed here.
        if any(inserts):
            embeds = self.model.shared(input_ids)
            for i, ins in enumerate(inserts):
                for pos, vec in ins:
                    embeds[i, pos] = vec.to(device=embeds.device, dtype=embeds.dtype)
            hidden = self.model.encoder(inputs_embeds=embeds).last_hidden_state
        else:
            hidden = self.model(input_ids=input_ids).last_hidden_state
        return hidden.float(), None

    def encode(self, text, kwargs, lora_stack):
        min_length = kwargs.get("min_length", self.min_length)
        disable_weights = kwargs.get("disable_weights")
        sections = self.tokenizer.tokenize_with_weights(text, min_length=max(min_length or 1, 1),
                                                        disable_weights=disable_weights)
        cond, pooled = encode_weighted(sections, self._forward_rows, self.tokenizer)
        return {"cond": cond, "pooled_output": None}

    def generate(self, text, kwargs, gen_kwargs):
        raise RuntimeError(
            f"engine '{self.name}' cannot generate text; load a causal_lm engine for that")


class CompositeEngine:
    """Combines sub-engine outputs the way comfy's SDXL / SD3 / Flux models do."""

    PART_KINDS = {
        "sdxl": {"clip_l": "sdxl_clip_l", "clip_g": "clip_g"},
        "sd3": {"clip_l": "sdxl_clip_l", "clip_g": "clip_g", "t5": "t5"},
        "flux": {"clip_l": "clip_l", "t5": "t5"},
    }

    def __init__(self, name, kind, parts):
        self.name = name
        self.kind = kind
        self.parts = parts
        self.lora = LoraPatcher()

    @property
    def device(self):
        return next(iter(self.parts.values())).device

    @property
    def dtype(self):
        return next(iter(self.parts.values())).dtype

    def encode(self, text, kwargs, lora_stack):
        if self.kind == "flux":
            t5_kwargs = {**kwargs, "min_length": max(int(kwargs.get("min_length") or 0), 256)}
            t5 = self.parts["t5"].encode(text, t5_kwargs, lora_stack)
            l = self.parts["clip_l"].encode(text, kwargs, lora_stack)
            return {"cond": t5["cond"], "pooled_output": l["pooled_output"]}
        outs = {k: part.encode(text, kwargs, lora_stack) for k, part in self.parts.items()}
        if self.kind == "sdxl":
            l_cond, g_cond = outs["clip_l"]["cond"], outs["clip_g"]["cond"]
            cut = min(l_cond.shape[1], g_cond.shape[1])
            cond = torch.cat([l_cond[:, :cut], g_cond[:, :cut]], dim=-1)
            return {"cond": cond, "pooled_output": outs["clip_g"]["pooled_output"]}
        if self.kind == "sd3":
            l_cond, g_cond, t5_cond = outs["clip_l"]["cond"], outs["clip_g"]["cond"], outs["t5"]["cond"]
            cut = min(l_cond.shape[1], g_cond.shape[1])
            lg = torch.cat([l_cond[:, :cut], g_cond[:, :cut]], dim=-1)
            lg = torch.nn.functional.pad(lg, (0, 4096 - lg.shape[-1]))
            l_pooled = outs["clip_l"]["pooled_output"]
            g_pooled = outs["clip_g"]["pooled_output"]
            if l_pooled is None:
                l_pooled = torch.zeros((1, 768))
            if g_pooled is None:
                g_pooled = torch.zeros((1, 1280))
            pooled = torch.cat((l_pooled, g_pooled), dim=-1)
            return {"cond": torch.cat([lg, t5_cond], dim=-2), "pooled_output": pooled}
        raise RuntimeError(f"unknown composite kind: {self.kind}")

    def generate(self, text, kwargs, gen_kwargs):
        raise RuntimeError(
            f"engine '{self.name}' cannot generate text; load a causal_lm engine for that")


# ---------------------------------------------------------------------------
# Qwen-Image engine
# ---------------------------------------------------------------------------

QWEN_TEXT_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
QWEN_IMAGE_TEMPLATE = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
QWEN_IM_START = 151644
QWEN_USER_ID = 872
QWEN_NEWLINE_ID = 198
QWEN_IMAGE_PAD = 151655
QWEN_PAD = 151643


def comfy_images_to_list(images):
    """comfy image tensor(s) [B,H,W,C] float 0..1 -> list of HWC uint8 arrays."""
    out = []
    for img in images:
        if not isinstance(img, torch.Tensor):
            continue
        if img.dim() == 4:
            for i in range(img.shape[0]):
                out.append((img[i].cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8))
        else:
            out.append((img.cpu().float().clamp(0, 1).numpy() * 255).round().astype(np.uint8))
    return out


class QwenImageEngine:
    def __init__(self, name, source, device, dtype, tokenizer_src=None):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.name = name
        self.kind = "qwen_image"
        self.device = device
        self.dtype = dtype
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(source, torch_dtype=dtype,
                                                                     **_hf_attn_kwargs(device))
        self.model.to(device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(source)
        self.tokenizer = WeightedTokenizer(
            get_tokenizer("qwen", tokenizer_src), max_length=99999999,
            pad_with_end=False, pad_to_max_length=False, min_length=1,
            has_start_token=False, has_end_token=False, start_token=None,
            end_token=None, pad_token=QWEN_PAD, disable_weights=True,
            embedding_size=3584, embedding_key="qwen25_7b", embedding_dirs=None)
        self.lora = LoraPatcher()

    def _strip_template(self, token_ids, out, attn):
        template_end = -1
        count = 0
        for i, v in enumerate(token_ids):
            if v == QWEN_IM_START:
                template_end = i
                count += 1
                if count >= 2:
                    break
        if template_end == -1:
            return out, attn
        if len(token_ids) > template_end + 3 and \
                token_ids[template_end + 1] == QWEN_USER_ID and \
                token_ids[template_end + 2] == QWEN_NEWLINE_ID:
            template_end += 3
        return out[:, template_end:], attn[:, template_end:]

    def encode(self, text, kwargs, lora_stack):
        images = kwargs.get("images") or []
        if text.startswith("<|im_start|>") or text.startswith("<|start_header_id|>"):
            llama_text = text
        else:
            template = kwargs.get("llama_template")
            if template is not None:
                llama_text = template.format(text)
            elif images:
                llama_text = QWEN_IMAGE_TEMPLATE.format(text)
            else:
                llama_text = QWEN_TEXT_TEMPLATE.format(text)
        if kwargs.get("prevent_empty_text") and text == "":
            llama_text = (QWEN_TEXT_TEMPLATE.format(" ") if not images
                          else QWEN_IMAGE_TEMPLATE.format(" "))

        ids = self.tokenizer.tokenizer(llama_text, add_special_tokens=False)["input_ids"]
        model_inputs = None
        if images:
            pil_images = comfy_images_to_list(images)
            image_inputs = self.processor.image_processor(images=pil_images, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
            grid_thw = image_inputs["image_grid_thw"].to(self.device)
            merge_size = getattr(self.processor.image_processor, "merge_size", 2)
            merges = (grid_thw.prod(dim=1) // (merge_size * merge_size)).tolist()
            expanded = []
            merge_iter = iter(merges)
            for tid in ids:
                if tid == QWEN_IMAGE_PAD:
                    expanded.extend([QWEN_IMAGE_PAD] * next(merge_iter))
                else:
                    expanded.append(tid)
            input_ids = torch.tensor([expanded], device=self.device, dtype=torch.long)
            model_inputs = {"input_ids": input_ids,
                            "attention_mask": torch.ones_like(input_ids),
                            "pixel_values": pixel_values, "image_grid_thw": grid_thw}
        else:
            input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
            model_inputs = {"input_ids": input_ids,
                            "attention_mask": torch.ones_like(input_ids)}

        length = model_inputs["input_ids"].shape[1]
        if is_xla(self.device):
            bucket = next((b for b in TPU_BUCKETS if b >= length), None)
            if bucket and bucket > length:
                pad = torch.full((1, bucket - length), QWEN_PAD, device=self.device, dtype=torch.long)
                model_inputs["input_ids"] = torch.cat([model_inputs["input_ids"], pad], dim=1)
                model_inputs["attention_mask"] = torch.cat(
                    [model_inputs["attention_mask"], torch.zeros_like(pad)], dim=1)

        with torch.no_grad():
            outputs = self.model.model(**model_inputs)
        hidden = outputs.last_hidden_state[:, :length].float()
        attn = model_inputs["attention_mask"][:, :length]
        out, attn = self._strip_template(ids, hidden, attn)
        result = {"cond": out}
        if attn.sum() != attn.numel():
            result["attention_mask"] = attn
        return result

    def generate(self, text, kwargs, gen_kwargs):
        raise RuntimeError(
            f"engine '{self.name}' cannot generate text; load a causal_lm engine for that")


# ---------------------------------------------------------------------------
# causal LM engine (Generate Text)
# ---------------------------------------------------------------------------

class CausalLMEngine:
    def __init__(self, name, source, device, dtype, tokenizer_src=None):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        self.name = name
        self.kind = "causal_lm"
        self.device = device
        self.dtype = dtype
        config = AutoConfig.from_pretrained(source)
        try:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(source, torch_dtype=dtype,
                                                                     **_hf_attn_kwargs(device))
        except (ValueError, OSError):
            self.model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype,
                                                              **_hf_attn_kwargs(device))
        self.model.to(device)
        self.model.eval()
        self.processor = None
        try:
            self.processor = AutoProcessor.from_pretrained(source)
        except Exception:
            pass
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src or source)
        self.model_type = getattr(config, "model_type", "")
        self.lora = LoraPatcher()

    def _build_inputs(self, text, kwargs):
        images = kwargs.get("image")
        video = kwargs.get("video")
        skip_template = kwargs.get("skip_template", False)
        if kwargs.get("audio") is not None:
            raise RuntimeError("audio inputs are not supported by the causal_lm engine yet")

        image_list = []
        if isinstance(images, torch.Tensor):
            image_list.extend(comfy_images_to_list([images]))
        if isinstance(video, torch.Tensor):
            frames = video if video.dim() == 4 else video.unsqueeze(0)
            sampled = frames[::24]  # 24 fps -> 1 fps like comfy
            image_list.extend(comfy_images_to_list([sampled]))
        if image_list and self.model_type.startswith("gemma"):
            text = text.replace("<image_soft_token>", "<start_of_image>")

        multimodal = bool(image_list) and self.processor is not None \
            and hasattr(self.processor, "image_processor")
        if multimodal:
            if skip_template:
                prompt = text
            else:
                messages = [{"role": "user", "content":
                             [{"type": "image"}] * len(image_list) + [{"type": "text", "text": text}]}]
                prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[prompt], images=image_list, return_tensors="pt")
        else:
            if skip_template or self.tokenizer.chat_template is None:
                prompt = text
            else:
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def generate(self, text, kwargs, gen_kwargs):
        inputs = self._build_inputs(text, kwargs)
        do_sample = bool(gen_kwargs.get("do_sample", True))
        max_length = int(gen_kwargs.get("max_length", 256))
        temperature = float(gen_kwargs.get("temperature", 1.0))
        top_k = int(gen_kwargs.get("top_k", 50))
        top_p = float(gen_kwargs.get("top_p", 0.95))
        min_p = float(gen_kwargs.get("min_p", 0.0))
        repetition_penalty = float(gen_kwargs.get("repetition_penalty", 1.0))
        presence_penalty = float(gen_kwargs.get("presence_penalty", 0.0))
        seed = gen_kwargs.get("seed")

        gen_args = {"max_new_tokens": max_length, "do_sample": do_sample}
        if do_sample:
            gen_args.update(temperature=max(temperature, 1e-4), top_k=max(top_k, 0),
                            top_p=top_p, min_p=min_p)
        if repetition_penalty != 1.0:
            gen_args["repetition_penalty"] = repetition_penalty
        if presence_penalty:
            gen_args["presence_penalty"] = presence_penalty

        if seed is not None:
            torch.manual_seed(int(seed) % (2 ** 63))

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_args)
        new_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def encode(self, text, kwargs, lora_stack):
        raise RuntimeError(
            "this engine only supports text generation; connect it to Generate Text style nodes")


# ---------------------------------------------------------------------------
# LoRA injection (kohya format)
# ---------------------------------------------------------------------------

LORA_PREFIXES = [
    ("lora_te1_", "clip_l"), ("lora_te1.", "clip_l"),
    ("lora_te2_", "clip_g"), ("lora_te2.", "clip_g"),
    ("lora_te_", None), ("lora_te.", None),
    ("lora_t5_", "t5"), ("lora_t5.", "t5"),
    ("lora_llm_", None), ("lora_llm.", None),
]


def parse_lora_file(path):
    """-> {role: {module_key: {"up","down","alpha"}}}"""
    sd = _load_safetensors(path)
    lora = {}
    unknown = 0
    for key, value in sd.items():
        if key.endswith(".alpha"):
            continue
        module_part = None
        role = None
        for prefix, prefix_role in LORA_PREFIXES:
            if key.startswith(prefix):
                module_part = key[len(prefix):]
                role = prefix_role
                break
        if module_part is None and (".lora_up.weight" in key or ".lora_down.weight" in key):
            module_part = key
        if module_part is None or (".lora_up.weight" not in module_part
                                   and ".lora_down.weight" not in module_part):
            unknown += 1
            continue
        module_key = module_part.split(".lora_")[0]
        entry = lora.setdefault(role or "*", {}).setdefault(module_key, {})
        if ".lora_up.weight" in module_part:
            entry["up"] = value.float()
            alpha_key = key.replace(".lora_up.weight", ".alpha")
            if alpha_key in sd:
                entry["alpha"] = float(sd[alpha_key])
        else:
            entry["down"] = value.float()
    if unknown:
        log(f"lora {os.path.basename(path)}: ignoring {unknown} unrecognized keys")
    return lora


def _module_map(model):
    out = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            out[name] = name
            out[name.replace(".", "_")] = name
    return out


def _resolve_module(model, module_key):
    modules = _module_map(model)
    if module_key in modules:
        return modules[module_key]
    normalized = module_key.replace(".", "_")
    if normalized in modules:
        return modules[normalized]
    dotted_guess = module_key.replace("_", ".")
    if dotted_guess in modules:
        return modules[dotted_guess]
    return None


class LoraPatcher:
    """Applies kohya lora stacks by in-place weight surgery. Pristine weights
    of touched modules are kept on CPU and restored before each re-apply."""

    def __init__(self):
        self.applied_sig = None
        self._pristine = {}  # model_id -> {module_name: cpu tensor}

    def apply(self, engine, lora_stack, registry):
        sig = json.dumps(lora_stack)
        if sig == self.applied_sig:
            return
        self.revert(registry)
        if not lora_stack:
            return
        deltas = {}
        for lora_name, strength in lora_stack:
            if strength == 0:
                continue
            path = registry.resolve_lora(lora_name)
            if path is None:
                raise FileNotFoundError(f"LoRA not found on worker: {lora_name}")
            lora = parse_lora_file(path)
            for role, modules in lora.items():
                for module_key, entry in modules.items():
                    if "up" not in entry or "down" not in entry:
                        continue
                    for model, model_id in registry.models_for_role(engine, role):
                        name = _resolve_module(model, module_key)
                        if name is None:
                            continue
                        rank = entry["down"].shape[0]
                        scale = entry.get("alpha", rank) / rank
                        delta = (entry["up"] @ entry["down"]) * (scale * strength)
                        key = (model_id, name)
                        deltas[key] = deltas.get(key, 0) + delta
        named_cache = {}
        applied = 0
        for (model_id, name), delta in deltas.items():
            model = registry.model_by_id(model_id)
            if model_id not in named_cache:
                named_cache[model_id] = dict(model.named_modules())
            module = named_cache[model_id][name]
            store = self._pristine.setdefault(model_id, {})
            if name not in store:
                store[name] = module.weight.detach().cpu().clone()
            module.weight.data.copy_(
                store[name].to(module.weight.device, module.weight.dtype)
                + delta.to(module.weight.device, module.weight.dtype))
            applied += 1
        if applied == 0:
            raise RuntimeError(
                f"LoRA {lora_stack[0][0]} matched no modules on the worker; "
                "its keys are not text-encoder LoRA keys the worker understands")
        self.applied_sig = sig

    def revert(self, registry):
        if self.applied_sig is None:
            return
        for model_id, store in self._pristine.items():
            try:
                model = registry.model_by_id(model_id)
            except KeyError:
                continue
            named = dict(model.named_modules())
            for name, weight in store.items():
                if name in named:
                    named[name].weight.data.copy_(
                        weight.to(named[name].weight.device, named[name].weight.dtype))
        self.applied_sig = None


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

class EngineRegistry:
    def __init__(self, loras_dir=None, embeddings_dir=None):
        self.engines = OrderedDict()
        self.default_name = None
        self.loras_dir = loras_dir
        self.embeddings_dir = embeddings_dir
        self.tokenizer_overrides = {}
        self._infer_lock = threading.Lock()
        self._cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def _embedding_dirs(self):
        return [self.embeddings_dir] if self.embeddings_dir else []

    def _tokenizer_for(self, kind, spec_override):
        if spec_override:
            return spec_override
        if kind in self.tokenizer_overrides:
            return self.tokenizer_overrides[kind]
        return self.tokenizer_overrides.get({"sdxl_clip_l": "clip_l"}.get(kind, kind))

    def build_engine(self, spec, _top=True):
        kind = spec.get("kind", "")
        if engines_native.dispatch_is_native(spec):
            engine = engines_native.build_native_engine(spec, self)
            if engine is None:
                raise ValueError(f"unknown engine kind: {kind}")
            engine.load_seconds = engine.load_seconds or 0
            self.engines[spec["name"]] = engine
            if _top and self.default_name is None:
                self.default_name = spec["name"]
            return engine
        name = spec["name"]
        device = detect_device(spec.get("device", "auto"))
        dtype = {"auto": pick_dtype(device), "bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[spec.get("dtype", "auto")]
        tok = self._tokenizer_for(kind, spec.get("tokenizer"))
        embedding_dirs = self._embedding_dirs()
        t0 = time.time()
        if kind in ("clip_l", "sdxl_clip_l", "clip_g"):
            variant = "sd1_clip_l" if kind == "clip_l" else kind
            engine = ClipVariantEngine(name, variant, spec["source"], device, dtype,
                                       tokenizer_src=tok, embedding_dirs=embedding_dirs)
        elif kind == "t5":
            engine = T5Engine(name, spec["source"], device, dtype,
                              min_length=spec.get("min_length", 77), tokenizer_src=tok,
                              embedding_dirs=embedding_dirs)
        elif kind == "qwen_image":
            engine = QwenImageEngine(name, spec["source"], device, dtype, tokenizer_src=tok)
        elif kind == "causal_lm":
            engine = CausalLMEngine(name, spec["source"], device, dtype, tokenizer_src=tok)
        elif kind in ("sdxl", "sd3", "flux"):
            parts = {}
            for role, part_source in spec["components"].items():
                part_name = f"{name}.{role}"
                if part_source in self.engines:
                    part = self.engines[part_source]
                else:
                    part_kind = CompositeEngine.PART_KINDS[kind][role]
                    part = self.build_engine({"name": part_name, "kind": part_kind,
                                              "source": part_source, "dtype": spec.get("dtype", "auto"),
                                              "device": spec.get("device", "auto")}, _top=False)
                parts[role] = part
            engine = CompositeEngine(name, kind, parts)
        else:
            raise ValueError(f"unknown engine kind: {kind}")
        engine.load_seconds = round(time.time() - t0, 1)
        self.engines[name] = engine
        if _top and self.default_name is None:
            self.default_name = name
        return engine

    def unload(self, name):
        engine = self.engines.pop(name, None)
        if engine is None:
            raise KeyError(f"model not loaded: {name}")
        if self.default_name == name:
            self.default_name = next(iter(self.engines), None)
        self._cache.clear()
        if hasattr(engine, "unload"):
            engine.unload()
        del engine

    def set_default(self, name):
        if name not in self.engines:
            raise KeyError(f"model not loaded: {name}")
        self.default_name = name
        self._cache.clear()

    def get_default(self):
        if self.default_name is None:
            raise RuntimeError("no model loaded; POST /v1/models/load first")
        return self.engines[self.default_name]

    def resolve_lora(self, lora_name):
        if not self.loras_dir:
            return None
        base = os.path.basename(lora_name)
        for candidate in (base, base + ".safetensors"):
            path = os.path.join(self.loras_dir, candidate)
            if os.path.isfile(path):
                return path
        return None

    def list_loras(self):
        if not self.loras_dir or not os.path.isdir(self.loras_dir):
            return []
        return sorted(f[:-len(".safetensors")] for f in os.listdir(self.loras_dir)
                      if f.endswith(".safetensors"))

    def models_for_role(self, engine, role):
        """(model, engine_name, family) triples under an engine, filtered by
        lora role. Family is derived from the engine class, not its name.
        Known roles that match nothing are skipped instead of applied
        everywhere (a clip_g lora must not hit a clip_l-only engine)."""
        out = []

        def walk(eng):
            if isinstance(eng, CompositeEngine):
                for p in eng.parts.values():
                    walk(p)
                return
            if not hasattr(eng, "model"):
                return
            if isinstance(eng, ClipVariantEngine):
                family = "clip_l" if eng.variant.endswith("clip_l") else "clip_g"
            elif isinstance(eng, T5Engine):
                family = "t5"
            else:
                family = None
            out.append((eng.model, eng.name, family))

        walk(engine)
        if role in (None, "*"):
            return [(m, n) for m, n, _f in out]
        filtered = [(m, n) for m, n, family in out if family == role]
        if role in ("clip_l", "clip_g", "t5"):
            return filtered
        llm = [(m, n) for m, n, family in out if family is None]
        return filtered or llm

    def model_by_id(self, model_id):
        def find(engines):
            for eng in engines:
                if eng.name == model_id:
                    return eng
                if isinstance(eng, CompositeEngine):
                    found = find(list(eng.parts.values()))
                    if found is not None:
                        return found
            return None
        found = find(list(self.engines.values()))
        if found is None:
            raise KeyError(model_id)
        return found.model

    @staticmethod
    def _has_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return True
        if isinstance(obj, dict):
            return any(EngineRegistry._has_tensor(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(EngineRegistry._has_tensor(v) for v in obj)
        return False

    def _cache_key(self, engine_name, text, kwargs, lora_stack):
        if self._has_tensor(kwargs):
            return None
        try:
            return json.dumps([engine_name, text, kwargs, lora_stack], sort_keys=True)
        except TypeError:
            return None

    def encode(self, text, kwargs, lora_stack):
        engine = self.get_default()
        with self._infer_lock:
            key = self._cache_key(engine.name, text, kwargs, lora_stack)
            if key is not None:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    self.cache_hits += 1
                    return cached
            self.cache_misses += 1
            if lora_stack and engine.lora is not None:
                engine.lora.apply(engine, lora_stack, self)
            if getattr(engine, "handles_lora_internally", False):
                result = engine.encode(text, kwargs, lora_stack,
                                       lora_resolver=self.resolve_lora)
            else:
                result = engine.encode(text, kwargs, lora_stack)
            if result.get("pooled_output") is None and "cond" in result:
                result["pooled_output"] = torch.zeros(
                    (result["cond"].shape[0], result["cond"].shape[-1]))
            result = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in result.items()}
            mark_step(engine.device)
            if key is not None:
                self._cache[key] = result
                while len(self._cache) > EMBED_CACHE_SIZE:
                    self._cache.popitem(last=False)
            return result

    def generate(self, text, kwargs, gen_kwargs, lora_stack):
        engine = self.get_default()
        with self._infer_lock:
            if lora_stack and engine.lora is not None:
                engine.lora.apply(engine, lora_stack, self)
            if getattr(engine, "handles_lora_internally", False):
                out = engine.generate(text, kwargs, gen_kwargs, lora_stack,
                                      lora_resolver=self.resolve_lora)
            else:
                out = engine.generate(text, kwargs, gen_kwargs)
            mark_step(engine.device)
            return out

    def clear_cache(self):
        self._cache.clear()
        stats = {"previous_hits": self.cache_hits, "previous_misses": self.cache_misses}
        self.cache_hits = 0
        self.cache_misses = 0
        return stats

    def status(self):
        loaded = []
        for name, eng in self.engines.items():
            entry = {"name": name, "kind": getattr(eng, "kind", type(eng).__name__),
                     "load_seconds": getattr(eng, "load_seconds", None)}
            if isinstance(eng, CompositeEngine):
                entry["components"] = {k: v.name for k, v in eng.parts.items()}
            loaded.append(entry)
        engine = self.engines.get(self.default_name)
        return {
            "proto": PROTOCOL_VERSION,
            "device": str(engine.device) if engine is not None else None,
            "dtype": str(engine.dtype) if engine is not None else None,
            "models": loaded,
            "default_model": self.default_name,
            "cache": {"size": len(self._cache), "hits": self.cache_hits,
                      "misses": self.cache_misses},
            "loras": self.list_loras(),
            "gpu": gpu_summary(),
        }


def gpu_summary():
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {"type": "cuda", "name": props.name,
                    "vram_gb": round(props.total_memory / 1e9, 1)}
        if xm is not None:
            return {"type": "xla"}
        return {"type": "cpu"}
    except Exception:
        return {"type": "unknown"}
