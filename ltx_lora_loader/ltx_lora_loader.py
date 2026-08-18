import os
import json

import folder_paths
import comfy.utils
import comfy.lora

try:
    from comfy.lora import load_lora_for_models as _load_lora
except (ImportError, AttributeError):
    from comfy.sd import load_lora_for_models as _load_lora

from aiohttp import web
from server import PromptServer


def _is_audio_key(key: str) -> bool:
    key = key.lower()
    return key.startswith("diffusion_model.transformer_blocks.") and "audio" in key


def _is_video_key(key: str) -> bool:
    key = key.lower()
    return key.startswith("diffusion_model.transformer_blocks.") and "audio" not in key


# MiniMax H3's transformer (comfy.ldm.minimax.model.MiniMaxH3Model) is a single
# joint-attention stack: self.blocks[i] mixes packed text/audio/video tokens
# together in one pass, so a LoRA delta on blocks.*.attn/mlp/adaln_proj cannot
# be attributed to one modality and cannot be scaled independently per modality.
# Only these are actually modality-specific:
#   video: video_patch_proj, final_layer.video_out
#   audio: audio_patch_proj, final_layer.audio_out
#   text:  condition_proj, token_refiner.*
# Everything else (blocks.*, rope, time_embedder) is "joint" and only scales
# with the master strength - it cannot be split into v/a/t.
def _is_minimax_h3_key(key: str) -> bool:
    key = key.lower()
    return any(tag in key for tag in (
        "video_patch_proj", "audio_patch_proj", "condition_proj",
        "token_refiner", "final_layer.video_out", "final_layer.audio_out",
    ))


def _minimax_h3_modality(key: str) -> str:
    key = key.lower()
    if "video_patch_proj" in key or "final_layer.video_out" in key:
        return "video"
    if "audio_patch_proj" in key or "final_layer.audio_out" in key:
        return "audio"
    if "condition_proj" in key or "token_refiner" in key:
        return "text"
    return "joint"


def _apply_normal(model, clip, lora_name: str, weights: dict, lora_str: float):
    print(f"[PlagueKind | LTX_lora_loader] '{lora_name}' (normal) {len(weights)} keys @ {lora_str:.3f}")
    if weights and lora_str != 0.0:
        model, clip = _load_lora(model, clip, weights, lora_str, lora_str)
    return model, clip


def _apply_ltx_split(model, clip, lora_name: str, weights: dict, lora_str: float, v_mult: float, a_mult: float):
    video_weights = {k: v for k, v in weights.items() if _is_video_key(k)}
    audio_weights = {k: v for k, v in weights.items() if _is_audio_key(k)}

    # keys don't match the LTX transformer_blocks prefix - fall back to treating
    # the whole dict as a single (video-slot) weight set.
    if not video_weights and not audio_weights:
        video_weights = weights

    v_strength = lora_str * v_mult
    a_strength = lora_str * a_mult

    print(
        f"[PlagueKind | LTX_lora_loader] '{lora_name}' (LTX) "
        f"V:{len(video_weights)} keys @ {v_strength:.3f}  "
        f"A:{len(audio_weights)} keys @ {a_strength:.3f}"
    )

    if video_weights and v_strength != 0.0:
        model, clip = _load_lora(model, clip, video_weights, v_strength, v_strength)
    if audio_weights and a_strength != 0.0:
        model, clip = _load_lora(model, clip, audio_weights, a_strength, a_strength)

    return model, clip


def _apply_minimax_split(model, clip, lora_name: str, weights: dict, lora_str: float, v_mult: float, a_mult: float, t_mult: float):
    # keys that don't match any MiniMax H3 I/O prefix land in "joint" by default,
    # so a non-MiniMax lora forced into this mode still applies at full strength.
    buckets = {"video": {}, "audio": {}, "text": {}, "joint": {}}
    for k, v in weights.items():
        buckets[_minimax_h3_modality(k)][k] = v
    mults = {"video": v_mult, "audio": a_mult, "text": t_mult, "joint": 1.0}

    print(
        f"[PlagueKind | LTX_lora_loader] '{lora_name}' (MiniMax H3) "
        f"J:{len(buckets['joint'])}@{lora_str:.3f}  "
        f"V:{len(buckets['video'])}@{lora_str * v_mult:.3f}  "
        f"A:{len(buckets['audio'])}@{lora_str * a_mult:.3f}  "
        f"T:{len(buckets['text'])}@{lora_str * t_mult:.3f}"
    )

    for name, bucket in buckets.items():
        strength = lora_str * mults[name]
        if bucket and strength != 0.0:
            model, clip = _load_lora(model, clip, bucket, strength, strength)

    return model, clip


def _apply_slot(model, clip, lora_name: str, lora_str: float, v_mult: float, a_mult: float,
                 t_mult: float = 1.0, mode: str = "auto"):
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.isfile(lora_path):
        print(f"[PlagueKind | LTX_lora_loader] LoRA not found: {lora_name}")
        return model, clip

    weights = comfy.utils.load_torch_file(lora_path, safe_load=True)

    if mode == "normal":
        return _apply_normal(model, clip, lora_name, weights, lora_str)
    if mode == "ltx":
        return _apply_ltx_split(model, clip, lora_name, weights, lora_str, v_mult, a_mult)
    if mode == "minimax":
        return _apply_minimax_split(model, clip, lora_name, weights, lora_str, v_mult, a_mult, t_mult)

    # "auto" (legacy stack_data saved before the mode toggle existed) - sniff the
    # keys to pick a scheme, same behavior this node had before.
    if any(_is_minimax_h3_key(k) for k in weights):
        return _apply_minimax_split(model, clip, lora_name, weights, lora_str, v_mult, a_mult, t_mult)
    return _apply_ltx_split(model, clip, lora_name, weights, lora_str, v_mult, a_mult)


@PromptServer.instance.routes.get("/plaguekind/ltx_lora_loader/keycounts")
async def pk_ltx_keycounts(request):
    lora_name = request.rel_url.query.get("lora", "")
    if not lora_name:
        return web.json_response({"scheme": "none", "v": 0, "a": 0, "t": 0, "j": 0})

    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.isfile(lora_path):
        return web.json_response({"scheme": "none", "v": 0, "a": 0, "t": 0, "j": 0})

    try:
        import safetensors
        with safetensors.safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        try:
            weights = comfy.utils.load_torch_file(lora_path, safe_load=True)
            keys = list(weights.keys())
        except Exception:
            return web.json_response({"scheme": "error", "v": -1, "a": -1, "t": -1, "j": -1})

    if any(_is_minimax_h3_key(k) for k in keys):
        counts = {"video": 0, "audio": 0, "text": 0, "joint": 0}
        for k in keys:
            counts[_minimax_h3_modality(k)] += 1
        return web.json_response({
            "scheme": "minimax_h3",
            "v": counts["video"], "a": counts["audio"], "t": counts["text"], "j": counts["joint"],
        })

    v_count = sum(1 for k in keys if _is_video_key(k))
    a_count = sum(1 for k in keys if _is_audio_key(k))
    scheme = "ltx" if (v_count or a_count) else "generic"
    return web.json_response({"scheme": scheme, "v": v_count, "a": a_count, "t": 0, "j": 0})


@PromptServer.instance.routes.get("/plaguekind/ltx_lora_loader/refresh")
async def pk_ltx_refresh(request):
    return web.json_response({"loras": folder_paths.get_filename_list("loras")})


class LTX_lora_loader:
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["normal", "ltx", "minimax"], {"default": "normal"}),
                "stack_data": ("STRING", {"default": "[]", "multiline": False}),
            },
            "optional": {
                "clip": ("CLIP",),
            },
            "hidden": {
                "available_loras": (lora_list,),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply_stack"
    CATEGORY = "PlagueKind/loaders"

    def apply_stack(self, model, mode, stack_data, clip=None, available_loras=None):
        m = model
        c = clip

        try:
            data = json.loads(stack_data)
        except Exception:
            print("[PlagueKind | LTX_lora_loader] Failed to parse stack_data JSON - returning unchanged.")
            return (m, c)

        for row in data:
            if not row.get("on"):
                continue
            lora_name = row.get("lora", "None")
            if lora_name in ("None", "", None):
                continue

            lora_str = float(row.get("str", 1.0))
            v_mult = float(row.get("v", 1.0))
            a_mult = float(row.get("a", 1.0))
            t_mult = float(row.get("t", 1.0))

            m, c = _apply_slot(m, c, lora_name, lora_str, v_mult, a_mult, t_mult, mode)

        return (m, c)


NODE_CLASS_MAPPINGS = {
    "LTX_lora_loader": LTX_lora_loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX_lora_loader": "LoRA Loader Stack ( LTX / MiniMax H3 Compatible )",
}

