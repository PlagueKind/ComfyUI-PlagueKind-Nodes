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


def _apply_slot(model, clip, lora_name: str, lora_str: float, v_mult: float, a_mult: float):
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.isfile(lora_path):
        print(f"[PlagueKind | LTX_lora_loader] LoRA not found: {lora_name}")
        return model, clip

    weights = comfy.utils.load_torch_file(lora_path, safe_load=True)

    video_weights = {k: v for k, v in weights.items() if _is_video_key(k)}
    audio_weights = {k: v for k, v in weights.items() if _is_audio_key(k)}

    v_strength = lora_str * v_mult
    a_strength = lora_str * a_mult

    print(
        f"[PlagueKind | LTX_lora_loader] '{lora_name}' "
        f"V:{len(video_weights)} keys @ {v_strength:.3f}  "
        f"A:{len(audio_weights)} keys @ {a_strength:.3f}"
    )

    if video_weights and v_strength != 0.0:
        model, clip = _load_lora(model, clip, video_weights, v_strength, v_strength)

    if audio_weights and a_strength != 0.0:
        model, clip = _load_lora(model, clip, audio_weights, a_strength, a_strength)

    return model, clip


@PromptServer.instance.routes.get("/plaguekind/ltx_lora_loader/keycounts")
async def pk_ltx_keycounts(request):
    lora_name = request.rel_url.query.get("lora", "")
    if not lora_name:
        return web.json_response({"v": 0, "a": 0})

    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.isfile(lora_path):
        return web.json_response({"v": 0, "a": 0})

    try:
        import safetensors
        with safetensors.safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
    except Exception:
        try:
            weights = comfy.utils.load_torch_file(lora_path, safe_load=True)
            keys = list(weights.keys())
        except Exception:
            return web.json_response({"v": -1, "a": -1})

    v_count = sum(1 for k in keys if _is_video_key(k))
    a_count = sum(1 for k in keys if _is_audio_key(k))
    return web.json_response({"v": v_count, "a": a_count})


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

    def apply_stack(self, model, stack_data, clip=None, available_loras=None):
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

            m, c = _apply_slot(m, c, lora_name, lora_str, v_mult, a_mult)

        return (m, c)


NODE_CLASS_MAPPINGS = {
    "LTX_lora_loader": LTX_lora_loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX_lora_loader": "LoRA Loader Stack ( LTX Compatible )",
}
