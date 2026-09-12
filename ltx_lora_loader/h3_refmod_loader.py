"""
H3_refmod_loader — LoRA-loader-style stack UI for ComfyUI-MiniMaxH3Mod's
"Load H3 RefMods" node.

This node does NOT reimplement or fork any of ComfyUI-MiniMaxH3Mod's logic.
At call time it looks up the real, unmodified `MiniMaxH3RefModsLoader` class
through ComfyUI's own global node registry (the same dict ComfyUI populates
when it loads every custom_nodes package) and calls its public `.load()`
method with keyword arguments built to match its own INPUT_TYPES() schema
(`mod_1..mod_8`, `strength_1..8`, `copies_1..8`, `show_info`,
`max_total_tokens`). All mod discovery, safetensors loading, caching, token
budgeting, and the H3_REF_MODS bundle format stay exactly as
ComfyUI-MiniMaxH3Mod defines them.

If ComfyUI-MiniMaxH3Mod isn't installed, this node degrades to an empty
bundle with a console warning instead of crashing the graph.
"""

import json

import nodes as comfy_nodes  # ComfyUI's own module -> global NODE_CLASS_MAPPINGS

from aiohttp import web
from server import PromptServer

_BACKING_NODE_KEY = "MiniMaxH3RefModsLoader"
_FALLBACK_NONE = "(none)"
_FALLBACK_MAX_SLOTS = 8


def _get_backing_cls():
    """The real, unmodified MiniMaxH3RefModsLoader class, if the pack is installed."""
    return comfy_nodes.NODE_CLASS_MAPPINGS.get(_BACKING_NODE_KEY)


def _none_sentinel(cls) -> str:
    return getattr(cls, "NONE", _FALLBACK_NONE) if cls else _FALLBACK_NONE


def _max_slots(cls) -> int:
    return int(getattr(cls, "MAX_SLOTS", _FALLBACK_MAX_SLOTS)) if cls else _FALLBACK_MAX_SLOTS


def _list_refmod_names_with_none():
    """Reads the mod list straight from their own INPUT_TYPES() schema - no
    reimplementation of their folder-scanning/metadata-filtering logic."""
    cls = _get_backing_cls()
    if cls is None:
        return [_FALLBACK_NONE]
    try:
        return list(cls.INPUT_TYPES()["required"]["mod_1"][0])
    except Exception as e:
        print(f"[PlagueKind | H3_refmod_loader] couldn't read mod list from {_BACKING_NODE_KEY}: {e}")
        return [_FALLBACK_NONE]


def _rows_to_kwargs(rows, cls, show_info, max_total_tokens):
    """Translate our [{on, mod, str, copies}, ...] rows into their mod_i/
    strength_i/copies_i keyword schema, padding unused slots with (none)."""
    none_name = _none_sentinel(cls)
    max_slots = _max_slots(cls)
    kwargs = {"show_info": bool(show_info), "max_total_tokens": int(max_total_tokens)}

    slot = 0
    overflow = 0
    for row in rows:
        if not isinstance(row, dict) or not row.get("on"):
            continue
        mod_name = row.get("mod", none_name)
        if mod_name in (none_name, "", None):
            continue
        strength = float(row.get("str", 1.0))
        if strength <= 0.0:
            continue
        if slot >= max_slots:
            overflow += 1
            continue
        slot += 1
        kwargs[f"mod_{slot}"] = mod_name
        kwargs[f"strength_{slot}"] = strength
        kwargs[f"copies_{slot}"] = max(1, min(10, int(row.get("copies", 1))))

    if overflow:
        print(f"[PlagueKind | H3_refmod_loader] {overflow} extra enabled row(s) ignored - "
              f"{_BACKING_NODE_KEY} only supports {max_slots} slots.")

    for i in range(slot + 1, max_slots + 1):
        kwargs[f"mod_{i}"] = none_name
        kwargs[f"strength_{i}"] = 1.0
        kwargs[f"copies_{i}"] = 1

    return kwargs


@PromptServer.instance.routes.get("/plaguekind/h3_refmod_loader/refresh")
async def pk_h3_refmod_refresh(request):
    return web.json_response({"mods": _list_refmod_names_with_none()})


class H3_refmod_loader:
    @classmethod
    def INPUT_TYPES(cls):
        mod_list = _list_refmod_names_with_none()
        return {
            "required": {
                "show_info": ("BOOLEAN", {"default": False,
                    "tooltip": "Print full details (tokens, layout, source, pool) of every loaded mod to the console."}),
                "stack_data": ("STRING", {"default": "[]", "multiline": False}),
            },
            "optional": {
                "max_total_tokens": ("INT", {"default": 0, "min": 0, "max": 1048576,
                    "tooltip": "0 disables the budget. Positive values reject bundles exceeding this token count after copies."}),
            },
            "hidden": {
                "available_mods": (mod_list,),
            },
        }

    RETURN_TYPES = ("H3_REF_MODS", "STRING")
    RETURN_NAMES = ("mods", "prompt_hint")
    FUNCTION = "apply_stack"
    CATEGORY = "PlagueKind/loaders"

    @classmethod
    def IS_CHANGED(cls, show_info, stack_data, max_total_tokens=0, available_mods=None):
        backing_cls = _get_backing_cls()
        if backing_cls is None:
            return float("nan")
        try:
            rows = json.loads(stack_data)
        except Exception:
            rows = []
        kwargs = _rows_to_kwargs(rows, backing_cls, show_info, max_total_tokens)
        try:
            return backing_cls.IS_CHANGED(**kwargs)
        except Exception:
            return float("nan")

    def apply_stack(self, show_info, stack_data, max_total_tokens=0, available_mods=None):
        backing_cls = _get_backing_cls()
        if backing_cls is None:
            print(f"[PlagueKind | H3_refmod_loader] {_BACKING_NODE_KEY} not found - "
                  f"install ComfyUI-MiniMaxH3Mod to use this node. Returning an empty bundle.")
            return ([], "")

        try:
            rows = json.loads(stack_data)
        except Exception:
            print("[PlagueKind | H3_refmod_loader] Failed to parse stack_data JSON - returning empty bundle.")
            rows = []

        kwargs = _rows_to_kwargs(rows, backing_cls, show_info, max_total_tokens)

        try:
            mods, hint = backing_cls().load(**kwargs)
        except Exception as e:
            print(f"[PlagueKind | H3_refmod_loader] {_BACKING_NODE_KEY}.load() failed: {e}")
            return ([], "")

        return (mods, hint)


NODE_CLASS_MAPPINGS = {
    "H3_refmod_loader": H3_refmod_loader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3_refmod_loader": "RefMod Loader Stack ( MiniMax H3 RefMods )",
}
