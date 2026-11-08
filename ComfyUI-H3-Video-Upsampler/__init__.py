# Generated from ComfyUI-Hand-Tie-Clips (https://github.com/sandpies/ComfyUI-Hand-Tie-Clips) by tools/build_upsampler_pack.py.
# Edit it there, not here: the next build overwrites this file.
"""H3 Video Upsampler: a finished MiniMax H3 clip, upscaled with the H3 latent
upscaler and one light re-draw step in tiles (the original MMH3 recipe), plus a
long-form save node."""
from .tiled_upsampler import register_folder as _register_folder
from .video_upsampler import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    # PyAV is ComfyUI's own dependency; a missing or broken one costs only this node.
    from .save_video import NODE_CLASS_MAPPINGS as _SAVE, NODE_DISPLAY_NAME_MAPPINGS as _SAVE_N
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_SAVE}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_SAVE_N}
except Exception as _e:  # noqa: BLE001
    print(f"[H3VideoUpsampler] H3 Save Video not loaded: {_e!r}", flush=True)

_register_folder()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
