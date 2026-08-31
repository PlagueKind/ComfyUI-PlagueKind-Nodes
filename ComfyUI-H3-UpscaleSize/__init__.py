"""ComfyUI-H3-UpscaleSize - one width/height for a whole upscale rig.

Standalone copy of H3UpscaleSizeLD from LD-H3-Studio, so a graph can use the
node without installing the whole kit. Same class, same widget order.
"""

from .upscale_size import H3UpscaleSizeLD

NODE_CLASS_MAPPINGS = {"H3UpscaleSizeLD": H3UpscaleSizeLD}
NODE_DISPLAY_NAME_MAPPINGS = {"H3UpscaleSizeLD": "\U0001f4d0 H3 Upscale Size - LD"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
