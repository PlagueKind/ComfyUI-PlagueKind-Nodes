"""H3 Ultimate Upscale (LD fork) - the full latent re-enhancement loop.

A DETACHED FORK of Comfyui-MMH3-UltimateUpscale by bbaudio-2025, kept
because nodes.py carries the joint tiled sampler this rig depends on:
spatial_process_joint() runs one forward per tile per STEP and feather-
accumulates them into a single full-frame prediction, so one latent is
stepped once. The upstream behaviour - each tile solving the whole
trajectory alone and the finished tiles being blended - is what caused
the seams, and no overlap or blend setting can fix that because they all
act downstream of it.

The git remote and .github are gone, and pyproject no longer names the
upstream repository, so neither a pull nor a Manager update can reach
this. The original is parked beside it as *.disabled.

NODE CLASS NAMES ARE DELIBERATELY UNCHANGED - saved workflows key off
them. Only the display names carry the (LD) marker."""
from comfy_api.latest import ComfyExtension
from typing_extensions import override

from .nodes import (
    MMH3UltimateUpscale,
    MMH3LatentUpscaleWithModelParams,
    MMH3LatentUpscaleParams,
    MMH3TemporalSplitParams,
    MMH3SpatialSplitParams,
    LTX25UltimateUpscale,
    LTX25LatentUpscaleParams,
    LTX25TemporalSplitParams,
    LTX25SpatialSplitParams,
    LTX25ReferenceParams,
    LTX25ICLoRALoader,
)

NODE_CLASS_MAPPINGS = {
    "MMH3UltimateUpscale": MMH3UltimateUpscale,
    "MMH3LatentUpscaleWithModelParams": MMH3LatentUpscaleWithModelParams,
    "MMH3LatentUpscaleParams": MMH3LatentUpscaleParams,
    "MMH3TemporalSplitParams": MMH3TemporalSplitParams,
    "MMH3SpatialSplitParams": MMH3SpatialSplitParams,
    "LTX25UltimateUpscale": LTX25UltimateUpscale,
    "LTX25LatentUpscaleParams": LTX25LatentUpscaleParams,
    "LTX25TemporalSplitParams": LTX25TemporalSplitParams,
    "LTX25SpatialSplitParams": LTX25SpatialSplitParams,
    "LTX25ICLoRALoader": LTX25ICLoRALoader,
    "LTX25ReferenceParams": LTX25ReferenceParams,
}

# front-end JS: auto-show/hide tile size vs rows/cols inputs on the two
# Spatial Split Params nodes based on the tile_size_mode combo
WEB_DIRECTORY = "./web"

NODE_DISPLAY_NAME_MAPPINGS = {
    "MMH3UltimateUpscale": "MMH3 Ultimate Upscale (LD)",
    "MMH3LatentUpscaleWithModelParams": "MMH3 Latent Upscale with Model Params (LD)",
    "MMH3LatentUpscaleParams": "MMH3 Latent Upscale Params (LD)",
    "MMH3TemporalSplitParams": "MMH3 Temporal Split Params (LD)",
    "MMH3SpatialSplitParams": "MMH3 Spatial Split Params (LD)",
    "LTX25UltimateUpscale": "LTX25 Ultimate Upscale (LD)",
    "LTX25LatentUpscaleParams": "LTX25 Latent Upscale Params (LD)",
    "LTX25TemporalSplitParams": "LTX25 Temporal Split Params (LD)",
    "LTX25SpatialSplitParams": "LTX25 Spatial Split Params (LD)",
    "LTX25ICLoRALoader": "LTX25 IC-LoRA Loader (MSR) (LD)",
    "LTX25ReferenceParams": "LTX25 Reference Params (LD)",
}


class MMH3UltimateUpscaleExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type]:
        return [
            MMH3UltimateUpscale,
            MMH3LatentUpscaleWithModelParams,
            MMH3LatentUpscaleParams,
            MMH3TemporalSplitParams,
            MMH3SpatialSplitParams,
            LTX25UltimateUpscale,
            LTX25LatentUpscaleParams,
            LTX25TemporalSplitParams,
            LTX25SpatialSplitParams,
            LTX25ReferenceParams,
            LTX25ICLoRALoader,
        ]


async def comfy_entrypoint() -> MMH3UltimateUpscaleExtension:
    return MMH3UltimateUpscaleExtension()
