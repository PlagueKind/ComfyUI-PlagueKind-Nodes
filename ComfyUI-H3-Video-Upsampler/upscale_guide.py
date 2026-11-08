# Generated from ComfyUI-Hand-Tie-Clips (https://github.com/sandpies/ComfyUI-Hand-Tie-Clips) by tools/build_upsampler_pack.py.
# Edit it there, not here: the next build overwrites this file.
"""The h3upscale guide, for the upsampler's re-draw.

Alissonerdx's h3upscale LoRA (trained with his ai-toolkit fork, branch
`minimax-h3-latent-guides`) learned an IC-LoRA-style guide: the source,
reduced by `reference_downscale_factor`, packed as clean condition rows that
sit ON the target's clock and on every factor-th origin of the target's patch
grid (`target_frame_grid[::f, ::f]`, `target_grid_stride_v1`). Without that
guide the LoRA is a texture nudge on a light re-draw; with it, the re-draw
restores detail toward the film's own content instead of inventing it.

Core's keyframe path already matches the trainer on everything except that
spatial grid: time placement, condition timestep, noise augmentation (0.999),
row order and patchify of the cond latent at its own size. Core assumes a
keyframe covers the FULL target grid, so `install()` teaches PackedLayout one
thing: a keyframe carrying `htc_grid_stride` > 1 has `vt * (Hp/f) * (Wp/f)`
rows positioned at `frame[::f, ::f]`. Keyframes without that key are
untouched, so the patch is inert for every other graph, and it is installed
only when a guided re-draw actually runs.

HTC Lab's prototype node patches the same way under the same marker, so
whichever loads first installs it and the other leaves it alone. The layout
was checked row for row against the fork's `build_packed_sequence` (37/37,
2026-10-04); tools/check_upscale_guide.py repeats that when the fork's
packing.py is on disk.
"""
from __future__ import annotations

import json

import torch

TAG = "H3VideoUpsampler guide"
STRIDE_KEY = "htc_grid_stride"
SUPPORTED_SPATIAL = ("target_crop_v2",)
SUPPORTED_POSITION = ("target_grid_stride_v1",)
# The caption every pair in the dataset carries (manifest.json `caption`). The
# LoRA learned the guide under this text, so the re-draw is conditioned on it
# rather than on the hop's prompt. Not in the LoRA header, hence a constant.
CAPTION = ("h3upscale, increase resolution and restore fine details while "
           "preserving the original content and visual style.")


# --- the LoRA header ---------------------------------------------------------

def factor_from_meta(meta):
    """The guide factor a LoRA header declares; 0 when it is not a guide LoRA.

    Raises on guide geometry this code does not implement, rather than packing
    rows the LoRA was never trained on."""
    meta = meta or {}
    try:
        factor = int(meta.get("reference_downscale_factor", 0) or 0)
    except (TypeError, ValueError):
        factor = 0
    if factor <= 1:
        return 0
    spatial = meta.get("minimax_h3_guide_spatial_version")
    position = meta.get("minimax_h3_guide_position_version")
    if spatial not in SUPPORTED_SPATIAL or position not in SUPPORTED_POSITION:
        raise ValueError(f"{TAG}: unsupported guide geometry spatial={spatial!r} "
                         f"position={position!r}; this implements "
                         f"{SUPPORTED_SPATIAL[0]} / {SUPPORTED_POSITION[0]}")
    if str(meta.get("guide_latent_only", "true")).lower() != "true":
        raise ValueError(f"{TAG}: the LoRA was trained with the guide shown to "
                         "Qwen; not supported")
    rope = json.loads(meta.get("minimax_h3_reference_rope", "{}") or "{}")
    if rope.get("guide_rope_layout", "overlap") != "overlap":
        raise ValueError(f"{TAG}: guide_rope_layout "
                         f"{rope['guide_rope_layout']!r} not supported")
    return factor


def lora_factor(path):
    """`factor_from_meta` for a LoRA file, reading only its header."""
    import comfy.utils
    _, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    return factor_from_meta(meta)


# --- pixels ------------------------------------------------------------------

def guide_pixels(frames, w_px, h_px, factor, batch=16):
    """[N, H, W, 3] canvas frames -> [N, h_px/f, w_px/f, 3], 0..1.

    The trainer fits the source to the output canvas, then reduces it by the
    factor. The upscaler STRETCHES the canvas to the output size (it does not
    crop), so the guide stretches too -- a crop would shift it off the content
    it is meant to sit on. Bicubic, antialiased, on the GPU when there is one.
    """
    import comfy.model_management as mm
    dev = mm.get_torch_device()
    gw, gh = int(w_px) // int(factor), int(h_px) // int(factor)
    out = torch.empty((int(frames.shape[0]), gh, gw, 3), dtype=torch.float32)
    for s in range(0, int(frames.shape[0]), batch):
        x = frames[s:s + batch].to(dev, torch.float32).permute(0, 3, 1, 2)
        x = torch.nn.functional.interpolate(x, size=(int(h_px), int(w_px)),
                                            mode="bicubic", align_corners=False,
                                            antialias=True).clamp_(0.0, 1.0)
        x = torch.nn.functional.interpolate(x, size=(gh, gw), mode="bicubic",
                                            align_corners=False,
                                            antialias=True).clamp_(0.0, 1.0)
        out[s:s + batch] = x.permute(0, 2, 3, 1).cpu()
    return out


# --- the layout patch --------------------------------------------------------

class _ShapeOnly:
    """Stands in for a strided keyframe latent while core builds the layout.

    Core reads only `latent.shape[2]` there; the real latent rides the payload's
    `cond_video_latents`, which model_base takes from the real keyframe list."""

    def __init__(self, vt):
        self.shape = (1, 0, vt, 0, 0)


def _strided_rows(layout_init):
    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None):
        strides = [int(kf.get(STRIDE_KEY, 1)) if kf.get("latent") is not None else 0
                   for kf in (keyframes or ())]
        if not any(s > 1 for s in strides):
            return layout_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                               keyframes=keyframes, refs=refs)
        stand_in = [dict(kf, latent=_ShapeOnly(kf["latent"].shape[2])) if s > 1 else kf
                    for kf, s in zip(keyframes, strides)]
        layout_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                    keyframes=stand_in, refs=refs)
        _shrink(self, latent_h, latent_w, keyframes, strides)
    __init__._htc_strided = True
    return __init__


def _shrink(layout, latent_h, latent_w, keyframes, strides):
    """Cut each strided keyframe's full-grid segment down to its coarse grid."""
    hp, wp = latent_h // 2, latent_w // 2
    cond = [i for i, (_, _, kind) in enumerate(layout.segments) if kind == "cond"]
    video_kfs = [(kf, s) for kf, s in zip(keyframes, strides) if s > 0]
    if len(cond) != len(video_kfs):
        raise RuntimeError(f"{TAG}: {len(cond)} cond segments for {len(video_kfs)} video keyframes")
    keep = torch.ones(layout.seq_len, dtype=torch.bool)
    for seg_i, (kf, f) in zip(cond, video_kfs):
        if f <= 1:
            continue
        start, stop, _ = layout.segments[seg_i]
        vt = kf["latent"].shape[2]
        gh, gw = kf["latent"].shape[3] // 2, kf["latent"].shape[4] // 2
        if hp % f or wp % f or (gh, gw) != (hp // f, wp // f):
            raise ValueError(
                f"{TAG}: guide latent grid {gh}x{gw} is not the target patch grid {hp}x{wp} / {f}. "
                f"The output canvas must be a multiple of {32 * f} px on both axes.")
        m = torch.zeros(vt, hp, wp, dtype=torch.bool)
        m[:, ::f, ::f] = True
        keep[start:stop] = m.reshape(-1)
    # every row index past a dropped row moves down by the rows dropped before it
    new_index = torch.cumsum(keep.to(torch.long), 0) - 1
    layout.position_ids = layout.position_ids[keep]
    layout.img_update = layout.img_update[keep[layout.img_pos]]
    layout.img_pos = new_index[layout.img_pos[keep[layout.img_pos]]]
    layout.audio_pos = new_index[layout.audio_pos]
    segs, off = [], 0
    for start, stop, kind in layout.segments:
        n = int(keep[start:stop].sum())
        segs.append((off, off + n, kind))
        off += n
    layout.segments = segs
    layout.seq_len = off


def install():
    """Teach core's PackedLayout strided keyframes. Idempotent."""
    import comfy.ldm.minimax.model as mm
    if not getattr(mm.PackedLayout.__init__, "_htc_strided", False):
        mm.PackedLayout.__init__ = _strided_rows(mm.PackedLayout.__init__)
