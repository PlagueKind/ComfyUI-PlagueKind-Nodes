"""
UPSCALE SIZE - LD
=================

One place that decides how big the upscale is, so the number cannot disagree
with itself.

The MMH3 Ultimate Upscale rig asks for the target size in two separate nodes:
'MMH3 Latent Upscale with Model Params' (what the upscaler renders) and
'MMH3 Spatial Split Params' (what the tiler thinks it is splitting). Nothing
checks that they match. When they drift apart the tiler solves its grid against
the WRONG frame - you do not get an error, you get tiling you can see, because
the tiles were cut for a different picture than the one being cut.

This node is the single source of truth: feed it the generation size the Studio
already knows, pick how much bigger you want it, and wire the ONE width and the
ONE height into every field that asks. They cannot drift after that.

Aspect ratio is held, not assumed. The source ratio is measured from the
width/height you feed in, and both output edges are snapped to `align` (32 for
H3 - the upscaler's grid). Snapping two numbers independently would bend the
ratio, so this searches the snapped candidates around the ideal pair and keeps
the one whose ratio is closest to the source, breaking ties on area.

Three ways to say how big:

    megapixels  absolute target area. 1 MP = 1024x1024 = 1,048,576 px, the
                same convention the H3 upscaler's own 'megapixels' mode uses.
    multiple    linear scale factor. 2.0 = twice the width AND twice the
                height (so four times the area).
    long_edge   pin the longer side to a pixel count, let the other follow.

The report output prints what you actually got, including the real megapixel
count and the linear scale, because a target and a snapped result are not the
same number and you should be able to see both.

Nothing here clamps. If you ask for less than you started with, you get less
than you started with and the report says so - the H3 upscaler itself will
refuse a downscale, and that refusal is more useful than this node quietly
deciding it knew better.
"""

_MP = 1024.0 * 1024.0


def _snap(v, align):
    return max(align, int(round(float(v) / align)) * align)


def _best_pair(ideal_w, ideal_h, aspect, target_area, align):
    """Snapped (w, h) whose ratio sits closest to `aspect`, ties on area."""
    best = None
    base_w = _snap(ideal_w, align)
    base_h = _snap(ideal_h, align)
    for dw in (-1, 0, 1):
        w = base_w + dw * align
        if w < align:
            continue
        for dh in (-1, 0, 1):
            h = base_h + dh * align
            if h < align:
                continue
            ratio_err = abs((w / float(h)) - aspect) / aspect
            area_err = abs((w * h) - target_area) / max(1.0, target_area)
            # ratio is the thing people notice; area is the thing they asked for
            score = ratio_err * 4.0 + area_err
            if best is None or score < best[0]:
                best = (score, w, h)
    return best[1], best[2]


class H3UpscaleSizeLD:
    CATEGORY = "PlagueKind/upscaling"
    FUNCTION = "run"
    RETURN_TYPES = ("INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "scale", "report")
    DESCRIPTION = (
        "Turn the generation size + a target into ONE upscale width/height, "
        "aspect held, snapped to the model grid. Wire the same two outputs "
        "into every node that asks for the upscale size."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 480, "min": 16, "max": 16384, "step": 1,
                    "tooltip": "Source WIDTH - the size the video was "
                               "generated at. Wire this from UnpackLD's "
                               "'width' so it follows the Studio panel and "
                               "you never type it twice."}),
                "height": ("INT", {
                    "default": 608, "min": 16, "max": 16384, "step": 1,
                    "tooltip": "Source HEIGHT - the size the video was "
                               "generated at. Wire this from UnpackLD's "
                               "'height'."}),
                "mode": (["megapixels", "multiple", "long_edge"], {
                    "default": "megapixels",
                    "tooltip": "How the target is expressed.\n"
                               "megapixels = absolute area (1 MP = 1024x1024 "
                               "= 1,048,576 px).\n"
                               "multiple = linear factor; 2.0 doubles BOTH "
                               "edges, which is 4x the area.\n"
                               "long_edge = pin the longer side in pixels."}),
                "megapixels": ("FLOAT", {
                    "default": 2.0, "min": 0.05, "max": 16.0, "step": 0.05,
                    "tooltip": "[megapixels mode] Target area. Note this is "
                               "AREA, not a multiplier: 2 MP at 1:1 is "
                               "1440x1440, not 2048x2048 - 2048x2048 is 4 MP. "
                               "Use 'multiple' if you want to think in "
                               "'twice as wide'."}),
                "multiple": ("FLOAT", {
                    "default": 2.0, "min": 0.25, "max": 8.0, "step": 0.05,
                    "tooltip": "[multiple mode] Linear scale. 2.0 = both "
                               "edges doubled (4x area). 1.5 = 1.5x each "
                               "edge (2.25x area)."}),
                "long_edge": ("INT", {
                    "default": 1440, "min": 64, "max": 16384, "step": 32,
                    "tooltip": "[long_edge mode] Pixel length of the LONGER "
                               "side. The short side follows the source "
                               "ratio."}),
                "align": ("INT", {
                    "default": 32, "min": 1, "max": 256, "step": 1,
                    "tooltip": "Both output edges are snapped to a multiple "
                               "of this. 32 for MiniMax H3 - its latent token "
                               "is 16px and the upscaler works on a 2x2 patch "
                               "grid on top of that. Do not lower it unless "
                               "you know the model tolerates it."}),
            }
        }

    def run(self, width, height, mode, megapixels, multiple, long_edge, align):
        src_w = max(1, int(width))
        src_h = max(1, int(height))
        align = max(1, int(align))
        aspect = src_w / float(src_h)

        if mode == "multiple":
            ideal_w = src_w * float(multiple)
            ideal_h = src_h * float(multiple)
        elif mode == "long_edge":
            if src_w >= src_h:
                ideal_w = float(long_edge)
                ideal_h = ideal_w / aspect
            else:
                ideal_h = float(long_edge)
                ideal_w = ideal_h * aspect
        else:  # megapixels
            target = float(megapixels) * _MP
            ideal_h = (target / aspect) ** 0.5
            ideal_w = ideal_h * aspect

        target_area = ideal_w * ideal_h
        out_w, out_h = _best_pair(ideal_w, ideal_h, aspect, target_area, align)

        out_mp = (out_w * out_h) / _MP
        src_mp = (src_w * src_h) / _MP
        # linear scale, measured on area so a bent ratio cannot hide in it
        scale = ((out_w * out_h) / float(src_w * src_h)) ** 0.5
        src_ratio = aspect
        out_ratio = out_w / float(out_h)
        ratio_drift = abs(out_ratio - src_ratio) / src_ratio * 100.0

        lines = [
            "{}x{}  ->  {}x{}".format(src_w, src_h, out_w, out_h),
            "  {:.2f} MP -> {:.2f} MP   (linear {:.3f}x, area {:.2f}x)".format(
                src_mp, out_mp, scale, scale * scale),
            "  aspect {:.4f} -> {:.4f}  (drift {:.2f}%, both edges /{})".format(
                src_ratio, out_ratio, ratio_drift, align),
        ]
        if mode == "megapixels":
            lines.append("  asked {:.2f} MP, snapping landed on {:.2f} MP"
                         .format(float(megapixels), out_mp))
        elif mode == "multiple":
            lines.append("  asked {:.2f}x linear, snapping landed on {:.3f}x"
                         .format(float(multiple), scale))
        else:
            lines.append("  asked long edge {}, got {}"
                         .format(int(long_edge), max(out_w, out_h)))
        if out_w < src_w or out_h < src_h:
            lines.append("  WARNING: this is a DOWNSCALE. The H3 latent "
                         "upscaler only runs at scale >= 1.0 and will raise.")
        report = "\n".join(lines)
        print("[H3UpscaleSizeLD] "
              + report.replace("\n", "\n[H3UpscaleSizeLD] "))
        return (out_w, out_h, float(scale), report)


NODE_CLASS_MAPPINGS = {"H3UpscaleSizeLD": H3UpscaleSizeLD}
NODE_DISPLAY_NAME_MAPPINGS = {"H3UpscaleSizeLD": "\U0001f4d0 H3 Upscale Size - LD"}
