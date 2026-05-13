import math
import comfy.utils
import torch.nn.functional as F


class UnifiedResizeImageMask:

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    scale_modes = [
        "Dimensions (W × H)",
        "Multiplier",
        "Longer Side",
        "Shorter Side",
        "Total Pixels (MP)",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_mode": (s.scale_modes,),
                "long_side_target": ("INT", {"default": 1024}),
                "short_side_target": ("INT", {"default": 768}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "multiplier": ("FLOAT", {"default": 1.0}),
                "megapixels": ("FLOAT", {"default": 1.0}),
                "upscale_method": (s.upscale_methods,),
                "crop": (s.crop_methods,),
                "divisible_by": ("INT", {"default": 32, "min": 1, "max": 512}),
                "maintain_aspect": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")

    FUNCTION = "resize"
    CATEGORY = "image"

    # -------------------------
    # SIZE LOGIC
    # -------------------------
    def resolve_size(self, mode, w, h, kw):

        if mode == "Dimensions (W × H)":
            return kw["width"], kw["height"]

        if mode == "Multiplier":
            return int(w * kw["multiplier"]), int(h * kw["multiplier"])

        if mode == "Longer Side":
            target = kw["long_side_target"]
            scale = target / (w if w >= h else h)
            return int(w * scale), int(h * scale)

        if mode == "Shorter Side":
            target = kw["short_side_target"]
            scale = target / (w if w <= h else h)
            return int(w * scale), int(h * scale)

        if mode == "Total Pixels (MP)":
            aspect = w / h
            mp = kw["megapixels"] * 1_000_000
            nw = int(math.sqrt(mp * aspect))
            nh = int(mp / nw)
            return nw, nh

        return w, h

    def snap(self, v, div):
        if div <= 1:
            return v
        return max(div, (v // div) * div)

    def apply_divisible(self, w, h, div, maintain_aspect):

        if div <= 1:
            return w, h

        if maintain_aspect:
            aspect = w / h
            if w >= h:
                w = self.snap(w, div)
                h = int(w / aspect)
            else:
                h = self.snap(h, div)
                w = int(h * aspect)
            return w, h

        return self.snap(w, div), self.snap(h, div)

    # -------------------------
    # IMAGE PIPELINE
    # -------------------------
    def resize_image(self, x, target_w, target_h, method):

        x = x.movedim(-1, 1)

        ow = x.shape[3]
        oh = x.shape[2]

        scale = max(target_w / ow, target_h / oh)

        sw = int(ow * scale)
        sh = int(oh * scale)

        x = comfy.utils.common_upscale(x, sw, sh, method, False)

        _, _, ch, cw = x.shape

        top = (ch - target_h) // 2
        left = (cw - target_w) // 2

        x = x[:, :, top:top + target_h, left:left + target_w]

        return x.movedim(1, -1)

    # -------------------------
    # MASK PIPELINE (FIXED - NO PIL)
    # -------------------------
    def resize_mask(self, mask, target_w, target_h):

        m = mask.unsqueeze(1).float()

        oh = m.shape[2]
        ow = m.shape[3]

        scale = max(target_w / ow, target_h / oh)

        sw = int(ow * scale)
        sh = int(oh * scale)

        m = F.interpolate(m, size=(sh, sw), mode="bilinear", align_corners=False)

        top = (sh - target_h) // 2
        left = (sw - target_w) // 2

        m = m[:, :, top:top + target_h, left:left + target_w]

        return m.squeeze(1)

    # -------------------------
    # MAIN
    # -------------------------
    def resize(
        self,
        image,
        mask=None,
        scale_mode=None,
        upscale_method=None,
        crop="center",
        divisible_by=32,
        width=1024,
        height=1024,
        multiplier=1.0,
        megapixels=1.0,
        long_side_target=1024,
        short_side_target=768,
        maintain_aspect=True
    ):

        orig_h = image.shape[1]
        orig_w = image.shape[2]

        kw = {
            "width": width,
            "height": height,
            "multiplier": multiplier,
            "megapixels": megapixels,
            "long_side_target": long_side_target,
            "short_side_target": short_side_target,
        }

        w, h = self.resolve_size(scale_mode, orig_w, orig_h, kw)
        w, h = self.apply_divisible(w, h, divisible_by, maintain_aspect)

        img = self.resize_image(image, w, h, upscale_method)

        if mask is not None:
            mask = self.resize_mask(mask, w, h)

        return (img, mask, w, h)


NODE_CLASS_MAPPINGS = {
    "UnifiedResizeImageMask": UnifiedResizeImageMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedResizeImageMask": "Unified Resize Image / Mask (Clean)"
}
