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
                "long_side_target": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "short_side_target": ("INT", {"default": 768, "min": 1, "max": 16384}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
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
    CATEGORY = "PlagueKind/image"

    def safe_dim(self, v):
        return max(1, int(round(v)))

    def resolve_size(self, mode, w, h, kw):
        w = max(1, int(w))
        h = max(1, int(h))

        if mode == "Dimensions (W × H)":
            return self.safe_dim(kw["width"]), self.safe_dim(kw["height"])

        if mode == "Multiplier":
            return self.safe_dim(w * kw["multiplier"]), self.safe_dim(h * kw["multiplier"])

        if mode == "Longer Side":
            target = max(1, int(kw["long_side_target"]))
            scale = target / max(w, h)
            return self.safe_dim(w * scale), self.safe_dim(h * scale)

        if mode == "Shorter Side":
            target = max(1, int(kw["short_side_target"]))
            scale = target / min(w, h)
            return self.safe_dim(w * scale), self.safe_dim(h * scale)

        if mode == "Total Pixels (MP)":
            aspect = w / h if h else 1.0
            area = max(0.000001, float(kw["megapixels"])) * 1_000_000.0
            nw = max(1, int(round(math.sqrt(area * aspect))))
            nh = max(1, int(round(area / max(1, nw))))
            return nw, nh

        return w, h

    def snap(self, v, div):
        v = max(1, int(v))
        if div <= 1:
            return v
        return max(div, (v // div) * div)

    def apply_divisible(self, w, h, div, maintain_aspect):
        w = max(1, int(w))
        h = max(1, int(h))

        if div <= 1:
            return w, h

        if maintain_aspect:
            aspect = w / h if h else 1.0

            if w >= h:
                w = self.snap(w, div)
                h = self.snap(max(1, int(round(w / aspect))), div)
            else:
                h = self.snap(h, div)
                w = self.snap(max(1, int(round(h * aspect))), div)

            return max(1, w), max(1, h)

        return max(1, self.snap(w, div)), max(1, self.snap(h, div))

    def resize_image(self, x, target_w, target_h, method, crop_mode):
        x = x.movedim(-1, 1)

        if crop_mode == "center":
            ow = x.shape[3]
            oh = x.shape[2]

            scale = max(target_w / ow, target_h / oh)

            sw = max(1, int(round(ow * scale)))
            sh = max(1, int(round(oh * scale)))

            x = comfy.utils.common_upscale(
                x,
                sw,
                sh,
                method,
                False
            )

            _, _, ch, cw = x.shape

            top = max(0, (ch - target_h) // 2)
            left = max(0, (cw - target_w) // 2)

            x = x[:, :, top:top + target_h, left:left + target_w]

        else:
            x = comfy.utils.common_upscale(
                x,
                target_w,
                target_h,
                method,
                False
            )

        return x.movedim(1, -1)

    def resize_mask(self, mask, target_w, target_h, crop_mode):
        m = mask.unsqueeze(1).float()

        if crop_mode == "center":
            oh = m.shape[2]
            ow = m.shape[3]

            scale = max(target_w / ow, target_h / oh)

            sw = max(1, int(round(ow * scale)))
            sh = max(1, int(round(oh * scale)))

            m = F.interpolate(
                m,
                size=(sh, sw),
                mode="bilinear",
                align_corners=False
            )

            top = max(0, (sh - target_h) // 2)
            left = max(0, (sw - target_w) // 2)

            m = m[:, :, top:top + target_h, left:left + target_w]

        else:
            m = F.interpolate(
                m,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            )

        return m.squeeze(1)

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
        w, h = max(1, int(w)), max(1, int(h))

        img = self.resize_image(
            image,
            w,
            h,
            upscale_method,
            crop
        )

        if mask is not None:
            mask = self.resize_mask(
                mask,
                w,
                h,
                crop
            )

        return {
            "ui": {"text": [f"Resized to: {w} × {h}"]},
            "result": (img, mask, w, h)
        }


NODE_CLASS_MAPPINGS = {
    "UnifiedResizeImageMask": UnifiedResizeImageMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedResizeImageMask": "Unified Resize Image / Mask (Clean)"
}
