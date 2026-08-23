import math
import torch
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

    # Same presets (and same order) as ComfyUI core's built-in ResolutionSelector node.
    aspect_ratios = [
        "Auto (Input Image)",
        "1:1 (Square)",
        "2:3 (Portrait Photo)",
        "3:2 (Photo)",
        "3:4 (Portrait Standard)",
        "4:3 (Standard)",
        "9:16 (Portrait Widescreen)",
        "16:9 (Widescreen)",
        "21:9 (Ultrawide)",
    ]

    aspect_ratio_values = {
        "1:1 (Square)": (1.0, 1.0),
        "2:3 (Portrait Photo)": (2.0, 3.0),
        "3:2 (Photo)": (3.0, 2.0),
        "3:4 (Portrait Standard)": (3.0, 4.0),
        "4:3 (Standard)": (4.0, 3.0),
        "9:16 (Portrait Widescreen)": (9.0, 16.0),
        "16:9 (Widescreen)": (16.0, 9.0),
        "21:9 (Ultrawide)": (21.0, 9.0),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scale_mode": (s.scale_modes,),
                "aspect_ratio": (s.aspect_ratios,),
                "long_side_target": ("FLOAT", {"default": 1024.0, "min": 1.0, "max": 16384.0}),
                "short_side_target": ("FLOAT", {"default": 768.0, "min": 1.0, "max": 16384.0}),
                "width": ("FLOAT", {"default": 1024.0, "min": 1.0, "max": 16384.0}),
                "height": ("FLOAT", {"default": 1024.0, "min": 1.0, "max": 16384.0}),
                "multiplier": ("FLOAT", {"default": 1.0}),
                "megapixels": ("FLOAT", {"default": 1.0}),
                "upscale_method": (s.upscale_methods,),
                "crop": (s.crop_methods,),
                "divisible_by": ("FLOAT", {"default": 32.0, "min": 1.0, "max": 512.0}),
                "maintain_aspect": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")

    FUNCTION = "resize"
    CATEGORY = "PlagueKind/image"

    def safe_dim(self, v):
        return max(1, int(round(v)))

    def resolve_size(self, mode, w, h, kw, aspect_override=None):
        w = max(1.0, float(w))
        h = max(1.0, float(h))

        # Aspect-driven modes (everything but explicit W x H) can have their
        # source aspect ratio overridden by the aspect_ratio preset dropdown
        # instead of being derived from the connected image (or the 1x1
        # fallback when no image is connected).
        if aspect_override is not None:
            w, h = aspect_override

        if mode == "Dimensions (W × H)":
            return self.safe_dim(kw["width"]), self.safe_dim(kw["height"])

        if mode == "Multiplier":
            return self.safe_dim(w * kw["multiplier"]), self.safe_dim(h * kw["multiplier"])

        if mode == "Longer Side":
            target = max(1.0, float(kw["long_side_target"]))
            scale = target / max(w, h)
            return self.safe_dim(w * scale), self.safe_dim(h * scale)

        if mode == "Shorter Side":
            target = max(1.0, float(kw["short_side_target"]))
            scale = target / min(w, h)
            return self.safe_dim(w * scale), self.safe_dim(h * scale)

        if mode == "Total Pixels (MP)":
            aspect = w / h if h else 1.0
            area = max(0.000001, float(kw["megapixels"])) * 1_000_000.0
            nw = max(1, int(round(math.sqrt(area * aspect))))
            nh = max(1, int(round(area / max(1, nw))))
            return nw, nh

        return self.safe_dim(w), self.safe_dim(h)

    def snap(self, v, div):
        v = max(1, int(v))
        div = max(1, int(round(div)))
        if div <= 1:
            return v
        return max(div, (v // div) * div)

    def apply_divisible(self, w, h, div, maintain_aspect):
        w = max(1, int(w))
        h = max(1, int(h))
        div = max(1, int(round(div)))

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
        image=None,
        mask=None,
        scale_mode=None,
        aspect_ratio="Auto (Input Image)",
        upscale_method="bilinear",
        crop="center",
        divisible_by=32.0,
        width=1024.0,
        height=1024.0,
        multiplier=1.0,
        megapixels=1.0,
        long_side_target=1024.0,
        short_side_target=768.0,
        maintain_aspect=True
    ):
        if image is not None:
            orig_h = image.shape[1]
            orig_w = image.shape[2]
        else:
            # No image connected: act as a plain resolution selector
            # (t2i / t2v use case). Aspect-dependent modes (Multiplier,
            # Longer/Shorter Side, MP) fall back to a square 1x1 source;
            # use "Dimensions (W x H)" mode for exact width/height control,
            # or pick a preset from aspect_ratio to override the 1x1 fallback.
            orig_h = 1
            orig_w = 1

        kw = {
            "width": width,
            "height": height,
            "multiplier": multiplier,
            "megapixels": megapixels,
            "long_side_target": long_side_target,
            "short_side_target": short_side_target,
        }

        # aspect_ratio is only meaningful for modes that derive width/height
        # from a source aspect ratio; "Dimensions (W x H)" sets both values
        # explicitly, so the preset is ignored there.
        aspect_override = None
        if scale_mode != "Dimensions (W × H)" and aspect_ratio in self.aspect_ratio_values:
            aspect_override = self.aspect_ratio_values[aspect_ratio]

        w, h = self.resolve_size(scale_mode, orig_w, orig_h, kw, aspect_override)
        w, h = self.apply_divisible(w, h, divisible_by, maintain_aspect)
        w, h = max(1, int(w)), max(1, int(h))

        img = torch.zeros((1, 1, 1, 3))
        if image is not None:
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
    "UnifiedResizeImageMask": "Unified Resize Image / Mask"
}
