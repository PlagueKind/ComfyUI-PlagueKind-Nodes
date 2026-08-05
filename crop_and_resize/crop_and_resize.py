import os
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import comfy.utils
from PIL import Image

import folder_paths


ASPECT_PRESETS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
    "3:2": (3, 2),
    "2:3": (2, 3),
}


class VisualCropAndResize:

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    post_crop_methods = ["disabled", "center"]

    aspect_modes = ["Free", "1:1", "4:3", "3:4", "16:9", "9:16", "21:9", "3:2", "2:3", "Custom"]

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
                "aspect_ratio": (s.aspect_modes,),
                "custom_ratio_w": ("INT", {"default": 1, "min": 1, "max": 100}),
                "custom_ratio_h": ("INT", {"default": 1, "min": 1, "max": 100}),
                "crop_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "crop_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "crop_w": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1.0, "step": 0.001}),
                "crop_h": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1.0, "step": 0.001}),
                "show_crop_values": ("BOOLEAN", {"default": False}),

                "scale_mode": (s.scale_modes,),
                "long_side_target": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "short_side_target": ("INT", {"default": 768, "min": 1, "max": 16384}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "multiplier": ("FLOAT", {"default": 1.0}),
                "megapixels": ("FLOAT", {"default": 1.0}),
                "upscale_method": (s.upscale_methods,),
                "post_crop": (s.post_crop_methods,),
                "divisible_by": ("INT", {"default": 32, "min": 1, "max": 512}),
                "maintain_aspect": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height", "x", "y")

    FUNCTION = "run"
    CATEGORY = "PlagueKind/image"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix = "cropresize_" + "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))

    # ---- crop ----

    def get_ratio(self, aspect_ratio, custom_w, custom_h):
        if aspect_ratio == "Free":
            return None
        if aspect_ratio == "Custom":
            return float(custom_w) / float(max(1, custom_h))
        rw, rh = ASPECT_PRESETS[aspect_ratio]
        return rw / rh

    def resolve_crop_box(self, w, h, cx, cy, cw, ch, ratio):
        px = min(w - 1, max(0, int(round(cx * w))))
        py = min(h - 1, max(0, int(round(cy * h))))
        pw = max(1, int(round(cw * w)))
        ph = max(1, int(round(ch * h)))

        pw = min(pw, w - px)
        ph = min(ph, h - py)

        if ratio is not None and ratio > 0:
            ideal_h = pw / ratio
            if ideal_h <= ph:
                ph = max(1, int(round(ideal_h)))
            else:
                pw = max(1, int(round(ph * ratio)))

            pw = min(pw, w - px)
            ph = min(ph, h - py)

        return px, py, max(1, pw), max(1, ph)

    def save_preview(self, image):
        results = []
        full_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            self.prefix, self.output_dir, image.shape[2], image.shape[1]
        )

        for i in range(image.shape[0]):
            arr = 255.0 * image[i].cpu().numpy()
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_folder, file), compress_level=1)

            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return results

    # ---- resize (mirrors UnifiedResizeImageMask) ----

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

            x = comfy.utils.common_upscale(x, sw, sh, method, False)

            _, _, ch, cw = x.shape
            top = max(0, (ch - target_h) // 2)
            left = max(0, (cw - target_w) // 2)

            x = x[:, :, top:top + target_h, left:left + target_w]
        else:
            x = comfy.utils.common_upscale(x, target_w, target_h, method, False)

        return x.movedim(1, -1)

    def resize_mask(self, mask, target_w, target_h, crop_mode):
        m = mask.unsqueeze(1).float()

        if crop_mode == "center":
            oh = m.shape[2]
            ow = m.shape[3]

            scale = max(target_w / ow, target_h / oh)

            sw = max(1, int(round(ow * scale)))
            sh = max(1, int(round(oh * scale)))

            m = F.interpolate(m, size=(sh, sw), mode="bilinear", align_corners=False)

            top = max(0, (sh - target_h) // 2)
            left = max(0, (sw - target_w) // 2)

            m = m[:, :, top:top + target_h, left:left + target_w]
        else:
            m = F.interpolate(m, size=(target_h, target_w), mode="bilinear", align_corners=False)

        return m.squeeze(1)

    # ---- combined ----

    def run(
        self,
        aspect_ratio="Free",
        custom_ratio_w=1,
        custom_ratio_h=1,
        crop_x=0.0,
        crop_y=0.0,
        crop_w=1.0,
        crop_h=1.0,
        show_crop_values=False,
        scale_mode=None,
        upscale_method="bilinear",
        post_crop="center",
        divisible_by=32,
        width=1024,
        height=1024,
        multiplier=1.0,
        megapixels=1.0,
        long_side_target=1024,
        short_side_target=768,
        maintain_aspect=True,
        image=None,
        mask=None,
    ):
        ui = {}

        kw = {
            "width": width,
            "height": height,
            "multiplier": multiplier,
            "megapixels": megapixels,
            "long_side_target": long_side_target,
            "short_side_target": short_side_target,
        }

        if image is None:
            # No image connected: act as a plain resolution selector, same
            # as Unified Resize's no-image fallback (t2i / t2v use case).
            # There's nothing to crop, so aspect_ratio / crop_x-y-w-h are
            # ignored here and the canvas widget has nothing to draw.
            w, h = self.resolve_size(scale_mode, 1, 1, kw)
            w, h = self.apply_divisible(w, h, divisible_by, maintain_aspect)
            w, h = max(1, int(w)), max(1, int(h))

            ui["text"] = [f"Resized to: {w} × {h} (no image — resolution only)"]
            empty = torch.zeros((1, 1, 1, 3))
            return {"ui": ui, "result": (empty, mask, w, h, 0, 0)}

        orig_h = image.shape[1]
        orig_w = image.shape[2]

        ratio = self.get_ratio(aspect_ratio, custom_ratio_w, custom_ratio_h)
        cx, cy, cw, ch = self.resolve_crop_box(orig_w, orig_h, crop_x, crop_y, crop_w, crop_h, ratio)

        cropped = image[:, cy:cy + ch, cx:cx + cw, :]
        cropped_mask = mask[:, cy:cy + ch, cx:cx + cw] if mask is not None else None

        w, h = self.resolve_size(scale_mode, cw, ch, kw)
        w, h = self.apply_divisible(w, h, divisible_by, maintain_aspect)
        w, h = max(1, int(w)), max(1, int(h))

        out_img = self.resize_image(cropped, w, h, upscale_method, post_crop)
        out_mask = self.resize_mask(cropped_mask, w, h, post_crop) if cropped_mask is not None else None

        try:
            # Not "images": that key triggers ComfyUI's built-in preview
            # strip on the node. This custom key is only read by our own
            # crop_canvas widget in crop_and_resize.js.
            ui["pk_crop_preview"] = self.save_preview(image)
            ui["text"] = [f"Resized to: {w} × {h}"]
        except Exception as e:
            print(f"[PlagueKind-Nodes] VisualCropAndResize preview save failed: {e}")

        return {"ui": ui, "result": (out_img, out_mask, w, h, cx, cy)}


NODE_CLASS_MAPPINGS = {
    "VisualCropAndResize": VisualCropAndResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualCropAndResize": "Visual Crop + Resize (BBox)"
}
