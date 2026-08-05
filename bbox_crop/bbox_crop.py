import os
import random

import numpy as np
import torch
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


class VisualBBoxCrop:

    aspect_modes = ["Free", "1:1", "4:3", "3:4", "16:9", "9:16", "21:9", "3:2", "2:3", "Custom"]

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
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height", "x", "y")

    FUNCTION = "crop"
    CATEGORY = "PlagueKind/image"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix = "bboxcrop_" + "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))

    def get_ratio(self, aspect_ratio, custom_w, custom_h):
        if aspect_ratio == "Free":
            return None
        if aspect_ratio == "Custom":
            return float(custom_w) / float(max(1, custom_h))
        rw, rh = ASPECT_PRESETS[aspect_ratio]
        return rw / rh

    def resolve_box(self, w, h, cx, cy, cw, ch, ratio):
        # Normalized box -> pixel box, clamped to image bounds.
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
        # Save the *uncropped* input as a temp preview so the frontend
        # widget has something to draw the drag box on top of.
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

    def crop(
        self,
        aspect_ratio="Free",
        custom_ratio_w=1,
        custom_ratio_h=1,
        crop_x=0.0,
        crop_y=0.0,
        crop_w=1.0,
        crop_h=1.0,
        show_crop_values=False,
        image=None,
        mask=None,
    ):
        ui = {}

        if image is None:
            empty = torch.zeros((1, 1, 1, 3))
            return {"ui": ui, "result": (empty, mask, 1, 1, 0, 0)}

        orig_h = image.shape[1]
        orig_w = image.shape[2]

        ratio = self.get_ratio(aspect_ratio, custom_ratio_w, custom_ratio_h)
        px, py, pw, ph = self.resolve_box(orig_w, orig_h, crop_x, crop_y, crop_w, crop_h, ratio)

        cropped = image[:, py:py + ph, px:px + pw, :]

        cropped_mask = None
        if mask is not None:
            cropped_mask = mask[:, py:py + ph, px:px + pw]

        try:
            # Not "images": that key triggers ComfyUI's built-in preview
            # strip on the node. This custom key is only read by our own
            # crop_canvas widget in bbox_crop.js.
            ui["pk_crop_preview"] = self.save_preview(image)
            ui["bbox"] = [[px, py, pw, ph, orig_w, orig_h]]
        except Exception as e:
            print(f"[PlagueKind-Nodes] VisualBBoxCrop preview save failed: {e}")

        return {"ui": ui, "result": (cropped, cropped_mask, pw, ph, px, py)}


NODE_CLASS_MAPPINGS = {
    "VisualBBoxCrop": VisualBBoxCrop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualBBoxCrop": "Visual Crop (BBox)"
}
