# ComfyUI-PlagueKind-Nodes

ComfyUI custom nodes providing unified image and mask resizing with multiple scaling modes, aspect-ratio preservation, center crop alignment, stable tensor-based mask transformations, and advanced LoRA stacking with audio/video branch control. The LoRA loader also functions as a standard LoRA loader for all compatible models, not limited to LTX workflows.

---

# Unified Resize Image / Mask

A single ComfyUI node that ensures consistent resizing behavior between images and masks using a unified geometric pipeline.

<img width="256" height="256" alt="Screenshot_20260513_233656" src="https://github.com/user-attachments/assets/9c4f69dd-8e9a-4ad8-a28e-66760a087793" />

### Features

* Multiple scaling modes:

  * Dimensions (W × H)
  * Multiplier
  * Longer Side
  * Shorter Side
  * Total Pixels (MP)

* Aspect-ratio preservation option

* Center crop alignment

* Divisible-by constraint (useful for latent models like LTX-2.3 / SDXL workflows, where other nodes only do one side.) Set divisible by 1 to disable.

* Unified image + mask transformation pipeline

* Stable tensor-based mask resizing (no PIL dependency issues)

### Why this node exists

Default ComfyUI workflows often suffer from:

* mask stretching inconsistencies
* image/mask misalignment after resize
* inconsistent crop behavior between pipelines

This node ensures both image and mask follow identical geometric transformations for predictable inpainting and compositing results.

### Node

**Unified Resize Image / Mask (Clean)**
Category: image

---

# Visual Crop + Resize (BBox)
 
Visual, drag-to-crop tools with aspect-ratio locking, available as a standalone crop node or combined with the resize pipeline in a single node.

 <img width="256" height="256" alt="Screenshot_20260806_001136" src="https://github.com/user-attachments/assets/753640d8-a821-4614-8665-1752bc1b2cd6" />

### Features
 
* Interactive drag-and-resize crop box overlay, drawn directly on the node
* Corner-handle resizing with aspect-ratio lock:
  * Free
  * 1:1, 4:3, 3:4, 16:9, 9:16, 21:9, 3:2, 2:3
  * Custom (numeric ratio)
* Normalized crop coordinates (0–1), so the box holds its relative position if the source resolution changes
* Optional numeric override of the crop box, hideable via a single toggle
* Outputs the crop origin (`x`, `y`) in source-pixel space for compositing the result back onto the original image
* Combined node chains the crop straight into the same scaling modes, divisible-by constraint, and post-scale center crop as Unified Resize
### Why these nodes exist
 
Cropping to a specific region or aspect ratio in ComfyUI normally means eyeballing pixel math or reaching for external tools. These nodes let a crop be drawn directly on the node after a single run, then reused and fine-tuned in place.
 
### Nodes
 
**Visual Crop (BBox)**
Category: image
Crop only, no resize. Outputs the cropped image/mask plus width, height, x, y.
 
**Visual Crop + Resize (BBox)**
Category: image
Crop followed by the full Unified Resize pipeline in one node.
 
---

# LTX LoRA Loader Stack (PlagueKind)

A 10-slot LoRA stacking node designed for LTX-2.3 workflows, featuring independent video and audio branch strength control per LoRA, optional CLIP passthrough, and structured stacking for advanced diffusion pipelines. This node also works as a standard LoRA loader for any compatible model.

<img width="256" height="171" alt="Screenshot_20260529_184720" src="https://github.com/user-attachments/assets/ed7e8083-8f1c-4b1e-89fb-8f60f2025f34" />


### Features

* Up to 10 stacked LoRA slots
* Per-slot enable / disable control
* Independent strength system:

  * S = master LoRA strength
  * V = video branch multiplier
  * A = audio branch multiplier
* Effective strengths:

  * Video = S × V
  * Audio = S × A
* Works as a standard LoRA loader for general models
* LoRA folder browser with search + nested directory support
* Missing LoRA detection warning
* Drag-and-drop slot reordering
* Optional CLIP input passthrough
* JSON-based stack serialization inside ComfyUI workflows

### Why this node exists

LTX-2.3 separates transformer processing into distinct audio and video branches, but most LoRA loaders treat all weights uniformly.

This node solves that limitation by allowing:

* targeted modulation of audio vs visual influence
* per-slot stacking instead of single LoRA application
* structured control over multi-LoRA compositions

### Node

**LoRA Loader Stack ( LTX Compatible )**
Category: PlagueKind/loaders

---

## Installation

### Manual install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes.git
```

### ComfyUI Manager

This node is also available via **ComfyUI Manager** for one-click installation.

Restart ComfyUI.

---

## Requirements

No external dependencies required beyond standard ComfyUI installation.

Uses:

* torch
* comfy.utils
* comfy.lora

---

## License

MIT License

---

## Support

If you find this project useful and want to support development:

Monero (XMR):
`865BrcfWLdwELwuq5faV1uVTbh93zVK6AUYLY2c3mX6sFfAGRfS6axe1kBTYYKuM7ccN7zBZDAZvnT7E4NKmUazySdbpc7p`

Thank you for your support.
