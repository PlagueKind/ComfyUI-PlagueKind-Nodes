# ComfyUI-PlagueKind-Nodes

ComfyUI custom nodes providing unified image and mask resizing with multiple scaling modes, aspect-ratio preservation, center crop alignment, and stable tensor-based mask transformations.

---

## Unified Resize Image / Mask

A single ComfyUI node that ensures consistent resizing behavior between images and masks using a unified geometric pipeline.

---

## Features

- Multiple scaling modes:
  - Dimensions (W × H)
  - Multiplier
  - Longer Side
  - Shorter Side
  - Total Pixels (MP)

- Aspect-ratio preservation option
- Center crop alignment
- Divisible-by constraint (useful for latent models like LTX-2.3 / SDXL workflows, where other nodes only do one side.)
- Unified image + mask transformation pipeline
- Stable tensor-based mask resizing (no PIL dependency issues)

---

## Why this node exists

Default ComfyUI workflows often suffer from:

- mask stretching inconsistencies  
- image/mask misalignment after resize  
- inconsistent crop behavior between pipelines  

This node ensures both image and mask follow identical geometric transformations for predictable inpainting and compositing results.

---

## Installation

### Manual install

Clone into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes.git

Clone into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes.git
