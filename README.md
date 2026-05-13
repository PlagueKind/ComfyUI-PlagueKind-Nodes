# Comfyui-PlagueKind-Nodes
ComfyUI custom node providing unified image and mask resizing with support for multiple scaling modes, aspect-ratio preservation, center crop alignment, and stable tensor-based mask transformations.

Custom nodes for ComfyUI focused on controlled image and mask processing workflows.

---

## Unified Resize Image / Mask

A single ComfyUI node that provides unified resizing logic for both images and masks with consistent geometry handling.

### Features

- Multiple scaling modes:
  - Dimensions (W × H)
  - Multiplier
  - Longer Side
  - Shorter Side
  - Total Pixels (MP)

- Aspect-ratio preservation option
- Center crop alignment
- Divisible-by constraint (useful for latent models like SDXL)
- Unified image + mask transformation pipeline
- Stable tensor-based mask resizing (no PIL dependency issues)

---

## Why this node exists

ComfyUI default workflows often suffer from:
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
