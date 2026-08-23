# ComfyUI-PlagueKind-Nodes

ComfyUI custom nodes providing unified image and mask resizing with multiple scaling modes, aspect-ratio preservation, center crop alignment, stable tensor-based mask transformations, advanced LoRA stacking with audio/video branch control, and MiniMax-H3 attention/LoRA-compatibility fixes. The LoRA loader also functions as a standard LoRA loader for all compatible models, not limited to LTX workflows.

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

A 10-slot LoRA stacking node designed for LTX-2.3 workflows, featuring independent video and audio branch strength control per LoRA, optional CLIP passthrough, and structured stacking for advanced diffusion pipelines. This node also works as a standard LoRA loader for any compatible model, and supports MiniMax H3 with per-modality strength controls.

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
* Top-level mode toggle (normal / ltx / minimax) with auto-detection via LoRA key-name sniffing
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

# H3 SLA Attention

Block-sparse attention for MiniMax-H3.

### Features

* `sparsity_ratio` — fraction of key blocks skipped (default 0.90). 0.85 is lightx2v's shipped value; 0.90 measured ~15% faster. Below ~0.60 the kernel is slower than dense, so it's a real floor, not a safe fallback.
* `block_size` (64 / 128) — how many sequence tokens share one key selection. Matters far more for audio than video: 128 forces 1.6 s of audio down one attention pattern and speech comes out robotic; 64 is clean for ~2% more time.
* `min_seq_len` — sequences shorter than this stay dense, protecting the short text-refiner attention and short/low-res clips where sparsity would cost more than it saves.
* `dense_last_steps` — run the final N sampling steps at full attention to recover fine detail, since the last step's error is the one you actually see.
* `protect_audio` — always attends the [text | cond | audio] prefix regardless of top-k, since audio is only ~1% of the packed sequence and plain top-k can drop it entirely. Costs ~7%.
* `enabled` — bypass toggle for a like-for-like dense speed baseline without rewiring the graph.
* Follows ComfyUI's actual `WrapperExecutor` chain and derives its position in the sampler from `sample_sigmas`/`sigmas` rather than counting model invocations, so it stays correctly synced and composes cleanly with step-caching/forecasting accelerators (e.g. Spectrum) that skip some denoiser evaluations.
* Automatic dense fallback if Triton is missing, the GPU is unsupported, or the ComfyUI attention API changes — a broken patch never blocks a run, it just runs dense.

Measured end-to-end on a 5090 at 768p/15s with the SLA turbo LoRA: ~44 s/it dense vs. ~31 s/it at sparsity 0.85 and ~25 s/it at 0.90 (1.4–1.75x), with no extra VRAM. Attention is only ~30 s of that 44 s step, so the ceiling is ~3.17x however fast attention itself gets — the widely-quoted 2.5x is an eight-GPU number. Sparsity did not turn out to drive H3 speech artifacts; step count did, so prefer 6+ sampling steps over lowering `sparsity_ratio`.

### Node

**H3 SLA Attention**
Category: PlagueKind/model_patches/minimax

---

# H3 AdaLN LoRA Fix

Makes dense (full-base) MiniMax-H3 LoRAs — including turbo LoRAs — work on pruned / curve-form H3 checkpoints, instead of being silently skipped with 51 `ERROR lora ... adaln_proj` lines per model load. Drop it on the MODEL wire anywhere after your LoRA loader(s) — rgthree's Power Lora Loader, ComfyUI's own, any stack. It operates on the patches those loaders already attached, so it needs no knowledge of which LoRAs were chosen and nothing upstream has to change.

### Why this node exists

A pruned checkpoint stores its AdaLN (timestep-modulation) projections over an 8-wide curve basis instead of the dense 2688-wide time embedding used by full H3 LoRAs. Without this fix, ComfyUI can't reconcile the shapes, logs one `ERROR` line per incompatible key, and drops all 51 — including on H3 turbo LoRAs, which is where this most commonly shows up. This node rebases those weights onto whichever basis the target model actually uses, so they apply instead of being dropped, and works in both directions (dense LoRA → pruned base, or curve-form LoRA → full base).

Measured on real H3 turbo LoRAs, the restored contribution is only ~0.02% of the modulation signal — so the practical benefit is a clean log, not a visible quality change.

### Features

* `mode`:
  * `port` — rebase the AdaLN weights onto the basis the model actually uses. Quiet log, timestep modulation restored.
  * `strip` — drop the incompatible weights. Quiet log, output identical to having no fix at all.
  * `off` — passthrough, leaves the errors in place.
* Fails safe: if the rebase itself errors, the unfixed model still runs and generates — it just logs the same errors it would have without the node.

### Node

**H3 AdaLN LoRA Fix**
Category: PlagueKind/model_patches/minimax

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

H3 SLA Attention additionally requires Triton and a supported GPU; if either is missing, the node logs a warning and the pack loads without it rather than blocking ComfyUI startup.

---

## License

MIT License

---

## Support

If you find this project useful and want to support development:

Monero (XMR):
`865BrcfWLdwELwuq5faV1uVTbh93zVK6AUYLY2c3mX6sFfAGRfS6axe1kBTYYKuM7ccN7zBZDAZvnT7E4NKmUazySdbpc7p`

Thank you for your support.
