# ComfyUI-PlagueKind-Nodes

ComfyUI custom nodes for MiniMax H3 (and general) video workflows: unified image/mask resizing and visual crop tools, stacked LoRA / RefMod / media loaders, a combined reference-to-video prompt encoder, MiniMax-H3 model patches (block-sparse attention, residual cache, LoRA compatibility fix), and a full H3 video upscaler suite (latent upscale + light re-draw + long-form save).

The pack loads with no external dependencies beyond a standard ComfyUI install; every optional part degrades with a logged warning instead of blocking startup.

Current version: 1.6.4

---

# Unified Resize Image / Mask

A single ComfyUI node that ensures consistent resizing behavior between images and masks using a unified geometric pipeline.

### Features

* Multiple scaling modes:
  * Dimensions (W × H)
  * Multiplier
  * Longer Side
  * Shorter Side
  * Total Pixels (MP)
* Aspect-ratio presets (Auto / 1:1 / 2:3 / 3:2 / 3:4 / 4:3 / 9:16 / 16:9 / 21:9) — the same presets as ComfyUI core's ResolutionSelector; ignored in Dimensions mode
* Upscale method choice: nearest-exact, bilinear, area, bicubic, lanczos (resized through linear color for cleaner scaling)
* Center crop alignment (or disabled)
* Divisible-by constraint (useful for latent models like LTX-2.3 / SDXL workflows, where other nodes only do one side.) Set divisible by 1 to disable.
* Unified image + mask transformation pipeline
* Stable tensor-based mask resizing (no PIL dependency issues)
* Works with no image connected: acts as a plain resolution selector for t2i / t2v

### Why this node exists

Default ComfyUI workflows often suffer from:

* mask stretching inconsistencies
* image/mask misalignment after resize
* inconsistent crop behavior between pipelines

This node ensures both image and mask follow identical geometric transformations for predictable inpainting and compositing results.

### Node

**Unified Resize Image / Mask**
Category: PlagueKind/image
Outputs: image, mask, width, height

---

# Visual Crop (BBox) / Visual Crop + Resize (BBox)

Visual, drag-to-crop tools with aspect-ratio locking, available as a standalone crop node or combined with the resize pipeline in a single node.

### Features

* Interactive drag-and-resize crop box overlay, drawn directly on the node
* Corner-handle resizing with aspect-ratio lock:
  * Free
  * 1:1, 4:3, 3:4, 16:9, 9:16, 21:9, 3:2, 2:3
  * Custom (numeric ratio)
* Normalized crop coordinates (0–1), so the box holds its relative position if the source resolution changes
* Optional numeric override of the crop box, hideable via a single toggle
* Outputs the crop origin (`x`, `y`) in source-pixel space for compositing the result back onto the original image
* Combined node chains the crop straight into the same scaling modes, upscale methods, divisible-by constraint, and post-scale center crop as Unified Resize

### Why these nodes exist

Cropping to a specific region or aspect ratio in ComfyUI normally means eyeballing pixel math or reaching for external tools. These nodes let a crop be drawn directly on the node after a single run, then reused and fine-tuned in place.

### Nodes

**Visual Crop (BBox)**
Category: PlagueKind/image
Crop only, no resize. Outputs the cropped image/mask plus width, height, x, y.

**Visual Crop + Resize (BBox)**
Category: PlagueKind/image
Crop followed by the full Unified Resize pipeline in one node.

---

# LoRA Loader Stack

A 10-slot LoRA stacking node with a per-slot strength system, independent branch controls, and a folder-browser UI. Designed for LTX-2.3 and MiniMax H3 workflows, but works as a standard LoRA loader for any compatible model.

### Features

* Up to 10 stacked LoRA slots
* Per-slot enable / disable control
* Top-level mode toggle:
  * `normal` — one strength for the whole LoRA
  * `ltx` — video / audio branch split (S × V, S × A)
  * `minimax` — per-modality split for MiniMax H3's joint-attention stack: joint keys (S), video (S × V), audio (S × A), text (S × T). Keys that match no H3 I/O prefix stay joint, so a non-H3 LoRA forced into this mode still applies at full strength
* Per-slot key counts (J / V / A / T) reported for the selected LoRA, so you can see which scheme a file actually has before applying it
* Works as a standard LoRA loader for general models
* LoRA folder browser with search + nested directory support
* Missing LoRA detection warning
* Drag-and-drop slot reordering
* Optional CLIP input passthrough
* JSON-based stack serialization inside ComfyUI workflows

### Why this node exists

LTX-2.3 separates transformer processing into distinct audio and video branches, and MiniMax H3's joint-attention stack has a few genuinely modality-specific layers (patch projections, final outputs, text refiner) — but most LoRA loaders treat all weights uniformly.

This node solves that limitation by allowing:

* targeted modulation of audio vs visual vs text influence
* per-slot stacking instead of single LoRA application
* structured control over multi-LoRA compositions

### Node

**LoRA Loader Stack ( LTX / MiniMax H3 Compatible )**
Category: PlagueKind/loaders

---

# RefMod Loader Stack

A LoRA-loader-style stack UI for ComfyUI-MiniMaxH3Mod's `Load H3 RefMods` node. It does not reimplement any of that pack's logic — at call time it looks up the real `MiniMaxH3RefModsLoader` class through ComfyUI's node registry and calls its own `.load()` method, so mod discovery, safetensors loading, caching, token budgeting, and the H3_REF_MODS bundle format stay exactly as ComfyUI-MiniMaxH3Mod defines them.

### Features

* Up to 8 stacked RefMod slots (matching the backing node)
* Per-slot: mod picker, strength, copies (1–10), component filter (All / Visual / Audio), and visual / audio strength (0–1)
* Optional total-token budget (`max_total_tokens`, 0 = disabled)
* `show_info` console details for every loaded mod
* Outputs the H3_REF_MODS bundle plus a `prompt_hint` string
* Live-updating mod list; degrades to an empty bundle with a console warning if ComfyUI-MiniMaxH3Mod isn't installed

### Node

**RefMod Loader Stack ( MiniMax H3 RefMods )**
Category: PlagueKind/loaders

---

# PK Media Loader

Nine image + three video file slots in one node — the core Load Image and Load Video nodes, combined.

### Features

* `image_1..image_9` — image file pickers (uploadable), loaded by the core Load Image class itself (identical tensors to the stock node)
* `video_1..video_3` — video file pickers (uploadable), decoded to frame batches through the same pyav path the core video stack uses
* Per-video audio toggles (default on): when off, that video's audio channel is left empty
* `bundle` output — a BUNDLE dict keyed `input_1..input_15` (images 1–9, video frames 1–3, video audio 1–3) that drops straight into Bundle Out (PlagueKind)
* `image_1` output — image slot 1 as a plain IMAGE tensor, so the resize chain can stay wired without routing through Bundle Out
* Slots left on (none) are skipped; a file that can't be read warns and skips instead of killing the run
* Tabbed slot UI with live image/video previews in the frontend

### Node

**PK Media Loader ( 9 Images + 3 Videos )**
Category: PlagueKind/loaders

---

# Bundle In / Bundle Out (PlagueKind)

Dynamic multi-input → single bundle → multi-output plumbing for carrying mixed-type reference sets through a graph.

### Features

* 20 wildcard (`*`) slots on each end — any type connects
* Mapping between Bundle In's `input_i` and Bundle Out's `output_i` is positional
* Frontend auto expand/collapse and connected-type labeling
* Bundle Out's bundle input is optional (unwired → all empty outputs)

### Nodes

**Bundle In (PlagueKind)**
Category: PlagueKind/utils

**Bundle Out (PlagueKind)**
Category: PlagueKind/utils

---

# PK MiniMax H3 Combined Reference to Video

One prompt encode that can carry, all at once:

* first / last frame keyframes
* native reference images (auto-expanding slots, up to 9)
* native-style reference videos (up to 3) with index-paired soundtracks, and standalone reference audios (up to 3)
* a Media Loader bundle (H3_REFS)
* a RefMod bundle (H3_REF_MODS — shared with ComfyUI-MiniMaxH3Mod / the Fantastic Prompt Builder's loaders)

Built directly on ComfyUI's own `comfy_extras.nodes_minimax_h3` module — the same helpers the native H3 nodes use — so no other custom node pack is required.

### Features

* Keyframes and reference images can be used together. The native nodes treat them as mutually exclusive, but that was never a model-level requirement — `minimax_keyframes` and `minimax_refs` feed the same packed layout independently. This node allows both
* `ref_image_size`:
  * `match` — scale each reference (down only, aspect kept) to the generation's pixel area
  * `max` — use the reference pipeline's 2048px short edge for best identity fidelity (reference tokens ride through every sampling step, so it can be several times slower)
  * `none` — show references to the text encoder only; no VAE-encoded DiT reference block
* `vlm_resolution` / `vlm_video_resolution` — Qwen3-VL presentation resolution for stills / sampled video frames, independent of the resolution actually VAE-encoded as the DiT reference, so visual-token cost doesn't scale with reference fidelity (0 = same as the DiT reference)
* `reference_fps` — playback rate assumed when reconstructing RefMod video timestamps
* `max_total_tokens` — refuse RefMod bundles over this many reference tokens (0 = no limit)
* Outputs: conditioning, empty latent, and a `reference_map` string mapping each `<Picture n>` / `<Video n>` / `<Audio n>` tag to what it is

### Node

**PK MiniMax H3 Combined Reference to Video**
Category: PlagueKind/minimax

---

# PK MiniMax H3 Image

The still-image variant of the Combined node, rebuilt around ComfyUI's own p-h3-image conditioning internals.

### Features

* Sizing matches the native image node: every reference image is aspect-preserving, down-only, scaled to fit the generation's pixel area — `width`/`height` of 0 means "use the first reference image's size after scaling, or 2912x2912 without images"
* Latent math matches the native image node: one video temporal cell plus the two audio latents it spans
* Same reference_images autogrow, RefMod bundle, VLM presentation resolution, `reference_fps`, `max_total_tokens`, and three outputs as the Combined node
* The model needs ModelSamplingMiniMaxH3 at 1.0/1.0 (same as the native image node)

### Node

**PK MiniMax H3 Image**
Category: PlagueKind/minimax

---

# H3 SLA Attention

Block-sparse attention for MiniMax-H3. Each query block is scored against every key block with one small pooled matmul, and only the top `1 - sparsity_ratio` fraction is actually attended. Nothing is trained or loaded: the published SLA files contain only ordinary LoRA tensors, and the sparsity is decided at runtime from q and k — the LoRA's job is to make the model tolerate the sparsity, not to provide it.

### Features

* `sparsity_ratio` — fraction of key blocks skipped (default 0.80). 0.85 is the value the SLA turbo LoRA was distilled against; 0.90 is validated and ~15% faster. Break-even is about 0.60 — below that the kernel is slower than dense, so it's a real floor, not a safe fallback. 0.0 disables sparsity without removing the node
* `block_size` (32 / 64 / 128, default 32) — how many sequence tokens share one key selection. Matters far more for audio than video: H3 packs audio at 80 rows per second, so a 128-row block forces 1.6 s of audio down one attention pattern and speech comes out robotic; 64 is clean for ~2% more time, and 32 improves quality further for a marginal slowdown
* `min_seq_len` (default 12228) — sequences shorter than this stay dense, protecting the short text-refiner attention and short/low-res clips where sparsity would cost more than it saves
* `dense_last_steps` — run the final N sampling steps at full attention, since the last step's error is the one you actually see
* `dense_steps` — explicit 0-based step indices to force dense (e.g. `0,1` or `0-2`), on top of `dense_last_steps` — early steps set global composition and prompt adherence
* `protect_audio` — always attend blocks overlapping actual language tokens, target audio, and audio-reference segments, regardless of top-k, since audio is only ~1% of the packed sequence and plain top-k can drop it entirely. Disable only for turbo-style uniform sparsity
* `reference_protection` (Protect Vid/Ref) — `Heavy Enforcement` / `Light` / `Off` (default): protect image/video reference blocks with a reinforced quota or fixed 0.85 reference sparsity
* `engine` — `comfy_kitchen` (default: ComfyUI's compiled sol_attn kernel, needs comfy-kitchen>=0.2.32) or `triton` (this pack's own kernel; all other widgets apply fully to it)
* `dense_backend` — kernel used on every dense fall-through: `pytorch`, `comfy_kitchen`, the `sage:*` sage-attention modes, or `auto`
* `disable_fp16_accum` (default on) — force off the fp16/bf16 reduced-precision matmul reduction path for this model's sampling run; measured to cost quality on H3 with no throughput gain
* `stabilize_motion` — bias each layer's block selection toward last step's choice so near-ties don't flip into faint double-exposures on fast motion
* `tail_correction` (experimental) — fold one pooled term in for all key blocks top-k left out, so nothing leaves the softmax
* `use_int8_qk` (experimental) — int8 QK dot product on the selected top-k blocks (PV stays full precision)
* `enabled` — bypass toggle for a like-for-like dense speed baseline without rewiring the graph
* Follows ComfyUI's actual `WrapperExecutor` chain and derives its position in the sampler from `sample_sigmas`/`sigmas` rather than counting model invocations, so it stays correctly synced and composes cleanly with step-caching/forecasting accelerators (e.g. Spectrum) that skip some denoiser evaluations
* Automatic dense fallback if the kernel is missing, the GPU is unsupported, or the ComfyUI attention API changes — a broken patch never blocks a run, it just runs dense

Measured end-to-end on a 5090 at 768p/15s with the SLA turbo LoRA: ~44 s/it dense vs. ~31 s/it at sparsity 0.85 and ~25 s/it at 0.90 (1.4–1.75x), with no extra VRAM. Attention is only ~30 s of that 44 s step, so the ceiling is ~3.17x however fast attention itself gets — the widely-quoted 2.5x is an eight-GPU number. Sparsity did not turn out to drive H3 speech artifacts; step count did, so prefer 6+ sampling steps over lowering `sparsity_ratio`.

### Node

**H3 SLA Attention**
Category: PlagueKind/model_patches/minimax

---

# H3 MiniMax Cache

Approximate block-stack residual cache for MiniMax-H3 (experimental). H3 is a 33B model, so every sampling step is expensive and video sampling means a lot of steps — this node skips re-running the transformer block stack on steps whose features haven't moved much.

### Features

* `reuse_threshold` (default 0.05) — maximum accumulated relative feature change allowed before a skip is refused; higher skips more work and can cost fidelity
* `start_percent` / `end_percent` (default 0.15 / 0.90) — caching is withheld early (structure) and stopped late (detail polish)
* `max_steps` (default 2) — maximum consecutive block-stack skips
* `device` — `auto` (keep the cached residual with the model), `cuda`, or `cpu` (offload to system RAM for tight VRAM)
* Patches a *cloned* model patcher only — nothing about the core model class is touched globally, so a misbehaving cache only affects the clone you fed the sampler
* Fails safe: if the patch fails, the model passes through unchanged
* Ported from silveroxides/ComfyUI-UtilsCollection's UC_MiniMaxH3Cache (MIT), with permission. Composes with H3 SLA Attention in either order — the two patch different things

### Node

**H3 MiniMax Cache**
Category: PlagueKind/model_patches/minimax

---

# H3 AdaLN LoRA Fix

Makes dense (full-base) MiniMax-H3 LoRAs — including turbo LoRAs — work on pruned / curve-form H3 checkpoints, instead of being silently skipped with 51 `ERROR lora ... adaln_proj` lines per model load. Drop it on the MODEL wire anywhere after your LoRA loader(s) — rgthree's Power Lora Loader, ComfyUI's own, any stack. It operates on the patches those loaders already attached, so it needs no knowledge of which LoRAs were chosen and nothing upstream has to change.

### Why this node exists

A pruned checkpoint stores its AdaLN (timestep-modulation) projections over an 8-wide curve basis instead of the dense 2688-wide time embedding used by full H3 LoRAs. Without this fix, ComfyUI can't reconcile the shapes, logs one `ERROR` line per incompatible key, and drops all 51 — including on H3 turbo LoRAs, which is where this most commonly shows up. This node rebases those weights onto whichever basis the target model actually uses, so they apply instead of being dropped, and works in both directions (dense LoRA → pruned base, or curve-form LoRA → full base).

Measured on real H3 turbo LoRAs, the restored contribution is only ~0.02% of the modulation signal — so the practical benefit is a clean log, not a visible quality change.

### Features

* `mode` (default `port`):
  * `port` — rebase the AdaLN weights onto the basis the model actually uses. Quiet log, timestep modulation restored
  * `strip` — drop the incompatible weights. Quiet log, output identical to having no fix at all
  * `off` — passthrough, leaves the errors in place
* Fails safe: if the rebase itself errors, the unfixed model still runs and generates — it just logs the same errors it would have without the node

### Node

**H3 AdaLN LoRA Fix**
Category: PlagueKind/model_patches/minimax

---

# H3 Video Upsampler Suite

A finished MiniMax H3 clip in, a bigger and sharper one out. The chain is: fit the frames onto H3's grid, encode, enlarge in latent space with an H3 latent upscaler, then run a light re-draw in overlapping chunks (stepped together, so there are no seams) with a feathered tile merge, and decode. Derived from sandpies/ComfyUI-Hand-Tie-Clips, and replaces the pack's earlier H3 Ultimate Upscale (LD) / H3 Upscale Size nodes (v1.6.0).

### Features

* Small-face close-up re-draw: faces are found in the source frames (YuNet, bundled), tracked and smoothed, cut out with context, enlarged to close-up size, re-drawn by the same upsampler, and blended back with a feathered oval — so medium-far faces keep their person instead of getting plausible stranger faces. Needs OpenCV; without it the output is unchanged
* Two re-draw methods:
  * `regen2k` — NVlabs' Open-H3-Regenerate-2K LoRA re-draws detail from the enlarged clip (faster)
  * `ic` — in-context: each piece's own low-res draft stays in view as `<Video 1>` while it is re-drawn, and it hears the soundtrack, so faces, lettering and mouths stay the clip's own (slower). Needs the official `minimax_h3_ref2va_int8_convrot` model, an IC LoRA (Kijai's 200 MB rank-21 resize recommended) plus the HyperFlow 8-step LoRA (+ LMS Sharp)
* Long clips are cut into overlapping pieces (~4 s) that are re-drawn together (stepped joins) or crossfaded, and frames cut into overlapping tiles when even a piece is too much for the GPU
* `tile_tokens` auto-learns what your card can take and remembers it for the next run
* Audio retiming: a finished clip's soundtrack is re-timed to sit under the right frames for non-24 fps clips (the output soundtrack is always the original)
* Sparse or dense attention for the re-draw; optional text conditioning ("describe the look more than the objects")

### Model files

* Latent upscaler in `models/latent_upscale_models` — from huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler: the 3D file (the recipe) or the 2D file (takes the `sharpness` widget)
* `regen2k` method: `open_h3_regenerate_2k_r256pp_bf16.safetensors` (NVlabs) in `models/loras`
* `ic` method: IC LoRA (`minimax_h3_open_regenerate_2k_ic_lora_avg_rank_21_bf16.safetensors` recommended) + `minimax_h3_hyperflow_8step_v1.0.safetensors` (+ `minimax_h3_lms_v1.0_r64.safetensors`) in `models/loras`

### Nodes

**H3 Video Upsampler**
Category: PlagueKind/minimax/upscaler
The finished clip (frames + optional soundtrack + optional text conditioning) upsampled to about 2 MP in one node. Outputs images, audio, and a report.

**H3 Video Upsampler (file to file)**
Category: PlagueKind/minimax/upscaler
The same upsampler from a video file to an MP4, streaming — never holding the film's frames in memory.

**H3 Video Frames (lean)**
Category: PlagueKind/minimax/upscaler
Get Video Components, lean: a frame window, a frame rate, and a size, applied while decoding.

**H3 Load Video (path)**
Category: PlagueKind/minimax/upscaler
A VIDEO from a file path: no upload, no copy, no size limit.

**H3 Save Video (long-form)**
Category: PlagueKind/minimax/upscaler
Frames (+ soundtrack) to MP4 in one streaming pass, without the long-form RAM bill. NVENC (h264 / hevc / av1) on an RTX card or libx264 / libx265 / SVT-AV1 on CPU; constant quality (high / max / archive), not a fixed bitrate; optional in-save frame generation x2/x3/x4 via NVIDIA Video Effects (needs the nvvfx package); a keyframe forced on every hard cut and at least every two seconds, so editors scrub the film cleanly; the soundtrack is trimmed or padded to the picture and interleaved as it goes.

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
* comfy.utils / comfy.lora / folder_paths
* aiohttp, server (ComfyUI's own HTTP layer)

Optional, per feature:

* **H3 SLA Attention** — the `triton` engine needs Triton (ships with CUDA PyTorch); the `comfy_kitchen` engine needs comfy-kitchen>=0.2.32; the `sage:*` dense backends need the sageattention package. Any failure falls back to dense with a logged warning
* **H3 Video Upsampler** — the model files listed above; OpenCV for the small-face pass (the YuNet detector model is bundled); the `ic` method's preflight stops before any work and says what's missing
* **H3 Save Video frame generation** — an RTX card and the nvvfx package (installed with the NVIDIA RTX nodes)
* **PK MiniMax H3 Combined / Image** — a ComfyUI build with native MiniMax H3 support (`comfy_extras.nodes_minimax_h3`)
* **RefMod Loader Stack** — ComfyUI-MiniMaxH3Mod for real mods; without it the node returns an empty bundle with a console warning

---

## Credits

* H3 Video Upsampler suite — derived from sandpies/ComfyUI-Hand-Tie-Clips (H3 Video Upsampler 0.7.1: +2D latent upscaler; 0.7.0: in-context re-draw method)
* H3 MiniMax Cache — ported from silveroxides/ComfyUI-UtilsCollection's UC_MiniMaxH3Cache (MIT), with permission
* PK MiniMax H3 Combined — VLM presentation-resolution helpers from silveroxides/ComfyUI-UtilsCollection (AGPLv3), with permission; RefMod bundle helpers from Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder (MIT), with permission
* Re-draw models — NVlabs' Open-H3-Regenerate-2K / -IC LoRAs, LBH-123-AI's Minimax_h3_latent_Upscaler, Kijai's rank-21 resize
* YuNet face detector (bundled ONNX)

---

## License

MIT License

---

## Support

If you find this project useful and want to support development:

Monero (XMR):
`865BrcfWLdwELwuq5faV1uVTbh93zVK6AUYLY2c3mX6sFfAGRfS6axe1kBTYYKuM7ccN7zBZDAZvnT7E4NKmUazySdbpc7p`

Thank you for your support.
