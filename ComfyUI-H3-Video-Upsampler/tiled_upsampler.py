# Generated from ComfyUI-Hand-Tie-Clips (https://github.com/sandpies/ComfyUI-Hand-Tie-Clips) by tools/build_upsampler_pack.py.
# Edit it there, not here: the next build overwrites this file.
"""HTC Tiled Upsampler -- a finished H3 clip made bigger and sharper, in one node.

WHAT IT DOES, in order, once per time chunk of the clip:

1. **Upscale in latent space.** LBH-123-AI's learned H3 latent upscaler takes
   the chunk's video latent to the size the megapixels widget asks for. Audio
   is never touched.
2. **Re-draw it lightly, in tiles.** One sampling step at denoise 0.2 on the
   `simple` schedule sharpens what the upscale smeared. The frame is cut into
   the fewest equal tiles whose forward fits a fixed token budget, and the
   tiles are stepped TOGETHER: one forward per tile, merged into one latent
   under a raised-cosine feather, before anything moves on. Every tile slices
   one noise field drawn at the chunk's full size, so two tiles covering the
   same spot start from the same noise. Together that is why there are no tile
   seams.
3. **Join the chunks.** Chunks overlap by one 17-frame grid step. Each chunk
   after the first has its frame-0 keyframe pinned to the previous chunk's
   re-drawn frame, and the overlap is crossfaded linearly.

Then the video is decoded with the VAE and the audio with the audio VAE.

THE RECIPE IS FIXED. Every number below is the one from the user's
MMH3 Ultimate Upscale graph (bbaudio-2025's Comfyui-MMH3-UltimateUpscale,
MIT), which is where this recipe was proven: 102-frame chunks with a 17-frame
overlap, anchor 0.999, auto tiling at 70,000 tokens per forward with 128 px
overlaps and a 256 px floor, joint steps on, one `simple` step at denoise 0.2,
seed 444, upscaler in fp16. Only the size is a widget; it defaults to 2 MP
(the proving run used 4, a stress test). This file is HTC's own
code; tools/check_tiled_upsampler.py runs it against that pack, when it is
installed, and requires identical output.

The upscaler network layout, key names and the 48 normalisation numbers are
LBH-123-AI's (the weights only load in the layout they were trained in; see
THIRD_PARTY_NOTICES.md). The runner is this pack's.
"""
from __future__ import annotations

import collections
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

TAG = "H3VideoUpsampler"

# --- the recipe -------------------------------------------------------------
UPSCALER_FILE = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
# The same 3D weights under every name LBH-123-AI has published them: HF renamed
# minimax_h3_latent_upscaler_3d_bf16 to ..._3d_conv_v1_bf16 on 2026-11-10, and
# saved workflows hold either. Any of them loads whichever of them is on disk.
UPSCALER_3D_NAMES = (UPSCALER_FILE, "minimax_h3_latent_upscaler_3d_bf16.safetensors")
# LBH-123-AI's 2D release (2026-11-10): Conv2d per frame + shared temporal
# blocks, 182M params, with 6 sharpness tiers the network is conditioned on.
UPSCALER_2D_FILE = "minimax_h3_latent_upscaler_2d_conv_v1_bf16.safetensors"
SHARPNESS = 1               # 2D tier: 1 most faithful (the 3D's look) .. 6 sharpest
SHARPNESS_MAX = 6
SHARPNESS_TIP = ("Sharpness tier of the 2D latent upscaler (" + UPSCALER_2D_FILE + "): "
                 "1 = most faithful to the clip (softest, about the 3D upscaler's look), "
                 "6 = sharpest (most new detail, furthest from the clip). The 3D upscaler "
                 "has no tiers and ignores it.")
MEGAPIXELS = 2.0            # 1 MP = 1024 x 1024 px, the only widget
SIZE_ALIGN = 32             # both edges snap to this, aspect held
CHUNK_FRAMES = 102          # pixel frames per time chunk (a multiple of 17)
CHUNK_OVERLAP = 17          # pixel frames shared by neighbouring chunks
ANCHOR_AUG = 0.999          # frame-0 pin's visual cond noise aug (H3 default)
TOKEN_BUDGET = 70000        # largest sequence one tile forward may carry
TILE_OVERLAP_PX = 128       # desired overlap between tiles, both axes
MIN_TILE_PX = 256           # no tile smaller than this
MAX_GRID_SIDE = 9           # tile grid search limit per axis
SCHEDULER, STEPS, DENOISE = "simple", 1, 0.2
SEED = 444
UPSCALER_DTYPE = torch.float16
_FLUSH_S = [0.0]           # seconds spent in sample_tile's cache flush (log only)
_SAMPLE_S = [0.0]          # seconds spent inside the sampler itself (log only)
_LAST_OOM = [0]            # OOM retries in the most recent redraw_chunk call (log only)
_LAST_BUDGET = [TOKEN_BUDGET]  # budget redraw_chunk / upsample finished at
_LAST_TOKENS = [0]          # tokens per forward of the last redraw_chunk grid (incl. cond rows)


class _DefaultInt(int):
    """Sentinel int equal to CHUNK_FRAMES that marks an un-overridden default."""


_DEFAULT_CHUNK_FRAMES = _DefaultInt(CHUNK_FRAMES)
REGEN2K_MIN_CHUNK_FRAMES = 68
REGEN2K_SEED = 42
REGEN2K_SIGMAS = torch.tensor([0.8, 24.0 / 35.0, 0.48, 0.0], dtype=torch.float32)
REGEN2K_SIGMAS_RELEASE = torch.tensor([0.909375, 6.0 / 7.0, 2.0 / 3.0, 0.0], dtype=torch.float32)
REGEN2K_SIGMAS_2STEP = torch.tensor([0.7, 0.494118, 0.0], dtype=torch.float32)
REGEN2K_SHIFT = 6.0


def regen2k_sigmas(start=0.8, steps=3, shift=REGEN2K_SHIFT):
    """Regen2K's grid from any `start`: `steps` even steps in unshifted sigma,
    then shifted (s*x / (1 + (s-1)*x)). start 0.8 gives REGEN2K_SIGMAS exactly
    ([0.8, 24/35, 0.48, 0]); lower starts from less noise, so the re-draw
    re-imagines less (a face stays closer to the source) and sharpens less."""
    start = float(start)
    raw = start / (shift - (shift - 1.0) * start)
    sig = [start] + [shift * x / (1.0 + (shift - 1.0) * x)
                     for x in (raw * (1.0 - i / steps) for i in range(1, steps))] + [0.0]
    return torch.tensor(sig, dtype=torch.float32)
# Structure lock: the re-draw's large shapes held to the 3D upscaler's (see
# structure_lock_fn). 0 = off, the recipe as it was. LOCK_SIGMA is the blur in
# latent cells (16 px each) that separates "layout" from "detail".
STRUCTURE_LOCK = 0.0
LOCK_SIGMA = 2.0
# attention: ComfyUI's Model Sparse Attention (sol-attn) on every step, or dense.
# chunk_join: crossfade two independent re-draws over the shared frames, or
#   stepped -- every chunk (and every tile of a tiled chunk) takes each step
#   together and is merged before the next, so the film is one grid in time
#   and space.
# chunking / tiling: each one ALLOWS a split when the clip does not fit
#   tile_tokens; neither forces one. chunking cuts time into 102-frame chunks
#   (no shorter than 68 -- Regen2K's 3 steps from 0.8 invent on short pieces);
#   tiling cuts space. Time is cut before space: a tile sees part of the frame,
#   which is where invention comes from, while a join in time can be stepped
#   together (chunk_join=stepped). Both off = the whole clip in one whole-frame
#   pass.
# lock_anchor: what structure_lock holds the large shapes to. source = the
#   clip's own latent, enlarged by plain interpolation (nothing invented, so
#   errors the 3D upscaler adds -- a blown patch, a colour shift -- are pulled
#   back too); upscaler = the 3D upscaler's output.
ATTENTION_MODES = ("dense", "sparse")
JOIN_MODES = ("crossfade", "stepped")
LOCK_ANCHORS = ("source", "upscaler")
# tile_context (draft): what a tile knows about the rest of its frame. none =
#   only its own window (and structure_lock's anchor). global pass = before the
#   tiles, the chunk is re-drawn WHOLE-FRAME at the largest size that fits the
#   budget, with the same steps; the tiles then hold their large shapes to that
#   result (structure_lock at full strength), so the layout -- a face, a hand, a
#   line -- is decided by a pass that saw all of it, and the tiles add detail.
#   Only chunks that are actually tiled pay for it.
TILE_CONTEXTS = ("none", "global pass")
WHOLE_CLIP_MAX = 4          # chunking off only up to this many tile_tokens (else chunked)

# --- the guided re-draw (not part of the recipe) ----------------------------
#
# With an h3upscale LoRA on the re-draw model, the same one step also gets the
# LoRA's guide: the film at 1/factor of the output size, packed on every
# factor-th patch of each tile. The recipe's numbers do not move; the output
# size snaps to 32 * factor, tiles snap to the same grid, the guide's rows
# count against the token budget, and the conditioning is the LoRA's caption.
# Without a guide every line below behaves exactly as before.
STRIDE_KEY = "htc_grid_stride"     # upscale_guide.STRIDE_KEY; the layout patch reads it

# factor: the LoRA's reference_downscale_factor. cond: the caption
# conditioning, used for every chunk. decode(video latent) -> frames and
# encode(frames) -> video latent are the chain's VAE; decode may be None and
# upsample_chain fills in its own. pixels(frames, w_px, h_px, factor) shrinks
# canvas frames to the guide (upscale_guide.guide_pixels).
Guide = collections.namedtuple("Guide", "factor cond decode encode pixels")

FOLDER = "latent_upscale_models"
SPATIAL = 16                # H3 video VAE: pixels per latent cell
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0   # audio latent steps per pixel frame (40 vs 24 Hz)

# The statistics LBH-123-AI's weights were trained against (24 channels).
LATENT_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328,
    -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
    -0.2553254961967468, -0.26907554268836975, -0.5376840829849243,
    -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
    0.25928452610969543, -0.30133944749832153, 0.211341992020607,
    -1.1206848621368408, 0.3581933379173279, -0.04225143790245056,
    0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
LATENT_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887,
    1.7549455165863037, 1.5636216402053833, 2.194143533706665,
    0.9653137922286987, 1.0569885969161987, 0.841948926448822,
    0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
    0.6936293244361877, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180145264, 3.276226282119751,
    3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)


def register_folder():
    """models/latent_upscale_models, unless something registered it already."""
    import folder_paths
    if FOLDER not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path(
            FOLDER, os.path.join(folder_paths.models_dir, FOLDER))


def choices():
    try:
        import folder_paths
        names = [n for n in folder_paths.get_filename_list(FOLDER)
                 if n.endswith(".safetensors")]
    except Exception:  # noqa: BLE001
        names = []
    # Both 3D names always listed (saved workflows hold either, and ComfyUI
    # refuses a combo value missing from the list); the installed one first.
    first = default_upscaler(names)
    rest = [n for n in UPSCALER_3D_NAMES if n != first]
    return [first] + rest + sorted(n for n in names if n not in UPSCALER_3D_NAMES)


def default_upscaler(names=None):
    """The 3D file under the name it is installed as (the new name when neither is)."""
    if names is None:
        try:
            import folder_paths
            names = folder_paths.get_filename_list(FOLDER)
        except Exception:  # noqa: BLE001
            names = []
    for n in UPSCALER_3D_NAMES:
        if n in names:
            return n
    return UPSCALER_FILE


def resolve_upscaler(name):
    """`name`'s full path; a 3D name falls back to the other 3D names. -> (name, path)."""
    import folder_paths
    path = folder_paths.get_full_path(FOLDER, str(name))
    if path:
        return str(name), path
    if str(name) in UPSCALER_3D_NAMES:
        for alt in UPSCALER_3D_NAMES:
            path = folder_paths.get_full_path(FOLDER, alt)
            if path:
                return alt, path
    raise RuntimeError(f"{TAG}: {name!r} is not in models/{FOLDER}. Download "
                       f"{UPSCALER_FILE} (3D) or {UPSCALER_2D_FILE} (2D) from "
                       "huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler "
                       f"into models/{FOLDER}.")


def is_2d_file(name):
    """True when `name` holds LBH-123-AI's 2D upscaler (read from its keys only)."""
    try:
        from safetensors import safe_open
        _, path = resolve_upscaler(name)
        with safe_open(path, framework="pt", device="cpu") as f:
            return any(k.startswith("resizer.") for k in f.keys())
    except Exception:  # noqa: BLE001
        return False


# --- size -------------------------------------------------------------------

def target_size(src_w, src_h, megapixels, align=SIZE_ALIGN):
    """(w, h) in pixels: `megapixels` of area at the source aspect, both edges
    on the `align` grid.

    Of the nine grid points around the ideal size, the one whose aspect is
    closest wins; area only breaks ties, weighted a quarter as much, because a
    bent aspect is what people notice and the area is what they asked for.
    """
    src_w, src_h, align = max(1, int(src_w)), max(1, int(src_h)), max(1, int(align))
    aspect = src_w / float(src_h)
    area = float(megapixels) * 1024.0 * 1024.0
    ideal_h = (area / aspect) ** 0.5
    ideal_w = ideal_h * aspect
    area = ideal_w * ideal_h

    def snap(v):
        return max(align, int(round(float(v) / align)) * align)

    best = None
    for dw in (-1, 0, 1):
        w = snap(ideal_w) + dw * align
        if w < align:
            continue
        for dh in (-1, 0, 1):
            h = snap(ideal_h) + dh * align
            if h < align:
                continue
            score = (abs(w / float(h) - aspect) / aspect * 4.0
                     + abs(w * h - area) / max(1.0, area))
            if best is None or score < best[0]:
                best = (score, w, h)
    return best[1], best[2]


# --- time: frames, tokens, chunks -------------------------------------------

def frames_in(tokens):
    """Pixel frames covered by the first `tokens` video latent steps."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(tokens)))


def tokens_to_reach(frames):
    """Fewest video latent steps whose frames reach at least `frames`."""
    k = got = 0
    while got < frames:
        got += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k


def audio_span(f0, f1):
    """Audio latent steps [a0, a1) under pixel frames [f0, f1)."""
    return round(f0 * FRAME_RESCALE), round(f1 * FRAME_RESCALE)


def _nearest_key_step(frame, tokens):
    """The key step (index % 5 == 0) whose start frame is nearest `frame`."""
    best_k, best_f, best_d = 0, 0, frame
    for k in range(0, tokens + 1, 5):
        f = frames_in(k)
        if abs(f - frame) < best_d:
            best_k, best_f, best_d = k, f, abs(f - frame)
    return best_k, best_f


def chunks(tokens, length=CHUNK_FRAMES, overlap=CHUNK_OVERLAP):
    """[(k0, f0, k1, f1)]: each chunk's latent steps and frames.

    Boundaries sit on key steps, a chunk never starts after the previous one
    ended, and the last chunk ends exactly on the clip's last frame.
    """
    total = frames_in(tokens)
    hop = length - overlap
    out, prev_end, i = [], 0, 0
    while True:
        start, end = i * hop, min(i * hop + length, total)
        if i == 0:
            k0, f0 = 0, 0
        else:
            k0, f0 = _nearest_key_step(start, tokens)
            if k0 > prev_end:
                k0, f0 = prev_end, frames_in(prev_end)
        if end >= total:
            k1, f1 = tokens, total
        else:
            k1, f1 = _nearest_key_step(end, tokens)
            if k1 <= k0:
                k1 = k0 + 5
                f1 = frames_in(k1)
            if k1 >= tokens:
                k1, f1 = tokens, total
        out.append((k0, f0, k1, f1))
        if k1 >= tokens:
            return out
        prev_end = k1
        i += 1


def regen2k_chunks(tokens, length=CHUNK_FRAMES, overlap=CHUNK_OVERLAP,
                   w_px=0, h_px=0, budget=TOKEN_BUDGET, overlap_px=TILE_OVERLAP_PX):
    """[(k0, f0, k1, f1)] for Regen2K: eliminates short tail-stub chunks (< 20 steps).

    A 192-frame (57-step) clip under standard 102/17 f chunking ends with a 7-step
    (22-frame) stub `(50, 170, 57, 192)` just to cover the last 2 steps (5 frames);
    3 high-sigma Euler steps on a 7-step stub hallucinate end-of-clip artifacts.
    1. When the final chunk adds <= 5 new steps (<= 17 frames) and absorbing it into
       the previous chunk keeps the same spatial tile grid under `budget`, merge them.
    2. Otherwise, when the final chunk is shorter than 20 steps (68 frames), slide its
       start back on the 5-step key grid so it runs at least 20 steps (bounded by
       `pk0 + 5` so chunk starts strictly advance and never exceed the previous chunk's
       length).
    """
    plan = chunks(tokens, length, overlap)
    if len(plan) < 2:
        return plan
    pk0, pf0, pk1, pf1 = plan[-2]
    lk0, lf0, lk1, lf1 = plan[-1]
    rem_steps = lk1 - pk1
    merged_steps = lk1 - pk0
    if rem_steps <= 5:
        can_absorb = True
        if w_px > 0 and h_px > 0:
            ov = snap_overlap(overlap_px)
            g_base = solve_grid(w_px, h_px, pk1 - pk0, overlap_px=ov, budget=int(budget))
            g_merged = solve_grid(w_px, h_px, merged_steps, overlap_px=ov, budget=int(budget))
            can_absorb = (g_merged is not None and g_base is not None
                          and (g_merged[0], g_merged[1]) == (g_base[0], g_base[1]))
        if can_absorb:
            plan[-2] = (pk0, pf0, lk1, lf1)
            plan.pop()
            return plan
    min_steps = max(5, min((int(length) // 17) * 5, (REGEN2K_MIN_CHUNK_FRAMES // 17) * 5))
    if (lk1 - lk0) < min_steps:
        new_k0 = max(pk0 + 5, ((lk1 - min_steps) // 5) * 5)
        if new_k0 < lk0:
            plan[-1] = (new_k0, frames_in(new_k0), lk1, lf1)
    return plan


# --- space: the tile grid ---------------------------------------------------

def _equal_split(total_px, count, want_overlap_px, grid=32):
    """(tile_px, overlap_px) for `count` equal tiles covering `total_px`
    exactly, the overlap on the `grid` and as close to the wanted one as the
    divisibility allows. None when no such split exists."""
    if count <= 1:
        return -(-int(total_px) // grid) * grid, 0
    lo = -(-((int(total_px) + (count - 1) * int(want_overlap_px)) // count)
           // grid) * grid
    hi = int(total_px) - grid * (count - 1)
    for tile in range(lo, hi + 1, grid):
        spare = count * tile - int(total_px)
        if spare % (count - 1) == 0:
            ov = spare // (count - 1)
            if ov % grid == 0 and 0 <= ov <= tile - grid:
                return tile, ov
    return None


def seq_len(tile_w_px, tile_h_px, tokens):
    """One forward's sequence length: 2x2-patched cells x video latent steps."""
    return (int(tile_w_px) // 32) * (int(tile_h_px) // 32) * int(tokens)


def factor_seq_len(w_px, h_px, tokens, factor=1):
    """One forward's sequence for a whole frame, the guide's rows included."""
    return (seq_len(w_px, h_px, tokens) if int(factor) <= 1
            else guided_seq_len(w_px, h_px, tokens, factor))


def guided_seq_len(tile_w_px, tile_h_px, tokens, factor):
    """`seq_len` plus the guide's rows: one per factor-th patch, every step."""
    f = int(factor)
    return (seq_len(tile_w_px, tile_h_px, tokens)
            + (int(tile_w_px) // (32 * f)) * (int(tile_h_px) // (32 * f)) * int(tokens))


def solve_grid(frame_w_px, frame_h_px, tokens, overlap_px=TILE_OVERLAP_PX,
               budget=TOKEN_BUDGET, min_tile_px=MIN_TILE_PX,
               max_side=MAX_GRID_SIDE, factor=1):
    """The fewest equal tiles whose forward fits `budget`.

    -> (rows, cols, tile_w_px, tile_h_px, overlap_w_px, overlap_h_px) or None.
    Among grids of the same size, the one that splits the longer frame axis
    more wins (strips, not a checkerboard), then the squarer one. A guide
    `factor` > 1 puts tiles and overlaps on the 32 * factor grid and counts the
    guide's rows in the budget.
    """
    grid = 32 * int(factor) if int(factor) > 1 else 32
    for n in range(1, max_side * max_side + 1):
        best = None
        for rows in range(1, max_side + 1):
            if n % rows:
                continue
            cols = n // rows
            if cols > max_side:
                continue
            across = _equal_split(frame_w_px, cols, overlap_px, grid)
            down = _equal_split(frame_h_px, rows, overlap_px, grid)
            if across is None or down is None:
                continue
            tw, ow = across
            th, oh = down
            if tw < min_tile_px or th < min_tile_px:
                continue
            cost = (guided_seq_len(tw, th, tokens, factor) if int(factor) > 1
                    else seq_len(tw, th, tokens))
            if cost > budget:
                continue
            long_split = rows if frame_h_px >= frame_w_px else cols
            rank = (-long_split, abs(rows - cols))
            if best is None or rank < best[0]:
                best = (rank, rows, cols, tw, th, ow, oh)
        if best is not None:
            return best[1:]
    return None


def _axis(size, tile, overlap, floor):
    """Tile starts, extents and seam overlaps along one axis, in latent cells.
    A leftover edge tile smaller than `floor` is pulled back to `floor`."""
    if size <= tile:
        return [0], [size], [0]
    stride = tile - overlap
    n = math.ceil((size - overlap) / stride)
    if (n - 1) * stride + tile < size:
        n += 1
    starts = [i * stride for i in range(n)]
    extents = [min(tile, size - s) for s in starts]
    if floor > 0 and n >= 2 and size - starts[-1] < floor:
        pulled = size - floor
        if starts[-2] < pulled < starts[-2] + extents[-2]:
            starts[-1], extents[-1] = pulled, size - pulled
    seams = [0] + [max(0, starts[i - 1] + extents[i - 1] - starts[i])
                   for i in range(1, n)]
    return starts, extents, seams


def snap_overlap(px):
    """Tile overlap in px, rounded to the 32 px patch grid, 0..512."""
    return max(0, min(512, int(round(float(px) / 32.0)) * 32))


def tile_grid(h, w, tokens, factor=1, budget=TOKEN_BUDGET, overlap_px=TILE_OVERLAP_PX):
    """The tiles for an (h, w)-cell chunk of `tokens` video steps.

    -> list of (r0, c0, rows, cols, seam_top, seam_left, top, left, bottom,
    right) in latent cells, plus the solved grid for the log. `budget` is the
    largest sequence one tile forward may carry: the recipe's TOKEN_BUDGET
    unless a caller lowers it to fit a smaller card.
    """
    got = solve_grid(w * SPATIAL, h * SPATIAL, tokens, overlap_px=snap_overlap(overlap_px),
                     budget=int(budget), factor=factor)
    if got is None:
        raise RuntimeError(
            f"{TAG}: no tile grid up to {MAX_GRID_SIDE}x{MAX_GRID_SIDE} keeps "
            f"a {w * SPATIAL}x{h * SPATIAL}px chunk of {tokens} latent steps "
            f"under {int(budget)} tokens per forward. Lower megapixels, or raise "
            f"the tile budget.")
    _r, _c, tw, th, ow, oh = got
    tw, th, ow, oh = tw // SPATIAL, th // SPATIAL, ow // SPATIAL, oh // SPATIAL
    floor = MIN_TILE_PX // SPATIAL
    r0s, rh, rseam = _axis(h, th, oh, floor)
    c0s, cw, cseam = _axis(w, tw, ow, floor)
    odd = sorted({x * SPATIAL for x in rh + cw if x % 2})
    if odd:
        raise RuntimeError(f"{TAG}: a tile side of {odd} px is not a multiple "
                           "of 32; H3 patchifies 2x2 latent cells.")
    if int(factor) > 1:
        # The guide sits on every factor-th PATCH of the tile, and a tile's
        # guide window is cut from the film's guide at (r0, c0) / factor, so
        # every edge and origin must land on a whole guide patch.
        step = 2 * int(factor)
        off = sorted({x * SPATIAL for x in r0s + rh + c0s + cw if x % step})
        if off:
            raise RuntimeError(f"{TAG}: tile edges {off} px are off the "
                               f"{step * SPATIAL} px guide grid")
    tiles = []
    nr, nc = len(r0s), len(c0s)
    for i in range(nr):
        for j in range(nc):
            tiles.append((r0s[i], c0s[j], rh[i], cw[j],
                          rseam[i] or oh, cseam[j] or ow,
                          i > 0, j > 0, i < nr - 1, j < nc - 1))
    return tiles, got


def feather(rows, cols, seam_top, seam_left, top, left, bottom, right, device):
    """A tile's weight when tiles are merged: 1 inside, a raised cosine down to
    0 across any side that has a neighbour. Raised cosine, not linear, so the
    weight arrives at 1 without a kink that would print as a faint line."""
    def ramp(n):
        i = torch.arange(n, device=device, dtype=torch.float32)
        return 0.5 * (1.0 - torch.cos(math.pi * i / n))

    m = torch.ones((rows, cols), device=device, dtype=torch.float32)
    if left and seam_left > 0:
        m[:, :seam_left] *= ramp(min(seam_left, cols))[None, :]
    if right and seam_left > 0:
        m[:, -seam_left:] *= torch.flip(ramp(min(seam_left, cols)), (0,))[None, :]
    if top and seam_top > 0:
        m[:seam_top, :] *= ramp(min(seam_top, rows))[:, None]
    if bottom and seam_top > 0:
        m[-seam_top:, :] *= torch.flip(ramp(min(seam_top, rows)), (0,))[:, None]
    return m


# --- conditioning -----------------------------------------------------------

def _resize_planes(lt, h, w, dtype=None):
    """[B, C, T, H, W] -> [B, C, T, h, w], every H x W plane bilinear."""
    b, c, t, hh, ww = lt.shape
    x = lt if dtype is None else lt.to(dtype)
    x = F.interpolate(x.reshape(b * c * t, 1, hh, ww), size=(h, w),
                      mode="bilinear", align_corners=False)
    return x.reshape(b, c, t, h, w)


def clean_refs(cond):
    """Reference blocks whose row counts come from their real latents.

    H3 reserves reference rows from a block's metadata but fills them from the
    block's latent. A block whose metadata disagrees, or a picture block with
    no latent, would reserve rows nothing fills. Rewrite the metadata from the
    latent and drop picture blocks without one.
    """
    out = []
    for emb, d in cond:
        d = dict(d)
        if d.get("minimax_refs"):
            kept = []
            for blk in d["minimax_refs"]:
                blk = dict(blk)
                lt = blk.get("latent")
                if lt is None:
                    if blk.get("kind") not in ("image", "video", "video_audio"):
                        kept.append(blk)
                    continue
                blk["latent_h"], blk["latent_w"] = int(lt.shape[3]), int(lt.shape[4])
                if blk.get("kind") in ("video", "video_audio"):
                    blk["latent_t"] = int(lt.shape[2])
                kept.append(blk)
            d["minimax_refs"] = kept
        out.append([emb, d])
    return out


def _keyframe_in(kf, f0, f1):
    """The part of keyframe `kf` lying wholly inside frames [f0, f1), re-based
    to f0, or None."""
    idx = kf["resolved_frame_index"]
    lt, al = kf.get("latent"), kf.get("audio_latent")
    if lt is None and al is None:
        return {"resolved_frame_index": idx - f0} if f0 <= idx < f1 else None
    out = {}
    if lt is not None:
        first = last = None
        pos = idx
        for k in range(lt.shape[2]):
            span = FRAME_PER_TOKEN[k % 5]
            if f0 <= pos and pos + span <= f1:
                first = k if first is None else first
                last = k + 1
            pos += span
        if first is None:
            return None
        out["latent"] = lt[:, :, first:last].contiguous()
        out["resolved_frame_index"] = idx + frames_in(first) - f0
    if al is not None:
        a0 = max(0, math.ceil((f0 - idx) * FRAME_RESCALE))
        a1 = min(al.shape[-1], math.floor((f1 - idx) / FRAME_RESCALE))
        if a1 > a0:
            out["audio_latent"] = al[..., a0:a1].contiguous()
            out.setdefault("resolved_frame_index", max(0, idx - f0))
    return out if ("latent" in out or "audio_latent" in out) else None


def cond_for_chunk(cond, f0, f1, h, w):
    """The keyframes that fall inside the chunk, re-based to it, their video
    resized to the chunk's (upscaled) grid."""
    out = []
    for emb, d in cond:
        d = dict(d)
        if d.get("minimax_keyframes"):
            kept = [k for k in (_keyframe_in(kf, f0, f1)
                                for kf in d["minimax_keyframes"]) if k is not None]
            for kf in kept:
                lt = kf.get("latent")
                if lt is not None and (lt.shape[3] != h or lt.shape[4] != w):
                    kf["latent"] = _resize_planes(lt, h, w)
            if kept:
                d["minimax_keyframes"] = kept
            else:
                d.pop("minimax_keyframes", None)
        out.append([emb, d])
    return out


def pin_chunk_start(cond, done_video, f0, aug=ANCHOR_AUG):
    """Frame 0 of this chunk pinned to what the previous chunk re-drew there."""
    k = tokens_to_reach(f0)
    if k >= done_video.shape[2]:
        raise RuntimeError(f"{TAG}: the finished chunks stop before frame {f0}")
    pin = {"resolved_frame_index": 0,
           "latent": done_video[:, :, k:k + 1].contiguous()}
    out = []
    for emb, d in cond:
        d = dict(d)
        rest = [kf for kf in (d.get("minimax_keyframes") or [])
                if kf.get("resolved_frame_index") != 0 or "latent" not in kf]
        d["minimax_keyframes"] = [pin] + rest
        d["minimax_visual_cond_noise_aug"] = max(0.0, min(1.0, float(aug)))
        out.append([emb, d])
    return out


def cond_for_tile(cond, h, w, r0, c0, rows, cols):
    """Keyframe video cropped to one tile. A keyframe on another grid is
    resized to the chunk's first, so the crop has the tile's row count."""
    out = []
    for emb, d in cond:
        d = dict(d)
        if d.get("minimax_keyframes"):
            kfs = []
            for kf in d["minimax_keyframes"]:
                kf = dict(kf)
                lt = kf.get("latent")
                f = int(kf.get(STRIDE_KEY, 1))
                if lt is not None and f > 1:
                    # The guide is on its own 1/f grid: cut the tile's window
                    # there, never resized up to the chunk's grid.
                    kf["latent"] = lt[:, :, :, r0 // f:(r0 + rows) // f,
                                      c0 // f:(c0 + cols) // f].contiguous()
                elif lt is not None:
                    if lt.shape[3] != h or lt.shape[4] != w:
                        lt = _resize_planes(lt, h, w, torch.float32)
                    kf["latent"] = lt[:, :, :, r0:r0 + rows, c0:c0 + cols].contiguous()
                kfs.append(kf)
            d["minimax_keyframes"] = kfs
        if any(b.get(TILE_CROP_KEY) for b in (d.get("minimax_refs") or ())):
            d["minimax_refs"] = [_crop_ref(b, h, w, r0, c0, rows, cols)
                                 if b.get(TILE_CROP_KEY) else b for b in d["minimax_refs"]]
        out.append([emb, d])
    return out


# A reference block marked with this key shows the same frames as the clip
# being re-drawn (IC's <Video 1> draft): a tile gets the matching window of it,
# so each tile sees the draft of what it re-draws, as IC was trained.
TILE_CROP_KEY = "htc_tile_crop"


def _crop_ref(blk, h, w, r0, c0, rows, cols):
    """The tile's window of a whole-frame reference video, on the reference's
    own grid (proportional, snapped outward to even rows/cols for the 2x2
    patch)."""
    lt = blk.get("latent")
    if lt is None:
        return blk
    rh, rw = int(lt.shape[3]), int(lt.shape[4])

    def span(a, n, full, ref):
        lo = int(math.floor(a * ref / float(full) / 2.0)) * 2
        hi = int(math.ceil((a + n) * ref / float(full) / 2.0)) * 2
        return max(0, lo), min(ref, max(hi, lo + 2))
    y0, y1 = span(r0, rows, h, rh)
    x0, x1 = span(c0, cols, w, rw)
    if (y0, y1, x0, x1) == (0, rh, 0, rw):
        return blk
    blk = dict(blk)
    blk["latent"] = lt[:, :, :, y0:y1, x0:x1].contiguous()
    blk["latent_h"], blk["latent_w"] = y1 - y0, x1 - x0
    return blk


def encode_steps(encode, frames, steps):
    """`steps` video latent steps from the frames that cover them.

    H3's VAE encodes 17-frame clips to 5 steps each, then drops the last 3
    steps -- right for a whole clip (17g+5 frames, 5g+2 steps), short for a
    chunk cut out of a film: 102 frames come back as 27 steps, not 30. Each
    17-frame clip encodes on its own, so padding the frames to the next whole
    clip length (repeating the last frame, as the encoder itself pads) and
    keeping the first `steps` steps gives exactly what those frames encode to.
    """
    n = frames_in(steps)
    if int(frames.shape[0]) < n:
        raise RuntimeError(f"{TAG}: {int(frames.shape[0])} frames for {steps} latent "
                           f"steps, which cover {n}")
    frames = frames[:n]
    full = 5 if n <= 5 else 17 * math.ceil((n - 5) / 17) + 5
    if full > n:
        frames = torch.cat([frames, frames[-1:].expand(full - n, -1, -1, -1)], dim=0)
    lat = encode(frames)
    if int(lat.shape[2]) < steps:
        raise RuntimeError(f"{TAG}: {full} frames encoded to {int(lat.shape[2])} "
                           f"latent steps, needed {steps}")
    return lat[:, :, :steps]


def add_guide(cond, guide_latent, factor):
    """The h3upscale guide appended as a strided keyframe at frame 0. After
    `pin_chunk_start`, which drops every other latent keyframe at frame 0."""
    kf = {"latent": guide_latent, "resolved_frame_index": 0, STRIDE_KEY: int(factor)}
    out = []
    for emb, d in cond:
        d = dict(d)
        d["minimax_keyframes"] = list(d.get("minimax_keyframes") or []) + [kf]
        out.append([emb, d])
    return out


# --- the upscaler network, in the layout its weights were trained in --------

def _gn(c):
    return nn.GroupNorm(32, c)


class _Res(nn.Module):
    def __init__(self, c, emb):
        super().__init__()
        self.in_layers = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv3d(c, c, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb, 2 * c))
        self.out_norm = _gn(c)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(0.0),
                                        nn.Conv3d(c, c, 3, padding=1))

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb).to(h.dtype)[..., None, None, None] \
            .chunk(2, dim=1)
        return x + self.out_layers(self.out_norm(h) * (1 + scale) + shift)


class _Temporal(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.norm = _gn(c)
        self.dwconv = nn.Conv3d(c, c, (k, 1, 1), padding=(k // 2, 0, 0), groups=c)
        self.pwconv = nn.Conv3d(c, c, 1)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class LatentUpscaler(nn.Module):
    def __init__(self, sd):
        super().__init__()
        c = int(sd["conv_in.weight"].shape[0])
        cin = int(sd["conv_in.weight"].shape[1])
        emb = int(sd["embed.0.weight"].shape[0])
        self.conv_in = nn.Conv3d(cin, c, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.in_blocks = self._blocks(sd, "in_blocks", c, emb)
        self.out_blocks = self._blocks(sd, "out_blocks", c, emb)
        self.norm_out = _gn(c)
        self.conv_out = nn.Conv3d(c, cin, 3, padding=1)

    @staticmethod
    def _blocks(sd, name, c, emb):
        # Which slot is a residual block and which a temporal conv is read
        # from the keys, so a file with another rhythm refuses rather than
        # loading wrong.
        slots = {}
        for k in sd:
            if k.startswith(name + "."):
                i = int(k.split(".")[1])
                if k.endswith(".dwconv.weight"):
                    slots[i] = _Temporal(c, int(sd[k].shape[2]))
                elif ".in_layers." in k:
                    slots.setdefault(i, None)
        for i in [i for i, m in slots.items() if m is None]:
            slots[i] = _Res(c, emb)
        if sorted(slots) != list(range(len(slots))):
            raise RuntimeError(f"{TAG}: {name} has gaps in its block list; "
                               "not an upscaler file this node reads.")
        return nn.ModuleList(slots[i] for i in range(len(slots)))

    def forward(self, x, size, scale):
        emb = self.embed(torch.tensor([[float(scale) - 1.0]],
                                      device=x.device, dtype=x.dtype))
        emb = emb.expand(x.shape[0], -1)
        x = self.conv_in(x)
        for b in self.in_blocks:
            x = b(x, emb) if isinstance(b, _Res) else b(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for b in self.out_blocks:
            x = b(x, emb) if isinstance(b, _Res) else b(x)
        return self.conv_out(F.silu(self.norm_out(x)))


# --- the 2D upscaler (LBH-123-AI 2d_conv_v1, MIT), in its trained layout -----
#
# Every frame runs through 2D residual blocks; after every other block a
# temporal block (one shared by all the in-blocks, one by all the out-blocks)
# mixes neighbouring frames. Between the halves the features are resized
# bilinearly. The scale is told per axis, and a sharpness tier (1..6) rides
# the same embedding. Clips longer than CHUNK_2D latent frames run in pieces
# with the blocks' whole temporal reach (24 latent frames for 2d_conv_v1) of
# context each side, exactly as the reference node does (temporal_chunking
# "chunked", 32 frames).

CHUNK_2D = 32               # latent frames per forward (the reference node's default)
TEMPORAL_EVERY_2D = 2


class _Res2D(nn.Module):
    def __init__(self, c, emb):
        super().__init__()
        self.in_layers = nn.Sequential(_gn(c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb, 2 * c))
        self.out_norm = _gn(c)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(0.0),
                                        nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb).to(h.dtype)[..., None, None].chunk(2, dim=1)
        return x + self.out_layers(self.out_norm(h) * (1 + scale) + shift)


class _Temporal2D(nn.Module):
    """GroupNorm per frame (not over the clip), then a depthwise conv over time."""

    def __init__(self, c, k):
        super().__init__()
        self.norm = _gn(c)
        self.dwconv = nn.Conv3d(c, c, (k, 1, 1), padding=(k // 2, 0, 0), groups=c)
        self.pwconv = nn.Conv3d(c, c, 1)

    def forward(self, x):
        b, c, t, h, w = x.shape
        n = self.norm(x.transpose(1, 2).reshape(b * t, c, h, w))
        n = n.reshape(b, t, c, h, w).transpose(1, 2)
        return x + self.pwconv(self.dwconv(F.silu(n)))


class _Embed2D(nn.Module):
    """Fourier code of scale_h and scale_w (and of tier - 1) -> an MLP."""

    def __init__(self, in_dim, emb, freqs):
        super().__init__()
        self.freqs = int(freqs)
        self.tiers = in_dim > 4 * self.freqs
        self.linear = nn.Sequential(nn.Linear(in_dim, emb), nn.SiLU(), nn.Linear(emb, emb))

    def forward(self, scale_h, scale_w, tier):
        dev = self.linear[0].weight.device
        f = 2.0 ** torch.arange(self.freqs, device=dev).float()
        parts = []
        for v in (float(scale_h), float(scale_w)):
            a = torch.tensor([v], device=dev, dtype=torch.float32)[:, None] * f[None]
            parts += [torch.sin(a), torch.cos(a)]
        if self.tiers:
            a = (torch.tensor([float(tier)], device=dev, dtype=torch.float32) - 1.0)[:, None] * f[None]
            parts += [torch.sin(a), torch.cos(a)]
        return self.linear(torch.cat(parts, dim=1).to(self.linear[0].weight.dtype))


class _Resizer2D(nn.Module):
    def __init__(self, sd, c, emb):
        super().__init__()
        cin = int(sd["resizer.conv_in.weight"].shape[1])
        e_in = int(sd["resizer.embed.linear.0.weight"].shape[1])
        self.conv_in = nn.Conv2d(cin, c, 3, padding=1)
        self.embed = _Embed2D(e_in, emb, 16 if e_in == 96 else e_in // 4)
        n_in = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("resizer.in_blocks."))
        n_out = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("resizer.out_blocks."))
        self.in_blocks = nn.ModuleList(_Res2D(c, emb) for _ in range(n_in))
        self.out_blocks = nn.ModuleList(_Res2D(c, emb) for _ in range(n_out))
        self.norm_out = _gn(c)
        self.conv_out = nn.Conv2d(c, cin, 3, padding=1)


class LatentUpscaler2D(nn.Module):
    IS_2D = True

    def __init__(self, sd):
        super().__init__()
        # Only the layout 2d_conv_v1 ships is built here; the reference code's
        # optional parts (coord channels, SE, large kernels, learned
        # PixelShuffle, multi-scale temporal) refuse rather than load wrong.
        extra = sorted({p for k in sd for p in (".se.", ".dw.", "post_up.", "learned_up.",
                                                 ".branches.") if p in k})
        cin = int(sd["resizer.conv_in.weight"].shape[1])
        if extra or cin != int(sd["resizer.conv_out.weight"].shape[0]):
            raise RuntimeError(f"{TAG}: this 2D upscaler file has parts this node does not "
                               f"build ({', '.join(extra) or 'coordinate channels'}).")
        c = int(sd["resizer.conv_in.weight"].shape[0])
        emb = int(sd["resizer.embed.linear.0.weight"].shape[0])
        self.resizer = _Resizer2D(sd, c, emb)
        n_t = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("temporal_blocks."))
        k = int(sd["temporal_blocks.0.dwconv.weight"].shape[2])
        self.temporal_blocks = nn.ModuleList(_Temporal2D(c, k) for _ in range(n_t))
        self.tiers = self.resizer.embed.tiers
        # context frames each side of a piece: the temporal reach of every block
        n_temporal = sum(1 for i in range(len(self.resizer.in_blocks))
                         if i % TEMPORAL_EVERY_2D == 0)
        n_temporal += sum(1 for i in range(len(self.resizer.out_blocks))
                          if i % TEMPORAL_EVERY_2D == 0)
        self.context = max(k // 2, 1) * max(n_temporal, 1)

    def _through(self, blocks, tb, x, emb, b, t):
        for i, blk in enumerate(blocks):
            x = blk(x, emb)
            if i % TEMPORAL_EVERY_2D == 0 and tb is not None:
                v = tb(x.reshape(b, t, *x.shape[1:]).transpose(1, 2))
                x = v.transpose(1, 2).reshape(b * t, *x.shape[1:])
        return x

    def _full(self, x, hw, tier):
        b, c, t, h, w = x.shape
        r = self.resizer
        emb = r.embed(hw[0] / h, hw[1] / w, tier).expand(b * t, -1)
        y = r.conv_in(x.transpose(1, 2).reshape(b * t, c, h, w))
        tb = self.temporal_blocks
        y = self._through(r.in_blocks, tb[0] if len(tb) > 0 else None, y, emb, b, t)
        y = F.interpolate(y, size=hw, mode="bilinear")
        y = self._through(r.out_blocks, tb[1] if len(tb) > 1 else None, y, emb, b, t)
        y = r.conv_out(F.silu(r.norm_out(y)))
        return y.reshape(b, t, c, *hw).transpose(1, 2)

    def forward(self, x, size, scale=None, tier=SHARPNESS):
        """x [B, 24, T, H, W] normalised -> [B, 24, T, size[1], size[2]]."""
        t = int(x.shape[2])
        hw = (int(size[1]), int(size[2]))
        tier = int(tier) if self.tiers else None
        if t <= CHUNK_2D or not len(self.temporal_blocks):
            return self._full(x, hw, tier)
        out = x.new_zeros((x.shape[0], x.shape[1], t) + hw)
        for s0 in range(0, t, CHUNK_2D):
            s1 = min(t, s0 + CHUNK_2D)
            lo, hi = max(0, s0 - self.context), min(t, s1 + self.context)
            out[:, :, s0:s1] = self._full(x[:, :, lo:hi].contiguous(), hw,
                                          tier)[:, :, s0 - lo:s1 - lo]
        return out


def _call_net(net, x, size, scale, sharpness):
    """One forward of either upscaler; only the 2D one takes the sharpness tier."""
    if getattr(net, "IS_2D", False):
        return net(x, size, scale, sharpness)
    return net(x, size, scale)


def clamp_sharpness(v):
    try:
        return max(1, min(SHARPNESS_MAX, int(v)))
    except (TypeError, ValueError):
        return SHARPNESS


_NET = {}
_TIER_NOTE = set()


def _note_sharpness(net, name, sharpness):
    """Say once per file which tier runs, or that this file has none."""
    key = (str(name), int(sharpness))
    if key in _TIER_NOTE:
        return
    _TIER_NOTE.add(key)
    if getattr(net, "IS_2D", False) and getattr(net, "tiers", False):
        print(f"[{TAG}] 2D upscaler sharpness {int(sharpness)}/{SHARPNESS_MAX} "
              "(1 = most faithful, 6 = sharpest)", flush=True)
    elif int(sharpness) != SHARPNESS:
        print(f"[{TAG}] sharpness {int(sharpness)} ignored: {name} has no sharpness "
              f"tiers (only the 2D upscaler {UPSCALER_2D_FILE} does)", flush=True)


def load_upscaler(name):
    """The network for `name`, kept on the CPU in fp16 between chunks."""
    if name in _NET:
        return _NET[name]
    from safetensors.torch import load_file
    if not str(name).endswith(".safetensors"):
        raise RuntimeError(f"{TAG}: {name!r}: only .safetensors files are "
                           "read (no pickle).")
    found, path = resolve_upscaler(name)
    sd = load_file(path, device="cpu")
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items()
              if k.startswith("upscaler.")}
    if "resizer.conv_in.weight" in sd and "resizer.embed.linear.0.weight" in sd:
        if int(sd["resizer.conv_in.weight"].shape[1]) != 24:
            raise RuntimeError(f"{TAG}: {name!r} does not take 24-channel H3 latents.")
        net, kind = LatentUpscaler2D(sd), "2D"
    else:
        if "conv_in.weight" not in sd or "embed.0.weight" not in sd:
            raise RuntimeError(f"{TAG}: {name!r} is not an H3 latent upscaler.")
        if int(sd["conv_in.weight"].shape[1]) != 24:
            raise RuntimeError(f"{TAG}: {name!r} does not take 24-channel H3 latents.")
        net, kind = LatentUpscaler(sd), "3D"
    net.load_state_dict({k: (v.to(torch.float16) if v.dtype == torch.float8_e4m3fn
                             else v) for k, v in sd.items()}, strict=True)
    net = net.eval().requires_grad_(False).to(UPSCALER_DTYPE)
    _NET.clear()
    _NET[name] = net
    alias = f" (installed as {found})" if found != str(name) else ""
    tiers = (f", {SHARPNESS_MAX} sharpness tiers" if getattr(net, "tiers", False) else "")
    print(f"[{TAG}] upscaler {name}{alias}: {kind}, "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.0f}M params{tiers}", flush=True)
    return net


def _device():
    import comfy.model_management as mm
    return mm.get_torch_device()


def _free_all_gpu():
    """Unload any resident ComfyUI models (VAE, CLIP, DiT) and empty CUDA cache."""
    try:
        import gc
        import comfy.model_management as mm
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_oom(exc):
    """True when `exc` is a CUDA or ComfyUI out-of-memory error."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    try:
        import comfy.model_management as mm
        if isinstance(exc, getattr(mm, "OOM_EXCEPTION", ())):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg or "alloc" in msg


def upscale_chunk(video, name, w_px, h_px, sharpness=SHARPNESS):
    """One chunk's video latent resized to (w_px, h_px) by the upscaler.
    `sharpness` is the 2D upscaler's tier (the 3D one has none).

    The network is told the scale actually achieved (mean of the two axes),
    and runs in fp16 end to end, normalisation included, as in the recipe.
    Unloads resident ComfyUI models before moving the 512-channel 3D Conv
    upscaler to CUDA, and automatically falls back to CPU (float32) if CUDA
    VRAM is exhausted so the latent upscale stage can never OOM.
    """
    _, _, t, h_in, w_in = video.shape
    scale = (w_px / (w_in * SPATIAL) + h_px / (h_in * SPATIAL)) / 2.0
    h_out = max(1, int(round(h_px / SPATIAL)))
    w_out = max(1, int(round(w_px / SPATIAL)))
    if scale < 1.0 and (w_out < w_in or h_out < h_in):
        raise RuntimeError(f"{TAG}: {w_px}x{h_px} is smaller than the clip "
                           f"({w_in * SPATIAL}x{h_in * SPATIAL}); this only "
                           "upscales. Raise megapixels.")
    if (h_out, w_out) == (h_in, w_in):
        return video
    dev = _device()
    if dev.type == "cuda":
        _free_all_gpu()
    net = load_upscaler(name)
    sharpness = clamp_sharpness(sharpness)
    _note_sharpness(net, name, sharpness)
    try:
        net = net.to(device=dev, dtype=UPSCALER_DTYPE)
        mean = torch.tensor(LATENT_MEAN, device=dev, dtype=UPSCALER_DTYPE).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENT_STD, device=dev, dtype=UPSCALER_DTYPE).view(1, -1, 1, 1, 1)
        with torch.inference_mode():
            x = (video.to(device=dev, dtype=UPSCALER_DTYPE, copy=True) - mean) / std
            out = _call_net(net, x, (t, h_out, w_out), scale, sharpness)
            del x
            out = out * std + mean
        return out.to(device="cpu", dtype=video.dtype)
    except Exception as exc:
        if dev.type != "cuda" or not _is_oom(exc):
            raise
        print(f"[{TAG}] CUDA VRAM tight during the latent upscale ({exc}); "
              "sharing load with CPU...", flush=True)
        try:
            net.to("cpu")
            torch.cuda.empty_cache()
        except Exception:
            pass
        cpu_dev = torch.device("cpu")
        net = net.to(device=cpu_dev, dtype=torch.float32)
        mean = torch.tensor(LATENT_MEAN, device=cpu_dev, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENT_STD, device=cpu_dev, dtype=torch.float32).view(1, -1, 1, 1, 1)
        with torch.inference_mode():
            x = (video.to(device=cpu_dev, dtype=torch.float32, copy=True) - mean) / std
            out = _call_net(net, x, (t, h_out, w_out), scale, sharpness)
            del x
            out = out * std + mean
        net.to(dtype=UPSCALER_DTYPE)
        return out.to(device="cpu", dtype=video.dtype)
    finally:
        net.to("cpu")
        if dev.type == "cuda":
            torch.cuda.empty_cache()


def upscale_chunks_batch(videos, name, w_px, h_px, sharpness=SHARPNESS):
    """Upscale a list of chunk video latents `[cv_0, cv_1, ...]` in one pass so
    the upscaler (3D 345M or 2D 182M params) is loaded onto GPU once instead of
    21 times across a 10-hop chain."""
    if not videos:
        return []
    _, _, _t0, h_in, w_in = videos[0].shape
    scale = (w_px / (w_in * SPATIAL) + h_px / (h_in * SPATIAL)) / 2.0
    h_out = max(1, int(round(h_px / SPATIAL)))
    w_out = max(1, int(round(w_px / SPATIAL)))
    if scale < 1.0 and (w_out < w_in or h_out < h_in):
        raise RuntimeError(f"{TAG}: {w_px}x{h_px} is smaller than the clip "
                           f"({w_in * SPATIAL}x{h_in * SPATIAL}); this only "
                           "upscales. Raise megapixels.")
    if (h_out, w_out) == (h_in, w_in):
        return list(videos)
    # If a test monkeypatches `upscale_chunk`, honour the monkeypatch per chunk.
    if upscale_chunk is not _DEFAULT_UPSCALE_CHUNK:
        return [upscale_chunk(v, name, w_px, h_px) for v in videos]
    dev = _device()
    if dev.type == "cuda":
        _free_all_gpu()
    net = load_upscaler(name)
    sharpness = clamp_sharpness(sharpness)
    _note_sharpness(net, name, sharpness)
    out_list = []
    cpu = torch.device("cpu")

    def run(v, d, dtype):
        mean = torch.tensor(LATENT_MEAN, device=d, dtype=dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENT_STD, device=d, dtype=dtype).view(1, -1, 1, 1, 1)
        with torch.inference_mode():
            x = (v.to(device=d, dtype=dtype, copy=True) - mean) / std
            o = _call_net(net, x, (int(v.shape[2]), h_out, w_out), scale, sharpness)
            del x
            return (o * std + mean).to(device="cpu", dtype=v.dtype)

    try:
        on_gpu = dev.type == "cuda"
        # A CPU-only device runs `run(v, cpu, float32)` below: weights to match.
        net = net.to(device=dev, dtype=UPSCALER_DTYPE if on_gpu else torch.float32)
        for i, v in enumerate(videos):
            if not on_gpu and dev.type == "cuda":
                # The previous chunk fell back to the CPU; give the GPU another try
                # rather than running every remaining chunk at CPU speed.
                try:
                    net = net.to(device=dev, dtype=UPSCALER_DTYPE)
                    on_gpu = True
                except Exception as exc:  # noqa: BLE001
                    if not _is_oom(exc):
                        raise
                    net = net.to(device=cpu, dtype=torch.float32)
            if not on_gpu:
                out_list.append(run(v, cpu, torch.float32))
                continue
            try:
                out_list.append(run(v, dev, UPSCALER_DTYPE))
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                print(f"[{TAG}] CUDA VRAM tight during the latent upscale of chunk {i + 1}/"
                      f"{len(videos)} ({exc}); this chunk on the CPU, the next one back on "
                      "the GPU", flush=True)
                net = net.to(device=cpu, dtype=torch.float32)
                torch.cuda.empty_cache()
                on_gpu = False
                out_list.append(run(v, cpu, torch.float32))
    finally:
        net.to(device="cpu", dtype=UPSCALER_DTYPE)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out_list


_DEFAULT_UPSCALE_CHUNK = upscale_chunk


# --- the upscaler through ComfyUI core, when it has it ----------------------
#
# Core (Kijai, ComfyUI PR #17412) runs the same network with its GroupNorm
# statistics taken over the whole clip while working 16 latent frames at a
# time. So the film is upscaled ONCE and every chunk is cut from it: no chunk
# starts the re-draw toned a little differently from its neighbour, which is
# what upscaling chunk by chunk does (each chunk normalised over its own span).
# Without it (older ComfyUI), or when the whole film does not fit, the chunks
# are upscaled one by one as before.

def _core_upscaler(name):
    """Core's model patcher for `name`, loaded by core's own Load Latent Upscale
    Model, or None when this ComfyUI has no H3 upscaler (or the file isn't one)."""
    key = "core:" + str(name)
    if key in _NET:
        return _NET[key]
    try:
        from comfy.ldm.minimax.latent_upscaler import MiniMaxH3LatentUpscaler
        from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
    except Exception:  # noqa: BLE001 -- ComfyUI before Oct 2026
        return None
    try:
        patcher = LatentUpscaleModelLoader.execute(name).args[0]
    except Exception as exc:  # noqa: BLE001 -- e.g. an `upscaler.`-prefixed file
        print(f"[{TAG}] core could not load {name!r} ({exc}); the bundled upscaler "
              "instead", flush=True)
        return None
    if not isinstance(getattr(patcher, "model", None), MiniMaxH3LatentUpscaler):
        return None
    _NET.clear()
    _NET[key] = patcher
    print(f"[{TAG}] upscaler {name}: ComfyUI core's (whole-clip statistics)", flush=True)
    return patcher


def _core_run(patcher, video, size):
    import comfy.model_management as mm
    mm.load_models_gpu([patcher], memory_required=patcher.model.memory_required(video.shape, size))
    with torch.inference_mode():
        out = patcher.model(video.to(device=patcher.load_device, dtype=patcher.model_dtype()),
                            size)
    return out.to(device="cpu", dtype=video.dtype)


def upscale_film(video, plan, name, w_px, h_px, sharpness=SHARPNESS):
    """Every chunk of `plan` upscaled, as `upscale_chunks_batch` returns them.

    Through core when it can: the whole film in one pass when that fits the
    device, otherwise chunk by chunk with core's faster kernels. Anything else
    (no core, a monkeypatched `upscale_chunk`, out of memory) is the bundled
    network chunk by chunk.

    The 2D upscaler (core has none) runs the whole film in one call -- its own
    32-frame pieces with 24 frames of context keep memory flat -- and every
    chunk is cut from it, so no chunk edge cuts its temporal blocks short.
    """
    raw = [video[:, :, k0:k1].contiguous() for (k0, _f0, k1, _f1) in plan]
    if upscale_chunk is not _DEFAULT_UPSCALE_CHUNK or not raw:
        return upscale_chunks_batch(raw, name, w_px, h_px)
    _, _, tv, h_in, w_in = video.shape
    h_out = max(1, int(round(h_px / SPATIAL)))
    w_out = max(1, int(round(w_px / SPATIAL)))
    if (h_out, w_out) == (h_in, w_in) or h_out < h_in or w_out < w_in:
        return upscale_chunks_batch(raw, name, w_px, h_px)  # its no-op / its refusal
    if is_2d_file(name):
        film = upscale_chunks_batch([video], name, w_px, h_px, sharpness)[0]
        out = [film[:, :, k0:k1].contiguous() for (k0, _f0, k1, _f1) in plan]
        del film
        print(f"[{TAG}] 2D upscale: the whole film in one call ({int(tv)} latent frames), "
              f"{len(plan)} chunk(s) cut from it", flush=True)
        return out
    try:
        name = resolve_upscaler(name)[0]   # core loads by the installed name
    except RuntimeError:
        pass                               # load_upscaler says what is missing
    dev = _device()
    if dev.type == "cuda":
        _free_all_gpu()
    patcher = _core_upscaler(name)
    if patcher is None:
        return upscale_chunks_batch(raw, name, w_px, h_px)
    import comfy.model_management as mm
    size = (int(tv), h_out, w_out)
    need = patcher.model.memory_required(video.shape, size) + patcher.model_size()
    whole = len(plan) == 1 or need <= mm.get_free_memory(dev) * MEMORY_HEADROOM
    try:
        if whole:
            film = _core_run(patcher, video, size)
            out = [film[:, :, k0:k1].contiguous() for (k0, _f0, k1, _f1) in plan]
            del film
            if len(plan) > 1:
                print(f"[{TAG}] 3D upscale: the whole film in one pass "
                      f"({need / 1073741824:.1f} GB), {len(plan)} chunks cut from it", flush=True)
        else:
            print(f"[{TAG}] 3D upscale: the whole film needs {need / 1073741824:.1f} GB, "
                  f"more than is free; chunk by chunk", flush=True)
            out = [_core_run(patcher, v, (int(v.shape[2]), h_out, w_out)) for v in raw]
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise
        print(f"[{TAG}] 3D upscale through core ran out of memory ({exc}); "
              "the bundled upscaler chunk by chunk instead", flush=True)
        out = None
    finally:
        _free_all_gpu()
    return out if out is not None else upscale_chunks_batch(raw, name, w_px, h_px)


# --- sampling ---------------------------------------------------------------

def recipe_sigmas(model, denoise=DENOISE, steps=STEPS, start_sigma=0.0):
    """One `simple` step at `denoise`, exactly as Core's BasicScheduler.

    The recipe's 0.2 is a fraction of the SCHEDULE, not of the noise: under
    H3's video shift (12) the step starts at sigma 0.75. Below it the re-draw
    repaints less; at 0 there is no re-draw and this returns None.

    `start_sigma` > 0 (0.2.0) replaces `denoise`: the `simple` schedule over
    the model's (shifted) sigmas, cut where it drops to `start_sigma`, then
    `steps` steps spread evenly down that tail to 0. `steps` 1 with
    `start_sigma` 0 is the 0.1.0 recipe exactly.
    """
    import comfy.samplers
    steps = max(1, int(steps))
    ms = model.get_model_object("model_sampling")
    if float(start_sigma) > 0.0:
        full = comfy.samplers.calculate_sigmas(ms, SCHEDULER, 1000).cpu().float()
        tail = full[full <= float(start_sigma) + 1e-6]
        if tail.numel() < 2:
            return None
        idx = torch.linspace(0, tail.numel() - 1, steps + 1).round().long()
        idx = torch.unique_consecutive(idx)
        sig = tail[idx]
        sig[-1] = 0.0
        return sig
    if float(denoise) <= 0.0:
        return None
    total = int(steps / float(denoise))
    sig = comfy.samplers.calculate_sigmas(ms, SCHEDULER, total).cpu()
    return sig[-(steps + 1):]


# --- the memory plan: tile_tokens = 0 learns this machine ---------------------
#
# A fixed token budget is a guess about someone else's card. With tile_tokens 0
# the upsampler starts from what it learned on THIS GPU last time (70000 the
# first time), measures every chunk -- activation memory = peak minus what is
# still resident after the chunk (mostly weights, which ComfyUI's dynamic
# loading grows into any spare VRAM, so the raw peak says little) -- and stores
# the budget the next run should start at: grown, at most 2x a run, toward
# 85% of the card when there was room; after an OOM, the budget that worked,
# and never again up to the one that failed. Keyed by GPU, its memory, the
# attention mode and the low-VRAM chunking, which all change the cost of a token.

MEMORY_FILE = "htc_upsampler_memory.json"
MEMORY_HEADROOM = 0.85
MEMORY_GROWTH = 2.0


def memory_key(attention="dense", attn_chunks=4, ffn_chunks=2, factor=1):
    """This machine's key for the memory plan, or None off CUDA."""
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(_device())
    except Exception:  # noqa: BLE001
        return None
    gb = round(props.total_memory / 2 ** 30)
    return f"{props.name}|{gb}GB|{attention}|a{int(attn_chunks)}f{int(ffn_chunks)}|g{int(factor)}"


def _memory_path():
    import folder_paths
    try:
        d = folder_paths.get_user_directory()
    except Exception:  # noqa: BLE001
        d = folder_paths.get_temp_directory()
    return os.path.join(d, MEMORY_FILE)


def _memory_load():
    import json
    try:
        with open(_memory_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def resolve_budget(budget, key):
    """`budget` as given, or (0) what this machine learned. -> (budget, auto, note)."""
    if int(budget or 0) > 0:
        return int(budget), False, f"{int(budget)} (fixed)"
    entry = _memory_load().get(key) if key else None
    if entry and int(entry.get("budget", 0)) > 0:
        return int(entry["budget"]), True, (f"{int(entry['budget'])} (auto, learned over "
                                            f"{int(entry.get('runs', 1))} run(s))")
    return TOKEN_BUDGET, True, f"{TOKEN_BUDGET} (auto, first run on this machine)"


def next_budget(used, final, samples, total_bytes, oom_at=None):
    """Where the next run starts. -> (budget, oom_at).

    `used` is the budget this run started at, `final` where it finished (lower
    after an OOM retry). `samples` are (tokens per forward, activation bytes,
    resident bytes) of chunks that ran without an OOM."""
    used, final = int(used), int(final)
    if final < used:                        # ran out: start where it worked
        oom_at = min(int(oom_at), used) if oom_at else used
        return max(1024, final // 1024 * 1024), oom_at
    fit = None
    for tokens, act, resident in samples:
        if tokens <= 0 or act <= 0:
            continue
        room = MEMORY_HEADROOM * float(total_bytes) - float(resident)
        if room <= 0:
            continue
        f = int(tokens * room / float(act))
        fit = f if fit is None else min(fit, f)
    nb = used
    if fit is not None and fit > used:          # grow, never past 2x a run
        nb = max(used, min(fit, int(used * MEMORY_GROWTH)) // 1024 * 1024)
    if oom_at and nb > int(int(oom_at) * 0.9):  # and never back near a failure
        nb = max(1024, min(used, int(int(oom_at) * 0.9) // 1024 * 1024))
    return nb, oom_at


def record_budget(key, used, final, samples):
    """Store what this run learned for `key`. -> a note for the log."""
    import json
    import time as _time
    if not key:
        return ""
    data = _memory_load()
    entry = data.get(key) or {}
    total = torch.cuda.get_device_properties(_device()).total_memory
    nb, oom_at = next_budget(used, final, samples, total, entry.get("oom_at"))
    entry.update(budget=int(nb), oom_at=oom_at, runs=int(entry.get("runs", 0)) + 1,
                 updated=_time.strftime("%Y-%m-%d %H:%M:%S"))
    if samples:
        t, a, r = max(samples, key=lambda x: x[0])
        entry["last"] = {"tokens": int(t), "activation_mb": round(a / 2 ** 20),
                         "resident_mb": round(r / 2 ** 20)}
    data[key] = entry
    try:
        path = _memory_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
    except OSError as exc:
        return f"not stored ({exc!r})"
    return (f"next run starts at {nb}" + (" (ran out of memory at "
                                          f"{oom_at})" if oom_at else ""))


def _block_forward_takes_attention(model):
    """Whether this model's patched H3 block forwards accept core's `attention`.

    Core calls every replaced ("double_block") block with attention=... ,
    sparse attention's producer or None. ComfyUI-KJNodes before its
    2026-09-06 fix patches DiTBlock.forward without that argument, and the
    run ends in the first block with "minimax_block_lowmem_forward() got an
    unexpected keyword argument 'attention'". -> False when this model's
    block patches would raise that.
    """
    import inspect
    try:
        from comfy.ldm.minimax.model import DiTBlock
        if "attention" not in inspect.signature(DiTBlock.forward).parameters:
            return True         # this core never passes it: nothing to clash with
    except Exception:  # noqa: BLE001 -- no core module to compare against
        return True
    for name, obj in (getattr(model, "object_patches", None) or {}).items():
        parts = str(name).split(".")
        if (len(parts) < 4 or parts[-1] != "forward" or parts[-3] != "blocks"
                or not parts[-2].isdigit()):
            continue
        try:
            if "attention" not in inspect.signature(obj).parameters:
                return False
        except (TypeError, ValueError):
            continue
    return True


def apply_sparse_attention(model):
    """ComfyUI's Model Sparse Attention (sol-attn, its defaults) on every step
    of the re-draw. -> (model, note). Kijai's low-VRAM attention composes with
    it. A ComfyUI without the node runs dense."""
    try:
        from comfy_extras.nodes_sparse_attention import (apply_block_sparse_attention,
                                                         parse_block_list)
    except Exception as exc:  # noqa: BLE001
        return model, f"dense (this ComfyUI has no sparse attention: {exc.__class__.__name__})"
    if not _block_forward_takes_attention(model):
        return model, ("dense (ComfyUI-KJNodes is too old for sparse attention: "
                       "update it)")
    try:
        m = apply_block_sparse_attention(
            model, tau=1.3, topk_ratio=0.0, vsa=False, start_percent=0.0, end_percent=1.0,
            min_tokens=12288, dense_blocks=parse_block_list(""),
            sink_conditioning="exact_kv_and_rows", extra_tokens=256, verbose=False)
    except Exception as exc:  # noqa: BLE001 -- another core's signature: dense, not a failed run
        return model, f"dense (sparse attention failed: {exc.__class__.__name__}: {exc})"
    return m, "sparse (sol-attn, tau 1.3)"


def _apply_low_vram_patches(model, attn_chunks=4, ffn_chunks=2):
    """Apply AdaLN port, 4-way attention head chunking, and 2-way SwiGLU FFN
    token chunking (as in Seanaiochat.json's MiniMaxLowVRAMAttention(4) +
    MiniMaxChunkFeedForward(2, 4096)) onto `model` so tiled upsampling uses
    minimal activation VRAM even when `upsample_dit` is wired from a bare
    UNETLoader. Mathematically identical to the unchunked forward."""
    if model is None or not hasattr(model, "get_model_object"):
        return model
    try:
        import nodes as _nodes_mod
        nmap = getattr(_nodes_mod, "NODE_CLASS_MAPPINGS", {}) or {}
        if hasattr(model, "patches") and "H3AdaLNLoRAFix" in nmap:
            _res = nmap["H3AdaLNLoRAFix"].execute(model=model, mode="port")
            model = (_res.args if hasattr(_res, "args") else tuple(_res))[0]
        if int(attn_chunks) > 1 and "MiniMaxLowVRAMAttention" in nmap:
            _res = nmap["MiniMaxLowVRAMAttention"].execute(
                model=model, head_chunks=int(attn_chunks))
            model = (_res.args if hasattr(_res, "args") else tuple(_res))[0]
        if int(ffn_chunks) > 1 and "MiniMaxChunkFeedForward" in nmap:
            _res = nmap["MiniMaxChunkFeedForward"].execute(
                model=model, chunks=int(ffn_chunks), seq_threshold=4096)
            model = (_res.args if hasattr(_res, "args") else tuple(_res))[0]
    except Exception:
        pass
    return model


def sample_tile(model, cond, sampler, sigmas, video, audio, noise, seed, lock=None):
    """One tile through `sigmas`, audio frozen. -> the tile's video latent.

    `noise` is a (video, audio) pair to use as is. SamplerCustomAdvanced's
    path, with a positive-only guider at cfg 1 and the x0 preview. `lock` is
    a `structure_lock_fn` for this tile, or None.
    """
    import comfy.model_management as mm
    import comfy.nested_tensor as nt
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview

    samples = comfy.sample.fix_empty_latent_channels(
        model, nt.NestedTensor((video, audio)), None, None)
    mask = nt.NestedTensor((
        torch.ones((1, 1, 1, video.shape[3], video.shape[4]),
                   device=video.device, dtype=torch.float32),
        torch.zeros((1, 32, 2, audio.shape[-1]),
                    device=audio.device, dtype=audio.dtype)))
    guider = comfy.samplers.CFGGuider(model)
    guider.inner_set_conds({"positive": cond})
    if lock is not None:
        import comfy.model_patcher as mp
        # On the guider's own options, not a model clone: the DiT stays staged.
        guider.model_options = mp.set_model_options_post_cfg_function(
            mp.create_model_options_clone(guider.model_options), lock)
    callback = latent_preview.prepare_callback(
        guider.model_patcher, sigmas.shape[-1] - 1, {})
    _ts = time.perf_counter()
    out = guider.sample(nt.NestedTensor(noise), samples, sampler, sigmas,
                        denoise_mask=mask, callback=callback,
                        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                        seed=seed)
    _SAMPLE_S[0] += time.perf_counter() - _ts
    res = out.to(mm.intermediate_device()).tensors[0]
    del out, guider, samples, mask
    _t = time.perf_counter()
    mm.soft_empty_cache()
    _FLUSH_S[0] += time.perf_counter() - _t
    return res


def _highpass_detail_merge(base_video, redrawn_video, sigma=8.0):
    """Keep `base_video` (full-frame 3D latent upscaler output, zero tiles) for
    tile-scale low frequencies (`> sigma` latent px, default 8.0 = 128 screen px)
    and take all mid + fine structural and texture detail from `redrawn_video`
    (tiled DiT re-draw): `base + (D - G_sigma(D))`.

    Eliminates tile-to-tile low-frequency DC/colour/contrast grid seams while
    preserving all mid-band facial/structural detail and fine texture from the
    `h3upscale` LoRA.
    """
    sigma = float(sigma or 0.0)
    if sigma <= 0.0 or base_video.shape != redrawn_video.shape or base_video.dim() < 5:
        return redrawn_video
    base_f = base_video.to(device=redrawn_video.device, dtype=torch.float32)
    red_f = redrawn_video.to(dtype=torch.float32)
    d = red_f - base_f
    return (base_f + (d - _blur_hw(d, sigma))).to(dtype=redrawn_video.dtype)


def _blur_hw(x, sigma):
    """[B, C, T, H, W] float32, every H x W plane Gaussian-blurred at `sigma`
    latent cells (reflect-padded, so a tile's edge is not pulled dark)."""
    b, c, t, h, w = x.shape
    radius = max(1, int(math.ceil(3.0 * sigma)))
    rad_h = min(radius, max(0, h - 1))
    rad_w = min(radius, max(0, w - 1))
    low = x.reshape(b * c * t, 1, h, w)
    if rad_w > 0:
        cw = torch.arange(-rad_w, rad_w + 1, device=low.device, dtype=torch.float32)
        kw = torch.exp(-0.5 * (cw / sigma) ** 2)
        kw = (kw / kw.sum()).view(1, 1, 1, -1)
        low = F.conv2d(F.pad(low, (rad_w, rad_w, 0, 0), mode="reflect"), kw)
    if rad_h > 0:
        ch = torch.arange(-rad_h, rad_h + 1, device=low.device, dtype=torch.float32)
        kh = torch.exp(-0.5 * (ch / sigma) ** 2)
        kh = (kh / kh.sum()).view(1, 1, -1, 1)
        low = F.conv2d(F.pad(low, (0, 0, rad_h, rad_h), mode="reflect"), kh)
    return low.reshape(b, c, t, h, w)


def structure_lock_fn(anchor, strength, sigma=LOCK_SIGMA):
    """A sampler_post_cfg_function holding the re-draw's layout to `anchor`.

    Regen2K starts from sigma 0.8: four parts noise to one part clip, so every
    tile and every chunk is free to re-imagine a face, a hand, a line of a
    cartoon. Each model call predicts the clean tile (x0); this moves that
    prediction's blur at `sigma` latent cells `strength` of the way onto the
    blur of `anchor` (the tile's 3D-upscaled source) and leaves the finer band
    to the model. Applied at every step, the later steps sharpen the source's
    layout rather than the one they made up -- and two chunks crossfading over
    the same frames, or two tiles over the same pixels, are held to the same
    source, so they agree. The last step's x0 is the output: at strength 1 it
    is the source's blur plus the re-draw's detail above it -- the split
    `detail_sigma` makes once, after the fact, made inside every step.

    The sampler hands over the flat AV pack (video first) of the nested
    latent; audio is left as predicted (it is frozen by the mask anyway).
    """
    shape = tuple(int(x) for x in anchor.shape)
    n = math.prod(shape[1:])
    strength = float(strength)
    held = {}

    def fn(args):
        d = args["denoised"]
        flat = d.reshape(d.shape[0], -1)
        if strength <= 0.0 or flat.shape[1] < n or d.shape[0] != shape[0]:
            return d
        a_low = held.get(d.device)
        if a_low is None:
            a_low = held[d.device] = _blur_hw(anchor.to(d.device, torch.float32), sigma)
        v = flat[:, :n].reshape(shape).float()
        v = v + strength * (a_low - _blur_hw(v, sigma))
        out = flat.clone()
        out[:, :n] = v.reshape(shape[0], -1).to(out.dtype)
        return out.reshape(d.shape)
    return fn


def _cond_extra_tokens(cond):
    """Extra non-strided conditioning rows (`minimax_refs` + `minimax_keyframes`)
    that ride every tile forward alongside the tile's own video tokens."""
    if not cond:
        return 0
    extra = 0
    try:
        d = cond[0][1] if len(cond[0]) > 1 and isinstance(cond[0][1], dict) else {}
        for blk in (d.get("minimax_refs") or ()):
            kind = blk.get("kind")
            if kind == "image":
                extra += max(0, int(blk.get("latent_h", 0)) // 2) * max(0, int(blk.get("latent_w", 0)) // 2)
            elif kind in ("video", "video_audio"):
                extra += (max(0, int(blk.get("latent_t", 0)))
                          * max(0, int(blk.get("latent_h", 0)) // 2)
                          * max(0, int(blk.get("latent_w", 0)) // 2))
            if kind in ("audio", "video_audio"):
                extra += max(0, int(blk.get("ref_audio_t", 0))) * 2
    except Exception:
        return 0
    return extra


def _cond_resized(cond, h, w):
    """`cond` with every (unstrided) keyframe latent resized to an h x w grid."""
    out = []
    for emb, d in cond:
        d = dict(d)
        if d.get("minimax_keyframes"):
            kfs = []
            for kf in d["minimax_keyframes"]:
                kf = dict(kf)
                lt = kf.get("latent")
                if lt is not None and int(kf.get(STRIDE_KEY, 1)) <= 1 and (
                        lt.shape[3] != h or lt.shape[4] != w):
                    kf["latent"] = _resize_planes(lt, h, w, torch.float32).to(lt.dtype)
                kfs.append(kf)
            d["minimax_keyframes"] = kfs
        out.append([emb, d])
    return out


def global_pass(model, cond, sampler, sigmas, video, audio, seed, budget,
                lock_src=None, structure_lock=0.0, lock_sigma=LOCK_SIGMA, factor=1):
    """`video` re-drawn whole-frame at the largest size whose sequence fits
    `budget`, then brought back to its own grid. -> (latent, note); the
    latent is None when it would not help (it fits whole already, a guided
    re-draw, or the frame would shrink below 8x8 latent cells)."""
    import comfy.nested_tensor as nt
    import comfy.sample
    if int(factor) > 1:
        return None, ""
    _, _, t, h, w = video.shape
    full = seq_len(w * SPATIAL, h * SPATIAL, t)
    scale = math.sqrt(float(budget) / float(full))
    if scale >= 1.0:
        return None, ""
    hg, wg = int(h * scale) // 2 * 2, int(w * scale) // 2 * 2
    if hg < 8 or wg < 8:
        return None, f"global pass skipped: {wg * SPATIAL}x{hg * SPATIAL}px is too small to help"
    small = F.interpolate(video.float(), size=(t, hg, wg), mode="area").to(video.dtype)
    field = comfy.sample.prepare_noise(nt.NestedTensor((small, audio)), seed)
    lock = None
    if lock_src is not None and float(structure_lock) > 0.0:
        lock = structure_lock_fn(
            F.interpolate(lock_src.float(), size=(t, hg, wg), mode="area").to(video.dtype),
            structure_lock, max(0.5, float(lock_sigma) * scale))
    out = sample_tile(model, _cond_resized(cond, hg, wg), sampler, sigmas, small, audio,
                      (field.tensors[0], field.tensors[1]), seed, lock=lock)
    back = F.interpolate(out.to(video.device).float(), size=(t, h, w), mode="trilinear",
                         align_corners=False).to(video.dtype)
    return back, f"global pass at {wg * SPATIAL}x{hg * SPATIAL}px"


def redraw_chunk(model, cond, sampler, sigmas, video, audio, seed, factor=1,
                 budget=TOKEN_BUDGET, overlap_px=TILE_OVERLAP_PX,
                 detail_sigma=0.0, noise_video=None, structure_lock=STRUCTURE_LOCK,
                 lock_sigma=LOCK_SIGMA, anchor=None, allow_tiles=True,
                 tile_context="none"):
    """Every tile stepped together, merged under the feather after each step.

    Step 0 hands each tile its slice of one noise field drawn at the chunk's
    full size; later steps add no noise. Audio rides along frozen. `factor`
    > 1 is a guided re-draw: tiles on the guide's grid, guide rows budgeted.
    When `detail_sigma > 0`, applies `_highpass_detail_merge(video, x, detail_sigma)`
    so low-frequency tile-grid shifts are stripped while fine detail is kept.
    `structure_lock` > 0 holds every tile's large shapes to its slice of
    `video` (the upscaled source) at every step; see `structure_lock_fn`.
    `anchor` replaces `video`
    as what structure_lock holds the large shapes to. `allow_tiles` False
    keeps the chunk whole-frame whatever the budget, and an OOM is an error
    instead of a retry in tiles. `tile_context` "global pass" re-draws the
    chunk whole-frame at a reduced size first and holds the tiles' large
    shapes to it (see TILE_CONTEXTS).
    Includes an automatic OOM guardrail: if a tile forward runs out of VRAM at
    `budget`, halves the tile token budget to split into smaller tiles and
    retries the chunk automatically.
    """
    import comfy.nested_tensor as nt
    import comfy.sample

    _, _, t, h, w = video.shape
    field = comfy.sample.prepare_noise(nt.NestedTensor((video, audio)), seed)
    nv, na = field.tensors[0], field.tensors[1]
    if noise_video is not None:
        # This chunk's frames of one film-wide field (noise=film): the frames
        # two chunks share get the same noise from both sides.
        nv = noise_video.to(device=nv.device, dtype=nv.dtype)
    dev, dt = video.device, video.dtype
    cur_budget = int(budget)
    _oom_retries = 0
    _LAST_OOM[0] = 0

    extra = _cond_extra_tokens(cond)
    lock_src = video if anchor is None else anchor.to(device=video.device, dtype=video.dtype)
    while True:
        tiles, grid = tile_grid(h, w, t, factor,
                                max(4096, cur_budget - extra) if allow_tiles else 1 << 40,
                                overlap_px)
        prepared = []
        for (r0, c0, rows, cols, st, sl, top, left, bottom, right) in tiles:
            prepared.append((r0, c0, rows, cols,
                             cond_for_tile(cond, h, w, r0, c0, rows, cols),
                             feather(rows, cols, st, sl, top, left, bottom, right,
                                     dev)[None, None, None]))
        locks = [structure_lock_fn(lock_src[:, :, :, r0:r0 + rows, c0:c0 + cols],
                                   structure_lock, lock_sigma)
                 if float(structure_lock) > 0.0 else None
                 for (r0, c0, rows, cols, *_rest) in prepared]
        tokens = (seq_len(grid[2], grid[3], t) if int(factor) <= 1
                  else f"{guided_seq_len(grid[2], grid[3], t, factor)} (guide incl.)")
        _LAST_TOKENS[0] = factor_seq_len(grid[2], grid[3], t, factor) + extra
        _sig_s = [round(float(s), 4) for s in sigmas]
        print(f"[{TAG}] chunk {w * SPATIAL}x{h * SPATIAL}px x {t} latent steps: "
              f"{grid[0]}x{grid[1]} tiles of {grid[2]}x{grid[3]}px, overlap "
              f"{grid[4]}/{grid[5]}px, {tokens} tokens per "
              f"forward, stepped together, sigmas {_sig_s}"
              + (f", high-pass detail={float(detail_sigma):g}" if float(detail_sigma or 0.0) > 0.0 else "")
              + (f", structure lock {float(structure_lock):g} at {float(lock_sigma):g} cells"
                 if float(structure_lock) > 0.0 else ""),
              flush=True)

        # Multi-step (0.2.0): every step runs over ALL tiles and is merged under
        # the feather before the next step starts, so neighbours share one state
        # at every sigma. Between steps x holds the sampler's output for a CONST
        # (flow) model, i.e. x_t / (1 - sigma); the next call's zero-noise
        # noise_scaling multiplies it back by (1 - sigma), so the hand-off is
        # exact. Only step 0 draws noise, from one full-chunk field.
        _t0, _calls, _FLUSH_S[0] = time.perf_counter(), 0, 0.0
        try:
            x = video
            if len(prepared) == 1:
                # 1x1 whole-chunk tile: no spatial tile seams to feather-merge
                # between steps, so run all `sigmas` in a single `sample_tile`
                # (`CFGGuider.sample`) call instead of re-staging the 32.4 GB
                # DynamicVRAM DiT on every step.
                r0, c0, rows, cols, tcond, _m = prepared[0]
                xt = x[:, :, :, r0:r0 + rows, c0:c0 + cols].contiguous()
                noise = (nv[:, :, :, r0:r0 + rows, c0:c0 + cols].contiguous(), na)
                ot = sample_tile(model, tcond, sampler, sigmas, xt, audio, noise, seed,
                                 lock=locks[0])
                _calls += 1
                x = ot.to(device=dev, dtype=dt)
                del ot, xt
            else:
                if str(tile_context) == "global pass":
                    g, g_note = global_pass(
                        model, cond, sampler, sigmas, video, audio, seed,
                        max(4096, cur_budget - extra), lock_src, structure_lock, lock_sigma,
                        factor)
                    if g is not None:
                        _calls += 1
                        locks = [structure_lock_fn(g[:, :, :, r0:r0 + rows, c0:c0 + cols],
                                                   1.0, lock_sigma)
                                 for (r0, c0, rows, cols, *_rest) in prepared]
                        del g
                    if g_note:
                        print(f"[{TAG}] {g_note}; the {grid[0]}x{grid[1]} tiles hold their "
                              "large shapes to it" if g_note.startswith("global pass at")
                              else f"[{TAG}] {g_note}", flush=True)
                wsum = torch.zeros((1, 1, 1, h, w), device=dev, dtype=torch.float32)
                for r0, c0, rows, cols, _tcond, m in prepared:
                    wsum[:, :, :, r0:r0 + rows, c0:c0 + cols] += m[:, :, 0]
                wsum = wsum.clamp(min=1e-8)
                for step in range(len(sigmas) - 1):
                    two = sigmas[step:step + 2]
                    acc = torch.zeros_like(x, dtype=torch.float32)
                    for (r0, c0, rows, cols, tcond, m), lock in zip(prepared, locks):
                        xt = x[:, :, :, r0:r0 + rows, c0:c0 + cols].contiguous()
                        if step == 0:
                            noise = (nv[:, :, :, r0:r0 + rows, c0:c0 + cols].contiguous(), na)
                        else:
                            noise = (torch.zeros_like(xt), torch.zeros_like(audio))
                        ot = sample_tile(model, tcond, sampler, two, xt, audio, noise, seed,
                                         lock=lock)
                        _calls += 1
                        acc[:, :, :, r0:r0 + rows, c0:c0 + cols] += \
                            ot.to(device=dev, dtype=torch.float32) * m
                        del ot, xt
                    x = (acc / wsum).to(dtype=dt)
                    del acc
                del wsum
            _dt = time.perf_counter() - _t0
            _LAST_OOM[0] = _oom_retries
            print(f"[{TAG}] chunk re-draw: {len(sigmas) - 1} step(s) x {len(prepared)} tile(s) "
                  f"({grid[0]}x{grid[1]}) in {_calls} sampler call(s), {_dt:.1f} s "
                  f"({_dt / max(1, _calls):.2f} s per call, cache flush {_FLUSH_S[0]:.1f} s), "
                  f"OOM retries {_oom_retries}"
                  + (f" (budget -> {cur_budget})" if cur_budget != int(budget) else ""),
                  flush=True)
            break
        except Exception as exc:
            if _is_oom(exc) and not allow_tiles:
                raise RuntimeError(
                    f"{TAG}: out of VRAM on a whole-frame {w * SPATIAL}x{h * SPATIAL}px chunk of "
                    f"{t} latent steps with tiling off. Turn tiling on, lower megapixels, or "
                    "(chunking on) lower tile_tokens for shorter chunks.") from exc
            if not _is_oom(exc) or cur_budget <= 16384:
                raise
            next_budget = max(16384, int(cur_budget * 0.65))
            if next_budget >= cur_budget:
                raise
            _oom_retries += 1
            print(f"[{TAG}] CUDA OOM at tile budget {cur_budget} ({exc}); "
                  f"lowering tile budget to {next_budget} (more, smaller tiles) and retrying chunk...",
                  flush=True)
            _free_all_gpu()
            cur_budget = next_budget

    if float(detail_sigma or 0.0) > 0.0:
        x = _highpass_detail_merge(video, x, sigma=float(detail_sigma))
    _LAST_BUDGET[0] = cur_budget
    return x, cur_budget


def _step_weights(plan, i, device):
    """Chunk i's weight per latent step in a stepped merge: 1 inside, ramping
    linearly to near 0 across the steps it shares with each neighbour, so the
    end of a chunk -- where it is weakest -- counts least."""
    k0, _f0, k1, _f1 = plan[i]
    w = torch.ones(k1 - k0, dtype=torch.float32, device=device)
    if i > 0:
        ov = max(0, plan[i - 1][2] - k0)
        if ov:
            w[:ov] = torch.arange(1, ov + 1, dtype=torch.float32, device=device) / (ov + 1)
    if i + 1 < len(plan):
        ov = max(0, k1 - plan[i + 1][0])
        if ov:
            w[-ov:] = torch.minimum(w[-ov:], torch.arange(ov, 0, -1, dtype=torch.float32,
                                                          device=device) / (ov + 1))
    return w.view(1, 1, -1, 1, 1)


def redraw_stepped(model, plan, videos, audios, conds, sampler, sigmas, seed,
                   mem_samples=None, budget=None, overlap_px=TILE_OVERLAP_PX,
                   anchors=None, structure_lock=0.0, lock_sigma=LOCK_SIGMA,
                   tile_context="none"):
    """Every chunk stepped TOGETHER (chunk_join=stepped). -> (film's video latent, notes).

    `videos` are the chunks' upscaled latents (plan order), `audios` their
    audio slices, `conds` their conditioning, `anchors` what `structure_lock`
    holds each chunk to (None = its upscaled latent). The film starts as the
    chunks crossfaded into one latent; each sigma step runs every chunk from
    its slice of that one state (the first step with its slice of one
    film-wide noise field) and merges the results before the next step --
    what spatial tiles have always done, along time.

    `budget` (tokens per forward) cuts a chunk that doesn't fit whole into
    tiles, as `redraw_chunk` does; then every TILE of every chunk is stepped
    together, each weighted by its feather times its chunk's `_step_weights`,
    so the film is one grid in time and space. None = whole-frame chunks."""
    import comfy.model_management as mm
    c, h, w = int(videos[0].shape[1]), int(videos[0].shape[3]), int(videos[0].shape[4])
    tv = int(plan[-1][2])
    dev, dt = videos[0].device, videos[0].dtype
    film = torch.zeros((1, c, tv, h, w), dtype=dt, device=dev)
    dummy = torch.zeros((1, 32, 2, 1), dtype=dt, device=dev)
    end = 0
    for (k0, f0, k1, _f1), v in zip(plan, videos):
        end, _ = _join_into(film, dummy, end, 0, v.to(dev, dt), dummy[..., :0], k0, f0)
    nv = torch.randn((1, c, tv, h, w), generator=torch.Generator().manual_seed(int(seed)),
                     dtype=torch.float32, device="cpu").to(dev, dt)
    # Every chunk's tiles: (r0, c0, rows, cols, cond, weight, lock, tokens).
    work, grids, notes = [], [], []
    wsum = torch.zeros((1, 1, tv, h, w), dtype=torch.float32, device=dev)
    for i, (k0, _f0, k1, _f1) in enumerate(plan):
        t = k1 - k0
        extra = _cond_extra_tokens(conds[i])
        tb = max(4096, int(budget) - extra) if budget is not None else 1 << 40
        tiles, grid = tile_grid(h, w, t, 1, tb, overlap_px)
        grids.append(grid[:2])
        src = videos[i] if anchors is None or anchors[i] is None else \
            anchors[i].to(device=dev, dtype=dt)
        tw = _step_weights(plan, i, dev)
        g = None
        if len(tiles) > 1 and str(tile_context) == "global pass":
            g, g_note = global_pass(model, conds[i], sampler, sigmas, videos[i], audios[i],
                                    seed, tb, src, structure_lock, lock_sigma)
            if g_note:
                notes.append(f"chunk {i + 1}: {g_note}")
        chunk = []
        for (r0, c0, rows, cols, st, sl, top, left, bottom, right) in tiles:
            m = feather(rows, cols, st, sl, top, left, bottom, right, dev)[None, None, None]
            wt = tw * m
            wsum[:, :, k0:k1, r0:r0 + rows, c0:c0 + cols] += wt
            if g is not None:
                lock = structure_lock_fn(g[:, :, :, r0:r0 + rows, c0:c0 + cols], 1.0, lock_sigma)
            elif float(structure_lock) > 0.0:
                lock = structure_lock_fn(src[:, :, :, r0:r0 + rows, c0:c0 + cols],
                                         structure_lock, lock_sigma)
            else:
                lock = None
            chunk.append((r0, c0, rows, cols, cond_for_tile(conds[i], h, w, r0, c0, rows, cols),
                          wt, lock, seq_len(cols * SPATIAL, rows * SPATIAL, t) + extra))
        work.append(chunk)
        del g, src
    wsum = wsum.clamp(min=1e-8)
    x = film
    _t0, _FLUSH_S[0], _SAMPLE_S[0] = time.perf_counter(), 0.0, 0.0
    for step in range(len(sigmas) - 1):
        two = sigmas[step:step + 2]
        acc = torch.zeros_like(x, dtype=torch.float32)
        for i, (k0, _f0, k1, _f1) in enumerate(plan):
            for (r0, c0, rows, cols, tcond, wt, lock, tokens) in work[i]:
                mm.throw_exception_if_processing_interrupted()
                xt = x[:, :, k0:k1, r0:r0 + rows, c0:c0 + cols].contiguous()
                noise = ((nv[:, :, k0:k1, r0:r0 + rows, c0:c0 + cols].contiguous()
                          if step == 0 else torch.zeros_like(xt)),
                         torch.zeros_like(audios[i]))
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                ot = sample_tile(model, tcond, sampler, two, xt, audios[i], noise, seed,
                                 lock=lock)
                if mem_samples is not None and torch.cuda.is_available():
                    mem_samples.append((tokens,
                                        int(torch.cuda.max_memory_allocated()
                                            - torch.cuda.memory_allocated()),
                                        int(torch.cuda.memory_allocated())))
                acc[:, :, k0:k1, r0:r0 + rows, c0:c0 + cols] += \
                    ot.to(device=dev, dtype=torch.float32) * wt
                del ot, xt
        x = (acc / wsum).to(dtype=dt)
        del acc
    _wall = time.perf_counter() - _t0
    n_tiles = sum(len(c_) for c_ in work)
    tiled = sorted({f"{a}x{b}" for a, b in grids if (a, b) != (1, 1)})
    notes.insert(0, f"{len(plan)} chunks" + (f" in {n_tiles} tiles ({', '.join(tiled)} per tiled chunk)"
                                             if tiled else "")
                 + f" x {len(sigmas) - 1} steps in {n_tiles * (len(sigmas) - 1)} sampler calls")
    _calls = max(1, n_tiles * (len(sigmas) - 1))
    notes.append(f"time: {_SAMPLE_S[0]:.1f} s sampling ({_SAMPLE_S[0] / _calls:.2f} s a call), "
                 f"{_FLUSH_S[0]:.1f} s cache flush, "
                 f"{max(0.0, _wall - _SAMPLE_S[0] - _FLUSH_S[0]):.1f} s setup and merging")
    return x, notes


def _crossfade(a, b, dim):
    n = a.shape[dim]
    wgt = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    shape = [1] * a.ndim
    shape[dim] = n
    return a + (b - a) * wgt.view(shape)


def join_chunk(done_v, done_a, v, a, k0, f0):
    """Append a finished chunk, crossfading where it overlaps what is done."""
    if done_v is None:
        return v, a
    a0 = round(f0 * FRAME_RESCALE)
    out_v = torch.zeros((1, done_v.shape[1], max(done_v.shape[2], k0 + v.shape[2]),
                         done_v.shape[3], done_v.shape[4]),
                        device=done_v.device, dtype=done_v.dtype)
    out_a = torch.zeros((1, 32, 2, max(done_a.shape[-1], a0 + a.shape[-1])),
                        device=done_a.device, dtype=done_a.dtype)
    out_v[:, :, :done_v.shape[2]] = done_v
    out_a[..., :done_a.shape[-1]] = done_a
    ov = min(done_v.shape[2] - k0, v.shape[2])
    if ov > 0:
        out_v[:, :, k0:k0 + ov] = _crossfade(out_v[:, :, k0:k0 + ov].clone(),
                                             v[:, :, :ov], 2)
        v = v[:, :, ov:]
    at = k0 + max(ov, 0)
    if v.shape[2] > 0:
        out_v[:, :, at:at + v.shape[2]] = v
    ova = min(done_a.shape[-1] - a0, a.shape[-1])
    if ova > 0:
        out_a[..., a0:a0 + ova] = _crossfade(out_a[..., a0:a0 + ova].clone(),
                                             a[..., :ova], 3)
        a = a[..., ova:]
    at = a0 + max(ova, 0)
    if a.shape[-1] > 0:
        out_a[..., at:at + a.shape[-1]] = a
    return out_v, out_a


def _join_into(out_v, out_a, end_v, end_a, v, a, k0, f0):
    """Write chunk `(v, a)` into preallocated `(out_v, out_a)` with
    `join_chunk`'s exact crossfade arithmetic. -> `(end_v, end_a)`."""
    if end_v == 0 and end_a == 0:
        nv, na = int(v.shape[2]), int(a.shape[-1])
        out_v[:, :, :nv] = v
        out_a[..., :na] = a
        return nv, na
    a0 = round(f0 * FRAME_RESCALE)
    ov = min(end_v - k0, v.shape[2])
    if ov > 0:
        out_v[:, :, k0:k0 + ov] = _crossfade(out_v[:, :, k0:k0 + ov].clone(),
                                             v[:, :, :ov], 2)
        v = v[:, :, ov:]
    at = k0 + max(ov, 0)
    if v.shape[2] > 0:
        out_v[:, :, at:at + v.shape[2]] = v
    end_v = max(end_v, at + int(v.shape[2]))
    ova = min(end_a - a0, a.shape[-1])
    if ova > 0:
        out_a[..., a0:a0 + ova] = _crossfade(out_a[..., a0:a0 + ova].clone(),
                                             a[..., :ova], 3)
        a = a[..., ova:]
    at = a0 + max(ova, 0)
    if a.shape[-1] > 0:
        out_a[..., at:at + a.shape[-1]] = a
    end_a = max(end_a, at + int(a.shape[-1]))
    return end_v, end_a


def _free(model):
    import comfy.model_management as mm
    if hasattr(model, "clone_base_uuid"):
        mm.unload_model_and_clones(model, unload_additional_models=False)
        mm.soft_empty_cache()


def _regen2k_chunk_frames(w_px, h_px, budget, chunk_frames=_DEFAULT_CHUNK_FRAMES,
                          overlap_px=TILE_OVERLAP_PX, explicit=None):
    """Pick chunk length from `budget` for regen2k mode (R0).

    If `chunk_frames` was not explicitly set on `HTCUpsampleSettings` and a
    102 f full-frame chunk exceeds `budget`, shorten `chunk_frames` in steps
    of 17 down to a floor of 68 f (20 steps) before allowing spatial tiles.
    """
    if explicit is None:
        explicit = not isinstance(chunk_frames, _DefaultInt)
    cf = int(chunk_frames)
    if not explicit and cf > REGEN2K_MIN_CHUNK_FRAMES:
        while cf > REGEN2K_MIN_CHUNK_FRAMES:
            steps = max(1, (cf // 17) * 5)
            if seq_len(w_px, h_px, steps) <= int(budget):
                break
            cf = max(REGEN2K_MIN_CHUNK_FRAMES, cf - 17)
    steps = max(1, (cf // 17) * 5)
    grid = solve_grid(w_px, h_px, steps, overlap_px=snap_overlap(overlap_px),
                      budget=int(budget))
    if grid is not None and grid[0] == 1 and grid[1] == 1:
        print(f"[{TAG}] regen2k: budget {int(budget)} -> chunk {cf} f full-frame",
              flush=True)
    elif grid is not None:
        print(f"[{TAG}] regen2k: budget {int(budget)} -> chunk {cf} f "
              f"-> {grid[0]}x{grid[1]} tiles", flush=True)
    return cf, grid


def upsample(model, conditioning, latent, sampler, upscaler, megapixels,
             denoise=DENOISE, guide=None, budget=TOKEN_BUDGET, steps=STEPS,
             start_sigma=0.0, overlap_px=TILE_OVERLAP_PX, detail_sigma=0.0,
             chunk_frames=_DEFAULT_CHUNK_FRAMES, chunk_overlap=CHUNK_OVERLAP,
             anchor=ANCHOR_AUG, seed=SEED, noise="per_chunk",
             attn_chunks=4, ffn_chunks=2, sigmas=None,
             structure_lock=STRUCTURE_LOCK, lock_sigma=LOCK_SIGMA,
             attention="dense", chunk_join="crossfade", chunking=True, tiling=True,
             lock_anchor="source", tile_context="none", ic=None, sharpness=SHARPNESS):
    """The whole recipe on an H3 AV latent. -> the upsampled AV latent dict.

    `sharpness` (1..6) is the 2D latent upscaler's tier; the 3D one ignores it.

    `conditioning` is one conditioning for the whole clip, or a callable
    `(f0, f1) -> conditioning` giving each chunk its own -- the chain passes
    the latter, because each hop was generated from its own prompt.

    `guide` (a Guide) makes the re-draw a guided one: each chunk's canvas
    frames, shrunk to 1/factor of the output, ride every tile as the h3upscale
    guide, and `guide.cond` replaces `conditioning`. Same step, same sigmas.

    `budget` caps one tile forward's tokens (default: the recipe's). Lower it
    and the same frame is cut into more, smaller tiles -- less VRAM, more
    forwards, the same feathered merge.

    `structure_lock` (0..1) holds each chunk's re-draw to the large shapes of
    its own 3D-upscaled frames (`structure_lock_fn`); 0 is the recipe as is.
    `attention`, `chunk_join`, `chunking`, `tiling` and `lock_anchor`: see
    ATTENTION_MODES and the notes above it.

    `ic` (an upsample_ic.ChunkConds) makes it the in-context re-draw: before the
    DiT is staged, each chunk's conditioning is built from that chunk's own
    low-res draft latent, which rides as <Video 1>; `conditioning` is then
    unused. The caller passes the IC-patched model and sigmas.
    """
    import comfy.nested_tensor as nt

    samples = latent["samples"]
    if not (getattr(samples, "is_nested", False) and len(samples.tensors) == 2
            and samples.tensors[0].ndim == 5 and samples.tensors[0].shape[1] == 24
            and samples.tensors[1].ndim == 4 and samples.tensors[1].shape[1] == 32):
        raise RuntimeError(f"{TAG}: wants a MiniMax H3 AV latent "
                           "(video [1,24,T,H,W] + audio [1,32,2,T]).")
    video, audio = samples.tensors
    if video.shape[0] != 1:
        raise RuntimeError(f"{TAG}: one clip at a time (batch 1).")
    factor = int(guide.factor) if guide is not None else 1
    if guide is not None:
        conditioning = guide.cond
    per_chunk = callable(conditioning)
    cond = None if per_chunk else clean_refs(conditioning)
    tv, ta = video.shape[2], audio.shape[-1]
    if guide is not None:
        w_px, h_px = target_size(video.shape[4] * SPATIAL, video.shape[3] * SPATIAL,
                                 megapixels, align=32 * factor)
    else:
        w_px, h_px = target_size(video.shape[4] * SPATIAL, video.shape[3] * SPATIAL,
                                 megapixels)
    model = _apply_low_vram_patches(model, attn_chunks, ffn_chunks)
    attn_note = "dense"
    if str(attention) == "sparse":
        model, attn_note = apply_sparse_attention(model)
    mem_key = memory_key(attention, attn_chunks, ffn_chunks, factor)
    budget, auto_budget, budget_note = resolve_budget(budget, mem_key)
    is_regen2k = sigmas is not None
    chunking, tiling = bool(chunking), bool(tiling)
    whole = not chunking
    if whole:
        _need = factor_seq_len(w_px, h_px, int(tv), factor)
        if _need > WHOLE_CLIP_MAX * int(budget):
            print(f"[{TAG}] chunking off: {frames_in(int(tv))} frames would be {_need} tokens "
                  f"in one pass, over {WHOLE_CLIP_MAX}x tile_tokens ({WHOLE_CLIP_MAX * int(budget)}); "
                  "chunking anyway", flush=True)
            whole = False
    if whole:
        if sigmas is not None:
            sigmas = (sigmas.detach().cpu().float() if isinstance(sigmas, torch.Tensor)
                      else torch.tensor(list(sigmas), dtype=torch.float32))
        else:
            sigmas = recipe_sigmas(model, denoise, steps, start_sigma)
        plan = [(0, 0, int(tv), frames_in(int(tv)))]
        print(f"[{TAG}] chunking off: {frames_in(int(tv))} frames in one pass "
              f"({factor_seq_len(w_px, h_px, int(tv), factor)} tokens whole-frame"
              + ("" if tiling else ", tiling off") + ")", flush=True)
    elif is_regen2k:
        sigmas = (sigmas.detach().cpu().float() if isinstance(sigmas, torch.Tensor)
                  else torch.tensor(list(sigmas), dtype=torch.float32))
        chunk_frames, _ = _regen2k_chunk_frames(w_px, h_px, budget, chunk_frames, overlap_px)
        # IC: a short tail always folds into the chunk before it (a 124-frame
        # clip is one 124-frame chunk, the IC training window)
        plan = regen2k_chunks(tv, int(chunk_frames), int(chunk_overlap),
                              w_px=(0 if ic is not None else w_px), h_px=h_px,
                              budget=budget, overlap_px=overlap_px)
    else:
        sigmas = recipe_sigmas(model, denoise, steps, start_sigma)
        plan = chunks(tv, int(chunk_frames), int(chunk_overlap))
    print(f"[{TAG}] {video.shape[4] * SPATIAL}x{video.shape[3] * SPATIAL} -> "
          f"{w_px}x{h_px} ({w_px * h_px / 1048576.0:.2f} MP), {len(plan)} "
          f"chunk(s), "
          + (f"sigmas {[round(float(s), 4) for s in sigmas]}" if sigmas is not None
             else "no re-draw (denoise 0)")
          + (f", h3upscale guide at {w_px // factor}x{h_px // factor}"
             if guide is not None and sigmas is not None else "")
          + f", tile_tokens {budget_note}, attention {attn_note}, chunk_join {chunk_join}, chunking "
          f"{'on' if chunking else 'off'}, tiling {'on' if tiling else 'off'}"
          + (f", structure lock {float(structure_lock):g} ({lock_anchor})"
             if float(structure_lock) > 0.0 else "")
          + (f", tile context {tile_context}" if str(tile_context) != "none" else ""),
          flush=True)

    # 3-Phase non-overlapping execution so VideoVAE, 3D LatentUpscaler, and the
    # 32.4 GB DynamicVRAM H3 DiT never coexist or churn in and out of VRAM
    # across 21 chunks:
    #   Phase 1: Decode & encode all `h3upscale` guide latents on VideoVAE once,
    #            store on CPU RAM (~1.5 MB/chunk), then unload VideoVAE.
    #   Phase 2: Load 3D `LatentUpscaler` onto GPU once, upscale all chunk video
    #            latents, store on CPU RAM (~6 MB/chunk), then offload upscaler.
    #   Phase 3: Stage the 32.4 GB H3 DiT onto GPU ONCE and step all chunks
    #            without unloading/re-staging the DiT between chunks.
    _tp = [time.perf_counter()]
    if _device().type == "cuda":
        _free(model)
    raw_cvs = [video[:, :, k0:k1].contiguous() for (k0, _f0, k1, _f1) in plan]
    glats = [None] * len(plan)
    if guide is not None and sigmas is not None:
        for i, cv_in in enumerate(raw_cvs):
            px = guide.decode(cv_in)[:frames_in(cv_in.shape[2])]
            glat = encode_steps(guide.encode, guide.pixels(px, w_px, h_px, factor),
                                int(cv_in.shape[2]))
            glats[i] = glat.cpu() if hasattr(glat, "cpu") else glat
            del px, glat
        if _device().type == "cuda":
            _free_all_gpu()
    if ic is not None and sigmas is not None:
        # Phase 1 (IC): every chunk's text encode (Qwen3-VL sees the chunk's
        # draft) and reference block, on CPU, before the DiT is staged, so the
        # text encoder and the DiT never share VRAM.
        _ti = time.perf_counter()
        ic_conds = {}
        for i, (k0, f0, k1, f1) in enumerate(plan):
            ic_conds[(f0, f1)] = ic.build(raw_cvs[i])
        if _device().type == "cuda":
            _free_all_gpu()
        conditioning = (lambda f0, f1, _c=ic_conds: _c[(f0, f1)])
        per_chunk = True
        print(f"[{TAG}] ic: {len(plan)} chunk(s) of {[p[3] - p[1] for p in plan]} frames, "
              f"draft kept in context as <Video 1>"
              + (f" ({ic.resized} shrunk to the reference canvas)" if ic.resized else "")
              + f", {time.perf_counter() - _ti:.1f} s", flush=True)

    _tp.append(time.perf_counter())
    upscaled_cvs = upscale_film(video, plan, upscaler, w_px, h_px, sharpness)
    anchors = [None] * len(plan)
    if (sigmas is not None and float(structure_lock) > 0.0
            and str(lock_anchor) == "source"):
        # The clip's own latent on the output grid: plain interpolation adds
        # nothing, so its blur is the source's large shapes and no one's guess.
        for i, (raw, up) in enumerate(zip(raw_cvs, upscaled_cvs)):
            anchors[i] = F.interpolate(
                raw.float(), size=tuple(int(x) for x in up.shape[2:]), mode="trilinear",
                align_corners=False).to(dtype=up.dtype)
    del raw_cvs
    _tp.append(time.perf_counter())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    film_nv = None
    _c0 = upscaled_cvs[0]
    if str(noise) == "film" and sigmas is not None and upscaled_cvs:
        film_nv = torch.randn(
            (1, int(_c0.shape[1]), tv, int(_c0.shape[3]), int(_c0.shape[4])),
            generator=torch.Generator().manual_seed(int(seed)),
            dtype=torch.float32, device="cpu").to(dtype=_c0.dtype)
    total_a = max((a0 + max(0, min(a1, int(ta)) - a0)
                   for (_k0, f0, _k1, f1) in plan
                   for a0, a1 in (audio_span(f0, f1),)), default=0)
    out_v = torch.zeros((1, int(_c0.shape[1]), int(tv), int(_c0.shape[3]), int(_c0.shape[4])),
                        dtype=_c0.dtype, device=_c0.device)
    out_a = torch.zeros((1, 32, 2, int(total_a)), dtype=audio.dtype, device=audio.device)
    end_v = end_a = 0
    _redraw_s = _join_s = 0.0
    _oom_total = 0
    init_budget = cur_budget = int(budget)
    mem_samples, _peak_all = [], 0
    stepped = False
    if str(chunk_join) == "stepped" and sigmas is not None and len(plan) > 1:
        if factor > 1:
            print(f"[{TAG}] chunk_join=stepped doesn't take the h3upscale guide; "
                  "crossfading instead", flush=True)
        else:
            conds, auds = [], []
            for i, (k0, f0, k1, f1) in enumerate(plan):
                a0, a1 = audio_span(f0, f1)
                auds.append(audio[:, :, :, a0:min(a1, ta)].contiguous())
                cv = upscaled_cvs[i]
                conds.append(cond_for_chunk(clean_refs(conditioning(f0, f1)) if per_chunk
                                            else cond, f0, f1, cv.shape[3], cv.shape[4]))
            _tr = time.perf_counter()
            while True:
                try:
                    out_v, s_notes = redraw_stepped(
                        model, plan, upscaled_cvs, auds, conds, sampler, sigmas, seed,
                        mem_samples, budget=(cur_budget if tiling else None),
                        overlap_px=overlap_px, anchors=anchors,
                        structure_lock=structure_lock, lock_sigma=lock_sigma,
                        tile_context=tile_context)
                    stepped = True
                    break
                except Exception as exc:
                    if not _is_oom(exc):
                        raise
                    _free_all_gpu()
                    mem_samples.clear()
                    nb = max(16384, int(cur_budget * 0.65))
                    if tiling and nb < cur_budget:
                        _oom_total += 1
                        print(f"[{TAG}] chunk_join=stepped ran out of VRAM at tile budget "
                              f"{cur_budget} ({exc}); stepping again at {nb} (more, smaller "
                              "tiles)", flush=True)
                        cur_budget = nb
                        continue
                    print(f"[{TAG}] chunk_join=stepped ran out of VRAM ({exc}); crossfading "
                          "chunk by chunk instead", flush=True)
                    break
            _redraw_s += time.perf_counter() - _tr
            if stepped:
                for _n in s_notes:
                    print(f"[{TAG}] chunk_join=stepped: {_n}", flush=True)
                print(f"[{TAG}] chunk_join=stepped: re-draw {_redraw_s:.1f} s", flush=True)
                if torch.cuda.is_available() and mem_samples:
                    _peak_all = max(a + r for _t, a, r in mem_samples)
                out_a[...] = audio[..., :int(total_a)]
                end_v, end_a = int(tv), int(total_a)
    for i, (k0, f0, k1, f1) in enumerate(plan if not stepped else []):
        a0, a1 = audio_span(f0, f1)
        cv = upscaled_cvs[i]
        upscaled_cvs[i] = None
        ca = audio[:, :, :, a0:min(a1, ta)].contiguous()
        if sigmas is None:
            _tj = time.perf_counter()
            end_v, end_a = _join_into(out_v, out_a, end_v, end_a, cv, ca, k0, f0)
            _join_s += time.perf_counter() - _tj
            continue
        ccond = cond_for_chunk(clean_refs(conditioning(f0, f1)) if per_chunk
                               else cond, f0, f1, cv.shape[3], cv.shape[4])
        if i > 0 and not is_regen2k:
            ccond = pin_chunk_start(ccond, out_v[:, :, :end_v], f0, anchor)
        glat = glats[i]
        glats[i] = None
        if glat is not None:
            want = (int(cv.shape[2]), int(cv.shape[3]) // factor, int(cv.shape[4]) // factor)
            got = tuple(int(x) for x in glat.shape[2:])
            if got != want:
                raise RuntimeError(f"{TAG}: guide latent {got} for a chunk that "
                                   f"needs {want} (steps, rows, cols)")
            ccond = add_guide(ccond, glat.to(cv.device, cv.dtype), factor)
            del glat
        if cur_budget < init_budget:
            _gi = solve_grid(int(cv.shape[4]) * SPATIAL, int(cv.shape[3]) * SPATIAL,
                             int(cv.shape[2]), overlap_px=snap_overlap(overlap_px),
                             budget=init_budget, factor=factor)
            _gc = solve_grid(int(cv.shape[4]) * SPATIAL, int(cv.shape[3]) * SPATIAL,
                             int(cv.shape[2]), overlap_px=snap_overlap(overlap_px),
                             budget=cur_budget, factor=factor)
            if _gi is not None and _gc is not None and _gi[:2] != _gc[:2]:
                print(f"[{TAG}] chunk {i + 1}/{len(plan)}: keeping lowered budget "
                      f"{cur_budget} ({_gc[0]}x{_gc[1]} tiles; would fit "
                      f"{_gi[0]}x{_gi[1]} at {init_budget})", flush=True)
        lock_anchor_t = anchors[i]      # not `anchor`: that is the pin strength above
        anchors[i] = None
        _tr = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _rc = redraw_chunk(model, ccond, sampler, sigmas, cv, ca, int(seed), factor,
                           cur_budget, overlap_px, detail_sigma=detail_sigma,
                           noise_video=(None if film_nv is None
                                        else film_nv[:, :, k0:k1]),
                           structure_lock=structure_lock, lock_sigma=lock_sigma,
                           anchor=lock_anchor_t, allow_tiles=tiling,
                           tile_context=tile_context)
        rv, cur_budget = (_rc if isinstance(_rc, tuple) else (_rc, cur_budget))
        _redraw_s += time.perf_counter() - _tr
        _oom_total += _LAST_OOM[0]
        if torch.cuda.is_available():
            _pk, _res = torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()
            _peak_all = max(_peak_all, _pk)
            if _LAST_OOM[0] == 0:
                mem_samples.append((int(_LAST_TOKENS[0]), int(_pk - _res), int(_res)))
        if i == 0:
            _, _g0 = tile_grid(int(cv.shape[3]), int(cv.shape[4]), int(cv.shape[2]),
                               factor, cur_budget, overlap_px)
            _tok0 = (seq_len(_g0[2], _g0[3], int(cv.shape[2])) if int(factor) <= 1
                     else guided_seq_len(_g0[2], _g0[3], int(cv.shape[2]), factor))
            _vram0 = (f"{torch.cuda.max_memory_allocated() / 1048576.0:.0f} MB"
                      if torch.cuda.is_available() else "0 MB")
            print(f"[{TAG}] chunk 1: peak VRAM {_vram0}, {_tok0} tokens/forward, "
                  f"budget {cur_budget}"
                  + ((" (auto: the next run starts there)" if auto_budget else
                      f"; set upsample_tile_tokens to {cur_budget} (or 0 = auto) to skip "
                      "the OOM next run") if cur_budget < init_budget else ""),
                  flush=True)
        _tj = time.perf_counter()
        end_v, end_a = _join_into(out_v, out_a, end_v, end_a, rv, ca, k0, f0)
        _join_s += time.perf_counter() - _tj
        del cv, rv
    _LAST_BUDGET[0] = cur_budget
    if end_v < out_v.shape[2]:
        out_v = out_v[:, :, :end_v]
    if end_a < out_a.shape[-1]:
        out_a = out_a[..., :end_a]
    assert end_v == int(tv) and end_a == int(total_a), (end_v, tv, end_a, total_a)
    _peak_mb = _peak_all / 1048576.0
    _lat_mb = (out_v.numel() * out_v.element_size()
               + out_a.numel() * out_a.element_size()) / 1048576.0
    _free(model)
    _tp.append(time.perf_counter())
    print(f"[{TAG}] timing: guide prep {_tp[1] - _tp[0]:.1f} s, " + ("2D" if is_2d_file(upscaler) else "3D") + " upscale "
          f"{_tp[2] - _tp[1]:.1f} s, re-draw {_redraw_s:.1f} s, join {_join_s:.1f} s "
          f"({len(plan)} chunk(s), OOM retries {_oom_total}, "
          f"joined latent {_lat_mb:.1f} MB"
          + (f", peak VRAM {_peak_mb:.0f} MB" if torch.cuda.is_available() else "")
          + ")", flush=True)
    if auto_budget and sigmas is not None:
        note = record_budget(mem_key, init_budget, cur_budget, mem_samples)
        if note:
            print(f"[{TAG}] auto tile_tokens: started at {init_budget}, {note}", flush=True)
    return {"samples": nt.NestedTensor((out_v, out_a))}


# --- inside the chain -------------------------------------------------------
#
# HandTieClips runs the recipe once over the WHOLE film after its last hop,
# not per hop: a hop re-drawn on its own would put a fresh texture step at
# every join, while one pass over the joined film lets the recipe's own
# 17-frame chunk crossfades run across the joins. It is delivery-only -- the
# chain has handed every pin on before this starts, so nothing here can reach
# what a hop inherits.

def segments(hops):
    """Split the chain's hops at restarts. A restart is a cut by design and
    opens on frame 0 of its own VAE cycle, so it cannot be joined to what came
    before on the latent grid -- each run between restarts is upsampled alone.
    `hops` are dicts with a `start` flag. -> [[hop, ...], ...]"""
    out = []
    for h in hops:
        if h["start"] or not out:
            out.append([])
        out[-1].append(h)
    return out


def stitch(seg):
    """One continuous latent for a run of hops. -> (video, audio).

    Each hop after the first drops the `drop` frames the master dropped (its
    regenerated overlap). That only joins cleanly on the latent grid when the
    drop is a whole number of latent steps AND the first kept step sits at the
    phase the running latent is at -- otherwise a 1-frame key step would be
    read as a 4-frame one and the timing would slip. Every H3 hop is 17k+5
    frames and every shipped overlap (5 / 22 / 39) meets both, but it is
    checked rather than assumed, with a message instead of wrong frames.
    """
    vids, auds, steps = [], [], 0
    for n, h in enumerate(seg):
        v, a = h["video"], h["audio"]
        if v is None or a is None:
            raise ValueError(f"hop {h['hop']} has no stored sampler latent "
                             "(a cache entry from an older build)")
        v = v if v.ndim == 5 else v.unsqueeze(0)
        a = a if a.ndim == 4 else a.unsqueeze(0)
        drop = 0 if n == 0 else int(h["drop"])
        k = tokens_to_reach(drop)
        if frames_in(k) != drop:
            raise ValueError(f"hop {h['hop']}: an overlap of {drop} frames is not "
                             "a whole number of latent steps")
        if n and k % 5 != steps % 5:
            raise ValueError(f"hop {h['hop']}: its kept frames start at VAE "
                             f"phase {k % 5}, the joined latent is at "
                             f"{steps % 5}")
        # Hand-tie cuts the previous hop's tail instead of this hop's head.
        tail = int(h.get("tail") or 0)
        end = int(v.shape[2])
        if tail:
            end = tokens_to_reach(frames_in(end) - tail)
            if frames_in(end) != frames_in(int(v.shape[2])) - tail:
                raise ValueError(f"hop {h['hop']}: a tail cut of {tail} frames is "
                                 "not a whole number of latent steps")
        kept = v[:, :, k:end]
        vids.append(kept)
        auds.append(a[..., round(drop * FRAME_RESCALE):
                      int(a.shape[-1]) - round(tail * FRAME_RESCALE)])
        steps += int(kept.shape[2])
    return torch.cat(vids, dim=2), torch.cat(auds, dim=-1)


def cond_by_frame(seg):
    """`(f0, f1) -> conditioning`: the conditioning of the hop that holds the
    chunk's middle frame, on the joined run's frame axis."""
    spans, at = [], 0
    for h in seg:
        spans.append((at, at + int(h["frames"]), h["cond"]))
        at += int(h["frames"])

    def pick(f0, f1):
        mid = (int(f0) + int(f1)) // 2
        for a, b, c in spans:
            if a <= mid < b:
                return c
        return spans[-1][2]
    return pick


def match_levels(frames, ref, chunk=16, in_place=False):
    """Each upsampled frame's per-channel mean moved onto the matching frame
    of `ref` (the chain's tone-corrected master at the canvas size), clamped
    to 0..1.

    `tone_compensate` corrects decoded pixels, and the upsampler decodes its
    own. Every mode's correction is (or is dominated by) a per-frame level
    shift, which does not depend on resolution, so a mean match carries it
    over -- and also takes back any level the light re-draw moved. It is not
    a contrast match: lut / gain_bias slopes are not reproduced.
    """
    out = frames if in_place else torch.empty_like(frames)
    for s in range(0, int(frames.shape[0]), chunk):
        up = frames[s:s + chunk].float()
        rf = ref[s:s + chunk].float()
        shift = rf.mean(dim=(1, 2), keepdim=True) - up.mean(dim=(1, 2), keepdim=True)
        out[s:s + chunk] = (up + shift).clamp_(0.0, 1.0).to(frames.dtype)
    return out


def upsample_chain(model, sampler, vae, hops, megapixels, master=None,
                   upscaler=UPSCALER_FILE, decode=None, denoise=DENOISE,
                   alloc=None, guide=None, budget=TOKEN_BUDGET, settings=None,
                   sigmas=None, out_hops=None, ic=None):
    """The recipe over a finished chain. -> (IMAGE [N,H,W,3], note).

    `hops` are per-hop dicts: video/audio (the hop's delivered sampler
    latent), cond, start, drop, frames, hop. `master`, when given, is the
    tone-corrected canvas master to match levels against. `decode` turns a
    video latent into frames (the chain passes its VAE); left None it uses
    `vae.decode`. `alloc(n, h, w)` makes the film buffer (the chain passes
    its master allocator, which spills to disk); left None it is float32 RAM.

    Each run is written into one preallocated film and level-matched in
    place. A 2 MP, 2877 f film is ~72 GB in float32; gathering the runs with
    `torch.cat` and matching into `empty_like` held three of those at once
    and choked a 64 GB machine at the very end of a two-minute render.
    """
    import comfy.nested_tensor as nt
    if decode is None:
        def decode(v):
            img = vae.decode(v)
            return img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
    if alloc is None:
        def alloc(n, h, w):
            return torch.empty((n, h, w, 3), dtype=torch.float32)
    if guide is not None and guide.decode is None:
        guide = guide._replace(decode=decode)
    st = resolve_settings(settings)
    if st["sampler"] is not None:
        sampler = st["sampler"]
    model = _apply_low_vram_patches(model, st["attn_chunks"], st["ffn_chunks"])
    # Unwired settings run the original recipe: one step from sigma 0.75
    # (`upsample_denoise=0.2` under H3's `shift=12` curve), 128 px tile overlap.
    # 3.0.0 ran 2 steps at 0.1 with 192 px; HTCUpsampleSettings can still.
    _chain_steps = int(st["steps"]) if float(denoise) > 0.0 else STEPS
    want = sum(int(h["frames"]) for h in hops)
    frames, at, runs = None, 0, 0
    _segs = segments(hops)
    _cur_budget = budget
    for seg in _segs:
        _t0 = time.perf_counter()
        v, a = stitch(seg)
        lat = {"samples": nt.NestedTensor((v, a))}
        if _cur_budget is not None:
            _LAST_BUDGET[0] = int(_cur_budget)
        _ukw = dict(
            steps=_chain_steps,
            start_sigma=(st["start_sigma"] if float(denoise) > 0.0 else 0.0),
            overlap_px=st["tile_overlap_px"], detail_sigma=st["detail_sigma"],
            chunk_frames=st["chunk_frames"], chunk_overlap=st["chunk_overlap"],
            anchor=st["anchor_strength"], seed=st["seed"], noise=st["noise"],
            attn_chunks=st["attn_chunks"], ffn_chunks=st["ffn_chunks"],
            structure_lock=st["structure_lock"], lock_sigma=st["lock_sigma"],
            attention=st["attention"], chunk_join=st["chunk_join"],
            chunking=st["chunking"], tiling=st["tiling"], lock_anchor=st["lock_anchor"],
            tile_context=st["tile_context"], sharpness=st["sharpness"],
        )
        if sigmas is not None:
            _ukw["sigmas"] = sigmas
        if ic is not None:
            _ukw["ic"] = ic
        out = upsample(model, cond_by_frame(seg), lat, sampler, upscaler,
                       megapixels, denoise, guide=guide, budget=_cur_budget,
                       **_ukw)
        if _cur_budget is not None and _LAST_BUDGET[0] < int(_cur_budget):
            _cur_budget = int(_LAST_BUDGET[0])
        vid = out["samples"].tensors[0]
        if out_hops is not None:
            _t_cur = 0
            for _hi, _h in enumerate(seg):
                _v_orig = _h["video"] if _h["video"].ndim == 5 else _h["video"].unsqueeze(0)
                _a_orig = _h["audio"] if _h["audio"].ndim == 4 else _h["audio"].unsqueeze(0)
                _drop_f = 0 if _hi == 0 else int(_h["drop"])
                _k_drop = tokens_to_reach(_drop_f)
                _tail_f = int(_h.get("tail") or 0)
                _k_end = (tokens_to_reach(frames_in(int(_v_orig.shape[2])) - _tail_f)
                          if _tail_f else int(_v_orig.shape[2]))
                _k_len = _k_end - _k_drop
                _a_drop = round(_drop_f * FRAME_RESCALE)
                _a_end = int(_a_orig.shape[-1]) - round(_tail_f * FRAME_RESCALE)
                out_hops.append({
                    **_h,
                    "start": (_hi == 0),
                    "video": vid[:, :, _t_cur:_t_cur + _k_len].detach().cpu().contiguous(),
                    "audio": _a_orig[..., _a_drop:_a_end].detach().cpu().contiguous(),
                    "drop": 0,
                    "tail": 0,
                })
                _t_cur += _k_len
        _free(model)
        if _device().type == "cuda":
            _free_all_gpu()
        _t1 = time.perf_counter()
        tv_seg = int(vid.shape[2])
        if vae is not None and tv_seg > 32:
            # Stream-decode in 30-token (102-frame) windows with a 5-token (17-frame)
            # causal warm-up prefix so ViT3DDecoder never allocates a full-film
            # 22-44 GB float32 tensor in RAM at once, writing each slice directly
            # into the disk-backed `frames` memmap.
            del out
            for k0 in range(0, tv_seg, 30):
                k1 = min(tv_seg, k0 + 30)
                k_in_start = max(0, k0 - 5) if k0 > 0 else 0
                k_in_end = min(tv_seg, k1 + 2) if k1 < tv_seg else tv_seg
                sub_z = vid[:, :, k_in_start:k_in_end].contiguous()
                part = decode(sub_z).float().cpu()
                del sub_z
                drop_head = 17 if k0 > 0 else 0
                keep_len = frames_in(k1) - frames_in(k0)
                part = part[drop_head:drop_head + keep_len]
                n = int(part.shape[0])
                if at + n > want:
                    raise ValueError(f"the upsampled film is at least {at + n} frames, "
                                     f"the chain delivered {want}")
                if frames is None:
                    frames = alloc(want, int(part.shape[1]), int(part.shape[2]))
                if master is not None:
                    part = match_levels(part, master[at:at + n], in_place=True)
                frames[at:at + n] = part.to(frames.dtype)
                del part
                at += n
            del vid
        else:
            part = decode(vid).float().cpu()
            del out, vid
            n = int(part.shape[0])
            if at + n > want:
                raise ValueError(f"the upsampled film is at least {at + n} frames, "
                                 f"the chain delivered {want}")
            if frames is None:
                frames = alloc(want, int(part.shape[1]), int(part.shape[2]))
            if master is not None:
                part = match_levels(part, master[at:at + n], in_place=True)
            frames[at:at + n] = part.to(frames.dtype)
            del part
            at += n
        runs += 1
        print(f"[{TAG}] run {runs}/{len(_segs)} (hop {seg[0]['hop']}"
              + (f"-{seg[-1]['hop']}" if len(seg) > 1 else "")
              + f"): upsample {_t1 - _t0:.1f} s, decode"
              + (" + level match" if master is not None else "")
              + f" {time.perf_counter() - _t1:.1f} s", flush=True)
    if at != want:
        raise ValueError(f"the upsampled film is {at} frames, "
                         f"the chain delivered {want}")
    note = (f"upsampled {int(frames.shape[2])}x{int(frames.shape[1])} "
            f"({frames.shape[1] * frames.shape[2] / 1048576.0:.2f} MP), "
            f"{runs} run(s)" + (", h3upscale guided" if guide is not None else ""))
    return frames, note


class HTCTiledUpsampler:
    """A finished H3 clip upscaled in latent space and lightly re-drawn in
    tiles, then decoded."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The H3 model the clip was made with."}),
                "conditioning": ("CONDITIONING", {
                    "tooltip": "The conditioning the clip was made with."}),
                "latent": ("LATENT", {"tooltip": "The finished H3 AV latent."}),
                "sampler": ("SAMPLER", {"tooltip": "The sampler for the light re-draw."}),
                "vae": ("VAE", {"tooltip": "The H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "The H3 audio VAE."}),
                "upscale_model": (choices(), {
                    "tooltip": "From models/latent_upscale_models. The recipe uses "
                               f"{UPSCALER_FILE} (3D); {UPSCALER_2D_FILE} (2D) also "
                               "loads and takes sharpness."}),
                "megapixels": ("FLOAT", {
                    "default": MEGAPIXELS, "min": 0.1, "max": 16.0, "step": 0.05,
                    "tooltip": "Output size as area, aspect kept (1 MP = 1024x1024). "
                               "1.5-2 is the usual range."}),
                # appended last: widgets_values are positional
                "sharpness": ("INT", {
                    "default": SHARPNESS, "min": 1, "max": SHARPNESS_MAX, "step": 1,
                    "tooltip": SHARPNESS_TIP}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    FUNCTION = "run"
    CATEGORY = "Hand Tie Clips"
    DESCRIPTION = (
        "Makes a finished MiniMax H3 clip bigger and sharper: the H3 latent "
        "upscaler per time chunk, then one light re-draw step in tiles that are "
        "stepped together, so there are no tile seams. The recipe is fixed; "
        "only the size is a widget.")

    def run(self, model, conditioning, latent, sampler, vae, audio_vae,
            upscale_model, megapixels, sharpness=SHARPNESS):
        import nodes
        from comfy_extras.nodes_audio import vae_decode_audio
        out = upsample(model, conditioning, latent, sampler, str(upscale_model),
                       float(megapixels), sharpness=int(sharpness))
        images = nodes.VAEDecode().decode(vae, out)[0]
        return (images, vae_decode_audio(audio_vae, out))


# --- the chain's upsample settings -------------------------------------------
#
# HandTieClips' upsample_settings socket. Unwired, the chain runs exactly
# these defaults: the original recipe (one simple step, 102 / 17 f chunks,
# anchor 0.999, 70000-token tiles with 128 px overlap, seed 444). 3.0.0 ran
# 2 steps at 0.1 with 192 px; measured 10-25 on one 576p -> 2 MP clip, the
# recipe's single step from 0.75 kept 14% more fine detail.

SETTINGS_TYPE = "HTC_UPSAMPLE"
ATTENTION_TIP = ("sparse (default) = ComfyUI's sparse attention: in side-by-side tests a "
                 "whole run was 28% faster and looked the same. dense = exact attention, "
                 "slower. A ComfyUI without sparse attention runs dense.")
JOIN_TIP = ("How the overlapping pieces of a long clip are joined. stepped (recommended) = all "
            "pieces are re-drawn together and agree at every step, so faces don't change at a "
            "join. crossfade = each piece is re-drawn on its own and the overlaps are blended "
            "(the old way; a face can shift at a join). Same speed.")
CHUNKING_TIP = ("On = a long clip may be cut into overlapping pieces (about 4 s each) when it's "
                "too much for the GPU at once. Off = the whole clip in one go: no joins, but "
                "VRAM grows with length, so very long clips are cut anyway.")
TILING_TIP = ("On = when even a short piece is too much for the GPU, each frame may also be "
              "cut into overlapping tiles. Off = never tile; if VRAM runs out, it stops with a "
              "message saying what to change.")
LOCK_ANCHOR_TIP = ("What structure_lock holds the large shapes to. source = the clip's own latent, "
                   "enlarged by plain interpolation. upscaler = the 3D upscaler's output.")
TILE_CONTEXT_TIP = ("Only matters when a chunk is cut into tiles. none = each tile sees its own "
                    "window. global pass = the chunk is first re-drawn whole at a smaller size, "
                    "and the tiles keep its large shapes.")
CHAIN_DEFAULTS = {
    "steps": STEPS, "start_sigma": 0.0, "noise": "per_chunk", "seed": SEED,
    "chunk_frames": _DEFAULT_CHUNK_FRAMES, "chunk_overlap": CHUNK_OVERLAP,
    "anchor_strength": ANCHOR_AUG, "tile_overlap_px": TILE_OVERLAP_PX, "detail_sigma": 0.0,
    "attn_chunks": 4, "ffn_chunks": 2, "sampler": None,
    "structure_lock": STRUCTURE_LOCK, "lock_sigma": LOCK_SIGMA,
    "attention": "sparse", "chunk_join": "stepped", "chunking": True, "tiling": True,
    "lock_anchor": "source", "tile_context": "none", "redraw_start": 0.6,
    # set by the chain node's upsample_sharpness widget, not the settings node
    "sharpness": SHARPNESS,
}


def resolve_settings(settings):
    """The settings node's dict over the chain defaults. -> a full dict."""
    out = dict(CHAIN_DEFAULTS)
    for k, v in (settings or {}).items():
        if k in out:
            out[k] = v
    if int(out["chunk_overlap"]) >= int(out["chunk_frames"]):
        raise ValueError(f"{TAG}: chunk_overlap ({out['chunk_overlap']}) must be "
                         f"smaller than chunk_frames ({out['chunk_frames']})")
    return out


class HTCUpsampleSettings:
    """Every setting of the chain's upsample, for HandTieClips' upsample_settings."""

    @classmethod
    def INPUT_TYPES(cls):
        d = CHAIN_DEFAULTS
        return {
            "required": {
                "steps": ("INT", {
                    "default": 3, "min": 1, "max": 8,
                    "tooltip": "Re-draw steps. 3 = the recipe. 2 = about a third faster; for "
                               "Regen2K it looked the same in side-by-side tests, for "
                               "upsample_redraw ic it starts later, so less is re-drawn."}),
                "start_sigma": ("FLOAT", {
                    "default": d["start_sigma"], "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": ("Leave at 0 (the Regen2K recipe; redraw_start sets how much "
                                "changes). Above 0: the older recipe, re-drawing from this noise "
                                "level.")}),
                "noise": (["per_chunk", "film"], {
                    "default": "per_chunk",
                    "tooltip": "per_chunk = each piece gets its own noise (Regen2K default). "
                               "film = one noise pattern across the whole film."}),
                # Not named seed: ComfyUI hangs a randomize control on any
                # widget called seed, and the seed must stay fixed.
                "redraw_seed": ("INT", {
                    "default": REGEN2K_SEED, "min": 0, "max": 0x7FFFFFFF,
                    "tooltip": "Noise seed for the re-draw. Same seed, same result."}),
                "chunk_frames": ("INT", {
                    "default": int(d["chunk_frames"]), "min": 34, "max": 1020, "step": 17,
                    "tooltip": "Frames per piece (a multiple of 17). Longer = fewer joins "
                               "but more VRAM. Regen2K never goes under 68."}),
                "chunk_overlap": ("INT", {
                    "default": d["chunk_overlap"], "min": 17, "max": 1003, "step": 17,
                    "tooltip": "Frames two neighbouring pieces share (a multiple of 17, less "
                               "than chunk_frames)."}),
                "anchor_strength": ("FLOAT", {
                    "default": d["anchor_strength"], "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Pre-Regen2K recipe only: how firmly each piece's first frame "
                               "follows the previous piece. 0.999 = H3's default."}),
                "tile_overlap_px": ("INT", {
                    "default": d["tile_overlap_px"], "min": 0, "max": 512, "step": 32,
                    "tooltip": "How much neighbouring tiles overlap, in pixels (128 = normal). "
                               "Only matters when frames are cut into tiles."}),
                "detail_sigma": ("FLOAT", {
                    "default": d["detail_sigma"], "min": 0.0, "max": 32.0, "step": 0.5,
                    "tooltip": "0 = off. Above 0, keeps the upscaler's large shapes and takes "
                               "only fine detail from the re-draw. A fix for tile-grid "
                               "shifts, not a sharpness control."}),
                "attn_chunks": ("INT", {
                    "default": d["attn_chunks"], "min": 1, "max": 16,
                    "tooltip": "VRAM saver; same result either way. Keep 4: lower needs far "
                               "more VRAM and can stall the PC."}),
                "ffn_chunks": ("INT", {
                    "default": d["ffn_chunks"], "min": 1, "max": 16,
                    "tooltip": "VRAM saver; same result either way. Keep 2, or raise it on "
                               "smaller cards."}),
                "attention": (list(ATTENTION_MODES), {
                    "default": d["attention"],
                    "tooltip": ATTENTION_TIP}),
                "chunk_join": (list(JOIN_MODES), {
                    "default": d["chunk_join"],
                    "tooltip": JOIN_TIP}),
                "chunking": ("BOOLEAN", {
                    "default": d["chunking"], "tooltip": CHUNKING_TIP}),
                "tiling": ("BOOLEAN", {
                    "default": d["tiling"], "tooltip": TILING_TIP}),
                "redraw_start": ("FLOAT", {
                    "default": d["redraw_start"], "min": 0.3, "max": 0.95, "step": 0.05,
                    "tooltip": "How much the re-draw is allowed to change. 0.6 (default) "
                               "keeps faces close to the chain's frames; 0.8 = the original "
                               "Regen2K recipe, sharper but faces drift more. Regen2K only."}),
            },
            "optional": {
                "sampler": ("SAMPLER", {
                    "tooltip": "Optional sampler for the re-draw. Unwired = euler, which "
                               "Regen2K is tuned for."}),
            },
        }

    RETURN_TYPES = (SETTINGS_TYPE,)
    RETURN_NAMES = ("upsample_settings",)
    FUNCTION = "build"
    CATEGORY = "Hand Tie Clips"
    DESCRIPTION = ("Settings for HandTieClips' upsample. Wire into its upsample_settings "
                   "socket; unwired, the chain uses these defaults.")

    def build(self, sampler=None, **kw):
        out = dict(kw)
        out["seed"] = out.pop("redraw_seed", REGEN2K_SEED)
        if out.get("chunk_frames") == CHUNK_FRAMES:
            out["chunk_frames"] = _DEFAULT_CHUNK_FRAMES
        out["sampler"] = sampler
        resolve_settings(out)
        return (out,)


NODE_CLASS_MAPPINGS = {"HTCTiledUpsampler": HTCTiledUpsampler,
                       "HTCUpsampleSettings": HTCUpsampleSettings}
NODE_DISPLAY_NAME_MAPPINGS = {"HTCTiledUpsampler": "H3 Tiled Upsampler",
                              "HTCUpsampleSettings": "HTC Upsample Settings"}
