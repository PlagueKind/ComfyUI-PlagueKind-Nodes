"""Port dense MiniMax-H3 AdaLN LoRA weights onto pruned (curve-form) checkpoints.

The problem
-----------
A pruned H3 checkpoint has no time embedder. Where the dense model computes
``adaln_proj.linear(silu(t_emb))`` with ``t_emb`` 2688-wide, the pruned model looks
up a row of a shared ``adaln_t_table [1025, 8]`` and computes
``adaln_proj.linear(table_row)`` with an ``[out, 8]`` weight
(``comfy/ldm/minimax/model.py``: ``AdalnProj``, and the ``use_adaln_curves`` branch
of ``MiniMaxH3Model``).

A LoRA trained on the dense base therefore carries ``A [r, 2688]`` for a weight that
is ``[out, 8]``. ComfyUI builds the delta, fails the ``.reshape(weight.shape)`` in
``comfy/weight_adapter/lora.py``, logs one ``ERROR lora ...`` line per key and skips
it -- 51 keys (50 DiT blocks + final_layer). Generation still runs, but the AdaLN
timestep-modulation half of the LoRA is silently gone, which for a step-distillation
("turbo") LoRA is the half that matters most.

The fix
-------
The pruned table spans the same 1025-point grid as the dense curve
(``t = i/1024``), and the dense curve is an affine function of the table::

    silu_grid[i] = c + V @ table[i]              V: [D, k]   c: [D]

Recover ``c`` and ``V`` with one least-squares solve, then rewrite the LoRA::

    dense:   W @ silu(t_emb)                 = W @ (c + V @ table)
    so:      dW_pruned = scale * B @ (A @ V)     -> [out, k], patches .weight
             db        = scale * B @ (A @ c)     -> [out],    patches .bias

Both land as ordinary ComfyUI ``("diff",)`` patches: no forward hooks, no per-step
cost, nothing for the dynamic-VRAM loader to trip over. The bias term is not
optional -- ``A @ c`` is the DC component of the curve, and dropping it makes the
port near-worthless.

Grid provenance matters
-----------------------
Different pruned builds carry different tables, so the grid must come from a time
embedder that matches. Measured against a hybrid fl2va/ref2va base: a grid derived
from a local non-pruned checkpoint fits at 9.0e-05 relative residual, while a grid
borrowed from an unrelated build bottoms out at 1.7e-03. We therefore derive the
grid from whichever non-pruned H3 checkpoint fits that model table best, and cache
the resulting basis keyed by a hash of the table itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import struct

import torch
import torch.nn.functional as F

log = logging.getLogger("H3AdaLN")

# adaln_t_table row count; t = i / (GRID_ROWS - 1), see MiniMaxH3Model._forward.
GRID_ROWS = 1025
# TimeEmbedder.freq_dim on every H3 config seen so far.
FREQ_DIM = 256
# Where we cache solved bases, under the ComfyUI models/ dir.
BASIS_SUBDIR = "h3_adaln"
# Name/key of a pre-baked grid, if one is dropped in instead.
GRID_FILENAME = "h3_silu_temb_grid.safetensors"
GRID_TENSOR_KEY = "silu_t_emb_grid"

ADALN_WEIGHT_SUFFIX = "adaln_proj.linear.weight"
TE_IN_W = "time_embedder.proj_in.weight"
TE_IN_B = "time_embedder.proj_in.bias"
TE_OUT_W = "time_embedder.proj_out.weight"
TE_OUT_B = "time_embedder.proj_out.bias"


# --------------------------------------------------------------------------- math


def silu_temb_grid(proj_in_w, proj_in_b, proj_out_w, proj_out_b,
                   rows=GRID_ROWS, freq_dim=FREQ_DIM):
    """The dense ``silu(t_emb(t))`` curve sampled on the table grid.

    Mirrors ``TimeEmbedder.forward`` exactly -- fp32 throughout, cos before sin --
    then applies the ``silu`` that ``AdalnProj.forward`` does on the dense path.
    Returns ``[rows, D]``.
    """
    t = torch.arange(rows, dtype=torch.float32) / float(rows - 1)
    half = freq_dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    hidden = F.silu(emb @ proj_in_w.float().T + proj_in_b.float())
    temb = hidden @ proj_out_w.float().T + proj_out_b.float()
    return F.silu(temb)


def solve_basis(table, grid):
    """Least-squares fit of ``grid[i] ~= c + V @ table[i]``.

    Returns ``(c [D], V [D, k], rel_residual)``. The residual is the honest quality
    number for the port -- report it, do not hide it.
    """
    table = table.detach().float().cpu()
    grid = grid.detach().float().cpu()
    if table.shape[0] != grid.shape[0]:
        raise ValueError("table has %d rows, grid has %d"
                         % (table.shape[0], grid.shape[0]))
    ones = torch.ones(table.shape[0], 1, dtype=torch.float32)
    x = torch.cat([ones, table], dim=1)
    sol = torch.linalg.lstsq(x, grid).solution          # [1 + k, D]
    resid = x @ sol - grid
    denom = grid.norm()
    rel = float(resid.norm() / denom) if float(denom) > 0 else float("inf")
    return sol[0].contiguous(), sol[1:].T.contiguous(), rel


def port_delta(lora_b, lora_a, scale, c, basis_v):
    """Dense LoRA factors -> (weight delta, bias delta) in the pruned basis.

    ``lora_a`` is ``[r, D]`` and ``lora_b`` is ``[out, r]``; the returned pair is
    ``[out, k]`` and ``[out]``. Computed in fp32 and left there -- the caller casts.

    From ``W @ silu(t_emb) = W @ (c + V @ table)``: the table-space weight is
    ``W @ V`` and the leftover ``W @ c`` is a constant, i.e. a bias.
    """
    a = lora_a.detach().float()
    b = lora_b.detach().float()
    d_w = (b @ (a @ basis_v)) * scale
    d_b = (b @ (a @ c)) * scale
    return d_w, d_b


def pinv_basis(basis_v):
    """``pinv(V)``: ``[k, D]``, mapping a point on the dense curve back to table
    coordinates. ``V`` has full column rank (it spans the curve), so this is well
    conditioned."""
    return torch.linalg.pinv(basis_v.detach().float())


def port_factors_reverse(lora_a, scale, c, pinv_v):
    """Curve LoRA ``A`` -> its dense-space factor, and the bias direction.

    The mirror of :func:`port_delta`, for a LoRA trained on a pruned checkpoint being
    applied to a full one. Inverting ``silu_grid = c + V @ table`` gives
    ``table = pinv(V) @ (silu - c)``, so the dense-space weight is ``W @ pinv(V)``
    and the constant it drags along, ``-W @ pinv(V) @ c``, comes back off as a bias.

    Returned **factored**, not multiplied out: a dense-space AdaLN delta is
    ``[96768, 2688]``, about 1 GB each and 53 GB across the 51 keys, so it must stay
    low rank and let ComfyUI form the product at patch time like any other LoRA.
    ``lora_a`` is ``[r, k]``; returns ``A' [r, D]`` and ``A' @ c [r]``.
    """
    a_dense = (lora_a.detach().float() @ pinv_v) * scale
    return a_dense, a_dense @ c


def port_delta_reverse(lora_b, lora_a, scale, c, basis_v):
    """Multiplied-out form of :func:`port_factors_reverse`.

    Only for tests and small shapes -- see the size warning above.
    """
    a_dense, a_c = port_factors_reverse(lora_a, scale, c, pinv_basis(basis_v))
    b = lora_b.detach().float()
    return b @ a_dense, -(b @ a_c)


def grid_from_time_embedder(time_embedder, rows=GRID_ROWS):
    """Grid computed from a live (non-pruned) model time embedder, or None.

    Preferred over scanning files when the loaded model is the dense one: it is by
    definition the right curve, with no guessing about which build it came from.
    """
    try:
        proj_in, proj_out = time_embedder.proj_in, time_embedder.proj_out
        tensors = (proj_in.weight, proj_in.bias, proj_out.weight, proj_out.bias)
        if any(t is None or t.device.type == "meta" for t in tensors):
            return None
        freq_dim = int(getattr(time_embedder, "freq_dim", FREQ_DIM))
        with torch.no_grad():   # these are live Parameters; no graph wanted
            return silu_temb_grid(*tensors, rows=rows, freq_dim=freq_dim).clone()
    except Exception:                                    # noqa: BLE001
        log.debug("[H3AdaLN] could not read the live time embedder", exc_info=True)
        return None


def tensor_hash(tensor):
    """Stable id for a tensor, used to key the basis cache."""
    t = tensor.detach().float().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


# The original name, kept because existing cache files are keyed with it.
table_hash = tensor_hash


# ------------------------------------------------------------------ checkpoint io


def _safetensors_header(path):
    """Read just the JSON header of a safetensors file. No tensor data."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            return None
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > (64 << 20):      # sanity: headers are small
            return None
        return json.loads(fh.read(size))


def _time_embedder_prefix(header):
    """Prefix under which a non-pruned H3 time embedder lives, or None.

    Rejects pruned checkpoints (they carry adaln_t_table and no time embedder).
    """
    if not header:
        return None
    if any(k.endswith("adaln_t_table") for k in header):
        return None
    for key in header:
        if key.endswith(TE_OUT_W):
            prefix = key[:-len(TE_OUT_W)]
            if all(prefix + n in header for n in (TE_IN_W, TE_IN_B, TE_OUT_B)):
                return prefix
    return None


def _detach_from_file(tensor):
    """Own the memory.

    ``safe_open`` hands back tensors backed by the file mapping, and ``.float()`` on
    an already-fp32 tensor is a no-op that keeps that backing. Using one after the
    handle closes is a segfault, not an exception, so copy while the file is open.
    """
    return tensor.to(torch.float32).clone()


def _load_time_embedder(path, prefix):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return [_detach_from_file(f.get_tensor(prefix + n))
                for n in (TE_IN_W, TE_IN_B, TE_OUT_W, TE_OUT_B)]


def _grid_from_checkpoint(path, rows):
    """Compute the silu(t_emb) grid from a non-pruned checkpoint, or None."""
    try:
        prefix = _time_embedder_prefix(_safetensors_header(path))
        if prefix is None:
            return None
        return silu_temb_grid(*_load_time_embedder(path, prefix), rows=rows)
    except Exception:                                    # noqa: BLE001
        log.debug("[H3AdaLN] could not derive a grid from %s", path, exc_info=True)
        return None


def _load_prebaked_grid(path):
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
            key = GRID_TENSOR_KEY if GRID_TENSOR_KEY in keys else (
                keys[0] if len(keys) == 1 else None)
            if key is None:
                return None
            return _detach_from_file(f.get_tensor(key))
    except Exception:                                    # noqa: BLE001
        log.debug("[H3AdaLN] could not read grid %s", path, exc_info=True)
        return None


# --------------------------------------------------------------- grid discovery


def _basis_dir():
    import folder_paths
    return os.path.join(folder_paths.models_dir, BASIS_SUBDIR)


def _candidate_grids(rows):
    """Yield ``(label, grid)`` for every usable grid source, best-effort.

    Non-pruned diffusion models first (they give a grid matched to the build the
    pruned checkpoint came from), then any pre-baked grid file lying around.
    """
    try:
        import folder_paths
    except Exception:                                    # noqa: BLE001
        return

    for name in folder_paths.get_filename_list("diffusion_models"):
        if not name.lower().endswith(".safetensors"):
            continue
        path = folder_paths.get_full_path("diffusion_models", name)
        if not path:
            continue
        grid = _grid_from_checkpoint(path, rows)
        if grid is not None:
            yield os.path.basename(name), grid

    seen = set()
    search = [_basis_dir()]
    search += list(folder_paths.get_folder_paths("loras"))
    search += list(folder_paths.get_folder_paths("diffusion_models"))
    packs = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(packs):
        search += [os.path.join(packs, d) for d in os.listdir(packs)
                   if os.path.isdir(os.path.join(packs, d))]
    for directory in search:
        path = os.path.join(directory, GRID_FILENAME)
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        grid = _load_prebaked_grid(path)
        if grid is not None:
            yield "%s (%s)" % (GRID_FILENAME, os.path.basename(directory)), grid


# ------------------------------------------------------------------ basis cache


def _cache_path(key):
    return os.path.join(_basis_dir(), "basis_%s.safetensors" % key)


def _read_cache(key):
    path = _cache_path(key)
    if not os.path.isfile(path):
        return None
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
            return (_detach_from_file(f.get_tensor("c")),
                    _detach_from_file(f.get_tensor("V")),
                    float(meta.get("residual", "nan")),
                    meta.get("source", "unknown") + " (cached)")
    except Exception:                                    # noqa: BLE001
        log.debug("[H3AdaLN] unreadable basis cache %s", path, exc_info=True)
        return None


def _write_cache(key, c, basis_v, residual, source):
    try:
        from safetensors.torch import save_file
        os.makedirs(_basis_dir(), exist_ok=True)
        save_file({"c": c.contiguous(), "V": basis_v.contiguous()},
                  _cache_path(key),
                  metadata={"residual": repr(residual), "source": source,
                            "table_hash": key})
    except Exception:                                    # noqa: BLE001
        log.debug("[H3AdaLN] could not write basis cache", exc_info=True)


def candidate_tables():
    """Yield ``(label, adaln_t_table)`` from every pruned checkpoint on disk.

    Needed for the reverse port: a dense base has no table of its own, so the
    coordinate system the curve LoRA was trained in has to come from a pruned
    checkpoint. Builds differ, so the caller scores them and takes the best fit.
    """
    try:
        import folder_paths
        from safetensors import safe_open
    except Exception:                                    # noqa: BLE001
        return

    for name in folder_paths.get_filename_list("diffusion_models"):
        if not name.lower().endswith(".safetensors"):
            continue
        path = folder_paths.get_full_path("diffusion_models", name)
        if not path:
            continue
        try:
            header = _safetensors_header(path)
            key = next((k for k in (header or ()) if k.endswith("adaln_t_table")),
                       None)
            if key is None:
                continue
            with safe_open(path, framework="pt") as f:
                table = _detach_from_file(f.get_tensor(key))
        except Exception:                                # noqa: BLE001
            log.debug("[H3AdaLN] could not read a table from %s", path, exc_info=True)
            continue
        # yielded outside the `with`, so the caller cannot outlive the mapping
        yield os.path.basename(name), table


def get_basis_for(table, grid, cache_key, use_cache=True, source="explicit"):
    """Solve (or reuse) the basis for one explicit table/grid pair."""
    if use_cache:
        cached = _read_cache(cache_key)
        if cached is not None:
            return cached
    c, basis_v, rel = solve_basis(table, grid)
    if use_cache:
        _write_cache(cache_key, c, basis_v, rel, source)
    return c, basis_v, rel, source


def get_basis_dense(grid, use_cache=True):
    """Best ``(c, V, residual, source)`` for a dense base, or None.

    The grid is known exactly (it is this model own curve); what is unknown is which
    pruned build the LoRA was trained against, so every table on disk is scored and
    the best fit wins. In practice the choice moves the ported delta by ~1 %, which
    is immaterial next to how small the delta already is -- but picking the best fit
    is still free.
    """
    grid_key = tensor_hash(grid)
    best = None
    for label, table in candidate_tables():
        if table.shape[0] != grid.shape[0]:
            continue
        key = "d%s%s" % (grid_key[:8], tensor_hash(table)[:8])
        try:
            c, basis_v, rel, _src = get_basis_for(
                table, grid, key, use_cache=use_cache, source=label)
        except Exception:                                # noqa: BLE001
            log.debug("[H3AdaLN] reverse basis solve failed for %s", label,
                      exc_info=True)
            continue
        log.debug("[H3AdaLN] table candidate %s -> residual %.3e", label, rel)
        if best is None or rel < best[2]:
            best = (c, basis_v, rel, label)
    return best


def get_basis(table, use_cache=True):
    """Best available ``(c, V, residual, source)`` for this table, or None.

    Every candidate grid is scored against the table and the best fit wins, so a
    grid from the wrong build loses to the right one automatically.
    """
    key = table_hash(table)
    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached

    best = None
    for label, grid in _candidate_grids(table.shape[0]):
        try:
            c, basis_v, rel = solve_basis(table, grid)
        except Exception:                                # noqa: BLE001
            log.debug("[H3AdaLN] basis solve failed for %s", label, exc_info=True)
            continue
        log.debug("[H3AdaLN] grid candidate %s -> residual %.3e", label, rel)
        if best is None or rel < best[2]:
            best = (c, basis_v, rel, label)

    if best is not None and use_cache:
        _write_cache(key, best[0], best[1], best[2], best[3])
    return best
