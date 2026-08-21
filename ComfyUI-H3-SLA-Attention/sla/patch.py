"""Wire the block-sparse kernel into MiniMax-H3 attention, at inference time.

The hook is ``transformer_options["optimized_attention_override"]``, read by
``wrap_attn`` in ``comfy/ldm/modules/attention.py``. H3 reaches it from its one
attention call site, ``comfy/ldm/minimax/model.py:172``.

The legacy ``set_model_attn1_patch`` hook does *not* work here: it is the
SD-UNet cross-attention path, and H3 is a DiT that never consults it. A patch
installed that way reports success and silently does nothing -- which is the
failure this whole module exists to avoid, hence the invocation counter below.

Layout at the call site: q/k/v arrive ``[1, 56, S, 128]`` bf16 with
``skip_reshape=True``, ``mask=None``, RoPE already applied. H3 does not pass
``skip_output_reshape``, so we owe it ``[1, S, 7168]`` back.
"""

from __future__ import annotations

import logging

import torch

from .block_map import get_block_map
from .kernel import block_sparse_attention

log = logging.getLogger("H3Utils")

_H3_HEAD_DIM = 128
_OK_DTYPES = (torch.bfloat16, torch.float16)


def _new_state():
    return {
        "calls": 0,        # sparse invocations this run
        "dense": 0,        # fall-throughs this run
        "step": 0,         # logical sampler step, 1-based
        "n_steps": 0,
        "last_step_index": None,
        "summarized": False,
        "seq": 0,
        "kept": 0,
        "blocks": 0,
        "pinned": 0,
        "backend": None,   # what we displaced
        "failed": None,    # first kernel failure, if any
    }


def _reset_run_state(state):
    """Reset per-sampling-run counters while preserving the displaced backend."""
    state["calls"] = 0
    state["dense"] = 0
    state["step"] = 0
    state["n_steps"] = 0
    state["last_step_index"] = None
    state["summarized"] = False
    state["seq"] = 0
    state["kept"] = 0
    state["blocks"] = 0
    state["pinned"] = 0
    state["failed"] = None


def _summarise(state, sparsity, blkq, blkk):
    """One line per sampling run. Never one per block -- there are 50 of those."""
    if state["calls"] == 0:
        log.warning(
            "[H3Utils] SLA: patch installed but never invoked -- attention was "
            "NOT sparsified. (%d dense fall-throughs; check that the model going "
            "into the sampler is the one this node returned.)", state["dense"],
        )
        return
    real = 1.0 - (state["kept"] / state["blocks"]) if state["blocks"] else 0.0
    log.info(
        "[H3Utils] SLA: %d calls | S=%d | blocks %d/%d kept (%.1f%% sparse, "
        "asked %.0f%%) | %d pinned | BLK=%dx%d | %d dense fall-throughs | "
        "displaced %s",
        state["calls"], state["seq"], state["kept"], state["blocks"], real * 100.0,
        sparsity * 100.0, state["pinned"], blkq, blkk, state["dense"],
        state["backend"] or "?",
    )
    if state["failed"] is not None:
        log.warning("[H3Utils] SLA: kernel fell back to dense at least once: %s",
                    state["failed"])


def _make_override(state, sparsity_ratio, blkq, blkk, min_seq_len,
                   protect_audio=True):
    topk_ratio = 1.0 - sparsity_ratio

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense():
            state["dense"] += 1
            return func(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)

        if state["backend"] is None:
            state["backend"] = getattr(func, "__name__", repr(func))

        to = kwargs.get("transformer_options") or {}

        # Anything that is not the packed H3 self-attention goes straight
        # through. The min_seq_len guard is what keeps the 2-block token refiner
        # (S = text length, a few hundred) and lower-resolution runs on dense
        # attention, where block sparsity would cost more than it saves.
        if (
            not skip_reshape
            or mask is not None
            or q.ndim != 4
            or q.shape[-1] != _H3_HEAD_DIM
            or q.dtype not in _OK_DTYPES
            or q.shape[2] < min_seq_len
            or to.get("_h3sla_dense", False)
        ):
            return dense()

        try:
            B, H, S, D = q.shape

            # [1, H, S, D] -> [1, S, H, D]. H3 builds q/k/v as [S, H, D] and
            # transposes for the call, so this transposes back onto the original
            # memory: contiguous already, and the copy is a no-op. A BHSD kernel
            # would instead cost a real ~1.3 GB copy per tensor at 768p/15s.
            qb, kb, vb = (t.transpose(1, 2) for t in (q, k, v))
            if not qb.is_contiguous():
                qb, kb, vb = qb.contiguous(), kb.contiguous(), vb.contiguous()

            # Pin the [text | cond | audio] prefix into every query's
            # selection. Audio is ~1% of the packed sequence, so plain top-k
            # routinely drops all of it and the soundtrack degrades while the
            # video stays fine. 0 when the layout is unavailable, which simply
            # disables the protection rather than guessing.
            prefix = int(to.get("_h3sla_prefix", 0) or 0) if protect_audio else 0
            if prefix >= S:
                prefix = 0

            lut, topk = get_block_map(qb, kb, topk_ratio, blkq, blkk,
                                      protect_upto=prefix)
            out = block_sparse_attention(qb, kb, vb, lut, topk, blkq, blkk)

            state["calls"] += 1
            state["seq"] = S
            state["kept"] = topk
            state["blocks"] = (S + blkk - 1) // blkk
            state["pinned"] = (prefix + blkk - 1) // blkk

            # [1, S, H, D] -> what the caller expects
            if skip_output_reshape:
                return out.transpose(1, 2)
            return out.reshape(B, S, H * D)

        except Exception as exc:  # noqa: BLE001 - a bad kernel must not kill the run
            if state["failed"] is None:
                state["failed"] = "%s: %s" % (exc.__class__.__name__, exc)
                log.debug("[H3Utils] SLA kernel failed", exc_info=True)
            return dense()

    return override


def _call_next_wrapper(executor, *args, **kwargs):
    """Advance the ComfyUI wrapper chain instead of jumping to the base model.

    Current ComfyUI ``WrapperExecutor`` instances are callable; calling them
    advances to the next registered wrapper. Calling ``executor.original`` here
    would bypass every wrapper installed after SLA (for example Spectrum's H3
    instrumentation), which makes those patches silently lose the native model
    call. The type fallback only preserves the tiny executor shims used by this
    package's historical unit tests and older compatibility harnesses.
    """
    if not isinstance(executor, type) and callable(executor):
        return executor(*args, **kwargs)
    return executor.original(*args, **kwargs)


def _sequence_scalar(value):
    """Return the first scalar from a Python sequence, or None when unavailable."""
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return float(value[0])
    return None


def _resolve_sampler_step(transformer_options):
    """Resolve the logical sampler step from ComfyUI's current/raw sigmas.

    Spectrum can forecast a sampler step without invoking the diffusion model at
    all. Counting wrapper invocations therefore counts *actual NFEs*, not sampler
    steps. ComfyUI exposes both the complete schedule (``sample_sigmas``) and the
    current raw sigma (``sigmas``), which lets SLA recover the real sampler
    position even when some intermediate model calls were skipped.
    """
    sample_sigmas = transformer_options.get("sample_sigmas")
    current_sigmas = transformer_options.get("sigmas")
    if sample_sigmas is None or current_sigmas is None:
        return None

    try:
        n_steps = len(sample_sigmas) - 1
    except TypeError:
        return None
    if n_steps < 1:
        return None

    # Keep dependency-free tests cheap and avoid requiring a Torch stub that
    # implements tensor reductions.
    if isinstance(sample_sigmas, (list, tuple)):
        current = _sequence_scalar(current_sigmas)
        if current is None:
            try:
                current = float(current_sigmas)
            except (TypeError, ValueError):
                return None
        try:
            values = [float(v) for v in sample_sigmas[:-1]]
        except (TypeError, ValueError):
            return None
        if not values:
            return None
        step_index = min(range(len(values)), key=lambda i: abs(values[i] - current))
        return step_index, n_steps

    try:
        schedule = sample_sigmas.reshape(-1)
        current = current_sigmas.reshape(-1)
        if schedule.numel() < 2 or current.numel() == 0:
            return None
        current_value = current[0].to(device=schedule.device, dtype=schedule.dtype)
        step_index = int(torch.argmin(torch.abs(schedule[:-1] - current_value)).item())
        return step_index, int(schedule.numel()) - 1
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _prepare_run_state(state, transformer_options, sparsity_ratio, blkq, blkk):
    """Synchronize SLA's run state with the real sampler schedule.

    Returns ``n_steps``. When current sigma metadata is unavailable, this falls
    back to the historical invocation counter so non-standard callers remain
    compatible.
    """
    resolved = _resolve_sampler_step(transformer_options)
    if resolved is None:
        n_steps = max(1, len(transformer_options.get("sample_sigmas", [])) - 1)
        if state["step"] >= n_steps:
            if not state["summarized"] and (state["calls"] or state["dense"]):
                _summarise(state, sparsity_ratio, blkq, blkk)
            _reset_run_state(state)
        state["n_steps"] = n_steps
        state["step"] += 1
        return n_steps

    step_index, n_steps = resolved
    last_step_index = state["last_step_index"]
    new_run = (
        (state["n_steps"] not in (0, n_steps))
        or (
            last_step_index is not None
            and (
                step_index < last_step_index
                or (state["summarized"] and step_index <= last_step_index)
            )
        )
    )

    if new_run:
        if not state["summarized"] and (state["calls"] or state["dense"]):
            _summarise(state, sparsity_ratio, blkq, blkk)
        _reset_run_state(state)

    state["n_steps"] = n_steps
    state["step"] = step_index + 1
    state["last_step_index"] = step_index
    return n_steps


def _make_wrapper(state, sparsity_ratio, blkq, blkk, dense_last_steps):
    """DIFFUSION_MODEL wrapper: per-step state, and the end-of-run summary.

    Registered once and then reused -- ComfyUI caches node outputs, so this
    closure outlives a single sampling run. Spectrum may skip diffusion-model
    calls on forecasted steps, therefore SLA derives its logical step from
    ``sample_sigmas`` + current ``sigmas`` instead of counting wrapper calls.
    """

    def wrapper(executor, x, timestep, context, transformer_options={},
                minimax_payload=None, **kwargs):
        to = transformer_options
        n_steps = _prepare_run_state(
            state, to, sparsity_ratio, blkq, blkk
        )

        # PackedLayout.segments is [(start, stop, kind), ...] over
        # [text | cond/ref | audio | video]; the video start is therefore the
        # length of everything that must stay exactly attended. It lives on the
        # payload, which never reaches the attention call site, so the wrapper
        # is the only place it can be picked up.
        prefix = 0
        layout = minimax_payload.get("layout") if minimax_payload else None
        for seg in getattr(layout, "segments", ()) or ():
            if len(seg) == 3 and seg[2] == "video":
                prefix = int(seg[0])
                break
        to["_h3sla_prefix"] = prefix

        to["_h3sla_dense"] = bool(
            dense_last_steps > 0 and state["step"] > n_steps - dense_last_steps
        )

        # Forward minimax_payload only when H3 actually supplied one. Nothing
        # stops a user wiring this node to a non-H3 model, and every other
        # diffusion model would raise TypeError on the unexpected kwarg -- a
        # crash mid-sampling rather than the graceful no-op they should get.
        if minimax_payload is not None:
            kwargs["minimax_payload"] = minimax_payload
        out = _call_next_wrapper(
            executor,
            x,
            timestep,
            context,
            transformer_options=transformer_options,
            **kwargs,
        )

        if state["step"] >= n_steps and not state["summarized"]:
            _summarise(state, sparsity_ratio, blkq, blkk)
            state["summarized"] = True
        return out

    return wrapper


def patch_h3_sla(model, sparsity_ratio=0.90, block_size=64, min_seq_len=8192,
                 dense_last_steps=0, protect_audio=True):
    """Return a clone of ``model`` whose H3 self-attention runs block-sparse.

    Weights are untouched; this only installs an attention override and a
    per-step wrapper on the clone.
    """
    blkq = int(block_size)
    # BLKK=64 is not a typo. On sm_120 the 128x128 tile needs 160 KB of shared
    # memory against a ~99 KB limit and cannot launch at all; 128x64 both fits
    # and measured fastest. LightX2V picks the same split for its sage2 path on
    # non-sm90 architectures.
    blkk = 64 if blkq == 128 else blkq

    state = _new_state()
    patched = model.clone()

    to = patched.model_options.get("transformer_options", {}).copy()
    to["optimized_attention_override"] = _make_override(
        state, float(sparsity_ratio), blkq, blkk, int(min_seq_len),
        bool(protect_audio))
    patched.model_options["transformer_options"] = to

    patched.add_wrapper_with_key(
        "diffusion_model", "h3_sla_state",
        _make_wrapper(state, float(sparsity_ratio), blkq, blkk,
                      int(dense_last_steps)),
    )

    log.info(
        "[H3Utils] SLA installed | sparsity=%.2f | BLK=%dx%d | min_seq_len=%d | "
        "dense_last_steps=%d | protect_audio=%s",
        sparsity_ratio, blkq, blkk, min_seq_len, dense_last_steps,
        protect_audio,
    )
    return patched
