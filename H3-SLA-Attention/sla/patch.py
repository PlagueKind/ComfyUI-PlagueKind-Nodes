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

from block_map import get_block_map
from kernel import block_sparse_attention

log = logging.getLogger("H3Utils")

_H3_HEAD_DIM = 128
_OK_DTYPES = (torch.bfloat16, torch.float16)


def _new_state():
    return {
        "calls": 0,        # sparse invocations this run
        "dense": 0,        # fall-throughs this run
        "step": 0,
        "seq": 0,
        "kept": 0,
        "blocks": 0,
        "pinned": 0,
        "backend": None,   # what we displaced
        "failed": None,    # first kernel failure, if any
    }


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


def _make_wrapper(state, sparsity_ratio, blkq, blkk, dense_last_steps):
    """DIFFUSION_MODEL wrapper: per-step state, and the end-of-run summary.

    Registered once and then reused -- ComfyUI caches node outputs, so this
    closure outlives a single sampling run. The step counter therefore has to
    reset itself, or every run after the first drifts permanently into the
    trailing-dense window and silently stops sparsifying.
    """

    def wrapper(executor, x, timestep, context, transformer_options={},
                minimax_payload=None, **kwargs):
        to = transformer_options
        n_steps = max(1, len(to.get("sample_sigmas", [])) - 1)

        if state["step"] >= n_steps:      # new run
            state["step"] = 0
            state["calls"] = 0
            state["dense"] = 0
            state["failed"] = None
        state["step"] += 1

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
        out = executor.original(x, timestep, context,
                                transformer_options=transformer_options,
                                **kwargs)

        if state["step"] >= n_steps:
            _summarise(state, sparsity_ratio, blkq, blkk)
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
