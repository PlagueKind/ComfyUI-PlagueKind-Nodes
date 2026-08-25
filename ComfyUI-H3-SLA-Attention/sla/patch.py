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

import importlib
import logging

import torch

from .block_map import get_block_map
from .kernel import block_sparse_attention

log = logging.getLogger("H3Utils")

_H3_HEAD_DIM = 128
_OK_DTYPES = (torch.bfloat16, torch.float16)

# Dense-path backend candidates, tried in order. "pytorch" is a core
# ComfyUI name. "comfy_kitchen" is ComfyUI's own "Comfy Kitchen" int8 backend
# (--use-ck-attention), confirmed against comfy/ldm/modules/attention.py; the
# extra entries are kept as harmless fallbacks in case a future version
# renames it. "sage:*" modes are handled separately by _build_sage_dense_fn,
# below, since each one needs a specific pv_accum_dtype, not just a name
# lookup.
_BACKEND_CANDIDATES = {
    "pytorch": (("comfy.ldm.modules.attention", "attention_pytorch"),),
    "comfy_kitchen": (
        ("comfy.ldm.modules.attention", "attention_comfy_kitchen_int8"),
        ("comfy.ldm.modules.attention", "attention_ck"),
        ("comfy.ldm.modules.attention", "attention_composable_kernel"),
    ),
}

# Every mode kijai/ComfyUI-KJNodes' PatchSageAttentionKJ node offers, kept in
# the same order and named to match its dropdown so the two are recognizable
# side by side. Each maps to the exact sageattention kernel + pv_accum_dtype
# KJNodes' _patch_modules uses for that mode -- pv_accum_dtype is not cosmetic,
# it is the single biggest speed/quality knob within sage (fp32 safest, fp16
# fastest and most prone to overflow-driven artifacts), so silently picking
# one for the user would defeat the point of exposing the modes at all.
_SAGE_KERNELS = {
    "qk_int8_pv_fp16_cuda": ("sageattn_qk_int8_pv_fp16_cuda", "fp32"),
    "qk_int8_pv_fp16_triton": ("sageattn_qk_int8_pv_fp16_triton", None),
    "qk_int8_pv_fp8_cuda": ("sageattn_qk_int8_pv_fp8_cuda", "fp32+fp32"),
    "qk_int8_pv_fp8_cuda++": ("sageattn_qk_int8_pv_fp8_cuda", "fp32+fp16"),
}
SAGE_MODES = ("sage:auto",) + tuple("sage:" + k for k in _SAGE_KERNELS)

_backend_cache = {}


def _build_sage_dense_fn(mode):
    """Build a dense-path callable for one ``sage:<kernel>`` mode.

    Calls the ``sageattention`` package directly -- the same package KJNodes'
    PatchSageAttentionKJ wraps -- rather than depending on KJNodes being
    installed, since its node isn't a stable importable API. The reshape /
    tensor_layout / mask handling below mirrors KJNodes' own ``attention_sage``
    exactly, since that is what makes swapping this in a drop-in replacement
    for what that node does globally, just scoped to this model's dense steps.
    Returns None (with a logged reason) if ``sageattention`` isn't installed
    or the specific kernel this mode needs isn't in it.
    """
    kernel = mode.split(":", 1)[1] if ":" in mode else mode

    try:
        import sageattention as _sa
    except ImportError:
        log.warning(
            "[H3Utils] SLA: dense_backend=%r needs the 'sageattention' "
            "package (pip install sageattention), which is not importable "
            "here.", mode)
        return None

    if kernel == "auto":
        def call_sage(q, k, v, is_causal, attn_mask, tensor_layout):
            return _sa.sageattn(q, k, v, is_causal=is_causal,
                                attn_mask=attn_mask, tensor_layout=tensor_layout)
    else:
        spec = _SAGE_KERNELS.get(kernel)
        if spec is None:
            log.warning("[H3Utils] SLA: unrecognized sage mode %r.", mode)
            return None
        fn_name, pv_accum_dtype = spec
        fn = getattr(_sa, fn_name, None)
        if fn is None:
            log.warning(
                "[H3Utils] SLA: dense_backend=%r needs sageattention.%s, "
                "which this install's sageattention package does not have "
                "(older/newer version?).", mode, fn_name)
            return None
        if pv_accum_dtype is None:
            def call_sage(q, k, v, is_causal, attn_mask, tensor_layout, _fn=fn):
                return _fn(q, k, v, is_causal=is_causal, attn_mask=attn_mask,
                          tensor_layout=tensor_layout)
        else:
            def call_sage(q, k, v, is_causal, attn_mask, tensor_layout,
                         _fn=fn, _pv=pv_accum_dtype):
                return _fn(q, k, v, is_causal=is_causal, attn_mask=attn_mask,
                          pv_accum_dtype=_pv, tensor_layout=tensor_layout)

    def attention_sage_dense(q, k, v, heads, mask=None, attn_precision=None,
                             skip_reshape=False, skip_output_reshape=False,
                             **kwargs):
        # H3's call site always uses skip_reshape=True (see module docstring
        # above), but both branches are handled for robustness against any
        # other model this node might get wired to.
        if skip_reshape:
            b, _, _, dim_head = q.shape
            tensor_layout = "HND"
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
            tensor_layout = "NHD"

        attn_mask = mask
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)

        out = call_sage(q, k, v, False, attn_mask, tensor_layout)

        if tensor_layout == "HND":
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        else:
            if skip_output_reshape:
                out = out.transpose(1, 2)
            else:
                out = out.reshape(b, -1, heads * dim_head)
        return out

    attention_sage_dense.__name__ = "attention_" + mode.replace(":", "_").replace("+", "p")
    return attention_sage_dense


def _resolve_backend(name):
    """Look up a specific dense-path attention function by name.

    This deliberately does NOT go through ``optimized_attention`` /
    ``func`` -- that resolves whatever the launch flags or an attention
    override node currently have active (e.g. ``--use-ck-attention``), which
    is exactly what a pinned ``dense_backend`` needs to bypass. Returns None
    for \"auto\" (meaning: use whatever ``func`` is, the old behaviour) or when
    nothing matching was found on this ComfyUI install.
    """
    if name in (None, "auto"):
        return None
    if name in _backend_cache:
        return _backend_cache[name]

    if name in SAGE_MODES:
        fn = _build_sage_dense_fn(name)
        _backend_cache[name] = fn
        return fn

    fn = None
    for module_name, attr in _BACKEND_CANDIDATES.get(name, ()):
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, attr, None)
            if fn is not None:
                break
        except Exception:  # noqa: BLE001 - a bad candidate must not crash
            continue
    if fn is None:
        log.warning(
            "[H3Utils] SLA: could not resolve dense_backend=%r on this "
            "ComfyUI install; dense steps will use whatever backend the "
            "environment already has active instead.", name)
    _backend_cache[name] = fn
    return fn


def _parse_step_spec(spec):
    """Parse a user-facing step spec ("0,2,4-6") into a frozenset of 0-based
    step indices. Blank input returns an empty set. A bad token is skipped
    with a warning rather than raising -- a typo here must not kill the run.
    """
    steps = set()
    if not spec:
        return frozenset()
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token[1:]:  # skip a leading '-' so "-1" isn't a range
                start_s, end_s = token.split("-", 1)
                start, end = int(start_s), int(end_s)
                steps.update(range(min(start, end), max(start, end) + 1))
            else:
                steps.add(int(token))
        except ValueError:
            log.warning("[H3Utils] SLA: ignoring unparsable dense_steps "
                        "token %r", token)
    return frozenset(steps)


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
        "dense_backend": None,  # what dense steps actually ran on
        "failed": None,    # first kernel failure, if any
        "fp16_saved": None,     # (fp16, bf16) matmul reduction flags to restore
        "call_idx": 0,       # this step's call count, for stabilize_motion
        "prev_lut": {},      # call_idx -> last sparse lut, for stabilize_motion
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
    # Catches the case the end-of-run clear (in the wrapper) can't: a
    # generation that got cancelled mid-run never reaches that clear, so
    # stabilize_motion's held tensors from the abandoned run would otherwise
    # sit in VRAM until this next-run detection fires here.
    state["prev_lut"].clear()


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
        "asked %.0f%%) | %d pinned | BLK=%dx%d | %d dense fall-throughs "
        "(on %s) | displaced %s",
        state["calls"], state["seq"], state["kept"], state["blocks"], real * 100.0,
        sparsity * 100.0, state["pinned"], blkq, blkk, state["dense"],
        state["dense_backend"] or "?", state["backend"] or "?",
    )
    if state["failed"] is not None:
        log.warning("[H3Utils] SLA: kernel fell back to dense at least once: %s",
                    state["failed"])


def _make_override(state, sparsity_ratio, blkq, blkk, min_seq_len,
                   protect_audio=True, dense_fn=None, stabilize_motion=False):
    """``dense_fn``, when not None, is a specific backend (e.g. always
    ``attention_pytorch``) that every dense fall-through uses instead of
    ``func``. Without it, ``func`` is whatever ``optimized_attention``
    currently resolves to -- which can be ck-attention under
    ``--use-ck-attention`` or an attention-override node, and running the
    dense steps on ck measurably loses quality versus pytorch. Pinning
    ``dense_fn`` makes the dense path deterministic regardless of what else
    is set globally; the sparse path is unaffected either way, since it
    never calls ``func`` at all.

    ``stabilize_motion`` carries each layer's previously-selected key blocks
    into ``get_block_map`` as a tie-breaker (see block_map.py); it costs one
    small dict lookup and store per call, nothing else. ``state["call_idx"]``
    is the layer identity this relies on -- it counts calls within the
    current step, reset to 0 by the wrapper at the top of every step, and it
    works because the model graph is static: the Nth attention call happens
    in the same layer every step.
    """
    topk_ratio = 1.0 - sparsity_ratio

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense():
            state["dense"] += 1
            call = dense_fn or func
            if state["dense_backend"] is None:
                state["dense_backend"] = getattr(call, "__name__", repr(call))
            return call(q, k, v, heads, mask=mask, attn_precision=attn_precision,
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

            call_idx = state["call_idx"]
            state["call_idx"] = call_idx + 1
            prev_lut = state["prev_lut"].get(call_idx) if stabilize_motion else None

            if stabilize_motion:
                lut, topk, history = get_block_map(
                    qb, kb, topk_ratio, blkq, blkk,
                    protect_upto=prefix, prev_lut=prev_lut, return_history=True,
                )
                # Only the bounded boundary slice is retained -- see
                # block_map.py. Storing the full lut here is what made this
                # grow to several GB per run at 768p with stabilize_motion on.
                state["prev_lut"][call_idx] = history
            else:
                lut, topk = get_block_map(qb, kb, topk_ratio, blkq, blkk,
                                          protect_upto=prefix, prev_lut=prev_lut)
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


def _set_fp16_accum(enabled):
    """Toggle the fast fp16/bf16 matmul-reduction path (what ``--fast
    fp16_accumulation`` turns on). Missing on older torch builds, hence the
    hasattr guards -- absence just means there is nothing to disable.
    """
    backend = torch.backends.cuda.matmul
    for attr in ("allow_fp16_reduced_precision_reduction",
                 "allow_bf16_reduced_precision_reduction"):
        if hasattr(backend, attr):
            setattr(backend, attr, enabled)


def _make_wrapper(state, sparsity_ratio, blkq, blkk, dense_last_steps,
                  dense_steps=frozenset(), disable_fp16_accum=False):
    """DIFFUSION_MODEL wrapper: per-step state, and the end-of-run summary.

    Registered once and then reused -- ComfyUI caches node outputs, so this
    closure outlives a single sampling run. Spectrum may skip diffusion-model
    calls on forecasted steps, therefore SLA derives its logical step from
    ``sample_sigmas`` + current ``sigmas`` instead of counting wrapper calls.

    ``dense_steps`` is a set of explicit 0-based step indices to force dense
    on top of the ``dense_last_steps`` tail -- useful for keeping early
    steps, which set global composition and prompt adherence, off the sparse
    path without paying for full-attention on every step.
    """

    def wrapper(executor, x, timestep, context, transformer_options={},
                minimax_payload=None, **kwargs):
        to = transformer_options
        n_steps = _prepare_run_state(
            state, to, sparsity_ratio, blkq, blkk
        )
        # Reset every step (real call, not nominal step -- Spectrum can skip
        # some), since call_idx identifies a layer by its position within one
        # step's sequence of override() calls, not across the whole run.
        state["call_idx"] = 0

        if disable_fp16_accum:
            if state["fp16_saved"] is None:
                backend = torch.backends.cuda.matmul
                state["fp16_saved"] = (
                    getattr(backend, "allow_fp16_reduced_precision_reduction", None),
                    getattr(backend, "allow_bf16_reduced_precision_reduction", None),
                )
            # Forced every call, not just once: some other node or a
            # concurrent model on the same process could flip these back on
            # between steps, and the whole point is that H3 never sees the
            # reduced-precision reduction path while this model is patched.
            _set_fp16_accum(False)

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

        step0 = state["step"] - 1  # 0-based, for dense_steps membership
        to["_h3sla_dense"] = bool(
            (dense_last_steps > 0 and state["step"] > n_steps - dense_last_steps)
            or step0 in dense_steps
        )

        # Forward minimax_payload only when H3 actually supplied one. Nothing
        # stops a user wiring this node to a non-H3 model, and every other
        # diffusion model would raise TypeError on the unexpected kwarg -- a
        # crash mid-sampling rather than the graceful no-op they should get.
        if minimax_payload is not None:
            kwargs["minimax_payload"] = minimax_payload
        try:
            out = _call_next_wrapper(
                executor,
                x,
                timestep,
                context,
                transformer_options=transformer_options,
                **kwargs,
            )
        except Exception:
            # This run is dead regardless of cause -- most commonly OOM.
            # stabilize_motion's held tensors must not survive to poison
            # the *next* attempt with the same already-scarce VRAM.
            # _reset_run_state only clears them lazily, once a new run's
            # first step is detected -- which never happens if the caller
            # does not retry with this same patched clone. Cleared here,
            # unconditionally, before the exception reaches ComfyUI.
            state["prev_lut"].clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        if state["step"] >= n_steps and not state["summarized"]:
            _summarise(state, sparsity_ratio, blkq, blkk)
            state["summarized"] = True
            # stabilize_motion's prev_lut only needs to survive step-to-step
            # within this run -- the whole point is comparing against *last
            # step*, not some earlier generation. Left unset it would sit on
            # its held tensors (one per layer, bounded, but real: same order
            # as a large activation) in VRAM for as long as this patched
            # model stays cached, doing nothing between separate runs.
            state["prev_lut"].clear()
            if disable_fp16_accum and state["fp16_saved"] is not None:
                orig_fp16, orig_bf16 = state["fp16_saved"]
                backend = torch.backends.cuda.matmul
                if orig_fp16 is not None:
                    backend.allow_fp16_reduced_precision_reduction = orig_fp16
                if orig_bf16 is not None:
                    backend.allow_bf16_reduced_precision_reduction = orig_bf16
                state["fp16_saved"] = None
        return out

    return wrapper


def patch_h3_sla(model, sparsity_ratio=0.90, block_size=64, min_seq_len=8192,
                 dense_last_steps=0, protect_audio=True, dense_backend="comfy_kitchen",
                 dense_steps="", disable_fp16_accum=True, stabilize_motion=False):
    """Return a clone of ``model`` whose H3 self-attention runs block-sparse.

    Weights are untouched; this only installs an attention override and a
    per-step wrapper on the clone.

    ``dense_backend`` pins every dense fall-through (short sequences,
    ``dense_last_steps``, and explicit ``dense_steps``) to a specific
    attention kernel -- default "comfy_kitchen" (Comfy Kitchen int8), since it's fast
    enough on the handful of dense steps this node runs to be worth its
    precision tradeoff there. "pytorch" pins the plain reference kernel
    instead if you want zero quantization anywhere in the dense path, at
    real cost to dense-step speed. "auto" restores the old behaviour of
    calling whatever the environment already resolved (e.g.
    ``--use-ck-attention`` or an attention-override node) -- note this can
    silently differ run to run if that global setting changes.

    ``disable_fp16_accum`` forces off the fp16/bf16 reduced-precision matmul
    reduction (what ``--fast fp16_accumulation`` enables) for the duration of
    this model's sampling run, regardless of the global launch flag --
    measured to cost quality on H3 for no throughput gain.

    ``stabilize_motion`` biases block selection toward what each layer picked
    last step (see block_map.py's ``prev_lut``), to cut down on a block
    flipping between two near-tied candidates step to step for no reason tied
    to actual content -- visible on fast motion as a faint double-exposure.
    Off by default: it's a real fix for that specific symptom, not a general
    quality dial, and adds a small amount of state to carry between steps.
    """
    blkq = int(block_size)
    # BLKK=64 is not a typo. On sm_120 the 128x128 tile needs 160 KB of shared
    # memory against a ~99 KB limit and cannot launch at all; 128x64 both fits
    # and measured fastest. LightX2V picks the same split for its sage2 path on
    # non-sm90 architectures.
    blkk = 64 if blkq == 128 else blkq

    dense_fn = _resolve_backend(dense_backend)
    dense_step_set = _parse_step_spec(dense_steps)

    state = _new_state()
    patched = model.clone()

    to = patched.model_options.get("transformer_options", {}).copy()
    to["optimized_attention_override"] = _make_override(
        state, float(sparsity_ratio), blkq, blkk, int(min_seq_len),
        bool(protect_audio), dense_fn=dense_fn,
        stabilize_motion=bool(stabilize_motion))
    patched.model_options["transformer_options"] = to

    patched.add_wrapper_with_key(
        "diffusion_model", "h3_sla_state",
        _make_wrapper(state, float(sparsity_ratio), blkq, blkk,
                      int(dense_last_steps), dense_steps=dense_step_set,
                      disable_fp16_accum=bool(disable_fp16_accum)),
    )

    log.info(
        "[H3Utils] SLA installed | sparsity=%.2f | BLK=%dx%d | min_seq_len=%d | "
        "dense_last_steps=%d | dense_steps=%s | dense_backend=%s | "
        "protect_audio=%s | disable_fp16_accum=%s | stabilize_motion=%s",
        sparsity_ratio, blkq, blkk, min_seq_len, dense_last_steps,
        sorted(dense_step_set) or "-", dense_backend, protect_audio,
        disable_fp16_accum, stabilize_motion,
    )
    return patched
