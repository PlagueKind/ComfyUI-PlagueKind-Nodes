"""Block-sparse attention forward kernel for MiniMax-H3.

Vendored and reduced from LightX2V (Apache-2.0):
  lightx2v/common/ops/attn/kernels/sla_kernel_ar.py  -- ``_attn_fwd``
  https://github.com/ModelTC/LightX2V

Only the forward pass survives the port; sampling runs under ``no_grad`` so the
backward half and the ``autograd.Function`` wrapper are dead weight here.

The ``_ar`` name upstream refers to the models it was written for, not to any
causal structure: the kernel walks whatever key blocks the lookup table names
and applies no triangular mask. That is what makes it usable for H3, whose
packed ``[text | cond/ref | audio | video]`` sequence is fully bidirectional.

Layout is BLHD -- ``(B, L, H, D)`` -- which is deliberate. H3 materialises q/k/v
as ``[S, H, D]`` and only then transposes to ``[1, H, S, D]`` for the attention
call, so transposing back is free, while a BHSD kernel would force a real
``.contiguous()`` copy costing ~1.3 GB per tensor at 768p/15s.

Two changes against upstream, both marked FIX below: masked loads now pass
``other=0.0``, because Triton leaves masked lanes undefined and ``0 * NaN`` is
NaN, not zero -- the sequence-tail block can poison a whole row otherwise.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

log = logging.getLogger("H3Utils")


@triton.jit
def _quantize_kernel(
    X,
    X_I8,
    X_SCALE,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Per-token symmetric int8 quantization, one token-tile per program.

    Same grid/offset pattern as block_map.py's ``_compress_kernel``: one read
    of X, one write each to X_I8 and X_SCALE, entirely in registers. This is
    the fix for doing it as chained eager torch ops (``.float()``, ``.round()``,
    ``.clamp()``, ...) instead -- each of those is a separate CUDA kernel that
    reads the whole tensor from global memory and writes a whole new one back
    out. For a tensor this size, ~6-7 unfused passes plus a full fp32-sized
    intermediate (from ``.float()`` on a bf16 tensor) is both slower and more
    memory-hungry than the bf16 baseline this was supposed to improve on.
    """
    idx_t = tl.program_id(0)
    idx_bh = tl.program_id(1)

    idx_b = idx_bh // H
    idx_h = idx_bh - idx_b * H

    offs_t = idx_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    t_mask = offs_t < L

    x_offset = idx_b * L * H * D + idx_h * D
    x = tl.load(
        X + x_offset + offs_t[:, None] * (H * D) + offs_d[None, :],
        mask=t_mask[:, None], other=0.0,
    )

    amax = tl.max(tl.abs(x), axis=1)
    scale = tl.where(amax > 0, amax / 127.0, 1.0)
    xi = (x / scale[:, None] + tl.where(x >= 0, 0.5, -0.5)).to(tl.int32)
    xi = tl.minimum(tl.maximum(xi, -127), 127)

    tl.store(
        X_I8 + x_offset + offs_t[:, None] * (H * D) + offs_d[None, :],
        xi.to(tl.int8), mask=t_mask[:, None],
    )
    tl.store(X_SCALE + idx_b * L * H + offs_t * H + idx_h, scale, mask=t_mask)


def quantize_per_token_int8(x, BLOCK_T=128):
    """(B, L, H, D) -> (int8 same shape, fp32 scale (B, L, H)).

    One fused kernel launch, one pass over ``x``. See ``_quantize_kernel``
    above for why this replaced an earlier eager-torch-ops version.

    Correct for Q and K, where the scale's own index (query/key TOKEN) is
    NOT the dimension QK^T contracts over (D is) -- so the scale factors
    cleanly out of that matmul. WRONG for V's use in P@V: there, the
    contracted dimension IS the key-token axis, so a per-token V scale would
    vary across exactly the index being summed and cannot be factored out
    after the matmul at all (there is no valid shape left to apply it to
    once that axis is gone). V needs ``quantize_per_channel_int8`` instead.
    """
    assert x.is_contiguous()
    B, L, H, D = x.shape
    x_i8 = torch.empty_like(x, dtype=torch.int8)
    x_scale = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(L, BLOCK_T), B * H)
    _quantize_kernel[grid](x, x_i8, x_scale, L, H, D, BLOCK_T)
    return x_i8, x_scale


@triton.jit
def _quantize_per_channel_kernel(
    X,
    X_I8,
    SCALE,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Elementwise quantize only -- SCALE is precomputed on the host (one
    global amax reduction over the whole L axis per (b, h, d), a small
    (B, H, D) result, not a full-tensor-sized intermediate) and just gets
    broadcast to every token here. No reduction inside this kernel at all,
    unlike ``_quantize_kernel`` above, because the scale must be identical
    for every token by construction -- that is the entire point.
    """
    idx_t = tl.program_id(0)
    idx_bh = tl.program_id(1)

    idx_b = idx_bh // H
    idx_h = idx_bh - idx_b * H

    offs_t = idx_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, D)
    t_mask = offs_t < L

    x_offset = idx_b * L * H * D + idx_h * D
    x = tl.load(
        X + x_offset + offs_t[:, None] * (H * D) + offs_d[None, :],
        mask=t_mask[:, None], other=0.0,
    )
    scale = tl.load(SCALE + idx_bh * D + offs_d)   # (D,), same for every token

    xi = (x / scale[None, :] + tl.where(x >= 0, 0.5, -0.5)).to(tl.int32)
    xi = tl.minimum(tl.maximum(xi, -127), 127)

    tl.store(
        X_I8 + x_offset + offs_t[:, None] * (H * D) + offs_d[None, :],
        xi.to(tl.int8), mask=t_mask[:, None],
    )


def quantize_per_channel_int8(x, BLOCK_T=128):
    """(B, L, H, D) -> (int8 same shape, fp32 scale (B, H, D)).

    One scale per (batch, head, D-channel), constant across every token --
    see ``quantize_per_token_int8``'s docstring for why V specifically needs
    this instead of per-token scaling. The amax reduction itself (a small
    (B, H, D) result) is plain torch -- cheap regardless of how it is
    computed, unlike an elementwise op over the full tensor -- and only the
    elementwise quantize step (the part that touches every element of ``x``)
    is the fused Triton kernel.
    """
    assert x.is_contiguous()
    B, L, H, D = x.shape
    amax = x.abs().amax(dim=1)                                    # (B, H, D)
    scale = torch.where(
        amax > 0, amax.float() / 127.0,
        torch.ones((B, H, D), device=x.device, dtype=torch.float32),
    ).contiguous()
    x_i8 = torch.empty_like(x, dtype=torch.int8)
    grid = (triton.cdiv(L, BLOCK_T), B * H)
    _quantize_per_channel_kernel[grid](x, x_i8, scale, L, H, D, BLOCK_T)
    return x_i8, scale


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT,
    OS,
    H: tl.constexpr,
    LQ: tl.constexpr,
    LK: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TAIL_MAX,
    TAIL_NUM,
    TAIL_DEN,
    HAS_TAIL: tl.constexpr,
    Q_I8,
    Q_SCALE,
    K_I8,
    K_SCALE,
    USE_INT8_QK: tl.constexpr,
    V_I8,
    V_SCALE,
    USE_INT8_PV: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    idx_b = idx_bh // H
    idx_h = idx_bh % H

    HD: tl.constexpr = H * D

    # Q/K/V/O: (B, L, H, D) contiguous.
    q_offset = idx_b * LQ * HD + idx_h * D
    kv_offset = idx_b * LK * HD + idx_h * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    OS_ptrs = OS + q_offset + offs_m[:, None] * HD + offs_d[None, :]
    LUT_ptr = LUT + lut_offset
    row_ok = offs_m < LQ

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    if USE_INT8_QK:
        # Pre-quantized on the host by quantize_per_token_int8 -- this is a
        # plain load, not a compute step. (B, L, H, D)/(B, L, H) layouts, same
        # b/h offset convention as Q/K above just without the D stride on the
        # scale.
        QI8_ptrs = Q_I8 + q_offset + offs_m[:, None] * HD + offs_d[None, :]
        QSCALE_ptrs = Q_SCALE + idx_b * LQ * H + idx_h + offs_m * H
        q_i8 = tl.load(QI8_ptrs, mask=row_ok[:, None], other=0)
        q_scale = tl.load(QSCALE_ptrs, mask=row_ok, other=1.0)
    else:
        Q_ptrs = Q + q_offset + offs_m[:, None] * HD + offs_d[None, :]
        # FIX vs upstream: other=0.0 on every masked load.
        q = tl.load(Q_ptrs, mask=row_ok[:, None], other=0.0)

    if USE_INT8_PV:
        # Per-channel, precomputed once on the host by
        # quantize_per_channel_int8 -- one (D,) vector for this (b, h),
        # constant across every key token, loaded once here rather than
        # once per loop iteration since it never changes within this program.
        VSCALE_ptrs = V_SCALE + idx_bh * D + offs_d
        v_scale_pc = tl.load(VSCALE_ptrs)

    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx).to(tl.int64)
        k_start = idx_n * BLOCK_N
        k_mask = (k_start + offs_n) < LK

        if USE_INT8_PV:
            VI8_ptrs = V_I8 + kv_offset + (k_start + offs_n)[:, None] * HD + offs_d[None, :]
            v_i8 = tl.load(VI8_ptrs, mask=k_mask[:, None], other=0)
        else:
            V_ptrs = V + kv_offset + (k_start + offs_n)[:, None] * HD + offs_d[None, :]
            v = tl.load(V_ptrs, mask=k_mask[:, None], other=0.0)

        if USE_INT8_QK:
            # Loading the int8 tile directly -- not quantizing here -- is
            # the entire point: half the bytes of the bf16 tile this replaces,
            # and no per-iteration amax/round/clamp cost, because that work
            # already happened once for the whole tensor in
            # quantize_per_token_int8.
            KI8_ptrs = K_I8 + kv_offset + (k_start + offs_n)[None, :] * HD + offs_d[:, None]
            KSCALE_ptrs = K_SCALE + idx_b * LK * H + idx_h + (k_start + offs_n) * H
            k_i8 = tl.load(KI8_ptrs, mask=k_mask[None, :], other=0)
            k_scale = tl.load(KSCALE_ptrs, mask=k_mask, other=1.0)

            qk_i32 = tl.dot(q_i8, k_i8, out_dtype=tl.int32)
            qk = qk_i32.to(tl.float32) * q_scale[:, None] * k_scale[None, :]
            qk = qk * (qk_scale * 1.4426950408889634)
        else:
            K_ptrs = K + kv_offset + (k_start + offs_n)[None, :] * HD + offs_d[:, None]
            k = tl.load(K_ptrs, mask=k_mask[None, :], other=0.0)
            qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)  # 1/ln(2), for exp2
        qk = tl.where(k_mask[None, :], qk, float("-inf"))

        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        qk = qk - new_m[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)

        if USE_INT8_PV:
            # p is >= 0 by construction (post-exp2), so this is a one-sided
            # [0, 127] quantization, not the symmetric [-127, 127] used for
            # Q/K -- and it MUST happen fresh every iteration, in registers,
            # unlike K/V's quantize-once host-side pass: p is a genuinely new
            # value each time (the local softmax numerator for THIS block,
            # before the running m_i/l_i merge below), not something that
            # could be precomputed. The reason this doesn't repeat the
            # earlier eager-ops mistake is that nothing here round-trips
            # through global memory -- p never leaves registers.
            p_amax = tl.max(p, 1)
            p_scale = tl.where(p_amax > 0, p_amax / 127.0, 1.0)
            p_i32 = (p / p_scale[:, None] + 0.5).to(tl.int32)
            p_i32 = tl.minimum(p_i32, 127)
            p_i8 = p_i32.to(tl.int8)

            pv_i32 = tl.dot(p_i8, v_i8, out_dtype=tl.int32)
            pv = pv_i32.to(tl.float32) * p_scale[:, None] * v_scale_pc[None, :]
            o_s = o_s * alpha[:, None] + pv
        else:
            o_s = o_s * alpha[:, None]
            o_s += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = new_m

    # Optional: one pooled term standing in for every key block that did NOT
    # make top-k, computed on the host in block_map.py from the same pooled
    # centroids the selection itself used. Same merge algebra as the loop
    # above, except this term's own local max/sum were already fixed on the
    # host (tail_max), so unlike a fresh block it needs its own rescale
    # (``beta``) into the new running max, not just the old accumulator's.
    if HAS_TAIL:
        tail_scalar_offset = idx_bh * M_BLOCKS + idx_m
        t_max = tl.load(TAIL_MAX + tail_scalar_offset)
        t_den = tl.load(TAIL_DEN + tail_scalar_offset)
        t_num = tl.load(TAIL_NUM + tail_scalar_offset * D + offs_d)

        new_m = tl.maximum(m_i, t_max)
        alpha = tl.math.exp2(m_i - new_m)
        beta = tl.math.exp2(t_max - new_m)
        o_s = o_s * alpha[:, None] + t_num[None, :] * beta
        l_i = l_i * alpha + t_den * beta
        m_i = new_m

    o_s = o_s / l_i[:, None]
    tl.store(OS_ptrs, o_s.to(OS.type.element_ty), mask=row_ok[:, None])


# (BLOCK_M, BLOCK_N) -> ladder of (num_warps, num_stages) to try, best first.
#
# These are not upstream's numbers. LightX2V hardcodes num_warps=4, num_stages=3,
# which on sm_120 needs 160 KB of shared memory against a 99 KB limit -- it does
# not merely run slowly, it fails to launch. Consumer Blackwell has less shared
# memory than the datacentre parts these kernels were tuned on, so we probe and
# fall back. Measured on a 5090 at S=32768, H=56, D=128 (ms, lower better):
#   128x64  w8 s3 -> 22.4   128x64  w4 s3 -> 23.9   128x128 w8 s2 -> 25.4
#   64x64   w4 s1 -> 26.8   64x128  w4 s2 -> 30.5   128x128 w4 s1 -> 40.5
_LADDER = {
    (128, 64): ((8, 3), (4, 3), (8, 2), (4, 1)),
    (128, 128): ((8, 2), (4, 2), (8, 1), (4, 1)),
    (64, 128): ((4, 2), (8, 2), (4, 1)),
    (64, 64): ((4, 1), (4, 3), (8, 3), (8, 1)),
    # UNBENCHMARKED on this kernel/hardware -- no measured (num_warps,
    # num_stages) exists for 32x32 here. (4, 2) is first based on external
    # evidence, not a local measurement: published Triton flash-attention
    # autotune sweeps that include a 32x32 tile consistently pair it with
    # num_warps=4, num_stages=2 rather than lower warp counts (e.g.
    # triton-lang/triton discussion #8261's BLOCK_BR=32,BLOCK_BC=32 config).
    # A local benchmark run measured 32x32 at the same speed as 64x64 with
    # the earlier (2, 1)-first ordering, so this reordering has not been
    # re-verified to actually help -- treat as still unproven, not a fix.
    (32, 32): ((4, 2), (2, 1), (4, 1), (2, 2)),
}
_CHOSEN: dict = {}


def block_sparse_attention(q, k, v, lut, topk, BLOCK_M, BLOCK_N, qk_scale=None,
                           tail=None, use_int8_qk=False, use_int8_pv=False):
    """Attend each query block to only the key blocks named in ``lut``.

    ``q``/``k``/``v`` are ``(B, L, H, D)`` contiguous; ``lut`` is
    ``(B, H, M_BLOCKS, topk)`` int32, contiguous. Returns ``(B, L, H, D)``.

    ``tail``, when given, is the ``(tail_max, tail_num, tail_den)`` triple
    from ``block_map.get_block_map`` -- one pooled term per query block
    covering every key block ``lut`` left out, merged in with the same
    online-softmax algebra the loop above uses between real blocks. ``None``
    (the default) reproduces the exact previous behaviour at zero added cost:
    the extra tensors are never touched and the branch is compiled out.

    ``use_int8_qk`` quantizes Q and K to int8 (per-token, symmetric, dynamic
    scale) before the QK dot product only -- PV stays in q/k/v's native
    dtype. SageAttention calls this split ``qk_int8_pv_fp16``; it is not
    full int8 attention. Quantization happens ONCE here, for the whole
    tensor, via ``quantize_per_token_int8`` -- not inside the kernel's topk
    loop, where a key block would otherwise get re-quantized once per query
    block that selects it (roughly ``(1 - sparsity_ratio) x M_BLOCKS`` times
    on average). The kernel just loads the already-int8 tiles.

    ``use_int8_pv`` independently quantizes P (the post-softmax weights) and
    V to int8 for the second matmul. Orthogonal to ``use_int8_qk`` -- P's
    value doesn't depend on how QK was computed, only on the (already
    dequantized) scores. V uses ``quantize_per_channel_int8``, NOT
    ``quantize_per_token_int8`` -- per-token would put the scale inside the
    dimension P@V contracts over (the key-token axis), where it cannot be
    factored back out after the matmul; per-channel keeps it constant across
    that axis instead, quantized once for the whole tensor same as Q/K. P
    itself cannot be precomputed -- it is quantized fresh every loop
    iteration, but entirely in registers, so this does not repeat the
    eager-torch-ops mistake ``quantize_per_token_int8``'s docstring
    describes: nothing here round-trips through global memory.

    MEMORY, both flags: each produces a NEW int8 buffer resident alongside
    the still-live bf16 original for the rest of this call -- the caller
    holds its own reference to q/k/v throughout this function's entire
    execution (it is synchronously suspended waiting for this call to
    return, so it cannot free anything until this function does, regardless
    of what this function does internally). This is a real, unavoidable
    peak-memory cost while both representations coexist, not a
    reference-holding bug with an in-function fix. Enabling both at once
    means THREE extra buffers (q_i8, k_i8, v_i8) alongside the bf16
    originals simultaneously; if memory is tight, enable only whichever one
    of the two actually matters for your workload rather than both.
    """
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert lut.is_contiguous()
    assert BLOCK_M in (32, 64, 128) and BLOCK_N in (32, 64, 128)

    B, LQ, H, D = q.shape
    LK = k.shape[1]
    device, dtype = q.device, q.dtype
    if qk_scale is None:
        qk_scale = D**-0.5

    M_BLOCKS = triton.cdiv(LQ, BLOCK_M)
    o_s = torch.empty_like(q)
    grid = (M_BLOCKS, B * H)

    has_tail = tail is not None
    if has_tail:
        tail_max, tail_num, tail_den = tail
        assert tail_max.is_contiguous() and tail_num.is_contiguous()
        assert tail_den.is_contiguous()
    else:
        # Triton needs real tensors on the signature even when the HAS_TAIL
        # branch compiles them out -- a 1-element placeholder on the same
        # device/dtype costs nothing and is never read.
        tail_max = tail_den = torch.empty(1, device=device, dtype=torch.float32)
        tail_num = torch.empty(1, device=device, dtype=torch.float32)

    use_int8_qk = bool(use_int8_qk)
    if use_int8_qk:
        # One pass over the whole tensor each, not per key-block selection --
        # see quantize_per_token_int8's docstring for why that distinction
        # is the entire point of this rewrite.
        #
        # NOTE on memory: q_i8/k_i8 are NEW buffers, resident alongside the
        # still-live q/k for the rest of this call. The caller (patch.py's
        # override()) holds its own reference to q/k throughout this entire
        # function's execution -- it is synchronously suspended waiting for
        # this call to return, so it cannot free anything until this
        # function does, regardless of what this function does internally.
        # There is no in-function fix for that: turning this on has a real,
        # unavoidable peak-memory cost while both representations coexist,
        # not a reference-holding bug.
        q_i8, q_scale = quantize_per_token_int8(q)
        k_i8, k_scale = quantize_per_token_int8(k)
    else:
        # Same placeholder pattern as the tail tensors above: Triton needs
        # real tensors on the signature even when the branch compiles out.
        q_i8 = k_i8 = torch.empty(1, device=device, dtype=torch.int8)
        q_scale = k_scale = torch.empty(1, device=device, dtype=torch.float32)

    use_int8_pv = bool(use_int8_pv)
    if use_int8_pv:
        v_i8, v_scale = quantize_per_channel_int8(v)   # same memory note as above
    else:
        v_i8 = torch.empty(1, device=device, dtype=torch.int8)
        v_scale = torch.empty(1, device=device, dtype=torch.float32)

    key = (BLOCK_M, BLOCK_N, D, has_tail, use_int8_qk, use_int8_pv)
    ladder = (_CHOSEN[key],) if key in _CHOSEN else _LADDER[(BLOCK_M, BLOCK_N)]

    last = None
    for cfg in ladder:
        num_warps, num_stages = cfg
        try:
            _attn_fwd[grid](
                q, k, v, qk_scale, topk, lut, o_s,
                H, LQ, LK, M_BLOCKS, D, BLOCK_M, BLOCK_N,
                tail_max, tail_num, tail_den, has_tail,
                q_i8, q_scale, k_i8, k_scale, use_int8_qk,
                v_i8, v_scale, use_int8_pv,
                num_warps=num_warps, num_stages=num_stages,
            )
        except triton.runtime.errors.OutOfResources as exc:
            last = exc
            continue
        _CHOSEN[key] = cfg
        return o_s

    raise last if last is not None else RuntimeError("no viable launch config")


def warm_launch_config(device, dtype, BLOCK_M, BLOCK_N, D=128,
                       has_tail=False, use_int8_qk=False, use_int8_pv=False):
    """Pre-select and cache the launch config for one ``block_sparse_attention``
    variant, on a single throwaway call.

    ``block_sparse_attention`` already caches its winning (num_warps,
    num_stages) config per ``(BLOCK_M, BLOCK_N, D, has_tail, use_int8_qk,
    use_int8_pv)`` (see ``_CHOSEN`` above), but the FIRST call for a
    never-before-seen key tries every ``_LADDER`` rung in order, catching
    ``OutOfResources`` on each failing one -- a different, larger set of
    allocations than every later call for that same key, which only ever
    launches the one cached winner.

    That difference matters once ComfyUI's own model compiler is active
    (the ``comfy.model_prefetch`` "malloc graph" CUDA-graph-capture path,
    Comfy-Org/ComfyUI PR #15861): it expects a captured region's allocation
    pattern to repeat exactly from one sampling step to the next, and the
    real first sparse call happens *inside* that captured region (H3's
    ``forward`` wraps its whole block loop in ``malloc_graph_begin``/
    ``end``). Probing the ladder there -- which only happens once per
    distinct settings combination, but once is enough -- reports as a graph
    break.

    Calling this once per combination from ``patch_h3_sla``, at node-execute
    time (before the model ever reaches the sampler), runs that same
    one-time probe outside any capture region, so the capture only ever sees
    the warmed, single-config path afterward.

    Best-effort and NOT validated against a real graph-break reproduction --
    only reasoned from the two code paths above. Never raises: a failed
    warm-up just means the first real call falls back to probing the ladder
    itself, exactly as it always has.
    """
    if device is None:
        return
    device = torch.device(device)
    if device.type != "cuda":
        return
    try:
        S = BLOCK_N * 2  # two key blocks: enough for a real lut entry
        q = torch.zeros((1, BLOCK_M, 1, D), device=device, dtype=dtype)
        k = torch.zeros((1, S, 1, D), device=device, dtype=dtype)
        v = torch.zeros((1, S, 1, D), device=device, dtype=dtype)
        lut = torch.zeros((1, 1, 1, 1), device=device, dtype=torch.int32)
        tail = None
        if has_tail:
            tail = (
                torch.zeros((1, 1, 1), device=device, dtype=torch.float32),
                torch.zeros((1, 1, 1, D), device=device, dtype=torch.float32),
                torch.zeros((1, 1, 1), device=device, dtype=torch.float32),
            )
        block_sparse_attention(q, k, v, lut, 1, BLOCK_M, BLOCK_N,
                               tail=tail, use_int8_qk=use_int8_qk,
                               use_int8_pv=use_int8_pv)
    except Exception:                                  # noqa: BLE001
        log.debug("[H3Utils] SLA: kernel warm-up failed; the first real "
                  "call will probe launch configs itself instead.",
                  exc_info=True)
