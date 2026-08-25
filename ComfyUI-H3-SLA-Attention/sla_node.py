"""The H3 SLA Attention node.

Drop it on the MODEL wire after the LoRA loaders, last before the sampler. It
replaces MiniMax-H3 self-attention with the block-sparse kernel that the
lightx2v SLA turbo LoRA was distilled against, which is the piece ComfyUI does
not otherwise have -- and the reason that LoRA gives no speedup on its own.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

log = logging.getLogger("H3Utils")

BLOCK_SIZES = ("64", "128")
# Matches kijai/ComfyUI-KJNodes' PatchSageAttentionKJ mode list exactly (see
# sla/patch.py for the per-mode kernel + pv_accum_dtype each one calls),
# plus this node's own "pytorch" / "ck" / "auto" choices.
DENSE_BACKENDS = (
    "pytorch",
    "comfy_kitchen",
    "sage:auto",
    "sage:qk_int8_pv_fp16_cuda",
    "sage:qk_int8_pv_fp16_triton",
    "sage:qk_int8_pv_fp8_cuda",
    "sage:qk_int8_pv_fp8_cuda++",
    "auto",
)


class H3SLAAttention(io.ComfyNode):
    """Run H3 self-attention over only the key blocks that matter.

    Each query block is scored against every key block with one small pooled
    matmul, and only the top ``1 - sparsity_ratio`` fraction is actually
    attended. Nothing here is trained and nothing is loaded: the published SLA
    files contain only ordinary LoRA tensors, and the sparsity is decided at
    runtime from q and k. The LoRA's job is to make the model tolerate the
    sparsity, not to provide it.

    Measured end-to-end on a 5090 at 768p/15s with the SLA turbo LoRA:
    ~44 s/it dense against ~31 s/it at sparsity 0.85 and ~25 s/it at 0.90, so
    1.4-1.75x, with no extra VRAM. Attention is only ~30 s of that 44 s step,
    so the Amdahl ceiling is 3.17x however fast attention gets. See SLA.md for
    why the widely-quoted 2.5x is an eight-GPU number.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SLAAttention",
            display_name="H3 SLA Attention",
            category="PlagueKind/model_patches/minimax",
            description=(
                "Block-sparse attention for MiniMax-H3, matching the inference "
                "path the lightx2v SLA turbo LoRA was trained for. Place it "
                "after your LoRA loader, last before the sampler. Long "
                "sequences benefit most, so the gain grows with resolution and "
                "duration; short ones fall back to dense automatically."),
            inputs=[
                io.Model.Input("model",
                    tooltip="MODEL, after the LoRA loader(s)."),
                io.Float.Input("sparsity_ratio", default=0.90, min=0.0, max=0.95,
                    step=0.05, round=False,
                    tooltip=(
                        "Fraction of key blocks skipped. 0.85 is what lightx2v "
                        "ships and what the SLA turbo LoRA was distilled "
                        "against; 0.90 is the value validated here and is "
                        "~15% faster. Sparsity did NOT turn out to drive the "
                        "speech artefacts on H3 -- step count did, so use 6 "
                        "steps rather than lowering this. Break-even is about "
                        "0.60 -- below that "
                        "this kernel is SLOWER than dense attention, so a low "
                        "value is a loss, not a safe fallback. If a LoRA "
                        "cannot take 0.7+, this node has nothing to offer it. "
                        "0.0 disables sparsity without removing the node.")),
                io.Combo.Input("block_size", options=list(BLOCK_SIZES),
                    default="64",
                    tooltip=(
                        "How many sequence tokens share one key selection. "
                        "Unrelated to the model's 128-wide heads. This matters "
                        "far more for audio than video: H3 packs audio at 80 "
                        "rows per second, so a 128-row block forces 1.6 s of "
                        "audio down one attention pattern, while the same 128 "
                        "rows are only 3% of a video frame. Speech came out "
                        "robotic at 128 and clean at 64, for about 2% more "
                        "time -- halving the block doubles the block count, so "
                        "the attention work is identical and only the routing "
                        "gets finer. Use 128 only if you generate without "
                        "meaningful audio.")),
                io.Int.Input("min_seq_len", default=4096, min=0, max=1000000,
                    step=1024, optional=True,
                    tooltip=(
                        "Sequences shorter than this stay dense. Guards two "
                        "things: the short text-refiner attention, which must "
                        "never be sparsified, and low-resolution or short "
                        "clips, where block selection would cost more than it "
                        "saves. Lower it only if you know your sequence is "
                        "long enough to benefit.")),
                io.Int.Input("dense_last_steps", default=1, min=0, max=8,
                    optional=True,
                    tooltip=(
                        "Run the last N sampling steps at full attention. 0 "
                        "matches lightx2v exactly. 1 costs a little speed and "
                        "can recover fine detail, since the final step's error "
                        "is the one you actually see. Stacks with dense_steps "
                        "below rather than replacing it.")),
                io.Boolean.Input("protect_audio", default=True,
                    label_on="protect", label_off="uniform (lightx2v parity)",
                    optional=True,
                    tooltip=(
                        "Always attend the [text | cond | audio] prefix, "
                        "whatever top-k picks. Audio is about 1% of the packed "
                        "sequence -- 19 key blocks of 1794 at 768p/15s -- so "
                        "plain top-k regularly drops all of it and the "
                        "soundtrack degrades while the video still looks fine. "
                        "Costs roughly 7%. Turn off only to reproduce "
                        "lightx2v's uniform selection exactly.")),
                io.Boolean.Input("enabled", default=True,
                    label_on="sparse", label_off="dense (bypass)",
                    optional=True,
                    tooltip=(
                        "Turn off to pass the model straight through, for a "
                        "like-for-like speed baseline without rewiring.")),
                # Everything below this line was added after the original
                # release. Kept at the end, in the order added, so old saved
                # workflows -- which can store widget values positionally --
                # keep lining up with the right inputs instead of shifting
                # onto whatever got inserted ahead of them.
                io.String.Input("dense_steps", default="0", optional=True,
                    tooltip=(
                        "Explicit 0-based step indices to force dense, on top "
                        "of dense_last_steps -- e.g. '0,1' or '0-2'. Early "
                        "steps set global composition and prompt adherence, "
                        "so keeping just those exact and sparsifying the rest "
                        "can fix prompt-following regressions without paying "
                        "for full attention on every step. Blank = none.")),
                io.Combo.Input("dense_backend", options=list(DENSE_BACKENDS),
                    default="comfy_kitchen", optional=True,
                    tooltip=(
                        "Attention kernel used on every dense fall-through "
                        "(short sequences, dense_last_steps, dense_steps). "
                        "'ck' (default, Comfy Kitchen int8) is fast enough on "
                        "the handful of dense steps this node runs to be "
                        "worth its precision tradeoff there. 'pytorch' pins "
                        "the plain reference kernel instead if you want zero "
                        "quantization anywhere in the dense path, at real "
                        "cost to dense-step speed -- dense steps are the "
                        "ones this node already decided must be exact, so "
                        "that's the safest choice if you're chasing maximum "
                        "quality over speed there. The "
                        "'sage:*' modes match kijai/ComfyUI-KJNodes' "
                        "PatchSageAttentionKJ dropdown one-for-one -- "
                        "'sage:auto' lets the sageattention package pick; "
                        "the rest pin a specific kernel + pv_accum_dtype: "
                        "fp16_cuda uses fp32 accum (safest), fp8_cuda uses "
                        "fp32+fp32 (safest of the fp8 pair), fp8_cuda++ uses "
                        "fp32+fp16 (faster, more overflow-prone -- try this "
                        "first if fp8_cuda works but you want more speed). "
                        "Needs the sageattention package installed; falls "
                        "back to whatever's already active with a logged "
                        "warning if it isn't, or if a specific kernel is "
                        "missing from your installed version. 'auto' "
                        "restores the old behaviour of using whatever "
                        "backend is already active globally. The sparse path "
                        "is unaffected by this setting either way -- it "
                        "never goes through backend selection at all.")),
                io.Boolean.Input("disable_fp16_accum", default=True,
                    label_on="disabled (recommended)", label_off="follow --fast",
                    optional=True,
                    tooltip=(
                        "Force off the fp16/bf16 reduced-precision matmul "
                        "reduction path for this model's sampling run, "
                        "regardless of the global --fast fp16_accumulation "
                        "flag. Measured to cost quality on H3 with no "
                        "throughput gain. Turn off only to A/B against the "
                        "global flag.")),
                io.Boolean.Input("stabilize_motion", default=True,
                    label_on="on", label_off="off",
                    optional=True,
                    tooltip=(
                        "Bias each layer's block selection toward what it "
                        "picked last step, so a near-tie between two blocks "
                        "doesn't flip for no reason and show up as a faint "
                        "double-exposure on fast motion. Keeps a bounded set "
                        "of near-cutoff block choices per layer; it is a fix "
                        "for that one specific symptom, not a general quality "
                        "dial.")),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, sparsity_ratio=0.90, block_size="64",
                min_seq_len=4096, dense_last_steps=1, protect_audio=True,
                enabled=True, dense_steps="0", dense_backend="comfy_kitchen",
                disable_fp16_accum=True, stabilize_motion=True) -> io.NodeOutput:
        if not enabled:
            log.info("[H3Utils] SLA disabled; model passed through unchanged.")
            return io.NodeOutput(model)

        try:
            from .sla import patch_h3_sla
            patched = patch_h3_sla(
                model,
                sparsity_ratio=sparsity_ratio,
                block_size=int(block_size),
                min_seq_len=min_seq_len,
                dense_last_steps=dense_last_steps,
                dense_steps=dense_steps,
                dense_backend=dense_backend,
                disable_fp16_accum=disable_fp16_accum,
                protect_audio=protect_audio,
                stabilize_motion=stabilize_motion,
            )
        except Exception:                                # noqa: BLE001
            # Triton missing, an incompatible GPU, a ComfyUI API change -- none
            # of it should cost the user their run. Dense attention still works.
            log.exception("[H3Utils] SLA patch failed; passing the model through "
                          "unchanged (attention will NOT be sparsified).")
            return io.NodeOutput(model)

        return io.NodeOutput(patched)
