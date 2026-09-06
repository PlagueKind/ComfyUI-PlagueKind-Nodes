"""The H3 SLA Attention node.

Drop it on the MODEL wire after the LoRA loaders, last before the sampler. It
replaces MiniMax-H3 self-attention with the block-sparse kernel that the
SLA turbo LoRA was distilled against, which is the piece ComfyUI does
not otherwise have -- and the reason that LoRA gives no speedup on its own.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

log = logging.getLogger("H3Utils")

BLOCK_SIZES = ("32", "64", "128")
REFERENCE_PROTECTION_MODES = ("Heavy Enforcement", "Light", "Off")
# Mirrors an existing, widely-used sage-attention kernel dropdown's mode
# list exactly (see sla/patch.py for the per-mode kernel + pv_accum_dtype
# each one calls),
# plus this node's own "pytorch" / "comfy_kitchen" / "auto" choices.
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
                "sequences benefit most, so the gain grows with resolution and "
                "duration; short ones fall back to dense automatically."),
            inputs=[
                io.Model.Input("model",
                    tooltip="MODEL,"),
                io.Float.Input("sparsity_ratio", default=0.80, min=0.0, max=0.95,
                    step=0.05, round=False,
                    tooltip=(
                        "Fraction of key blocks skipped. 0.85 is the shipped default "
                        "and what the SLA turbo LoRA was distilled "
                        "against; 0.90 is the value validated here and is "
                        "~15% faster. Sparsity did NOT turn out to drive the "
                        "speech artefacts on H3 -- step count did, so use 6 "
                        "steps rather than lowering this. Break-even is about "
                        "0.60 -- below that "
                        "this kernel is SLOWER than dense attention,"
                        "0.0 disables sparsity without removing the node.")),
                io.Combo.Input("block_size", options=list(BLOCK_SIZES),
                    default="32",
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
                        "meaningful audio. Coming down to 32 has increased "
                        "the quality even further for marginal slowdown ")),
                io.Int.Input("min_seq_len", default=12228, min=0, max=1000000,
                    step=1024, optional=True,
                    tooltip=(
                        "Sequences shorter than this stay dense. Guards two "
                        "things: the short text-refiner attention, which must "
                        "never be sparsified, and low-resolution or short "
                        "clips, where block selection would cost more than it "
                        "saves. Lower it only if you know your sequence is "
                        "long enough to benefit.")),
                io.Int.Input("dense_last_steps", default=0, min=0, max=8,
                    optional=True,
                    tooltip=(
                        "Run the last N sampling steps at full attention. 0 "
                        "matches the turbo LoRA's original distillation setup "
                        "exactly. 1 costs a little speed and "
                        "can recover fine detail, since the final step's error "
                        "is the one you actually see. Stacks with dense_steps "
                        "below rather than replacing it.")),
                io.Boolean.Input("protect_audio", default=False,
                    label_on="protect", label_off="uniform (turbo parity)",
                    optional=True,
                    tooltip=(
                        "Always attend blocks overlapping actual language "
                        "tokens, target audio, and audio-reference segments. "
                        "Visual-reference blocks are controlled separately. "
                        "Disable only for turbo-style uniform sparsity; "
                        "testing found partial or unprotected audio unstable "
                        "for very little speed gain.")),
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
                io.String.Input("dense_steps", default="1", optional=True,
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
                        "'comfy_kitchen' (default, Comfy Kitchen int8) is fast enough on "
                        "the handful of dense steps this node runs to be "
                        "worth its precision tradeoff there. 'pytorch' pins "
                        "the plain reference kernel instead if you want zero "
                        "quantization anywhere in the dense path, at real "
                        "cost to dense-step speed -- dense steps are the "
                        "ones this node already decided must be exact, so "
                        "that's the safest choice if you're chasing maximum "
                        "quality over speed there. The "
                        "'sage:*' modes match an existing sage-attention "
                        "kernel dropdown one-for-one -- "
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
                io.Boolean.Input("stabilize_motion", default=False,
                    label_on="on", label_off="off",
                    optional=True,
                    tooltip=(
                        "Bias each layer's block selection toward what it "
                        "picked last step, so a near-tie between two blocks "
                        "doesn't flip for no reason and show up as a faint "
                        "double-exposure on fast motion. Only target-video "
                        "query rows are stabilized; text and audio choices "
                        "remain step-local. It is a fix for that one specific "
                        "symptom, not a general quality dial. - Uses more Vram ")),
                io.Combo.Input("reference_protection",
                    display_name="Protect Vid/Ref",
                    options=list(REFERENCE_PROTECTION_MODES), default="Off",
                    optional=True,
                    tooltip=(
                        "Protect Image/Video Reference. Heavy Enforcement "
                        "guarantees every Qwen vision-token, "
                        "conditioning/image-reference, and video-reference "
                        "block, matching the broad legacy prefix protection -- "
                        "this is a reinforcement of those blocks, not a "
                        "protection tuned for reference quality, and is most "
                        "likely unusable with max ref size mode. Light uses "
                        "fixed 0.85 reference sparsity and guarantees the "
                        "best-scoring 15% of each visual-reference range. Off "
                        "adds no special quota; references still participate "
                        "in ordinary top-k. Default Off preserves the precise "
                        "audio patch's fastest behaviour.")),
                io.Boolean.Input("tail_correction", default=False,
                    label_on="on (experimental)", label_off="off",
                    optional=True,
                    tooltip=(
                        "Instead of a hard zero for every key block topk left "
                        "out, fold in one pooled term standing in for all of "
                        "them, so nothing leaves the softmax -- same idea as "
                        "a widely-used block-sparse kernel's tail handling. Scored from "
                        "the same pooled centroids selection already computes, "
                        "so the only added cost is one more mean-pool of V "
                        "plus a small reduction per call, not a second "
                        "attention pass. Should help most at high sparsity, "
                        "where the discarded tail is largest. Off by default: "
                        "new, and its effect on H3 output quality specifically "
                        "hasn't been validated the way sparsity_ratio's "
                        "defaults have -- test before trusting it in a real "
                        "render.")),
                io.Boolean.Input("use_int8_qk", default=True,
                    label_on="on (experimental)", label_off="off",
                    optional=True,
                    tooltip=(
                        "Quantize Q and K to int8 (per-token, dynamic scale) "
                        "before the QK dot product on the selected topk "
                        "blocks -- PV stays full precision (the mirror "
                        "use_int8_pv toggle was removed from this node: its "
                        "hidden widget could still be converted to an input "
                        "socket and connected, which broke the node -- PV "
                        "quantization is no longer exposed here). This is "
                        "SageAttention's qk_int8_pv_fp16 split, not full int8 "
                        "attention, and it's a different lever from "
                        "dense_backend above: that setting only affects dense "
                        "fall-through steps, this affects the sparse compute "
                        "itself, on every sparse step. Ignored entirely when "
                        "engine is comfy_kitchen, which quantizes internally "
                        "regardless. UNTESTED ON HARDWARE: the "
                        "quantize/dequantize math checks out against exact "
                        "fp32 scores in isolation, but real speed, launch "
                        "stability, and output quality on H3 have not been "
                        "measured on a GPU. Try it against a known-good render "
                        "before trusting it, and expect to possibly hit a "
                        "launch failure on some GPU/Triton combinations before "
                        "it's been shaken out.")),
                io.Combo.Input("engine", options=["triton", "comfy_kitchen"],
                    default="comfy_kitchen", optional=True,
                    tooltip=(
                        "Which attention implementation runs the sparse path. "
                        "triton (default) is this node pack's own kernel -- "
                        "every other widget above applies to it fully. "
                        "comfy_kitchen instead calls comfy_kitchen's real "
                        "compiled sol_attn kernel (Comfy-Org/ComfyUI PR "
                        "#16072, needs comfy-kitchen>=0.2.32 installed): "
                        "genuine CUDA int8 compute and a built-in pooled tail "
                        "term, but it can only express ONE contiguous "
                        "protected range, has no reference-quota tier, and "
                        "has no cross-step stabilize_motion -- reference_"
                        "protection, multi-span protect_ranges, and "
                        "stabilize_motion are silently disabled (one-time log "
                        "warning each) rather than approximated, and "
                        "tail_correction/use_int8_qk above are ignored "
                        "since the real kernel quantizes internally "
                        "regardless, but now honours tail_correction's "
                        "on/off setting rather than always applying it. Both "
                        "fall back to dense the same as any other kernel "
                        "failure if their kernel is unavailable or throws.")),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, sparsity_ratio=0.80, block_size="32",
                min_seq_len=12228, dense_last_steps=0, protect_audio=False,
                enabled=True, dense_steps="1", dense_backend="comfy_kitchen",
                disable_fp16_accum=True, stabilize_motion=False,
                reference_protection="Off", tail_correction=False,
                use_int8_qk=True, engine="comfy_kitchen") -> io.NodeOutput:
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
                reference_protection=reference_protection,
                tail_correction=tail_correction,
                use_int8_qk=use_int8_qk,
                engine=engine,
            )
        except Exception:                                # noqa: BLE001
            # Triton missing, an incompatible GPU, a ComfyUI API change -- none
            # of it should cost the user their run. Dense attention still works.
            log.exception("[H3Utils] SLA patch failed; passing the model through "
                          "unchanged (attention will NOT be sparsified).")
            return io.NodeOutput(model)

        return io.NodeOutput(patched)
