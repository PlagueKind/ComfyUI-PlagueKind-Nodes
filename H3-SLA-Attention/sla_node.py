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
                io.Int.Input("min_seq_len", default=8192, min=0, max=1000000,
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
                        "matches lightx2v exactly. 1 costs a little speed and "
                        "can recover fine detail, since the final step's error "
                        "is the one you actually see.")),
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
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, sparsity_ratio=0.90, block_size="64",
                min_seq_len=8192, dense_last_steps=0, protect_audio=True,
                enabled=True) -> io.NodeOutput:
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
                protect_audio=protect_audio,
            )
        except Exception:                                # noqa: BLE001
            # Triton missing, an incompatible GPU, a ComfyUI API change -- none
            # of it should cost the user their run. Dense attention still works.
            log.exception("[H3Utils] SLA patch failed; passing the model through "
                          "unchanged (attention will NOT be sparsified).")
            return io.NodeOutput(model)

        return io.NodeOutput(patched)
