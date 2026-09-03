"""The H3 MiniMax Cache node.

Drop it on the MODEL wire, anywhere relative to H3 SLA Attention -- the two
patch different things (this one wraps the block-loop boundary and the outer
sampling call; SLA replaces per-block attention) and don't touch the same
object-patch keys, so order between them doesn't matter. Last before the
sampler either way.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

log = logging.getLogger("H3Utils")

DEVICE_MODES = ("auto", "cuda", "cpu")


class H3MiniMaxCache(io.ComfyNode):
    """Reuse MiniMax-H3's whole-block-stack residual across similar steps.

    H3 is a 33B model, so every sampling step is expensive and video sampling
    means a lot of steps. This patches a *cloned* model patcher only -- via
    ``ModelPatcher.add_object_patch`` on ``diffusion_model._forward`` plus a
    ``block_loop`` replacement and an ``OUTER_SAMPLE`` wrapper -- so nothing
    about the Core model class is touched globally. If the cache misbehaves,
    only the clone you fed the sampler is affected.

    Ported from silveroxides/ComfyUI-UtilsCollection's UC_MiniMaxH3Cache
    (MIT), with permission, in exchange for this pack's H3 SLA Attention
    node. See h3cache/patch.py for the implementation.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3MiniMaxCache",
            display_name="H3 MiniMax Cache",
            category="PlagueKind/model_patches/minimax",
            description=(
                "Approximate block-stack residual cache for MiniMax-H3. "
                "Skips re-running the transformer block stack on steps "
                "whose features haven't moved much, on a cloned model only."),
            inputs=[
                io.Model.Input("model",
                    tooltip="MiniMax-H3 MODEL to patch."),
                io.Float.Input("reuse_threshold", default=0.05, min=0.0,
                    max=1.0, step=0.01,
                    tooltip=(
                        "Maximum accumulated relative feature change allowed "
                        "before a skip is refused. Higher skips more work "
                        "and can cost fidelity; lower runs more real steps. "
                        "Start here and only raise it if quality holds.")),
                io.Float.Input("start_percent", default=0.15, min=0.0,
                    max=1.0, step=0.01,
                    tooltip=(
                        "Sampling progress at which cache reuse may begin. "
                        "Early steps set structure, so caching is withheld "
                        "until this point by default.")),
                io.Float.Input("end_percent", default=0.90, min=0.0,
                    max=1.0, step=0.01,
                    tooltip=(
                        "Sampling progress after which cache reuse stops. "
                        "Late steps polish detail, so the last stretch runs "
                        "dense by default.")),
                io.Int.Input("max_steps", default=2, min=1, max=10, step=1,
                    tooltip="Maximum number of consecutive block-stack skips."),
                io.Combo.Input("device", options=list(DEVICE_MODES),
                    default="auto",
                    tooltip=(
                        "auto keeps the cached residual with the model, "
                        "cuda requires CUDA, cpu offloads the residual to "
                        "system RAM (useful on tight VRAM).")),
                io.Boolean.Input("verbose", default=False,
                    tooltip="Log per-step cache decisions and a final skip summary."),
            ],
            outputs=[io.Model.Output()],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, reuse_threshold=0.05, start_percent=0.15,
                end_percent=0.90, max_steps=2, device="auto",
                verbose=False) -> io.NodeOutput:
        try:
            from .h3cache import patch_h3_minimax_cache
            patched = patch_h3_minimax_cache(
                model,
                reuse_threshold=reuse_threshold,
                start_percent=start_percent,
                end_percent=end_percent,
                max_steps=max_steps,
                device=device,
                verbose=verbose,
            )
        except Exception:                                # noqa: BLE001
            log.exception("[H3Utils] MiniMax Cache patch failed; passing the "
                          "model through unchanged (no caching will occur).")
            return io.NodeOutput(model)

        return io.NodeOutput(patched)
