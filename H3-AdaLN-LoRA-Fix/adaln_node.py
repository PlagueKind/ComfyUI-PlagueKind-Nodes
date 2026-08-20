"""The H3 AdaLN LoRA Fix node.

Drop it on the MODEL wire anywhere after the LoRA loaders -- rgthree's Power Lora
Loader, ComfyUI's own, any stack. It works on the patches those loaders already
attached, so it needs no knowledge of which LoRAs were chosen and nothing upstream
has to change.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

import adaln_patch

log = logging.getLogger("H3AdaLN")

MODES = ("port", "strip", "off")


class H3AdaLNLoRAFix(io.ComfyNode):
    """Make dense H3 LoRAs work on a pruned (curve-form) H3 base.

    A pruned checkpoint stores its AdaLN projections over an 8-wide curve basis
    instead of the dense 2688-wide time embedding, so a LoRA trained on the full
    base cannot be applied to them: ComfyUI logs one ``ERROR lora ...`` line per key
    and drops all 51. This node rebases those weights so they apply properly, which
    silences the log and restores the timestep-modulation weights.

    Measured on real H3 turbo LoRAs that restored contribution is only ~0.02 % of
    the modulation signal, so the clean log is the real benefit -- do not expect a
    visible quality change. See ADALN.md for the numbers.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3AdaLNLoRAFix",
            display_name="H3 AdaLN LoRA Fix",
            category="PlagueKind/model_patches/minimax",
            description=(
                "Rebases dense MiniMax-H3 LoRA AdaLN weights onto a pruned "
                "(curve-form) checkpoint, so they are applied instead of being "
                "skipped with 51 'ERROR lora ... adaln_proj' lines. Place it after "
                "your LoRA loader. Works in both directions: dense LoRA on a "
                "pruned base, or curve-form LoRA on a full one."),
            inputs=[
                io.Model.Input("model",
                    tooltip="MODEL, after the LoRA loader(s)."),
                io.Combo.Input("mode", options=list(MODES), default="port",
                    tooltip=(
                        "port: rebase the AdaLN weights onto the basis the model "
                        "actually uses - quiet log, and the timestep modulation is "
                        "restored. "
                        "strip: just drop the incompatible weights - quiet log, "
                        "output identical to having no fix at all. "
                        "off: passthrough, leaves the errors in place.")),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, mode="port") -> io.NodeOutput:
        try:
            patched, report = adaln_patch.fix_model(model, mode)
        except Exception:                                # noqa: BLE001
            # A failure here must never take the workflow down: the un-fixed model
            # still generates, it just logs the errors it logged before.
            log.exception("[H3AdaLN] AdaLN LoRA fix failed; passing the model "
                          "through unchanged.")
            return io.NodeOutput(model)

        log.info("[H3AdaLN] %s", adaln_patch.format_report(report))
        for note in report["notes"][:8]:
            log.warning("[H3AdaLN]   %s", note)
        if len(report["notes"]) > 8:
            log.warning("[H3AdaLN]   ... and %d more",
                        len(report["notes"]) - 8)
        return io.NodeOutput(patched)
