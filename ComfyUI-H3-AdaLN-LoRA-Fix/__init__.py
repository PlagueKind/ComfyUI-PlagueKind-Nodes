"""ComfyUI-H3-AdaLN-LoRA-Fix.

One node, H3 AdaLN LoRA Fix, which makes dense (full-base) MiniMax-H3 LoRAs work on
pruned / curve-form H3 checkpoints instead of being skipped with 51
``ERROR lora ... adaln_proj`` lines per model load. See README.md.

Registration is deliberately defensive: if anything in this package fails to import,
we log the traceback and expose no entrypoint, so ComfyUI skips the pack instead of
failing to start. A broken node must never stop ComfyUI from booting.

Note we must NOT define ``NODE_CLASS_MAPPINGS`` here, not even as an empty dict.
ComfyUI's loader (nodes.py, "V1 node definition") checks it with ``is not None`` and
returns early when it exists -- so an empty mapping would shadow ``comfy_entrypoint``
and silently register zero nodes.

To remove the pack entirely, delete this folder.
"""

import logging

log = logging.getLogger("H3AdaLN")

_EXTENSION = None
try:
    from comfy_api.latest import ComfyExtension

    from .adaln_node import H3AdaLNLoRAFix

    class H3AdaLNExtension(ComfyExtension):
        async def get_node_list(self):
            return [H3AdaLNLoRAFix]

    _EXTENSION = H3AdaLNExtension
except Exception:  # noqa: BLE001 - never block ComfyUI startup
    log.exception("[H3AdaLN] failed to load; the node will not be available. "
                  "ComfyUI startup is unaffected.")

if _EXTENSION is not None:
    async def comfy_entrypoint():
        return _EXTENSION()

    __all__ = ["comfy_entrypoint"]
else:
    __all__ = []
