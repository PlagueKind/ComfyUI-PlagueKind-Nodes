"""ComfyUI-H3-SLA-Attention.

H3 SLA Attention gives MiniMax-H3 the block-sparse attention backend that
ComfyUI does not otherwise have -- the inference path the lightx2v SLA turbo
LoRA was distilled against, and the reason that LoRA produces no speedup on
its own. H3 Adaptive Feed Forward is a separate, orthogonal MODEL patch --
see its own module docstring for why it exists, what it does and does not
overlap with H3SLAAttention, and why it does not reimplement kijai's
del-based low-VRAM attention restructuring (current ComfyUI's
AttentionTensorContainer already does the equivalent for any
optimized_attention_override node, including H3SLAAttention, automatically).
See README.md.

Registration is deliberately defensive: if anything in this package fails to
import -- missing Triton, an unsupported GPU, a ComfyUI API change -- we log the
traceback and expose no entrypoint, so ComfyUI skips the pack instead of failing
to start. A broken node must never stop ComfyUI from booting.

Note we must NOT define ``NODE_CLASS_MAPPINGS`` here, not even as an empty dict.
ComfyUI's loader (nodes.py, "V1 node definition") checks it with ``is not None``
and returns early when it exists -- so an empty mapping would shadow
``comfy_entrypoint`` and silently register zero nodes.

To remove the pack entirely, delete this folder.
"""

import logging

log = logging.getLogger("H3Utils")

_EXTENSION = None
try:
    from comfy_api.latest import ComfyExtension

    from .sla_node import H3SLAAttention
    from .h3_adaptive_feedforward import H3AdaptiveFeedForward

    class H3SLAExtension(ComfyExtension):
        async def get_node_list(self):
            return [H3SLAAttention, H3AdaptiveFeedForward]

    _EXTENSION = H3SLAExtension
except Exception:  # noqa: BLE001 - never block ComfyUI startup
    log.exception("[H3Utils] SLA failed to load; the node will not be available. "
                  "ComfyUI startup is unaffected.")

if _EXTENSION is not None:
    async def comfy_entrypoint():
        return _EXTENSION()

    __all__ = ["comfy_entrypoint"]
else:
    __all__ = []
