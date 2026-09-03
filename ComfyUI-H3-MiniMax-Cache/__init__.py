"""ComfyUI-H3-MiniMax-Cache.

One node, H3 MiniMax Cache, which adds a whole-block-stack residual cache to
MiniMax-H3 sampling: when consecutive steps produce near-identical features
it reuses the last block-stack residual instead of recomputing it. Ported
from silveroxides/ComfyUI-UtilsCollection's UC_MiniMaxH3Cache (MIT) with the
author's permission, given in exchange for this pack's H3 SLA Attention node.
See README.md.

Registration is deliberately defensive: if anything in this package fails to
import -- an unsupported Core version, a ComfyUI API change -- we log the
traceback and expose no entrypoint, so ComfyUI skips the pack instead of
failing to start. A broken node must never stop ComfyUI from booting.

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

    from .cache_node import H3MiniMaxCache

    class H3MiniMaxCacheExtension(ComfyExtension):
        async def get_node_list(self):
            return [H3MiniMaxCache]

    _EXTENSION = H3MiniMaxCacheExtension
except Exception:  # noqa: BLE001 - never block ComfyUI startup
    log.exception("[H3Utils] MiniMax Cache failed to load; the node will not be "
                  "available. ComfyUI startup is unaffected.")

if _EXTENSION is not None:
    async def comfy_entrypoint():
        return _EXTENSION()

    __all__ = ["comfy_entrypoint"]
else:
    __all__ = []
