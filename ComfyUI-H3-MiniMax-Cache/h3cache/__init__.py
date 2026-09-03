"""Whole-block-stack residual cache for MiniMax-H3.

Ported from silveroxides/ComfyUI-UtilsCollection's UC_MiniMaxH3Cache (MIT
License), with the author's permission, in exchange for this pack's H3 SLA
Attention node. See THIRD_PARTY_NOTICES in this folder's parent README.
"""

from __future__ import annotations

from .patch import patch_h3_minimax_cache

__all__ = ["patch_h3_minimax_cache"]
