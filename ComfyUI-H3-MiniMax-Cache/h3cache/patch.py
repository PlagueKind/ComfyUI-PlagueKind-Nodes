"""Whole-block-stack residual cache for MiniMax-H3.

Ported from silveroxides/ComfyUI-UtilsCollection's ``patcher_helpers.py``
(MiniMaxH3Cache / MiniMaxH3SamplingScope / run_minimax_h3_blocks /
minimax_h3_block_patch_forward / patch_minimax_h3_cache_model), MIT License,
with the author's permission, in exchange for this pack's H3 SLA Attention
node.

Current Core exposes per-block replacements but no replacement boundary
around the *complete* H3 block stack. Core does, however, route the whole
stack through ``MiniMaxH3Model._run_blocks``. ``h3_cache_run_blocks`` wraps
exactly that method with a "block_loop" replacement boundary, installed via
``ModelPatcher.add_object_patch`` on a cloned model only -- Core's model class
is never touched globally, and the original bound method is restored whenever
the clone is unpatched.

Everything outside the block stack (packing, continuation/cond_audio rows,
per-row denoise masks, the final PDD layer) stays Core's own code, so Core
updates to ``MiniMaxH3Model._forward`` no longer desync this file.
"""

from __future__ import annotations

import logging
import types
from collections.abc import Callable
from typing import Any

import torch

import comfy.patcher_extension
from comfy.ldm.minimax import model as minimax_model

log = logging.getLogger("H3Utils")

H3_MINIMAX_CACHE_OWNER_KEY = "plaguekind_h3_minimax_cache"


class H3CacheState:
    """Reuse the residual produced by the complete MiniMax H3 block stack."""

    def __init__(
        self,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        max_steps: int,
        device: str,
        verbose: bool,
    ) -> None:
        self.reuse_threshold = reuse_threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.max_steps = max_steps
        self.device = device
        self.verbose = verbose
        self.total_steps = 1
        self._residual_buffer: torch.Tensor | None = None
        self._cache_valid = False
        self.reset()

    def reset(self) -> None:
        # Deliberately do NOT drop/free self._residual_buffer here. This
        # method runs inside H3CacheSamplingScope's `finally` (via finish())
        # as well as mid-generation on a layout change -- i.e. it can run
        # after comfy.model_prefetch's malloc_scope="block" region for the
        # block stack has already been torn down / recycled. Letting the
        # buffer's refcount hit zero at that point (the old code's
        # `del residual`) frees a tensor whose storage may be associated
        # with an already-closed scope, which is what produced the CUDA
        # invalid-argument abort in cuMemFreeAsync. Instead we just mark the
        # cache logically empty and keep reusing the same buffer in-place
        # (see _store_residual). It only ever gets released via ordinary
        # Python GC when this whole H3CacheState -- and the model patcher
        # that owns it -- is discarded during a normal model unload, which
        # comfy's own memory manager controls and expects.
        self._cache_valid = False
        self.previous_feature_signature: torch.Tensor | None = None
        self.layout_signature: tuple[Any, ...] | None = None
        self.last_seen_timestep: float | None = None
        self.step_counter = 0
        self.accumulated_relative_l1 = 0.0
        self.consecutive_skips = 0
        self.run_count = 0
        self.skip_count = 0

    def begin(self, total_steps: int) -> None:
        self.reset()
        self.total_steps = max(1, total_steps)

    def finish(self) -> None:
        if self.verbose and self.run_count + self.skip_count:
            total = self.run_count + self.skip_count
            speedup = total / max(1, self.run_count)
            log.info(
                "[H3Utils] MiniMax Cache skipped %s/%s block-stack "
                "executions (%.2fx theoretical block-stack speedup).",
                self.skip_count,
                total,
                speedup,
            )
        self.reset()

    @staticmethod
    def _feature_signature(
        hidden_states: torch.Tensor,
        cache_ranges: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        max_dim = min(64, hidden_states.shape[-1])
        signatures = []
        for start, end in cache_ranges:
            length = end - start
            if length <= 0:
                continue
            stride = max(1, length // 100)
            sampled = hidden_states[start:end:stride, :max_dim]
            signatures.append(sampled.detach().abs().mean(dim=-1))

        if not signatures:
            stride = max(1, hidden_states.shape[0] // 100)
            sampled = hidden_states[::stride, :max_dim]
            return sampled.detach().abs().mean(dim=-1).float().cpu()
        return torch.cat(signatures).float().cpu()

    @staticmethod
    def _timestep_value(timestep: Any) -> float | None:
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 0:
                return None
            return float(timestep.detach().flatten()[0].item())
        if isinstance(timestep, (int, float)):
            return float(timestep)
        return None

    @property
    def cached_residual(self) -> torch.Tensor | None:
        return self._residual_buffer if self._cache_valid else None

    def _store_residual(self, residual: torch.Tensor) -> None:
        if self.device == "cuda" and residual.device.type != "cuda":
            raise ValueError(
                "H3 MiniMax Cache device is set to cuda, but the model is not running on CUDA."
            )

        target_device = torch.device("cpu") if self.device == "cpu" else residual.device
        residual = residual.detach()
        try:
            buf = self._residual_buffer
            # Overwrite the existing buffer in place instead of freeing it
            # and allocating a new one every RUN step. A realloc is only
            # needed on the first store or when shape/dtype/device actually
            # changes (e.g. after a layout change already called reset()).
            # copy_() into a stable, long-lived buffer never triggers the
            # free-of-scoped-memory path that reset() used to hit.
            if (buf is None or buf.shape != residual.shape or buf.dtype != residual.dtype
                    or buf.device != target_device):
                self._residual_buffer = torch.empty(
                    residual.shape, dtype=residual.dtype, device=target_device
                )
                buf = self._residual_buffer
            buf.copy_(residual, non_blocking=(target_device.type == "cuda"))
            self._cache_valid = True
        except torch.OutOfMemoryError:
            if self.device == "cuda":
                raise
            self._residual_buffer = residual.to("cpu", copy=True)
            self._cache_valid = True

    def _apply_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.cached_residual
        if residual is None:
            return hidden_states
        if residual.device != hidden_states.device or residual.dtype != hidden_states.dtype:
            residual = residual.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                non_blocking=True,
            )
        return hidden_states + residual

    def __call__(self, args: dict[str, Any], extra_options: dict[str, Any]) -> dict[str, Any]:
        original_block = extra_options["original_block"]
        hidden_states = args["img"]
        cache_ranges = tuple(tuple(pair) for pair in args.get("cache_ranges", ()))
        current_layout = (
            tuple(hidden_states.shape),
            hidden_states.dtype,
            hidden_states.device,
            args.get("block_count"),
            cache_ranges,
        )

        if self.layout_signature is None:
            self.layout_signature = current_layout
        elif self.layout_signature != current_layout:
            total_steps = self.total_steps
            self.reset()
            self.total_steps = total_steps
            self.layout_signature = current_layout

        timestep = self._timestep_value(args.get("timestep"))
        if timestep is None:
            return original_block(args)
        if self.last_seen_timestep != timestep:
            self.last_seen_timestep = timestep
            self.step_counter += 1

        progress = self.step_counter / self.total_steps
        in_cache_range = self.start_percent <= progress <= self.end_percent
        skip_reason = "initial step"

        if self.cached_residual is not None and self.previous_feature_signature is not None:
            current_signature = self._feature_signature(hidden_states, cache_ranges)
            current_float = current_signature.float()
            previous_float = self.previous_feature_signature.float()
            difference = (current_float - previous_float).abs().mean().item()
            denominator = previous_float.abs().mean().item() + 1e-6
            self.accumulated_relative_l1 += difference / denominator

            below_threshold = self.accumulated_relative_l1 < self.reuse_threshold
            below_skip_limit = self.consecutive_skips < self.max_steps
            if below_threshold and below_skip_limit and in_cache_range:
                self.skip_count += 1
                self.consecutive_skips += 1
                if self.verbose:
                    log.info(
                        "[H3Utils] MiniMax Cache step %s SKIP "
                        "(relative L1 %.4f < %.4f).",
                        self.step_counter,
                        self.accumulated_relative_l1,
                        self.reuse_threshold,
                    )
                return {"img": self._apply_residual(hidden_states)}

            reasons = []
            if not below_threshold:
                reasons.append("threshold reached")
            if not below_skip_limit:
                reasons.append("maximum consecutive skips reached")
            if not in_cache_range:
                reasons.append("outside cache range")
            skip_reason = ", ".join(reasons)

        if self.verbose:
            log.info(
                "[H3Utils] MiniMax Cache step %s RUN (%s).",
                self.step_counter,
                skip_reason,
            )

        self.run_count += 1
        self.consecutive_skips = 0
        self._cache_valid = False
        self.previous_feature_signature = self._feature_signature(hidden_states, cache_ranges)
        start_hidden_states = hidden_states.clone()
        result = original_block(args)
        output = result["img"]
        self._store_residual(output - start_hidden_states)
        self.accumulated_relative_l1 = 0.0
        return result


class H3CacheSamplingScope:
    """Give one cache state object an exact outer-sampling lifecycle."""

    def __init__(self, cache: H3CacheState) -> None:
        self.cache = cache

    def __call__(self, sample_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        sigmas = kwargs.get("sigmas")
        if sigmas is None and len(args) > 3:
            sigmas = args[3]
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 2:
            self.cache.reset()
            raise ValueError("H3 MiniMax Cache could not read the sampler sigma schedule.")

        self.cache.begin(len(sigmas) - 1)
        try:
            return sample_fn(*args, **kwargs)
        finally:
            self.cache.finish()


def h3_cache_run_blocks(
    self: minimax_model.MiniMaxH3Model,
    hidden_states: torch.Tensor,
    t_emb: torch.Tensor,
    mod_segments: Any,
    layout: Any,
    transformer_options: dict[str, Any],
) -> torch.Tensor:
    """Core ``_run_blocks`` behind one model-patchable "block_loop" boundary."""

    core_run_blocks = minimax_model.MiniMaxH3Model._run_blocks
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    if ("block_loop", 0) not in blocks_replace:
        return core_run_blocks(
            self, hidden_states, t_emb, mod_segments, layout, transformer_options
        )

    def block_loop_wrapper(block_args: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            "img": core_run_blocks(
                self,
                block_args["img"],
                block_args["t_emb"],
                block_args["mod_segments"],
                layout,
                block_args["transformer_options"],
            )
        }

    target_ranges = tuple(
        (start, end, kind)
        for start, end, kind in layout.segments
        if kind in ("audio", "video")
    )
    return blocks_replace[("block_loop", 0)](
        {
            "img": hidden_states,
            "timestep": transformer_options.get("sigmas"),
            "t_emb": t_emb,
            "mod_segments": mod_segments,
            "layout": layout,
            "transformer_options": transformer_options,
            "cache_ranges": tuple((start, end) for start, end, _ in target_ranges),
            "target_ranges": target_ranges,
            "block_count": len(self.blocks),
        },
        {"original_block": block_loop_wrapper},
    )["img"]


def patch_h3_minimax_cache(
    model: Any,
    reuse_threshold: float,
    start_percent: float,
    end_percent: float,
    max_steps: int,
    device: str,
    verbose: bool,
) -> Any:
    """Return a cloned patcher with a reversible H3 block-loop cache."""

    if start_percent > end_percent:
        raise ValueError("Cache start percent must not exceed end percent.")

    patched_model = model.clone()
    diffusion_model = patched_model.model.diffusion_model
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError(
            "H3 MiniMax Cache requires a MiniMax H3 diffusion model; "
            f"received {diffusion_model.__class__.__name__}."
        )

    cache = H3CacheState(
        reuse_threshold=reuse_threshold,
        start_percent=start_percent,
        end_percent=end_percent,
        max_steps=max_steps,
        device=device,
        verbose=verbose,
    )
    if hasattr(patched_model, "model_options"):
        patched_model.model_options[H3_MINIMAX_CACHE_OWNER_KEY] = True
    if not hasattr(minimax_model.MiniMaxH3Model, "_run_blocks"):
        raise RuntimeError(
            "H3 MiniMax Cache requires a ComfyUI version whose MiniMaxH3Model "
            "provides _run_blocks; please update ComfyUI."
        )
    bound_run_blocks = types.MethodType(h3_cache_run_blocks, diffusion_model)
    patched_model.add_object_patch("diffusion_model._run_blocks", bound_run_blocks)
    patched_model.set_model_patch_replace(cache, "dit", "block_loop", 0)
    patched_model.add_wrapper(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        H3CacheSamplingScope(cache),
    )
    return patched_model
