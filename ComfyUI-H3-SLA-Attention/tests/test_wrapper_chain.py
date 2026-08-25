"""Regression tests for ComfyUI DIFFUSION_MODEL wrapper composition.

This file deliberately avoids importing Triton or ComfyUI. It loads SLA's
``patch.py`` with tiny dependency stubs and emulates the current ComfyUI
``WrapperExecutor`` contract closely enough to prove that SLA advances the
wrapper chain instead of jumping directly to the original model method.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_PATCH = _ROOT / "sla" / "patch.py"
_PACKAGE = "h3_sla_wrapper_chain_test"


def _load_patch_module():
    """Load patch.py without requiring torch, Triton, or a ComfyUI checkout."""

    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()
    torch_stub.float16 = object()
    sys.modules.setdefault("torch", torch_stub)

    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(_ROOT)]
    sys.modules[_PACKAGE] = package

    sla_package = types.ModuleType(f"{_PACKAGE}.sla")
    sla_package.__path__ = [str(_ROOT / "sla")]
    sys.modules[f"{_PACKAGE}.sla"] = sla_package

    block_map = types.ModuleType(f"{_PACKAGE}.sla.block_map")
    block_map.get_block_map = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("attention routing is outside this wrapper-chain test")
    )
    block_map.get_protected_block_ranges = lambda *args, **kwargs: ()
    sys.modules[block_map.__name__] = block_map

    kernel = types.ModuleType(f"{_PACKAGE}.sla.kernel")
    kernel.block_sparse_attention = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("attention kernel is outside this wrapper-chain test")
    )
    sys.modules[kernel.__name__] = kernel

    spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE}.sla.patch",
        _PATCH,
        submodule_search_locations=None,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sla_patch = _load_patch_module()


class WrapperExecutor:
    """Minimal behavioral copy of ComfyUI's current WrapperExecutor."""

    def __init__(self, original, wrappers, idx=0):
        self.original = original
        self.wrappers = list(wrappers)
        self.idx = idx
        self.is_last = idx == len(self.wrappers)

    def __call__(self, *args, **kwargs):
        return WrapperExecutor(self.original, self.wrappers, self.idx + 1).execute(
            *args, **kwargs
        )

    def execute(self, *args, **kwargs):
        if self.is_last:
            return self.original(*args, **kwargs)
        return self.wrappers[self.idx](self, *args, **kwargs)


class WrapperChainRegression(unittest.TestCase):
    def test_run_reset_clears_motion_history(self):
        """GPU LUTs and choices from one video must not survive its run."""

        state = sla_patch._new_state()
        marker = object()
        state["prev_lut"][0] = marker
        state["call_idx"] = 17

        sla_patch._reset_run_state(state)

        self.assertEqual(state["prev_lut"], {})
        self.assertEqual(state["call_idx"], 0)

    def test_end_of_run_releases_motion_history(self):
        """Release history immediately rather than waiting for another run."""

        state = sla_patch._new_state()
        state["prev_lut"][0] = object()
        sla_wrapper = sla_patch._make_wrapper(state, 0.90, 64, 64, 0)
        executor = WrapperExecutor(lambda *a, **k: None, [sla_wrapper])

        executor.execute(
            object(),
            object(),
            object(),
            transformer_options={"sample_sigmas": [1.0, 0.0]},
        )

        self.assertEqual(state["prev_lut"], {})

    def test_sla_advances_to_downstream_wrapper_before_original(self):
        events = []
        state = sla_patch._new_state()
        sla_wrapper = sla_patch._make_wrapper(
            state,
            sparsity_ratio=0.90,
            blkq=64,
            blkk=64,
            dense_last_steps=0,
        )

        def downstream(executor, *args, **kwargs):
            events.append("downstream")
            return executor(*args, **kwargs)

        def original(*args, **kwargs):
            events.append("original")
            return "ok"

        transformer_options = {"sample_sigmas": [1.0, 0.0]}
        executor = WrapperExecutor(original, [sla_wrapper, downstream])
        result = executor.execute(
            object(),
            object(),
            object(),
            transformer_options=transformer_options,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(events, ["downstream", "original"])

    def test_h3_payload_is_forwarded_through_the_chain(self):
        seen = {}
        state = sla_patch._new_state()
        sla_wrapper = sla_patch._make_wrapper(state, 0.90, 64, 64, 0)

        def downstream(executor, *args, **kwargs):
            seen.update(kwargs)
            return executor(*args, **kwargs)

        executor = WrapperExecutor(lambda *a, **k: None, [sla_wrapper, downstream])
        payload = {"layout": None}
        executor.execute(
            object(),
            object(),
            object(),
            transformer_options={"sample_sigmas": [1.0, 0.0]},
            minimax_payload=payload,
        )

        self.assertIs(seen["minimax_payload"], payload)

    def test_audio_range_is_read_without_pinning_reference_segments(self):
        state = sla_patch._new_state()
        sla_wrapper = sla_patch._make_wrapper(state, 0.90, 64, 64, 0)
        seen = {}

        class Layout:
            segments = [
                (0, 512, "text"),
                (512, 8192, "cond"),
                (8192, 10192, "audio"),
                (10192, 120000, "video"),
            ]

        def downstream(executor, *args, **kwargs):
            seen.update(kwargs["transformer_options"])
            return executor(*args, **kwargs)

        executor = WrapperExecutor(
            lambda *a, **k: None, [sla_wrapper, downstream]
        )
        executor.execute(
            object(), object(), object(),
            transformer_options={"sample_sigmas": [1.0, 0.0]},
            minimax_payload={"layout": Layout()},
        )

        self.assertEqual(seen["_h3sla_protected_ranges"], ((8192, 10192),))
        self.assertEqual(seen["_h3sla_prefix"], 10192)

    def test_spectrum_forecasts_do_not_merge_sla_run_state(self):
        """Skipped NFEs must not make SLA count across independent sampler runs."""

        state = sla_patch._new_state()
        sla_wrapper = sla_patch._make_wrapper(state, 0.90, 64, 64, 0)
        summaries = []
        original_summarise = sla_patch._summarise

        def capture_summary(current_state, *args):
            summaries.append(current_state["calls"])

        def downstream(executor, *args, **kwargs):
            # MiniMax-H3 has 50 transformer blocks, hence 50 attention calls
            # for each actual model evaluation.
            state["calls"] += 50
            return executor(*args, **kwargs)

        executor = WrapperExecutor(lambda *a, **k: None, [sla_wrapper, downstream])

        schedule19 = [float(19 - i) for i in range(20)]
        schedule3 = [3.0, 2.0, 1.0, 0.0]

        def run(schedule, actual_indices):
            for idx in actual_indices:
                executor.execute(
                    object(),
                    object(),
                    object(),
                    transformer_options={
                        "sample_sigmas": schedule,
                        "sigmas": [schedule[idx]],
                    },
                )

        try:
            sla_patch._summarise = capture_summary

            # Mirrors the observed Spectrum patterns: 10/19 actuals, then
            # 12/19 actuals, then two 2/3-actual refinement passes.
            run(schedule19, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
            run(schedule19, [0, 1, 3, 5, 7, 8, 10, 12, 14, 15, 17, 18])
            run(schedule3, [0, 2])
            run(schedule3, [0, 2])
        finally:
            sla_patch._summarise = original_summarise

        self.assertEqual(summaries, [500, 600, 100, 100])
        self.assertTrue(state["summarized"])
        self.assertEqual(state["step"], 3)
        self.assertEqual(state["last_step_index"], 2)

    def test_dense_last_steps_uses_logical_sampler_position(self):
        """Forecasted steps must not shift SLA's trailing-dense window."""

        state = sla_patch._new_state()
        sla_wrapper = sla_patch._make_wrapper(state, 0.90, 64, 64, 2)
        dense_flags = []

        def downstream(executor, *args, **kwargs):
            dense_flags.append(kwargs["transformer_options"]["_h3sla_dense"])
            return executor(*args, **kwargs)

        executor = WrapperExecutor(lambda *a, **k: None, [sla_wrapper, downstream])
        schedule = [float(19 - i) for i in range(20)]

        # Only actual model evaluations reach SLA. Steps 17 and 18 are the
        # logical last two sampler steps even though they are just calls 3/4.
        for idx in [0, 5, 17, 18]:
            executor.execute(
                object(),
                object(),
                object(),
                transformer_options={
                    "sample_sigmas": schedule,
                    "sigmas": [schedule[idx]],
                },
            )

        self.assertEqual(dense_flags, [False, False, True, True])

    def test_legacy_non_callable_test_executor_still_works(self):
        """Keep compatibility with the package's older unit-test executor shim."""

        seen = []

        class LegacyExecutor:
            @staticmethod
            def original(*args, **kwargs):
                seen.append("original")
                return "ok"

        result = sla_patch._call_next_wrapper(LegacyExecutor, 1, 2, three=3)
        self.assertEqual(result, "ok")
        self.assertEqual(seen, ["original"])


if __name__ == "__main__":
    unittest.main()
