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
