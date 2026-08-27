"""Dependency-free tests for visual-reference quota arithmetic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


def _load_block_map_helpers():
    root = Path(__file__).resolve().parents[1]
    source = root / "sla" / "block_map.py"

    torch_stub = types.ModuleType("torch")
    triton_stub = types.ModuleType("triton")
    triton_stub.__path__ = []
    triton_stub.jit = lambda function: function
    language_stub = types.ModuleType("triton.language")

    names = ("torch", "triton", "triton.language")
    previous = {name: sys.modules.get(name) for name in names}
    sys.modules["torch"] = torch_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "h3_sla_reference_quota_test", source
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


block_map = _load_block_map_helpers()


class ReferenceQuotaArithmetic(unittest.TestCase):
    def test_block_map_uses_its_protect_ranges_argument(self):
        code = block_map.get_block_map.__code__
        self.assertIn("protect_ranges", code.co_varnames)
        self.assertNotIn("protected_ranges", code.co_names)

    def test_reference_ranges_exclude_already_protected_blocks(self):
        ranges = block_map.get_reference_quota_block_ranges(
            reference_ranges=((64, 192), (256, 512)),
            protect_upto=0,
            protected_ranges=((128, 320),),
            BLKK=64,
            NK=16,
        )
        self.assertEqual(ranges, ((1, 2), (5, 8)))

    def test_legacy_protected_prefix_is_also_subtracted(self):
        ranges = block_map.get_reference_quota_block_ranges(
            reference_ranges=((0, 256),),
            protect_upto=128,
            protected_ranges=(),
            BLKK=64,
            NK=16,
        )
        self.assertEqual(ranges, ((2, 4),))

    def test_manual_sparsity_rounds_up_to_the_promised_minimum(self):
        keep = block_map.get_reference_quota_keep_count
        self.assertEqual(keep(10, 0.80), 2)
        self.assertEqual(keep(10, 0.90), 1)
        self.assertEqual(keep(3, 0.0), 3)
        self.assertEqual(keep(3, 0.99), 1)
        self.assertEqual(keep(10, None), 0)


if __name__ == "__main__":
    unittest.main()
