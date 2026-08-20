"""AdaLN LoRA porting: basis recovery, port maths, and patch surgery.

The port is only worth doing if it reproduces what the dense LoRA would have done,
so the central test builds a grid that is exactly affine in the table and asserts
the ported patches match the dense contribution to floating-point tolerance.

Skipped automatically when torch or ComfyUI are not importable.
"""

import importlib.util
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))  # custom_nodes/<pkg> -> root

try:
    import torch
    if _COMFY_ROOT not in sys.path:
        sys.path.insert(0, _COMFY_ROOT)
    _spec = importlib.util.spec_from_file_location(
        "h3u", os.path.join(_PKG_DIR, "__init__.py"),
        submodule_search_locations=[_PKG_DIR])
    h3u = importlib.util.module_from_spec(_spec)
    sys.modules["h3u"] = h3u
    _spec.loader.exec_module(h3u)
    from h3u import adaln, adaln_patch
    AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    AVAILABLE = False
    REASON = str(exc)

try:
    from comfy.model_patcher import ModelPatcher
    from comfy.weight_adapter import LoRAAdapter
    PATCHER = True
except Exception as exc:  # noqa: BLE001
    PATCHER = False
    PATCHER_REASON = str(exc)


OUT, RANK, DENSE, CURVE, ROWS = 64, 4, 32, 8, 33


def affine_case(seed=0):
    """A table, an exactly-affine grid over it, and the (c, V) that generate it."""
    g = torch.Generator().manual_seed(seed)
    table = torch.randn(ROWS, CURVE, generator=g)
    basis_v = torch.randn(DENSE, CURVE, generator=g)
    c = torch.randn(DENSE, generator=g)
    grid = c[None, :] + table @ basis_v.T
    return table, grid, c, basis_v


def lora_factors(seed=1):
    g = torch.Generator().manual_seed(seed)
    lora_a = torch.randn(RANK, DENSE, generator=g)      # down
    lora_b = torch.randn(OUT, RANK, generator=g)        # up
    return lora_b, lora_a


@unittest.skipUnless(AVAILABLE, "torch/pack not importable")
class TestBasisSolve(unittest.TestCase):

    def test_recovers_known_affine_basis(self):
        table, grid, c, basis_v = affine_case()
        got_c, got_v, rel = adaln.solve_basis(table, grid)
        self.assertLess(rel, 1e-5)
        self.assertTrue(torch.allclose(got_c, c, atol=1e-4))
        self.assertTrue(torch.allclose(got_v, basis_v, atol=1e-4))

    def test_residual_reports_a_bad_fit(self):
        table, grid, _, _ = affine_case()
        grid = grid + torch.randn(grid.shape, generator=torch.Generator().manual_seed(7))
        _, _, rel = adaln.solve_basis(table, grid)
        self.assertGreater(rel, 1e-2)

    def test_row_count_mismatch_is_an_error(self):
        table, grid, _, _ = affine_case()
        with self.assertRaises(ValueError):
            adaln.solve_basis(table[:-1], grid)

    def test_table_hash_is_stable_and_discriminating(self):
        table, _, _, _ = affine_case()
        self.assertEqual(adaln.table_hash(table), adaln.table_hash(table.clone()))
        other = table.clone()
        other[0, 0] += 1.0
        self.assertNotEqual(adaln.table_hash(table), adaln.table_hash(other))


@unittest.skipUnless(AVAILABLE, "torch/pack not importable")
class TestPortMaths(unittest.TestCase):

    def test_port_reproduces_the_dense_contribution(self):
        """The whole point: ported patches must equal what the dense LoRA did."""
        table, grid, c, basis_v = affine_case()
        lora_b, lora_a = lora_factors()
        scale = 0.75

        d_w, d_b = adaln.port_delta(lora_b, lora_a, scale, c, basis_v)
        self.assertEqual(tuple(d_w.shape), (OUT, CURVE))
        self.assertEqual(tuple(d_b.shape), (OUT,))

        for row in range(ROWS):
            dense = scale * (lora_b @ (lora_a @ grid[row]))
            ported = d_w @ table[row] + d_b
            self.assertTrue(torch.allclose(dense, ported, atol=1e-4),
                            "row %d: dense and ported contributions differ" % row)

    def test_dropping_the_bias_term_would_break_it(self):
        """Guards the comment in adaln.py: the DC term is not optional."""
        table, grid, c, basis_v = affine_case()
        lora_b, lora_a = lora_factors()
        d_w, _ = adaln.port_delta(lora_b, lora_a, 1.0, c, basis_v)
        dense = lora_b @ (lora_a @ grid[0])
        self.assertFalse(torch.allclose(dense, d_w @ table[0], atol=1e-3))

    def test_scale_is_linear(self):
        table, grid, c, basis_v = affine_case()
        lora_b, lora_a = lora_factors()
        one_w, one_b = adaln.port_delta(lora_b, lora_a, 1.0, c, basis_v)
        half_w, half_b = adaln.port_delta(lora_b, lora_a, 0.5, c, basis_v)
        self.assertTrue(torch.allclose(one_w * 0.5, half_w, atol=1e-6))
        self.assertTrue(torch.allclose(one_b * 0.5, half_b, atol=1e-6))


@unittest.skipUnless(AVAILABLE, "torch/pack not importable")
class TestReversePortMaths(unittest.TestCase):
    """Curve-form LoRA applied to a dense base -- the mirror of the above."""

    def test_reverse_port_reproduces_the_curve_contribution(self):
        table, grid, c, basis_v = affine_case()
        g = torch.Generator().manual_seed(5)
        lora_a = torch.randn(RANK, CURVE, generator=g)      # curve-form: [r, k]
        lora_b = torch.randn(OUT, RANK, generator=g)
        scale = 0.75

        d_w, d_b = adaln.port_delta_reverse(lora_b, lora_a, scale, c, basis_v)
        self.assertEqual(tuple(d_w.shape), (OUT, DENSE))
        self.assertEqual(tuple(d_b.shape), (OUT,))

        for row in range(ROWS):
            curve = scale * (lora_b @ (lora_a @ table[row]))
            dense = d_w @ grid[row] + d_b
            self.assertTrue(torch.allclose(curve, dense, atol=1e-4),
                            "row %d: curve and reverse-ported differ" % row)

    def test_reverse_bias_term_is_required(self):
        table, grid, c, basis_v = affine_case()
        g = torch.Generator().manual_seed(5)
        lora_a = torch.randn(RANK, CURVE, generator=g)
        lora_b = torch.randn(OUT, RANK, generator=g)
        d_w, _ = adaln.port_delta_reverse(lora_b, lora_a, 1.0, c, basis_v)
        curve = lora_b @ (lora_a @ table[0])
        self.assertFalse(torch.allclose(curve, d_w @ grid[0], atol=1e-3))

    def test_round_trip_forward_then_reverse(self):
        """Porting to the curve basis and back must land where it started."""
        table, grid, c, basis_v = affine_case()
        lora_b, lora_a = lora_factors()
        fwd_w, fwd_b = adaln.port_delta(lora_b, lora_a, 1.0, c, basis_v)
        # feed the curve-space weight back as a rank-full "LoRA" and reverse it
        eye = torch.eye(fwd_w.shape[0])
        back_w, back_b = adaln.port_delta_reverse(eye, fwd_w, 1.0, c, basis_v)
        for row in (0, ROWS // 2, ROWS - 1):
            original = lora_b @ (lora_a @ grid[row])
            round_trip = back_w @ grid[row] + back_b + fwd_b
            self.assertTrue(torch.allclose(original, round_trip, atol=1e-3))


@unittest.skipUnless(AVAILABLE, "torch/pack not importable")
class TestSiluGrid(unittest.TestCase):

    def test_matches_comfy_time_embedder(self):
        """Our reimplementation must track comfy/ldm/minimax/model.py exactly."""
        try:
            from comfy.ldm.minimax.model import TimeEmbedder
        except Exception as exc:  # noqa: BLE001
            self.skipTest("comfy minimax model not importable: %s" % exc)

        torch.manual_seed(3)
        embedder = TimeEmbedder(adaln.FREQ_DIM, 48, 24, operations=torch.nn)
        with torch.no_grad():
            for param in embedder.parameters():
                param.copy_(torch.randn(param.shape) * 0.1)
            t = torch.arange(ROWS, dtype=torch.float32) / float(ROWS - 1)
            expected = torch.nn.functional.silu(embedder(t))
            got = adaln.silu_temb_grid(
                embedder.proj_in.weight, embedder.proj_in.bias,
                embedder.proj_out.weight, embedder.proj_out.bias, rows=ROWS)
        self.assertEqual(tuple(got.shape), (ROWS, 24))
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))


# ------------------------------------------------------------------ patch surgery


class TinyAdaln(torch.nn.Module if AVAILABLE else object):
    def __init__(self, out=OUT, curve=CURVE):
        super().__init__()
        self.linear = torch.nn.Linear(curve, out, bias=True)


class TinyBlock(torch.nn.Module if AVAILABLE else object):
    def __init__(self):
        super().__init__()
        self.adaln_proj = TinyAdaln()


class TinyDiT(torch.nn.Module if AVAILABLE else object):
    def __init__(self, table, curves=True):
        super().__init__()
        self.blocks = torch.nn.ModuleList([TinyBlock()])
        self.final_layer = TinyBlock()
        self.other = torch.nn.Linear(4, 4)
        self.register_buffer("adaln_t_table", table.clone())
        self.use_adaln_curves = curves


class TinyModel(torch.nn.Module if AVAILABLE else object):
    def __init__(self, dit):
        super().__init__()
        self.diffusion_model = dit


BLOCK_W = "diffusion_model.blocks.0.adaln_proj.linear.weight"
BLOCK_B = "diffusion_model.blocks.0.adaln_proj.linear.bias"
FINAL_W = "diffusion_model.final_layer.adaln_proj.linear.weight"
OTHER_W = "diffusion_model.other.weight"


@unittest.skipUnless(AVAILABLE and PATCHER, "comfy ModelPatcher not importable")
class TestPatchSurgery(unittest.TestCase):

    def setUp(self):
        self.table, self.grid, self.c, self.basis_v = affine_case()
        self.lora_b, self.lora_a = lora_factors()
        self.model = ModelPatcher(TinyModel(TinyDiT(self.table)),
                                  torch.device("cpu"), torch.device("cpu"))
        self._real_get_basis = adaln_patch.get_basis
        adaln_patch.get_basis = lambda table, use_cache=True: (
            self.c, self.basis_v, 1e-9, "test-basis")

    def tearDown(self):
        adaln_patch.get_basis = self._real_get_basis

    def dense_lora(self, alpha=None):
        return LoRAAdapter(set(), (self.lora_b, self.lora_a, alpha, None, None, None))

    def fitting_lora(self):
        good_a = torch.zeros(RANK, CURVE)
        return LoRAAdapter(set(), (self.lora_b, good_a, None, None, None, None))

    def add_dense(self, keys=(BLOCK_W, FINAL_W), strength=1.0, alpha=None):
        self.model.add_patches({k: self.dense_lora(alpha) for k in keys}, strength)

    # -- detection ---------------------------------------------------------

    def test_scan_finds_only_the_mismatched(self):
        self.add_dense()
        self.model.add_patches({OTHER_W: self.dense_lora()}, 1.0)
        found = adaln_patch.scan_mismatched(self.model.patches,
                                            self.model.model_state_dict())
        self.assertEqual(set(found), {BLOCK_W, FINAL_W})

    def test_scan_keeps_a_correctly_shaped_lora(self):
        self.model.add_patches({BLOCK_W: self.fitting_lora()}, 1.0)
        found = adaln_patch.scan_mismatched(self.model.patches,
                                            self.model.model_state_dict())
        self.assertEqual(found, {})

    def test_scan_ignores_plain_diff_patches(self):
        self.model.add_patches({BLOCK_W: ("diff", (torch.zeros(OUT, CURVE),))}, 1.0)
        found = adaln_patch.scan_mismatched(self.model.patches,
                                            self.model.model_state_dict())
        self.assertEqual(found, {})

    # -- modes -------------------------------------------------------------

    def test_off_is_passthrough(self):
        self.add_dense()
        out, report = adaln_patch.fix_model(self.model, "off")
        self.assertIs(out, self.model)
        self.assertEqual(report["status"], "disabled")

    def test_non_h3_base_is_passthrough(self):
        """No curve table and no time embedder: not an H3 model, leave it alone."""
        model = ModelPatcher(TinyModel(TinyDiT(self.table, curves=False)),
                             torch.device("cpu"), torch.device("cpu"))
        model.add_patches({BLOCK_W: self.dense_lora()}, 1.0)
        out, report = adaln_patch.fix_model(model, "port")
        self.assertIs(out, model)
        self.assertIn("not a MiniMax-H3 model", report["status"])

    def test_clean_model_is_passthrough(self):
        out, report = adaln_patch.fix_model(self.model, "port")
        self.assertIs(out, self.model)
        self.assertEqual(report["status"], "no mismatched AdaLN patches")

    def test_strip_removes_exactly_the_offenders(self):
        self.add_dense()
        self.model.add_patches({OTHER_W: self.dense_lora()}, 1.0)
        out, report = adaln_patch.fix_model(self.model, "strip")
        self.assertEqual(report["stripped"], 2)
        self.assertNotIn(BLOCK_W, out.patches)
        self.assertNotIn(FINAL_W, out.patches)
        self.assertNotIn(BLOCK_B, out.patches)
        self.assertIn(OTHER_W, out.patches)
        # the input model is left alone
        self.assertIn(BLOCK_W, self.model.patches)

    def test_strip_keeps_a_good_patch_on_the_same_key(self):
        self.model.add_patches({BLOCK_W: self.fitting_lora()}, 1.0)
        self.add_dense(keys=(BLOCK_W,))
        out, report = adaln_patch.fix_model(self.model, "strip")
        self.assertEqual(report["stripped"], 1)
        self.assertEqual(len(out.patches[BLOCK_W]), 1)

    # -- port --------------------------------------------------------------

    def test_port_emits_weight_and_bias_diffs(self):
        self.add_dense(keys=(BLOCK_W,))
        out, report = adaln_patch.fix_model(self.model, "port")
        self.assertEqual(report["ported"], 1)
        self.assertEqual(report["effective_mode"], "port")
        self.assertIn(BLOCK_W, out.patches)
        self.assertIn(BLOCK_B, out.patches)
        self.assertEqual(out.patches[BLOCK_W][0][1][0], "diff")
        self.assertEqual(out.patches[BLOCK_B][0][1][0], "diff")

    def test_ported_patch_reproduces_the_dense_contribution(self):
        self.add_dense(keys=(BLOCK_W,), strength=0.8)
        out, _ = adaln_patch.fix_model(self.model, "port")
        d_w = out.patches[BLOCK_W][0][1][1][0].float()
        d_b = out.patches[BLOCK_B][0][1][1][0].float()
        for row in range(ROWS):
            dense = 0.8 * (self.lora_b @ (self.lora_a @ self.grid[row]))
            self.assertTrue(
                torch.allclose(dense, d_w @ self.table[row] + d_b, atol=1e-4))

    def test_alpha_over_rank_scaling_is_honoured(self):
        self.add_dense(keys=(BLOCK_W,), alpha=float(RANK) / 2.0)
        out, _ = adaln_patch.fix_model(self.model, "port")
        d_w = out.patches[BLOCK_W][0][1][1][0].float()
        expected, _ = adaln.port_delta(self.lora_b, self.lora_a, 0.5,
                                       self.c, self.basis_v)
        self.assertTrue(torch.allclose(d_w, expected, atol=1e-4))

    def test_two_stacked_loras_accumulate(self):
        self.add_dense(keys=(BLOCK_W,), strength=1.0)
        self.add_dense(keys=(BLOCK_W,), strength=0.5)
        out, report = adaln_patch.fix_model(self.model, "port")
        self.assertEqual(report["ported"], 2)
        self.assertEqual(len(out.patches[BLOCK_W]), 1)
        d_w = out.patches[BLOCK_W][0][1][1][0].float()
        expected, _ = adaln.port_delta(self.lora_b, self.lora_a, 1.5,
                                       self.c, self.basis_v)
        self.assertTrue(torch.allclose(d_w, expected, atol=1e-4))

    def test_port_falls_back_to_strip_without_a_basis(self):
        adaln_patch.get_basis = lambda table, use_cache=True: None
        self.add_dense(keys=(BLOCK_W,))
        out, report = adaln_patch.fix_model(self.model, "port")
        self.assertEqual(report["effective_mode"], "strip")
        self.assertEqual(report["stripped"], 1)
        self.assertNotIn(BLOCK_W, out.patches)
        self.assertTrue(report["notes"])

    def test_port_refuses_an_untrustworthy_basis(self):
        adaln_patch.get_basis = lambda table, use_cache=True: (
            self.c, self.basis_v, 0.5, "bad-grid")
        self.add_dense(keys=(BLOCK_W,))
        _, report = adaln_patch.fix_model(self.model, "port")
        self.assertEqual(report["effective_mode"], "strip")
        self.assertIn("too poor to trust", report["notes"][0])

    def test_dora_is_reported_not_silently_ported(self):
        weights = (self.lora_b, self.lora_a, None, None, torch.ones(OUT), None)
        self.model.add_patches({BLOCK_W: LoRAAdapter(set(), weights)}, 1.0)
        out, report = adaln_patch.fix_model(self.model, "port")
        self.assertEqual(report["ported"], 0)
        self.assertEqual(report["unportable"], 1)
        self.assertNotIn(BLOCK_W, out.patches)

    def test_patches_uuid_changes(self):
        self.add_dense(keys=(BLOCK_W,))
        before = self.model.patches_uuid
        out, _ = adaln_patch.fix_model(self.model, "strip")
        self.assertNotEqual(before, out.patches_uuid)

    # -- reporting ---------------------------------------------------------

    def test_report_is_one_line(self):
        self.add_dense()
        _, report = adaln_patch.fix_model(self.model, "port")
        line = adaln_patch.format_report(report)
        self.assertNotIn("\n", line)
        self.assertIn("ported 2", line)
        self.assertIn("test-basis", line)


@unittest.skipUnless(AVAILABLE and PATCHER, "comfy ModelPatcher not importable")
class TestDenseBaseSurgery(unittest.TestCase):
    """A dense base with a curve-form LoRA: the case that was missed at first."""

    def setUp(self):
        self.table, self.grid, self.c, self.basis_v = affine_case()
        g = torch.Generator().manual_seed(11)
        self.lora_a = torch.randn(RANK, CURVE, generator=g)   # curve-form LoRA
        self.lora_b = torch.randn(OUT, RANK, generator=g)
        dit = TinyDiT(self.table, curves=False)               # dense: DENSE-wide adaln
        dit.blocks = torch.nn.ModuleList([_DenseBlk()])
        dit.final_layer = _DenseBlk()
        dit.time_embedder = object()                          # presence marker only
        self.model = ModelPatcher(TinyModel(dit), torch.device("cpu"),
                                  torch.device("cpu"))
        self._real = adaln_patch.resolve_basis_reverse
        adaln_patch.resolve_basis_reverse = lambda dm: (
            (self.c, self.basis_v, 1e-9, "test-table"), None)

    def tearDown(self):
        adaln_patch.resolve_basis_reverse = self._real

    def add(self, strength=1.0):
        adapter = LoRAAdapter(set(), (self.lora_b, self.lora_a, None, None, None, None))
        self.model.add_patches({BLOCK_W: adapter}, strength)

    def test_detects_the_reverse_mismatch(self):
        self.add()
        found = adaln_patch.scan_mismatched(self.model.patches,
                                            self.model.model_state_dict())
        self.assertIn(BLOCK_W, found)

    def test_reports_the_direction(self):
        self.add()
        _, report = adaln_patch.fix_model(self.model, "port")
        self.assertEqual(report["direction"], "curve->dense")
        self.assertEqual(report["ported"], 1)

    def test_ported_patch_reproduces_the_curve_contribution(self):
        self.add(strength=0.6)
        out, _ = adaln_patch.fix_model(self.model, "port")
        adapter = out.patches[BLOCK_W][0][1]
        self.assertIsInstance(adapter, LoRAAdapter)      # low rank, not materialised
        up, down = adapter.weights[0].float(), adapter.weights[1].float()
        self.assertEqual(tuple(down.shape), (RANK, DENSE))
        d_b = out.patches[BLOCK_B][0][1][1][0].float()
        for row in range(ROWS):
            curve = 0.6 * (self.lora_b @ (self.lora_a @ self.table[row]))
            dense = up @ (down @ self.grid[row]) + d_b
            self.assertTrue(torch.allclose(curve, dense, atol=1e-4))

    def test_stays_low_rank_rather_than_materialising(self):
        """A multiplied-out dense AdaLN delta is ~1 GB per key; it must stay factored."""
        self.add()
        out, _ = adaln_patch.fix_model(self.model, "port")
        adapter = out.patches[BLOCK_W][0][1]
        stored = sum(w.numel() for w in adapter.weights if torch.is_tensor(w))
        self.assertLess(stored, OUT * DENSE)

    def test_two_stacked_loras_concatenate(self):
        self.add(strength=1.0)
        self.add(strength=0.5)
        out, report = adaln_patch.fix_model(self.model, "port")
        self.assertEqual(report["ported"], 2)
        adapter = out.patches[BLOCK_W][0][1]
        self.assertEqual(adapter.weights[0].shape[1], 2 * RANK)
        d_b = out.patches[BLOCK_B][0][1][1][0].float()
        up, down = adapter.weights[0].float(), adapter.weights[1].float()
        for row in (0, ROWS // 2, ROWS - 1):
            curve = 1.5 * (self.lora_b @ (self.lora_a @ self.table[row]))
            dense = up @ (down @ self.grid[row]) + d_b
            self.assertTrue(torch.allclose(curve, dense, atol=1e-4))

    def test_strip_still_works_on_a_dense_base(self):
        self.add()
        out, report = adaln_patch.fix_model(self.model, "strip")
        self.assertEqual(report["stripped"], 1)
        self.assertNotIn(BLOCK_W, out.patches)


class _DenseBlk(torch.nn.Module if AVAILABLE else object):
    def __init__(self):
        super().__init__()
        self.adaln_proj = TinyAdaln(out=OUT, curve=DENSE)


if __name__ == "__main__":
    if not AVAILABLE:
        print("skipped:", REASON)
    unittest.main()
