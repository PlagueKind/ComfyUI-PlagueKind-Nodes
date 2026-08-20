"""Block-sparse attention: kernel correctness and the ComfyUI override contract.

The contract half matters more than it looks. The failure mode this node exists
to avoid is not a crash -- it is a patch that installs cleanly, logs success and
never runs, because it hooked an API the model does not consult. So the central
test drives the override exactly the way ``wrap_attn`` does, with H3's real
argument shape, and asserts the sparse path actually fired.

The CUDA tests are skipped automatically without a GPU; the contract tests that
do not need a kernel run anywhere torch is importable.
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
    from h3u.sla import patch as sla_patch
    AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    AVAILABLE = False
    REASON = str(exc)

try:
    import triton  # noqa: F401
    CUDA = AVAILABLE and torch.cuda.is_available()
except Exception:  # noqa: BLE001
    CUDA = False

H, D = 56, 128          # MiniMax-H3: 56 heads, head_dim 128


def _sdpa(q, k, v):
    """Reference, in H3's [B, H, S, D] -> [B, S, H*D] convention."""
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    b, _, s, _ = q.shape
    return o.transpose(1, 2).reshape(b, s, -1)


@unittest.skipUnless(AVAILABLE, "torch or the package is unavailable")
class OverrideContract(unittest.TestCase):
    """Drive the override the way comfy/ldm/modules/attention.py:193 does."""

    def _call(self, override, q, k, v, **kw):
        # wrap_attn hands the override the *undecorated* backend as arg 0, then
        # q/k/v/heads, with H3's kwargs. mask is always None, skip_reshape True,
        # and skip_output_reshape is not passed at all.
        opts = dict(mask=None, skip_reshape=True,
                    transformer_options={}, _inside_attn_wrapper=True)
        opts.update(kw)
        return override(_backend, q, k, v, H, **opts)

    def test_short_sequence_falls_through_to_dense(self):
        """The text refiner must never be sparsified."""
        state = sla_patch._new_state()
        ov = sla_patch._make_override(state, 0.85, 128, 64, min_seq_len=8192)
        q = torch.randn(1, H, 512, D)
        out = self._call(ov, q, q.clone(), q.clone())
        self.assertEqual(state["calls"], 0)
        self.assertEqual(state["dense"], 1)
        self.assertEqual(out.shape, (1, 512, H * D))

    def test_records_the_backend_it_displaced(self):
        state = sla_patch._new_state()
        ov = sla_patch._make_override(state, 0.85, 128, 64, min_seq_len=8192)
        self._call(ov, *(torch.randn(1, H, 256, D),) * 3)
        self.assertEqual(state["backend"], "_backend")

    def test_masked_attention_falls_through(self):
        state = sla_patch._new_state()
        ov = sla_patch._make_override(state, 0.85, 128, 64, min_seq_len=0)
        q = torch.randn(1, H, 256, D)
        self._call(ov, q, q.clone(), q.clone(), mask=torch.zeros(1, 1))
        self.assertEqual(state["calls"], 0)

    def test_float32_falls_through_before_the_kernel(self):
        """fp32 never reaches triton -- the dtype guard catches it first."""
        state = sla_patch._new_state()
        ov = sla_patch._make_override(state, 0.85, 128, 64, min_seq_len=0)
        q = torch.randn(1, H, 256, D)
        out = self._call(ov, q, q.clone(), q.clone())
        self.assertEqual(state["calls"], 0)
        self.assertEqual(state["dense"], 1)
        self.assertIsNone(state["failed"], "should not have reached the kernel")
        self.assertEqual(out.shape, (1, 256, H * D))

    def test_kernel_failure_falls_back_instead_of_raising(self):
        """A broken kernel must cost speed, never the run.

        bf16 on CPU passes every guard and then cannot launch a CUDA kernel,
        which is the closest stand-in for a GPU/driver/triton mismatch.
        """
        state = sla_patch._new_state()
        ov = sla_patch._make_override(state, 0.85, 128, 64, min_seq_len=0)
        q = torch.randn(1, H, 256, D, dtype=torch.bfloat16)
        out = self._call(ov, q, q.clone(), q.clone())
        self.assertEqual(out.shape, (1, 256, H * D))
        self.assertIsNotNone(state["failed"], "the failure was not recorded")
        self.assertEqual(state["calls"], 0)
        self.assertEqual(state["dense"], 1)


@unittest.skipUnless(AVAILABLE, "torch or the package is unavailable")
class StepWrapper(unittest.TestCase):

    def setUp(self):
        # These runs make no attention calls, so the end-of-run summary fires
        # its "never invoked" warning every time. That warning is wanted in
        # production and only noise here.
        import logging
        self._log = logging.getLogger("H3Utils")
        self._lvl = self._log.level
        self._log.setLevel(logging.CRITICAL)

    def tearDown(self):
        self._log.setLevel(self._lvl)

    def _run(self, wrapper, state, n_steps):
        """One sampling run: n_steps forwards through the wrapper."""
        seen = []

        class Ex:
            @staticmethod
            def original(*a, **kw):
                seen.append(kw["transformer_options"]["_h3sla_dense"])
                return None

        to = {"sample_sigmas": [0.0] * (n_steps + 1)}
        for _ in range(n_steps):
            wrapper(Ex, None, None, None, transformer_options=to)
        return seen

    def test_counter_resets_between_runs(self):
        """ComfyUI caches node outputs, so this closure outlives one run.

        Without the reset the step counter keeps climbing and every later run
        sits permanently inside the trailing-dense window -- sparsity silently
        stops happening from run two onwards.
        """
        state = sla_patch._new_state()
        w = sla_patch._make_wrapper(state, 0.85, 128, 64, dense_last_steps=1)
        first = self._run(w, state, 4)
        second = self._run(w, state, 4)
        self.assertEqual(first, [False, False, False, True])
        self.assertEqual(second, first)

    def test_dense_last_steps_zero_never_forces_dense(self):
        state = sla_patch._new_state()
        w = sla_patch._make_wrapper(state, 0.85, 128, 64, dense_last_steps=0)
        self.assertEqual(self._run(w, state, 4), [False] * 4)

    def test_non_h3_models_are_not_passed_a_minimax_kwarg(self):
        """Nothing stops a user wiring this node to a WAN or Flux model.

        Every other diffusion model would raise TypeError on an unexpected
        minimax_payload kwarg -- a crash mid-sampling instead of the graceful
        no-op they should get.
        """
        seen = {}

        class Ex:
            @staticmethod
            def original(x, timestep, context, transformer_options=None, **kw):
                seen.update(kw)
                return None

        state = sla_patch._new_state()
        w = sla_patch._make_wrapper(state, 0.85, 64, 64, dense_last_steps=0)
        w(Ex, None, None, None, transformer_options={"sample_sigmas": [0.0] * 2})
        self.assertNotIn("minimax_payload", seen)

    def test_h3_still_receives_its_payload(self):
        seen = {}

        class Ex:
            @staticmethod
            def original(x, timestep, context, transformer_options=None, **kw):
                seen.update(kw)
                return None

        state = sla_patch._new_state()
        w = sla_patch._make_wrapper(state, 0.85, 64, 64, dense_last_steps=0)
        w(Ex, None, None, None, transformer_options={"sample_sigmas": [0.0] * 2},
          minimax_payload={"layout": None})
        self.assertIn("minimax_payload", seen)

    def test_prefix_is_read_from_the_packed_layout(self):
        """The video segment start is the length of what must stay exact.

        It lives on minimax_payload, which never reaches the attention call
        site, so the wrapper is the only place it can be picked up.
        """
        class Layout:
            segments = [(0, 512, "text"), (512, 800, "cond"),
                        (800, 2000, "audio"), (2000, 114785, "video")]

        state = sla_patch._new_state()
        w = sla_patch._make_wrapper(state, 0.85, 128, 64, dense_last_steps=0)
        to = {"sample_sigmas": [0.0] * 5}

        class Ex:
            @staticmethod
            def original(*a, **kw):
                return None

        w(Ex, None, None, None, transformer_options=to,
          minimax_payload={"layout": Layout()})
        self.assertEqual(to["_h3sla_prefix"], 2000)

    def test_missing_layout_disables_protection_rather_than_guessing(self):
        state = sla_patch._new_state()
        w = sla_patch._make_wrapper(state, 0.85, 128, 64, dense_last_steps=0)
        to = {"sample_sigmas": [0.0] * 5}

        class Ex:
            @staticmethod
            def original(*a, **kw):
                return None

        w(Ex, None, None, None, transformer_options=to, minimax_payload=None)
        self.assertEqual(to["_h3sla_prefix"], 0)


@unittest.skipUnless(CUDA, "needs CUDA and triton")
class Kernel(unittest.TestCase):

    def test_zero_sparsity_matches_dense_attention(self):
        """With every block kept, the sparse kernel is just attention."""
        from h3u.sla.block_map import get_block_map
        from h3u.sla.kernel import block_sparse_attention

        S = 4096
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3))
        lut, topk = get_block_map(q, k, 1.0, 128, 64)
        got = block_sparse_attention(q, k, v, lut, topk, 128, 64)
        ref = _sdpa(*(t.transpose(1, 2) for t in (q, k, v)))

        self.assertFalse(torch.isnan(got).any())
        rel = ((got.reshape(1, S, -1).float() - ref.float()).abs().max()
               / ref.float().abs().max()).item()
        self.assertLess(rel, 1e-2, "sparse kernel diverges from dense attention")

    def test_override_returns_h3s_expected_shape_and_fires(self):
        state = sla_patch._new_state()
        ov = sla_patch._make_override(state, 0.85, 128, 64, min_seq_len=1024)
        S = 8192
        q, k, v = (torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3))
        out = ov(_backend, q, k, v, H, mask=None, skip_reshape=True,
                 transformer_options={})
        self.assertEqual(state["calls"], 1, "the sparse path did not fire")
        self.assertEqual(state["dense"], 0)
        self.assertEqual(out.shape, (1, S, H * D))
        self.assertFalse(torch.isnan(out).any())

    def test_protected_prefix_is_always_selected(self):
        """Every query block must keep the whole prefix, at any sparsity.

        This is the audio fix. Audio is ~1% of the packed sequence, so plain
        top-k routinely drops all of it -- the soundtrack degrades while the
        video still looks fine.
        """
        from h3u.sla.block_map import get_block_map

        S, prefix = 16384, 2048          # prefix spans 32 key blocks at BLKK=64
        torch.manual_seed(0)
        q = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16)

        plain_lut, plain_topk = get_block_map(q, k, 0.15, 128, 64)
        lut, topk = get_block_map(q, k, 0.15, 128, 64, protect_upto=prefix)

        n_pinned = prefix // 64
        # Widened, so the pinned blocks do not evict what top-k chose.
        self.assertEqual(topk, plain_topk + n_pinned)

        # Every query block contains every prefix block.
        got = lut.long().sort(dim=-1).values[..., :n_pinned]
        want = torch.arange(n_pinned, device=lut.device)
        self.assertTrue(torch.equal(got, want.expand_as(got)))

    def test_pinning_selects_the_same_blocks_at_zero_sparsity(self):
        """Pinning must decide *which* blocks, never alter their values.

        The +inf written into the pooled scores is a selection device: it feeds
        torch.topk and nothing else, and the kernel recomputes real dot products
        from q and k. With every block kept, pinning can therefore only reorder
        the lookup table, not change what is attended.
        """
        from h3u.sla.block_map import get_block_map
        from h3u.sla.kernel import block_sparse_attention

        S = 4096
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3))
        l0, t0 = get_block_map(q, k, 1.0, 64, 64, protect_upto=0)
        l1, t1 = get_block_map(q, k, 1.0, 64, 64, protect_upto=1024)

        self.assertEqual(t0, t1)
        self.assertTrue(torch.equal(l0.long().sort(-1).values,
                                    l1.long().sort(-1).values),
                        "pinning changed which blocks are attended")

        o0 = block_sparse_attention(q, k, v, l0, t0, 64, 64)
        o1 = block_sparse_attention(q, k, v, l1, t1, 64, 64)
        # Same set, different visitation order -> online softmax accumulates in
        # that order, so expect a sub-ULP difference, not bitwise equality.
        diff = (o0.float() - o1.float()).abs().max().item()
        self.assertLess(diff, torch.finfo(torch.bfloat16).eps)

    def test_no_inf_or_nan_leaks_into_the_output(self):
        from h3u.sla.block_map import get_block_map
        from h3u.sla.kernel import block_sparse_attention

        S = 4096
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3))
        lut, topk = get_block_map(q, k, 0.10, 64, 64, protect_upto=1024)
        out = block_sparse_attention(q, k, v, lut, topk, 64, 64)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
        self.assertLess(lut.max().item(), (S + 63) // 64)

    def test_unprotected_selection_can_miss_the_prefix(self):
        """Shows the failure the pin exists to prevent."""
        from h3u.sla.block_map import get_block_map

        S, prefix = 16384, 2048
        torch.manual_seed(0)
        q = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16)
        lut, _ = get_block_map(q, k, 0.15, 128, 64)
        n_pinned = prefix // 64
        covered = (lut.long() < n_pinned).sum(-1)     # prefix blocks per query
        self.assertLess(covered.float().mean().item(), n_pinned,
                        "plain top-k unexpectedly kept the whole prefix")

    def test_sparse_output_is_close_to_dense(self):
        """Sparsity is an approximation, but not an unrecognisable one."""
        state = sla_patch._new_state()
        ov = sla_patch._make_override(state, 0.85, 128, 64, min_seq_len=1024)
        S = 8192
        torch.manual_seed(0)
        # Correlated q/k, so the top blocks carry most of the mass -- random
        # normals have no structure for block selection to find.
        base = torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16)
        q = base + 0.1 * torch.randn_like(base)
        k = base + 0.1 * torch.randn_like(base)
        v = torch.randn_like(base)

        got = ov(_backend, q, k, v, H, mask=None, skip_reshape=True,
                 transformer_options={})
        ref = _sdpa(q, k, v)
        cos = torch.nn.functional.cosine_similarity(
            got.float().flatten(), ref.float().flatten(), dim=0).item()
        self.assertGreater(cos, 0.9, "sparse output barely resembles dense")


def _backend(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
             skip_output_reshape=False, **kwargs):
    """Stand-in for ComfyUI's undecorated attention backend."""
    if not skip_reshape:
        b, s, _ = q.shape
        q, k, v = (t.view(b, s, heads, -1).transpose(1, 2) for t in (q, k, v))
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    if skip_output_reshape:
        return o
    b, _, s, _ = q.shape
    return o.transpose(1, 2).reshape(b, s, -1)


if __name__ == "__main__":
    unittest.main()
