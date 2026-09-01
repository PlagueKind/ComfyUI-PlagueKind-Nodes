import importlib.util
import os
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = REPO_ROOT / "ComfyUI-H3-UltimateUpscale-LD" / "nodes.py"
default_comfy_root = Path(__file__).resolve().parents[3]
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", default_comfy_root))
if not (COMFY_ROOT / "comfy").is_dir() or not (COMFY_ROOT / "folder_paths.py").is_file():
    raise RuntimeError(
        "Set COMFYUI_ROOT to a ComfyUI checkout before running this test "
        f"(resolved {COMFY_ROOT!s})."
    )

sys.path.insert(0, str(COMFY_ROOT))
spec = importlib.util.spec_from_file_location("plaguekind_upscale_pr_nodes", NODES_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class ExtendContextTrimTests(unittest.TestCase):
    def test_context_is_kept_only_for_first_temporal_chunk(self):
        latent = torch.randn(1, 16, 2, 4, 4)
        keyframe = {"kind": "context", "num_frames": 2, "latent": latent}

        first = module.trim_keyframe(keyframe, 0, 85)
        self.assertIsNot(first, keyframe)
        self.assertEqual(first["kind"], "context")
        self.assertEqual(first["num_frames"], 2)
        self.assertIs(first["latent"], latent)
        self.assertNotIn("resolved_frame_index", first)
        self.assertIsNone(module.trim_keyframe(keyframe, 68, 153))

    def test_context_audio_is_kept_only_for_first_temporal_chunk(self):
        audio = torch.randn(1, 32, 2, 20)
        keyframe = {"kind": "context_audio", "num_frames": 20, "audio_latent": audio}

        first = module.trim_keyframe(keyframe, 0, 85)
        self.assertEqual(first["kind"], "context_audio")
        self.assertIs(first["audio_latent"], audio)
        self.assertIsNone(module.trim_keyframe(keyframe, 68, 153))

    def test_context_payload_requires_matching_declared_length(self):
        with self.assertRaisesRegex(ValueError, "requires both 'latent' and 'num_frames'"):
            module.trim_keyframe({"kind": "context", "num_frames": 2}, 0, 85)

        mismatched = {
            "kind": "context_audio",
            "num_frames": 19,
            "audio_latent": torch.zeros(1, 32, 2, 20),
        }
        with self.assertRaisesRegex(ValueError, "contains 20 temporal frames"):
            module.trim_keyframe(mismatched, 0, 85)

        with self.assertRaisesRegex(ValueError, "must be a 5D tensor"):
            module.trim_keyframe({
                "kind": "context",
                "num_frames": 2,
                "latent": torch.zeros(2, 2),
            }, 0, 85)

        with self.assertRaisesRegex(ValueError, "contains no temporal frames"):
            module.trim_keyframe({
                "kind": "context_audio",
                "num_frames": 0,
                "audio_latent": torch.zeros(1, 32, 2, 0),
            }, 0, 85)

    def test_regular_absolute_keyframe_behavior_is_unchanged(self):
        latent = torch.randn(1, 16, 1, 4, 4)
        keyframe = {"resolved_frame_index": 0, "latent": latent}
        trimmed = module.trim_keyframe(keyframe, 0, 17)

        self.assertEqual(trimmed["resolved_frame_index"], 0)
        self.assertTrue(torch.equal(trimmed["latent"], latent))

    def test_unknown_unpositioned_keyframe_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "missing 'resolved_frame_index'"):
            module.trim_keyframe({"kind": "mystery", "latent": torch.zeros(1, 16, 1, 2, 2)}, 0, 17)

    def test_last_frame_anchor_and_frame_count_are_rebased_together(self):
        last = {"resolved_frame_index": 169, "latent": torch.zeros(1, 16, 1, 2, 2)}
        cond = [[torch.zeros(1), {
            "minimax_keyframes": [last],
            "minimax_frame_count": 170,
        }]]

        rebased = module.reanchor_conditioning(cond, 85, 170)
        metadata = rebased[0][1]
        self.assertEqual(metadata["minimax_frame_count"], 85)
        self.assertEqual(metadata["minimax_keyframes"][0]["resolved_frame_index"], 84)

    def test_noncontiguous_context_latent_can_be_spatially_resized(self):
        source = torch.randn(1, 16, 6, 4, 6)
        noncontiguous = source[:, :, 3:, :, :]
        self.assertFalse(noncontiguous.is_contiguous())
        context = {"kind": "context", "num_frames": 3, "latent": noncontiguous}
        cond = [[torch.zeros(1), {
            "minimax_keyframes": [context],
            "minimax_frame_count": 85,
        }]]

        rebased = module.reanchor_conditioning(cond, 0, 85, spatial=(8, 10))
        latent = rebased[0][1]["minimax_keyframes"][0]["latent"]
        self.assertEqual(tuple(latent.shape), (1, 16, 3, 8, 10))

    def test_reanchor_conditioning_drops_original_context_after_first_chunk(self):
        context = {"kind": "context", "num_frames": 1, "latent": torch.zeros(1, 16, 1, 2, 2)}
        anchor = {"resolved_frame_index": 68, "latent": torch.zeros(1, 16, 1, 2, 2)}
        cond = [[torch.zeros(1), {"minimax_keyframes": [context, anchor]}]]

        first = module.reanchor_conditioning(cond, 0, 85)
        self.assertEqual([kf.get("kind") for kf in first[0][1]["minimax_keyframes"]], ["context", None])

        later = module.reanchor_conditioning(cond, 68, 153)
        later_keyframes = later[0][1]["minimax_keyframes"]
        self.assertEqual(len(later_keyframes), 1)
        self.assertNotIn("kind", later_keyframes[0])
        self.assertEqual(later_keyframes[0]["resolved_frame_index"], 0)


if __name__ == "__main__":
    unittest.main()
