import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "bin" / "run_grandqc_artifact_analysis.py"
SPEC = importlib.util.spec_from_file_location("run_grandqc_artifact_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _FakeProperties:
    def __init__(self, gib: float):
        self.total_memory = int(gib * (1024 ** 3))


class _FakeCuda:
    def __init__(self, gib: float):
        self.gib = gib

    def get_device_properties(self, _index: int):
        return _FakeProperties(self.gib)


class _FakeTorch:
    def __init__(self, gib: float):
        self.cuda = _FakeCuda(gib)


class GrandQCTileSelectionTest(unittest.TestCase):
    def test_tissue_threshold_half_matches_binary_argmax(self):
        scores = np.array([
            [[0.8, 0.5, 0.49]],
            [[0.2, 0.5, 0.51]],
        ], dtype=np.float32)
        observed, probability = MODULE.tissue_decision_from_scores(scores, 0.5, np)
        np.testing.assert_array_equal(observed, scores.argmax(axis=0))
        np.testing.assert_array_equal(probability, scores[0])

    def test_lower_tissue_threshold_increases_sensitivity(self):
        scores = np.array([
            [[0.8, 0.4, 0.2]],
            [[0.2, 0.6, 0.8]],
        ], dtype=np.float32)
        strict, _ = MODULE.tissue_decision_from_scores(scores, 0.5, np)
        sensitive, _ = MODULE.tissue_decision_from_scores(scores, 0.3, np)
        self.assertGreater(np.count_nonzero(strict == 0), 0)
        self.assertGreater(np.count_nonzero(sensitive == 0), np.count_nonzero(strict == 0))

    def test_tissue_threshold_and_score_shape_are_validated(self):
        scores = np.ones((2, 2, 2), dtype=np.float32) / 2
        for threshold in (0.0, 1.0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                MODULE.tissue_decision_from_scores(scores, threshold, np)
        with self.assertRaises(ValueError):
            MODULE.tissue_decision_from_scores(np.ones((1, 2, 2)), 0.5, np)

    def test_clean_tissue_policy_removes_only_explicit_artifacts(self):
        classes = np.array([[1, 7, 2, 6], [7, 1, 3, 1]], dtype=np.uint8)
        tissue = np.array([[0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.uint8)
        observed = MODULE.build_clean_tissue_mask(
            classes, tissue, "tissue_minus_artifacts", np
        )
        expected = np.array([[255, 255, 0, 0], [0, 255, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(observed, expected)

    def test_legacy_clean_tissue_policy_remains_available(self):
        classes = np.array([[1, 7], [2, 1]], dtype=np.uint8)
        tissue = np.zeros_like(classes)
        observed = MODULE.build_clean_tissue_mask(
            classes, tissue, "artifact_normal_only", np
        )
        np.testing.assert_array_equal(observed, np.array([[255, 0], [0, 255]], dtype=np.uint8))

    def test_auto_uses_official_geometry_on_supported_gpu(self):
        size, meta = MODULE.resolve_artifact_tile_size(0, "cuda", _FakeTorch(16.0))
        self.assertEqual(size, 512)
        self.assertEqual(meta["mode"], "official_default")

    def test_auto_falls_back_for_small_gpu_or_cpu(self):
        self.assertEqual(MODULE.resolve_artifact_tile_size(0, "cuda", _FakeTorch(4.0))[0], 512)
        self.assertEqual(MODULE.resolve_artifact_tile_size(0, "cpu", _FakeTorch(64.0))[0], 512)

    def test_explicit_tile_size_is_validated(self):
        self.assertEqual(MODULE.resolve_artifact_tile_size(768, "cpu", _FakeTorch(0.0))[0], 768)
        with self.assertRaises(ValueError):
            MODULE.resolve_artifact_tile_size(750, "cuda", _FakeTorch(16.0))

    def test_auto_model_uses_empirically_validated_wsi_checkpoint(self):
        self.assertEqual(MODULE.resolve_artifact_mpp_model("auto", 0.25), 2.0)
        self.assertEqual(MODULE.resolve_artifact_mpp_model("auto", 0.10), 1.0)
        self.assertEqual(MODULE.resolve_artifact_mpp_model("auto", 0.15), 1.5)

    def test_probability_blend_is_identical_for_memmap_and_memory_arrays(self):
        scores = np.zeros((8, 5, 7), dtype=np.float32)
        weights = np.ones((5, 7), dtype=np.float32)
        scores[1, :, :3] = 0.8
        scores[7, :, 3:] = 0.9
        scores[2, 2, 2:5] = 1.2
        confidence = {
            "artifact_probability_threshold": 0.25,
            "artifact_margin_threshold": 0.05,
            "suppressed_pixels": 0,
        }
        expected, expected_meta = MODULE.finalize_probability_blend(
            scores.copy(), weights.copy(), np.zeros((5, 7), dtype=np.uint8),
            confidence.copy(), np, row_block=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score_mm = np.memmap(root / "scores.dat", dtype=np.float32, mode="w+", shape=scores.shape)
            weight_mm = np.memmap(root / "weights.dat", dtype=np.float32, mode="w+", shape=weights.shape)
            score_mm[:] = scores
            weight_mm[:] = weights
            observed, observed_meta = MODULE.finalize_probability_blend(
                score_mm, weight_mm, np.zeros((5, 7), dtype=np.uint8),
                confidence.copy(), np, row_block=3,
            )
        np.testing.assert_array_equal(observed, expected)
        self.assertEqual(observed_meta["suppressed_pixels"], expected_meta["suppressed_pixels"])

    def test_image_size_no_longer_selects_a_different_merge_algorithm(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("25_000_000", source)
        self.assertIn("probability_blend_memmap_windowed_overlap", source)


if __name__ == "__main__":
    unittest.main()
