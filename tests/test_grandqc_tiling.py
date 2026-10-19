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
