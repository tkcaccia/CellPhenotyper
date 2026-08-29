import importlib.util
import math
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "bin" / "gigatime_resolution.py"
SPEC = importlib.util.spec_from_file_location("gigatime_resolution", MODULE_PATH)
gigatime = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gigatime
SPEC.loader.exec_module(gigatime)


class GigaTIMEPhysicalResolutionTest(unittest.TestCase):
    def test_strict_target_mpp_uses_exact_upsampling_factor(self):
        source_mpp = 0.5473021326941083
        target_mpp = 0.25
        factor, metadata = gigatime.choose_downsample_factor(
            orig_h=13303,
            orig_w=11912,
            source_mpp=source_mpp,
            target_mpp=target_mpp,
            auto_threshold_mpix=100.0,
            max_side=4096,
            max_output_gib=1.25,
            num_channels=5,
            bytes_per_sample=1,
            strict_target_mpp=True,
        )

        self.assertTrue(math.isclose(factor, target_mpp / source_mpp, rel_tol=1.0e-12))
        self.assertTrue(math.isclose(metadata["effective_mpp"], target_mpp, rel_tol=1.0e-12))
        self.assertEqual(metadata["selected_shape_yx"], [29124, 26078])
        self.assertGreater(metadata["estimated_prediction_gib"], 1.25)

    def test_strict_target_mpp_does_not_round_noninteger_downsampling(self):
        factor, metadata = gigatime.choose_downsample_factor(
            orig_h=1000,
            orig_w=800,
            source_mpp=0.31,
            target_mpp=0.5,
            auto_threshold_mpix=100.0,
            max_side=4096,
            max_output_gib=8.0,
            num_channels=5,
            bytes_per_sample=1,
            strict_target_mpp=True,
        )

        self.assertTrue(math.isclose(factor, 0.5 / 0.31, rel_tol=1.0e-12))
        self.assertTrue(math.isclose(metadata["effective_mpp"], 0.5, rel_tol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
