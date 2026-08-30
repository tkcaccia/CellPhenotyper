import importlib.util
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
gigatime = None
torch = None
if TORCH_AVAILABLE:
    import torch  # type: ignore

    MODULE_PATH = Path(__file__).parents[1] / "bin" / "run_gigatime_on_crop.py"
    sys.path.insert(0, str(MODULE_PATH.parent))
    SPEC = importlib.util.spec_from_file_location("run_gigatime_on_crop", MODULE_PATH)
    gigatime = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    sys.modules[SPEC.name] = gigatime
    SPEC.loader.exec_module(gigatime)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is supplied by the pipeline runtime")
class GigaTIMERuntimeTest(unittest.TestCase):
    def setUp(self):
        gigatime._RUNTIME_BATCH_CAP = 0
        gigatime._RUNTIME_OOM_REDUCTIONS = 0

    def test_region_inference_covers_every_patch_on_cpu(self):
        class ZeroModel(torch.nn.Module):
            def forward(self, tensor):
                return torch.zeros(
                    (tensor.shape[0], len(gigatime.CHANNEL_NAMES), tensor.shape[2], tensor.shape[3]),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )

        image = np.full((4, 4, 3), 127, dtype=np.uint8)
        accum, counts = gigatime.run_region_inference(
            image_rgb=image,
            positions=[(0, 0), (0, 2), (2, 0), (2, 2)],
            model=ZeroModel(),
            device=torch.device("cpu"),
            patch_size=2,
            batch_size=4,
        )
        self.assertEqual(accum.shape, (len(gigatime.CHANNEL_NAMES), 4, 4))
        self.assertTrue(np.allclose(accum, 0.5, atol=1e-6))
        self.assertTrue(np.all(counts >= 1.0))

    def test_background_mask_cleanup_is_warning_free_and_preserves_thresholds(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[1:6, 1:6] = True
        mask[3, 3] = False
        mask[8, 8] = True

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cleaned = gigatime.clean_background_skip_mask(
                mask,
                close_radius=1,
                hole_area=2,
                min_obj_area=2,
            )

        self.assertFalse(any(issubclass(item.category, FutureWarning) for item in caught))
        self.assertTrue(cleaned[3, 3])
        self.assertFalse(cleaned[8, 8])

    @unittest.skipUnless(TORCH_AVAILABLE and torch.cuda.is_available(), "CUDA runtime required")
    def test_cuda_oom_reduces_batch_and_retries_same_positions(self):
        class BatchLimitedModel(torch.nn.Module):
            def forward(self, tensor):
                if tensor.shape[0] > 1:
                    raise torch.cuda.OutOfMemoryError("synthetic CUDA out of memory")
                return torch.zeros(
                    (tensor.shape[0], len(gigatime.CHANNEL_NAMES), tensor.shape[2], tensor.shape[3]),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )

        image = np.full((4, 4, 3), 127, dtype=np.uint8)
        accum, counts = gigatime.run_region_inference(
            image_rgb=image,
            positions=[(0, 0), (0, 2), (2, 0), (2, 2)],
            model=BatchLimitedModel().cuda(),
            device=torch.device("cuda"),
            patch_size=2,
            batch_size=4,
        )
        self.assertEqual(gigatime._RUNTIME_BATCH_CAP, 1)
        self.assertEqual(gigatime._RUNTIME_OOM_REDUCTIONS, 2)
        self.assertTrue(np.allclose(accum, 0.5, atol=1e-6))
        self.assertTrue(np.all(counts >= 1.0))


if __name__ == "__main__":
    unittest.main()
