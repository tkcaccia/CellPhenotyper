import importlib.util
import inspect
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

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

    @staticmethod
    def _area_api_variants(operation, legacy_name):
        # Exercise both public API signatures against the installed, real
        # connected-component algorithm. This does not install another skimage.
        modern_backend = "max_size" in inspect.signature(operation).parameters

        def inclusive(ar, *, max_size):
            kwargs = {"max_size": max_size} if modern_backend else {legacy_name: max_size + 1}
            return operation(ar, **kwargs)

        def legacy_holes(ar, area_threshold):
            return inclusive(ar, max_size=area_threshold - 1)

        def legacy_objects(ar, min_size):
            return inclusive(ar, max_size=min_size - 1)

        return {"modern_inclusive": inclusive,
                "legacy_exclusive": legacy_holes if legacy_name == "area_threshold" else legacy_objects}

    def test_hole_area_preserves_components_below_at_and_above_threshold(self):
        mask = np.ones((9, 20), dtype=bool)
        mask[3, 2:5] = False   # area 3: fill
        mask[3, 7:11] = False  # area 4: preserve
        mask[3, 13:18] = False # area 5: preserve
        expected = mask.copy()
        expected[3, 2:5] = True
        original = mask.copy()
        for api, operation in self._area_api_variants(gigatime.remove_small_holes, "area_threshold").items():
            with self.subTest(api=api), patch.object(gigatime, "remove_small_holes", operation):
                result = gigatime.clean_background_skip_mask(mask, close_radius=0, hole_area=4, min_obj_area=0)
                np.testing.assert_array_equal(result, expected)
                np.testing.assert_array_equal(mask, original)

    def test_object_area_preserves_components_below_at_and_above_threshold(self):
        mask = np.zeros((9, 20), dtype=bool)
        mask[3, 2:5] = True   # area 3: remove
        mask[3, 7:11] = True  # area 4: preserve
        mask[3, 13:18] = True # area 5: preserve
        expected = mask.copy()
        expected[3, 2:5] = False
        original = mask.copy()
        for api, operation in self._area_api_variants(gigatime.remove_small_objects, "min_size").items():
            with self.subTest(api=api), patch.object(gigatime, "remove_small_objects", operation):
                result = gigatime.clean_background_skip_mask(mask, close_radius=0, hole_area=0, min_obj_area=4)
                np.testing.assert_array_equal(result, expected)
                np.testing.assert_array_equal(mask, original)

    def test_disabled_or_unit_area_thresholds_preserve_background_masks(self):
        mixed = np.zeros((9, 9), dtype=bool)
        mixed[1:5, 1:5] = True
        mixed[2, 2] = False
        mixed[7, 7] = True
        for mask in (mixed, np.zeros_like(mixed), np.ones_like(mixed)):
            for threshold in (0, 1):
                with self.subTest(threshold=threshold, foreground=int(mask.sum())):
                    result = gigatime.clean_background_skip_mask(mask, close_radius=0,
                        hole_area=threshold, min_obj_area=threshold)
                    np.testing.assert_array_equal(result, mask)
                    self.assertEqual(result.dtype, np.dtype(bool))

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
