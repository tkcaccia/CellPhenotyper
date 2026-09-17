import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import tifffile


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
gigatime = None
quantify = None
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

    QUANT_SPEC = importlib.util.spec_from_file_location(
        "quantify_gigatime_intensity", Path(__file__).parents[1] / "bin" / "quantify_gigatime_intensity.py"
    )
    quantify = importlib.util.module_from_spec(QUANT_SPEC)
    assert QUANT_SPEC.loader is not None
    sys.modules[QUANT_SPEC.name] = quantify
    QUANT_SPEC.loader.exec_module(quantify)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is supplied by the pipeline runtime")
class GigaTIMERuntimeTest(unittest.TestCase):
    def setUp(self):
        gigatime._RUNTIME_BATCH_CAP = 0
        gigatime._RUNTIME_OOM_REDUCTIONS = 0

    def test_authoritative_source_mpp_overrides_tiff_and_shift_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = root / "crop.tif"
            shift_path = root / "shift.json"
            tifffile.imwrite(
                image_path,
                np.zeros((8, 9, 3), dtype=np.uint8),
                photometric="rgb",
                resolution=(10, 10),
                resolutionunit="CENTIMETER",
            )
            shift_path.write_text(json.dumps({"source_mpp": 900.0}))

            observed = gigatime.infer_source_mpp(
                str(image_path),
                shift_json=str(shift_path),
                source_mpp_override=0.273774374855905,
            )

            self.assertAlmostEqual(observed, 0.273774374855905)

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

    def test_background_skip_requires_empty_halo(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True

        skip_without_halo, _ = gigatime.background_block_skip_decision(
            mask, 100, 100, 0, 20, 0, 20, halo_px=0, max_tissue_fraction=0.0
        )
        skip_with_halo, fraction = gigatime.background_block_skip_decision(
            mask, 100, 100, 0, 20, 0, 20, halo_px=40, max_tissue_fraction=0.0
        )

        self.assertTrue(skip_without_halo)
        self.assertFalse(skip_with_halo)
        self.assertGreater(fraction, 0.0)

    def test_marker_score_qc_records_semantics_distribution_and_tissue_contrast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            tissue = np.zeros((8, 8), dtype=np.uint8)
            tissue[:, :4] = 1
            tissue_path = outdir / "tissue.tif"
            tifffile.imwrite(tissue_path, tissue)

            qc = gigatime.MarkerScoreQC(
                outdir=outdir,
                target_shape=(8, 8),
                channel_names=["marker_a", "marker_b"],
                tissue_mask_path=str(tissue_path),
                max_samples=64,
            )
            scores = np.zeros((2, 8, 8), dtype=np.float32)
            scores[0, :, :4] = 0.8
            scores[0, :, 4:] = 0.2
            scores[1] = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
            qc.accumulate_tile(0, 8, 0, 8, scores)
            result = qc.write_outputs()

            report = json.loads((outdir / "gigatime_marker_score_qc.json").read_text())
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                report["value_semantics"]["value_type"],
                "uncalibrated_virtual_marker_score",
            )
            self.assertFalse(report["value_semantics"]["calibrated_probability"])
            self.assertAlmostEqual(
                report["channels"][0]["tissue_background_difference"], 0.6, places=5
            )
            self.assertEqual(len(report["channel_correlation"]["matrix"]), 2)
            self.assertTrue((outdir / "gigatime_marker_score_qc.tsv").exists())
            self.assertTrue((outdir / "gigatime_marker_score_qc.png").exists())

    def test_uint8_store_is_rescaled_to_virtual_score_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            store_path = outdir / "gigatime_probs.zarr"
            root = quantify.zarr.open(str(store_path), mode="w", shape=(1, 2, 2), dtype="uint8")
            root[:] = np.asarray([[[0, 255], [128, 64]]], dtype=np.uint8)
            root.attrs["axes"] = "CYX"
            (outdir / "gigatime_metadata.json").write_text(
                json.dumps({"output_dtype": "uint8", "storage_scale_max": 255.0})
            )
            with quantify.LazyImageReader(str(store_path)) as reader:
                block = reader.read_block(0, 2, 0, 2)
            self.assertAlmostEqual(float(block.max()), 1.0, places=6)

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
