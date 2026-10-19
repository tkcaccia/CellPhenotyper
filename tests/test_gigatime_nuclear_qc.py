import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import zarr
from scipy.ndimage import gaussian_filter


BIN = Path(__file__).parents[1] / "bin"
sys.path.insert(0, str(BIN))
SPEC = importlib.util.spec_from_file_location("gigatime_nuclear_qc", BIN / "gigatime_nuclear_qc.py")
qc = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(qc)


class NuclearQCTest(unittest.TestCase):
    def test_diffuse_dapi_fails_despite_high_overlap_auc(self):
        yy, xx = np.mgrid[:256, :256]
        nuclei = np.zeros((256, 256), dtype=np.uint8)
        for cy in range(20, 256, 32):
            for cx in range(20, 256, 32):
                nuclei[(yy - cy) ** 2 + (xx - cx) ** 2 < 7 ** 2] = 1
        dapi = gaussian_filter(nuclei.astype(np.float32), sigma=11)
        dapi = (dapi - dapi.min()) / (dapi.max() - dapi.min())
        scorer = qc.NuclearAlignmentScore(nuclei.shape)
        scorer.add(dapi, nuclei, 0, 0)
        report = scorer.report()
        self.assertGreater(report["auc"], 0.90)
        self.assertGreater(report["prevalence_matched_dice"], 0.70)
        self.assertLess(report["nuclear_boundary_contrast"], 0.05)
        self.assertEqual(report["alignment_status"], "fail")

    def test_dark_image_border_fails_even_when_nuclei_are_detected(self):
        yy, xx = np.mgrid[:256, :256]
        nuclei = np.zeros((256, 256), dtype=np.uint16)
        for cy in range(16, 256, 32):
            for cx in range(16, 256, 32):
                nuclei[(yy - cy) ** 2 + (xx - cx) ** 2 < 7 ** 2] = 1
        edge = (yy < 32) | (xx < 32) | (yy >= 224) | (xx >= 224)
        dapi = np.where(nuclei > 0, np.where(edge, 0.2, 0.9), 0.05).astype(np.float32)
        scorer = qc.NuclearAlignmentScore(nuclei.shape)
        scorer.add(dapi, nuclei, 0, 0)
        report = scorer.report()
        self.assertEqual(report["alignment_status"], "pass")
        self.assertEqual(report["border_shadow_status"], "fail")
        self.assertEqual(report["status"], "fail")

    def test_aligned_and_shifted_nuclei_have_different_scores(self):
        yy, xx = np.mgrid[:128, :128]
        nuclei = np.zeros((128, 128), dtype=np.uint16)
        for cy, cx in ((25, 25), (25, 75), (75, 25), (75, 75)):
            nuclei[(yy - cy) ** 2 + (xx - cx) ** 2 < 9 ** 2] = 1
        dapi = np.where(nuclei > 0, 0.9, 0.05).astype(np.float32)
        aligned = qc.NuclearAlignmentScore(nuclei.shape)
        aligned.add(dapi, nuclei, 0, 0)
        displaced = qc.NuclearAlignmentScore(nuclei.shape)
        displaced.add(dapi, np.roll(nuclei, 20, axis=1), 0, 0)
        a, b = aligned.report(), displaced.report()
        self.assertEqual(a["status"], "pass")
        self.assertGreater(a["auc"], 0.99)
        self.assertLess(b["auc"], 0.6)
        self.assertEqual(b["status"], "fail")

    def test_reads_native_zarr_and_mask_without_jpeg_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / "gigatime"
            store = output / "gigatime_probs.zarr"
            output.mkdir()
            yy, xx = np.mgrid[:96, :96]
            labels = (((yy - 48) ** 2 + (xx - 48) ** 2) < 25 ** 2).astype(np.uint16)
            arr = zarr.open(str(store), mode="w", shape=(1, 96, 96), dtype="uint8")
            arr[0] = np.where(labels > 0, 230, 10).astype(np.uint8)
            (output / "gigatime_channels.json").write_text('["DAPI"]')
            (output / "gigatime_metadata.json").write_text(
                '{"original_shape_yx":[96,96],"inference_shape_yx":[96,96],"effective_mpp":0.25}'
            )
            mask = base / "stardist.tif"
            tifffile.imwrite(mask, labels)
            result = qc.assess(str(output), str(mask), str(base / "qc.json"), block_size=32)
            self.assertEqual(result["status"], "pass")
            self.assertGreater(result["alignment_score"], 0.8)
            self.assertTrue((base / "qc.png").exists())
            self.assertEqual(result["preview"], "qc.png")


if __name__ == "__main__":
    unittest.main()
