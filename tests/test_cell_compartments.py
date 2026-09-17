import csv
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile

SPEC = importlib.util.spec_from_file_location("expand_labels_to_cytoplasm", Path(__file__).parents[1] / "bin" / "expand_labels_to_cytoplasm.py")
compartments = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compartments)


class ArrayReader:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def read(self, y0, y1, x0, x1):
        return self.array[y0:y1, x0:x1]


class ArraySupport:
    def __init__(self, array):
        self.array = array

    def read(self, x0, y0, x1, y1):
        return self.array[y0:y1, x0:x1]


class PhysicalExpansionTest(unittest.TestCase):
    def expand(self, labels, tissue, radius, mx=1, my=1):
        return compartments.expand_labels_physical(labels, tissue, radius, mx, my, accelerate=False)

    def test_physical_radius_and_anisotropic_spacing(self):
        labels = np.zeros((11, 13), dtype=np.uint32)
        labels[5, 6] = 1000
        result = self.expand(labels, np.ones(labels.shape, dtype=bool), 2, mx=1, my=0.5)
        self.assertEqual(result[5, 8], 1000)
        self.assertEqual(result[9, 6], 1000)
        self.assertEqual(result[5, 9], 0)
        self.assertEqual(result[10, 6], 0)
        self.assertEqual(result[6, 7], 1000)
        self.assertEqual(result[7, 8], 0)

    def test_gap_cannot_be_crossed_even_if_same_component_connects_far_away(self):
        labels = np.zeros((15, 15), dtype=np.uint32)
        labels[7, 5] = 9
        support = np.ones(labels.shape, dtype=bool)
        support[2:13, 7] = False
        result = self.expand(labels, support, 6)
        self.assertTrue(np.all(result[:, 8:] == 0))
        self.assertTrue(np.all(result[~support] == 0))
        self.assertEqual(result[7, 6], 9)

    def test_diagonal_corner_cutting_is_forbidden(self):
        labels = np.zeros((4, 4), dtype=np.uint32)
        labels[1, 1] = 5
        support = np.zeros(labels.shape, dtype=bool)
        support[1, 1] = True
        support[2, 2] = True
        result = self.expand(labels, support, 10)
        self.assertEqual(result[2, 2], 0)

    def test_collisions_tie_break_by_original_id_and_preserve_nuclei(self):
        labels = np.zeros((9, 13), dtype=np.uint32)
        labels[4, 3] = 700
        labels[4, 9] = 10
        result = self.expand(labels, np.ones(labels.shape, dtype=bool), 5)
        self.assertEqual(result[4, 6], 10)
        np.testing.assert_array_equal(result[labels > 0], labels[labels > 0])
        self.assertEqual(set(np.unique(result)), {0, 10, 700})

    def test_outside_support_nucleus_is_retained_but_does_not_seed(self):
        labels = np.zeros((7, 7), dtype=np.uint32)
        labels[3, 3] = 1
        support = np.ones(labels.shape, dtype=bool)
        support[3, 3] = False
        result = self.expand(labels, support, 3)
        np.testing.assert_array_equal(result, labels)

    def test_tiled_result_and_flags_match_single_tile(self):
        labels = np.zeros((31, 45), dtype=np.uint32)
        labels[2:4, 3:5] = 100
        labels[13, 14] = 7000
        labels[14, 20] = 33
        labels[28, 43] = 8
        support = np.ones(labels.shape, dtype=bool)
        support[6:26, 17] = False
        support[2, 3] = False
        full = self.expand(labels, support, 4, 0.6, 0.9)
        all_stats = []
        for tile in (7, 12, 64):
            out = np.zeros_like(labels)
            stats = compartments.expand_physical_tiled(ArrayReader(labels), ArraySupport(support), 4, 0.6, 0.9, tile, out, accelerate=False)
            np.testing.assert_array_equal(out, full)
            all_stats.append(stats)
        self.assertEqual(all_stats[0], all_stats[1])
        self.assertEqual(all_stats[0], all_stats[2])
        self.assertEqual(all_stats[0][100]["nucleus_outside_support_pixels"], 1)
        self.assertEqual(all_stats[0][8]["image_truncation_flag"], 1)
        self.assertEqual(all_stats[0][7000]["tissue_truncation_flag"], 1)

    def test_optional_accelerator_matches_python(self):
        try:
            import numba  # noqa: F401
        except ImportError:
            self.skipTest("Numba optional")
        labels = np.zeros((17, 21), dtype=np.uint32)
        labels[7, 4] = 900
        labels[7, 13] = 2
        tissue = np.ones(labels.shape, dtype=bool)
        tissue[6:9, 8] = False
        expected = self.expand(labels, tissue, 3, 0.3, 0.5)
        actual = compartments.expand_labels_physical(labels, tissue, 3, 0.3, 0.5)
        np.testing.assert_array_equal(actual, expected)

    def test_zero_radius_empty_image_and_invalid_labels(self):
        labels = np.zeros((3, 3), dtype=np.int32)
        labels[1, 1] = 3
        tissue = np.ones(labels.shape, dtype=bool)
        np.testing.assert_array_equal(self.expand(labels, tissue, 0), labels)
        np.testing.assert_array_equal(self.expand(labels*0, tissue, 3), labels*0)
        labels[1, 1] = -1
        with self.assertRaisesRegex(ValueError, "Negative"):
            self.expand(labels, tissue, 2)

    def test_legacy_euclidean_api_is_unchanged(self):
        labels = np.zeros((15, 15), dtype=np.uint32)
        labels[7, 7] = 1
        result = compartments.expand_labels_nonoverlap(labels, 5)
        self.assertEqual(result[10, 11], 1)
        self.assertEqual(result[11, 11], 0)


class CompartmentExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.labels = np.zeros((40, 48), dtype=np.uint32)
        self.labels[8:10, 10:12] = 100
        self.labels[8:10, 17:19] = 4
        self.labels[38:40, 46:48] = 600
        self.tissue = np.ones((40, 48), dtype=np.uint8)
        self.tissue[:, 30] = 0
        tifffile.imwrite(self.root / "labels.tif", self.labels, tile=(16, 16), compression="deflate")
        tifffile.imwrite(self.root / "tissue.tif", self.tissue, tile=(16, 16), compression="deflate")
        (self.root / "shift.json").write_text(json.dumps({"crop_size": {"width": 48, "height": 40}, "offset_crop_to_original": {"dx": 0, "dy": 0}}))
        (self.root / "resolution.json").write_text(json.dumps({"status": "pass", "mpp_x": 0.5, "mpp_y": 0.5, "width_px": 48, "height_px": 40}))

    def tearDown(self):
        self.temp.cleanup()

    def test_three_compartments_partition_and_qc(self):
        with patch.object(tifffile.TiffPage, "asarray", side_effect=AssertionError("No whole-image decoding")):
            summary = compartments.write_compartments(self.root / "labels.tif", self.root / "whole.tif", 2,
                self.root / "resolution.json", self.root / "shift.json", self.root / "tissue.tif",
                compartments_dir=self.root / "compartments", tile_size=13, accelerate=False)
        whole = tifffile.imread(self.root / "whole.tif")
        nuclei = tifffile.imread(self.root / "compartments" / "labels_nucleus.tif")
        ring = tifffile.imread(self.root / "compartments" / "labels_perinuclear_ring.tif")
        np.testing.assert_array_equal(nuclei, self.labels)
        self.assertFalse(np.any((nuclei > 0) & (ring > 0)))
        np.testing.assert_array_equal(nuclei+ring, whole)
        self.assertFalse(np.any(ring[self.tissue == 0]))
        with (self.root / "compartments" / "compartment_qc.csv").open() as handle:
            rows = {int(row["label"]): row for row in csv.DictReader(handle)}
        self.assertEqual(set(rows), {4, 100, 600})
        self.assertEqual(rows[4]["crowding_flag"], "1")
        self.assertEqual(rows[100]["possible_neighbour_contamination_flag"], "1")
        self.assertEqual(rows[600]["image_truncation_flag"], "1")
        for label, row in rows.items():
            self.assertEqual(int(row["nucleus_pixels"]), int((nuclei == label).sum()))
            self.assertEqual(int(row["ring_pixels"]), int((ring == label).sum()))
        self.assertFalse(summary["io"]["whole_image_decode"])
        self.assertEqual(summary["provenance_schema_version"], "cellphenotyper.compartments.v1")
        self.assertEqual(set(summary["inputs"]), {"labels", "tissue_mask", "shift", "resolution"})
        for collection in (summary["inputs"], summary["output_artifacts"]):
            for record in collection.values():
                self.assertEqual(record["sha256"], hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest())
        bundle = self.root / "compartments" / "labels_whole_cell.tif"
        self.assertEqual(summary["outputs"]["whole_cell_bundle"], str(bundle))
        self.assertEqual(bundle.stat().st_ino, (self.root / "whole.tif").stat().st_ino)
        self.assertEqual(summary["output_artifacts"]["whole_cell"]["sha256"], summary["output_artifacts"]["whole_cell_bundle"]["sha256"])

    def test_original_label_frame_ignores_crop_offset_and_records_lineage(self):
        (self.root / "shift.json").write_text(json.dumps({"crop_size": {"width": 10, "height": 12},
            "offset_crop_to_original": {"dx": 17, "dy": 18}}))
        summary = compartments.write_compartments(self.root / "labels.tif", self.root / "whole.tif", 2,
            self.root / "resolution.json", self.root / "shift.json", self.root / "tissue.tif",
            compartments_dir=self.root / "compartments", label_frame="original", tissue_frame="original",
            tile_size=13, accelerate=False)
        self.assertEqual(summary["label_frame"], "original")
        self.assertEqual(summary["crop_offset_xy"], [0, 0])
        nuclei = tifffile.imread(self.root / "compartments" / "labels_nucleus.tif")
        np.testing.assert_array_equal(nuclei, self.labels)
        self.assertEqual(summary["inputs"]["shift"]["sha256"], hashlib.sha256((self.root / "shift.json").read_bytes()).hexdigest())

    def test_bundle_fallback_copy_and_existing_target_preserved(self):
        with patch.object(compartments.os, "link", side_effect=OSError("cross-device")):
            summary = compartments.write_compartments(self.root / "labels.tif", self.root / "whole.tif", 2,
                self.root / "resolution.json", self.root / "shift.json", self.root / "tissue.tif",
                compartments_dir=self.root / "compartments", tile_size=13, accelerate=False)
        bundle = Path(summary["outputs"]["whole_cell_bundle"])
        before = bundle.read_bytes()
        self.assertEqual(before, (self.root / "whole.tif").read_bytes())
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            compartments.write_compartments(self.root / "labels.tif", self.root / "whole.tif", 3,
                self.root / "resolution.json", self.root / "shift.json", self.root / "tissue.tif",
                compartments_dir=self.root / "compartments", tile_size=13, accelerate=False)
        self.assertEqual(bundle.read_bytes(), before)

    def test_original_frame_downsampled_support_maps_crop_offset(self):
        support = np.zeros((10, 20), dtype=np.uint8)
        support[:, 10:] = 1
        tifffile.imwrite(self.root / "downsampled.tif", support)
        reader = compartments.TissueSupportReader(self.root / "downsampled.tif", "original", (20, 20), (40, 80), (35, 10))
        try:
            actual = reader.read(0, 0, 20, 20)
            self.assertFalse(actual[:, :5].any())
            self.assertTrue(actual[:, 5:].all())
        finally:
            reader.close()

    def test_refuses_overwriting_canonical_labels(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            compartments.write_compartments(self.root / "labels.tif", self.root / "labels.tif", 2,
                self.root / "resolution.json", self.root / "shift.json", self.root / "tissue.tif", accelerate=False)


if __name__ == "__main__":
    unittest.main()
