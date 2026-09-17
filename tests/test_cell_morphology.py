import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile


SPEC = importlib.util.spec_from_file_location("profile_cell_morphology", Path(__file__).parents[1] / "bin" / "profile_cell_morphology.py")
morphology = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(morphology)


class MorphologyGeometryTest(unittest.TestCase):
    def test_rectangle_exact_physical_pixel_union(self):
        mask = np.ones((4, 8), dtype=bool)
        result = morphology.geometry_features(mask, 0.5, 0.25, x0=20, y0=12, dx=100, dy=30)
        self.assertEqual(result["mask_pixel_count"], 32)
        self.assertAlmostEqual(result["area_um2"], 4)
        self.assertAlmostEqual(result["perimeter_um"], 10)
        self.assertAlmostEqual(result["solidity"], 1)
        self.assertAlmostEqual(result["major_axis_um"], 4 * math.sqrt(4**2 / 12))
        self.assertAlmostEqual(result["minor_axis_um"], 4 * math.sqrt(1**2 / 12))
        self.assertAlmostEqual(result["orientation_rad"], 0)
        self.assertAlmostEqual(result["mask_centroid_x_um"], 62)
        self.assertAlmostEqual(result["mask_centroid_y_um"], 11)

    def test_scale_changes_lengths_and_areas_but_not_shape(self):
        yy, xx = np.mgrid[:41, :51]
        mask = ((xx-25)/18)**2 + ((yy-20)/8)**2 <= 1
        a = morphology.geometry_features(mask, 0.25, 0.25)
        b = morphology.geometry_features(mask, 0.5, 0.5)
        for key in ("major_axis_um", "minor_axis_um", "perimeter_um"):
            self.assertAlmostEqual(b[key], 2*a[key])
        self.assertAlmostEqual(b["area_um2"], 4*a["area_um2"])
        for key in ("eccentricity", "solidity", "circularity", "boundary_irregularity"):
            self.assertAlmostEqual(b[key], a[key])

    def test_rotated_ellipse_orientation_and_axes(self):
        yy, xx = np.mgrid[-80:81, -80:81]
        theta = math.pi / 6
        along = xx*math.cos(theta) + yy*math.sin(theta)
        across = -xx*math.sin(theta) + yy*math.cos(theta)
        result = morphology.geometry_features((along/40)**2 + (across/12)**2 <= 1, 0.25, 0.25)
        self.assertAlmostEqual(result["orientation_rad"], theta, delta=0.01)
        self.assertAlmostEqual(result["major_axis_um"], 20, delta=0.2)
        self.assertAlmostEqual(result["minor_axis_um"], 6, delta=0.1)

    def test_single_pixel_and_holes_are_explicit(self):
        one = morphology.geometry_features(np.ones((1, 1), dtype=bool), 1, 1)
        self.assertTrue(math.isnan(one["orientation_rad"]))
        self.assertEqual(one["perimeter_um"], 4)
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False
        hole = morphology.geometry_features(mask, 1, 1)
        self.assertEqual(hole["perimeter_um"], 24)
        self.assertAlmostEqual(hole["solidity"], 24/25)

    def test_empty_cell_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty canonical"):
            morphology.geometry_features(np.zeros((2, 2), dtype=bool), 1, 1)


class AppearanceTest(unittest.TestCase):
    def test_normalized_rgb_is_identical_across_supported_intensity_encodings(self):
        yy, xx = np.mgrid[:8, :8]
        rgb = np.repeat(np.where((xx+yy) % 2, 32, 224).astype(np.uint8)[..., None], 3, axis=2)
        mask = np.ones((8, 8), dtype=bool)
        baseline = morphology.appearance_features(rgb, mask, .5, .5)
        for converted, white in ((rgb.astype(np.uint16)*257, None), (rgb.astype(float)/255., 1.)):
            actual = morphology.appearance_features(converted, mask, .5, .5, white_level=white)
            for key in baseline:
                self.assertAlmostEqual(actual[key], baseline[key], places=12, msg=key)

    def test_physical_texture_lag_records_effective_pixel_offsets(self):
        from cell_morphology_io import texture_sampling
        coarse = texture_sampling(.5, .5, .5)
        fine = texture_sampling(.25, .25, .5)
        self.assertEqual(coarse["offsets_px"], [[1, 0], [0, 1], [1, 1], [-1, 1]])
        self.assertEqual(fine["offsets_px"], [[2, 0], [0, 2], [2, 2], [-2, 2]])
        self.assertEqual(coarse["effective_offsets_um"], fine["effective_offsets_um"])
        self.assertEqual(texture_sampling(.3, .4, .5)["effective_offsets_um"][2], [.6, .4])

    def test_large_lag_retains_intensities_with_explicit_missing_texture(self):
        rgb = np.full((2, 2, 3), 120, np.uint8)
        result = morphology.appearance_features(rgb, np.ones((2, 2), bool), .25, .25, texture_lag_um=10)
        self.assertTrue(math.isfinite(result["rgb_od_mean_mean"]))
        self.assertEqual(result["od_texture_pair_count"], 0)
        self.assertTrue(math.isnan(result["od_glcm_contrast"]))

    def test_invalid_physical_lag_or_levels_fail(self):
        rgb, mask = np.ones((2, 2, 3), np.uint8), np.ones((2, 2), bool)
        for lag in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                morphology.appearance_features(rgb, mask, .25, .25, texture_lag_um=lag)
        with self.assertRaisesRegex(ValueError, "levels"):
            morphology.appearance_features(rgb, mask, .25, .25, levels=1)

    def test_background_and_other_cells_do_not_contaminate_features(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:10, 2:10] = True
        mask[4:6, 4:6] = False
        image = np.full((12, 12, 3), 120, dtype=np.uint8)
        clean = morphology.appearance_features(image, mask, 0.5, 0.5)
        image[~mask] = 0
        dirty = morphology.appearance_features(image, mask, 0.5, 0.5)
        self.assertEqual(clean, dirty)
        self.assertEqual(clean["od_glcm_contrast"], 0)
        self.assertEqual(clean["od_gradient_mean_per_um"], 0)
        self.assertAlmostEqual(clean["rgb_od_mean_mean"], -math.log(121/256))

    def test_texture_detects_within_nucleus_variation_and_physical_gradient(self):
        yy, xx = np.mgrid[:10, :10]
        image = np.repeat(np.where((xx+yy) % 2, 20, 220).astype(np.uint8)[..., None], 3, axis=2)
        mask = np.ones((10, 10), dtype=bool)
        a = morphology.appearance_features(image, mask, 0.5, 0.5)
        b = morphology.appearance_features(image, mask, 1, 1)
        self.assertGreater(a["od_glcm_contrast"], 0)
        self.assertGreater(a["rgb_od_mean_std"], 0)
        self.assertGreater(a["od_glcm_entropy_bits"], 0)
        self.assertAlmostEqual(a["od_gradient_mean_per_um"], b["od_gradient_mean_per_um"]*2)

    def test_single_pixel_has_missing_texture_not_fabricated_zero(self):
        result = morphology.appearance_features(np.full((1, 1, 3), 120, dtype=np.uint8), np.ones((1, 1), dtype=bool), 1, 1)
        self.assertEqual(result["od_texture_pair_count"], 0)
        self.assertTrue(math.isnan(result["od_glcm_contrast"]))


class WindowReaderTest(unittest.TestCase):
    def test_pyramidal_tiff_reads_level_zero(self):
        image = np.arange(64*64, dtype=np.uint16).reshape(64, 64)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pyramid.tif"
            with tifffile.TiffWriter(path) as writer:
                writer.write(image, subifds=1, compression="deflate", tile=(16, 16))
                writer.write(image[::2, ::2], subfiletype=1, compression="deflate", tile=(16, 16))
            with morphology.WindowReader(path) as reader:
                self.assertEqual(reader.shape, (64, 64))
                np.testing.assert_array_equal(reader.read(20, 22, 40, 50), image[22:50, 20:40])

    def test_zarr_level_zero_windows_when_available(self):
        try:
            import zarr
        except ImportError:
            self.skipTest("Optional Zarr dependency unavailable")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.zarr"
            group = zarr.open_group(str(path), mode="w")
            image = np.arange(32*48*3, dtype=np.uint16).reshape(32, 48, 3)
            group.create_array("0", data=image, chunks=(16, 16, 3))
            with morphology.WindowReader(path) as reader:
                np.testing.assert_array_equal(reader.read(12, 4, 22, 15), image[4:15, 12:22])

    def test_compressed_tiled_and_stripped_windows_without_full_image_decode(self):
        image = np.arange(65*79*3, dtype=np.uint16).reshape(65, 79, 3)
        with tempfile.TemporaryDirectory() as temp:
            for name, opts in (("tile", {"tile": (16, 16)}), ("strip", {"rowsperstrip": 13})):
                path = Path(temp) / f"{name}.tif"
                tifffile.imwrite(path, image, photometric="rgb", compression="deflate", **opts)
                with patch.object(tifffile.TiffPage, "asarray", side_effect=AssertionError("Whole page decode prohibited")):
                    with morphology.WindowReader(path, cache_mb=0.005) as reader:
                        self.assertEqual(reader.backend, "tiff_segment_windows")
                        for bounds in ((5, 11, 72, 49), (-4, -3, 20, 18), (60, 61, 79, 65)):
                            x0, y0, x1, y1 = bounds
                            np.testing.assert_array_equal(reader.read(*bounds), image[max(0,y0):y1, max(0,x0):x1])

    def test_planar_rgb_and_grayscale_windows(self):
        with tempfile.TemporaryDirectory() as temp:
            rgb = np.arange(3*37*41, dtype=np.uint16).reshape(3, 37, 41)
            for compressed in (False, True):
                path = Path(temp) / f"planar{compressed}.tif"
                tifffile.imwrite(path, rgb, photometric="rgb", planarconfig="separate", compression="deflate" if compressed else None, rowsperstrip=7)
                with morphology.WindowReader(path) as reader:
                    np.testing.assert_array_equal(reader.read(10, 12, 20, 30), np.moveaxis(rgb[:, 12:30, 10:20], 0, -1))
            path = Path(temp) / "labels.tif"
            tifffile.imwrite(path, rgb[0], compression="deflate", tile=(16, 16))
            with morphology.WindowReader(path) as reader:
                np.testing.assert_array_equal(reader.read(13, 17, 30, 37), rgb[0, 17:37, 13:30])

    def test_oversized_compressed_strip_fails_before_decode(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "strip.tif"
            tifffile.imwrite(path, np.zeros((128, 128), dtype=np.uint16), compression="deflate")
            with self.assertRaisesRegex(ValueError, "retile"):
                morphology.WindowReader(path, max_segment_mb=0.001)


class ProfileIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.labels = np.zeros((64, 64), dtype=np.uint32)
        self.labels[0:4, 0:5] = 4
        self.labels[20:27, 28:33] = 9001
        self.labels[23, 30] = 0
        self.rgb = np.full((64, 64, 3), 150, dtype=np.uint8)
        self.rows = [dict(label=4, x=2, y=2, xmin=-1, ymin=-1, xmax=4, ymax=3, geometry_source="cellvitpp"),
                     dict(label=9001, x=30, y=23, xmin=29, ymin=21, xmax=31, ymax=25, geometry_source="hovernet")]
        (self.root / "shift.json").write_text(json.dumps({"source_mpp": 0.25, "offset_crop_to_original": {"dx": 100, "dy": 200}}))
        self.write_inputs()

    def tearDown(self):
        self.temp.cleanup()

    def write_inputs(self):
        tifffile.imwrite(self.root / "labels.tif", self.labels, compression="deflate", tile=(16, 16))
        tifffile.imwrite(self.root / "image.tif", self.rgb, photometric="rgb", compression="deflate", tile=(16, 16))
        with (self.root / "objects.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["label", "x", "y", "xmin", "ymin", "xmax", "ymax", "geometry_source"])
            writer.writeheader()
            writer.writerows(self.rows)

    def run_profile(self):
        return morphology.profile_cells(self.root / "labels.tif", self.root / "image.tif", self.root / "objects.csv", self.root / "shift.json", self.root / "output", tile_size=11)

    def test_profile_recovers_actual_mask_bounds_and_preserves_noncontiguous_ids(self):
        summary = self.run_profile()
        self.assertEqual(summary["cells"], 2)
        with (self.root / "output" / "cell_morphology.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["label"] for row in rows], ["4", "9001"])
        self.assertEqual(rows[0]["mask_bbox_xmin"], "0")
        self.assertEqual(rows[0]["mask_bbox_xmax"], "5")
        self.assertEqual(rows[1]["mask_bbox_xmin"], "28")
        self.assertEqual(rows[1]["mask_pixel_count"], "34")
        self.assertEqual(rows[0]["touches_image_edge"], "1")
        self.assertEqual(rows[0]["geometry_source"], "cellvitpp")
        self.assertFalse(summary["io"]["whole_image_decode"])

    def test_duplicate_ids_fail(self):
        self.rows.append(self.rows[0].copy())
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_profile()

    def test_missing_mask_id_fails(self):
        self.labels[self.labels == 9001] = 0
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "absent from mask"):
            self.run_profile()

    def test_unmatched_mask_id_fails(self):
        self.labels[48, 48] = 700
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "absent from objects"):
            self.run_profile()

    def test_empty_specimen_writes_valid_empty_profile(self):
        self.labels[:] = 0
        self.rows = []
        self.write_inputs()
        summary = self.run_profile()
        self.assertEqual(summary["cells"], 0)
        self.assertEqual(summary["mask_pixels"], 0)

    def test_missing_calibration_fails(self):
        (self.root / "shift.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "source_mpp"):
            self.run_profile()

    def test_passed_resolution_report_overrides_stale_shift_calibration(self):
        (self.root / "shift.json").write_text(json.dumps({"source_mpp": 1000, "offset_crop_to_original": {"dx": 30, "dy": 50}}))
        with self.assertRaisesRegex(ValueError, "plausible nuclear-image"):
            morphology.read_calibration(self.root / "shift.json")
        report = self.root / "resolution.json"
        report.write_text(json.dumps({"status": "pass", "mpp_x": 0.3, "mpp_y": 0.4}))
        self.assertEqual(morphology.read_calibration(self.root / "shift.json", report), (0.3, 0.4, 30, 50))
        report.write_text(json.dumps({"status": "fail", "mpp_x": 0.3, "mpp_y": 0.4}))
        with self.assertRaisesRegex(ValueError, "status 'pass'"):
            morphology.read_calibration(self.root / "shift.json", report)

    def test_shift_image_dimension_mismatch_fails(self):
        (self.root / "shift.json").write_text(json.dumps({"source_mpp": 0.25, "crop_size": {"width": 100, "height": 100}}))
        with self.assertRaisesRegex(ValueError, "crop_size differs"):
            self.run_profile()


if __name__ == "__main__":
    unittest.main()
