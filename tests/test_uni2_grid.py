import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "bin" / "uni2_grid.py"
SPEC = importlib.util.spec_from_file_location("uni2_grid", MODULE_PATH)
grid = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = grid
SPEC.loader.exec_module(grid)


class UNI2GridAssignmentTest(unittest.TestCase):
    def test_spatial_grid_geometry_matches_scaled_inner_square(self):
        context, inner, stride = grid.resolve_spatial_grid_geometry(
            model_tile_size=224,
            inner_square_size=90,
            source_mpp=0.08706,
            target_mpp=0.25,
        )
        self.assertEqual(context, 643)
        self.assertEqual(inner, 258)
        self.assertEqual(stride, 258)
        self.assertEqual(context - stride, 385)

    def test_spatial_grid_stride_can_be_decoupled_from_inner_square(self):
        context, inner, stride = grid.resolve_spatial_grid_geometry(
            model_tile_size=224,
            inner_square_size=28,
            grid_stride_size=56,
            source_mpp=0.273774374855905,
            target_mpp=0.25,
        )
        self.assertEqual((context, inner, stride), (205, 26, 51))

    def test_centered_lattice_cores_are_adjacent_and_cover_axis(self):
        for length in (1, 89, 90, 91, 180, 205, 1000):
            centers, starts, ends = grid.centered_axis_lattice(length, 90)
            self.assertTrue(np.all(centers >= 0))
            self.assertTrue(np.all(centers < length))
            self.assertLessEqual(int(starts[0]), 0)
            self.assertGreaterEqual(int(ends[-1]), length)
            np.testing.assert_array_equal(starts[1:], ends[:-1])

    def test_rounded_centroid_and_grid_assignment_use_same_coordinate(self):
        cx = np.asarray([511.4, 511.6, 1023.6])
        cy = np.asarray([511.6, 511.4, 1023.6])
        center_x, center_y, rows, cols, tile_ids, tile_w, tile_h = grid.assign_rounded_centers_to_grid(
            cx, cy, height=1024, width=1024, grid_rows=2, grid_cols=2
        )
        self.assertEqual((tile_w, tile_h), (512, 512))
        self.assertEqual(center_x.tolist(), [511, 512, 1023])
        self.assertEqual(center_y.tolist(), [512, 511, 1023])
        self.assertEqual(rows.tolist(), [1, 0, 1])
        self.assertEqual(cols.tolist(), [0, 1, 1])
        self.assertEqual(tile_ids.tolist(), [2, 1, 3])

    def test_every_assigned_center_lies_inside_its_clipped_grid_core(self):
        values = np.linspace(0.0, 999.0, 10001)
        center_x, center_y, rows, cols, _, tile_w, tile_h = grid.assign_rounded_centers_to_grid(
            values, values[::-1], height=1000, width=1000, grid_rows=10, grid_cols=10
        )
        for x, y, row, col in zip(center_x, center_y, rows, cols):
            self.assertGreaterEqual(x, col * tile_w)
            self.assertLess(x, min(1000, (col + 1) * tile_w))
            self.assertGreaterEqual(y, row * tile_h)
            self.assertLess(y, min(1000, (row + 1) * tile_h))


if __name__ == "__main__":
    unittest.main()
