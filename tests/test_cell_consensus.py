import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "bin" / "build_cell_consensus.py"
SPEC = importlib.util.spec_from_file_location("build_cell_consensus", MODULE_PATH)
consensus = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = consensus
SPEC.loader.exec_module(consensus)


def cell(source, source_id, x, y, contour=None):
    return consensus.Cell(source, source_id, x, y, contour or [])


class ConsensusMatchingTest(unittest.TestCase):
    def test_matching_never_places_two_predictions_from_one_detector_together(self):
        cells = [
            cell("stardist", "s1", 10, 10),
            cell("stardist", "s2", 11, 10),
            cell("hovernet", "h1", 10.2, 10),
            cell("cellvitpp", "c1", 10.4, 10),
        ]
        by_source = {
            source: [i for i, value in enumerate(cells) if value.source == source]
            for source in ("stardist", "hovernet", "cellvitpp")
        }
        dsu = consensus.DisjointSet(cells)
        for _, left, right in consensus.candidate_edges(cells, by_source, 2.0):
            dsu.union(left, right)

        groups = {}
        for index, value in enumerate(cells):
            groups.setdefault(dsu.find(index), []).append(value)
        for group in groups.values():
            sources = [value.source for value in group]
            self.assertEqual(len(sources), len(set(sources)))
        self.assertEqual(sorted(len(group) for group in groups.values()), [1, 3])

    def test_matching_respects_radius(self):
        cells = [cell("stardist", "s1", 0, 0), cell("hovernet", "h1", 5, 0)]
        edges = consensus.candidate_edges(cells, {"stardist": [0], "hovernet": [1]}, 4.99)
        self.assertEqual(edges, [])

    def test_geometry_priority_is_applied_before_centroid_distance(self):
        hover = cell("hovernet", "h1", 0.1, 0, [[0, 0], [1, 0], [0, 1]])
        cellvit = cell("cellvitpp", "c1", 20, 0, [[20, 0], [21, 0], [20, 1]])
        star = cell("stardist", "s1", 0, 0)
        selected = consensus.choose_geometry([star, hover, cellvit], ["cellvitpp", "hovernet"])
        self.assertIs(selected, cellvit)

    def test_geometry_falls_back_to_next_source_with_a_contour(self):
        hover = cell("hovernet", "h1", 0, 0, [[0, 0], [1, 0], [0, 1]])
        cellvit = cell("cellvitpp", "c1", 0, 0)
        selected = consensus.choose_geometry([hover, cellvit], ["cellvitpp", "hovernet"])
        self.assertIs(selected, hover)

    def test_seed_pixels_are_unique_when_centroids_round_to_same_pixel(self):
        records = [
            {"label": 1, "x": 4.1, "y": 5.1},
            {"label": 2, "x": 4.2, "y": 5.2},
            {"label": 3, "x": 4.3, "y": 5.3},
        ]
        seeds = consensus.allocate_unique_seed_pixels(records, width=10, height=10)
        self.assertEqual(len(set(seeds.values())), 3)
        self.assertEqual(seeds[1], (4, 5))
        self.assertTrue(all(0 <= x < 10 and 0 <= y < 10 for x, y in seeds.values()))

    @unittest.skipUnless(
        importlib.util.find_spec("rasterio") and importlib.util.find_spec("shapely"),
        "rasterio and shapely are supplied by the pipeline runtime",
    )
    def test_overlapping_polygons_cannot_erase_consensus_labels(self):
        from shapely.geometry import Polygon

        records = [
            {
                "label": 1, "x": 10.0, "y": 10.0, "support": 2,
                "polygon": Polygon([(5, 5), (15, 5), (15, 15), (5, 15)]),
            },
            {
                "label": 2, "x": 10.0, "y": 10.0, "support": 3,
                "polygon": Polygon([(4, 4), (16, 4), (16, 16), (4, 16)]),
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "labels.tif"
            consensus.write_mask(records, 32, 32, output, tile_size=16, compression="deflate")
            coverage = consensus.validate_mask_label_coverage(output, expected_count=2)
        self.assertEqual(coverage["present_label_count"], 2)
        self.assertEqual(coverage["missing_label_count"], 0)


if __name__ == "__main__":
    unittest.main()
