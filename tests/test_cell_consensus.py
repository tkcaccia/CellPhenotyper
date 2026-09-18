import importlib.util
import gzip
import json
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
    def test_load_cells_accepts_compressed_hovernet_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hovernet_cells.json.gz"
            with gzip.open(path, "wt") as handle:
                json.dump({"cells": [{"id": "h1", "centroid": [2, 3],
                                      "type_id": 2, "type": "lymphocyte"}]}, handle)
            cells = consensus.load_cells(path, "hovernet")
        self.assertEqual(len(cells), 1)
        self.assertEqual((cells[0].source_id, cells[0].x, cells[0].y), ("h1", 2.0, 3.0))

    def test_count_qc_accounts_for_non_exhaustive_monusac_scope(self):
        by_source = {
            "stardist": list(range(100)),
            "hovernet": list(range(15)),
            "cellvitpp": list(range(90)),
        }
        qc = consensus.detector_count_qc(by_source, warning_ratio=2.0)
        self.assertEqual(qc["status"], "expected_non_exhaustive_scope_imbalance")
        self.assertAlmostEqual(qc["broad_scope_max_min_ratio"], 100 / 90)
        self.assertAlmostEqual(qc["all_detector_max_min_ratio"], 100 / 15)
        self.assertTrue(qc["detector_scopes"]["hovernet"]["known_non_exhaustive"])

    def test_count_qc_flags_disagreement_between_broad_scope_detectors(self):
        by_source = {
            "stardist": list(range(100)),
            "hovernet": list(range(40)),
            "cellvitpp": list(range(30)),
        }
        qc = consensus.detector_count_qc(by_source, warning_ratio=2.0)
        self.assertEqual(qc["status"], "review_broad_scope_detector_imbalance")

    def test_count_qc_reviews_when_scoped_support_detector_is_empty(self):
        by_source = {
            "stardist": list(range(100)),
            "hovernet": [],
            "cellvitpp": list(range(90)),
        }
        qc = consensus.detector_count_qc(by_source, warning_ratio=2.0)
        self.assertEqual(qc["status"], "review_missing_scoped_support_output")
        self.assertEqual(qc["missing_scoped_support_detectors"], ["hovernet"])

    def test_count_qc_fails_when_a_broad_detector_is_empty(self):
        qc = consensus.detector_count_qc(
            {"stardist": list(range(100)), "hovernet": list(range(20)), "cellvitpp": []},
            warning_ratio=2.0,
        )
        self.assertEqual(qc["status"], "failed_missing_broad_detector_output")
        self.assertEqual(qc["missing_broad_scope_detectors"], ["cellvitpp"])

    def test_spatial_coverage_reports_missing_detector_regions(self):
        cells = [
            cell("stardist", "s1", 10, 10),
            cell("stardist", "s2", 90, 90),
            cell("cellvitpp", "c1", 10, 10),
            cell("cellvitpp", "c2", 90, 90),
            cell("hovernet", "h1", 10, 10),
        ]
        qc = consensus.detector_spatial_coverage(cells, width=100, height=100, bins=2)
        self.assertEqual(qc["broad_scope_jointly_occupied_bins"], 2)
        self.assertEqual(qc["broad_scope_bins_without_hovernet"], 1)
        self.assertEqual(qc["broad_scope_bins_without_hovernet_fraction"], 0.5)

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

    def test_group_agreement_separates_instance_support_from_phenotype_evidence(self):
        group = [
            consensus.Cell("stardist", "s1", 10, 10, []),
            consensus.Cell("hovernet", "h1", 10.5, 10, [], cell_type="lymphocyte"),
            consensus.Cell("cellvitpp", "c1", 10, 10.5, [], cell_type="inflammatory"),
        ]
        result = consensus.summarize_group_agreement(group, mpp=0.25, match_radius_um=4.0)
        self.assertEqual(result["instance_support"], 3)
        self.assertEqual(result["phenotype_evidence_count"], 2)
        self.assertEqual(result["phenotype_status"], "detector_specific_evidence_only")
        self.assertTrue(result["has_broad_instance_consensus"])
        self.assertEqual(result["broad_instance_support"], 2)
        self.assertEqual(result["scoped_support_sources"], ["hovernet"])
        self.assertNotIn("phenotype_consensus", result)
        self.assertGreater(result["agreement_score"], 0.9)

    def test_scoped_support_does_not_inflate_broad_detector_agreement(self):
        two = [cell("stardist", "s1", 1, 1), cell("cellvitpp", "c1", 1, 1)]
        three = two + [cell("hovernet", "h1", 1, 1)]
        score_two = consensus.summarize_group_agreement(two, 0.25, 4.0)["agreement_score"]
        score_three = consensus.summarize_group_agreement(three, 0.25, 4.0)["agreement_score"]
        self.assertEqual(score_two, score_three)

    def test_broad_pair_policy_abstains_from_one_broad_plus_scoped_support(self):
        group = [cell("stardist", "s1", 1, 1), cell("hovernet", "h1", 1, 1)]
        accepted, decision = consensus.component_is_accepted(group, 2, "broad_pair")
        self.assertFalse(accepted)
        self.assertEqual(decision, "abstained_missing_broad_detector_consensus")
        legacy_accepted, _ = consensus.component_is_accepted(group, 2, "any_two")
        self.assertTrue(legacy_accepted)

    def test_scoped_detector_does_not_shift_canonical_broad_centroid_score(self):
        broad = [cell("stardist", "s1", 0, 0), cell("cellvitpp", "c1", 1, 0)]
        with_far_scoped = broad + [cell("hovernet", "h1", 10, 0)]
        result = consensus.summarize_group_agreement(with_far_scoped, 0.25, 4.0)
        self.assertAlmostEqual(result["broad_centroid_distance_um_max"], 0.25)
        self.assertAlmostEqual(result["all_source_centroid_distance_um_max"], 2.5)

    def test_transitive_scoped_match_does_not_bypass_broad_pair_radius(self):
        group = [
            cell("stardist", "s1", 0, 0),
            cell("hovernet", "h1", 4, 0),
            cell("cellvitpp", "c1", 8, 0),
        ]
        accepted, decision = consensus.component_is_accepted(
            group, 2, "broad_pair", mpp=1.0, match_radius_um=4.0
        )
        self.assertFalse(accepted)
        self.assertEqual(decision, "abstained_broad_detector_distance")

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

    @unittest.skipUnless(importlib.util.find_spec("shapely"), "shapely is supplied by the pipeline runtime")
    def test_detector_agreement_reports_detection_and_boundary_metrics(self):
        square_a = [[0, 0], [2, 0], [2, 2], [0, 2]]
        square_b = [[1, 0], [3, 0], [3, 2], [1, 2]]
        groups = [
            [cell("stardist", "s1", 1, 1, square_a), cell("hovernet", "h1", 2, 1, square_b)],
            [cell("stardist", "s2", 10, 10, square_a)],
        ]
        rows = consensus.detector_agreement_benchmark(
            groups, {"stardist": [0, 2], "hovernet": [1]}, mpp=0.5
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["matched_pairs"], 1)
        self.assertEqual(row["source_a_match_fraction"], 1.0)
        self.assertEqual(row["source_b_match_fraction"], 0.5)
        self.assertAlmostEqual(row["centroid_distance_um_median"], 0.5)
        self.assertAlmostEqual(row["polygon_iou_median"], 1.0 / 3.0)
        self.assertAlmostEqual(row["hausdorff_distance_um_median"], 0.5)

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
