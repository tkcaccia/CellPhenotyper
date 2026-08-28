import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
