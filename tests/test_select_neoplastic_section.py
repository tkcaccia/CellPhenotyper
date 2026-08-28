import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "bin" / "select_neoplastic_section.py"
SPEC = importlib.util.spec_from_file_location("select_neoplastic_section", MODULE_PATH)
selector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


@unittest.skipUnless(importlib.util.find_spec("shapely"), "shapely is supplied by the pipeline runtime")
class NeoplasticSectionTest(unittest.TestCase):
    def test_multipolygon_is_split_and_named_stably(self):
        payload = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"value": 2, "classification": "color_2"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                        [[[20, 0], [25, 0], [25, 5], [20, 5], [20, 0]]],
                    ],
                },
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sections.geojson"
            path.write_text(json.dumps(payload))
            sections = selector.load_sections(path, "standard")
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0]["section_id"], "standard_class_2_component_1")
        self.assertEqual(sections[0]["area_px2"], 100.0)

    def test_named_cell_counts_choose_most_neoplastic_component(self):
        payload = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"value": 1}, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}},
                {"type": "Feature", "properties": {"value": 2}, "geometry": {"type": "Polygon", "coordinates": [[[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]]]}},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geojson = root / "sections.geojson"
            objects = root / "objects.csv"
            geojson.write_text(json.dumps(payload))
            with objects.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["x", "y", "cellvitpp_type"])
                writer.writeheader()
                writer.writerows([
                    {"x": 1, "y": 1, "cellvitpp_type": "connective"},
                    {"x": 21, "y": 1, "cellvitpp_type": "neoplastic"},
                    {"x": 22, "y": 2, "cellvitpp_type": "Neoplastic"},
                    {"x": 100, "y": 100, "cellvitpp_type": "neoplastic"},
                ])
            sections = selector.load_sections(geojson, "standard")
            audit = selector.count_cells(objects, sections, {"neoplastic"}, 8)
        selected = selector.select_section(sections)
        self.assertEqual(selected["class_value"], 2)
        self.assertEqual(selected["neoplastic_cells"], 2)
        self.assertEqual(audit["consensus_cells_assigned"], 3)

    def test_ties_are_deterministic(self):
        sections = [
            {"section_id": "b", "neoplastic_cells": 2, "total_cells": 4, "area_px2": 20},
            {"section_id": "a", "neoplastic_cells": 2, "total_cells": 4, "area_px2": 20},
        ]
        self.assertEqual(selector.select_section(sections)["section_id"], "a")


if __name__ == "__main__":
    unittest.main()
