import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "bin" / "run_cellvitpp.py"
SPEC = importlib.util.spec_from_file_location("run_cellvitpp", MODULE_PATH)
cellvit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = cellvit
SPEC.loader.exec_module(cellvit)


class CellVitBatchTest(unittest.TestCase):
    def test_explicit_batch_is_clamped_to_supported_cli_range(self):
        self.assertEqual(cellvit.auto_batch_size(1, 0), 2)
        self.assertEqual(cellvit.auto_batch_size(16, 0), 16)
        self.assertEqual(cellvit.auto_batch_size(100, 0), 48)

    @patch("subprocess.check_output", side_effect=RuntimeError("no nvidia-smi"))
    def test_detection_failure_uses_valid_low_memory_batch(self, _):
        self.assertEqual(cellvit.auto_batch_size(0, 0), 2)

    def test_paths_are_stable_for_ray_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                paths = cellvit.resolve_runtime_paths("crop.tif", "shift.json", "output")
            finally:
                os.chdir(previous)
        self.assertTrue(all(path.is_absolute() for path in paths))
        self.assertEqual(paths[0], (Path(directory) / "crop.tif").resolve())
        self.assertEqual(paths[-1], (Path(directory) / "output").resolve())

    def test_pannuke_ids_match_official_cellvit_output(self):
        self.assertEqual(
            cellvit.PANNUKE_TYPE_MAP,
            {
                "1": "neoplastic", "2": "inflammatory", "3": "connective",
                "4": "dead", "5": "epithelial",
            },
        )

    def test_normalization_preserves_id_and_emits_name(self):
        payload = {
            "type_map": {"1": "Neoplastic", "3": "Connective"},
            "wsi_metadata": {"base_mpp": 0.25},
            "cells": [{"centroid": [12, 34], "contour": [], "type": 1, "type_prob": 0.8}],
        }
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "cells.json"
            output = Path(directory) / "normalized.json"
            raw.write_text(json.dumps(payload))
            cellvit.normalize_output(raw, output, "pannuke", {"source_mpp": 0.25})
            normalized = json.loads(output.read_text())
        self.assertEqual(normalized["type_map"]["1"], "neoplastic")
        self.assertEqual(normalized["cells"][0]["type_id"], 1)
        self.assertEqual(normalized["cells"][0]["type"], "neoplastic")


if __name__ == "__main__":
    unittest.main()
