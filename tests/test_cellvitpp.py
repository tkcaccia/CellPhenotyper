import importlib.util
import json
import hashlib
import os
import shutil
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
    def test_zero_clean_mask_core_is_conservative(self):
        import numpy as np
        import tifffile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = root / 'mask.tif'
            shift = root / 'shift.json'
            shift.write_text(json.dumps({'crop_size': {'width': 16, 'height': 16}}))
            pixels = np.zeros((2, 2), dtype=np.uint8)
            pixels[0, 1] = 255
            tifffile.imwrite(mask, pixels)
            shard = {'core': [0, 0, 8, 8]}
            evidence = cellvit.empty_clean_core_evidence(mask, shift, shard, 16, 16)
            self.assertEqual(evidence['mask_xyxy'], [0, 0, 1, 1])
            self.assertEqual(evidence['nonzero_mask_pixels'], 0)
            shard['core'] = [0, 0, 9, 8]
            self.assertIsNone(cellvit.empty_clean_core_evidence(mask, shift, shard, 16, 16))

    def test_missing_native_cells_requires_exact_zero_patch_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            raw.mkdir()
            prepared = root / 'cellvit_input.tif'
            prepared.write_bytes(b'checked-input')
            log = root / 'native.log'
            log.write_text(f'No patches sampled from {prepared}\n'
                           'No cells have been extracted\nFinished processing\n')
            record = {'native_exit_code': 0, 'truncated': False,
                      'sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
                      'bytes': log.stat().st_size}
            evidence = cellvit.no_patch_native_evidence(log, record, prepared, raw)
            self.assertEqual(evidence['raw_output_files'], 0)
            (raw / 'partial.pt').write_bytes(b'partial native output')
            self.assertIsNone(cellvit.no_patch_native_evidence(log, record, prepared, raw))
            (raw / 'partial.pt').unlink()
            log.write_text(f'No patches sampled from {prepared}\nTotal patches sampled: 3\n'
                           'No cells have been extracted\nFinished processing\n')
            record['sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
            self.assertIsNone(cellvit.no_patch_native_evidence(log, record, prepared, raw))
            log.write_text('No cells have been extracted\n')
            record['sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
            self.assertIsNone(cellvit.no_patch_native_evidence(log, record, prepared, raw))
            log.write_text(f'No patches sampled from {prepared}\nNo cells have been extracted\n')
            record['sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
            record['truncated'] = True
            self.assertIsNone(cellvit.no_patch_native_evidence(log, record, prepared, raw))

    def test_empty_native_source_declares_zero_rows_at_sam_width(self):
        class FakeTorch:
            float32 = 'float32'
            float64 = 'float64'

            @staticmethod
            def empty(shape, dtype):
                return {'shape': shape, 'dtype': dtype}

            @staticmethod
            def save(payload, path):
                Path(path).write_text(json.dumps(payload))

        with tempfile.TemporaryDirectory() as directory, \
                patch('cellvit_models._torch', return_value=FakeTorch):
            root = Path(directory)
            cells, graph = cellvit.materialize_empty_native_source(
                root, 0.25, 0.25, 1280, {'policy': 'tested-zero-clean-core'})
            self.assertEqual(json.loads(cells.read_text())['cells'], [])
            tensors = json.loads(graph.read_text())
            self.assertEqual(tensors['x']['shape'], [0, 1280])
            self.assertEqual(tensors['positions']['shape'], [0, 2])

    def test_native_log_capture_records_exact_successful_output(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'native.log'
            record = cellvit.run_native_with_bounded_log(
                [sys.executable, '-c', 'print("No patches sampled from /tmp/core.tif"); '
                 'print("No cells have been extracted")'], os.environ.copy(), log)
            self.assertEqual(record['native_exit_code'], 0)
            self.assertFalse(record['truncated'])
            self.assertEqual(record['sha256'], hashlib.sha256(log.read_bytes()).hexdigest())

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

    def test_model_names_resolve_to_the_official_cached_checkpoints(self):
        cache = Path("/models")
        self.assertEqual(cellvit.required_model_path(cache, "HIPT"), cache / "CellViT-256-x40-AMP.pth")
        self.assertEqual(cellvit.required_model_path(cache, "SAM"), cache / "CellViT-SAM-H-x40-AMP.pth")

    def test_runtime_environment_keeps_ray_and_caches_in_task_output(self):
        with tempfile.TemporaryDirectory() as directory:
            outdir = Path(directory) / "output"
            env, runtime_dir, ray_temp_dir = cellvit.prepare_runtime_env(outdir)
            try:
                self.assertEqual(Path(env["RAY_TMPDIR"]), ray_temp_dir)
                self.assertEqual(Path(env["TMPDIR"]), ray_temp_dir)
                self.assertEqual(Path(env["XDG_CACHE_HOME"]), runtime_dir / "cache")
                self.assertEqual(Path(env["MPLCONFIGDIR"]), runtime_dir / "matplotlib")
                self.assertEqual(ray_temp_dir.parent, Path("/tmp"))
                cellvit.validate_ray_temp_dir(ray_temp_dir)
            finally:
                shutil.rmtree(ray_temp_dir, ignore_errors=True)

    def test_long_ray_temp_path_is_rejected_before_inference(self):
        long_path = Path("/tmp") / ("x" * 90)
        with self.assertRaisesRegex(RuntimeError, "AF_UNIX"):
            cellvit.validate_ray_temp_dir(long_path)

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
