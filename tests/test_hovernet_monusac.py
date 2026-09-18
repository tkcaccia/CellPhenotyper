import importlib.util
import gzip
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "bin" / "run_hovernet_monusac.py"
SPEC = importlib.util.spec_from_file_location("run_hovernet_monusac", MODULE_PATH)
hovernet = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hovernet
SPEC.loader.exec_module(hovernet)


class MonusacTypeInfoTest(unittest.TestCase):
    def test_model_scope_is_explicitly_non_exhaustive(self):
        self.assertFalse(hovernet.MONUSAC_MODEL_SCOPE["exhaustive_nuclei_detector"])
        self.assertIn("fibroblasts", hovernet.MONUSAC_MODEL_SCOPE["known_omissions"][0])

    def test_type_ids_match_checkpoint_taxonomy(self):
        self.assertEqual(
            [hovernet.MONUSAC_TYPE_INFO[str(value)][0] for value in range(1, 5)],
            ["epithelial", "lymphocyte", "macrophage", "neutrophil"],
        )

    def test_type_map_has_background_plus_four_nucleus_classes(self):
        self.assertEqual(set(hovernet.MONUSAC_TYPE_INFO), {"0", "1", "2", "3", "4"})

    def test_target_mpp_maps_to_expected_objective_power(self):
        self.assertEqual(hovernet.objective_power_for_mpp(0.25), 40)
        self.assertEqual(hovernet.objective_power_for_mpp(0.5), 20)

    def test_task_paths_are_resolved_before_upstream_changes_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                paths = hovernet.resolve_runtime_paths(
                    "crop.tif", "shift.json", "hovernet_repo", "model.tar", "output"
                )
            finally:
                os.chdir(previous)
        self.assertTrue(all(path.is_absolute() for path in paths))
        self.assertEqual(paths[0], (Path(directory) / "crop.tif").resolve())
        self.assertEqual(paths[-1], (Path(directory) / "output").resolve())

    def test_shared_mask_disables_the_upstream_automatic_mask(self):
        mask_dir = Path("/tmp/hovernet-mask")
        self.assertEqual(
            hovernet.inference_mask_args(mask_dir),
            [f"--input_mask_dir={mask_dir}"],
        )
        self.assertEqual(hovernet.inference_mask_args(None), [])

    def test_checkpoint_hash_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.tar"
            checkpoint.write_bytes(b"checkpoint")
            self.assertEqual(
                hovernet.file_sha256(checkpoint),
                "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef",
            )

    def test_transient_cache_storage_reports_logical_and_allocated_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "a.bin").write_bytes(b"a" * 17)
            (root / "nested" / "b.bin").write_bytes(b"b" * 31)
            usage = hovernet.directory_storage(root)
        self.assertEqual(usage["logical_bytes"], 48)
        self.assertEqual(usage["files"], 2)
        self.assertGreaterEqual(usage["allocated_bytes"], usage["logical_bytes"])

    def test_cache_resume_patches_only_an_isolated_runtime_copy(self):
        allocation = '''        self.wsi_pred_map = np.lib.format.open_memmap(
            "%s/pred_map.npy" % self.cache_path,
            mode="w+",
            shape=tuple(self.wsi_proc_shape) + (out_ch,),
            dtype=np.float32,
        )'''
        inference = "        self.__get_raw_prediction(chunk_info_list, patch_info_list)"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            outdir = root / "output"
            cache = outdir / "cache"
            prediction_cache = root / "completed_cache"
            (repo / "infer").mkdir(parents=True)
            cache.mkdir(parents=True)
            prediction_cache.mkdir()
            upstream = repo / "infer" / "wsi.py"
            upstream.write_text(allocation + "\n" + inference + "\n")
            (prediction_cache / "pred_map.npy").write_bytes(b"complete")

            runtime = hovernet.prepare_cache_resume_runtime(repo, outdir, cache, prediction_cache)
            patched = (runtime / "infer" / "wsi.py").read_text()

            self.assertEqual(upstream.read_text(), allocation + "\n" + inference + "\n")
            self.assertIn("HOVERNET_RESUME_PRED_MAP", patched)
            self.assertIn("skipping raw inference", patched)
            self.assertTrue((cache / "pred_map.npy").is_symlink())
            self.assertEqual((cache / "pred_map.npy").resolve(), (prediction_cache / "pred_map.npy").resolve())

    def test_zarr_patch_preserves_float32_and_int32_storage(self):
        constants = tuple(
            value for value in hovernet.enable_zarr_cache_runtime.__code__.co_consts
            if isinstance(value, str)
        )
        assemble = next(value for value in constants if value.startswith("def _assemble_and_flush"))
        allocation = next(value for value in constants if "self.wsi_inst_map = np.lib.format.open_memmap" in value)
        postproc_load = '    wsi_pred_map_ptr = np.load(pred_map_mmap_path, mmap_mode="r")'
        pred_path = 'wsi_pred_map_mmap_path = "%s/pred_map.npy" % self.cache_path'
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "infer").mkdir()
            source = "import numpy as np\n" + assemble + postproc_load + "\n" + allocation + "\n" + pred_path + "\n" + pred_path
            (runtime / "infer" / "wsi.py").write_text(source)
            hovernet.enable_zarr_cache_runtime(runtime)
            patched = (runtime / "infer" / "wsi.py").read_text()
        self.assertIn("pred_map.zarr", patched)
        self.assertIn("pred_inst.zarr", patched)
        self.assertIn("dtype=np.float32", patched)
        self.assertIn("dtype=np.int32", patched)
        self.assertIn("Blosc.BITSHUFFLE", patched)
        self.assertIn("min(1024, int(tile_shape[idx])", patched)
        self.assertNotIn("chunk_pred_map = np.zeros", patched)
        self.assertIn("patches_by_storage_chunk", patched)
        self.assertIn("storage_block = np.zeros", patched)

    def test_streaming_tile_patch_removes_bulky_outputs_and_compresses_json(self):
        constants = tuple(
            value for value in hovernet.enable_lightweight_tile_runtime.__code__.co_consts
            if isinstance(value, str)
        )
        directories = next(value for value in constants if "rm_n_mkdir(self.output_dir + '/mat/')" in value)
        bulky = next(value for value in constants if 'mat_dict = {' in value)
        json_writer = next(value for value in constants if 'self.__save_json(save_path, inst_info_dict, None)' in value)
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "infer").mkdir()
            (runtime / "infer" / "tile.py").write_text(
                "import glob\n" + directories + "\n" + bulky + "\n" + json_writer + "\n"
            )
            hovernet.enable_lightweight_tile_runtime(runtime)
            patched = (runtime / "infer" / "tile.py").read_text()
        self.assertIn('gzip.open(save_path + ".gz"', patched)
        self.assertNotIn("sio.savemat(save_path, mat_dict)", patched)
        self.assertNotIn("cv2.imwrite(save_path, cv2.cvtColor(overlaid_img", patched)
        self.assertNotIn("rm_n_mkdir(self.output_dir + '/mat/')", patched)

    def test_cache_measurement_patch_is_explicitly_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "infer").mkdir()
            upstream = "        rm_n_mkdir(self.cache_path)  # clean up all cache\n"
            (runtime / "infer" / "wsi.py").write_text(upstream)
            hovernet.enable_cache_measurement_runtime(runtime)
            patched = (runtime / "infer" / "wsi.py").read_text()
        self.assertIn('HOVERNET_MEASURE_CACHE_STORAGE', patched)
        self.assertIn("rm_n_mkdir(self.cache_path)", patched)

    def test_streaming_merge_owns_halo_cells_once_and_omits_contours(self):
        class Mask:
            def contains_xy(self, x, y):
                return x < 150 and y < 150

        records = [
            {"name": "tile_0000000", "tile_origin_xy": [0, 0],
             "core_xyxy": [0, 0, 100, 100]},
            {"name": "tile_0000001", "tile_origin_xy": [80, 0],
             "core_xyxy": [100, 0, 200, 100]},
        ]
        payloads = [
            {"nuc": {"1": {"centroid": [90, 40], "contour": [[89, 39]], "type": 2, "type_prob": .9},
                     "2": {"centroid": [110, 40], "contour": [[109, 39]], "type": 1, "type_prob": .8}}},
            {"nuc": {"1": {"centroid": [10, 40], "contour": [[9, 39]], "type": 2, "type_prob": .9},
                     "2": {"centroid": [30, 40], "contour": [[29, 39]], "type": 1, "type_prob": .8}}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / "json"
            raw.mkdir(parents=True)
            for record, payload in zip(records, payloads):
                with gzip.open(raw / f"{record['name']}.json.gz", "wt") as handle:
                    json.dump(payload, handle)
            output = root / "cells.json.gz"
            summary = hovernet.normalize_streaming_tiles(
                records, root / "raw", output, 1.0, {"execution_mode": "streaming_tiles"},
                clean_mask=Mask(), include_contours=False,
            )
            with gzip.open(output, "rt") as handle:
                result = json.load(handle)
        self.assertEqual([cell["centroid"] for cell in result["cells"]], [[90.0, 40.0], [110.0, 40.0]])
        self.assertTrue(all("contour" not in cell for cell in result["cells"]))
        self.assertEqual(summary["input_tile_cells_including_halo_duplicates"], 4)
        self.assertEqual(summary["retained_unique_core_cells"], 2)

    def test_streaming_batches_append_valid_json_without_retaining_raw_records(self):
        records = [
            {"name": "tile_0000000", "tile_origin_xy": [0, 0], "core_xyxy": [0, 0, 10, 10]},
            {"name": "tile_0000001", "tile_origin_xy": [10, 0], "core_xyxy": [10, 0, 20, 10]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw" / "json"
            raw.mkdir(parents=True)
            for index, record in enumerate(records):
                with gzip.open(raw / f"{record['name']}.json.gz", "wt") as handle:
                    json.dump({"nuc": {"1": {"centroid": [5, 5], "type": index + 1}}}, handle)
            stream = io.StringIO()
            first = True
            for index, record in enumerate(records):
                _, retained, first = hovernet.append_streaming_cells(
                    [record], Path(directory) / "raw", stream, 1.0,
                    clean_mask=None, include_contours=False,
                    tile_index_offset=index, first=first,
                )
                self.assertEqual(retained, 1)
            cells = json.loads(f"[{stream.getvalue()}]")
            self.assertFalse(any(raw.iterdir()))
        self.assertEqual([cell["id"] for cell in cells], ["0:1", "1:1"])
        self.assertEqual([cell["centroid"] for cell in cells], [[5.0, 5.0], [15.0, 5.0]])

    def test_normal_runtime_is_copied_to_writable_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "read_only_repo"
            outdir = root / "output"
            repo.mkdir()
            outdir.mkdir()
            (repo / "run_infer.py").write_text("print('ok')\n")
            repo.chmod(0o555)
            try:
                runtime = hovernet.prepare_runtime_repo(repo, outdir)
                (runtime / "debug.log").write_text("writable\n")
            finally:
                repo.chmod(0o755)

            self.assertEqual(runtime, outdir / "hovernet_runtime")
            self.assertEqual((runtime / "run_infer.py").read_text(), "print('ok')\n")
            self.assertEqual((runtime / "debug.log").read_text(), "writable\n")

    def test_unreadable_checkpoint_fails_before_expensive_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.tar"
            checkpoint.write_bytes(b"model")
            checkpoint.chmod(0)
            try:
                with self.assertRaisesRegex(PermissionError, "not readable"):
                    hovernet.require_readable_file(checkpoint, "MoNuSAC checkpoint")
            finally:
                checkpoint.chmod(0o600)

    def test_normalizes_current_upstream_nuc_key_and_coordinates(self):
        payload = {
            "mag": 40,
            "nuc": {
                "7": {
                    "centroid": [20.0, 30.0],
                    "contour": [[19.0, 29.0], [21.0, 29.0], [20.0, 31.0]],
                    "type": 2,
                    "type_prob": 0.9,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "normalized.json"
            raw_path.write_text(json.dumps(payload))
            hovernet.normalize_output(raw_path, output_path, 0.5, {"target_mpp": 0.25})
            cells = json.loads(output_path.read_text())["cells"]
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["id"], "7")
        self.assertEqual(cells[0]["centroid"], [10.0, 15.0])
        self.assertEqual(cells[0]["type_id"], 2)
        self.assertEqual(cells[0]["type"], "lymphocyte")

    def test_unknown_checkpoint_class_remains_auditable(self):
        self.assertEqual(hovernet.monusac_type_name(99), "unknown_99")


if __name__ == "__main__":
    unittest.main()
