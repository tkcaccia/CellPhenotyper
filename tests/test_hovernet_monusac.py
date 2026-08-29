import importlib.util
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
