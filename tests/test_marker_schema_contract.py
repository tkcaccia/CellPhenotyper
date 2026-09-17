import copy
import csv
import importlib.util
import json
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parents[1] / "bin"))
import quantify_gigatime_intensity as quantify

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    import run_gigatime_on_crop as gigatime


class MarkerSchemaContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.shape = (33, 41)
        rng = np.random.default_rng(29)
        self.scores = rng.random((23, *self.shape), dtype=np.float32)
        nuclei = np.zeros(self.shape, dtype=np.uint32)
        nuclei[5:12, 6:14] = 1
        nuclei[19:30, 24:38] = 17
        nuclei[22:25, 28:32] = 0
        whole = nuclei.copy()
        whole[3:14, 4:16] = 1
        whole[17:32, 22:40] = 17
        ring = np.where(nuclei == 0, whole, 0).astype(np.uint32)
        self.masks = {}
        for name, array in (("nuclei", nuclei), ("cyto", whole), ("ring", ring)):
            path = self.root / f"{name}.tif"
            tifffile.imwrite(path, array, compression="deflate", tile=(16, 16))
            self.masks[name] = str(path)
        self.metadata = {
            "inference_shape_yx": list(self.shape), "original_shape_yx": list(self.shape),
            "model_channels": quantify.CANONICAL_CHANNEL_NAMES, "output_dtype": "float32",
            "resolution_contract": "exact_mpp_v2", "source_mpp": 0.25, "effective_mpp": 0.25,
            "downsample_factor": 1.0,
            "inference_complete": True,
            "model_arithmetic": "cpu_float32",
            "model_provenance": {"checkpoints": [{"status": "verified", "sha256": "a"*64}]},
        }
        self.schema = quantify.build_marker_schema(self.metadata, self.masks)
        self.metadata["marker_schema"] = self.schema
        self.write_store()

    def tearDown(self):
        self.temp.cleanup()

    def write_store(self, scores=None, names=None, output_dtype="float32"):
        self.image = self.root / "gigatime_probs.ome.tif"
        tifffile.imwrite(self.image, self.scores if scores is None else scores,
                         ome=True, compression="deflate", tile=(16, 16),
                         metadata={"axes": "CYX", "Channel": {"Name": names or quantify.CANONICAL_CHANNEL_NAMES}})
        meta = {**self.metadata, "output_dtype": output_dtype}
        if output_dtype != "float32":
            meta["storage_scale_max"] = np.iinfo(np.dtype(output_dtype)).max
        (self.root / "gigatime_metadata.json").write_text(json.dumps(meta))

    def validate(self, **kwargs):
        with quantify.LazyImageReader(str(self.image)) as reader:
            return quantify.validate_restart_contract(reader, "nuclei", self.masks["nuclei"], **kwargs)

    def test_roles_precision_and_checkpoint_are_explicit(self):
        self.assertEqual(self.schema["default_phenotype_excluded_channels"], ["TRITC", "Cy5"])
        self.assertEqual(self.schema["channel_roles"]["DAPI"], "nuclear_counterstain")
        self.assertEqual(self.schema["channel_roles"]["CD8"], "predicted_biological_marker")
        self.assertEqual(self.schema["reduction_precision"], "float64")
        self.assertEqual(self.schema["checkpoint_sha256"], "a"*64)
        self.assertEqual(self.validate()["status"], "equivalent_float32_restart")

    def test_missing_or_invalid_checkpoint_cannot_produce_authoritative_schema(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["model_provenance"]["checkpoints"][0]["status"] = "missing"
        with self.assertRaisesRegex(ValueError, "verified model checkpoint"):
            quantify.build_marker_schema(metadata, self.masks)

    def test_uint8_is_never_claimed_equivalent(self):
        self.write_store(np.round(self.scores*255).astype(np.uint8), output_dtype="uint8")
        with self.assertRaisesRegex(ValueError, "not unquantized float32"):
            self.validate()
        result = self.validate(allow_legacy_research=True)
        self.assertEqual(result["status"], "legacy_research_not_equivalent")
        self.assertFalse(result["equivalent_to_authoritative_integrated"])

    def test_cli_legacy_mode_is_explicit_and_status_is_carried_into_tables(self):
        self.write_store(np.round(self.scores*255).astype(np.uint8), output_dtype="uint8")
        out = self.root / "cli"
        command = [sys.executable, str(Path(quantify.__file__)), "--image", str(self.image),
            "--mask", self.masks["nuclei"], "--mask-name", "nuclei",
            "--out-quant-csv", str(out/"quant.csv"), "--out-mean-csv", str(out/"mean.csv"),
            "--out-stats-csv", str(out/"stats.csv"), "--out-summary-json", str(out/"summary.json")]
        strict = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(strict.returncode, 0)
        self.assertIn("Incompatible GigaTIME restart", strict.stderr)
        self.assertFalse(out.exists())
        legacy = subprocess.run(command + ["--allow-legacy-research"], capture_output=True, text=True)
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
        self.assertIn("authority_status=legacy_research_not_equivalent", legacy.stdout)
        with (out/"quant.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(all(row["quantification_status"] == "legacy_research_not_equivalent" for row in rows))
        self.assertEqual(json.loads((out/"summary.json").read_text())["authority_status"], "legacy_research_not_equivalent")

    def test_subset_and_permutation_fail_instead_of_changing_marker_space(self):
        names = quantify.CANONICAL_CHANNEL_NAMES
        self.write_store(self.scores[:5], names[:5])
        with self.assertRaisesRegex(ValueError, "subset/order"):
            self.validate()
        self.write_store(self.scores[::-1], names[::-1])
        with self.assertRaisesRegex(ValueError, "subset/order"):
            self.validate()

    def test_requested_checkpoint_and_schema_mismatch_even_in_legacy_mode(self):
        with self.assertRaisesRegex(ValueError, "checkpoint SHA256 differs"):
            self.validate(checkpoint_sha256="b"*64, allow_legacy_research=True)
        expected = copy.deepcopy(self.schema)
        expected["checkpoint_sha256"] = "b"*64
        expected["schema_sha256"] = quantify.marker_schema_hash(expected)
        with self.assertRaisesRegex(ValueError, "requested marker schema"):
            self.validate(expected_schema=expected)

    def test_changed_mask_or_compartment_is_incompatible(self):
        changed = np.ones(self.shape, dtype=np.uint32)
        tifffile.imwrite(self.masks["nuclei"], changed)
        with self.assertRaisesRegex(ValueError, "mask or semantics differs"):
            self.validate()

    def test_channel_sidecar_cannot_hide_reordered_image_header(self):
        self.write_store(self.scores[::-1], quantify.CANONICAL_CHANNEL_NAMES[::-1])
        (self.root / "gigatime_channels.json").write_text(json.dumps(quantify.CANONICAL_CHANNEL_NAMES))
        with self.assertRaisesRegex(ValueError, "sidecar conflicts"):
            self.validate()

    @unittest.skipUnless(TORCH_AVAILABLE, "CPU torch required for ring integrity test")
    def test_ring_mask_cannot_be_a_second_copy_of_whole_cell_mask(self):
        with self.assertRaisesRegex(ValueError, "ring mask overlaps"):
            gigatime.build_quantifiers(nuclei_mask_path=self.masks["nuclei"],
                cyto_mask_path=self.masks["cyto"], ring_mask_path=self.masks["cyto"],
                target_shape=self.shape, channel_names=quantify.CANONICAL_CHANNEL_NAMES, block_size=8)

    def test_missing_schema_and_precision_corruption_fail(self):
        self.metadata.pop("marker_schema")
        self.write_store()
        with self.assertRaisesRegex(ValueError, "missing versioned marker schema"):
            self.validate()
        self.metadata["marker_schema"] = copy.deepcopy(self.schema)
        self.metadata["marker_schema"]["prediction_precision"] = "float16"
        self.metadata["marker_schema"]["schema_sha256"] = quantify.marker_schema_hash(self.metadata["marker_schema"])
        self.write_store()
        with self.assertRaisesRegex(ValueError, "precision contract mismatch"):
            self.validate()

    @unittest.skipUnless(TORCH_AVAILABLE, "CPU torch required for integrated quantifier fixture")
    def test_integrated_and_float_tiff_restart_are_numerically_identical_for_all_compartments(self):
        quantifiers = gigatime.build_quantifiers(nuclei_mask_path=self.masks["nuclei"],
            cyto_mask_path=self.masks["cyto"], ring_mask_path=self.masks["ring"],
            target_shape=self.shape, channel_names=quantify.CANONICAL_CHANNEL_NAMES, block_size=8)
        try:
            for q in quantifiers:
                q.marker_schema = self.schema
                for y0, y1, x0, x1 in gigatime.iter_output_tiles(*self.shape, 11):
                    q.accumulate_tile(y0, y1, x0, x1, self.scores[:, y0:y1, x0:x1])
                summary = q.write_outputs(self.root / "integrated", "fixture")
                self.assertEqual(summary["objects_quantified"], 2)
                self.assertEqual(summary["authority_status"], "authoritative_full_precision_integrated")
                self.assertTrue(Path(summary["summary_json"]).exists())
                with quantify.LazyImageReader(str(self.image)) as image, quantify.LazyMaskReader(self.masks[q.mask_name]) as mask:
                    contract = quantify.validate_restart_contract(image, q.mask_name, self.masks[q.mask_name], expected_schema=self.schema, checkpoint_sha256="a"*64)
                    self.assertTrue(contract["equivalent_to_authoritative_integrated"])
                    restart, _, _ = quantify.quantify_blockwise(image, mask, image.channel_names, mask_name=q.mask_name, block_size=7)
                with Path(summary["quant_csv"]).open() as handle:
                    integrated = list(csv.DictReader(handle))
                self.assertEqual(len(integrated), len(restart))
                for before, after in zip(integrated, restart):
                    for name, value in after.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            self.assertAlmostEqual(float(before[name]), float(value), places=12, msg=f"{q.mask_name}/{name}")
                        else:
                            self.assertEqual(before[name], str(value))
        finally:
            for q in quantifiers:
                q.close()

    @unittest.skipUnless(TORCH_AVAILABLE, "CPU torch required for storage helper fixture")
    def test_float_zarr_restart_retains_all_marker_values(self):
        store = self.root / "gigatime_probs.zarr"
        root = quantify.zarr.open_group(str(store), mode="w")
        root.attrs.update(gigatime._zarr_root_attrs(self.metadata, quantify.CANONICAL_CHANNEL_NAMES))
        array = gigatime._create_zarr_dataset(root, "0", shape=self.scores.shape, dtype="float32", chunks=(23, 16, 16))
        array[:] = self.scores
        array.attrs["axes"] = "CYX"
        quantify.finalize_zarr_storage(store, self.metadata)
        with quantify.LazyImageReader(str(store)) as image:
            result = quantify.validate_restart_contract(image, "nuclei", self.masks["nuclei"], expected_schema=self.schema)
            self.assertTrue(result["equivalent_to_authoritative_integrated"])
            np.testing.assert_array_equal(image.read_block(0, self.shape[0], 0, self.shape[1]), self.scores)

    @unittest.skipUnless(TORCH_AVAILABLE, "CPU torch required for staged writer fixture")
    def test_staged_float_resume_recomputes_nonempty_quantification(self):
        out = self.root / "resumed"
        out.mkdir()
        source = self.root / "he.tif"
        tifffile.imwrite(source, np.full((*self.shape, 3), 127, dtype=np.uint8), photometric="rgb")
        buffer = out / "_gigatime_level0.cyx.bin"
        self.scores.tofile(buffer)
        expected = gigatime.make_level0_contract(self.metadata, quantify.CANONICAL_CHANNEL_NAMES, "float32", quantify.sha256_file(source), patch_size=8, stride=4, factor=1)
        (out / "_gigatime_level0.schema.json").write_text(json.dumps({"state": "complete", "contract": expected}))
        quantifiers = gigatime.build_quantifiers(nuclei_mask_path=self.masks["nuclei"], cyto_mask_path=self.masks["cyto"], ring_mask_path=self.masks["ring"], target_shape=self.shape, channel_names=quantify.CANONICAL_CHANNEL_NAMES, block_size=16)
        qc = gigatime.MarkerScoreQC(outdir=out, target_shape=self.shape, channel_names=quantify.CANONICAL_CHANNEL_NAMES, max_samples=100)
        gigatime.blockwise_write_ometiff_outputs(str(source), 0, out, None, torch.device("cpu"),
            patch_size=8, stride=4, batch_size=1, tile_size=16, compression="deflate", jpeg_quality=90,
            output_dtype="float32", predictor=False, pyramid=False, factor=1,
            metadata=self.metadata, output_channel_indices=list(range(23)), output_channel_names=quantify.CANONICAL_CHANNEL_NAMES,
            quantifiers=quantifiers, score_qc=qc, quant_dir=out / "quant", jpg_exporter=None,
            jpg_channel_indices=[], sample_id="fixture", resume_level0_buffer=True,
            skip_background_blocks=False, skip_background_mask_path="", skip_background_downsample=8,
            skip_background_min_fraction=0, skip_background_close_radius=0,
            skip_background_min_obj_area=0, skip_background_hole_area=0)
        np.testing.assert_array_equal(tifffile.imread(out / "gigatime_probs.ome.tif"), self.scores)
        for name in ("nuclei", "cyto", "ring"):
            with (out / "quant" / f"fixture_{name}_gigatime_quantification.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
        self.assertFalse(buffer.exists())

    @unittest.skipUnless(TORCH_AVAILABLE, "CPU torch required for buffer contract fixture")
    def test_size_only_incomplete_or_changed_precision_buffer_is_rejected(self):
        path = self.root / "buffer.bin"
        path.write_bytes(b"1234")
        manifest = self.root / "manifest.json"
        expected = gigatime.make_level0_contract(self.metadata, quantify.CANONICAL_CHANNEL_NAMES, "float32", "c"*64, patch_size=8, stride=4, factor=1)
        with self.assertRaisesRegex(ValueError, "without a complete schema"):
            gigatime.validate_level0_resume(path, manifest, expected, 4)
        manifest.write_text(json.dumps({"state": "in_progress", "contract": expected}))
        with self.assertRaisesRegex(ValueError, "incomplete buffer"):
            gigatime.validate_level0_resume(path, manifest, expected, 4)
        manifest.write_text(json.dumps({"state": "complete", "contract": expected}))
        changed = {**expected, "dtype": "uint8"}
        with self.assertRaisesRegex(ValueError, "changed marker"):
            gigatime.validate_level0_resume(path, manifest, changed, 4)
        manifest.write_text(json.dumps({"state": "complete", "contract": changed}))
        with self.assertRaisesRegex(ValueError, "all 23 float32"):
            gigatime.validate_level0_resume(path, manifest, changed, 4)


if __name__ == "__main__":
    unittest.main()
