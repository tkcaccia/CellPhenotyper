import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "bin" / "extract_titan_section_embedding.py"
SPEC = importlib.util.spec_from_file_location("extract_titan_section_embedding", MODULE_PATH)
titan = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = titan
SPEC.loader.exec_module(titan)


class TitanEmbeddingContractTest(unittest.TestCase):
    def test_source_patch_size_preserves_physical_512_at_20x(self):
        self.assertEqual(titan.source_patch_size(0.5, 0.5, 512), 512)
        self.assertEqual(titan.source_patch_size(0.547304, 0.5, 512), 468)

    def test_pathofmpred_feature_schema_is_exact(self):
        columns = titan.feature_columns()
        self.assertEqual(len(columns), 768)
        self.assertEqual(columns[0], "titan_000")
        self.assertEqual(columns[-1], "titan_767")

    def test_embedding_csv_rejects_wrong_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            with self.assertRaisesRegex(RuntimeError, "768"):
                titan.write_embedding_csv(path, "sample", "section", np.zeros(767))

    def test_embedding_csv_writes_named_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.csv"
            titan.write_embedding_csv(path, "sample", "section", np.arange(768))
            lines = path.read_text().splitlines()
        self.assertTrue(lines[0].startswith("sample_id,section_id,titan_000"))
        self.assertTrue(lines[0].endswith("titan_767"))

    def test_local_resolver_returns_snapshot_checkpoint(self):
        try:
            import huggingface_hub
        except ImportError:
            self.skipTest("huggingface_hub is not installed locally")
        original = huggingface_hub.hf_hub_download
        try:
            with tempfile.TemporaryDirectory() as directory:
                snapshot = Path(directory)
                checkpoint = snapshot / "conch_v1_5_pytorch_model.bin"
                checkpoint.write_bytes(b"test")
                self.assertEqual(
                    titan.install_local_titan_file_resolver(directory),
                    snapshot.resolve(),
                )
                self.assertEqual(
                    huggingface_hub.hf_hub_download(
                        "MahmoodLab/TITAN", filename=checkpoint.name,
                    ),
                    str(checkpoint.resolve()),
                )
        finally:
            huggingface_hub.hf_hub_download = original

    def test_missing_absolute_snapshot_is_rejected_before_hub_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-titan-snapshot"
            with self.assertRaisesRegex(FileNotFoundError, "not accessible"):
                titan.install_local_titan_file_resolver(str(missing))

    def test_local_resolver_falls_back_when_transformers_cache_is_read_only(self):
        try:
            from transformers import dynamic_module_utils
        except ImportError:
            self.skipTest("transformers is not installed locally")
        original_cache = dynamic_module_utils.HF_MODULES_CACHE
        original_fallback = os.environ.get("CELLPHENOTYPER_HF_MODULES_CACHE")
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                snapshot = root / "snapshot"
                snapshot.mkdir()
                (snapshot / "configuration_titan.py").write_text("# test\n")
                blocker = root / "not-a-directory"
                blocker.write_text("blocked")
                fallback = root / "fallback"
                dynamic_module_utils.HF_MODULES_CACHE = str(blocker)
                os.environ["CELLPHENOTYPER_HF_MODULES_CACHE"] = str(fallback)
                self.assertEqual(
                    titan.install_local_titan_file_resolver(str(snapshot)),
                    snapshot.resolve(),
                )
                self.assertTrue(
                    (fallback / "transformers_modules" / "snapshot" / "configuration_titan.py").is_file()
                )
        finally:
            dynamic_module_utils.HF_MODULES_CACHE = original_cache
            if original_fallback is None:
                os.environ.pop("CELLPHENOTYPER_HF_MODULES_CACHE", None)
            else:
                os.environ["CELLPHENOTYPER_HF_MODULES_CACHE"] = original_fallback

    def test_singularity_binds_absolute_titan_model(self):
        root = Path(__file__).parents[1]
        config = (root / "nextflow.config").read_text()
        bind_block = config.split("def singularity_bind_targets = [", 1)[1].split("]", 1)[0]
        self.assertNotIn("titan_model", bind_block)
        module = (root / "modules" / "extract_titan_section_embedding.nf").read_text()
        self.assertIn("containerOptions", module)
        self.assertIn("workflow.containerEngine == 'singularity'", module)
        self.assertIn('return "-B ${modelPath}:${modelPath}:ro"', module)

    def test_ome_rgb_loader_is_used(self):
        source = MODULE_PATH.read_text()
        self.assertIn("def open_vips_rgb", source)
        self.assertIn("image = open_vips_rgb(args.image)", source)
        self.assertIn("TITAN requires an RGB input image", source)


if __name__ == "__main__":
    unittest.main()
