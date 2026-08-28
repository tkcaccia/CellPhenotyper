import importlib.util
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


if __name__ == "__main__":
    unittest.main()
