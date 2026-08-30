import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class ProcessCodeFingerprintTest(unittest.TestCase):
    def assert_fingerprint(self, module: str, required_files: list[str]):
        text = (ROOT / module).read_text()
        self.assertIn("MessageDigest.getInstance('SHA-256')", text)
        self.assertIn("codeFingerprint", text)
        for required in required_files:
            self.assertIn(required, text)

    def test_consensus_fingerprint_tracks_python_implementation(self):
        self.assert_fingerprint(
            "modules/build_cell_consensus.nf",
            ["scriptPath", "Cell consensus code fingerprint"],
        )

    def test_both_uni2_processes_track_grid_helper(self):
        for module in (
            "modules/extract_uni2_embeddings.nf",
            "modules/extract_uni2_embeddings_shared.nf",
        ):
            self.assert_fingerprint(module, ["uni2_script", "bin/uni2_grid.py"])

    def test_gigatime_tracks_hardware_and_resolution_helpers(self):
        self.assert_fingerprint(
            "modules/run_gigatime_on_crop.nf",
            ["gigatime_script", "bin/gigatime_hardware.py", "bin/gigatime_resolution.py"],
        )

    def test_post_cluster_image_writers_track_metadata_helpers(self):
        expectations = {
            "modules/grow_to_tissue.nf": ["grow_script", "bin/ome_tiff_metadata.py"],
            "modules/refine_grown_tissue_medsam.nf": [
                "refine_script", "bin/medsam_border_refine.py", "bin/ome_tiff_metadata.py",
            ],
            "modules/select_neoplastic_section.nf": [
                "scriptPath", "bin/grow_to_tissue.py", "bin/ome_tiff_metadata.py",
            ],
        }
        for module, files in expectations.items():
            self.assert_fingerprint(module, files)

    def test_titan_tracks_its_image_loader(self):
        self.assert_fingerprint(
            "modules/extract_titan_section_embedding.nf",
            ["scriptPath"],
        )


if __name__ == "__main__":
    unittest.main()
