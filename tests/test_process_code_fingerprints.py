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
            self.assert_fingerprint(module, ["uni2_script", "bin/uni2_grid.py", "bin/uni2_embedding_io.py"])

    def test_binary_embedding_consumers_track_format_helpers(self):
        for module in ("build_spatial_cell_profiles", "discover_tissue_hierarchy", "prepare_hierarchy_features"):
            self.assertIn("uni2_embedding_io.py", (ROOT / "modules" / f"{module}.nf").read_text())
        self.assertIn("uni2_embedding_io.R", (ROOT / "modules/run_kodama_analysis.nf").read_text())

    def test_reference_mapping_producers_and_export_track_shared_contract(self):
        for name in ("map_cell_reference_atlas", "map_region_reference_atlas", "export_spatialdata"):
            self.assertIn("reference_mapping_io.py", (ROOT / "modules" / f"{name}.nf").read_text())

    def test_cohort_export_tracks_portable_consumer_contract(self):
        self.assertIn("cohort_niche_io.py", (ROOT / "modules/export_spatialdata.nf").read_text())
        self.assertIn("cohort_niche_io.py", (ROOT / "lib/PipelineHelpers.groovy").read_text())

    def test_uni2_grid_processes_track_geometry_and_mask_helpers(self):
        self.assert_fingerprint(
            "modules/build_uni2_spatial_grid.nf",
            ["scriptPath", "bin/uni2_grid.py", "bin/ome_tiff_metadata.py"],
        )
        self.assert_fingerprint(
            "modules/grid_clusters_to_mask.nf",
            [
                "scriptPath",
                "bin/labels_to_cluster_mask.py",
                "bin/tiff_preview.py",
                "bin/ome_tiff_metadata.py",
            ],
        )

    def test_gigatime_tracks_hardware_and_resolution_helpers(self):
        self.assert_fingerprint(
            "modules/run_gigatime_on_crop.nf",
            [
                "gigatime_script",
                "bin/gigatime_hardware.py",
                "bin/gigatime_resolution.py",
                "bin/gigatime_seam_qc.py",
            ],
        )

    def test_post_cluster_image_writers_track_metadata_helpers(self):
        expectations = {
            "modules/grow_to_tissue.nf": [
                "grow_script",
                "bin/tiff_preview.py",
                "bin/ome_tiff_metadata.py",
            ],
            "modules/refine_grown_tissue_medsam.nf": [
                "refine_script", "bin/medsam_border_refine.py", "bin/annealed_wand_boundary.py", "bin/ome_tiff_metadata.py",
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

    def test_cluster_interpretation_tracks_assessment_script(self):
        self.assert_fingerprint(
            "modules/assess_cluster_interpretation.nf",
            ["scriptPath", "Cluster interpretation assessment code fingerprint"],
        )

    def test_new_detector_and_gigatime_paths_invalidate_resume_cache(self):
        expectations = {
            "modules/prepare_input_ometiff.nf": [
                "resolution_validator_script",
                "generic_converter_script",
                "btf_converter_script",
            ],
            "modules/prepare_roi_geojson.nf": ["roi_script", "roi_validator_script"],
            "modules/run_grandqc_artifact_analysis.nf": ["grandqc_script"],
            "modules/prepare_analysis_crop.nf": ["prepare_crop_script"],
            "modules/crop_grandqc_clean_mask.nf": ["grandqc_crop_mask_script"],
            "modules/run_stardist_roi_segmentation.nf": ["stardist_script", "bin/grandqc_mask.py"],
            "modules/run_hovernet_monusac.nf": ["hovernet_script", "bin/grandqc_mask.py"],
            "modules/run_cellvitpp.nf": ["cellvit_script", "bin/grandqc_mask.py"],
            "modules/run_gigatime_kodama.nf": ["gigatime_kodama_loader_script", "r_script"],
        }
        for module, tracked in expectations.items():
            text = (ROOT / module).read_text()
            self.assertIn("PipelineHelpers.codeFingerprint", text)
            for item in tracked:
                self.assertIn(item, text)


if __name__ == "__main__":
    unittest.main()
