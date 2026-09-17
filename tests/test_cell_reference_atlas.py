import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

BIN = Path(__file__).parents[1] / "bin"
sys.path.insert(0, str(BIN))
import cell_reference_atlas as atlas


MARKER = "predicted__nucleus__CD3__mean"
DEFINITION = {"model": "UNI2", "revision": "immutable-test-revision", "context_um": 56,
              "pooling": "test_pooling", "preprocessing": "fixed_test_transform"}
MARKER_DEFINITION = {"schema_version": "cellphenotyper.gigatime.v1", "marker_names": ["CD3", "CD8"],
    "default_phenotype_excluded_channels": ["TRITC", "Cy5"], "checkpoint_sha256": "b" * 64,
    "prediction_precision": "float32", "reduction_precision": "float64", "model_arithmetic": "float32",
    "prediction_settings": {"patch_size": 256, "stride": 128}, "effective_mpp": .25,
    "compartment": "nucleus", "compartment_semantics": "canonical_nuclear_mask",
    "value_semantics": "uncalibrated_virtual_marker_score"}


def make_profile(root, sample, values, labels=None, unit="cell", markers=None,
                 definition=None, extra_block=None, label_column="tissue_domain", generic=False,
                 marker_definition=None, marker_verified=True, include_marker_block=True):
    root = Path(root)
    root.mkdir()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    n = len(values)
    if generic:
        stem, uid, key = "observation_profiles", "observation_uid", "observation_id"
    elif unit == "cell":
        stem, uid, key = "cell_profiles", "cell_uid", "cell_id"
    else:
        stem, uid, key = "region_profiles", "region_uid", "region_id"
    table = pd.DataFrame({uid: [f"{sample}:{i:03d}" for i in range(n)], "sample_id": sample,
        key: [f"{i:03d}" for i in range(n)], label_column: labels if labels is not None else ["1"] * n,
        "phenotype": ["predicted_phenotype"] * n, MARKER: markers if markers is not None else [0.1] * n})
    table.to_csv(root / f"{stem}.csv", index=False)
    table[[uid, "sample_id", key]].to_csv(root / "feature_rows.csv", index=False)
    np.save(root / "context.npy", values)
    manifest = {"schema_version": "1.0.0", "observation_unit": unit, "observation_count": n,
        "biological_marker_features": [MARKER], "feature_blocks": {
            "context": {"path": "context.npy", "shape": list(values.shape),
                "feature_names": [f"feature_{i}" for i in range(values.shape[1])],
                "feature_definition": DEFINITION if definition is None else definition}}}
    if extra_block is not None:
        block = np.asarray(extra_block, dtype=np.float32)
        if block.ndim == 1:
            block = block[:, None]
        np.save(root / "local.npy", block)
        manifest["feature_blocks"]["local"] = {"path": "local.npy", "shape": list(block.shape),
            "feature_definition": {**DEFINITION, "context_um": 22.5}}
    if include_marker_block:
        from build_cell_profiles import export_table_block
        arrays = root / "feature_blocks"
        arrays.mkdir()
        manifest["feature_blocks"]["markers_nucleus"] = export_table_block(table, [MARKER], "markers_nucleus", arrays,
            MARKER_DEFINITION if marker_definition is None else marker_definition, compatible=marker_verified)
    (root / f"{stem}_manifest.json").write_text(json.dumps(manifest))
    return root, table


class ReferenceAtlasTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def reference(self, unit="cell"):
        a, _ = make_profile(self.root / "a", "A", [-0.2, 0, 0.2], unit=unit)
        b, _ = make_profile(self.root / "b", "B", [4.8, 5, 5.2], unit=unit)
        destination = self.root / "atlas_v1"
        manifest = atlas.build_atlas([a, b], destination, "v1", ["context"])
        return a, b, destination, manifest

    def test_arbitrary_slide_cluster_labels_are_never_merged(self):
        a, b, destination, manifest = self.reference()
        self.assertEqual([r["reference_group"] for r in manifest["groups"]], ["A:tissue_domain:1", "B:tissue_domain:1"])
        self.assertEqual(manifest["observation_unit"], "cell")
        refs = pd.read_csv(destination / "reference_observations.csv", dtype=str)
        self.assertEqual(set(refs.observation_uid), {f"{s}:{i:03d}" for s in ("A", "B") for i in range(3)})
        self.assertEqual(refs.observation_id.iloc[0], "000")
        self.assertTrue(all(r["assignment_supported"] for r in manifest["groups"]))
        with self.assertRaisesRegex(ValueError, "sample-scoped"):
            atlas.build_atlas([a, b], self.root / "shared", "v2", ["context"], label_scope="shared")

    def test_reference_build_is_immutable_and_source_order_is_stable(self):
        a, b, destination, manifest = self.reference()
        with self.assertRaises(FileExistsError):
            atlas.build_atlas([a], destination, "v2", ["context"])
        other = atlas.build_atlas([b, a], self.root / "atlas_reordered", "v1", ["context"])
        self.assertEqual(manifest["atlas_id"], other["atlas_id"])
        self.assertEqual(manifest["groups"], other["groups"])
        self.assertEqual(manifest["files"], other["files"])

    def test_mapping_retains_every_row_and_original_discovery_label(self):
        _, _, destination, _ = self.reference()
        query, original = make_profile(self.root / "query", "Q", [0, 5, 100, np.nan], labels=["99", "77", "99", "77"])
        result = atlas.map_profile(query, destination)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.tissue_domain.astype(str).tolist(), original.tissue_domain.tolist())
        self.assertEqual(result.reference_assignment.tolist(), ["A:tissue_domain:1", "B:tissue_domain:1", "unknown", "unknown"])
        self.assertEqual(result.reference_status.tolist(), ["assigned", "assigned", "outside_reference", "missing_features"])
        self.assertEqual(result.cell_id.tolist(), ["000", "001", "002", "003"])
        self.assertFalse(any("confidence" in c or "probability" in c for c in result))

    def test_same_width_with_different_definition_is_incompatible(self):
        _, _, destination, _ = self.reference()
        query, _ = make_profile(self.root / "different", "Q", [1, 2], definition={**DEFINITION, "revision": "other-model"})
        with self.assertRaisesRegex(ValueError, "Incompatible feature"):
            atlas.map_profile(query, destination)

    def test_dimension_missing_block_and_missing_definition_are_rejected(self):
        _, _, destination, _ = self.reference()
        query, _ = make_profile(self.root / "wide", "Q", [[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, "Incompatible feature"):
            atlas.map_profile(query, destination)
        with self.assertRaisesRegex(ValueError, "missing feature block"):
            atlas.load_profile(query, ["absent"])
        empty, _ = make_profile(self.root / "empty_schema", "E", [1, 2, 3], definition={})
        with self.assertRaisesRegex(ValueError, "lacks a feature_definition"):
            atlas.build_atlas([empty], self.root / "bad", "v1", ["context"])

    def test_unverified_reference_compatibility_is_rejected(self):
        query, _ = make_profile(self.root / "unverified", "Q", [1, 2, 3])
        path = query / "cell_profiles_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["feature_blocks"]["context"]["reference_compatible"] = False
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "not reference-compatible"):
            atlas.build_atlas([query], self.root / "bad", "v1", ["context"])

    def test_singletons_stay_unmatched_even_at_the_reference_mean(self):
        reference, _ = make_profile(self.root / "one", "A", [5])
        destination = self.root / "singleton"
        manifest = atlas.build_atlas([reference], destination, "v1", ["context"])
        self.assertIsNone(manifest["groups"][0]["acceptance_radius"])
        query, _ = make_profile(self.root / "query", "Q", [5])
        result = atlas.map_profile(query, destination)
        self.assertEqual(result.reference_assignment.iloc[0], "unknown")
        self.assertEqual(result.reference_status.iloc[0], "insufficient_reference")

    def test_tied_reference_prototypes_are_explicitly_ambiguous(self):
        a, _ = make_profile(self.root / "a", "A", [-1, 0, 1])
        b, _ = make_profile(self.root / "b", "B", [-1, 0, 1])
        destination = self.root / "atlas"
        atlas.build_atlas([a, b], destination, "v1", ["context"])
        query, _ = make_profile(self.root / "q", "Q", [0])
        self.assertEqual(atlas.map_profile(query, destination).reference_status.iloc[0], "ambiguous_reference")

    def test_per_group_distributions_and_real_representatives(self):
        _, _, destination, manifest = self.reference()
        quantiles = np.load(destination / "prototype_quantiles.npy", allow_pickle=False)
        self.assertEqual(quantiles.shape, (2, 3, 1))
        self.assertEqual(manifest["groups"][0]["representative_observation_uids"][0], "A:001")
        self.assertIn("hashes", manifest["source_profiles"][0])
        with np.load(destination / "feature_distributions.npz", allow_pickle=False) as distributions:
            self.assertEqual(distributions["p05_p50_p95_0"].shape, (3, 1))

    def test_reviewed_and_predicted_descriptions_are_stored_separately(self):
        a, _ = make_profile(self.root / "a", "A", [-1, 0, 1], labels=["stromal"] * 3, label_column="reviewed_label")
        b, _ = make_profile(self.root / "b", "B", [-1, 0, 1], labels=["stromal"] * 3, label_column="reviewed_label")
        descriptions = {"reviewed": {"reviewed_label:stromal": "Reviewed stromal morphology"},
                        "predicted": {"reviewed_label:stromal": "Model connective-like pattern"}}
        result = atlas.build_atlas([a, b], self.root / "atlas", "v1", ["context"], label_column="reviewed_label", descriptions=descriptions)
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(result["groups"][0]["reviewed_description"], "Reviewed stromal morphology")
        self.assertEqual(result["groups"][0]["predicted_description"], "Model connective-like pattern")

    def test_tissue_region_profiles_map_and_retrieve_region_identities(self):
        _, _, destination, manifest = self.reference(unit="tissue_region")
        self.assertEqual(manifest["observation_unit"], "tissue_region")
        query, _ = make_profile(self.root / "query_region", "Q", [0, 80], unit="tissue_region", generic=True)
        result = atlas.map_profile(query, destination)
        self.assertEqual(result.reference_status.tolist(), ["assigned", "outside_reference"])
        neighbours = atlas.search_similar(query, destination, ["Q:000"], ["context"], k=2)
        self.assertEqual(neighbours.reference_observation_uid.iloc[0], "A:001")
        self.assertTrue(neighbours.observation_unit.eq("tissue_region").all())
        self.assertNotIn("reference_cell_uid", neighbours)

    def test_observation_unit_mismatch_cannot_cross_cell_and_region(self):
        a, _, destination, _ = self.reference()
        region, _ = make_profile(self.root / "region", "R", [0], unit="tissue_region")
        with self.assertRaisesRegex(ValueError, "Incompatible feature"):
            atlas.map_profile(region, destination)
        with self.assertRaisesRegex(ValueError, "Incompatible feature"):
            atlas.build_atlas([a, region], self.root / "mixed", "v1", ["context"])

    def test_missing_region_observation_unit_is_rejected(self):
        region, _ = make_profile(self.root / "region", "R", [0, 1, 2], unit="tissue_region")
        path = region / "region_profiles_manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["observation_unit"]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "observation_unit"):
            atlas.load_profile(region, ["context"])

    def test_similarity_ties_are_deterministic_and_discordance_filters(self):
        reference, _ = make_profile(self.root / "ref", "A", [0, 0, 1], markers=[0.1, 0.9, 0.8])
        destination = self.root / "atlas"
        atlas.build_atlas([reference], destination, "v1", ["context"])
        query, _ = make_profile(self.root / "query", "Q", [0], markers=[0.1])
        plain = atlas.search_similar(query, destination, ["Q:000"], ["context"], k=2)
        self.assertEqual(plain.reference_observation_uid.tolist(), ["A:000", "A:001"])
        discordant = atlas.search_similar(query, destination, ["Q:000"], ["context"], k=2,
            marker_column=MARKER, min_marker_difference=0.5)
        self.assertEqual(discordant.reference_observation_uid.tolist(), ["A:001", "A:002"])
        self.assertTrue((discordant.absolute_marker_difference >= 0.5).all())

    def test_unverified_reference_marker_disables_only_requested_discordance(self):
        for index, options in enumerate(({"include_marker_block": False}, {"marker_verified": False},
                {"marker_definition": {**MARKER_DEFINITION, "prediction_precision": "uint8"}})):
            with self.subTest(options=options):
                reference, _ = make_profile(self.root / f"ref_{index}", "A", [0, 0, 1], markers=[.1, .9, .8], **options)
                destination = self.root / f"atlas_{index}"
                manifest = atlas.build_atlas([reference], destination, "v1", ["context"])
                self.assertEqual(manifest["marker_filter_contracts"][MARKER]["status"], "unavailable")
                query, _ = make_profile(self.root / f"query_{index}", "Q", [0], markers=[.1])
                self.assertEqual(len(atlas.search_similar(query, destination, ["Q:000"], ["context"], k=2)), 2)
                self.assertEqual(len(atlas.map_profile(query, destination)), 1)
                with self.assertRaisesRegex(ValueError, "Marker discordance unavailable"):
                    atlas.search_similar(query, destination, ["Q:000"], ["context"], marker_column=MARKER, min_marker_difference=.5)

    def test_mixed_reference_marker_definitions_do_not_create_score_compatibility(self):
        a, _ = make_profile(self.root / "a", "A", [0, 0, 1])
        b, _ = make_profile(self.root / "b", "B", [0, 0, 1],
            marker_definition={**MARKER_DEFINITION, "checkpoint_sha256": "c" * 64})
        destination = self.root / "atlas"
        manifest = atlas.build_atlas([a, b], destination, "v1", ["context"])
        self.assertEqual(manifest["marker_filter_contracts"][MARKER]["status"], "incompatible")
        self.assertTrue(len(atlas.search_similar(a, destination, ["A:000"], ["context"])))
        with self.assertRaisesRegex(ValueError, "reference_marker_definitions_differ"):
            atlas.search_similar(a, destination, ["A:000"], ["context"], marker_column=MARKER, min_marker_difference=.1)

    def test_query_marker_model_compartment_precision_and_settings_are_checked_outside_distance(self):
        _, _, destination, _ = self.reference()
        changes = [{"checkpoint_sha256": "c" * 64}, {"model_arithmetic": "bfloat16"},
            {"prediction_precision": "uint8"}, {"reduction_precision": "float32"},
            {"compartment": "perinuclear_ring"}, {"compartment_semantics": "other_compartment"},
            {"prediction_settings": {"patch_size": 512, "stride": 256}}, {"effective_mpp": .5}]
        for index, change in enumerate(changes):
            with self.subTest(change=change):
                query, _ = make_profile(self.root / f"query_{index}", "Q", [0],
                    marker_definition={**MARKER_DEFINITION, **change})
                self.assertEqual(len(atlas.search_similar(query, destination, ["Q:000"], ["context"], k=1)), 1)
                with self.assertRaisesRegex(ValueError, "Marker discordance"):
                    atlas.search_similar(query, destination, ["Q:000"], ["context"], marker_column=MARKER, min_marker_difference=.1)

    def test_query_marker_block_payload_and_reference_flag_are_checked(self):
        _, _, destination, _ = self.reference()
        for index, fault in enumerate(("missing", "unverified", "hash", "values", "dtype")):
            with self.subTest(fault=fault):
                query, _ = make_profile(self.root / f"query_{index}", "Q", [0])
                path = query / "cell_profiles_manifest.json"
                manifest = json.loads(path.read_text())
                block = manifest["feature_blocks"]["markers_nucleus"]
                if fault == "missing":
                    del manifest["feature_blocks"]["markers_nucleus"]
                elif fault == "unverified":
                    block["reference_compatible"] = False
                elif fault == "hash":
                    block["sha256"] = "0" * 64
                else:
                    values = np.array([[.9]], dtype=np.float32) if fault == "values" else np.array([[1]], dtype=np.uint8)
                    np.save(query / block["path"], values)
                    block["sha256"] = atlas.digest(query / block["path"])
                path.write_text(json.dumps(manifest))
                self.assertEqual(len(atlas.search_similar(query, destination, ["Q:000"], ["context"], k=1)), 1)
                with self.assertRaisesRegex(ValueError, "Marker discordance unavailable"):
                    atlas.search_similar(query, destination, ["Q:000"], ["context"], marker_column=MARKER, min_marker_difference=.1)

    def test_missing_unrelated_marker_block_and_values_do_not_block_verified_filter(self):
        other = "predicted__nucleus__CD8__mean"
        unverified = "predicted__perinuclear_ring__CD8__mean"
        reference, table = make_profile(self.root / "ref", "A", [0, 0, 1], markers=[.1, .9, .8])
        path = reference / "cell_profiles_manifest.json"
        manifest = json.loads(path.read_text())
        table[other] = np.nan
        table[unverified] = np.nan
        table.to_csv(reference / "cell_profiles.csv", index=False)
        manifest["biological_marker_features"].extend([other, unverified])
        block = manifest["feature_blocks"]["markers_nucleus"]
        values = np.column_stack([table[MARKER], table[other]]).astype(np.float32)
        np.save(reference / block["path"], values)
        block.update({"feature_names": [MARKER, other], "shape": list(values.shape), "sha256": atlas.digest(reference / block["path"])})
        path.write_text(json.dumps(manifest))
        destination = self.root / "atlas"
        result = atlas.build_atlas([reference], destination, "v1", ["context"])
        self.assertEqual(result["marker_filter_contracts"][MARKER]["status"], "compatible")
        self.assertEqual(result["marker_filter_contracts"][unverified]["status"], "unavailable")
        query, _ = make_profile(self.root / "query", "Q", [0], markers=[.1])
        hits = atlas.search_similar(query, destination, ["Q:000"], ["context"], marker_column=MARKER, min_marker_difference=.5)
        self.assertEqual(hits.reference_observation_uid.tolist(), ["A:001", "A:002"])

    def test_legacy_atlas_keeps_morphology_search_but_cannot_claim_marker_compatibility(self):
        a, _, destination, _ = self.reference()
        path = destination / "atlas_manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["marker_filter_contracts"]
        atlas.write_json(path, manifest)
        (destination / "atlas_manifest.sha256").write_text(atlas.digest(path) + "\n")
        self.assertTrue(len(atlas.search_similar(a, destination, ["A:000"], ["context"])))
        with self.assertRaisesRegex(ValueError, "legacy_atlas_without_marker_definition"):
            atlas.search_similar(a, destination, ["A:000"], ["context"], marker_column=MARKER, min_marker_difference=.1)

    def test_discordance_retains_raw_marker_precision_not_float32_projection(self):
        values = [.1000000001, .1000000002, .1000000003]
        reference, _ = make_profile(self.root / "ref", "A", [0, 0, 1], markers=values)
        destination = self.root / "atlas"
        atlas.build_atlas([reference], destination, "v1", ["context"])
        query, _ = make_profile(self.root / "query", "Q", [0], markers=[.1])
        hits = atlas.search_similar(query, destination, ["Q:000"], ["context"], marker_column=MARKER, min_marker_difference=1e-11)
        self.assertEqual(hits.reference_marker_score.tolist(), values)
        self.assertEqual(hits.absolute_marker_difference.tolist(), [value - .1 for value in values])

    def test_filter_accepts_the_actual_canonical_marker_producer_contract(self):
        from test_cell_profiles import versioned_markers, write_marker_summary, options
        from build_cell_profiles import build_profiles
        directory, schema = versioned_markers(self.root)
        write_marker_summary(directory, schema)
        _, manifest = build_profiles(options(self.root, marker_quant_dir=directory))
        profile = atlas.load_profile(self.root / "profiles", ["markers_nucleus"])
        column = "predicted__nucleus__CD8__mean"
        definition = atlas.marker_filter_definition(profile, column)
        self.assertEqual(definition["feature_definition"], manifest["feature_blocks"]["markers_nucleus"]["feature_definition"])
        self.assertEqual(definition["block_precision"], "float32")
        self.assertEqual(profile.hashes["marker_filter_blocks"]["markers_nucleus"], manifest["feature_blocks"]["markers_nucleus"]["sha256"])

    def test_search_missing_features_and_markers_are_explicit(self):
        _, _, destination, _ = self.reference()
        query, _ = make_profile(self.root / "query", "Q", [np.nan, 1], markers=[0.1, np.nan])
        result = atlas.search_similar(query, destination, ["Q:000", "Q:001"], ["context"],
            marker_column=MARKER, min_marker_difference=0.3)
        self.assertEqual(result.search_status.tolist(), ["missing_features", "missing_marker"])
        self.assertEqual(result.columns.tolist(), atlas.SEARCH_COLUMNS)
        self.assertTrue(result.reference_observation_uid.isna().all())

    def test_search_excludes_self_and_optionally_same_sample(self):
        a, _, destination, _ = self.reference()
        result = atlas.search_similar(a, destination, ["A:001"], ["context"], k=10)
        self.assertNotIn("A:001", result.reference_observation_uid.tolist())
        result = atlas.search_similar(a, destination, ["A:001"], ["context"], exclude_same_sample=True)
        self.assertTrue(result.reference_sample_id.eq("B").all())

    def test_feature_groups_are_balanced_independently_and_search_can_select_subset(self):
        a, _ = make_profile(self.root / "a", "A", [-1, 0, 1], extra_block=[[-1] * 8, [0] * 8, [1] * 8])
        destination = self.root / "atlas"
        atlas.build_atlas([a], destination, "v1", ["context", "local"])
        features = np.load(destination / "reference_features.npy")
        np.testing.assert_allclose(np.square(features[:, :1]).sum(), np.square(features[:, 1:]).sum(), rtol=1e-6)
        query, _ = make_profile(self.root / "q", "Q", [0])
        neighbours = atlas.search_similar(query, destination, ["Q:000"], ["context"], k=1)
        self.assertEqual(neighbours.reference_observation_uid.iloc[0], "A:001")
        with self.assertRaisesRegex(ValueError, "missing feature block local"):
            atlas.map_profile(query, destination)

    def test_row_alignment_and_duplicate_ids_fail(self):
        a, _ = make_profile(self.root / "a", "A", [1, 2, 3])
        path = a / "feature_rows.csv"
        rows = pd.read_csv(path, dtype=str).iloc[::-1]
        rows.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Feature-row identities"):
            atlas.load_profile(a, ["context"])
        b, _ = make_profile(self.root / "b", "A", [1, 2, 3])
        c, _ = make_profile(self.root / "c", "A", [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "Duplicate observation"):
            atlas.build_atlas([b, c], self.root / "bad", "v1", ["context"])

    def test_tampered_atlas_artifacts_are_rejected(self):
        _, _, destination, _ = self.reference()
        with (destination / "reference_observations.csv").open("a") as handle:
            handle.write("tampered\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            atlas.load_atlas(destination)

    def test_cli_build_map_search_and_nonoverwrite(self):
        reference, _ = make_profile(self.root / "ref", "A", [-1, 0, 1])
        query, _ = make_profile(self.root / "q", "Q", [0, 50])
        destination = self.root / "atlas"
        command = [sys.executable, str(BIN / "cell_reference_atlas.py")]
        commands = [
            ["build", "--profiles", str(reference), "--outdir", str(destination), "--version", "v1", "--feature-groups", "context"],
            ["map", "--query", str(query), "--atlas", str(destination), "--output", str(self.root / "mapping.csv")],
            ["search", "--query", str(query), "--atlas", str(destination), "--output", str(self.root / "search.csv"),
             "--cell-uid", "Q:000", "--feature-groups", "context", "--k", "2"],
        ]
        for args in commands:
            result = subprocess.run(command + args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(pd.read_csv(self.root / "mapping.csv")), 2)
        self.assertEqual(len(pd.read_csv(self.root / "search.csv")), 2)
        repeated = subprocess.run(command + commands[1], capture_output=True, text=True)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("FileExistsError", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
