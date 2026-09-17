import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

BIN = Path(__file__).parents[1] / "bin"
sys.path.insert(0, str(BIN))
import cell_reference_atlas as atlas
import reference_mapping_io as mapping
from tests.test_cell_reference_atlas import MARKER, make_profile


class ReferenceMappingContractTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def bundle(self, unit="cell", generic=False, reference_values=None, query_values=None):
        reference_values = [-.2, 0, .2] if reference_values is None else reference_values
        query_values = [0, 100, np.nan] if query_values is None else query_values
        reference, _ = make_profile(self.root / "reference", "R", reference_values, unit=unit)
        query, original = make_profile(self.root / "query", "Q", query_values, unit=unit, generic=generic)
        destination = self.root / "atlas"
        frozen = atlas.build_atlas([reference], destination, "frozen-test-v1", ["context"])
        output = self.root / "mapping" / "reference_assignments.csv"
        result = mapping.write_reference_mapping(query, destination, output, atlas.map_profile)
        return query, destination, output, result, original, frozen

    def record(self, output):
        path = output.with_suffix(".mapping.json")
        return path, json.loads(path.read_text())

    def replace_assignments(self, output, mutate):
        frame = pd.read_csv(output, dtype=str, keep_default_na=False)
        mutate(frame)
        frame.to_csv(output, index=False)
        path, receipt = self.record(output)
        receipt["assignments"]["sha256"] = mapping._sha256(output)
        path.write_text(json.dumps(receipt))

    def test_complete_cell_receipt_preserves_unknown_and_projects_only_interpretation(self):
        query, destination, output, result, original, frozen = self.bundle()
        receipt, expected = self.record(output)
        frame, record, paths = mapping.load_reference_mapping(receipt, query, expected_unit="cell")
        self.assertEqual(record, expected)
        self.assertEqual(set(paths), {"receipt", "assignments", "atlas_manifest"})
        self.assertEqual(paths["atlas_manifest"].read_bytes(), (destination / "atlas_manifest.json").read_bytes())
        self.assertEqual(record["format"], "cellphenotyper_reference_mapping")
        self.assertEqual(record["schema_version"], "1.0.0")
        self.assertEqual(record["atlas"]["atlas_id"], frozen["atlas_id"])
        self.assertEqual(record["atlas"]["version"], "frozen-test-v1")
        self.assertEqual(record["atlas"]["feature_groups"], ["context"])
        self.assertEqual(record["source_profile"]["feature_rows"]["sha256"], mapping._sha256(query / "feature_rows.csv"))
        self.assertEqual(record["source_profile"]["feature_blocks"]["context"]["sha256"], mapping._sha256(query / "context.npy"))
        self.assertEqual(frame.cell_id.tolist(), ["000", "001", "002"])
        self.assertEqual(frame.observation_id.tolist(), frame.cell_id.tolist())
        self.assertEqual(frame.reference_status.tolist(), ["assigned", "outside_reference", "missing_features"])
        self.assertEqual(frame.reference_assignment.tolist(), ["R:tissue_domain:1", "unknown", "unknown"])
        self.assertTrue(np.isnan(frame.reference_margin).all())
        self.assertTrue(np.isnan(frame.loc[2, mapping.NUMERIC_COLUMNS].astype(float)).all())
        self.assertNotIn(MARKER, frame)
        self.assertNotIn("tissue_domain", frame)
        self.assertEqual(len(frame.columns), len(record["source_profile"]["identity_columns"]) + 7)
        pd.testing.assert_frame_equal(result[original.columns], atlas.load_profile(query, ["context"]).cells[original.columns])

    def test_region_and_generic_observation_aliases_remain_bound(self):
        for generic in (False, True):
            with self.subTest(generic=generic), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                query, _, output, _, _, _ = self.bundle("tissue_region", generic)
                frame, record, _ = mapping.load_reference_mapping(output.with_suffix(".mapping.json"), query, expected_unit="tissue_region")
                self.assertTrue(frame.observation_unit.eq("tissue_region").all())
                self.assertEqual(frame.observation_id.tolist(), ["000", "001", "002"])
                self.assertEqual("region_uid" in frame, not generic)
                self.assertEqual(record["source_profile"]["format"], "observation_profiles" if generic else "region_profiles")
                with self.assertRaisesRegex(ValueError, "observation_unit"):
                    mapping.load_reference_mapping(output.with_suffix(".mapping.json"), query, expected_unit="cell")

    def test_literal_na_and_leading_zero_identity_fields_survive(self):
        query, destination, _, _, _, _ = self.bundle()
        path = query / "cell_profiles.csv"
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["sample_id"] = "NA"
        frame["cell_id"] = ["000", "NA", "090"]
        frame["cell_uid"] = ["NA:000", "NA:NA", "NA:090"]
        frame.to_csv(path, index=False)
        frame[["cell_uid", "sample_id", "cell_id"]].to_csv(query / "feature_rows.csv", index=False)
        output = self.root / "literal_ids.csv"
        mapping.write_reference_mapping(query, destination, output, atlas.map_profile)
        loaded, _, _ = mapping.load_reference_mapping(output.with_suffix(".mapping.json"), query)
        self.assertEqual(loaded.cell_id.tolist(), ["000", "NA", "090"])
        self.assertEqual(loaded.sample_id.tolist(), ["NA"] * 3)
        self.assertEqual(loaded.observation_uid.tolist(), frame.cell_uid.tolist())

    def test_singleton_and_ambiguous_reference_unknowns_are_not_promoted(self):
        query, _, output, _, _, _ = self.bundle(reference_values=[0], query_values=[0])
        frame, _, _ = mapping.load_reference_mapping(output.with_suffix(".mapping.json"), query)
        self.assertEqual(frame.reference_status.tolist(), ["insufficient_reference"])
        self.assertEqual(frame.reference_assignment.tolist(), ["unknown"])
        self.assertTrue(np.isnan(frame.reference_radius.iloc[0]))
        a, _ = make_profile(self.root / "a", "A", [-1, 0, 1])
        b, _ = make_profile(self.root / "b", "B", [-1, 0, 1])
        destination = self.root / "tied_atlas"
        atlas.build_atlas([a, b], destination, "tie-v1", ["context"])
        output = self.root / "tied.csv"
        mapping.write_reference_mapping(query, destination, output, atlas.map_profile)
        frame, _, _ = mapping.load_reference_mapping(output.with_suffix(".mapping.json"), query)
        self.assertEqual(frame.reference_status.tolist(), ["ambiguous_reference"])
        self.assertEqual(frame.reference_assignment.tolist(), ["unknown"])
        self.assertEqual(frame.reference_margin.iloc[0], 0)

    def test_copy_roundtrip_never_requires_upstream_atlas_or_original_profile(self):
        query, destination, output, _, _, _ = self.bundle()
        copied_profile = self.root / "portable_profile"
        copied_bundle = self.root / "portable_bundle"
        shutil.copytree(query, copied_profile)
        shutil.copytree(output.parent, copied_bundle)
        query.rename(self.root / "original_profile_offline")
        destination.rename(self.root / "original_atlas_offline")
        output.parent.rename(self.root / "original_mapping_offline")
        frame, _, paths = mapping.load_reference_mapping(copied_bundle / "reference_assignments.mapping.json", copied_profile)
        self.assertEqual(len(frame), 3)
        self.assertTrue(all(path.parent == copied_bundle.resolve() for path in paths.values()))
        self.assertEqual(sorted(p.name for p in copied_bundle.iterdir()), ["reference_assignments.atlas.json", "reference_assignments.csv", "reference_assignments.mapping.json"])

    def test_csv_and_atlas_byte_corruption_are_rejected(self):
        query, _, output, _, _, _ = self.bundle()
        original = output.read_bytes()
        output.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            mapping.load_reference_mapping(output.with_suffix(".mapping.json"), query)
        output.write_bytes(original)
        frozen = output.with_suffix(".atlas.json")
        frozen.write_bytes(frozen.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "identity/hash"):
            mapping.load_reference_mapping(output.with_suffix(".mapping.json"), query)

    def test_foreign_profile_reusing_local_ids_is_rejected(self):
        query, _, output, _, _, _ = self.bundle()
        foreign, _ = make_profile(self.root / "foreign", "Other", [0, 100, np.nan])
        with self.assertRaisesRegex(ValueError, "Source profile hashes/identity"):
            mapping.load_reference_mapping(output.with_suffix(".mapping.json"), foreign)
        same_ids, _ = make_profile(self.root / "same_ids", "Q", [0, 100, np.nan], markers=[.9, .8, .7])
        with self.assertRaisesRegex(ValueError, "Source profile hashes/identity"):
            mapping.load_reference_mapping(output.with_suffix(".mapping.json"), same_ids)

    def test_source_manifest_table_rows_and_selected_features_are_all_bound(self):
        query, _, output, _, _, _ = self.bundle()
        receipt = output.with_suffix(".mapping.json")
        for name in ("cell_profiles_manifest.json", "cell_profiles.csv", "feature_rows.csv", "context.npy"):
            with self.subTest(name=name):
                path = query / name
                original = path.read_bytes()
                if path.suffix == ".npy":
                    values = np.load(path)
                    values[0, 0] = 1
                    np.save(path, values)
                else:
                    path.write_bytes(original + b" ")
                with self.assertRaises((ValueError, pd.errors.ParserError)):
                    mapping.load_reference_mapping(receipt, query)
                path.write_bytes(original)

    def test_rehashed_uid_sample_order_alias_and_measurement_changes_are_rejected(self):
        query, _, output, _, _, _ = self.bundle()
        original_csv, original_receipt = output.read_bytes(), output.with_suffix(".mapping.json").read_bytes()
        changes = {
            "uid": lambda frame: frame.__setitem__("cell_uid", ["foreign", "Q:001", "Q:002"]),
            "sample": lambda frame: frame.__setitem__("sample_id", "Other"),
            "alias": lambda frame: frame.__setitem__("observation_id", ["bad", "001", "002"]),
            "measurement": lambda frame: frame.__setitem__(MARKER, ".99"),
            "discovery": lambda frame: frame.__setitem__("tissue_domain", "forbidden_relabel"),
            "order": lambda frame: frame.iloc.__setitem__(slice(None), frame.iloc[::-1].to_numpy()),
            "duplicate": lambda frame: frame.iloc.__setitem__(1, frame.iloc[0]),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                self.replace_assignments(output, change)
                with self.assertRaisesRegex(ValueError, "identities/order or original"):
                    mapping.load_reference_mapping(output.with_suffix(".mapping.json"), query)
                output.write_bytes(original_csv)
                output.with_suffix(".mapping.json").write_bytes(original_receipt)

    def test_csv_schema_row_count_and_malformed_width_are_checked_beyond_hashes(self):
        query, _, output, _, _, _ = self.bundle()
        original_csv, original_receipt = output.read_bytes(), output.with_suffix(".mapping.json").read_bytes()
        frame = pd.read_csv(output, dtype=str, keep_default_na=False)
        altered = [frame.iloc[:-1], frame[list(reversed(frame.columns))], frame.drop(columns=[MARKER]),
                   frame.assign(undeclared_column="extra")]
        for index, bad in enumerate(altered):
            with self.subTest(index=index):
                bad.to_csv(output, index=False)
                receipt, record = self.record(output)
                record["assignments"]["sha256"] = mapping._sha256(output)
                record["assignments"]["row_count"] = len(bad)
                record["assignments"]["columns"] = list(bad.columns)
                receipt.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "schema/order/count"):
                    mapping.load_reference_mapping(receipt, query)
                output.write_bytes(original_csv)
                receipt.write_bytes(original_receipt)
        output.write_bytes(original_csv + b"short,row\n")
        receipt, record = self.record(output)
        record["assignments"]["sha256"] = mapping._sha256(output)
        receipt.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "row width"):
            mapping.load_reference_mapping(receipt, query)

    def test_rehashed_foreign_group_bad_status_or_numeric_semantics_are_rejected(self):
        query, _, output, _, _, _ = self.bundle()
        original_csv, original_receipt = output.read_bytes(), output.with_suffix(".mapping.json").read_bytes()
        changes = [(0, "reference_atlas_id", "foreign"), (0, "reference_assignment", "foreign"),
                   (0, "reference_nearest_group", "foreign"), (0, "reference_status", "invented"),
                   (0, "reference_distance", "inf"), (0, "reference_distance", "not_numeric"),
                   (0, "reference_distance", ""), (0, "reference_radius", "999"),
                   (0, "reference_margin", "1"), (1, "reference_assignment", "R:tissue_domain:1"),
                   (1, "reference_status", "assigned"), (2, "reference_distance", "0"),
                   (2, "reference_nearest_group", "R:tissue_domain:1")]
        for row, key, value in changes:
            with self.subTest(key=key, value=value):
                self.replace_assignments(output, lambda frame: frame.at.__setitem__((row, key), value))
                with self.assertRaises(ValueError):
                    mapping.load_reference_mapping(output.with_suffix(".mapping.json"), query)
                output.write_bytes(original_csv)
                output.with_suffix(".mapping.json").write_bytes(original_receipt)

    def test_decimal_radius_roundtrip_is_exact_without_ulp_tolerance(self):
        frozen = {"atlas_id": "test", "groups": [{"reference_group": "group", "acceptance_radius": 1.8371172801977502}]}
        frame = pd.DataFrame({"reference_assignment": ["group"], "reference_status": ["assigned"],
                              "reference_distance": ["0.125"], "reference_radius": ["1.8371172801977502"],
                              "reference_margin": [""], "reference_nearest_group": ["group"], "reference_atlas_id": ["test"]})
        mapping._validate_interpretations(frame, frozen, [True])
        self.assertEqual(frame.reference_radius.iloc[0], frozen["groups"][0]["acceptance_radius"])
        frame["reference_radius"] = "1.8371172801977504"
        with self.assertRaisesRegex(ValueError, "radius differs"):
            mapping._validate_interpretations(frame, frozen, [True])

    def test_rehashed_atlas_identity_or_group_corruption_is_rejected(self):
        query, _, output, _, _, _ = self.bundle()
        frozen_path = output.with_suffix(".atlas.json")
        original = frozen_path.read_bytes()
        receipt_path, original_record = self.record(output)
        for key, value in (("atlas_id", "f" * 24), ("version", "swapped-atlas"), ("observation_unit", "tissue_region")):
            with self.subTest(key=key):
                frozen = json.loads(original)
                frozen[key] = value
                frozen_path.write_text(json.dumps(frozen))
                record = json.loads(json.dumps(original_record))
                record["atlas"]["sha256"] = mapping._sha256(frozen_path)
                record["atlas"][key] = value
                receipt_path.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "atlas ID"):
                    mapping.load_reference_mapping(receipt_path, query)
        frozen = json.loads(original)
        frozen["groups"][0]["reference_group"] = "foreign"
        frozen_path.write_text(json.dumps(frozen))
        record = json.loads(json.dumps(original_record))
        record["atlas"]["sha256"] = mapping._sha256(frozen_path)
        receipt_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "nearest group"):
            mapping.load_reference_mapping(receipt_path, query)

    def test_receipt_paths_cannot_escape_follow_symlinks_or_alias_artifacts(self):
        query, _, output, _, _, _ = self.bundle()
        receipt, original = self.record(output)
        external = self.root / "external.csv"
        external.write_bytes(output.read_bytes())
        (output.parent / "escaped.csv").symlink_to(external)
        for name in (str(external), "../external.csv", "escaped.csv", ".", output.with_suffix(".atlas.json").name):
            with self.subTest(name=name):
                record = json.loads(json.dumps(original))
                record["assignments"]["path"] = name
                receipt.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    mapping.load_reference_mapping(receipt, query)

    def test_existing_output_members_are_never_overwritten(self):
        query, destination, output, _, _, _ = self.bundle()
        for name in ("existing.csv", "existing.atlas.json", "existing.mapping.json"):
            with self.subTest(name=name):
                directory = self.root / name.replace(".", "_")
                directory.mkdir()
                path = directory / name
                path.write_text("user-owned")
                with self.assertRaises(FileExistsError):
                    mapping.write_reference_mapping(query, destination, directory / "existing.csv", atlas.map_profile)
                self.assertEqual(path.read_text(), "user-owned")
                self.assertEqual(list(directory.iterdir()), [path])

    def test_source_mutation_during_mapping_cannot_produce_completion(self):
        query, destination, _, _, _, _ = self.bundle()
        output = self.root / "raced.csv"
        def mutate(query_dir, atlas_dir):
            result = atlas.map_profile(query_dir, atlas_dir)
            values = np.load(query / "context.npy")
            values[0, 0] = 2
            np.save(query / "context.npy", values)
            return result
        with self.assertRaisesRegex(ValueError, "source changed"):
            mapping.write_reference_mapping(query, destination, output, mutate)
        self.assertFalse(output.with_suffix(".mapping.json").exists())
        self.assertFalse(output.exists())

    def test_atlas_mutation_during_mapping_cannot_produce_completion(self):
        query, destination, _, _, _, _ = self.bundle()
        output = self.root / "raced_atlas.csv"
        def mutate(query_dir, atlas_dir):
            result = atlas.map_profile(query_dir, atlas_dir)
            path = destination / "prototypes.npy"
            values = np.load(path)
            values[0, 0] += 1
            np.save(path, values)
            return result
        with self.assertRaisesRegex(ValueError, "atlas artifact changed"):
            mapping.write_reference_mapping(query, destination, output, mutate)
        self.assertFalse(output.with_suffix(".mapping.json").exists())

    def test_source_mutation_while_writing_leaves_no_completion(self):
        query, destination, _, _, _, _ = self.bundle()
        output = self.root / "late_race.csv"
        original = mapping._read_assignments
        def mutate(*args):
            result = original(*args)
            path = query / "cell_profiles_manifest.json"
            path.write_bytes(path.read_bytes() + b" ")
            return result
        with patch.object(mapping, "_read_assignments", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "Source profile changed"):
                mapping.write_reference_mapping(query, destination, output, atlas.map_profile)
        self.assertTrue(output.exists())
        self.assertFalse(output.with_suffix(".mapping.json").exists())

    def test_legacy_unreceipted_csv_cannot_claim_binding(self):
        query, _, output, _, _, _ = self.bundle()
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            mapping.load_reference_mapping(output, query)
        output.with_suffix(".mapping.json").unlink()
        with self.assertRaises(FileNotFoundError):
            mapping.load_reference_mapping(output.with_suffix(".mapping.json"), query)

    def test_consumer_import_never_loads_scipy_sklearn_or_mapping_runtime(self):
        script = """import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('sklearn', 'scipy', 'cell_reference_atlas'):
            raise AssertionError('ML import: ' + fullname)
sys.meta_path.insert(0, Reject())
import reference_mapping_io
print(reference_mapping_io.FORMAT)
"""
        query, _, output, _, _, _ = self.bundle()
        script += "\nreference_mapping_io.load_reference_mapping(sys.argv[2], sys.argv[3])\n"
        result = subprocess.run([sys.executable, "-c", script, str(BIN), str(output.with_suffix(".mapping.json")), str(query)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Parquet runtime not installed")
    def test_parquet_profile_missing_measurements_are_preserved(self):
        query, destination, _, _, _, _ = self.bundle()
        path = query / "cell_profiles.csv"
        frame = pd.read_csv(path, dtype={"cell_uid": str, "sample_id": str, "cell_id": str})
        frame.loc[1, MARKER] = np.nan
        frame.to_parquet(query / "cell_profiles.parquet", index=False)
        path.unlink()
        output = self.root / "parquet_source.csv"
        mapping.write_reference_mapping(query, destination, output, atlas.map_profile)
        loaded, _, _ = mapping.load_reference_mapping(output.with_suffix(".mapping.json"), query)
        self.assertEqual(loaded.cell_id.tolist(), ["000", "001", "002"])


if __name__ == "__main__":
    unittest.main()
