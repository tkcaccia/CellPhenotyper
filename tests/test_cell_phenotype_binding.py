"""Source-bound taxonomy contracts using synthetic CellViT tokens, no Torch."""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import build_cell_profiles as profiles
import cellvit_embeddings
from cell_phenotype_io import phenotype_definition
from cell_profile_io import sha256_file
from test_cellvit_profile_binding import profile_inputs, bound_bundle, bind_canonical_objects


def typed_fixture(directory, *, source_tag="first", mutation=None):
    directory.mkdir(exist_ok=True)
    args = profile_inputs(directory)
    source = directory / "source"
    exporter = cellvit_embeddings.export_embeddings

    def typed_export(raw_json, retained_json, *positional, **kwargs):
        payload = json.loads(retained_json.read_text())
        payload.update({"taxonomy": "pannuke", "type_map": {
            "1": "neoplastic", "2": "inflammatory", "3": "epithelial"}})
        for cell, code in zip(payload["cells"], (1, 2, 3)):
            cell.update(type_id=code, type=payload["type_map"][str(code)])
        payload["cells"][0]["type_name"] = "neoplastic"
        if mutation:
            mutation(payload)
        retained_json.write_text(json.dumps(payload))
        return exporter(raw_json, retained_json, *positional, **kwargs)

    # Only the graph-array loader is replaced by the shared fixture. Actual
    # exporter, receipt producer, strict bundle loader and phenotype binder run.
    with patch.object(cellvit_embeddings, "export_embeddings", side_effect=typed_export):
        bound_bundle(args, source, population_variant=source_tag)
    objects = bind_canonical_objects(args, source)
    objects["cellvitpp_type"] = ["neoplastic", "", "inflammatory"]
    objects.to_csv(args.objects, index=False)
    return args, source


def test_actual_profile_attachment_verifies_taxonomy_without_changing_labels(tmp_path):
    args, source = typed_fixture(tmp_path)
    before = {p: sha256_file(p) for p in source.rglob("*") if p.is_file()}
    cells, manifest = profiles.build_profiles(args)
    binding = manifest["phenotype_definition"]
    assert binding["verified"] is True
    assert cells.cellvitpp_id.tolist() == ["007", "", "NA"]
    assert cells.phenotype.tolist() == ["neoplastic", "unknown", "inflammatory"]
    assert cells.cell_id.tolist() == ["2", "1", "3"]
    assert binding["definition"]["taxonomy"] == "pannuke"
    assert binding["definition"]["type_map"] == {"1": "neoplastic", "2": "inflammatory", "3": "epithelial"}
    assert binding["source_binding"]["canonical_mapped_cells"] == 2
    assert binding["source_binding"]["canonical_unknown_without_source"] == 1
    assert binding["source_binding"]["retained_population_sha256"] == sha256_file(source / "cellvit_cells.json")
    assert str(source) not in json.dumps(binding)
    assert before == {p: sha256_file(p) for p in before}
    saved = json.loads((Path(args.outdir) / "cell_profiles_manifest.json").read_text())
    assert saved["phenotype_definition"] == binding


def test_definition_is_portable_but_source_hash_remains_specimen_specific(tmp_path):
    records = []
    for name in ("a", "b"):
        args, _ = typed_fixture(tmp_path / name, source_tag=name)
        records.append(profiles.build_profiles(args)[1]["phenotype_definition"])
    assert records[0]["definition"] == records[1]["definition"]
    assert records[0]["source_binding"]["retained_population_sha256"] != records[1]["source_binding"]["retained_population_sha256"]


@pytest.mark.parametrize("index,label,expected", [
    (0, "inflammatory", "differs from its exact retained"),
    (1, "neoplastic", "without a CellViT source ID must be unknown"),
    (2, "NEOPLASTIC", "differs from its exact retained"),
])
def test_foreign_or_fabricated_canonical_phenotypes_fail(tmp_path, index, label, expected):
    args, _ = typed_fixture(tmp_path)
    objects = pd.read_csv(args.objects, dtype={"cellvitpp_id": str}, keep_default_na=False)
    objects.loc[index, "cellvitpp_type"] = label
    objects.to_csv(args.objects, index=False)
    with pytest.raises(ValueError, match=expected):
        profiles.build_profiles(args)


@pytest.mark.parametrize("mutation,expected", [
    (lambda p: p.update(taxonomy="binary"), "taxonomy differs"),
    (lambda p: p["type_map"].update({"1": "inflammatory"}), "type_map code"),
    (lambda p: p["cells"][0].update(type_name="inflammatory"), "type_name and normalized type disagree"),
    (lambda p: p["type_map"].update({"1": "Neoplastic"}), "not producer-normalized"),
    (lambda p: p["cells"][0].update(type_id=8), "unmapped type code"),
])
def test_rehashed_but_semantically_contradictory_source_bundle_fails(tmp_path, mutation, expected):
    args, _ = typed_fixture(tmp_path, mutation=mutation)
    with pytest.raises(ValueError, match=expected):
        profiles.build_profiles(args)


def test_missing_taxonomy_metadata_stays_unverified(tmp_path):
    args, _ = typed_fixture(tmp_path, mutation=lambda p: p.pop("type_map"))
    cells, manifest = profiles.build_profiles(args)
    assert manifest["phenotype_definition"]["verified"] is False
    assert "lacks explicit normalized taxonomy" in manifest["phenotype_definition"]["source_binding"]["reason"]
    assert cells.phenotype.tolist() == ["neoplastic", "unknown", "inflammatory"]


@pytest.mark.parametrize("code", [None, 99])
def test_producer_unmapped_type_remains_explicit_unknown_without_relabeling(tmp_path, code):
    label = f"unknown_{code}"
    def mutation(payload):
        payload["cells"][0].update(type_id=code, type=label, type_name=label)
    args, _ = typed_fixture(tmp_path, mutation=mutation)
    objects = pd.read_csv(args.objects, dtype={"cellvitpp_id": str}, keep_default_na=False)
    objects.loc[0, "cellvitpp_type"] = label
    objects.to_csv(args.objects, index=False)
    cells, manifest = profiles.build_profiles(args)
    assert cells.phenotype.iloc[0] == label
    assert manifest["phenotype_definition"]["verified"] is True
    assert "missing categorical assignments" in manifest["phenotype_definition"]["definition"]["unknown_label_semantics"]


def test_unverified_record_does_not_follow_provenance_paths_or_upgrade_labels():
    cells = pd.DataFrame({"phenotype": ["neoplastic"]})
    record = {"reference_compatible": False, "sources": [{"role": "retained_population", "path": "/do/not/follow"}]}
    assert phenotype_definition(cells, record, source="/also/not/read")["verified"] is False
    assert phenotype_definition(cells)["verified"] is False
    assert cells.phenotype.tolist() == ["neoplastic"]


def test_helper_rechecks_exact_population_bytes_and_ignores_recorded_paths(tmp_path):
    args, source = typed_fixture(tmp_path)
    cells, manifest = profiles.build_profiles(args)
    record = copy.deepcopy(manifest["feature_blocks"]["cellvit"])
    for entry in record["sources"]:
        entry["path"] = "/untrusted/unavailable/recorded/path"
    assert phenotype_definition(cells, record, source=source)["verified"] is True
    path = source / "cellvit_cells.json"
    path.write_text(path.read_text().replace('"neoplastic"', '"epithelial"'))
    with pytest.raises(ValueError, match="SHA256/size binding mismatch"):
        phenotype_definition(cells, record, source=source)


@pytest.mark.parametrize("change,expected", [
    (lambda c: c.loc.__setitem__((0, "cellvitpp_id"), "7"), "absent from retained population"),
    (lambda c: c.loc.__setitem__((2, "cellvitpp_id"), "007"), "Duplicate canonical"),
    (lambda c: c.loc.__setitem__((0, "cellvitpp_id"), " 007"), "Invalid literal"),
])
def test_direct_helper_checks_literal_source_ids_without_numeric_aliases(tmp_path, change, expected):
    args, source = typed_fixture(tmp_path)
    cells, manifest = profiles.build_profiles(args)
    change(cells)
    with pytest.raises(ValueError, match=expected):
        phenotype_definition(cells, manifest["feature_blocks"]["cellvit"], source=source)
