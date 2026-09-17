"""Real SpatialData I/O of frozen reference interpretations; no learned models."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sd = pytest.importorskip("spatialdata")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import cell_reference_atlas as atlas
import export_spatialdata as exporter
from cell_profile_io import sha256_file
from test_spatialdata_export import specimen, affine, measured_package
from test_spatialdata_hierarchy import hierarchy_specimen, linked_inputs


def mapping_bundle(profile, destination, unit="cell"):
    """Use actual atlas construction and map CLI, not handwritten receipts."""
    destination.mkdir(parents=True)
    stem, uid, instance = (("cell_profiles", "cell_uid", "cell_id") if unit == "cell" else
                           ("region_profiles", "region_uid", "region_id"))
    original = json.loads((profile / f"{stem}_manifest.json").read_text())
    query = pd.read_csv(profile / f"{stem}.csv", dtype={uid: str, instance: str, "sample_id": str}, keep_default_na=False)
    reference = destination / "reference_profiles"
    reference.mkdir()
    rows = pd.concat([query.iloc[:1]] * 6, ignore_index=True)
    rows["sample_id"] = "independent synthetic reference"
    rows[uid] = [f"reference:{unit}:{i}" for i in range(6)]
    rows[instance] = [str(i + 1) for i in range(6)]
    rows["reviewed_label"] = "synthetic_group"
    rows.to_csv(reference / f"{stem}.csv", index=False)
    rows[[uid, "sample_id", instance]].to_csv(reference / "feature_rows.csv", index=False)
    group = "context"
    source = np.load(profile / original["feature_blocks"][group]["path"], allow_pickle=False)
    values = (source[0] + np.linspace(-.05, .05, 6, dtype=np.float32)[:, None]).astype(np.float32)
    np.save(reference / "context.npy", values, allow_pickle=False)
    block = dict(original["feature_blocks"][group], path="context.npy", shape=list(values.shape),
                 sha256=sha256_file(reference / "context.npy"))
    header = {"schema_version": "1.0.0", "observation_unit": unit,
        "cell_count" if unit == "cell" else "region_count": len(rows),
        "feature_blocks": {group: block}, "files": {name: sha256_file(reference / name)
            for name in (f"{stem}.csv", "feature_rows.csv", "context.npy")}}
    (reference / f"{stem}_manifest.json").write_text(json.dumps(header))
    frozen = destination / "atlas"
    atlas.build_atlas([reference], frozen, "synthetic-v1", [group], label_column="reviewed_label")
    output = destination / "reference_assignments.csv"
    completed = subprocess.run([sys.executable, str(ROOT / "bin/cell_reference_atlas.py"), "map",
        "--query", str(profile), "--atlas", str(frozen), "--output", str(output)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = output.with_suffix(".mapping.json")
    assert receipt.is_file()
    return receipt


def expected_mapping(receipt, profile, unit):
    from reference_mapping_io import load_reference_mapping
    return load_reference_mapping(receipt, profile, expected_unit=unit)[0]


def assert_mapping(table, expected, portable, receipt):
    identity = "cell_uid" if portable["observation_unit"] == "cell" else "region_uid"
    assert table.obs_names.tolist() == expected[identity].tolist()
    for name in exporter.REFERENCE_FIELDS:
        original = expected[name].to_numpy()
        actual = table.obs[name].to_numpy()
        if original.dtype.kind in "fiu":
            assert actual.dtype == original.dtype
            np.testing.assert_array_equal(actual, original)
        else:
            assert actual.tolist() == original.tolist()
    metadata = json.loads(table.uns["cellphenotyper"]["reference_mapping_json"])
    assert metadata == portable
    assert metadata["receipt_json"] == receipt.read_text()
    assert metadata["atlas_manifest_json"] == receipt.with_name(receipt.name.replace(".mapping.json", ".atlas.json")).read_text()
    assert metadata["artifacts"]["receipt"]["sha256"] == sha256_file(receipt)
    assert not any("profile" in key for key in metadata["artifacts"])


def test_cell_reference_interpretations_are_additive_portable_and_exact(specimen, tmp_path, monkeypatch):
    inputs, pixels, labels, context = specimen
    receipt = mapping_bundle(inputs["profile_dir"], tmp_path / "cell_mapping")
    expected = expected_mapping(receipt, inputs["profile_dir"], "cell")
    assert expected.reference_status.tolist() == ["assigned", "missing_features"]
    originals = {p: sha256_file(p) for p in inputs["profile_dir"].rglob("*") if p.is_file()}
    monkeypatch.setattr(atlas, "map_profile", lambda *args, **kwargs: pytest.fail("Export must not recompute mapping"))
    output = tmp_path / "reference.zarr"
    summary = exporter.export_spatialdata(**inputs, cell_reference_mapping=receipt, outdir=output)
    loaded = sd.SpatialData.read(output)
    table = loaded.tables["cells"]
    assert set(summary["reference_mappings"]) == {"cell"}
    assert_mapping(table, expected, summary["reference_mappings"]["cell"], receipt)
    assert table.obs.cell_id.tolist() == ["001", "7"]
    assert table.obs.instance_id.tolist() == [1, 7]
    assert table.obs.tissue_domain.tolist() == [1, 2]
    np.testing.assert_array_equal(table.obs.predicted__nucleus__CD3__mean, [.1, np.nan])
    np.testing.assert_array_equal(table.obsm["context"], context)
    np.testing.assert_array_equal(table.obsp["neighbors_25um"].toarray(), [[0, 1], [1, 0]])
    np.testing.assert_array_equal(loaded.labels["canonical_cells"].data.compute(), labels)
    np.testing.assert_array_equal(loaded.images["he_image"].data.compute(), np.moveaxis(pixels, -1, 0))
    assert all(sha256_file(path) == digest for path, digest in originals.items())
    # An exported table must retain the complete interpretation without original
    # mapping/atlas directories being mounted in a new environment.
    receipt.parent.rename(tmp_path / "unmounted_mapping_sources")
    relocated = sd.SpatialData.read(output)
    assert json.loads(relocated.tables["cells"].uns["cellphenotyper"]["reference_mapping_json"]) == summary["reference_mappings"]["cell"]


def test_cell_and_region_mappings_keep_their_own_hierarchy_instances(hierarchy_specimen, tmp_path):
    inputs = linked_inputs(hierarchy_specimen, tmp_path)
    assay = measured_package(hierarchy_specimen[0], tmp_path / "independent_assay")
    cell_receipt = mapping_bundle(inputs["profile_dir"], tmp_path / "cell_map")
    region_profile = inputs["hierarchy_dir"] / "region_profiles"
    region_receipt = mapping_bundle(region_profile, tmp_path / "region_map", "tissue_region")
    expected_cell = expected_mapping(cell_receipt, inputs["profile_dir"], "cell")
    expected_region = expected_mapping(region_receipt, region_profile, "tissue_region")
    assert expected_region.reference_status.tolist() == ["assigned", "outside_reference", "missing_features"]
    output = tmp_path / "linked_reference.zarr"
    summary = exporter.export_spatialdata(**inputs, cell_reference_mapping=cell_receipt,
        region_reference_mapping=region_receipt, measured_assay=[assay], outdir=output)
    loaded = sd.SpatialData.read(output)
    assert_mapping(loaded.tables["cells"], expected_cell, summary["reference_mappings"]["cell"], cell_receipt)
    regions = loaded.tables["hierarchy_region_profiles"]
    assert_mapping(regions, expected_region, summary["reference_mappings"]["region"], region_receipt)
    assert regions.obs.instance_id.tolist() == [1, 2, 3]
    assert regions.obs.spatial_region.astype(str).eq("hierarchy_regions").all()
    assert not {"cell_id", "cell_uid"} & set(regions.obs)
    assert not set(exporter.REFERENCE_FIELDS) & set(loaded.tables["cell_hierarchy_overlaps"].obs)
    for name, matrix in hierarchy_specimen[3].items():
        np.testing.assert_array_equal(regions.obsm[name], matrix)
    np.testing.assert_array_equal(affine(loaded.labels["hierarchy_regions"]), [[.5, 0, 50], [0, .5, 100], [0, 0, 1]])
    measured_record = summary["measured_modalities"][0]
    assert measured_record["profile_registry_binding"]["status"] == "verified_additive_hierarchy_source_registry"
    measured = loaded.tables[measured_record["table"]]
    assert measured.obs_names.tolist() == loaded.tables["cells"].obs_names.tolist()
    np.testing.assert_array_equal(measured.X, [[16777217.25, np.nan], [np.nan, np.nan]])
    assert measured.X.dtype == np.float64
    assert not set(exporter.REFERENCE_FIELDS) & set(measured.obs)


def test_region_mapping_without_hierarchy_is_never_silently_ignored(specimen, tmp_path):
    with pytest.raises(ValueError, match="requires.*hierarchy"):
        exporter.export_spatialdata(**specimen[0], region_reference_mapping=tmp_path / "not_read.mapping.json",
                                   outdir=tmp_path / "invalid.zarr")
    assert not (tmp_path / "invalid.zarr").exists()


def test_cell_mapping_cannot_be_attached_to_region_instances(hierarchy_specimen, tmp_path):
    inputs = linked_inputs(hierarchy_specimen, tmp_path)
    receipt = mapping_bundle(inputs["profile_dir"], tmp_path / "cell_map")
    with pytest.raises(ValueError):
        exporter.export_spatialdata(**inputs, region_reference_mapping=receipt, outdir=tmp_path / "wrong_unit.zarr")
    assert not (tmp_path / "wrong_unit.zarr").exists()


def test_cell_mapping_requires_the_exact_final_linked_profile(hierarchy_specimen, tmp_path):
    receipt = mapping_bundle(hierarchy_specimen[0]["profile_dir"], tmp_path / "before_link_map")
    inputs = linked_inputs(hierarchy_specimen, tmp_path)
    with pytest.raises(ValueError, match="Source profile hashes/identity"):
        exporter.export_spatialdata(**inputs, cell_reference_mapping=receipt, outdir=tmp_path / "old_profile.zarr")
    assert not (tmp_path / "old_profile.zarr").exists()


@pytest.mark.parametrize("fault", ["assignment_bytes", "atlas_bytes", "receipt_bytes", "stale_profile", "legacy_csv"])
def test_mapping_integrity_failures_stop_before_store_creation(specimen, tmp_path, fault):
    inputs = specimen[0]
    receipt = mapping_bundle(inputs["profile_dir"], tmp_path / "map")
    if fault in {"assignment_bytes", "atlas_bytes"}:
        suffix = ".csv" if fault == "assignment_bytes" else ".atlas.json"
        path = receipt.with_name(receipt.name.replace(".mapping.json", suffix))
        path.write_bytes(path.read_bytes() + b"\n")
    elif fault == "receipt_bytes":
        receipt.write_text("{}")
    elif fault == "legacy_csv":
        receipt = receipt.with_name(receipt.name.replace(".mapping.json", ".csv"))
    else:
        manifest_path = inputs["profile_dir"] / "cell_profiles_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["extra_review_note"] = "same cells, different immutable profile version"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, KeyError)):
        exporter.export_spatialdata(**inputs, cell_reference_mapping=receipt, outdir=tmp_path / "bad.zarr")
    assert not (tmp_path / "bad.zarr").exists()


def test_reference_field_collision_does_not_overwrite_original_profile(specimen, tmp_path):
    inputs = specimen[0]
    root = inputs["profile_dir"]
    table = pd.read_parquet(root / "cell_profiles.parquet")
    table["Reference_Status"] = ["original measurement", "original missingness"]
    table.to_csv(root / "cell_profiles.csv", index=False)
    table.to_parquet(root / "cell_profiles.parquet", index=False)
    path = root / "cell_profiles_manifest.json"
    manifest = json.loads(path.read_text())
    for name in ("cell_profiles.csv", "cell_profiles.parquet"):
        manifest["files"][name] = sha256_file(root / name)
    path.write_text(json.dumps(manifest))
    receipt = mapping_bundle(root, tmp_path / "map")
    with pytest.raises(ValueError, match="collide"):
        exporter.export_spatialdata(**inputs, cell_reference_mapping=receipt, outdir=tmp_path / "collision.zarr")
    assert not (tmp_path / "collision.zarr").exists()


@pytest.mark.parametrize("fault", ["source_after_write", "written_value", "written_metadata"])
def test_source_changes_and_exact_readback_failures_are_detected(specimen, tmp_path, monkeypatch, fault):
    import zarr
    inputs = specimen[0]
    receipt = mapping_bundle(inputs["profile_dir"], tmp_path / "map")
    original_write = sd.SpatialData.write
    def damaged_write(self, path, *args, **kwargs):
        result = original_write(self, path, *args, **kwargs)
        if fault == "source_after_write":
            receipt.write_text(receipt.read_text() + "\n")
        else:
            group = zarr.open_group(path, mode="r+")
            if fault == "written_value":
                group["tables"]["cells"]["obs"]["reference_distance"][0] = 123.25
            else:
                attributes = dict(group.attrs["cellphenotyper"])
                attributes["reference_mappings"]["cell"]["status"] = "changed"
                group.attrs["cellphenotyper"] = attributes
        return result
    monkeypatch.setattr(sd.SpatialData, "write", damaged_write)
    with pytest.raises(RuntimeError, match="[Rr]eference"):
        exporter.export_spatialdata(**inputs, cell_reference_mapping=receipt, outdir=tmp_path / "damaged.zarr")


def test_payload_change_after_shared_loader_validation_is_rejected(specimen, tmp_path, monkeypatch):
    import reference_mapping_io
    inputs = specimen[0]
    receipt = mapping_bundle(inputs["profile_dir"], tmp_path / "map")
    original_load = reference_mapping_io.load_reference_mapping
    def changed_after_validation(*args, **kwargs):
        frame, record, paths = original_load(*args, **kwargs)
        path = paths["assignments"]
        path.write_bytes(path.read_bytes() + b"\n")
        return frame, record, paths
    monkeypatch.setattr(reference_mapping_io, "load_reference_mapping", changed_after_validation)
    with pytest.raises(ValueError, match="payload changed after validation"):
        exporter.export_spatialdata(**inputs, cell_reference_mapping=receipt, outdir=tmp_path / "raced.zarr")
    assert not (tmp_path / "raced.zarr").exists()


def test_export_cli_accepts_verified_mapping_receipt(specimen, tmp_path):
    inputs = specimen[0]
    receipt = mapping_bundle(inputs["profile_dir"], tmp_path / "map")
    command = [sys.executable, str(ROOT / "bin/export_spatialdata.py")]
    for key, value in {**inputs, "cell_reference_mapping": receipt, "outdir": tmp_path / "cli.zarr"}.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])["reference_mappings"] == ["cell"]
