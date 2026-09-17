"""Real two-specimen reference-mapping/SpatialData workflow acceptance.

All images, features and atlas observations are small synthetic fixtures. This
exercises numerical mapping, process staging and real SpatialData write/read;
it is not learned inference or independent biological validation.
"""
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sd = pytest.importorskip("spatialdata")
from test_spatialdata_export import specimen as specimen_fixture
from test_spatialdata_hierarchy import hierarchy_specimen as hierarchy_fixture, refresh_hierarchy
from cell_profile_io import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COLUMNS = ["reference_assignment", "reference_status", "reference_distance",
                     "reference_radius", "reference_margin", "reference_nearest_group", "reference_atlas_id"]
NUMERIC_REFERENCE = {"reference_distance", "reference_radius", "reference_margin"}


def atlas_python():
    configured = os.environ.get("CELLPHENOTYPER_ATLAS_TEST_PYTHON")
    local = ROOT / ".venv-spatial/bin/python"
    return configured or (str(local) if local.is_file() else sys.executable)


def atlas_cli(*args):
    result = subprocess.run([atlas_python(), str(ROOT / "bin/cell_reference_atlas.py"), *map(str, args)],
                            capture_output=True, text=True, timeout=45,
                            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def tree_hashes(directory):
    return {path: sha256_file(path) for path in directory.rglob("*") if path.is_file()}


def query_specimen(directory, sample, *, alternate=False):
    directory.mkdir()
    base = specimen_fixture.__wrapped__(directory)
    inputs = hierarchy_fixture.__wrapped__(base, directory)[0]
    profile, hierarchy = inputs["profile_dir"], inputs["hierarchy_dir"]
    cells = pd.read_parquet(profile / "cell_profiles.parquet")
    cells["sample_id"] = sample
    cells["cell_uid"] = [f"{sample}:seg:{value}" for value in cells.cell_id]
    cells.to_csv(profile / "cell_profiles.csv", index=False)
    cells.to_parquet(profile / "cell_profiles.parquet", index=False)
    cells[["sample_id", "cell_id", "cell_uid"]].to_csv(profile / "feature_rows.csv", index=False)
    if alternate:
        np.save(profile / "local.npy", np.array([[100., 100.], [np.nan, np.nan]], np.float32))
    path = profile / "cell_profiles_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["sample_id"] = sample
    manifest["files"] = {name: sha256_file(profile / name) for name in manifest["files"]}
    for record in manifest["feature_blocks"].values():
        record["sha256"] = sha256_file(profile / record["path"])
    path.write_text(json.dumps(manifest))

    hierarchy_id = f"hierarchy-{sample}"
    for relative in ("region_profiles/region_profiles.csv", "region_profiles/feature_rows.csv", "grid_subdomains.csv"):
        path = hierarchy / relative
        rows = pd.read_csv(path, dtype={"region_id": str})
        rows["sample_id"] = sample
        if "region_uid" in rows:
            rows["region_uid"] = [f"{sample}::{hierarchy_id}::region_{value}" for value in rows.region_id]
        rows.to_csv(path, index=False)
    for relative in ("hierarchy_summary.json", "region_profiles/region_profiles_manifest.json"):
        path = hierarchy / relative
        metadata = json.loads(path.read_text())
        metadata["hierarchy_id"] = hierarchy_id
        if "sample_id" in metadata:
            metadata["sample_id"] = sample
        path.write_text(json.dumps(metadata))
    if alternate:
        for name in ("local", "context"):
            path = hierarchy / "region_profiles" / f"{name}.npy"
            values = np.load(path, allow_pickle=False)
            np.save(path, values[[1, 0, 2]])
    refresh_hierarchy(hierarchy)
    record = {"sample_id": sample, "existing_profiles": str(profile), "hierarchy": str(hierarchy),
              "image": str(inputs["image"]), "labels": str(inputs["labels"]), "shift": str(inputs["shift"]),
              "resolution": str(inputs["resolution_json"]), "tissue_geojson": str(inputs["tissue_geojson"]),
              "tissue_coordinates": "crop_pixels"}
    return record


def reference_profile(directory, query, unit):
    """Independent reference observations with the same declared feature schema."""
    directory.mkdir()
    stem = "cell" if unit == "cell" else "region"
    source = Path(query)
    table_name = f"{stem}_profiles.csv"
    manifest_name = f"{stem}_profiles_manifest.json"
    template = pd.read_csv(source / table_name, dtype={f"{stem}_id": str}, keep_default_na=False)
    rows = pd.concat([template.iloc[[0]]] * 3, ignore_index=True)
    rows["sample_id"] = f"independent_{stem}_reference"
    rows[f"{stem}_id"] = ["001", "002", "003"]
    rows[f"{stem}_uid"] = [f"independent_{stem}:reference:{i}" for i in range(3)]
    rows.to_csv(directory / table_name, index=False)
    rows[["sample_id", f"{stem}_id", f"{stem}_uid"]].to_csv(directory / "feature_rows.csv", index=False)
    original = json.loads((source / manifest_name).read_text())
    manifest = {"schema_version": "1.0.0", "observation_unit": unit, f"{stem}_count": 3,
                "sample_id": rows.sample_id.iloc[0], "feature_blocks": {}, "files": {}}
    groups = ["local"] if unit == "cell" else ["local", "context"]
    for name in groups:
        source_record = original["feature_blocks"][name]
        center = np.load(source / source_record["path"], allow_pickle=False)[0]
        values = np.stack([center - .05, center, center + .05]).astype(np.float32)
        np.save(directory / f"{name}.npy", values)
        manifest["feature_blocks"][name] = {**copy.deepcopy(source_record), "path": f"{name}.npy",
            "shape": list(values.shape), "sha256": sha256_file(directory / f"{name}.npy")}
    manifest["files"] = {path.name: sha256_file(path) for path in directory.iterdir() if path.is_file()}
    (directory / manifest_name).write_text(json.dumps(manifest))
    return groups


@pytest.fixture(scope="module")
def mapping_dataset(tmp_path_factory):
    if not shutil.which("nextflow"):
        pytest.skip("Nextflow unavailable")
    root = tmp_path_factory.mktemp("reference_mapping_workflow")
    records = [query_specimen(root / sample, sample, alternate=sample == "sample_B")
               for sample in ("sample_A", "sample_B")]
    atlases = {}
    receipts = {}
    for unit in ("cell", "region"):
        query = Path(records[0]["existing_profiles"] if unit == "cell" else records[0]["hierarchy"])
        if unit == "region":
            query /= "region_profiles"
        reference = root / f"independent_{unit}_reference"
        groups = reference_profile(reference, query, "cell" if unit == "cell" else "tissue_region")
        atlas = root / f"frozen_{unit}_atlas"
        atlas_cli("build", "--profiles", reference, "--outdir", atlas,
                  "--version", f"synthetic-{unit}-v1", "--feature-groups", *groups,
                  "--label-column", "tissue_domain" if unit == "cell" else "subdomain_id")
        atlases[unit] = atlas
        for row in records:
            query = Path(row["existing_profiles"] if unit == "cell" else row["hierarchy"])
            if unit == "region":
                query /= "region_profiles"
            output = root / f"{row['sample_id']}_{unit}_mapping" / "reference_assignments.csv"
            atlas_cli("map", "--query", query, "--atlas", atlas, "--output", output)
            receipt = output.with_suffix(".mapping.json")
            assert receipt.is_file(), "Real mapping CLI must produce its validation receipt"
            receipts[row["sample_id"], unit] = receipt
    return root, records, atlases, receipts


def run_workflow(directory, rows, *, export=True, cell_atlas=None, region_atlas=None):
    directory.mkdir(exist_ok=True)
    samples = directory / "samples.json"
    samples.write_text(json.dumps(rows))
    params = {"cell_profile_samples": str(samples), "outdir_base": str(directory / "output"),
              "cell_profiles_spatialdata": export, "spatialdata_python": sys.executable,
              "cell_atlas_python": atlas_python(), "spatialdata_tile_size": 16,
              "cell_profiles_cpus": 1, "cell_profiles_memory_gb": 2,
              "cell_reference_atlas": str(cell_atlas) if cell_atlas else "",
              "region_reference_atlas": str(region_atlas) if region_atlas else ""}
    config = directory / "params.json"
    config.write_text(json.dumps(params))
    return subprocess.run([shutil.which("nextflow"), "-log", str(directory / "nextflow.log"),
        "run", str(ROOT / "cell_profiles.nf"), "-params-file", str(config), "-ansi-log", "false",
        "-work-dir", str(directory / "work"), "-with-trace", str(directory / "trace.tsv")],
        cwd=directory, env=dict(os.environ, NXF_OFFLINE="true", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        capture_output=True, text=True, timeout=180)


def assert_mapping_readback(table, csv_path, receipt_path, atlas_path, uid_column):
    expected = pd.read_csv(csv_path, dtype={uid_column: str}, keep_default_na=False, float_precision="round_trip")
    assert table.obs_names.tolist() == expected[uid_column].tolist()
    for name in REFERENCE_COLUMNS:
        assert name in table.obs
        if name in NUMERIC_REFERENCE:
            numbers = np.asarray([float(value) if value != "" else np.nan for value in expected[name]], dtype=float)
            np.testing.assert_array_equal(table.obs[name].to_numpy(float), numbers)
        else:
            assert table.obs[name].astype(str).tolist() == expected[name].astype(str).tolist()
    metadata = json.loads(table.uns["cellphenotyper"]["reference_mapping_json"])
    assert metadata["receipt_json"] == receipt_path.read_text()
    assert metadata["atlas_manifest_json"] == atlas_path.read_text()
    for name, path in (("receipt", receipt_path), ("assignments", csv_path), ("atlas_manifest", atlas_path)):
        assert metadata["artifacts"][name] == {"filename": path.name, "sha256": sha256_file(path),
                                               "size_bytes": path.stat().st_size}
    return expected, metadata


def test_actual_two_specimen_fresh_cell_and_existing_region_reference_export(mapping_dataset, tmp_path):
    root, original, atlases, receipts = mapping_dataset
    rows = copy.deepcopy(original)
    for row in rows:
        row["region_reference_mapping"] = str(receipts[row["sample_id"], "region"])
    frozen = tree_hashes(root)
    result = run_workflow(tmp_path, rows, cell_atlas=atlases["cell"])
    assert result.returncode == 0, result.stdout + result.stderr
    statuses = {}
    for row in rows:
        sample = row["sample_id"]
        package = sd.SpatialData.read(tmp_path / "output/20_spatialdata" / sample / "spatialdata.zarr")
        cell_source = tmp_path / "output/21_reference_mapping" / sample / "reference_assignments.csv"
        cell_expected, _ = assert_mapping_readback(package.tables["cells"], cell_source,
            cell_source.with_suffix(".mapping.json"), cell_source.with_suffix(".atlas.json"), "cell_uid")
        region_receipt = receipts[sample, "region"]
        region_source = region_receipt.with_name("reference_assignments.csv")
        region_expected, _ = assert_mapping_readback(package.tables["hierarchy_region_profiles"], region_source,
            region_receipt, region_source.with_suffix(".atlas.json"), "region_uid")
        for table in (package.tables["cells"], package.tables["hierarchy_region_profiles"]):
            assert table.obs.sample_id.astype(str).eq(sample).all()
            assert all(uid.startswith(sample + ":") for uid in table.obs_names)
        assert set(package.attrs["cellphenotyper"]["reference_mappings"]) == {"cell", "region"}
        assert cell_expected.reference_atlas_id.eq(json.loads((atlases["cell"] / "atlas_manifest.json").read_text())["atlas_id"]).all()
        assert region_expected.reference_atlas_id.eq(json.loads((atlases["region"] / "atlas_manifest.json").read_text())["atlas_id"]).all()
        original_cells = pd.read_parquet(Path(row["existing_profiles"]) / "cell_profiles.parquet")
        assert package.tables["cells"].obs.cell_id.astype(str).tolist() == original_cells.cell_id.tolist()
        np.testing.assert_array_equal(package.tables["cells"].obs.predicted__nucleus__CD3__mean, original_cells.predicted__nucleus__CD3__mean)
        statuses[sample] = (cell_expected.reference_status.tolist(), region_expected.reference_status.tolist())
    assert statuses == {
        "sample_A": (["assigned", "outside_reference"], ["assigned", "outside_reference", "missing_features"]),
        "sample_B": (["outside_reference", "missing_features"], ["outside_reference", "assigned", "missing_features"])}
    assert tree_hashes(root) == frozen
    assert not (tmp_path / "output/19_cell_profiles").exists()
    assert not (tmp_path / "output/23_region_reference_mapping").exists()  # attachment, not remapping
    trace = pd.read_csv(tmp_path / "trace.tsv", sep="\t")
    assert len(trace) == 6 and trace.status.eq("COMPLETED").all() and trace.exit.eq(0).all()
    exports = [path for path in (tmp_path / "work").rglob(".command.sh") if "export_spatialdata.py" in path.read_text()]
    assert len(exports) == 2
    for command in exports:
        assert "--cell-reference-mapping" in command.read_text() and "--region-reference-mapping" in command.read_text()
        for kind in ("cell_reference", "region_reference"):
            assert {path.name for path in (command.parent / kind).iterdir()} == {
                "reference_assignments.csv", "reference_assignments.atlas.json", "reference_assignments.mapping.json"}


def test_actual_global_region_atlas_maps_only_specimens_with_hierarchy(mapping_dataset, tmp_path):
    root, original, atlases, _ = mapping_dataset
    rows = copy.deepcopy(original)
    rows[1].pop("hierarchy")
    frozen = tree_hashes(root)
    result = run_workflow(tmp_path, rows, region_atlas=atlases["region"])
    assert result.returncode == 0, result.stdout + result.stderr
    stage = tmp_path / "output/23_region_reference_mapping/sample_A/artifact_hierarchy"
    assert {path.name for path in stage.iterdir()} == {
        "reference_assignments.csv", "reference_assignments.atlas.json", "reference_assignments.mapping.json"}
    source = stage / "reference_assignments.csv"
    for sample in ("sample_A", "sample_B"):
        package = sd.SpatialData.read(tmp_path / "output/20_spatialdata" / sample / "spatialdata.zarr")
        cells = package.tables["cells"]
        assert cells.obs.sample_id.astype(str).eq(sample).all()
        assert all(uid.startswith(sample + ":") for uid in cells.obs_names)
        assert not set(REFERENCE_COLUMNS) & set(cells.obs)
        assert "reference_mapping_json" not in cells.uns["cellphenotyper"]
        if sample == "sample_A":
            expected, _ = assert_mapping_readback(package.tables["hierarchy_region_profiles"], source,
                source.with_suffix(".mapping.json"), source.with_suffix(".atlas.json"), "region_uid")
            assert expected.sample_id.eq(sample).all()
            assert expected.reference_status.tolist() == ["assigned", "outside_reference", "missing_features"]
            assert set(package.attrs["cellphenotyper"]["reference_mappings"]) == {"region"}
        else:
            assert "hierarchy_region_profiles" not in package.tables
            assert "cell_hierarchy_overlaps" not in package.tables
            assert package.attrs["cellphenotyper"]["reference_mappings"] == {}
    assert not (tmp_path / "output/23_region_reference_mapping/sample_B").exists()
    assert not (tmp_path / "output/24_cell_tissue_links/sample_B").exists()
    assert not (tmp_path / "output/21_reference_mapping").exists()
    assert not (tmp_path / "output/19_cell_profiles").exists()
    assert tree_hashes(root) == frozen
    trace = pd.read_csv(tmp_path / "trace.tsv", sep="\t")
    assert len(trace) == 4 and trace.status.eq("COMPLETED").all() and trace.exit.eq(0).all()
    exports = [path for path in (tmp_path / "work").rglob(".command.sh") if "export_spatialdata.py" in path.read_text()]
    assert len(exports) == 2
    assert sum("--region-reference-mapping" in path.read_text() for path in exports) == 1
    assert not any("--cell-reference-mapping" in path.read_text() for path in exports)


@pytest.mark.parametrize("case", ["ignored_cell", "ignored_region", "duplicate_cell", "duplicate_region",
                                  "wrong_unit", "foreign_specimen", "region_without_hierarchy"])
def test_reference_attachments_cannot_be_ignored_conflicted_or_cross_bound(mapping_dataset, tmp_path, case):
    _, records, atlases, receipts = mapping_dataset
    row = copy.deepcopy(records[0])
    options = {}
    if case in {"ignored_cell", "duplicate_cell", "wrong_unit"}:
        row["cell_reference_mapping"] = str(receipts["sample_A", "region" if case == "wrong_unit" else "cell"])
    else:
        row["region_reference_mapping"] = str(receipts["sample_B" if case == "foreign_specimen" else "sample_A", "region"])
    if case.startswith("ignored"):
        options["export"] = False
    elif case == "duplicate_cell":
        options["cell_atlas"] = atlases["cell"]
    elif case == "duplicate_region":
        options["region_atlas"] = atlases["region"]
    elif case == "region_without_hierarchy":
        row.pop("hierarchy")
    result = run_workflow(tmp_path, [row], **options)
    assert result.returncode != 0, result.stdout + result.stderr
    diagnostic = (result.stdout + result.stderr).lower()
    if case.startswith("ignored"):
        assert "must not be silently ignored" in diagnostic, diagnostic
    elif case.startswith("duplicate"):
        assert "choose" in diagnostic and "not both" in diagnostic, diagnostic
    elif case == "region_without_hierarchy":
        assert "requires the corresponding tissue hierarchy" in diagnostic, diagnostic
    else:
        # These reach the real Python content validator, not just filename checks.
        expected = "observation_unit" if case == "wrong_unit" else "source profile hashes/identity differ"
        assert expected in diagnostic, diagnostic
    assert "compilation error" not in diagnostic and "no such variable" not in diagnostic
    assert not (tmp_path / "output/20_spatialdata").exists()
