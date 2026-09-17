"""Real tiny pooled fit + per-specimen SpatialData wiring, without model inference.

Fixtures/fitting use the separate ML environment; export/readback use SpatialData
without scikit-learn. Source profiles and within-specimen graphs stay immutable.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sd = pytest.importorskip("spatialdata")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from cohort_niche_io import COHORT_COLUMNS


def ml_python():
    configured = os.environ.get("CELLPHENOTYPER_ATLAS_TEST_PYTHON")
    local = ROOT / ".venv-spatial/bin/python"
    return configured or (str(local) if local.is_file() else sys.executable)


def hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    if not shutil.which("nextflow"):
        pytest.skip("Nextflow unavailable")
    root = tmp_path_factory.mktemp("cohort_export_sources")
    script = """import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'tests'))
from test_cohort_niche_workflow import specimen
root = Path(sys.argv[2])
rows = [specimen(root / name, name) for name in ('B', 'A')]
for row in rows:
    polygon = root / row['sample_id'] / 'domains.geojson'
    polygon.write_text(json.dumps({'type':'FeatureCollection','features':[
        {'type':'Feature','properties':{'value':1},'geometry':{'type':'Polygon',
         'coordinates':[[[0,0],[64,0],[64,64],[0,64],[0,0]]]}}]}))
    row.update(tissue_geojson=str(polygon), tissue_coordinates='crop_pixels')
(root / 'records.json').write_text(json.dumps(rows))
"""
    result = subprocess.run([ml_python(), "-c", script, str(ROOT), str(root)],
        capture_output=True, text=True, timeout=45,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    assert result.returncode == 0, result.stdout + result.stderr
    return root, json.loads((root / "records.json").read_text())


def run_workflow(directory, rows, *, fit=True, bundle=None, export=True,
                 label="run", resume=False, script=None, extra=None):
    directory.mkdir(exist_ok=True)
    samples = directory / "samples.json"
    samples.write_text(json.dumps(rows))
    params = {"cell_profile_samples": str(samples), "outdir_base": str(directory / "output"),
              "cell_profiles_spatialdata": export, "spatialdata_python": sys.executable,
              "cell_atlas_python": ml_python(), "spatialdata_tile_size": 16,
              "spatialdata_pyramid_levels": 0, "cell_profiles_cpus": 1, "cell_profiles_memory_gb": 2,
              "cohort_niches_enable": fit, "cohort_niches_bundle": str(bundle) if bundle else None,
              "cohort_niches_max_k": 2, "cohort_niches_fixed_k": 2,
              "cohort_niches_repeats": 2, "cohort_niches_fit_limit": 100,
              "cohort_niches_max_working_mb": 64, "cell_profiles_domain_source": "none"}
    params.update(extra or {})
    parameters = directory / "params.json"
    parameters.write_text(json.dumps(params))
    config = directory / "trace.config"
    config.write_text("trace.fields = 'task_id,hash,process,name,status,exit'\n")
    trace = directory / f"{label}.tsv"
    command = [shutil.which("nextflow"), "-log", str(directory / f"{label}.log"),
               "run", str(script or ROOT / "cell_profiles.nf"), "-params-file", str(parameters),
               "-c", str(config), "-ansi-log", "false", "-work-dir", str(directory / "work"),
               "-with-trace", str(trace)]
    if resume:
        command.append("-resume")
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=180,
        env=dict(os.environ, NXF_OFFLINE="true", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    return result, pd.read_csv(trace, sep="\t") if trace.exists() else None


def assert_ok(result):
    assert result.returncode == 0, result.stdout + result.stderr


def readback(directory, rows, bundle=None):
    expected = pd.read_parquet(bundle / "cohort_niche_assignments.parquet") if bundle else None
    identities = set()
    for row in rows:
        sample = row["sample_id"]
        profile = Path(row["existing_profiles"])
        source = pd.read_parquet(profile / "cell_profiles.parquet")
        package = sd.SpatialData.read(directory / "output/20_spatialdata" / sample / "spatialdata.zarr")
        table = package.tables["cells"]
        assert len(table.obs) == 24
        assert table.obs_names.tolist() == source.cell_uid.tolist()
        assert table.obs.sample_id.astype(str).eq(sample).all()
        field_names = json.loads(table.uns["cellphenotyper"]["field_name_mapping_json"])["obs"]
        for name in ("cell_id", "niche_id", "niche_status", "x_um", "y_um"):
            actual = table.obs[field_names[name]]
            if source[name].dtype.kind in "biuf":
                np.testing.assert_array_equal(actual.to_numpy(), source[name].to_numpy())
            else:
                assert actual.astype(str).tolist() == source[name].astype(str).tolist()
        if bundle:
            wanted = expected[expected.sample_id == sample]
            assert wanted.cell_uid.tolist() == source.cell_uid.tolist()
            for name in COHORT_COLUMNS:
                if wanted[name].dtype.kind in "biuf":
                    np.testing.assert_array_equal(table.obs[name].to_numpy(), wanted[name].to_numpy())
                else:
                    assert table.obs[name].astype(str).tolist() == wanted[name].astype(str).tolist()
            record = json.loads(table.uns["cellphenotyper"]["cohort_niches_json"])
            assert record == package.attrs["cellphenotyper"]["cohort_niches"]
            assert record["source_identity"]["manifest_sha256"] == hashes(profile)["cell_profiles_manifest.json"]
            assert record["bundle_files"] == hashes(bundle)
            identities.add(record["cohort_niche_model_id"])
        else:
            assert not set(COHORT_COLUMNS) & set(table.obs)
            assert package.attrs["cellphenotyper"]["cohort_niches"] == {"status": "not_provided"}
    assert len(identities) == (1 if bundle else 0)


@pytest.fixture(scope="module")
def fitted(dataset, tmp_path_factory):
    source, rows = dataset
    before = hashes(source)
    run = tmp_path_factory.mktemp("cohort_fit_export")
    result, trace = run_workflow(run, rows, label="first")
    assert_ok(result)
    assert trace.process.value_counts().to_dict() == {"EXPORT_SPATIALDATA": 2, "FIT_COHORT_NICHES": 1}
    assert trace.status.eq("COMPLETED").all()
    bundle = run / "output/25_cohort_niches/cohort_niches"
    readback(run, rows, bundle)
    assert hashes(source) == before
    return run, rows, bundle, trace


def test_fresh_two_specimen_fit_broadcast_export_and_unchanged_resume(fitted, dataset):
    run, rows, bundle, first = fitted
    source, _ = dataset
    before = hashes(source)
    result, trace = run_workflow(run, rows, label="unchanged", resume=True)
    assert_ok(result)
    assert len(trace) == 3 and trace.status.eq("CACHED").all()
    assert set(trace.hash) == set(first.hash)
    readback(run, rows, bundle)
    exports = [path for path in (run / "work").rglob(".command.sh") if "export_spatialdata.py" in path.read_text()]
    assert len(exports) == 2
    for command in exports:
        assert "--cohort-niches 'cohort_niches_input'" in command.read_text()
        assert hashes(command.parent / "cohort_niches_input") == hashes(bundle)
    assert hashes(source) == before


def test_external_global_bundle_broadcast_resume_and_same_stat_corruption(fitted, tmp_path):
    _, original, original_bundle, _ = fitted
    rows = list(reversed(original))
    bundle = tmp_path / "external_cohort"
    shutil.copytree(original_bundle, bundle)
    run = tmp_path / "external_run"
    result, first = run_workflow(run, rows, fit=False, bundle=bundle, label="first")
    assert_ok(result)
    assert len(first) == 2 and first.process.eq("EXPORT_SPATIALDATA").all()
    readback(run, rows, bundle)
    result, cached = run_workflow(run, rows, fit=False, bundle=bundle, label="unchanged", resume=True)
    assert_ok(result)
    assert cached.status.eq("CACHED").all() and set(cached.hash) == set(first.hash)
    old_outputs = hashes(run / "output/20_spatialdata")
    path = bundle / "cohort_niche_summary.json"
    old = path.read_bytes()
    changed = old.replace(b'"1.0.0"', b'"9.0.0"', 1)
    assert changed != old and len(changed) == len(old)
    stat = path.stat()
    path.write_bytes(changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    result, failed = run_workflow(run, rows, fit=False, bundle=bundle, label="corrupt", resume=True)
    assert result.returncode != 0
    assert "Cohort source SHA256 mismatch" in result.stdout + result.stderr
    assert len(failed) >= 2 and failed.status.isin(["FAILED", "ABORTED"]).all()
    assert set(failed.loc[failed.status.eq("FAILED"), "name"]) == {
        "EXPORT_SPATIALDATA (A)", "EXPORT_SPATIALDATA (B)"}
    assert set(failed.hash).isdisjoint(first.hash)
    assert hashes(run / "output/20_spatialdata") == old_outputs


def test_existing_bundle_refuses_a_changed_profile_registry(fitted, tmp_path):
    _, original, bundle, _ = fitted
    row = copy.deepcopy(original[0])
    changed = tmp_path / "same_cells_changed_registry"
    shutil.copytree(row["existing_profiles"], changed)
    manifest = changed / "cell_profiles_manifest.json"
    record = json.loads(manifest.read_text())
    record["fixture_registry_revision"] = "Different final registry, identical cell observations"
    manifest.write_text(json.dumps(record))
    row["existing_profiles"] = str(changed)
    result, failed = run_workflow(tmp_path / "wrong_profile", [row], fit=False, bundle=bundle)
    assert result.returncode != 0
    assert "Cohort source SHA256 mismatch" in result.stdout + result.stderr
    assert not failed.empty and failed.status.eq("FAILED").all()
    assert not list((tmp_path / "wrong_profile/output/20_spatialdata").rglob("spatialdata.zarr"))


def test_disabled_cohort_preserves_export_without_new_assignments(dataset, tmp_path):
    source, rows = dataset
    before = hashes(source)
    result, trace = run_workflow(tmp_path, rows, fit=False)
    assert_ok(result)
    assert len(trace) == 2 and trace.process.eq("EXPORT_SPATIALDATA").all()
    readback(tmp_path, rows)
    assert not (tmp_path / "output/25_cohort_niches").exists()
    assert hashes(source) == before


@pytest.mark.parametrize("feature_storage", ["table", "arrays"])
def test_main_atlas_subworkflow_fits_and_exports_exact_new_final_profiles(dataset, tmp_path, feature_storage):
    source, rows = dataset
    before = hashes(source)
    project = tmp_path / "project"
    project.mkdir()
    for name in ("lib", "modules", "subworkflows"):
        shutil.copytree(ROOT / name, project / name)
    for name in ("bin", "resources"):
        (project / name).symlink_to(ROOT / name, target_is_directory=True)
    shutil.copyfile(ROOT / "nextflow.config", project / "nextflow.config")
    workflow = project / "probe.nf"
    workflow.write_text("""import groovy.json.JsonSlurper
nextflow.enable.dsl=2
include { RUN_CELL_PROFILE_ATLAS } from './subworkflows/run_cell_profile_atlas'
workflow {
    def records = new JsonSlurper().parse(file(params.cell_profile_samples).toFile())
    def source = { name -> Channel.fromList(records.collect { tuple(it.sample_id, file(it[name], checkIfExists:true)) }) }
    def keyed = { name -> Channel.fromList(records.collect {
        tuple("${it.sample_id}::standard", it.sample_id, 'standard', file(it[name], checkIfExists:true)) }) }
    RUN_CELL_PROFILE_ATLAS(source('image'), source('image'), source('labels'), source('objects'),
        source('shift'), source('resolution'), source('support'), keyed('support'), Channel.empty(),
        keyed('support'), Channel.empty(), Channel.empty(), keyed('tissue_geojson'), Channel.empty(),
        Channel.empty(), Channel.empty(), Channel.empty(), Channel.empty(), Channel.empty(),
        Channel.empty(), Channel.empty(),
        [uni2:false, markers:false, cellvit:false, hierarchy:false, grid:false, both:false,
         run_cytoplasm:false, have_final_domains:true, primary_variant:'standard', run_cluster_geojson:true],
        TaskRuntime.forArtifacts(params))
}
""")
    result, trace = run_workflow(tmp_path, rows, script=workflow, extra={
        "cell_profiles_domain_source": "refined", "cell_neighborhood_radii_um": "5", "cell_niche_max_k": 2,
        "cell_neighborhood_feature_storage": feature_storage, "cell_neighborhood_row_batch_size": 7,
        "cell_neighborhood_column_batch_size": 3})
    assert_ok(result)
    assert trace.process.value_counts().to_dict() == {
        "RUN_CELL_PROFILE_ATLAS:BUILD_SPATIAL_CELL_PROFILES": 2,
        "RUN_CELL_PROFILE_ATLAS:FIT_COHORT_NICHES": 1,
        "RUN_CELL_PROFILE_ATLAS:EXPORT_SPATIALDATA": 2}
    assert trace.status.eq("COMPLETED").all()
    final_rows = [{**row, "existing_profiles": str(tmp_path / "output/19_cell_profiles" / row["sample_id"] / "cell_profiles")}
                  for row in rows]
    bundle = tmp_path / "output/25_cohort_niches/cohort_niches"
    readback(tmp_path, final_rows, bundle)
    # This route must not accidentally fit/export the fixture's old registry.
    sources = json.loads((bundle / "cohort_niche_model.json").read_text())["sources"]
    assert {value["manifest_sha256"] for value in sources}.isdisjoint(
        {hashes(Path(row["existing_profiles"]))["cell_profiles_manifest.json"] for row in rows})
    assert hashes(source) == before
    if feature_storage == "arrays":
        from neighborhood_feature_io import FeatureColumns
        for row in final_rows:
            reader = FeatureColumns(row["existing_profiles"])
            assert reader.groups
            package = sd.SpatialData.read(tmp_path / "output/20_spatialdata" / row["sample_id"] / "spatialdata.zarr")
            table = package.tables["cells"]
            record = json.loads(table.uns["cellphenotyper"]["neighborhood_features_json"])
            assert record == package.attrs["cellphenotyper"]["neighborhood_features"]
            for group, source_group in reader.groups.items():
                columns = source_group["columns"]
                target = record["groups"][group]
                assert target["logical_columns"] == columns
                assert target["aggregation_definition"] == reader.group_definitions[group]
                assert not set(columns) & set(table.obs)
                np.testing.assert_array_equal(table.obsm[target["obsm"]], reader.read(slice(None), columns))
            reader.recheck()
        assert all(value.get("store_files") for value in sources)
        result, cached = run_workflow(tmp_path, rows, script=workflow, label="unchanged", resume=True, extra={
            "cell_profiles_domain_source": "refined", "cell_neighborhood_radii_um": "5", "cell_niche_max_k": 2,
            "cell_neighborhood_feature_storage": feature_storage, "cell_neighborhood_row_batch_size": 7,
            "cell_neighborhood_column_batch_size": 3})
        assert_ok(result)
        assert len(cached) == 5 and cached.status.eq("CACHED").all()
        assert set(cached.hash) == set(trace.hash)


@pytest.mark.parametrize("fit,export,script,extra,diagnostic", [
    (True, True, "cell_profiles.nf", {}, "Choose cohort_niches_enable or cohort_niches_bundle"),
    (False, False, "cell_profiles.nf", {}, "cohort_niches_bundle requires cell_profiles_spatialdata=true"),
    (False, True, "main.nf", {"cell_profiles_enable": False}, "cohort_niches_bundle requires cell_profiles_enable=true"),
])
def test_ignored_or_conflicting_external_bundle_fails_before_tasks(dataset, tmp_path, fit, export, script, extra, diagnostic):
    _, rows = dataset
    result, _ = run_workflow(tmp_path, rows, fit=fit, export=export,
        bundle=tmp_path / "not_read", script=ROOT / script, extra=extra)
    assert result.returncode != 0
    assert diagnostic in result.stdout + result.stderr
    assert not list((tmp_path / "work").rglob(".command.sh"))
