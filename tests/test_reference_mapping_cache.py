"""Actual small Nextflow cache/staging checks; no model inference or containers.

Deep caching deliberately reads input bytes. These tests retain size and mtime
when corrupting inputs, so ordinary metadata-based caching cannot detect them.
"""
import io
import json
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("spatialdata")
from test_reference_mapping_workflow import atlas_cli, atlas_python, query_specimen, reference_profile
from cell_profile_io import sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cache_specimen(tmp_path):
    if not shutil.which("nextflow"):
        pytest.skip("Nextflow unavailable")
    data = tmp_path / "data"
    data.mkdir()
    row = query_specimen(data / "query", "cache_sample")
    row.pop("hierarchy")  # This check needs only two cells and a tiny image.
    query = Path(row["existing_profiles"])
    reference = data / "independent_reference"
    groups = reference_profile(reference, query, "cell")
    atlas_parent = data / "frozen"
    atlas_parent.mkdir()
    frozen = atlas_parent / query.name  # Deliberate query/atlas basename collision.
    atlas_cli("build", "--profiles", reference, "--outdir", frozen, "--version", "cache-test-v1",
              "--feature-groups", *groups, "--label-column", "tissue_domain")
    return data, row, query, frozen


def configure_run(directory, row, *, frozen=None, export=False):
    directory.mkdir()
    samples = directory / "samples.json"
    samples.write_text(json.dumps([row]))
    params = {"cell_profile_samples": str(samples), "outdir_base": str(directory / "output"),
              "cell_profiles_spatialdata": export, "spatialdata_python": sys.executable,
              "cell_atlas_python": atlas_python(), "spatialdata_tile_size": 16,
              "cell_profiles_cpus": 1, "cell_profiles_memory_gb": 2,
              "cell_reference_atlas": str(frozen) if frozen is not None else "",
              "region_reference_atlas": ""}
    (directory / "params.json").write_text(json.dumps(params))


def run_once(directory, label, *, resume=False, workflow=None):
    command = [shutil.which("nextflow"), "-log", str(directory / f"{label}.log"),
               "run", str(workflow or ROOT / "cell_profiles.nf"), "-params-file", str(directory / "params.json"),
               "-ansi-log", "false", "-work-dir", str(directory / "work"),
               "-with-trace", str(directory / f"{label}.trace.tsv")]
    if resume:
        command.append("-resume")
    return subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, NXF_OFFLINE="true", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))


def task_trace(directory, label, process, *, failed=False):
    table = pd.read_csv(directory / f"{label}.trace.tsv", sep="\t", keep_default_na=False)
    selected = table[table["name"].str.contains(process, regex=False)]
    if failed:
        # Production compute_medium retries one failed task; neither attempt
        # may be a cached success after source corruption.
        assert len(selected) in (1, 2) and selected.status.eq("FAILED").all(), table.to_string()
    else:
        assert len(selected) == 1, table.to_string()
    return selected.iloc[-1]


def assert_ok(result):
    assert result.returncode == 0, result.stdout + result.stderr


def corrupt_without_metadata_change(path, *, csv=False):
    previous = path.stat()
    directory_mtime = path.parent.stat().st_mtime_ns
    old = path.read_bytes()
    if csv:
        new = old.replace(b"reference_status", b"reference_statuz", 1)
    else:
        values = np.load(path, allow_pickle=False)
        values[0, 0] += np.float32(.125)
        stream = io.BytesIO()
        np.save(stream, values, allow_pickle=False)
        new = stream.getvalue()
    assert old != new and len(old) == len(new)
    path.write_bytes(new)
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert path.stat().st_size == previous.st_size
    assert path.stat().st_mtime_ns == previous.st_mtime_ns
    assert path.parent.stat().st_mtime_ns == directory_mtime


def test_same_basename_mapping_stages_separately_and_resume_checks_actual_source_bytes(cache_specimen, tmp_path):
    _, row, query, frozen = cache_specimen
    assert query.name == frozen.name and query != frozen
    run = tmp_path / "mapping_run"
    configure_run(run, row, frozen=frozen)
    assert_ok(run_once(run, "first"))
    first = task_trace(run, "first", "MAP_CELL_REFERENCE_ATLAS")
    assert first.status == "COMPLETED"
    commands = [path for path in (run / "work").rglob(".command.sh") if "cell_reference_atlas.py" in path.read_text()]
    assert len(commands) == 1
    command = commands[0]
    assert "--query 'query_profiles' --atlas 'frozen_reference'" in command.read_text()
    assert (command.parent / "query_profiles").resolve() == query.resolve()
    assert (command.parent / "frozen_reference").resolve() == frozen.resolve()
    receipt = run / "output/21_reference_mapping/cache_sample/reference_assignments.mapping.json"
    receipt_hash = sha256_file(receipt)

    assert_ok(run_once(run, "unchanged", resume=True))
    unchanged = task_trace(run, "unchanged", "MAP_CELL_REFERENCE_ATLAS")
    assert unchanged.status == "CACHED" and unchanged["hash"] == first["hash"]

    corrupt_without_metadata_change(query / "local.npy")
    result = run_once(run, "corrupt", resume=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "Source feature block hash differs from manifest: local" in result.stdout + result.stderr
    failed = task_trace(run, "corrupt", "MAP_CELL_REFERENCE_ATLAS", failed=True)
    assert failed.status == "FAILED" and failed["hash"] != first["hash"]
    assert sha256_file(receipt) == receipt_hash  # Failed work never publishes a new completion.


def test_region_mapping_resume_checks_unchanged_metadata_directory_payload(tmp_path):
    if not shutil.which("nextflow"):
        pytest.skip("Nextflow unavailable")
    from test_region_reference_workflow import region_profiles
    query = tmp_path / "query/region_profiles"
    reference = tmp_path / "reference/region_profiles"
    region_profiles(query, "query_region", [0, 100, np.nan])
    region_profiles(reference, "independent_region", [-.2, 0, .2])
    path = query / "region_profiles_manifest.json"
    manifest = json.loads(path.read_text())
    for block in manifest["feature_blocks"].values():
        block["sha256"] = sha256_file(query / block["path"])
    path.write_text(json.dumps(manifest))
    frozen = tmp_path / "frozen/region_profiles"
    atlas_cli("build", "--profiles", reference, "--outdir", frozen, "--version", "region-cache-v1",
              "--feature-groups", "local", "context", "--label-column", "subdomain_id")
    run = tmp_path / "region_run"
    (run / "modules").mkdir(parents=True)
    (run / "lib").mkdir()
    shutil.copyfile(ROOT / "modules/map_region_reference_atlas.nf", run / "modules/map_region_reference_atlas.nf")
    for library in (ROOT / "lib").glob("*.groovy"):
        shutil.copyfile(library, run / "lib" / library.name)
    (run / "bin").symlink_to(ROOT / "bin", target_is_directory=True)
    params = {"query": str(query), "frozen": str(frozen), "outdir_base": str(run / "output"),
              "cell_atlas_python": atlas_python(), "cell_profiles_cpus": 1, "cell_profiles_memory_gb": 2,
              "_executor_max_cpus": 2, "_executor_max_memory_gb": 4}
    (run / "params.json").write_text(json.dumps(params))
    workflow = run / "workflow.nf"
    workflow.write_text("""nextflow.enable.dsl=2
include { MAP_REGION_REFERENCE_ATLAS } from './modules/map_region_reference_atlas'
workflow {
    def query = file(params.query, checkIfExists: true)
    def reference = file(params.frozen, checkIfExists: true)
    def inputs = Channel.of(tuple('query_region::tile', 'query_region', 'tile', query, reference))
        .map { key, id, variant, profiles, atlas ->
            tuple(key, id, variant, profiles, atlas, PipelineHelpers.atlasTaskFingerprint('region_reference_mapping', projectDir, [profiles, atlas]))
        }
    def plan = TaskRuntime.create(HardwarePolicy.resolve(params, 2, 4, false, 0d), 'cpu')
    MAP_REGION_REFERENCE_ATLAS(inputs, plan)
}
""")
    assert_ok(run_once(run, "first", workflow=workflow))
    first = task_trace(run, "first", "MAP_REGION_REFERENCE_ATLAS")
    assert first.status == "COMPLETED"
    assert_ok(run_once(run, "unchanged", resume=True, workflow=workflow))
    unchanged = task_trace(run, "unchanged", "MAP_REGION_REFERENCE_ATLAS")
    assert unchanged.status == "CACHED" and unchanged["hash"] == first["hash"]
    corrupt_without_metadata_change(query / "local.npy")
    result = run_once(run, "corrupt", resume=True, workflow=workflow)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "Source feature block hash differs from manifest: local" in result.stdout + result.stderr
    failed = task_trace(run, "corrupt", "MAP_REGION_REFERENCE_ATLAS", failed=True)
    assert failed["hash"] != first["hash"]


@pytest.mark.parametrize("artifact", ["profile_feature", "mapping_csv"])
def test_export_resume_cannot_cache_unchanged_metadata_source_or_mapping_corruption(cache_specimen, tmp_path, artifact):
    data, row, query, frozen = cache_specimen
    mapping_dir = data / "mapping"
    mapping_dir.mkdir()
    assignments = mapping_dir / "reference_assignments.csv"
    atlas_cli("map", "--query", query, "--atlas", frozen, "--output", assignments)
    row["cell_reference_mapping"] = str(assignments.with_suffix(".mapping.json"))
    run = tmp_path / "export_run"
    configure_run(run, row, export=True)
    assert_ok(run_once(run, "first"))
    first = task_trace(run, "first", "EXPORT_SPATIALDATA")
    assert first.status == "COMPLETED"
    store = run / "output/20_spatialdata/cache_sample/spatialdata.zarr"
    assert store.is_dir()
    old_store_hashes = {str(path.relative_to(store)): sha256_file(path) for path in store.rglob("*") if path.is_file()}

    assert_ok(run_once(run, "unchanged", resume=True))
    unchanged = task_trace(run, "unchanged", "EXPORT_SPATIALDATA")
    assert unchanged.status == "CACHED" and unchanged["hash"] == first["hash"]

    corrupt_without_metadata_change(query / "local.npy" if artifact == "profile_feature" else assignments,
                                    csv=artifact == "mapping_csv")
    result = run_once(run, "corrupt", resume=True)
    assert result.returncode != 0, result.stdout + result.stderr
    diagnostic = (result.stdout + result.stderr).lower()
    if artifact == "mapping_csv":
        assert "reference assignments hash differs" in diagnostic, diagnostic
    else:
        assert "hash" in diagnostic and "local" in diagnostic, diagnostic
    failed = task_trace(run, "corrupt", "EXPORT_SPATIALDATA", failed=True)
    assert failed.status == "FAILED" and failed["hash"] != first["hash"]
    assert {str(path.relative_to(store)): sha256_file(path) for path in store.rglob("*") if path.is_file()} == old_store_hashes


def copy_atlas_project(destination):
    """Only the small workflow/code surface required by these real stages."""
    destination.mkdir()
    for name in ("cell_profiles.nf", "nextflow.config"):
        shutil.copyfile(ROOT / name, destination / name)
    for folder in ("modules", "lib"):
        shutil.copytree(ROOT / folder, destination / folder)
    (destination / "resources").mkdir()
    shutil.copytree(ROOT / "resources/empty_embeddings_placeholder", destination / "resources/empty_embeddings_placeholder")
    (destination / "bin").mkdir()
    for name in ("cell_reference_atlas.py", "reference_mapping_io.py", "export_spatialdata.py",
                 "integrate_measured_assay.py", "cell_profile_io.py", "profile_cell_morphology.py",
                 "link_cell_tissue_hierarchy.py", "cell_morphology_io.py", "cohort_niche_io.py", "neighborhood_feature_io.py",
                 "activate_source_python.sh"):
        shutil.copyfile(ROOT / "bin" / name, destination / "bin" / name)


@pytest.mark.parametrize("stage", ["mapping", "export"])
def test_actual_code_only_edit_invalidates_upstream_mapping_and_export_cache(cache_specimen, tmp_path, stage):
    data, row, query, frozen = cache_specimen
    project = tmp_path / "copied_project"
    copy_atlas_project(project)
    script = project / "bin" / ("cell_reference_atlas.py" if stage == "mapping" else "export_spatialdata.py")
    script.write_text(script.read_text() + '\nprint("CACHE_CODE_ONLY_FIRST")\n')
    if stage == "export":
        directory = data / "mapping"
        directory.mkdir()
        assignments = directory / "reference_assignments.csv"
        atlas_cli("map", "--query", query, "--atlas", frozen, "--output", assignments)
        row["cell_reference_mapping"] = str(assignments.with_suffix(".mapping.json"))
    source_hashes = {str(path): sha256_file(path) for path in data.rglob("*") if path.is_file()}
    run = tmp_path / "code_run"
    configure_run(run, row, frozen=frozen if stage == "mapping" else None, export=stage == "export")
    workflow = project / "cell_profiles.nf"
    process = "MAP_CELL_REFERENCE_ATLAS" if stage == "mapping" else "EXPORT_SPATIALDATA"
    assert_ok(run_once(run, "first", workflow=workflow))
    first = task_trace(run, "first", process)
    assert first.status == "COMPLETED"
    assert_ok(run_once(run, "unchanged", resume=True, workflow=workflow))
    unchanged = task_trace(run, "unchanged", process)
    assert unchanged.status == "CACHED" and unchanged["hash"] == first["hash"]

    # No input-data change and no file size/mtime change: only code bytes.
    previous = script.stat()
    original = script.read_bytes()
    changed = original.replace(b"CACHE_CODE_ONLY_FIRST", b"CACHE_CODE_ONLY_AFTER")
    assert original != changed and len(original) == len(changed)
    script.write_bytes(changed)
    os.utime(script, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert script.stat().st_size == previous.st_size and script.stat().st_mtime_ns == previous.st_mtime_ns
    assert_ok(run_once(run, "code_changed", resume=True, workflow=workflow))
    rerun = task_trace(run, "code_changed", process)
    assert rerun.status == "COMPLETED" and rerun["hash"] != first["hash"]
    logs = [path for path in (run / "work").rglob(".command.out") if "CACHE_CODE_ONLY_AFTER" in path.read_text()]
    assert len(logs) == 1  # The new actual producer code executed.
    assert {str(path): sha256_file(path) for path in data.rglob("*") if path.is_file()} == source_hashes


def test_actual_same_stat_helper_edit_ignores_stale_python_bytecode(cache_specimen, tmp_path):
    """A new Nextflow hash is insufficient if Python executes old helper .pyc."""
    from test_spatialdata_neighborhood_features import add_store
    import spatialdata as sd

    data, row, query, _ = cache_specimen
    add_store(query)
    project = tmp_path / "copied_project"
    copy_atlas_project(project)
    helper = project / "bin/neighborhood_feature_io.py"
    helper.write_text(helper.read_text() + '\nprint("CACHE_NBH_HELPER_FIRST")\n')
    # Deliberately retain the exact stale timestamp/size-based bytecode that
    # caused the real reproduction. The task must ignore, not delete, this file.
    bytecode = Path(py_compile.compile(str(helper), doraise=True,
                                      invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP))
    bytecode_hash = sha256_file(bytecode)
    source_hashes = {str(path): sha256_file(path) for path in data.rglob("*") if path.is_file()}
    run = tmp_path / "helper_run"
    configure_run(run, row, export=True)
    workflow = project / "cell_profiles.nf"
    assert_ok(run_once(run, "first", workflow=workflow))
    first = task_trace(run, "first", "EXPORT_SPATIALDATA")
    assert first.status == "COMPLETED"
    assert_ok(run_once(run, "unchanged", resume=True, workflow=workflow))
    unchanged = task_trace(run, "unchanged", "EXPORT_SPATIALDATA")
    assert unchanged.status == "CACHED" and unchanged["hash"] == first["hash"]
    previous = helper.stat()
    original = helper.read_bytes()
    changed = original.replace(b"CACHE_NBH_HELPER_FIRST", b"CACHE_NBH_HELPER_AFTER")
    assert original != changed and len(original) == len(changed)
    helper.write_bytes(changed)
    os.utime(helper, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert helper.stat().st_size == previous.st_size and helper.stat().st_mtime_ns == previous.st_mtime_ns
    assert_ok(run_once(run, "helper_changed", resume=True, workflow=workflow))
    rerun = task_trace(run, "helper_changed", "EXPORT_SPATIALDATA")
    assert rerun.status == "COMPLETED" and rerun["hash"] != first["hash"]
    outputs = [path for path in (run / "work").rglob(".command.out")
               if "CACHE_NBH_HELPER_AFTER" in path.read_text()]
    assert len(outputs) == 1  # Prove the new helper actually executed.
    assert sha256_file(bytecode) == bytecode_hash
    assert b"CACHE_NBH_HELPER_FIRST" in bytecode.read_bytes()
    prefixes = list((run / "work").glob("*/*/.cellphenotyper-pycache.*"))
    assert len(prefixes) == 2 and all(not list(path.iterdir()) for path in prefixes)
    assert {str(path): sha256_file(path) for path in data.rglob("*") if path.is_file()} == source_hashes
    store = run / "output/20_spatialdata/cache_sample/spatialdata.zarr"
    table = sd.SpatialData.read(store).tables["cells"]
    assert table.obsm["neighborhood_group_0000"].dtype == np.dtype("float64")
    assert table.obs_names.tolist() == pd.read_parquet(query / "cell_profiles.parquet").cell_uid.tolist()


def test_streamed_content_fingerprint_nested_changes_names_symlinks_and_placeholders(tmp_path):
    if not shutil.which("nextflow"):
        pytest.skip("Nextflow unavailable")
    root = tmp_path / "original"
    (root / "nested").mkdir(parents=True)
    np.save(root / "nested/values.npy", np.array([[1., 2.]], np.float32))
    (root / "empty").mkdir()
    copied, changed, renamed = [tmp_path / name for name in ("copied", "changed", "renamed")]
    for target in (copied, changed, renamed):
        shutil.copytree(root, target, copy_function=shutil.copy2)
    corrupt_without_metadata_change(changed / "nested/values.npy")
    (renamed / "nested/values.npy").rename(renamed / "nested/different.npy")
    alias = tmp_path / "root_alias"
    alias.symlink_to(root, target_is_directory=True)
    placeholder = tmp_path / "placeholder"
    placeholder.write_text("optional input unavailable")
    escape, cycle = tmp_path / "escape", tmp_path / "cycle"
    escape.mkdir()
    cycle.mkdir()
    (escape / "foreign").symlink_to(root, target_is_directory=True)
    (cycle / "back").symlink_to(cycle, target_is_directory=True)
    project = tmp_path / "probe"
    (project / "lib").mkdir(parents=True)
    shutil.copyfile(ROOT / "lib/PipelineHelpers.groovy", project / "lib/PipelineHelpers.groovy")
    paths = {key: str(value) for key, value in locals().copy().items()
             if key in {"root", "copied", "changed", "renamed", "alias", "placeholder", "escape", "cycle"}}
    (project / "params.json").write_text(json.dumps(paths))
    (project / "probe.nf").write_text("""nextflow.enable.dsl=2
workflow {
    def original = PipelineHelpers.contentFingerprint([file(params.root)])
    assert original == PipelineHelpers.contentFingerprint([file(params.copied)])
    assert original == PipelineHelpers.contentFingerprint([file(params.alias)])
    assert original == PipelineHelpers.contentFingerprint([new nextflow.processor.TaskPath(file(params.alias), 'staged_root')])
    assert original != PipelineHelpers.contentFingerprint([file(params.changed)])
    assert original != PipelineHelpers.contentFingerprint([file(params.renamed)])
    assert PipelineHelpers.contentFingerprint([file(params.placeholder)]) ==~ /[0-9a-f]{64}/
    [params.escape, params.cycle].each { value ->
        def rejected = false
        try { PipelineHelpers.contentFingerprint([file(value)]) }
        catch (IllegalArgumentException expected) { rejected = true }
        assert rejected
    }
    println 'CONTENT_FINGERPRINT_CHECKS_PASSED'
}
""")
    result = subprocess.run([shutil.which("nextflow"), "run", str(project / "probe.nf"),
                             "-params-file", str(project / "params.json"), "-ansi-log", "false"],
                            cwd=project, env=dict(os.environ, NXF_OFFLINE="true"),
                            capture_output=True, text=True, timeout=45)
    assert_ok(result)
    assert "CONTENT_FINGERPRINT_CHECKS_PASSED" in result.stdout
