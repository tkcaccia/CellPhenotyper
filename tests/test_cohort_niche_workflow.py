"""Tiny real CPU cohort discovery through Nextflow; no learned-model inference."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from build_cell_profiles import build_profiles, parser
from assemble_spatial_cell_profiles import assemble


def hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def specimen(directory, sample):
    directory.mkdir()
    xy = [(x, y) for y in (3, 7, 11) for x in (3, 7, 11, 15)]
    xy += [(x, y) for y in (25, 40, 55) for x in (25, 35, 45, 55)]
    labels = np.zeros((64, 64), np.uint16)
    for label, (x, y) in enumerate(xy, 1):
        labels[y:y + 2, x:x + (2 if label <= 12 else 3)] = label
    image = np.full((64, 64, 3), (110, 75, 130), np.uint8)
    files = {name: directory / f"{name}.tif" for name in ("image", "labels", "support")}
    for name, values in (("image", image), ("labels", labels), ("support", np.ones((64, 64), np.uint8))):
        tifffile.imwrite(files[name], values)
    files["objects"] = directory / "objects.csv"
    pd.DataFrame({"label": [str(i) for i in range(1, 25)],
                  "x": [x + (.5 if i < 12 else 1.) for i,(x,y) in enumerate(xy)], "y": [y + .5 for x, y in xy],
                  "xmin": [x for x, y in xy], "ymin": [y for x, y in xy],
                  "xmax": [x + (2 if i < 12 else 3) for i,(x,y) in enumerate(xy)], "ymax": [y + 2 for x, y in xy]}).to_csv(files["objects"], index=False)
    files["shift"] = directory / "shift.json"
    files["shift"].write_text(json.dumps({"crop_size": {"width": 64, "height": 64},
        "offset_crop_to_original": {"dx": 100, "dy": 200}}))
    files["resolution"] = directory / "resolution.json"
    files["resolution"].write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5}))
    base = directory / "base"
    args = ["--sample-id", sample, "--outdir", str(base)]
    for name, flag in (("objects", "--objects"), ("image", "--image"), ("labels", "--labels"),
                       ("support", "--tissue-mask"), ("shift", "--shift"), ("resolution", "--resolution-json")):
        args += [flag, str(files[name])]
    build_profiles(parser().parse_args(args))
    files["existing_profiles"] = directory / "cell_profiles"
    assemble(base, files["support"], files["shift"], files["resolution"], files["existing_profiles"],
             radii_um=(5.,), max_k=2)
    return {"sample_id": sample, **{key: str(value) for key, value in files.items()}}


def run_nextflow(directory, rows, *, script=None, resume=False, enable=True, name="run", overrides=None):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow executable unavailable")
    samples = directory / "samples.json"
    samples.write_text(json.dumps(rows))
    params = {"cell_profile_samples": str(samples), "outdir_base": str(directory / "output"),
              "cell_profiles_spatialdata": False, "cell_atlas_python": sys.executable,
              "cell_profiles_cpus": 1, "cell_profiles_memory_gb": 2,
              "cohort_niches_enable": enable, "cohort_niches_max_k": 2,
              "cohort_niches_fixed_k": 2, "cohort_niches_repeats": 2,
              "cohort_niches_fit_limit": 100, "cohort_niches_max_working_mb": 64,
              "cell_profiles_domain_source": "none", "cell_neighborhood_radii_um": "5",
              "cell_niche_max_k": 2}
    params.update(overrides or {})
    params_file = directory / "params.json"
    params_file.write_text(json.dumps(params))
    trace_config = directory / "trace.config"
    trace_config.write_text("trace.fields = 'task_id,hash,process,name,status,exit,cpus,memory'\ntrace.raw = true\n")
    trace = directory / f"{name}.tsv"
    command = [nextflow, "-log", str(directory / f"{name}.log"), "run", str(script or ROOT / "cell_profiles.nf"),
               "-c", str(trace_config), "-ansi-log", "false", "-params-file", str(params_file), "-work-dir", str(directory / "work"),
               "-with-trace", str(trace)]
    if resume:
        command.append("-resume")
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=120,
        env=dict(os.environ, NXF_OFFLINE="true", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    return result, pd.read_csv(trace, sep="\t") if trace.exists() else None


def assert_cohort(directory, rows):
    output = directory / "output/25_cohort_niches/cohort_niches"
    for name in ("cohort_niche_model.json", "cohort_niche_assignments.parquet",
                 "cohort_niche_summary.json", "cohort_niches_completion.json"):
        assert (output / name).is_file(), name
    result = pd.read_parquet(output / "cohort_niche_assignments.parquet")
    expected = pd.concat([pd.read_parquet(Path(row["existing_profiles"]) / "cell_profiles.parquet")
                          for row in sorted(rows, key=lambda row: row["sample_id"])], ignore_index=True)
    for column in ("sample_id", "cell_id", "cell_uid"):
        assert result[column].tolist() == expected[column].tolist()
    assert len(result) == 48 and result.cell_uid.is_unique
    return result


def test_existing_two_specimens_run_real_cohort_and_resume_without_mutation(tmp_path):
    # Reverse input order and intentionally share directory basename/cell IDs.
    rows = [specimen(tmp_path / "b", "B"), specimen(tmp_path / "a", "A")]
    before = {row["sample_id"]: hashes(Path(row["existing_profiles"])) for row in rows}
    result, trace = run_nextflow(tmp_path, rows, name="first")
    assert result.returncode == 0, result.stdout + result.stderr
    assert trace.process.tolist() == ["FIT_COHORT_NICHES"]
    assert trace.status.tolist() == ["COMPLETED"]
    assert trace.cpus.tolist() == [1] and trace.memory.tolist() == [2 * 1024**3]
    first = assert_cohort(tmp_path, rows)
    assert not (tmp_path / "output/19_cell_profiles").exists()
    assert not (tmp_path / "output/20_spatialdata").exists()
    result, repeated = run_nextflow(tmp_path, rows, resume=True, name="unchanged")
    assert result.returncode == 0, result.stdout + result.stderr
    assert repeated.status.tolist() == ["CACHED"]
    assert repeated.hash.tolist() == trace.hash.tolist()
    pd.testing.assert_frame_equal(first, assert_cohort(tmp_path, rows))
    assert before == {row["sample_id"]: hashes(Path(row["existing_profiles"])) for row in rows}
    # A same-stat directory-child corruption must rerun and fail validation,
    # never return the successful cohort cached for the old source bytes.
    summary = Path(rows[0]["existing_profiles"]) / "neighborhood_summary.json"
    original = summary.read_bytes()
    changed = original.replace(b'"2.0.0"', b'"9.0.0"', 1)
    assert changed != original and len(changed) == len(original)
    stat = summary.stat()
    summary.write_bytes(changed)
    os.utime(summary, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    result, corrupted = run_nextflow(tmp_path, rows, resume=True, name="same_stat_corruption")
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stdout + result.stderr
    assert not corrupted.empty and corrupted.status.eq("FAILED").all()
    assert set(corrupted.hash).isdisjoint(trace.hash)  # retries must not reuse the successful old task


def test_full_atlas_subworkflow_assembles_then_fits_separate_cohort(tmp_path):
    rows = [specimen(tmp_path / "a", "A"), specimen(tmp_path / "b", "B")]
    project = tmp_path / "project"
    project.mkdir()
    for name in ("lib", "bin", "modules", "subworkflows", "resources"):
        shutil.copytree(ROOT / name, project / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(ROOT / "nextflow.config", project / "nextflow.config")
    workflow = project / "probe.nf"
    workflow.write_text("""import groovy.json.JsonSlurper
nextflow.enable.dsl=2
include { RUN_CELL_PROFILE_ATLAS } from './subworkflows/run_cell_profile_atlas'
workflow {
    def records = new JsonSlurper().parse(file(params.cell_profile_samples).toFile())
    def source = { name -> Channel.fromList(records.collect { tuple(it.sample_id, file(it[name], checkIfExists:true)) }) }
    RUN_CELL_PROFILE_ATLAS(source('image'), source('image'), source('labels'), source('objects'),
        source('shift'), source('resolution'), source('support'), Channel.empty(), Channel.empty(),
        Channel.empty(), Channel.empty(), Channel.empty(), Channel.empty(), Channel.empty(),
        Channel.empty(), Channel.empty(), Channel.empty(), Channel.empty(), Channel.empty(),
        Channel.empty(), Channel.empty(),
        [uni2:false, markers:false, cellvit:false, hierarchy:false, grid:false, both:false, run_cytoplasm:false],
        TaskRuntime.forArtifacts(params))
}
""")
    result, trace = run_nextflow(tmp_path, rows, script=workflow)
    assert result.returncode == 0, result.stdout + result.stderr
    assert trace.process.value_counts().to_dict() == {
        "RUN_CELL_PROFILE_ATLAS:BUILD_SPATIAL_CELL_PROFILES": 2,
        "RUN_CELL_PROFILE_ATLAS:FIT_COHORT_NICHES": 1}
    assert trace.status.eq("COMPLETED").all()
    assembled = [{"sample_id": row["sample_id"], "existing_profiles": str(tmp_path / "output/19_cell_profiles" / row["sample_id"] / "cell_profiles")} for row in rows]
    assert_cohort(tmp_path, assembled)
    model=json.loads((tmp_path/'output/25_cohort_niches/cohort_niches/cohort_niche_model.json').read_text())
    assert {'morphology','nuclear_texture'} <= set(model['source_feature_definitions'])
    assert {'own:morphology','morphology'} <= set(model['discovery']['scaling'])


def test_optional_cohort_is_absent_by_default_and_rejects_one_specimen(tmp_path):
    rows = [specimen(tmp_path / "a", "A")]
    result, _ = run_nextflow(tmp_path, rows, enable=False, name="disabled")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "output/25_cohort_niches").exists()
    result, _ = run_nextflow(tmp_path, rows, name="single")
    assert result.returncode != 0
    assert "at least two distinct specimens" in result.stdout + result.stderr
    assert not list((tmp_path / "work").rglob(".command.sh"))


def test_main_rejects_cohort_without_profiles_before_any_tasks(tmp_path):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow executable unavailable")
    result = subprocess.run([nextflow, "-log", str(tmp_path / "run.log"), "run", str(ROOT / "main.nf"),
        "-ansi-log", "false", "-work-dir", str(tmp_path / "work"), "--cohort_niches_enable", "true",
        "--cell_profiles_enable", "false", "--outdir_base", str(tmp_path / "output")],
        cwd=tmp_path, env=dict(os.environ, NXF_OFFLINE="true"), capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "cohort_niches_enable requires cell_profiles_enable=true" in result.stdout + result.stderr
    assert not list((tmp_path / "work").rglob(".command.sh"))


@pytest.mark.parametrize("overrides,expected", [
    ({"cohort_niches_fit_limit": 9}, "fit_limit>=10"),
    ({"cohort_niches_max_working_mb": 4096}, "exceeds the task memory allocation"),
    ({"cell_neighborhood_row_batch_size": 0}, "cell_neighborhood_row_batch_size must be a positive integer"),
    ({"cell_neighborhood_row_batch_size": "7; false"}, "cell_neighborhood_row_batch_size must be a positive integer"),
])
def test_invalid_cohort_working_limits_fail_before_the_python_task(tmp_path, overrides, expected):
    rows = [specimen(tmp_path / "a", "A"), specimen(tmp_path / "b", "B")]
    result, _ = run_nextflow(tmp_path, rows, overrides=overrides)
    assert result.returncode != 0
    assert expected in result.stdout + result.stderr
    assert not list((tmp_path / "work").rglob(".command.sh"))
