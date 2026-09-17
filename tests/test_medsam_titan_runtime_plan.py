"""Execute actual Nextflow task commands with model-free Python recorders."""

import csv
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULES = ("refine_grown_tissue_medsam", "extract_titan_section_embedding")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def probe(tmp_path, runtime, medsam_device="auto", targets=("medsam", "titan"), titan_gpu=0):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    (tmp_path / "modules").mkdir()
    (tmp_path / "lib").mkdir()
    (tmp_path / "bin").symlink_to(ROOT / "bin", target_is_directory=True)
    for name in MODULES:
        shutil.copy2(ROOT / "modules" / f"{name}.nf", tmp_path / "modules" / f"{name}.nf")
    for name in ("TaskRuntime.groovy", "HostRuntime.groovy", "HardwarePolicy.groovy", "PipelineHelpers.groovy", "ProcessCode.groovy"):
        shutil.copy2(ROOT / "lib" / name, tmp_path / "lib" / name)
    recorder = tmp_path / "record_model_arguments.py"
    recorder.write_text("""import argparse, json, os, sys
from pathlib import Path
p = argparse.ArgumentParser()
for name in ['sample-id', 'out', 'outdir', 'uncertainty-out', 'provenance-out', 'preview', 'medsam-device']:
    p.add_argument('--' + name)
p.add_argument('--max-workers', type=int)
p.add_argument('--gpu', type=int)
args, _ = p.parse_known_args()
record = vars(args) | {'argv': sys.argv[1:], 'used_model': False,
    'threads': {name: os.environ.get(name) for name in
        ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}}
if args.out:
    for name in [args.out, args.uncertainty_out, args.provenance_out, args.preview]:
        Path(name).write_text('model-free placeholder; not a raster')
    Path(args.out + '.provenance.json').write_text(json.dumps({'stub': True}))
    Path(args.sample_id + '_medsam_summary.json').write_text(json.dumps(record))
else:
    out = Path(args.outdir); out.mkdir()
    (out / 'titan_embedding.csv').write_text('model-free placeholder; no embedding')
    (out / 'titan_patch_features.h5').write_text('model-free placeholder; not HDF5')
    (out / 'titan_metadata.json').write_text(json.dumps(record))
""")
    model_dir = tmp_path / "local model snapshot"
    model_dir.mkdir()
    for name in ("image.tif", "seed.tif", "grown.tif", "tissue.tif", "kodama.png", "uncertainty.tif", "shift.json", "section.json"):
        (tmp_path / name).write_text("model-free command fixture\n")
    (tmp_path / "resolution.json").write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5}))
    combined = "\n".join((ROOT / "modules" / f"{name}.nf").read_text() for name in MODULES)
    params = {name: 0 for name in re.findall(r"params\.([A-Za-z_][A-Za-z_0-9]*)", combined)}
    params.update({
        "outdir_base": str(tmp_path / "published"), "publish_dir_mode": "copy",
        # Module include scopes snapshot these deliberately conflicting values.
        "compute_device": "auto", "_resolved_compute_device": "cpu", "hardware_profile": "auto",
        "_executor_max_cpus": 1, "_executor_max_memory_gb": 1,
        "medsam_refine_cpus": 1, "medsam_refine_memory_gb": 1, "medsam_refine_max_workers": 1,
        "medsam_refine_max_auto_workers": 4, "medsam_refine_auto_hardware": True,
        "titan_cpus": 1, "titan_memory_gb": 1, "hardware_auto": True,
        "grown_tissue_refine_script": recorder.name, "titan_script": recorder.name,
        "medsam_refine_time": "1m", "titan_time": "1m", "medsam_device": medsam_device,
        "medsam_checkpoint": str(model_dir / "medsam.pth"), "medsam_large_image_mode": "stream",
        "medsam_force_core_preservation": True, "medsam_image_guided_internal_refine": True,
        "medsam_pre_boundary_competition": True, "medsam_pre_boundary_radius": 64,
        "medsam_pre_boundary_downsample": 4, "medsam_pre_boundary_iterations": 16,
        "medsam_pre_boundary_initial_temperature": 2.0, "medsam_pre_boundary_final_temperature": 0.05,
        "medsam_pre_boundary_data_weight": 1.0, "medsam_pre_boundary_smoothness_weight": 0.3,
        "medsam_pre_boundary_edge_beta": 0.7, "medsam_pre_boundary_radius_um": None,
        "medsam_appearance_refine": True, "medsam_appearance_multiclass": False,
        "medsam_appearance_clusters": 24, "medsam_cluster_tile_size": 4096,
        "medsam_cluster_tile_overlap": 512, "medsam_core_erosion_um": 4.5,
        "input_resolution_override_mpp": .273774374855905,
        "titan_model": str(model_dir), "titan_revision": "locked-test-revision",
        "titan_cache_dir": str(tmp_path / "hf"), "titan_offline": True,
        "titan_target_mpp": .5, "titan_default_mpp": .5, "titan_patch_size": 512,
        "titan_min_tissue_coverage": .2, "titan_batch_size": 7, "titan_gpu": titan_gpu,
    })
    (tmp_path / "params.json").write_text(json.dumps(params))
    (tmp_path / "runtime.json").write_text(json.dumps(runtime))
    includes = {
        "medsam": "include { REFINE_GROWN_TISSUE_MEDSAM } from './modules/refine_grown_tissue_medsam'",
        "auxiliary": "include { REFINE_GROWN_TISSUE_MEDSAM as REFINE_CELL_AUXILIARY_MEDSAM } from './modules/refine_grown_tissue_medsam'",
        "titan": "include { EXTRACT_TITAN_SECTION_EMBEDDING } from './modules/extract_titan_section_embedding'",
    }
    medsam_tuple = "tuple('sample::standard', 'sample', 'standard', file('image.tif'), file('seed.tif'), file('grown.tif'), file('tissue.tif'), file('kodama.png'), file('resolution.json'), file('uncertainty.tif'))"
    calls = {
        "medsam": f"REFINE_GROWN_TISSUE_MEDSAM(Channel.of({medsam_tuple}), runtime_plan)",
        "auxiliary": f"REFINE_CELL_AUXILIARY_MEDSAM(Channel.of({medsam_tuple}), runtime_plan)",
        "titan": "EXTRACT_TITAN_SECTION_EMBEDDING(Channel.of(tuple('sample::standard', 'sample', 'standard', file('image.tif'), file('grown.tif'), file('shift.json'), file('section.json'))), runtime_plan)",
    }
    (tmp_path / "main.nf").write_text("""nextflow.enable.dsl=2
import groovy.json.JsonSlurperClassic
""" + "\n".join(includes[name] for name in targets) + """
workflow {
  def runtime_plan = new JsonSlurperClassic().parseText(file('runtime.json').text)
  params._resolved_compute_device = runtime_plan.compute_device
  params.hardware_profile = runtime_plan.profile
  params.medsam_refine_cpus = 99
  params.medsam_refine_max_workers = 99
  params.titan_cpus = 99
""" + "\n".join(calls[name] for name in targets) + "\n}\n")
    (tmp_path / "nextflow.config").write_text(f"""process {{
  executor = 'local'
  maxForks = 1
  beforeScript = '''
  python() {{ {shlex.quote(sys.executable)} "$@"; }}
  export -f python
  '''
}}
executor.queueSize = 1
trace.fields = 'task_id,name,status,exit,cpus,memory'
""")
    env = {key: value for key, value in os.environ.items() if not key.startswith("HF_")}
    result = subprocess.run([
        nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "main.nf"),
        "-params-file", str(tmp_path / "params.json"), "-ansi-log", "false",
        "-with-trace", str(tmp_path / "trace.tsv"), "-work-dir", str(tmp_path / "work"),
    ], cwd=tmp_path, env={**env, "NXF_OFFLINE": "true"}, text=True, capture_output=True, timeout=60)
    scripts = {path.parent: path.read_text() for path in (tmp_path / "work").glob("*/*/.command.sh")}
    trace_file = tmp_path / "trace.tsv"
    with trace_file.open() as handle:
        trace = list(csv.DictReader(handle, delimiter="\t"))
    return result, scripts, trace


def plan(device="gpu", profile="aggressive", workers=8, cpus=2, memory=.5):
    return {
        "schema_version": 1, "compute_device": device, "profile": profile,
        "cpu_budget": cpus, "memory_budget_gb": memory,
        "stages": {"medsam_refine": {"cpus": 8, "memory_gb": 30}, "titan": {"cpus": 8, "memory_gb": 24}},
        "settings": {"medsam_refine_max_workers": workers},
    }


def assert_resources(trace, cpus, memory):
    assert all(row["status"] == "COMPLETED" and int(row["cpus"]) == cpus for row in trace)
    for row in trace:
        value, unit = row["memory"].split()
        gb = float(value) / 1024 if unit == "MB" else float(value)
        assert gb == memory


def test_explicit_gpu_plan_reaches_both_actual_commands_resources_and_threads(tmp_path):
    result, scripts, trace = probe(tmp_path, plan())
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(trace) == 2 and len(scripts) == 2
    assert_resources(trace, 2, .5)
    medsam = json.loads((tmp_path / "published/14_medsam_refine_tissue/sample/sample_standard_medsam_summary.json").read_text())
    titan = json.loads((tmp_path / "published/17_titan/sample/titan_sample_standard/titan_metadata.json").read_text())
    assert medsam["medsam_device"] == "cuda" and medsam["max_workers"] == 2
    assert titan["gpu"] == 0
    for record in (medsam, titan):
        assert record["threads"] == dict.fromkeys(THREAD_VARS, "2")
        assert record["used_model"] is False
    for flag, value in [("--appearance-clusters", "24"), ("--medsam-core-erosion-um", "4.5"), ("--medsam-cluster-tile-size", "4096"), ("--pre-boundary-iterations", "16"), ("--pre-boundary-final-temperature", "0.05")]:
        assert medsam["argv"][medsam["argv"].index(flag) + 1] == value
    assert "--pre-boundary-competition" in medsam["argv"]
    assert "--appearance-refine" in medsam["argv"] and "--no-appearance-multiclass" in medsam["argv"]
    for flag, value in [("--batch-size", "7"), ("--target-mpp", "0.5"), ("--patch-size", "512"), ("--revision", "locked-test-revision")]:
        assert titan["argv"][titan["argv"].index(flag) + 1] == value
    assert "--offline" in titan["argv"]
    assert all("profile=aggressive" in command and "memory_budget_gb=0.5" in command for command in scripts.values())


@pytest.mark.parametrize(
    "device,override,workers,expected_device,expected_workers",
    [
        ("cpu", "auto", 4, "cpu", 2),
        ("gpu", "cpu", 1, "cpu", 1),
        ("cpu", "cuda:2", 8, "cuda:2", 2),
        ("gpu", "mps", 1, "mps", 1),
    ],
)
def test_medsam_explicit_device_and_planned_worker_cap_are_preserved(tmp_path, device, override, workers, expected_device, expected_workers):
    result, scripts, trace = probe(tmp_path, plan(device=device, workers=workers, memory=1.5), override, targets=("auxiliary",))
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(trace) == 1 and "REFINE_CELL_AUXILIARY_MEDSAM" in trace[0]["name"]
    assert_resources(trace, 2, 1.5)
    record = json.loads((tmp_path / "published/14_medsam_refine_tissue/sample/sample_standard_medsam_summary.json").read_text())
    assert record["medsam_device"] == expected_device and record["max_workers"] == expected_workers
    assert record["threads"] == dict.fromkeys(THREAD_VARS, "2")
    if expected_device in ("cpu", "mps"):
        assert all("auto_hardware=false" in command for command in scripts.values())


def test_titan_preserves_explicit_gpu_ordinal_under_managed_allocation(tmp_path):
    # This isolated native executor has no dynamic single-GPU admission. The
    # production config separately rejects nonzero ordinals when admission is on.
    result, _, trace = probe(tmp_path, plan(profile="conservative", cpus=1, memory=1.5), targets=("titan",), titan_gpu=2)
    assert result.returncode == 0, result.stdout + result.stderr
    assert_resources(trace, 1, 1.5)
    record = json.loads((tmp_path / "published/17_titan/sample/titan_sample_standard/titan_metadata.json").read_text())
    assert record["gpu"] == 2 and record["threads"] == dict.fromkeys(THREAD_VARS, "1")


def test_medsam_runtime_false_setting_is_not_replaced_by_stale_true_params(tmp_path):
    runtime = plan(workers=1)
    runtime["settings"]["hardware_auto"] = False
    result, scripts, trace = probe(tmp_path, runtime, targets=("medsam",))
    assert result.returncode == 0, result.stdout + result.stderr
    assert_resources(trace, 2, .5)
    record = json.loads((tmp_path / "published/14_medsam_refine_tissue/sample/sample_standard_medsam_summary.json").read_text())
    assert record["medsam_device"] == "cuda" and record["max_workers"] == 1
    assert all("auto_hardware=false" in command for command in scripts.values())


def test_titan_cpu_plan_fails_before_python_recorder(tmp_path):
    result, scripts, trace = probe(tmp_path, plan(device="cpu"), targets=("titan",))
    assert result.returncode != 0
    assert "TITAN requires a resolved GPU runtime" in result.stdout + result.stderr
    assert len(scripts) == 1 and len(trace) == 1 and trace[0]["status"] == "FAILED"
    assert not list((tmp_path / "work").glob("*/*/titan_sample_standard/titan_metadata.json"))
