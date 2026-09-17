"""Execute real Nextflow encoder commands using model-free argument recorders."""
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
MODULES = {
    "single": ("extract_uni2_embeddings", "EXTRACT_UNI2_EMBEDDINGS"),
    "shared": ("extract_uni2_embeddings_shared", "EXTRACT_UNI2_EMBEDDINGS_SHARED"),
    "cellvit": ("run_cellvitpp", "RUN_CELLVITPP"),
}


def runtime_probe(tmp_path, plan, targets, embedding_storage="csv"):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    (tmp_path / "modules").mkdir()
    (tmp_path / "lib").mkdir()
    for name in ("TaskRuntime.groovy", "HostRuntime.groovy", "HardwarePolicy.groovy", "PipelineHelpers.groovy", "ProcessCode.groovy"):
        shutil.copy2(ROOT / "lib" / name, tmp_path / "lib" / name)
    for module, _ in MODULES.values():
        shutil.copy2(ROOT / "modules" / f"{module}.nf", tmp_path / "modules" / f"{module}.nf")
    (tmp_path / "bin").symlink_to(ROOT / "bin", target_is_directory=True)
    recorder = tmp_path / "record_encoder_arguments.py"
    recorder.write_text("""import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--outdir', required=True)
p.add_argument('--paired-inner-square-outdir')
p.add_argument('--device')
p.add_argument('--embedding-storage')
p.add_argument('--embedding-mode')
for name in ['torch-threads','batch','rows-per-csv','cpus','memory-mb','ray-workers','ray-worker-cpus']:
    p.add_argument('--'+name, type=int)
args, _ = p.parse_known_args()
record = vars(args) | {'used_model': False}
for name in [args.outdir, args.paired_inner_square_outdir]:
    if name:
        output = Path(name); output.mkdir(parents=True, exist_ok=True)
        (output/'record_arguments.json').write_text(json.dumps(record))
        (output/'cellvit_cells.json').write_text(json.dumps({'pipeline_metadata': {'used_model': False}, 'cells': []}))
""")
    for name in ("image.tif", "mask.tif", "tissue.tif", "objects.csv", "shift.json"):
        (tmp_path / name).write_text("model-free command fixture; no image inference\n")
    (tmp_path / "resolution.json").write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5}))
    (tmp_path / "runtime.json").write_text(json.dumps(plan))
    modules = [ROOT / "modules" / f"{value[0]}.nf" for value in MODULES.values()]
    params = {key: 0 for path in modules for key in re.findall(r"params\.([A-Za-z_][A-Za-z_0-9]*)", path.read_text())}
    params.update({"outdir_base": str(tmp_path / "published"), "publish_dir_mode": "copy",
        "compute_device": "auto", "_resolved_compute_device": "cpu", "hardware_profile": "auto",
        "_executor_max_cpus": 48, "_executor_max_memory_gb": 96,
        "uni2_cpus": 48, "uni2_memory_gb": 96, "cellvit_cpus": 48, "cellvit_memory_gb": 96,
        "uni2_script": recorder.name, "cellvit_script": recorder.name,
        "uni2_time": "1m", "cellvit_time": "1m", "uni2_batch": 64, "uni2_rows_per_csv": 10000,
        "uni2_embedding_storage": embedding_storage,
        "uni2_torch_threads": 1, "cellvit_ray_workers": 1, "cellvit_ray_worker_cpus": 1,
        "hardware_auto": True, "uni2_auto_hardware": True, "uni2_max_auto_batch": 24,
        "uni2_force_full_image": False, "uni2_save_tiles": False, "gpu_debug_diagnostics": False,
        "uni2_mask_context_mode": "none", "uni2_tiles_root": "tiles", "uni2_encoder": "uni2-h",
        "uni2_backend": "auto", "uni2_pooling": "cls", "uni2_grid": "10x10",
        "uni2_paired_inner_square_mode": "token_subset", "cellvit_cache_dir": str(tmp_path / "cellvit_cache"),
        "cellvit_executable": "unused-model-free", "cellvit_model": "HIPT", "cellvit_taxonomy": "pannuke",
        "cellvit_amp": False, "cellvit_export_embeddings": False, "hf_token_env_file": "",
        "hf_token_env_var_name": "UNUSED_MODEL_FREE_TOKEN", "hf_home": str(tmp_path / "hf"),
        "hf_hub_cache": str(tmp_path / "hf/hub"), "hf_hub_offline": "1"})
    (tmp_path / "params.json").write_text(json.dumps(params))
    includes = "\n".join(f"include {{ {MODULES[key][1]} }} from './modules/{MODULES[key][0]}'" for key in targets)
    calls = {
        "single": "EXTRACT_UNI2_EMBEDDINGS(Channel.of(tuple('single', image, mask, resolution, 'tile', false, 'none', 255)), runtime_plan)",
        "shared": "EXTRACT_UNI2_EMBEDDINGS_SHARED(Channel.of(tuple('shared', image, mask, 'grid', objects, resolution)), runtime_plan)",
        "cellvit": "RUN_CELLVITPP(Channel.of(tuple('cellvit', image, shift, tissue, resolution)), runtime_plan)",
    }
    # Resolve/load only after all modules have been included. Deliberately stale
    # parameter mutations are not relied upon to reach imported module scopes.
    (tmp_path / "main.nf").write_text("""nextflow.enable.dsl=2
import groovy.json.JsonSlurperClassic
""" + includes + """
workflow {
  def runtime_plan = new JsonSlurperClassic().parseText(file('runtime.json').text)
  params.compute_device = runtime_plan.compute_device
  params._resolved_compute_device = runtime_plan.compute_device
  params.hardware_profile = runtime_plan.profile
  def image = file('image.tif', checkIfExists: true)
  def mask = file('mask.tif', checkIfExists: true)
  def resolution = file('resolution.json', checkIfExists: true)
  def objects = file('objects.csv', checkIfExists: true)
  def shift = file('shift.json', checkIfExists: true)
  def tissue = file('tissue.tif', checkIfExists: true)
""" + "\n".join(calls[key] for key in targets) + "\n}\n")
    interpreter = shlex.quote(sys.executable)
    (tmp_path / "nextflow.config").write_text(f"""process {{
  executor = 'local'
  beforeScript = '''
  python() {{ {interpreter} "$@"; }}
  nvidia-smi() {{ printf '8192, 16384\\n'; }}
  export -f python
  export -f nvidia-smi
  '''
}}
executor.queueSize = 1
trace.fields = 'task_id,name,status,exit,cpus,memory'
""")
    env = {key: value for key, value in os.environ.items() if not key.startswith("HF_")}
    result = subprocess.run([nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "main.nf"),
        "-params-file", str(tmp_path / "params.json"), "-ansi-log", "false", "-with-trace", str(tmp_path / "trace.tsv"),
        "-work-dir", str(tmp_path / "work")], cwd=tmp_path,
        env={**env, "NXF_OFFLINE": "true"}, text=True, capture_output=True, timeout=60)
    scripts = {path.parent: path.read_text() for path in (tmp_path / "work").glob("*/*/.command.sh")}
    trace_path = tmp_path / "trace.tsv"
    trace = list(csv.DictReader(trace_path.open(), delimiter="\t")) if trace_path.exists() else []
    return result, scripts, trace


def plan(device="gpu", profile="aggressive", memory=3.5, workers=2, worker_cpus=5):
    return {"schema_version": 1, "compute_device": device, "profile": profile,
        "cpu_budget": 3, "memory_budget_gb": memory,
        "stages": {"uni2": {"cpus": 2, "memory_gb": 10}, "cellvit": {"cpus": 8, "memory_gb": 24}},
        "settings": {"uni2_torch_threads": 6, "cellvit_ray_workers": workers, "cellvit_ray_worker_cpus": worker_cpus}}


@pytest.mark.parametrize("profile,workers,expected_batch,expected_workers", [("aggressive", 2, 24, 2), ("conservative", 20, 16, 3)])
def test_runtime_plan_reaches_actual_gpu_commands_resources_threads_and_ray(tmp_path, profile, workers, expected_batch, expected_workers):
    result, scripts, trace = runtime_probe(tmp_path, plan(profile=profile, workers=workers), ["single", "shared", "cellvit"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(scripts) == 3 and len(trace) == 3
    assert {row["status"] for row in trace} == {"COMPLETED"}
    for row in trace:
        assert float(row["memory"].split()[0]) == 3.5  # caps the stage's stale 10/24 GB allocation
        assert int(row["cpus"]) == (3 if "CELLVIT" in row["name"] else 2)
    for sample in ("single", "shared"):
        record = json.loads((tmp_path / f"published/09_embeddings/{sample}/embeddings_{sample}_tile/record_arguments.json").read_text())
        assert record["device"] == "cuda" and record["torch_threads"] == 2
        assert record["batch"] == expected_batch and record["rows_per_csv"] == 10000
        assert not record["used_model"]
    paired = tmp_path / "published/09_embeddings/shared/embeddings_shared_inner_square/record_arguments.json"
    assert json.loads(paired.read_text())["batch"] == expected_batch
    cell = json.loads((tmp_path / "published/03c_cellvitpp/cellvit/cellvit_cellvit/record_arguments.json").read_text())
    assert cell["cpus"] == 3 and cell["memory_mb"] == 3584
    assert cell["ray_workers"] == expected_workers and cell["ray_worker_cpus"] == 1
    assert cell["ray_workers"] * cell["ray_worker_cpus"] <= cell["cpus"]
    assert not cell["used_model"]
    assert all(profile in text for text in scripts.values())
    assert not any('test "cpu" = "gpu"' in text for text in scripts.values())


def test_cpu_low_memory_plan_preserves_uni2_safeguards(tmp_path):
    result, scripts, trace = runtime_probe(tmp_path, plan(device="cpu", profile="conservative", memory=2.5), ["single", "shared"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(scripts) == 2 and all(row["status"] == "COMPLETED" for row in trace)
    assert all(int(row["cpus"]) == 1 and float(row["memory"].split()[0]) == 2.5 for row in trace)
    for sample in ("single", "shared"):
        record = json.loads((tmp_path / f"published/09_embeddings/{sample}/embeddings_{sample}_tile/record_arguments.json").read_text())
        assert record["device"] == "cpu" and record["torch_threads"] == 1
        assert record["batch"] == 2 and record["rows_per_csv"] == 2000


def test_cellvit_rejects_explicit_cpu_plan_before_any_recorder_inference(tmp_path):
    result, scripts, trace = runtime_probe(tmp_path, plan(device="cpu"), ["cellvit"])
    assert result.returncode != 0
    assert "requires a resolved GPU runtime" in result.stdout + result.stderr
    assert len(scripts) == 1 and len(trace) == 1 and trace[0]["status"] == "FAILED"
    assert not list((tmp_path / "work").glob("*/*/cellvit_cellvit/record_arguments.json"))


@pytest.mark.parametrize("storage", ["csv", "binary"])
def test_storage_choice_reaches_single_shared_and_paired_outputs(tmp_path, storage):
    result, scripts, trace = runtime_probe(tmp_path, plan(device="cpu"), ["single", "shared"], storage)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(trace) == 2 and all(row["status"] == "COMPLETED" for row in trace)
    for sample, mode in (("single", "tile"), ("shared", "tile"), ("shared", "inner_square")):
        record = json.loads((tmp_path / f"published/09_embeddings/{sample}/embeddings_{sample}_{mode}/record_arguments.json").read_text())
        assert record["embedding_storage"] == storage
        assert record["embedding_mode"] == "tile"  # shared secondary has its own explicit inner-square writer
        assert not record["used_model"]


def test_invalid_storage_rejected_before_extraction(tmp_path):
    result, _, _ = runtime_probe(tmp_path, plan(device="cpu"), ["single"], "float16")
    assert result.returncode != 0
    assert "uni2_embedding_storage must be csv or binary" in result.stdout + result.stderr
    assert not list((tmp_path / "work").glob("*/*/embeddings_single_tile/record_arguments.json"))
