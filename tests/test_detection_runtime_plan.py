"""Run real detector module commands against explicit no-model recorders."""
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
    "grandqc": ("run_grandqc_artifact_analysis", "RUN_GRANDQC_ARTIFACT_ANALYSIS"),
    "stardist": ("run_stardist_roi_segmentation", "RUN_STARDIST_ROI_SEGMENTATION"),
    "hovernet": ("run_hovernet_monusac", "RUN_HOVERNET_MONUSAC"),
}


def plan(device="gpu", profile="aggressive", memory=3.5, cpus=3):
    return {"schema_version": 1, "compute_device": device, "profile": profile,
            "cpu_budget": cpus, "memory_budget_gb": memory,
            "stages": {"grandqc": {"cpus": 2, "memory_gb": 96},
                       "stardist": {"cpus": 6, "memory_gb": 96},
                       "hovernet": {"cpus": 8, "memory_gb": 96}},
            "settings": {"hovernet_postproc_workers": 30}}


def run_probe(tmp_path, runtime, targets, overrides=None):
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
    recorder = tmp_path / "record_detection_arguments.py"
    recorder.write_text("""import argparse, gzip, json, os
from pathlib import Path
p = argparse.ArgumentParser(allow_abbrev=False)
p.add_argument('--image', '--in', dest='image')
p.add_argument('--outdir', required=True)
p.add_argument('--device')
p.add_argument('--sample-id')
p.add_argument('--gpu')
p.add_argument('--cache-backend')
p.add_argument('--execution-mode')
p.add_argument('--export-contours', action='store_true')
for name in ['memory-budget-gb','min-area','prob','nms','artifact-overlap-fraction','target-mpp']:
    p.add_argument('--'+name, type=float)
for name in ['big-block-size','inference-workers','postproc-workers','patch-size','artifact-tile-size','batch-size','chunk-shape','tile-shape','stream-core-size','stream-halo','stream-batch-tiles','tile-jpeg-quality']:
    p.add_argument('--'+name, type=int)
p.add_argument('--tiles', nargs=2, type=int)
p.add_argument('--big-tiles', nargs=2, type=int)
args, _ = p.parse_known_args()
output = Path(args.outdir); output.mkdir(parents=True, exist_ok=True)
record = vars(args) | {'used_model': False, 'environment': {key: os.environ.get(key) for key in
    ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','TF_NUM_INTRAOP_THREADS','LOKY_MAX_CPU_COUNT']}}
(output/'record_arguments.json').write_text(json.dumps(record))
# These are conspicuous engineering placeholders, not predicted biological data.
if args.sample_id:
    names = [args.sample_id+'_grandqc_clean_tissue_mask.tif', args.sample_id+'_grandqc_artifact_mask.tif']
elif output.name == 'stardist_out':
    names = ['crop_roi.tif','labels.tif','objects.csv','roi_all_crop.geojson','shift.json']
else:
    names = ['hovernet_cells.json.gz']
    for transient in ['cache','input','input_mask','raw','raw_cells','runtime_cache','hovernet_runtime']:
        path = output/transient; path.mkdir(); (path/'large-transient-sentinel').write_text('temporary')
for name in names:
    if name.endswith('.gz'):
        with gzip.open(output/name, 'wt') as handle: handle.write('{"cells":[]}')
    else:
        (output/name).write_text('MODEL-FREE RUNTIME COMMAND FIXTURE; NOT BIOLOGICAL RESULTS\\n')
""")
    for name in ("image.tif", "roi.geojson", "shift.json", "tissue.tif"):
        (tmp_path / name).write_text("model-free command fixture; no inference\n")
    (tmp_path / "runtime.json").write_text(json.dumps(runtime))
    params = {key: 0 for module, _ in MODULES.values()
              for key in re.findall(r"params\.([A-Za-z_][A-Za-z_0-9]*)", (ROOT / "modules" / f"{module}.nf").read_text())}
    params.update({"outdir_base": str(tmp_path / "published"), "publish_dir_mode": "copy",
        "compute_device": "auto", "_resolved_compute_device": "cpu", "hardware_profile": "auto",
        "_executor_max_cpus": 48, "_executor_max_memory_gb": 96, "hardware_auto": True,
        "host_arch": "amd64", "enable_stardist_gpu_on_arm64": False,
        "grandqc_script": recorder.name, "grandqc_device": "auto", "grandqc_time": "1m",
        "grandqc_cpus": 48, "grandqc_memory_gb": 96, "grandqc_cache_dir": str(tmp_path / "grandqc_cache"),
        "grandqc_bootstrap_deps": False, "grandqc_download_models": False, "grandqc_create_geojson": False,
        "grandqc_patch_size": 384, "grandqc_artifact_tile_size": 640, "grandqc_artifact_overlap_fraction": 0.2,
        "grandqc_default_source_mpp": 0.5, "grandqc_artifact_mpp_model": "auto", "grandqc_tissue_mpp_model": 10,
        "stardist_script": recorder.name, "stardist_time": "1m", "stardist_cpus": 48, "stardist_memory_gb": 96,
        "stardist_auto_hardware": True, "stardist_tiles_x": 32, "stardist_tiles_y": 32,
        "stardist_big_block_size": 2048, "stardist_big_tiles_x": 8, "stardist_big_tiles_y": 8,
        "stardist_min_auto_tiles": 2, "stardist_max_auto_block_size": 8192, "hardware_max_auto_block_size": 8192,
        "stardist_model": "auto", "stardist_keras_home": str(tmp_path / "empty_model_cache"),
        "stardist_pretrained_zip": "", "stardist_precomputed_labels_full": "", "stardist_pythonpath": "",
        "stardist_autoinstall_runtime": False, "gpu_debug_diagnostics": False,
        "write_full_labels": False, "full_format": "tif", "allow_huge_tif": False,
        "stardist_prob": 0.42, "stardist_nms": 0.31, "stardist_min_area": 11, "stardist_big_mode": "auto",
        "hovernet_script": recorder.name, "hovernet_time": "1m", "hovernet_cpus": 48, "hovernet_memory_gb": 96,
        "hovernet_postproc_workers": 1, "hovernet_prediction_cache": "", "hovernet_cache_backend": "zarr", "hovernet_repo_dir": "unused-model-free",
        "hovernet_monusac_checkpoint": "unused-model-free", "hovernet_target_mpp": 0.25, "hovernet_default_mpp": 0.5,
        "hovernet_gpu": 0, "hovernet_batch_size": 8, "hovernet_chunk_shape": 8192, "hovernet_tile_shape": 2048,
        "hovernet_execution_mode": "streaming_tiles", "hovernet_stream_core_size": 4096,
        "hovernet_stream_halo": 256, "hovernet_stream_batch_tiles": 64,
        "hovernet_tile_jpeg_quality": 92,
        "hovernet_export_contours": False})
    params.update(overrides or {})
    (tmp_path / "params.json").write_text(json.dumps(params))
    includes = "\n".join(f"include {{ {MODULES[key][1]} }} from './modules/{MODULES[key][0]}'" for key in targets)
    calls = {"grandqc": "RUN_GRANDQC_ARTIFACT_ANALYSIS(Channel.of(tuple('grandqc', image)), runtime_plan)",
             "stardist": "RUN_STARDIST_ROI_SEGMENTATION(Channel.of(tuple('stardist', image, roi, shift, tissue)), runtime_plan)",
             "hovernet": "RUN_HOVERNET_MONUSAC(Channel.of(tuple('hovernet', image, shift, tissue)), runtime_plan)"}
    (tmp_path / "main.nf").write_text("nextflow.enable.dsl=2\nimport groovy.json.JsonSlurperClassic\n" + includes + """
workflow {
  def runtime_plan = new JsonSlurperClassic().parseText(file('runtime.json').text)
  params.compute_device = runtime_plan.compute_device
  params._resolved_compute_device = runtime_plan.compute_device
  params.hardware_profile = runtime_plan.profile
  def image = file('image.tif', checkIfExists:true)
  def roi = file('roi.geojson', checkIfExists:true)
  def shift = file('shift.json', checkIfExists:true)
  def tissue = file('tissue.tif', checkIfExists:true)
""" + "\n".join(calls[key] for key in targets) + "\n}\n")
    interpreter = shlex.quote(sys.executable)
    (tmp_path / "nextflow.config").write_text(f"""process {{
  executor = 'local'
  beforeScript = '''
  python() {{ {interpreter} "$@"; }}
  export -f python
  '''
}}
// Model-free metadata tests may exercise a large reported RAM allocation;
// the recorder never allocates image/model tensors.
executor.memory = '128 GB'
executor.cpus = 8
executor.queueSize = 1
trace.fields = 'task_id,name,status,exit,cpus,memory'
""")
    env = {key: value for key, value in os.environ.items() if not key.startswith("HF_")}
    result = subprocess.run([nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "main.nf"),
        "-params-file", str(tmp_path / "params.json"), "-ansi-log", "false", "-with-trace", str(tmp_path / "trace.tsv"),
        "-work-dir", str(tmp_path / "work")], cwd=tmp_path, text=True, capture_output=True, timeout=60,
        env={**env, "NXF_OFFLINE": "true", "CUDA_VISIBLE_DEVICES": "GPU-model-free-sentinel"})
    trace_path = tmp_path / "trace.tsv"
    trace = list(csv.DictReader(trace_path.open(), delimiter="\t")) if trace_path.exists() else []
    records = {key: json.loads(path.read_text()) for key in targets
               for path in (tmp_path / "published").glob(f"*/{key}/*/record_arguments.json")}
    scripts = [path.read_text() for path in (tmp_path / "work").glob("*/*/.command.sh")]
    return result, records, trace, scripts


def test_actual_gpu_commands_use_plan_caps_and_preserve_scientific_options(tmp_path):
    result, records, trace, scripts = run_probe(tmp_path, plan(), list(MODULES))
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(records) == len(trace) == len(scripts) == 3
    assert all(row["status"] == "COMPLETED" for row in trace)
    for row in trace:
        assert float(row["memory"].split()[0]) == 3.5
        assert int(row["cpus"]) == (2 if "GRANDQC" in row["name"] else 3)
    grandqc, star, hover = (records[key] for key in MODULES)
    assert grandqc["device"] == "cuda" and grandqc["environment"]["OMP_NUM_THREADS"] == "2"
    assert (grandqc["patch_size"], grandqc["artifact_tile_size"], grandqc["artifact_overlap_fraction"]) == (384, 640, 0.2)
    assert star["memory_budget_gb"] == 3.5 and star["environment"]["LOKY_MAX_CPU_COUNT"] == "3"
    assert star["device"] == "cuda"
    assert (star["prob"], star["nms"], star["min_area"]) == (0.42, 0.31, 11)
    assert star["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-model-free-sentinel"
    assert hover["inference_workers"] == 3 and hover["postproc_workers"] == 1
    assert hover["cache_backend"] == "zarr"
    assert hover["execution_mode"] == "streaming_tiles"
    assert (hover["stream_core_size"], hover["stream_halo"], hover["stream_batch_tiles"]) == (4096, 256, 64)
    assert (hover["target_mpp"], hover["batch_size"], hover["chunk_shape"], hover["tile_shape"]) == (0.25, 8, 8192, 2048)
    assert all(not record["used_model"] for record in records.values())
    assert all("aggressive" in script for script in scripts)


def test_cpu_plan_grandqc_auto_is_concrete_cpu_even_without_hardware_autotuning(tmp_path):
    result, records, trace, _ = run_probe(tmp_path, plan(device="cpu", memory=0.5, cpus=1), ["grandqc", "stardist"],
                                          {"hardware_auto": False})
    assert result.returncode == 0, result.stdout + result.stderr
    assert records["grandqc"]["device"] == "cpu"  # no Python auto→MPS or CUDA rediscovery
    assert records["stardist"]["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert records["stardist"]["device"] == "cpu"
    assert records["stardist"]["memory_budget_gb"] == 0.5
    assert all(int(row["cpus"]) == 1 and row["memory"] == "512 MB" for row in trace)


@pytest.mark.parametrize("device,override", [("gpu", "cpu"), ("cpu", "cuda"), ("cpu", "mps")])
def test_grandqc_manual_native_device_override_remains_authoritative(tmp_path, device, override):
    result, records, _, _ = run_probe(tmp_path, plan(device=device), ["grandqc"], {"grandqc_device": override})
    assert result.returncode == 0, result.stdout + result.stderr
    assert records["grandqc"]["device"] == override


@pytest.mark.parametrize("profile,memory,tiles,block,big_tiles", [
    ("conservative", 36, [32, 32], 2048, [8, 8]),
    ("balanced", 26, [16, 16], 6144, [3, 3]),
    ("aggressive", 36, [8, 8], 8192, [2, 2]),
])
def test_stardist_existing_tuning_uses_concrete_profile_and_actual_stage_ram(tmp_path, profile, memory, tiles, block, big_tiles):
    result, records, _, _ = run_probe(tmp_path, plan(profile=profile, memory=memory), ["stardist"])
    assert result.returncode == 0, result.stdout + result.stderr
    star = records["stardist"]
    assert (star["tiles"], star["big_block_size"], star["big_tiles"]) == (tiles, block, big_tiles)


@pytest.mark.parametrize("arm_gpu", [False, True])
def test_stardist_arm64_fallback_and_explicit_opt_in_control_cuda_visibility(tmp_path, arm_gpu):
    result, records, _, _ = run_probe(tmp_path, plan(memory=36), ["stardist"], {
        "host_arch": "aarch64", "enable_stardist_gpu_on_arm64": arm_gpu})
    assert result.returncode == 0, result.stdout + result.stderr
    star = records["stardist"]
    assert star["environment"]["CUDA_VISIBLE_DEVICES"] == ("GPU-model-free-sentinel" if arm_gpu else "")
    assert star["device"] == ("cuda" if arm_gpu else "cpu")
    assert star["tiles"] == ([8, 8] if arm_gpu else [32, 32])


def test_hovernet_plan_worker_setting_is_capped_by_cpus_and_ram(tmp_path):
    runtime = plan(memory=36)
    runtime["settings"]["hovernet_postproc_workers"] = 5
    result, records, _, _ = run_probe(tmp_path, runtime, ["hovernet"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert records["hovernet"]["postproc_workers"] == 3  # not stale param=1, nor requested=5 > task.cpus


def test_hovernet_task_cleans_all_transients_on_exit(tmp_path):
    text = (ROOT / "modules" / "run_hovernet_monusac.nf").read_text(encoding="utf-8")
    assert "trap cleanup_hovernet_transients EXIT" in text
    assert "trap 'exit 143' TERM" in text
    for transient in ("cache", "input", "input_mask", "raw", "raw_cells", "runtime_cache", "hovernet_runtime"):
        assert f'"hovernet_${{sample_id}}/{transient}"' in text
    result, records, _, _ = run_probe(tmp_path, plan(memory=36), ["hovernet"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hovernet" in records
    assert not list((tmp_path / "work").glob("*/*/hovernet_hovernet/*/large-transient-sentinel"))


def test_hovernet_cpu_plan_fails_before_model_free_recorder(tmp_path):
    result, records, trace, _ = run_probe(tmp_path, plan(device="cpu"), ["hovernet"])
    assert result.returncode != 0 and "requires a resolved GPU runtime" in result.stdout + result.stderr
    assert not records and len(trace) == 1 and trace[0]["status"] == "FAILED"
