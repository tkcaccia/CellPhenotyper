import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "bin" / "gigatime_hardware.py"
PIPELINE_MODULE_PATH = Path(__file__).parents[1] / "modules" / "run_gigatime_on_crop.nf"
SPEC = importlib.util.spec_from_file_location("gigatime_hardware", MODULE_PATH)
hardware = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = hardware
SPEC.loader.exec_module(hardware)


def settings(**overrides):
    values = {
        "enabled": True,
        "profile": "balanced",
        "requested_batch": 1,
        "requested_block": 1024,
        "requested_output_gib": 2.0,
        "max_auto_batch": 16,
        "max_auto_block": 3072,
        "max_auto_output_gib": 16.0,
        "task_memory_gib": 18.0,
        "min_free_system_gib": 8.0,
        "system_mem_available_gib": 23.0,
        "cuda_mem_free_gib": 14.5,
        "cuda_mem_total_gib": 15.5,
        "use_cuda": True,
    }
    values.update(overrides)
    return hardware.choose_gigatime_hardware_settings(**values)


class GigaTIMEHardwarePolicyTest(unittest.TestCase):
    def test_pipeline_passes_a_concrete_hardware_profile(self):
        module = PIPELINE_MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn("val(runtime_plan)", module)
        self.assertIn("TaskRuntime.profile(runtime_plan)", module)
        self.assertNotIn("params._resolved_hardware_profile", module)
        self.assertIn("--hardware-profile ${resolved_hardware_profile}", module)

    def test_cuda_batch_uses_vram_tier_instead_of_host_ram_tier(self):
        result = settings()
        self.assertEqual(result["derived_caps"]["memory_batch"], 4)
        self.assertEqual(result["derived_caps"]["gpu_batch"], 8)
        self.assertEqual(result["effective"]["batch_size"], 8)
        self.assertEqual(result["effective"]["block_size"], 1024)

    def test_cpu_batch_remains_limited_by_host_memory(self):
        result = settings(use_cuda=False, cuda_mem_free_gib=None, cuda_mem_total_gib=None)
        self.assertEqual(result["effective"]["batch_size"], 4)

    def test_auto_caps_remain_hard_limits(self):
        result = settings(max_auto_batch=4, max_auto_block=768)
        self.assertEqual(result["effective"]["batch_size"], 4)
        self.assertEqual(result["effective"]["block_size"], 768)

    def test_high_memory_gpu_can_exceed_legacy_batch_ceiling(self):
        result = settings(
            profile="aggressive",
            max_auto_batch=32,
            task_memory_gib=80.0,
            min_free_system_gib=6.0,
            system_mem_available_gib=100.0,
            cuda_mem_free_gib=72.0,
            cuda_mem_total_gib=80.0,
        )
        self.assertEqual(result["effective"]["batch_size"], 24)
        self.assertEqual(result["effective"]["block_size"], 3072)

    def test_unknown_cuda_memory_keeps_conservative_requested_batch(self):
        result = settings(cuda_mem_free_gib=None, cuda_mem_total_gib=None)
        self.assertEqual(result["effective"]["batch_size"], 1)

    def test_low_available_ram_fails_before_inference(self):
        with self.assertRaisesRegex(RuntimeError, "cannot run safely"):
            settings(system_mem_available_gib=10.0)


@pytest.mark.parametrize("requested,resolved,stage,policy_budget,expected", [
    ("auto", None, "", (2, 4), "conservative"),
    ("auto", None, "", (32, 96), "aggressive"),
    ("auto", None, "conservative", (32, 96), "conservative"),
    ("auto", None, "auto", None, "balanced"),
    ("aggressive", "conservative", "", None, "conservative"),
    ("conservative", "aggressive", " CONSERVATIVE ", None, "conservative"),
])
def test_actual_nextflow_command_uses_effective_hardware_policy(tmp_path, requested, resolved, stage, policy_budget, expected):
    """Execute the real module command with an argument recorder, never a model.

    Module include precedes runtime policy resolution just as it does in main.nf.
    A portable memory probe shim avoids reading Linux /proc on the test host;
    the actual policy flag still travels through the generated shell and CLI.
    """
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    root = MODULE_PATH.parents[1]
    (tmp_path / "modules").mkdir()
    (tmp_path / "lib").mkdir()
    shutil.copy2(PIPELINE_MODULE_PATH, tmp_path / "modules/run_gigatime_on_crop.nf")
    shutil.copy2(root / "lib/HardwarePolicy.groovy", tmp_path / "lib/HardwarePolicy.groovy")
    shutil.copy2(root / "lib/TaskRuntime.groovy", tmp_path / "lib/TaskRuntime.groovy")
    shutil.copy2(root / "lib/HostRuntime.groovy", tmp_path / "lib/HostRuntime.groovy")
    (tmp_path / "bin").symlink_to(root / "bin", target_is_directory=True)
    recorder = tmp_path / "record_gigatime_arguments.py"
    recorder.write_text("""import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--hardware-profile', choices=['conservative', 'balanced', 'aggressive'], required=True)
p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
p.add_argument('--task-memory-gb', type=float, required=True)
p.add_argument('--outdir', required=True)
args, extra = p.parse_known_args()
Path(args.outdir, 'hardware_argument_probe.json').write_text(json.dumps({'profile': args.hardware_profile, 'device': args.device, 'task_memory_gb': args.task_memory_gb, 'used_model': False}))
""")
    names = ["image.tif", "shift.json", "nuclei.tif", "whole.tif", "tissue.tif", "ring.tif"]
    for name in names:
        (tmp_path / name).write_text("synthetic command probe; not image data\n")
    # Unused interpolated arguments still need defined parameters. Override the
    # actual policy/resource/environment controls with valid, bounded values.
    module = PIPELINE_MODULE_PATH.read_text()
    params = {name: 0 for name in re.findall(r"params\.([A-Za-z_][A-Za-z_0-9]*)", module)}
    params.update({"outdir_base": str(tmp_path / "published"), "publish_dir_mode": "copy",
        "_executor_max_cpus": 1, "_executor_max_memory_gb": 1, "gigatime_cpus": 1,
        "gigatime_memory_gb": 1, "gigatime_time": "1m", "gigatime_script": recorder.name,
        "compute_device": "cpu", "_resolved_compute_device": "cpu", "hf_token_env_file": "",
        "hf_home": str(tmp_path / "hf"), "hf_hub_cache": str(tmp_path / "hf/hub"), "hf_hub_offline": "1",
        "gigatime_hf_token_env_var_name": "CELL_PHENOTYPER_UNUSED_PROBE_TOKEN",
        "hf_token_env_var_name": "CELL_PHENOTYPER_UNUSED_PROBE_TOKEN",
        "hardware_profile": requested, "_resolved_hardware_profile": resolved,
        "gigatime_hardware_profile": stage, "hardware_auto": True, "gigatime_auto_hardware": None,
        "hardware_max_auto_batch": 16, "hardware_max_auto_block_size": 1024,
        "hardware_max_auto_output_gib": 2, "hardware_min_free_system_gb": 1,
        "gigatime_memory_wait_minutes": 0, "gigatime_min_usable_memory_gb": 0,
        "gigatime_integrated_quantification": False, "expand_um": -1,
        "gigatime_output_format": "ome_tiff", "gigatime_output_dtype": "float32",
        "gigatime_output_compression": "deflate", "gigatime_output_channels": "", "gigatime_jpg_markers": ""})
    (tmp_path / "params.json").write_text(json.dumps(params))
    policy = ""
    if policy_budget:
        policy = f"""def hardware_plan = HardwarePolicy.resolve(params, {policy_budget[0]}, {policy_budget[1]}, false, 0.0d)
  params.hardware_profile = hardware_plan.profile
  params._resolved_hardware_profile = hardware_plan.profile
  runtime_plan.profile = hardware_plan.profile
  println 'EFFECTIVE_POLICY=' + hardware_plan.profile
"""
    # Opposing include-time defaults prove resources/device come from the value
    # input, not the stale params captured by the module. No CUDA code is run.
    planned_device = "gpu" if policy_budget else "cpu"
    planned_profile = resolved or (requested if requested != "auto" else "balanced")
    planned_memory = 0.5 if stage == " CONSERVATIVE " else 2
    files = ", ".join(f"file('{name}', checkIfExists: true)" for name in names)
    (tmp_path / "main.nf").write_text(f"""nextflow.enable.dsl=2
include {{ RUN_GIGATIME_ON_CROP }} from './modules/run_gigatime_on_crop'
workflow {{
  def runtime_plan = [schema_version:1, compute_device:'{planned_device}', profile:'{planned_profile}',
    cpu_budget:2, memory_budget_gb:{planned_memory}, stages:[gigatime:[cpus:2, memory_gb:{planned_memory}]],
    settings:[gigatime_cpus:2, gigatime_memory_gb:{planned_memory}]]
  {policy}
  params._resolved_compute_device = '{planned_device}'
  params.gigatime_cpus = 2
  RUN_GIGATIME_ON_CROP(Channel.of(tuple('probe', {files})), runtime_plan)
}}
""")
    interpreter = shlex.quote(sys.executable)
    (tmp_path / "nextflow.config").write_text(f"""process {{
  executor = 'local'
  maxForks = 1
  beforeScript = '''
  python() {{
    if [[ "$#" -eq 1 && "$1" == "-" ]]; then
      {interpreter} -c 'import sys; sys.stdin.read(); print(100.0)'
    else
      {interpreter} "$@"
    fi
  }}
  export -f python
  '''
}}
trace.fields = 'task_id,name,status,cpus,memory'
""")
    env = {key: value for key, value in os.environ.items() if not key.startswith("HF_")}
    result = subprocess.run([nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "main.nf"),
        "-params-file", str(tmp_path / "params.json"), "-ansi-log", "false", "-with-trace", str(tmp_path / "trace.tsv"),
        "-work-dir", str(tmp_path / "work")], cwd=tmp_path,
        env={**env, "NXF_OFFLINE": "true"}, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    if policy_budget:
        policy_expected = "conservative" if policy_budget[0] <= 4 else "aggressive"
        assert f"EFFECTIVE_POLICY={policy_expected}" in result.stdout
    probe = json.loads((tmp_path / "published/05_gigatime/probe/gigatime_probe/hardware_argument_probe.json").read_text())
    assert probe == {"profile": expected, "device": "cuda" if planned_device == "gpu" else "cpu",
                     "task_memory_gb": planned_memory, "used_model": False}
    scripts = list((tmp_path / "work").glob("*/*/.command.sh"))
    assert len(scripts) == 1
    command = scripts[0].read_text()
    assert re.search(rf"--hardware-profile\s+{expected}\s", command)
    assert not re.search(r"--hardware-profile\s+auto\s", command)
    trace = (tmp_path / "trace.tsv").read_text().splitlines()
    assert len(trace) == 2 and "COMPLETED" in trace[1]
    row = dict(zip(trace[0].split("\t"), trace[1].split("\t")))
    assert row["cpus"] == "2" and row["memory"] == ("512 MB" if planned_memory == 0.5 else "2 GB")


if __name__ == "__main__":
    unittest.main()
