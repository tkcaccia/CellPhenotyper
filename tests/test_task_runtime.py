"""Execute the real runtime-plan helper through Nextflow's Groovy runtime."""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_nextflow(project, source, *, trace=False):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    project.mkdir(parents=True, exist_ok=True)
    (project / "lib").mkdir(exist_ok=True)
    for name in ("TaskRuntime.groovy", "HardwarePolicy.groovy", "HostRuntime.groovy"):
        shutil.copy2(ROOT / "lib" / name, project / "lib" / name)
    (project / "main.nf").write_text(source)
    (project / "nextflow.config").write_text("process.executor = 'local'\ntrace.fields = 'name,status,cpus,memory'\n")
    command = [nextflow, "-log", str(project / "nextflow.log"), "run", str(project / "main.nf"), "-ansi-log", "false"]
    if trace:
        command += ["-with-trace", str(project / "trace.tsv")]
    result = subprocess.run(command, cwd=project, env={**os.environ, "NXF_OFFLINE": "true"},
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    project = tmp_path_factory.mktemp("task_runtime_helper")
    result = run_nextflow(project, r"""nextflow.enable.dsl=2
import groovy.json.JsonOutput
import groovy.json.JsonSlurper

def check(Closure action) {
  try { return [ok:true, value:action.call()] }
  catch (Throwable error) { return [ok:false, type:error.class.name, message:error.message] }
}
def fresh() {
  [schema_version:1, compute_device:'cpu', profile:'balanced', cpu_budget:3, memory_budget_gb:1.5,
   stages:[encoder:[cpus:8, memory_gb:9], small:[cpus:1, memory_gb:0.25]],
   settings:[disabled:false, zero:0, empty:'', absent:null]]
}
workflow {
  def result = [:]
  def hardware = [profile:'balanced', cpu_budget:3, memory_budget_gb:1.5,
    stages:fresh().stages, updates:fresh().settings]
  def original = TaskRuntime.create(hardware, 'gpu')
  def encoded = JsonOutput.toJson(original)
  def restored = new JsonSlurper().parseText(encoded)
  hardware.stages.encoder.cpus = 1
  hardware.updates.zero = 99
  result.roundtrip = [encoded:encoded, cpu:TaskRuntime.cpus(restored,'encoder'),
    memory:TaskRuntime.memory(restored,'encoder'), small:TaskRuntime.memory(restored,'small'),
    device:TaskRuntime.device(restored), profile:TaskRuntime.profile(restored),
    original_cpu:TaskRuntime.cpus(original,'encoder'), original_zero:TaskRuntime.setting(original,'zero',17)]
  result.settings = ['disabled','zero','empty','absent','missing'].collectEntries { name ->
    [(name):TaskRuntime.setting(restored,name,'fallback')]
  }
  result.invalid = [:]
  [null, [], [schema_version:2], 'unresolved'].eachWithIndex { value,index ->
    result.invalid["plan_${index}"] = check { TaskRuntime.device(value) }
  }
  ['schema_version':0, 'compute_device':'auto', 'profile':'auto', 'cpu_budget':0,
   'memory_budget_gb':0, 'stages':[], 'settings':[]].each { key,value ->
    result.invalid["field_${key}"] = check { def plan=fresh(); plan[key]=value; TaskRuntime.profile(plan) }
  }
  [null, 0, -1, 1.5, 'NaN', 'Infinity', 'auto', 2147483648L].eachWithIndex { value,index ->
    result.invalid["cpu_${index}"] = check { def plan=fresh(); plan.stages.encoder.cpus=value; TaskRuntime.cpus(plan,'encoder') }
  }
  [null, 0, -1, 'NaN', 'Infinity', 'auto', 1e-30, 1e30].eachWithIndex { value,index ->
    result.invalid["memory_${index}"] = check { def plan=fresh(); plan.stages.encoder.memory_gb=value; TaskRuntime.memory(plan,'encoder') }
  }
  result.invalid.missing_stage = check { TaskRuntime.cpus(fresh(),'missing') }
  result.invalid.bad_stage = check { def plan=fresh(); plan.stages.encoder='bad'; TaskRuntime.memory(plan,'encoder') }
  result.invalid.invalid_create = check { TaskRuntime.create([profile:'balanced',cpu_budget:2,memory_budget_gb:2,stages:[encoder:[cpus:0,memory_gb:1]],updates:[:]],'cpu') }
  result.one_byte = check { def plan=fresh(); plan.stages.small.memory_gb=new BigDecimal('0.000000000931322574615478515625'); (TaskRuntime.memory(plan,'small') as nextflow.util.MemoryUnit).toBytes() }

  def values = [hardware_auto:false, hardware_profile:'balanced', _executor_max_cpus:8,
    _executor_max_memory_gb:8, cell_profiles_cpus:3, cell_profiles_memory_gb:4,
    tissue_hierarchy_cpus:2, tissue_hierarchy_memory:'512 MB']
  result.artifacts = TaskRuntime.forArtifacts(values)
  def low = new LinkedHashMap(values); low._executor_max_cpus=1; low._executor_max_memory_gb=1
  result.low_artifacts = TaskRuntime.forArtifacts(low)
  def high = new LinkedHashMap(values)
  high._executor_max_cpus=100000; high.cell_profiles_cpus=100000
  high._executor_max_memory_gb=100000; high.cell_profiles_memory_gb=100000
  result.host = [cpus:Runtime.runtime.availableProcessors(), memory:HostRuntime.memoryGb(), plan:TaskRuntime.forArtifacts(high)]
  result.artifact_errors = [:]
  ['_executor_max_cpus':0, '_executor_max_memory_gb':'auto', 'cell_profiles_cpus':1.5,
   'cell_profiles_memory_gb':-1].each { key,value ->
    result.artifact_errors[key] = check { def invalid=new LinkedHashMap(values); invalid[key]=value; TaskRuntime.forArtifacts(invalid) }
  }
  println 'TASK_RUNTIME_JSON=' + JsonOutput.toJson(result)
}
""")
    line = next(line for line in result.stdout.splitlines() if line.startswith("TASK_RUNTIME_JSON="))
    return json.loads(line.split("=", 1)[1])


def test_serialized_plan_preserves_effective_caps_and_detaches_source_maps(probe):
    data = probe["roundtrip"]
    assert data["cpu"] == data["original_cpu"] == 3
    assert data["memory"] == "1.5 GB" and data["small"] == "0.25 GB"
    assert data["device"] == "gpu" and data["profile"] == "balanced"
    assert data["original_zero"] == 0
    assert json.loads(data["encoded"])["schema_version"] == 1


def test_settings_preserve_false_zero_and_empty_but_fallback_for_null(probe):
    assert probe["settings"] == {"disabled": False, "zero": 0, "empty": "", "absent": "fallback", "missing": "fallback"}


def test_invalid_schema_and_resource_values_fail_explicitly(probe):
    failures = {name: result for name, result in probe["invalid"].items() if result["ok"]}
    assert failures == {}, f"Accepted invalid runtime inputs: {failures}"
    assert all(result["message"] for result in probe["invalid"].values())
    assert probe["one_byte"] == {"ok": True, "value": 1}


def test_artifact_plan_respects_profile_executor_and_host_caps(probe):
    plan = probe["artifacts"]
    assert plan["compute_device"] == "cpu"
    assert plan["cpu_budget"] <= 3 and plan["memory_budget_gb"] <= 4
    assert plan["stages"]["hierarchy_features"]["memory_gb"] == 0.5
    low = probe["low_artifacts"]
    assert low["cpu_budget"] == 1 and low["memory_budget_gb"] == 1
    assert all(value["cpus"] <= 1 and value["memory_gb"] <= 1 for value in low["stages"].values())
    assert all(value <= 1 for name, value in low["settings"].items() if name.endswith("_memory_gb"))
    assert low["settings"]["tissue_hierarchy_memory"] == "0.5 GB"
    host = probe["host"]
    assert host["plan"]["cpu_budget"] <= host["cpus"]
    assert host["plan"]["memory_budget_gb"] <= host["memory"]
    assert all(not value["ok"] for value in probe["artifact_errors"].values())


def test_fractional_memory_and_budget_cap_reach_actual_task(tmp_path):
    run_nextflow(tmp_path, r'''nextflow.enable.dsl=2
process PROBE {
  cpus { TaskRuntime.cpus(runtime_plan,'probe') }
  memory { TaskRuntime.memory(runtime_plan,'probe') }
  input:
  val runtime_plan
  output:
  path 'resource_probe.txt'
  script:
  """
  printf '%s\n' 'cpus=${task.cpus}' 'bytes=${task.memory.toBytes()}' 'setting=${TaskRuntime.setting(runtime_plan,"flag",true)}' > resource_probe.txt
  """
}
workflow {
  def plan = [schema_version:1,compute_device:'cpu',profile:'balanced',cpu_budget:1,memory_budget_gb:0.5,
    stages:[probe:[cpus:8,memory_gb:4]],settings:[flag:false]]
  PROBE(plan)
}
''', trace=True)
    with (tmp_path / "trace.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["status"] == "COMPLETED" and rows[0]["cpus"] == "1" and rows[0]["memory"] == "512 MB"
    output = next((tmp_path / "work").glob("*/*/resource_probe.txt")).read_text()
    assert output.splitlines() == ["cpus=1", "bytes=536870912", "setting=false"]
