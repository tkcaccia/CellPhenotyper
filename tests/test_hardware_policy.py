import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow is unavailable")
def test_hardware_policy_respects_budgets_and_scales_profiles(tmp_path: Path) -> None:
    project = tmp_path / "policy_probe"
    (project / "lib").mkdir(parents=True)
    shutil.copy2(ROOT / "lib" / "HardwarePolicy.groovy", project / "lib" / "HardwarePolicy.groovy")
    (project / "main.nf").write_text(
        """nextflow.enable.dsl=2
import groovy.json.JsonOutput
workflow {
  def fast = HardwarePolicy.resolve([hardware_auto:true, hardware_profile:'auto'], 32, 96, true, 80.0d)
  def small = HardwarePolicy.resolve([hardware_auto:true, hardware_profile:'auto'], 2, 4, false, 0.0d)
  println 'POLICY_JSON=' + JsonOutput.toJson([fast:fast, small:small])
}
""",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["NXF_OFFLINE"] = "true"
    completed = subprocess.run(
        ["nextflow", "run", str(project / "main.nf")],
        cwd=project,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
        check=True,
    )
    line = next(item for item in completed.stdout.splitlines() if item.startswith("POLICY_JSON="))
    payload = json.loads(line.split("=", 1)[1])

    assert payload["fast"]["profile"] == "aggressive"
    assert payload["small"]["profile"] == "conservative"
    assert all(stage["cpus"] <= 32 and stage["memory_gb"] <= 96 for stage in payload["fast"]["stages"].values())
    assert all(stage["cpus"] <= 2 and stage["memory_gb"] <= 4 for stage in payload["small"]["stages"].values())
    assert payload["fast"]["stages"]["cell_consensus"]["cpus"] == 2
    assert "uni2_route_compare" in payload["fast"]["stages"]
    assert "cluster_assessment" in payload["fast"]["stages"]
    assert payload["small"]["stages"]["cluster_assessment"]["memory_gb"] <= 4
    assert payload["fast"]["runtime"]["kodama_n_cores"] == payload["fast"]["stages"]["kodama"]["cpus"]


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow is unavailable")
def test_atlas_policy_strict_units_low_budgets_and_shared_caps(tmp_path):
    (tmp_path / "lib").mkdir()
    for source in (ROOT / "lib").glob("*.groovy"):
        shutil.copy2(source, tmp_path / "lib" / source.name)
    (tmp_path / "main.nf").write_text("""nextflow.enable.dsl=2
import groovy.json.JsonOutput
workflow {
  def tiny = HardwarePolicy.resolve([hardware_auto:true, hardware_profile:'aggressive',
    cell_profiles_cpus:99,cell_profiles_memory_gb:99,tissue_hierarchy_cpus:99,tissue_hierarchy_memory:'99 GB'],1,2,true,80d)
  def manual = HardwarePolicy.resolve([hardware_auto:false,cell_profiles_cpus:3,cell_profiles_memory_gb:7,
    tissue_hierarchy_cpus:2,tissue_hierarchy_memory:'1536 MB',tissue_hierarchy_device:'cpu'],64,96,true,80d)
  def fractional = HardwarePolicy.resolve([hardware_auto:true,cell_profiles_cpus:1,cell_profiles_memory_gb:2,
    tissue_hierarchy_cpus:1,tissue_hierarchy_memory:'512 MB'],32,96,false,0d)
  def invalid = []
  [['tissue_hierarchy_memory','invalid'],['tissue_hierarchy_memory','16'],['tissue_hierarchy_memory','0 GB'],
   ['tissue_hierarchy_memory','-1 GB'],['tissue_hierarchy_memory','NaN GB'],['tissue_hierarchy_memory','16GiB'],
   ['tissue_hierarchy_memory','1 TB'],['tissue_hierarchy_memory','0.000000000001 MB'],
   ['cell_profiles_memory_gb','many'],['cell_profiles_memory_gb',0],['cell_profiles_cpus',0],
   ['tissue_hierarchy_cpus','2.5']].each { entry ->
    def values=[hardware_auto:false]; values[entry[0]]=entry[1]
    try { HardwarePolicy.resolve(values,4,16,false,0d); invalid.add([key:entry[0],value:entry[1],rejected:false]) }
    catch (IllegalArgumentException error) { invalid.add([key:entry[0],value:entry[1],rejected:true,error:error.message]) }
  }
  def plan = TaskRuntime.create(fractional,'cpu')
  def requested = '512 MB' as nextflow.util.MemoryUnit
  def resolved = TaskRuntime.memory(plan,'hierarchy_discovery') as nextflow.util.MemoryUnit
  println 'ATLAS_POLICY_JSON=' + JsonOutput.toJson([tiny:tiny,manual:manual,fractional:fractional,
    invalid:invalid,requested_bytes:requested.toBytes(),resolved_bytes:resolved.toBytes(),plan:plan])
}
""")
    result = subprocess.run(["nextflow", "run", str(tmp_path / "main.nf"), "-ansi-log", "false"],
        cwd=tmp_path, env={**os.environ, "NXF_OFFLINE": "true"}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(next(line.split("=", 1)[1] for line in result.stdout.splitlines()
        if line.startswith("ATLAS_POLICY_JSON=")))
    profile_stages = ("cell_profiles", "cell_tissue_links", "reference_mapping", "region_reference_mapping", "spatialdata")
    hierarchy_stages = ("hierarchy_features", "hierarchy_discovery")
    for stage in profile_stages + hierarchy_stages:
        assert payload["tiny"]["stages"][stage]["cpus"] == 1
        assert 0 < payload["tiny"]["stages"][stage]["memory_gb"] <= 2
    for stage in profile_stages:
        assert payload["manual"]["stages"][stage] == {"cpus": 3, "memory_gb": 7}
        assert payload["fractional"]["stages"][stage]["cpus"] == 1
        assert payload["fractional"]["stages"][stage]["memory_gb"] <= 2
    for stage in hierarchy_stages:
        assert payload["manual"]["stages"][stage] == {"cpus": 2, "memory_gb": 1.5}
        assert payload["fractional"]["stages"][stage] == {"cpus": 1, "memory_gb": .5}
    assert payload["manual"]["updates"]["tissue_hierarchy_memory"] == "1.5 GB"
    for section in ("tiny", "manual", "fractional"):
        policy = payload[section]
        assert policy["updates"]["cell_profiles_cpus"] == max(policy["stages"][name]["cpus"] for name in profile_stages)
        assert policy["updates"]["cell_profiles_memory_gb"] == max(policy["stages"][name]["memory_gb"] for name in profile_stages)
        assert "tissue_hierarchy_device" not in policy["updates"]
    assert all(item["rejected"] and item["key"] in item["error"] for item in payload["invalid"])
    assert payload["requested_bytes"] == payload["resolved_bytes"] == 512 * 1024**2
