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
