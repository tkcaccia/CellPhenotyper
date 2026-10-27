"""The continuation DAG has only bounded CPU manifest/Bora/export edges."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow is unavailable")
@pytest.mark.parametrize("export", [False, True])
def test_stub_dag_never_makes_durable_scientific_outputs(tmp_path: Path, export: bool):
    spec = tmp_path / "inputs.json"
    spec.write_text(json.dumps({
        "schema": "cellphenotyper-bora-global-inputs-v1",
        "evidence_class": "verified_gpu_inventory",
        "run_root": str(tmp_path), "producer_source": str(tmp_path),
        "inputs": {}, "limits": {"memory_budget_bytes": 1024**3},
    }))
    sif = tmp_path / "stub-cpu.sif"
    sif.write_bytes(b"stub-only; never used as a runtime")
    manifest_dir = tmp_path / "durable-manifest"
    bora_dir = tmp_path / "durable-bora"
    command = ["nextflow", "run", str(ROOT / "bora_orchestrate_fullslide.nf"),
               "-stub-run", "--bora_fullslide_input_spec", str(spec),
               "--bora_fullslide_manifest_output", str(manifest_dir),
               "--bora_fullslide_output", str(bora_dir),
               "--bora_fullslide_cpu_sif", str(sif),
               "--bora_fullslide_memory_gb", "2", "--compute_device", "cpu",
               "-work-dir", str(tmp_path / "work")]
    if export:
        base = tmp_path / "existing-project.json"
        base.write_text("stub-only")
        command += ["--bora_fullslide_base_project", str(base),
                    "--bora_fullslide_project_output", str(tmp_path / "new-project.json")]
    env = dict(os.environ, NXF_OFFLINE="true", NXF_ANSI_LOG="false",
               NXF_OPTS="-Xms64m -Xmx512m")
    run = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True,
                         text=True, timeout=90, check=False)
    output = run.stdout + run.stderr
    assert run.returncode == 0, output
    assert output.count("Submitted process > BUILD_BORA_GLOBAL_MANIFEST") == 1
    assert output.count("Submitted process > RUN_ADMITTED_BORA_GLOBAL") == 1
    assert output.count("Submitted process > EXPORT_ADMITTED_BORA_PROJECT") == int(export)
    assert not manifest_dir.exists() and not bora_dir.exists()
    assert not (tmp_path / "new-project.json").exists()
