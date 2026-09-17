"""Actual task-directive probes; stubs never execute model/scientific code."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from test_tissue_hierarchy_workflow import workflow_fixture

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "build_spatial_cell_profiles": "cell_profiles",
    "fit_cohort_niches": "cohort_niches",
    "link_cell_tissue_hierarchy": "cell_tissue_links",
    "map_cell_reference_atlas": "reference_mapping",
    "map_region_reference_atlas": "region_reference_mapping",
    "export_spatialdata": "spatialdata",
    "prepare_hierarchy_features": "hierarchy_features",
    "discover_tissue_hierarchy": "hierarchy_discovery",
}


def test_atlas_module_resource_directives_only_consume_explicit_plan():
    for module, stage in STAGES.items():
        source = (ROOT / f"modules/{module}.nf").read_text()
        assert source.count("val(runtime_plan)") == 1
        assert source.index("tuple val(") < source.index("val(runtime_plan)") < source.index("output:")
        assert f"cpus {{ TaskRuntime.cpus(runtime_plan, '{stage}') }}" in source
        assert f"memory {{ TaskRuntime.memory(runtime_plan, '{stage}') }}" in source
        assert "params._executor_max" not in source
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            assert f"{variable}=${{task.cpus}}" in source
    feature_module = (ROOT / "modules/prepare_hierarchy_features.nf").read_text()
    assert "--device '${params.tissue_hierarchy_device}'" in feature_module
    assert "TaskRuntime.device" not in feature_module


@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow is unavailable")
def test_all_actual_atlas_tasks_obey_plan_despite_conflicting_global_caps(tmp_path):
    workflow_fixture(tmp_path)
    for module in STAGES:
        shutil.copyfile(ROOT / f"modules/{module}.nf", tmp_path / f"modules/{module}.nf")
    for index in range(16):
        (tmp_path / f"input-{index}.dat").write_bytes(b"stub-only immutable placeholder")
    values = json.loads((tmp_path / "params.json").read_text())
    values.update({"_executor_max_cpus": 999, "_executor_max_memory_gb": 999,
        "cell_profiles_cpus": 999, "cell_profiles_memory_gb": 999,
        "tissue_hierarchy_cpus": 999, "tissue_hierarchy_memory": "999 GB",
        "cell_atlas_python": "MUST_NOT_EXECUTE_PYTHON", "spatialdata_python": "MUST_NOT_EXECUTE_PYTHON",
        "tissue_hierarchy_python": "MUST_NOT_EXECUTE_MODEL_PYTHON"})
    (tmp_path / "params.json").write_text(json.dumps(values))
    imports = "\n".join(f"include {{ {module.upper()} }} from './modules/{module}'" for module in STAGES)
    (tmp_path / "workflow.nf").write_text("nextflow.enable.dsl=2\n" + imports + "\n" + """
workflow {
    source = (0..<16).collect { index -> file("input-${index}.dat", checkIfExists:true) }
    allocations = HardwarePolicy.resolve([hardware_auto:false,cell_profiles_cpus:1,cell_profiles_memory_gb:2,
        tissue_hierarchy_cpus:1,tissue_hierarchy_memory:'512 MB'],2,4,false,0d)
    plan = TaskRuntime.create(allocations,'cpu')
    BUILD_SPATIAL_CELL_PROFILES(Channel.of(tuple('s1',source[0],source[1],source[2],source[3],source[4],source[5],
        source[6],source[7],source[8],source[9],source[10],source[11],source[12],[:])),plan)
    FIT_COHORT_NICHES(Channel.of(tuple(['s1','s2'],[source[0],source[1]])),plan)
    LINK_CELL_TISSUE_HIERARCHY(Channel.of(tuple('s1',source[0],source[1],source[2],source[3],false)),plan)
    MAP_CELL_REFERENCE_ATLAS(Channel.of(tuple('s1',source[0],source[1],PipelineHelpers.atlasTaskFingerprint('cell_reference_mapping',projectDir,[source[0],source[1]]))),plan)
    MAP_REGION_REFERENCE_ATLAS(Channel.of(tuple('s1::tile','s1','tile',source[0],source[1],PipelineHelpers.atlasTaskFingerprint('region_reference_mapping',projectDir,[source[0],source[1]]))),plan)
    EXPORT_SPATIALDATA(Channel.of(tuple('s1',source[0],source[1],source[2],source[3],source[4],source[5],source[6],
        'crop_pixels',false,[source[7]],[source[8]],[packages:false,shapes:false],source[9],false,
        [source[10]],[source[11]],[cell:false,region:false],source[9],false,PipelineHelpers.atlasTaskFingerprint('spatialdata_export',projectDir,[source[0],source[7],source[8],source[9]]))),plan)
    PREPARE_HIERARCHY_FEATURES(Channel.of(tuple('s1',source[0],source[1],source[2],source[3],source[4],
        source[5],source[6],'model.safetensors')),plan)
    DISCOVER_TISSUE_HIERARCHY(Channel.of(tuple('s1::tile','s1','tile',source[0],source[1],source[2],
        source[3],source[4],source[5],source[6],source[7],source[8])),plan)
}
""")
    (tmp_path / "probe.config").write_text("""
trace { fields='task_id,name,status,exit,cpus,memory'; raw=true }
executor { name='local'; cpus=2; memory='4 GB'; queueSize=2 }
""")
    result = subprocess.run(["nextflow", "-log", str(tmp_path / "probe.log"), "run", "workflow.nf",
        "-c", "probe.config", "-params-file", "params.json", "-stub-run", "-ansi-log", "false",
        "-with-trace", "trace.tsv", "-work-dir", "work"], cwd=tmp_path,
        env={**os.environ, "NXF_OFFLINE": "true"}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    with (tmp_path / "trace.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 8
    for row in rows:
        assert row["status"] == "COMPLETED" and row["exit"] == "0"
        assert row["cpus"] == "1"
        hierarchy = row["name"].startswith(("PREPARE_HIERARCHY_FEATURES", "DISCOVER_TISSUE_HIERARCHY"))
        assert int(row["memory"]) == (512 * 1024**2 if hierarchy else 2 * 1024**3)
