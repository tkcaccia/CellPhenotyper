"""Real Nextflow module commands, with argument recorders instead of R/model work."""

import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULES = ("run_kodama_analysis.nf", "run_gigatime_kodama.nf")


@pytest.mark.parametrize("allocated,requested,expected,representation,export_requested", [
    (8, 8, 8, "umap2d", False), (2, 8, 2, "kodama_graph", False),
    (2, 1, 1, "pca", True), (2, 0, 2, "pca", False),
])
def test_actual_kodama_commands_obey_explicit_runtime_plan(tmp_path, allocated, requested, expected, representation, export_requested):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    if allocated > (os.cpu_count() or 1):
        pytest.skip(f"Allocation probe needs {allocated} host CPUs")
    (tmp_path / "modules").mkdir()
    (tmp_path / "lib").mkdir()
    (tmp_path / "empty").mkdir()
    (tmp_path / "objects.csv").write_text("label_id,x,y\n")
    (tmp_path / "bin").symlink_to(ROOT / "bin", target_is_directory=True)
    for module in MODULES:
        shutil.copy2(ROOT / "modules" / module, tmp_path / "modules" / module)
    for helper in ("PipelineHelpers.groovy", "TaskRuntime.groovy", "HardwarePolicy.groovy", "HostRuntime.groovy", "ProcessCode.groovy"):
        shutil.copy2(ROOT / "lib" / helper, tmp_path / "lib" / helper)

    recorder = tmp_path / "record_r_arguments"
    recorder.write_text(f"#!{sys.executable}\n" + """import json, sys
from pathlib import Path
if sys.argv[1:] == ['-']:
    sys.stdin.read()  # The module's R package preflight, not executed as R.
else:
    output = Path('gigatime_kodama_output' if 'load_gigatime' in sys.argv[1] or sys.argv[2].startswith('gigatime_') else 'kodama_output')
    output.mkdir(exist_ok=True)
    name = 'analysis_arguments.json' if '--n-cores' in sys.argv else 'loader_arguments.json'
    (output / name).write_text(json.dumps({'argv': sys.argv[1:], 'used_model': False}))
""")
    recorder.chmod(0o755)
    combined = "\n".join((ROOT / "modules" / name).read_text() for name in MODULES)
    params = {name: 0 for name in re.findall(r"params\.([A-Za-z_][A-Za-z_0-9]*)", combined)}
    params.update({
        "outdir_base": str(tmp_path / "published"), "publish_dir_mode": "copy",
        # These initial limits deliberately disagree with the resolved value
        # input; they must not silently override its stage allocation.
        "_executor_max_cpus": 1, "_executor_max_memory_gb": 1,
        "r_cpus": 1, "r_memory_gb": 1, "r_time": "1m", "kodama_n_cores": 1,
        "r_data_loader_script": "bin/load_kodama_rawdata.R",
        "gigatime_kodama_loader_script": "bin/load_gigatime_kodama_rawdata.R",
        "r_script": "bin/run_kodama_analysis.R", "kodama_rscript": str(recorder),
        "kodama_r_library_dir": str(tmp_path / "empty"), "kodama_embedding_mode": "tile,inner_square",
        "kodama_dims_to_run": 20, "kodama_spark_top_features": 100, "kodama_landmarks": 10000,
        "kodama_ncomp": 50, "kodama_save_selected_features": False,
        "kodama_backend": "cpu", "kodama_gpu_device": 0,
        "cluster_representation": representation, "kodama_export_native_graph": export_requested,
    })
    (tmp_path / "params.json").write_text(json.dumps(params))
    (tmp_path / "main.nf").write_text(f"""nextflow.enable.dsl=2
include {{ RUN_KODAMA_ANALYSIS }} from './modules/run_kodama_analysis'
include {{ RUN_GIGATIME_KODAMA }} from './modules/run_gigatime_kodama'
workflow {{
  def runtime_plan = [schema_version:1, compute_device:'cpu', profile:'balanced',
    cpu_budget:{allocated}, memory_budget_gb:2, stages:[kodama:[cpus:{allocated}, memory_gb:2]],
    settings:[r_cpus:{allocated}, r_memory_gb:2, kodama_n_cores:{requested}]]
  // Late params remain non-authoritative, matching the real workflow's order.
  params.r_cpus = {allocated}
  params.kodama_n_cores = {requested}
  def empty = file('empty', checkIfExists:true)
  RUN_KODAMA_ANALYSIS(Channel.of(tuple('uni2_probe', empty, empty, empty, empty, file('objects.csv'))), runtime_plan)
  RUN_GIGATIME_KODAMA(Channel.of(tuple('marker_probe', empty)), runtime_plan)
}}
""")
    (tmp_path / "nextflow.config").write_text("""process.executor = 'local'
process.maxForks = 1
executor.queueSize = 1
trace.fields = 'task_id,name,status,cpus,memory'
""")
    completed = subprocess.run([
        nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "main.nf"),
        "-params-file", str(tmp_path / "params.json"), "-ansi-log", "false",
        "-with-trace", str(tmp_path / "trace.tsv"), "-work-dir", str(tmp_path / "work"),
    ], cwd=tmp_path, env={**os.environ, "NXF_OFFLINE": "true"}, text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    paths = [
        tmp_path / "published/10_kodama/uni2_probe/kodama_output/analysis_arguments.json",
        tmp_path / "published/10_kodama/marker_probe/gigatime/gigatime_kodama_output/analysis_arguments.json",
    ]
    for path in paths:
        recorded = json.loads(path.read_text())
        assert recorded["used_model"] is False
        argv = recorded["argv"]
        assert argv[argv.index("--n-cores") + 1] == str(expected)
        assert argv[argv.index("--kodama-ncomp") + 1] == "50"
        assert argv[argv.index("--export-native-graph") + 1] == str(export_requested or representation == "kodama_graph").lower()
        # Keep the KODAMA.matrix component input distinct from embedding/PCA dims.
        assert argv[argv.index("--dims-to-run") + 1] == "20"
    commands = list((tmp_path / "work").glob("*/*/.command.sh"))
    assert len(commands) == 2
    for path in commands:
        command = path.read_text()
        assert f"effective CPU cores={expected}; task allocation={allocated}" in command
        assert f"--n-cores {expected}" in command
        preflight = re.search(r'required_pkgs <- c\(([^)]*)\)', command).group(1)
        packages = set(re.findall(r'"([^"]+)"', preflight))
        assert ("Matrix" in packages) == (export_requested or representation == "kodama_graph")
        # CSV and binary UNI2 now both produce exact hashed input receipts,
        # independently of whether a native graph is requested.
        if "load_gigatime_kodama_rawdata.R" not in command:
            assert {"digest", "jsonlite"} <= packages
        else:
            assert ({"digest", "jsonlite"} <= packages) == (export_requested or representation == "kodama_graph")
    with (tmp_path / "trace.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 2
    assert all(row["status"] == "COMPLETED" and row["cpus"] == str(allocated) and row["memory"] == "2 GB" for row in rows)
