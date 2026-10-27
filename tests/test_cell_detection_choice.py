"""Detector choice contracts: no models, remote jobs or GPU work are launched."""
from __future__ import annotations

import json
import csv
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def validate(payload: dict) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "nextflow_schema.json").read_text())
    jsonschema.Draft7Validator(schema).validate(payload)


@pytest.mark.parametrize("mode,device", [("stardist", "cpu"), ("stardist", "gpu"),
                                         ("cellvitpp", "gpu"), ("cellvitpp", "auto")])
def test_schema_accepts_explicit_detector_choice(mode: str, device: str) -> None:
    validate({"cell_detection_mode": mode, "compute_device": device})


@pytest.mark.parametrize("payload", [
    {"cell_detection_mode": "cellvitpp", "compute_device": "cpu"},
    {"cell_detection_mode": "consensus", "compute_device": "gpu"},
    {"cell_detection_mode": "both", "compute_device": "gpu"},
    {"cell_detection_mode": "any_detector", "compute_device": "gpu"},
])
def test_schema_rejects_invalid_or_cpu_only_cellvit_choice(payload: dict) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(jsonschema.ValidationError):
        validate(payload)


def nextflow(project: Path, source: str, *extra: str) -> subprocess.CompletedProcess:
    if shutil.which("nextflow") is None:
        pytest.skip("Nextflow unavailable")
    project.mkdir(parents=True, exist_ok=True)
    script = project / "main.nf"
    script.write_text(source)
    # No project config is inherited: synthetic stubs always run as tiny local
    # CPU processes and never pull a container or invoke a model executable.
    (project / "nextflow.config").write_text("process.executor='local'\nprocess.cpus=1\nprocess.memory='128 MB'\nprocess.container=null\n")
    return subprocess.run(
        ["nextflow", "run", str(script), "-c", str(project / "nextflow.config"), *extra],
        cwd=project, env=dict(os.environ, NXF_OFFLINE="true", NXF_ANSI_LOG="false"),
        capture_output=True, text=True, timeout=90,
    )


def test_actual_nextflow_resolves_detector_sets_and_legacy_fallback(tmp_path: Path) -> None:
    source = f"""nextflow.enable.dsl=2
include {{ resolveCellDetectionSettings }} from '{ROOT}/modules/cell_detection_settings'
workflow {{
  def star = resolveCellDetectionSettings([cell_detection_mode:'stardist'], 'cpu')
  assert star.mode == 'stardist' && star.run_stardist && !star.run_cellvitpp && !star.run_hovernet && !star.primary_cellvitpp
  def cellvit = resolveCellDetectionSettings([cell_detection_mode:'cellvitpp'], 'gpu')
  assert cellvit.mode == 'cellvitpp' && !cellvit.run_stardist && cellvit.run_cellvitpp && !cellvit.run_hovernet && cellvit.primary_cellvitpp
  assert resolveCellDetectionSettings([cell_consensus_enable:false], 'cpu').mode == 'stardist'
  assert resolveCellDetectionSettings([cell_consensus_enable:true], 'gpu').mode == 'cellvitpp'
  assert resolveCellDetectionSettings([:], 'gpu').mode == 'cellvitpp'
  assert resolveCellDetectionSettings([cell_detection_mode:'stardist',cell_consensus_enable:true], 'cpu').mode == 'stardist'
  assert resolveCellDetectionSettings([cell_detection_mode:'cellvitpp',cell_consensus_enable:false], 'gpu').mode == 'cellvitpp'
  println 'DETECTOR_SELECTION_MATRIX_OK'
}}
"""
    result = nextflow(tmp_path / "matrix", source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DETECTOR_SELECTION_MATRIX_OK" in result.stdout


def test_actual_nextflow_parses_production_detector_and_primary_adapter_modules(tmp_path: Path) -> None:
    source = f"""nextflow.enable.dsl=2
include {{ resolveCellDetectionSettings }} from '{ROOT}/modules/cell_detection_settings'
include {{ RUN_STARDIST_ROI_SEGMENTATION }} from '{ROOT}/modules/run_stardist_roi_segmentation'
include {{ RUN_CELLVITPP }} from '{ROOT}/modules/run_cellvitpp'
include {{ EXPORT_CELLVIT_PRIMARY_SEGMENTATION }} from '{ROOT}/modules/export_cellvit_primary_segmentation'
include {{ RUN_PRIMARY_CELLVIT }} from '{ROOT}/modules/cell_segmentation_cellvit'
include {{ writePipelineExecutionReport }} from '{ROOT}/modules/execution_reporting'
workflow {{
  assert resolveCellDetectionSettings([cell_detection_mode:'cellvitpp'], 'gpu').primary_cellvitpp
  println 'PRODUCTION_DETECTOR_MODULES_PARSED_OK'
}}
"""
    result = nextflow(tmp_path / "production_parse", source, "-lib", str(ROOT / "lib"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PRODUCTION_DETECTOR_MODULES_PARSED_OK" in result.stdout


@pytest.mark.parametrize("mode", ["cellvitpp"])
def test_actual_nextflow_refuses_gpu_detector_on_cpu(tmp_path: Path, mode: str) -> None:
    source = f"""nextflow.enable.dsl=2
include {{ resolveCellDetectionSettings }} from '{ROOT}/modules/cell_detection_settings'
workflow {{ resolveCellDetectionSettings([cell_detection_mode:'{mode}'], 'cpu') }}
"""
    result = nextflow(tmp_path / mode, source)
    assert result.returncode != 0
    assert "gpu" in (result.stdout + result.stderr).lower(), result.stdout + result.stderr


@pytest.mark.parametrize("mode,device,expected", [
    ("cellvitpp", "cpu", "cell_detection_mode=cellvitpp requires GPU execution"),
    ("consensus", "cpu", "cellvitpp or stardist"),
])
def test_main_workflow_rejects_invalid_detector_before_submitting_processes(tmp_path: Path, mode: str,
                                                                          device: str, expected: str) -> None:
    if shutil.which("nextflow") is None:
        pytest.skip("Nextflow unavailable")
    # Exercise the real top-level workflow's parameter guard. This invocation
    # deliberately has no usable GPU request or tissue input and must stop
    # before launching even a stub process.
    result = subprocess.run([
        "nextflow", "run", str(ROOT / "main.nf"), "-stub-run",
        "-work-dir", str(tmp_path / "work"),
        "--cell_detection_mode", mode, "--compute_device", device,
        "--end_point", "convert", "--outdir_base", str(tmp_path / "output"),
        "--runtime_image_mode", "manual", "--max_cpus", "1", "--max_memory_gb", "2",
    ], cwd=tmp_path, env=dict(os.environ, NXF_OFFLINE="true", NXF_ANSI_LOG="false"),
       capture_output=True, text=True, timeout=90)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert expected in output, output
    assert "Submitted process" not in output
    assert not list((tmp_path / "work").glob("**/.command.sh"))


@pytest.mark.parametrize("mode,expected", [
    ("stardist", {"stardist"}),
    ("cellvitpp", {"cellvitpp"}),
])
def test_actual_cpu_stub_dag_runs_only_selected_detector_set(tmp_path: Path, mode: str,
                                                            expected: set[str]) -> None:
    project = tmp_path / mode
    process_definitions = "\n".join(f"""
process TEST_{detector.upper()} {{
  publishDir '{project}/markers', mode:'copy'
  input:
  val sample
  output:
  path '{detector}.txt'
  script:
  'exit 99'
  stub:
  \"\"\"printf 'synthetic detector stub only\\n' > {detector}.txt\"\"\"
}}
""" for detector in ("stardist", "cellvitpp", "hovernet"))
    source = f"""nextflow.enable.dsl=2
include {{ resolveCellDetectionSettings }} from '{ROOT}/modules/cell_detection_settings'
{process_definitions}
workflow {{
  def selection = resolveCellDetectionSettings([cell_detection_mode:'{mode}'], 'gpu')
  if (selection.run_stardist) TEST_STARDIST(Channel.value('synthetic'))
  if (selection.run_cellvitpp) TEST_CELLVITPP(Channel.value('synthetic'))
  if (selection.run_hovernet) TEST_HOVERNET(Channel.value('synthetic'))
}}
"""
    result = nextflow(project, source, "-stub-run")
    assert result.returncode == 0, result.stdout + result.stderr
    actual = {path.stem for path in (project / "markers").glob("*.txt")}
    assert actual == expected


def test_actual_nextflow_rejects_stardist_with_cellvit_identification(tmp_path: Path) -> None:
    source = f"""nextflow.enable.dsl=2
include {{ resolveCellDetectionSettings }} from '{ROOT}/modules/cell_detection_settings'
workflow {{ resolveCellDetectionSettings([cell_detection_mode:'stardist',cell_identification_enable:true], 'gpu') }}
"""
    result = nextflow(tmp_path / "conflict", source)
    assert result.returncode != 0
    output = (result.stdout + result.stderr).lower()
    assert "stardist" in output and "cell_identification_enable" in output, output


def test_actual_nextflow_stardist_does_not_require_unselected_cellvit_cache(tmp_path: Path) -> None:
    source = f"""nextflow.enable.dsl=2
include {{ resolveCellDetectionSettings }} from '{ROOT}/modules/cell_detection_settings'
include {{ resolveCellIdentificationSettings }} from '{ROOT}/modules/cell_identification_settings'
workflow {{
  def requested = [cell_detection_mode:'stardist',cell_identification_enable:false,
    cellvit_classifier_cache:'/missing/unselected/cellvit',
    cellvit_cache_dir:'/missing/unselected/cellvit-models',
    cellvit_fullslide_runtime_sif:'/missing/unselected/runtime.sif']
  def detector = resolveCellDetectionSettings(requested, 'cpu')
  def identification = resolveCellIdentificationSettings(requested, 'cpu')
  assert detector.mode == 'stardist' && detector.run_stardist && !detector.run_cellvitpp
  assert !identification.enabled && !identification.review
  println 'STARDIST_UNSELECTED_CELLVIT_DEPENDENCIES_NOT_REQUIRED'
}}
"""
    result = nextflow(tmp_path / "unused_cache", source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STARDIST_UNSELECTED_CELLVIT_DEPENDENCIES_NOT_REQUIRED" in result.stdout
    assert not list((tmp_path / "unused_cache" / "work").glob("**/.command.sh"))


@pytest.mark.parametrize("mode", ["stardist", "cellvitpp"])
def test_main_cpu_stubs_route_one_detector_into_common_output_channels(tmp_path: Path, mode: str) -> None:
    if shutil.which("nextflow") is None:
        pytest.skip("Nextflow unavailable")
    image = tmp_path / "synthetic.ome.tif"
    tifffile = pytest.importorskip("tifffile")
    numpy = pytest.importorskip("numpy")
    tifffile.imwrite(image, numpy.full((8, 8, 3), 180, dtype=numpy.uint8),
                     photometric="rgb", resolution=(40000, 40000), resolutionunit="CENTIMETER")
    config = tmp_path / "local.config"
    config.write_text("""docker.enabled=false
singularity.enabled=false
process {
  executor='local'
  withName: '.*' {
    cpus=1
    memory='128 MB'
    container=''
    beforeScript=''
    afterScript=''
    maxForks=1
  }
}
""")
    trace = tmp_path / "trace.tsv"
    params = tmp_path / "params.json"
    params.write_text(json.dumps({
        "cell_detection_mode": mode, "compute_device": "gpu" if mode == "cellvitpp" else "cpu",
        "host_arch": "amd64", "image_input": str(image), "folder_input": "", "roi_geojson": "",
        "cellvit_input_scope": "roi", "start_point": "convert",
        "end_point": "cell_identification" if mode == "cellvitpp" else "stardist",
        "outdir_base": str(tmp_path / "output"), "runtime_image_mode": "manual",
        "storage_preflight_mode": "off", "tma_enable": False,
        "max_cpus": 1, "max_memory_gb": 2, "publish_dir_mode": "copy",
        "gpu_scheduler_enable": False,
    }))
    # The CellViT ROI route is selected only for this CPU stub dataflow probe;
    # full-slide campaigns retain their default and are never invoked here.
    # Symbolic GPU configuration admits that route, but no container/GPU/model
    # command is executed: -stub-run and local CPU process overrides are forced.
    command = [
        "nextflow", "run", str(ROOT / "main.nf"), "-stub-run", "-c", str(config),
        "-params-file", str(params), "-work-dir", str(tmp_path / "work"), "-with-trace", str(trace),
    ]
    result = subprocess.run(command, cwd=tmp_path,
        env=dict(os.environ, NXF_OFFLINE="true", NXF_ANSI_LOG="false"),
        capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    with trace.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows and all(row["status"] == "COMPLETED" for row in rows), rows
    processes = "\n".join(row["name"] for row in rows)
    for script in (tmp_path / "work").glob("**/.command.sh"):
        command_text = script.read_text()
        assert "acquire_gpu_slot" not in command_text
        assert not any(model in command_text for model in (
            "run_cellvitpp.py", "run_stardist_roi_segmentation.py", "run_grandqc_artifact_analysis.py"))
    assert "PREPARE_ANALYSIS_CROP" in processes and "CROP_GRANDQC_CLEAN_MASK" in processes
    assert "RUN_HOVERNET_MONUSAC" not in processes and "BUILD_CELL_CONSENSUS" not in processes
    if mode == "stardist":
        assert "RUN_STARDIST_ROI_SEGMENTATION" in processes
        assert sum("RUN_STARDIST_ROI_SEGMENTATION" in row["name"] for row in rows) == 1
        assert "RUN_CELLVITPP" not in processes
        assert list((tmp_path / "output" / "03_stardist").glob("**/labels.tif"))
    else:
        assert "RUN_CELLVITPP" in processes and "EXPORT_CELLVIT_PRIMARY_SEGMENTATION" in processes
        assert sum("RUN_CELLVITPP" in row["name"] for row in rows) == 1
        assert "RUN_STARDIST_ROI_SEGMENTATION" not in processes
        assert list((tmp_path / "output" / "03b_cellvitpp").glob("**/labels.tif"))
