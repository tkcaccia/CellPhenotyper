import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from write_pipeline_execution_reports import preserve_trace, summarize_trace  # noqa: E402


def write_trace(path: Path, process: str, realtime: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["process", "status", "realtime", "peak_rss"])
        writer.writerow([process, "COMPLETED", realtime, "2 GB"])


def test_targeted_run_does_not_replace_full_pipeline_trace(tmp_path: Path) -> None:
    execution_dir = tmp_path / "00_execution"
    execution_dir.mkdir()
    current_trace = execution_dir / "trace.tsv"

    write_trace(current_trace, "FULL_PROCESS", "10m")
    trace_source, metadata = preserve_trace(
        execution_dir, "full_run", True, "convert", "titan"
    )
    assert trace_source.name == "full_pipeline_trace.tsv"
    assert metadata["run_name"] == "full_run"

    write_trace(current_trace, "TARGETED_PROCESS", "5s")
    trace_source, metadata = preserve_trace(
        execution_dir, "targeted_run", True, "grandqc", "grandqc"
    )
    assert metadata["run_name"] == "full_run"
    assert summarize_trace(trace_source)[0]["process"] == "FULL_PROCESS"
    assert (execution_dir / "run_traces" / "targeted_run_grandqc_to_grandqc.tsv").exists()

    stored = json.loads((execution_dir / "full_pipeline_run.json").read_text())
    assert stored["start_point"] == "convert"
    assert len(list((execution_dir / "run_traces").glob("*.tsv"))) == 2
