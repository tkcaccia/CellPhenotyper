import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from write_pipeline_execution_reports import list_files, preserve_trace, stage_summary, summarize_trace  # noqa: E402


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


def test_project_paths_remain_in_published_tree_for_relative_links(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    target = tmp_path / "work" / "task" / "consensus_sample"
    target.mkdir(parents=True)
    (target / "objects.csv").write_text("label,x,y\n1,1,1\n")
    published_parent = outdir / "03d_cell_consensus" / "sample"
    published_parent.mkdir(parents=True)
    published_dir = published_parent / "consensus_sample"
    published_dir.symlink_to(target, target_is_directory=True)

    records = list_files(outdir / "03d_cell_consensus")
    assert len(records) == 1
    record = records[0]
    assert record["absolute_path"] == str(published_dir / "objects.csv")
    assert record["resolved_target_path"] == str(target / "objects.csv")
    assert record["absolute_path"].startswith(str(outdir))

    consensus = next(row for row in stage_summary(outdir) if row["id"] == "cell_consensus")
    assert consensus["key_files"] == [str(published_dir / "objects.csv")]
