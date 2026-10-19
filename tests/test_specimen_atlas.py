import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from specimen_atlas import build_specimen_atlas, render_specimen_atlas  # noqa: E402


def record(path: Path, outdir: Path, stage_id: str, output_id: str) -> dict:
    return {
        "output_id": output_id,
        "stage_id": stage_id,
        "stage_title": stage_id.replace("_", " ").title(),
        "stage_folder": path.relative_to(outdir).parts[0],
        "relative_path": path.relative_to(outdir / path.relative_to(outdir).parts[0]).as_posix(),
        "absolute_path": str(path),
        "resolved_target_path": str(path),
        "size_bytes": path.stat().st_size,
    }


def test_atlas_groups_primary_and_cell_routes_and_preserves_semantics(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    execution = outdir / "00_execution"
    execution.mkdir(parents=True)
    sample = "sample&one"

    converted = outdir / "01_input" / sample / f"{sample}.converted_resolution.json"
    converted.parent.mkdir(parents=True)
    converted.write_text(
        json.dumps({"status": "pass", "width_px": 2000, "height_px": 1000, "effective_mpp": 0.5}),
        encoding="utf-8",
    )
    grid = outdir / "09_grid_tiles" / sample / f"{sample}_uni2_grid_preview.png"
    grid.parent.mkdir(parents=True)
    grid.write_bytes(b"grid")
    consensus = outdir / "03d_cell_consensus" / f"{sample}__cells" / "consensus_preview.png"
    consensus.parent.mkdir(parents=True)
    consensus.write_bytes(b"cells")
    uncertainty = outdir / "11_clustering" / sample / f"{sample}_cluster_kodama_uncertainty.png"
    uncertainty.parent.mkdir(parents=True)
    uncertainty.write_bytes(b"uncertainty")

    records = [
        record(converted, outdir, "input", "input_1"),
        record(grid, outdir, "grid_tiles", "grid_1"),
        record(consensus, outdir, "cell_consensus", "cells_1"),
        record(uncertainty, outdir, "clustering", "uncertainty_1"),
    ]
    signal_path = outdir / "02_grandqc" / sample / "qc.json"
    payload = build_specimen_atlas(
        execution_dir=execution,
        run_name="unsafe <run>",
        success=True,
        analysis_intent="tissue_domain_discovery",
        uni2_sampling_mode="both",
        cell_detection_mode="consensus",
        claim_ceiling="exploratory_description_only",
        project_records=records,
        quality_signals=[{"kind": "grandqc", "status": "pass", "details": "safe <detail>", "path": str(signal_path)}],
        uncertainty_register=[
            {
                "stage_id": "clustering",
                "stage_title": "Clustering",
                "stage_present": True,
                "uncertainty_implementation": "quantified_with_abstention",
                "decision_or_abstention": "abstain on ambiguity",
                "calibration_status": "not_calibrated",
                "interpretation_limit": "stability is not identity",
            }
        ],
    )

    assert [row["sample_id"] for row in payload["specimens"]] == [sample]
    specimen = payload["specimens"][0]
    assets = {
        layer["id"]: layer["assets"]
        for layer in specimen["layers"]
    }
    assert assets["domains"][0]["route"] == "grid tissue-domain primary; cell-centred auxiliary"
    assert assets["cells"][0]["route"] == "cell-centred"
    assert assets["uncertainty"][0]["output_id"] == "uncertainty_1"
    assert not assets["virtual_markers"]
    assert assets["cells"][0]["semantic"].startswith("predicted nuclear instances")
    assert specimen["quality_signals"][0]["evidence_href"].startswith("../02_grandqc/")
    assert str(tmp_path) not in json.dumps(payload)

    page = render_specimen_atlas(payload)
    assert "unsafe &lt;run&gt;" in page
    assert "sample&amp;one" in page
    assert "safe &lt;detail&gt;" in page
    assert "Uncertainty Coverage" in page
    assert "Not produced in this stage window" in page
    assert "uncalibrated virtual-marker score; not measured protein abundance" in page


def test_execution_report_writes_and_indexes_specimen_atlas(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    converted = outdir / "01_input" / "sample_a" / "sample_a.converted_resolution.json"
    converted.parent.mkdir(parents=True)
    converted.write_text(json.dumps({"status": "pass", "effective_mpp": 0.5}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "bin" / "write_pipeline_execution_reports.py"),
            "--outdir", str(outdir),
            "--run-name", "atlas integration",
            "--success", "true",
            "--start-point", "convert",
            "--end-point", "convert",
            "--analysis-intent", "exploratory",
            "--uni2-sampling-mode", "grid",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    execution = outdir / "00_execution"
    assert (execution / "specimen_atlas.html").exists()
    assert (execution / "specimen_atlas.json").exists()
    assert "Specimen atlas" in (execution / "index.html").read_text(encoding="utf-8")
    report = json.loads((execution / "final_report.json").read_text(encoding="utf-8"))
    assert report["specimen_atlas_html"] == str(execution / "specimen_atlas.html")
    manifest = json.loads((execution / "project_outputs.json").read_text(encoding="utf-8"))
    atlas_records = [
        row for row in manifest["records"]
        if row["relative_path"] in {"specimen_atlas.html", "specimen_atlas.json"}
    ]
    assert len(atlas_records) == 2
    assert all(row["output_id"].startswith("execution_") for row in atlas_records)


def test_quality_signals_are_specimen_scoped_with_global_fallback(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    execution = outdir / "00_execution"
    execution.mkdir(parents=True)
    records = []
    for index, sample in enumerate(("sample_a", "sample_b"), start=1):
        source = outdir / "01_input" / sample / f"{sample}.converted_resolution.json"
        source.parent.mkdir(parents=True)
        source.write_text("{}", encoding="utf-8")
        records.append(record(source, outdir, "input", f"input_{index}"))

    payload = build_specimen_atlas(
        execution_dir=execution,
        run_name="two specimens",
        success=True,
        analysis_intent="exploratory",
        uni2_sampling_mode="grid",
        cell_detection_mode="consensus",
        claim_ceiling="exploratory_description_only",
        project_records=records,
        quality_signals=[
            {"kind": "qc_a", "status": "pass", "details": "A", "path": str(outdir / "02_grandqc" / "sample_a" / "qc.json")},
            {"kind": "qc_b", "status": "review", "details": "B", "path": str(outdir / "02_grandqc" / "sample_b" / "qc.json")},
            {"kind": "human_review", "status": "pending", "details": "run level", "path": str(execution / "human_review.json")},
        ],
        uncertainty_register=[],
    )

    signals = {
        specimen["sample_id"]: {row["kind"] for row in specimen["quality_signals"]}
        for specimen in payload["specimens"]
    }
    assert signals["sample_a"] == {"qc_a", "human_review"}
    assert signals["sample_b"] == {"qc_b", "human_review"}


def test_base_sample_detector_assets_retain_cell_characterization_route(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    execution = outdir / "00_execution"
    execution.mkdir(parents=True)
    source = outdir / "01_input" / "sample" / "sample.converted_resolution.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    overlay = outdir / "03_stardist" / "sample" / "stardist_out" / "overlay.png"
    overlay.parent.mkdir(parents=True)
    overlay.write_bytes(b"overlay")

    payload = build_specimen_atlas(
        execution_dir=execution,
        run_name="routes",
        success=True,
        analysis_intent="tissue_domain_discovery",
        uni2_sampling_mode="grid",
        cell_detection_mode="consensus",
        claim_ceiling="exploratory_description_only",
        project_records=[
            record(source, outdir, "input", "input_1"),
            record(overlay, outdir, "stardist", "stardist_1"),
        ],
        quality_signals=[],
        uncertainty_register=[],
    )

    specimen = payload["specimens"][0]
    context = next(layer for layer in specimen["layers"] if layer["id"] == "context")
    cells = next(layer for layer in specimen["layers"] if layer["id"] == "cells")
    assert not context["assets"]
    assert cells["assets"][0]["route"] == "cell identification and marker phenotyping"
