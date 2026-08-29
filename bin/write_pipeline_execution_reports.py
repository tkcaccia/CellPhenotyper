#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path


STAGE_DEFS = [
    {"id": "input", "folder": "01_input", "title": "Input Conversion", "expected": [".ome.tif", ".source_resolution.json", ".converted_resolution.json"]},
    {"id": "grandqc", "folder": "02_grandqc", "title": "GrandQC Artifact QC", "expected": ["_grandqc_summary.json", "_grandqc_artifact_mask.tif", "_grandqc_clean_tissue_mask.tif"]},
    {"id": "stardist", "folder": "03_stardist", "title": "StarDist Segmentation", "expected": ["labels.tif", "objects.csv"]},
    {"id": "hovernet_monusac", "folder": "03b_hovernet_monusac", "title": "HoVer-Net MoNuSAC", "expected": ["hovernet_cells.json"]},
    {"id": "cellvitpp", "folder": "03c_cellvitpp", "title": "CellViT++", "expected": ["cellvit_cells.json"]},
    {"id": "cell_consensus", "folder": "03d_cell_consensus", "title": "Consensus Cell Identification", "expected": ["labels.tif", "objects.csv", "alignment.csv", "consensus_cells.geojson", "consensus_summary.json", "consensus_preview.png"]},
    {"id": "tma", "folder": "04_TMA", "title": "TMA Detection and Cell-to-Spot Assignment", "expected": ["_tma_summary.json", "_tma_spots.geojson", "_objects_tma_assigned.csv"]},
    {"id": "tissue_mask", "folder": "04_tissue_mask", "title": "Tissue Mask", "expected": ["_tissue_mask.tif"]},
    {"id": "gigatime", "folder": "05_gigatime", "title": "GigaTIME Virtual mIF + Marker Quantification", "expected": ["gigatime_probs.ome.tif", "gigatime_probs.zarr", "_gigatime_quantification.csv", "_gigatime_mean_intensity.csv", "_gigatime_intensity_stats.csv", "_gigatime_intensity_summary.json"]},
    {"id": "roi", "folder": "06_roi", "title": "ROI GeoJSON and Input ROI Mask", "expected": [".roi.geojson", "_input_roi_mask.tif", "_input_roi_mask_preview.png", "_input_roi_mask_labels.json"]},
    {"id": "cell_assignment", "folder": "07_cell_assignments", "title": "Cell Assignment", "expected": ["_objects_assigned.csv"]},
    {"id": "cytoplasm", "folder": "08_cytoplasm", "title": "Cytoplasm Expansion", "expected": ["_labels_cyto.tif"]},
    {"id": "embeddings", "folder": "09_embeddings", "title": "UNI-2 Embeddings", "expected": ["embeddings_"]},
    {"id": "kodama", "folder": "10_kodama", "title": "KODAMA", "expected": ["kodama_output", ".Rout"]},
    {"id": "clustering", "folder": "11_clustering", "title": "Clustering", "expected": ["_cluster.csv", ".Rout"]},
    {"id": "cluster_mask", "folder": "12_cluster_mask", "title": "Cluster Mask", "expected": ["_cluster_mask.tif"]},
    {"id": "grown_tissue", "folder": "13_grown_tissue", "title": "Grown Tissue", "expected": ["_grown_mask.ome.tif"]},
    {"id": "medsam_refine_tissue", "folder": "14_medsam_refine_tissue", "title": "MedSAM Refinement", "expected": ["_grown_mask_refined.ome.tif", "_medsam_editable_band.png", "_medsam_raw_vs_final_panel.png", "_medsam_kodama_membership.png"]},
    {"id": "cluster_geojson", "folder": "15_cluster_geojson", "title": "Cluster GeoJSON", "expected": [".geojson"]},
    {"id": "neoplastic_section", "folder": "16_neoplastic_section", "title": "Neoplastic-Enriched Tissue Section", "expected": ["selected_section.ome.tif", "section_neoplastic_counts.csv", "selected_section_summary.json", "selected_section_preview.png"]},
    {"id": "titan", "folder": "17_titan", "title": "TITAN Section Representation", "expected": ["titan_embedding.csv", "titan_patch_features.h5", "titan_metadata.json"]},
    {"id": "pathofmpred", "folder": "18_pathofmpred", "title": "PathoFMPred Research Predictions", "expected": ["pathofmpred_predictions.csv", "pathofmpred_research_report.html", "pathofmpred_continuous_radar.png", "pathofmpred_binary_predictions.png"]},
    {"id": "execution", "folder": "00_execution", "title": "Execution Metadata", "expected": ["trace.tsv", "timeline.html", "dag.html"]},
]


def bytes_to_human(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def seconds_to_human(total_seconds: float) -> str:
    seconds = max(0, int(round(total_seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def parse_duration_seconds(raw: str) -> float:
    text = (raw or "").strip()
    if not text:
        return 0.0
    unit_matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(d|h|m|s|ms)\b", text)
    if unit_matches:
        total = 0.0
        for value, unit in unit_matches:
            amount = float(value)
            if unit == "d":
                total += amount * 86400.0
            elif unit == "h":
                total += amount * 3600.0
            elif unit == "m":
                total += amount * 60.0
            elif unit == "s":
                total += amount
            elif unit == "ms":
                total += amount / 1000.0
        return total
    if text.endswith("ms"):
        try:
            return float(text[:-2]) / 1000.0
        except ValueError:
            return 0.0
    if text.endswith("s") and text.count(":") == 0:
        try:
            return float(text[:-1])
        except ValueError:
            return 0.0
    parts = text.split(":")
    if len(parts) in (2, 3):
        try:
            parts_f = [float(p) for p in parts]
        except ValueError:
            return 0.0
        if len(parts_f) == 2:
            return parts_f[0] * 60.0 + parts_f[1]
        return parts_f[0] * 3600.0 + parts_f[1] * 60.0 + parts_f[2]
    return 0.0


def parse_memory_bytes(raw: str) -> int:
    text = (raw or "").strip().upper().replace("IB", "B")
    if not text:
        return 0
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
        "TB": 1024 ** 4,
    }
    for suffix, mult in sorted(multipliers.items(), key=lambda item: len(item[0]), reverse=True):
        if text.endswith(suffix):
            try:
                return int(float(text[: -len(suffix)]) * mult)
            except ValueError:
                return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def summarize_trace(trace_path: Path) -> list[dict]:
    if not trace_path.exists():
        return []
    with trace_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    idx_process = header.index("process") if "process" in header else (header.index("name") if "name" in header else -1)
    idx_realtime = header.index("realtime") if "realtime" in header else -1
    idx_peak_rss = header.index("peak_rss") if "peak_rss" in header else -1
    idx_status = header.index("status") if "status" in header else -1
    grouped: dict[str, dict] = {}
    for cols in rows[1:]:
        if idx_process < 0 or idx_process >= len(cols):
            continue
        process_name = cols[idx_process]
        bucket = grouped.setdefault(process_name, {"tasks": 0, "realtime_s": 0.0, "peak_bytes": 0, "failed": 0})
        bucket["tasks"] += 1
        if idx_realtime >= 0 and idx_realtime < len(cols):
            bucket["realtime_s"] += parse_duration_seconds(cols[idx_realtime])
        if idx_peak_rss >= 0 and idx_peak_rss < len(cols):
            bucket["peak_bytes"] = max(bucket["peak_bytes"], parse_memory_bytes(cols[idx_peak_rss]))
        if idx_status >= 0 and idx_status < len(cols) and cols[idx_status] not in {"COMPLETED", "CACHED"}:
            bucket["failed"] += 1
    return sorted(
        [
            {
                "process": name,
                **vals,
                "realtime_human": seconds_to_human(vals["realtime_s"]),
                "peak_human": bytes_to_human(vals["peak_bytes"]),
            }
            for name, vals in grouped.items()
        ],
        key=lambda x: x["realtime_s"],
        reverse=True,
    )


def list_files(stage_dir: Path) -> list[dict]:
    if not stage_dir.exists():
        return []
    files = []
    seen_dirs: set[str] = set()
    for root, dirs, names in os.walk(stage_dir, followlinks=True):
        try:
            real_root = os.path.realpath(root)
        except OSError:
            real_root = root
        if real_root in seen_dirs:
            dirs[:] = []
            continue
        seen_dirs.add(real_root)
        for name in sorted(names):
            f = Path(root) / name
            if not f.is_file():
                continue
            try:
                size = f.stat().st_size
                resolved = str(f.resolve())
                rel = f.relative_to(stage_dir).as_posix()
            except OSError:
                continue
            files.append(
                {
                    "relative_path": rel,
                    "absolute_path": resolved,
                    "size_bytes": size,
                }
            )
    files.sort(key=lambda item: item["relative_path"])
    return files


def stage_summary(outdir: Path) -> list[dict]:
    rows = []
    for spec in STAGE_DEFS:
        stage_dir = outdir / spec["folder"]
        files = list_files(stage_dir)
        size_bytes = sum(f["size_bytes"] for f in files)
        key_files = []
        for needle in spec["expected"]:
            match = next((f["absolute_path"] for f in files if needle in f["relative_path"]), None)
            if match:
                key_files.append(match)
        rows.append(
            {
                "id": spec["id"],
                "folder": spec["folder"],
                "title": spec["title"],
                "present": bool(files),
                "file_count": len(files),
                "size_bytes": size_bytes,
                "size_human": bytes_to_human(size_bytes),
                "key_files": key_files,
                "files": files,
            }
        )
    return rows


def make_output_id(namespace: str, stage_id: str, relative_path: str) -> str:
    seed = f"{namespace}\t{stage_id}\t{relative_path}".encode("utf-8")
    digest = hashlib.sha1(seed).hexdigest()[:16]
    return f"{stage_id}_{digest}"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unnamed"


def preserve_trace(
    execution_dir: Path,
    run_name: str,
    success: bool,
    start_point: str,
    end_point: str,
) -> tuple[Path, dict | None]:
    """Snapshot each run trace and retain the latest successful full-pipeline trace."""
    trace_path = execution_dir / "trace.tsv"
    runs_dir = execution_dir / "run_traces"
    runs_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = runs_dir / f"{safe_name(run_name)}_{safe_name(start_point)}_to_{safe_name(end_point)}.tsv"
    if trace_path.exists():
        shutil.copy2(trace_path, snapshot_path)

    full_trace_path = execution_dir / "full_pipeline_trace.tsv"
    full_meta_path = execution_dir / "full_pipeline_run.json"
    final_stages = {"cluster_geojson", "neoplastic_section", "titan", "pathofmpred"}
    is_full_run = success and start_point == "convert" and end_point in final_stages
    if is_full_run and trace_path.exists():
        shutil.copy2(trace_path, full_trace_path)
        full_meta_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "success": success,
                    "start_point": start_point,
                    "end_point": end_point,
                    "trace_file": str(full_trace_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    full_meta = None
    if full_trace_path.exists() and full_meta_path.exists():
        try:
            full_meta = json.loads(full_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            full_meta = None
    return (full_trace_path if full_trace_path.exists() else trace_path), full_meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--success", required=True)
    ap.add_argument("--start-point", required=True)
    ap.add_argument("--end-point", required=True)
    ap.add_argument("--image-input", default="")
    ap.add_argument("--roi-geojson", default="")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    execution_dir = (outdir / "00_execution").resolve()
    execution_dir.mkdir(parents=True, exist_ok=True)

    success = args.success.strip().lower() == "true"
    trace_source, full_run_meta = preserve_trace(
        execution_dir,
        args.run_name,
        success,
        args.start_point,
        args.end_point,
    )

    stages = stage_summary(outdir)
    trace_rows = summarize_trace(trace_source)
    trace_history = []
    for run_trace in sorted((execution_dir / "run_traces").glob("*.tsv")):
        for row in summarize_trace(run_trace):
            trace_history.append({"trace": run_trace.name, **row})
    report_run_name = str(full_run_meta.get("run_name")) if full_run_meta else args.run_name
    report_start = str(full_run_meta.get("start_point")) if full_run_meta else args.start_point
    report_end = str(full_run_meta.get("end_point")) if full_run_meta else args.end_point
    report_success = bool(full_run_meta.get("success")) if full_run_meta else success

    manifest_path = execution_dir / "outputs_manifest.txt"
    manifest_lines = [
        "CellPhenotyper Output Manifest",
        f"Run name: {report_run_name}",
        f"Success: {str(report_success).lower()}",
        f"Stage window: {report_start} -> {report_end}",
        f"Timing trace: {trace_source}",
        "",
        f"Stage folders under: {outdir}",
    ]
    for s in stages:
        status = "PRESENT" if s["present"] else "MISSING"
        manifest_lines.append(f"{status}\t{outdir / s['folder']}\tfiles={s['file_count']}\tsize={s['size_human']}")
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    project_records = []
    for s in stages:
        for f in s["files"]:
            project_records.append(
                {
                    "output_id": make_output_id(str(outdir), s["id"], f["relative_path"]),
                    "stage_id": s["id"],
                    "stage_title": s["title"],
                    "stage_folder": s["folder"],
                    **f,
                }
            )

    project_json_path = execution_dir / "project_outputs.json"
    project_tsv_path = execution_dir / "project_outputs.tsv"
    project_payload = {
        "run_name": report_run_name,
        "success": report_success,
        "output_root": str(outdir),
        "stage_window": {"start": report_start, "end": report_end},
        "input_context": {
            "image_input": str(Path(args.image_input).resolve()) if args.image_input else None,
            "roi_geojson": str(Path(args.roi_geojson).resolve()) if args.roi_geojson else None,
        },
        "records": project_records,
    }
    project_json_path.write_text(json.dumps(project_payload, indent=2), encoding="utf-8")
    with project_tsv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["output_id", "stage_id", "stage_title", "stage_folder", "relative_path", "absolute_path", "size_bytes"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(project_records)

    report_md_path = execution_dir / "final_report.md"
    report_lines = [
        "# CellPhenotyper Final Report",
        "",
        f"- Run name: `{report_run_name}`",
        f"- Success: `{str(report_success).lower()}`",
        f"- Stage window: `{report_start} -> {report_end}`",
        f"- Timing trace: `{trace_source}`",
        f"- Output root: `{outdir}`",
        f"- Project outputs JSON: `{project_json_path}`",
        f"- Project outputs TSV: `{project_tsv_path}`",
        "",
        "## Stage Outputs",
        "",
        "| Folder | Stage | Status | Files | Size |",
        "|---|---|---:|---:|---:|",
    ]
    for s in stages:
        report_lines.append(f"| `{s['folder']}` | {s['title']} | {'PRESENT' if s['present'] else 'MISSING'} | {s['file_count']} | {s['size_human']} |")
    report_lines.extend(["", "## Key Result Files", ""])
    for s in stages:
        if s["key_files"]:
            report_lines.append(f"- `{s['folder']}`")
            for path in s["key_files"]:
                report_lines.append(f"  - `{path}`")
    report_lines.extend(["", "## Runtime And Memory (From trace.tsv)", ""])
    if trace_rows:
        report_lines.extend([
            "| Process | Tasks | Total Realtime | Total Realtime (s) | Peak RSS | Failed Tasks |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in trace_rows:
            report_lines.append(
                f"| `{row['process']}` | {row['tasks']} | {row['realtime_human']} | {row['realtime_s']:.1f} | {row['peak_human']} | {row['failed']} |"
            )
    else:
        report_lines.append("Trace file not available.")
    report_lines.extend(["", "## Run Timing History", ""])
    if trace_history:
        report_lines.extend([
            "| Run trace | Process | Tasks | Total Realtime | Peak RSS | Failed Tasks |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for row in trace_history:
            report_lines.append(
                f"| `{row['trace']}` | `{row['process']}` | {row['tasks']} | {row['realtime_human']} | {row['peak_human']} | {row['failed']} |"
            )
    else:
        report_lines.append("No preserved per-run traces are available.")
    report_md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    final_report_json_path = execution_dir / "final_report.json"
    final_report_json_path.write_text(
        json.dumps(
            {
                "run_name": report_run_name,
                "success": report_success,
                "stage_window": {"start": report_start, "end": report_end},
                "timing_trace": str(trace_source),
                "output_root": str(outdir),
                "stages": [
                    {
                        "id": s["id"],
                        "folder": s["folder"],
                        "stage": s["title"],
                        "present": s["present"],
                        "file_count": s["file_count"],
                        "size_bytes": s["size_bytes"],
                        "key_files": s["key_files"],
                    }
                    for s in stages
                ],
                "process_trace_summary": trace_rows,
                "run_trace_history": trace_history,
                "project_outputs_json": str(project_json_path),
                "project_outputs_tsv": str(project_tsv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Execution reports dir: {execution_dir}")
    print(f"Output manifest: {manifest_path}")
    print(f"Project outputs JSON: {project_json_path}")
    print(f"Project outputs TSV: {project_tsv_path}")
    print(f"Final report: {report_md_path}")
    print(f"Final report JSON: {final_report_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
