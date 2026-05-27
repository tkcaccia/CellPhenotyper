#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


STAGE_DEFS = [
    {"id": "input", "folder": "01_input", "title": "Input Conversion", "expected": [".ome.tif"]},
    {"id": "grandqc", "folder": "01a_grandqc", "title": "GrandQC Artifact QC", "expected": ["_grandqc_summary.json", "_grandqc_artifact_mask.tif", "_grandqc_clean_tissue_mask.tif"]},
    {"id": "stardist", "folder": "02_stardist", "title": "StarDist Segmentation", "expected": ["labels.tif", "objects.csv"]},
    {"id": "gigatime", "folder": "03_gigatime", "title": "GigaTIME Virtual mIF", "expected": ["gigatime_probs.ome.tif", "gigatime_probs.zarr"]},
    {"id": "tissue_mask", "folder": "04_tissue_mask", "title": "Tissue Mask", "expected": ["_tissue_mask.tif"]},
    {"id": "roi", "folder": "05_roi", "title": "ROI GeoJSON", "expected": [".roi.geojson"]},
    {"id": "roi_mask", "folder": "06_roi_mask", "title": "Input ROI Mask", "expected": ["_input_roi_mask.tif", "_input_roi_mask_preview.png", "_input_roi_mask_labels.json"]},
    {"id": "marker_quantification", "folder": "06_marker_quantification", "title": "Marker Quantification", "expected": ["_gigatime_quantification.csv", "_gigatime_mean_intensity.csv", "_gigatime_intensity_stats.csv", "_gigatime_intensity_summary.json"]},
    {"id": "cell_assignment", "folder": "07_cell_assignments", "title": "Cell Assignment", "expected": ["_objects_assigned.csv"]},
    {"id": "cytoplasm", "folder": "08_cytoplasm", "title": "Cytoplasm Expansion", "expected": ["_labels_cyto.tif"]},
    {"id": "embeddings", "folder": "10_embeddings", "title": "UNI-2 Embeddings", "expected": ["embeddings_"]},
    {"id": "kodama", "folder": "11_kodama", "title": "KODAMA", "expected": ["kodama_output"]},
    {"id": "kodama_logs", "folder": "12_kodama_logs", "title": "KODAMA Logs", "expected": [".Rout"]},
    {"id": "clustering", "folder": "13_clustering", "title": "Clustering", "expected": ["_cluster.csv"]},
    {"id": "clustering_logs", "folder": "14_clustering_logs", "title": "Clustering Logs", "expected": [".Rout"]},
    {"id": "cluster_mask", "folder": "15_cluster_mask", "title": "Cluster Mask", "expected": ["_cluster_mask.tif"]},
    {"id": "grown_tissue", "folder": "16_grown_tissue", "title": "Grown Tissue", "expected": ["_grown_mask.ome.tif"]},
    {"id": "medsam_refined_tissue", "folder": "17_medsam_refined_tissue", "title": "MedSAM Refinement", "expected": ["_grown_mask_refined.ome.tif", "_medsam_editable_band.png", "_medsam_raw_vs_final_panel.png", "_medsam_kodama_membership.png"]},
    {"id": "cluster_geojson", "folder": "18_cluster_geojson", "title": "Cluster GeoJSON", "expected": [".geojson"]},
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
    for f in sorted(p for p in stage_dir.rglob("*") if p.is_file()):
        files.append(
            {
                "relative_path": f.relative_to(stage_dir).as_posix(),
                "absolute_path": str(f.resolve()),
                "size_bytes": f.stat().st_size,
            }
        )
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


def make_output_id(run_name: str, stage_id: str, relative_path: str) -> str:
    seed = f"{run_name}\t{stage_id}\t{relative_path}".encode("utf-8")
    digest = hashlib.sha1(seed).hexdigest()[:16]
    return f"{stage_id}_{digest}"


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

    stages = stage_summary(outdir)
    trace_rows = summarize_trace(execution_dir / "trace.tsv")

    manifest_path = execution_dir / "outputs_manifest.txt"
    manifest_lines = [
        "CellPhenotyper Output Manifest",
        f"Run name: {args.run_name}",
        f"Success: {args.success}",
        f"Stage window: {args.start_point} -> {args.end_point}",
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
                    "output_id": make_output_id(args.run_name, s["id"], f["relative_path"]),
                    "stage_id": s["id"],
                    "stage_title": s["title"],
                    "stage_folder": s["folder"],
                    **f,
                }
            )

    project_json_path = execution_dir / "project_outputs.json"
    project_tsv_path = execution_dir / "project_outputs.tsv"
    project_payload = {
        "run_name": args.run_name,
        "success": args.success,
        "output_root": str(outdir),
        "stage_window": {"start": args.start_point, "end": args.end_point},
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
        f"- Run name: `{args.run_name}`",
        f"- Success: `{args.success}`",
        f"- Stage window: `{args.start_point} -> {args.end_point}`",
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
    report_md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    final_report_json_path = execution_dir / "final_report.json"
    final_report_json_path.write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "success": args.success,
                "stage_window": {"start": args.start_point, "end": args.end_point},
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
