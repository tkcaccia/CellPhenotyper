#!/usr/bin/env python3
"""Run one isolated CellPhenotyper workflow per image in a cohort folder.

Each sample receives its own top-level output and work directory. This keeps
large cohorts easy to inspect and permits one failed sample to be resumed
without disturbing completed samples.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = (
    ".ome.tiff",
    ".ome.tif",
    ".jpeg",
    ".tiff",
    ".btf",
    ".czi",
    ".vsi",
    ".svs",
    ".ndpi",
    ".scn",
    ".mrxs",
    ".vms",
    ".vmu",
    ".tif",
    ".png",
    ".jpg",
)


def image_suffix(path: Path) -> str:
    lower = path.name.lower()
    return next((suffix for suffix in IMAGE_SUFFIXES if lower.endswith(suffix)), "")


def sample_id(path: Path) -> str:
    suffix = image_suffix(path)
    return path.name[: -len(suffix)] if suffix else path.stem


def discover(input_dir: Path) -> list[Path]:
    selected: dict[str, Path] = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or not image_suffix(path):
            continue
        sid = sample_id(path)
        current = selected.get(sid)
        if current is None or IMAGE_SUFFIXES.index(image_suffix(path)) < IMAGE_SUFFIXES.index(image_suffix(current)):
            selected[sid] = path
    return [selected[sid] for sid in sorted(selected)]


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_sample_params(source: Path, destination: Path, image: Path) -> None:
    """Copy YAML parameters while forcing unambiguous single-image mode."""
    text = source.read_text(encoding="utf-8")
    replacements = {
        "folder_input": "null",
        "image_input": json.dumps(str(image)),
        "roi_geojson": "null",
    }
    for key, value in replacements.items():
        pattern = rf"(?m)^{re.escape(key)}\s*:.*$"
        replacement = f"{key}: {value}"
        if re.search(pattern, text):
            text = re.sub(pattern, replacement, text, count=1)
        else:
            text = replacement + "\n" + text
    destination.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CellPhenotyper serially with results/<sample>/<stage>/... organization."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--params-file", type=Path)
    parser.add_argument("--profile", default="docker")
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--nextflow", default="nextflow")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--clean-work-on-success",
        action="store_true",
        help="Remove only the completed sample's isolated Nextflow work directory after outputs are published.",
    )
    parser.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_root = args.output_root.resolve()
    project_dir = args.project_dir.resolve()
    params_file = (args.params_file or project_dir / "pipeline_paramers.yml").resolve()
    work_root = (args.work_root or output_root / "work").resolve()
    forwarded = list(args.pipeline_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    images = discover(input_dir)
    if not images:
        parser.error(f"no supported images found in {input_dir}")

    for image in images:
        sid = sample_id(image)
        if image_suffix(image) == ".vsi":
            companion = image.parent / f"_{sid}_"
            if not companion.is_dir():
                parser.error(f"missing VSI companion directory for {image.name}: {companion}")

    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "batch_status.json"
    status = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "samples": {sample_id(path): {"input": str(path), "status": "pending"} for path in images},
    }
    write_status(manifest_path, status)

    for image in images:
        sid = sample_id(image)
        sample_out = output_root / sid
        sample_work = work_root / sid
        execution_dir = sample_out / "00_execution"
        execution_dir.mkdir(parents=True, exist_ok=True)
        log_path = execution_dir / "nextflow.console.log"
        sample_params = execution_dir / "resolved_params.yml"
        write_sample_params(params_file, sample_params, image)
        command = [
            args.nextflow,
            "run",
            str(project_dir / "main.nf"),
            "-profile",
            args.profile,
            "-params-file",
            str(sample_params),
            "-work-dir",
            str(sample_work),
            "--outdir_base",
            str(sample_out),
            *forwarded,
        ]
        if not args.no_resume:
            command.append("-resume")

        status["samples"][sid].update(
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            output_dir=str(sample_out),
            work_dir=str(sample_work),
            log=str(log_path),
            command=command,
        )
        write_status(manifest_path, status)
        print(f"[BATCH] Starting {sid}: {image}", flush=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[BATCH] command={json.dumps(command)}\n")
            log_handle.flush()
            completed = subprocess.run(
                command,
                cwd=project_dir,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )

        status["samples"][sid].update(
            status="completed" if completed.returncode == 0 else "failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            returncode=completed.returncode,
        )
        write_status(manifest_path, status)
        if completed.returncode != 0:
            print(f"[BATCH] {sid} failed; see {log_path}", file=sys.stderr, flush=True)
            return completed.returncode
        if args.clean_work_on_success and sample_work.is_dir():
            shutil.rmtree(sample_work)
            status["samples"][sid]["work_cleaned"] = True
            write_status(manifest_path, status)
        print(f"[BATCH] Completed {sid}", flush=True)

    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    status["status"] = "completed"
    write_status(manifest_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
