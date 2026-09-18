#!/usr/bin/env python3
"""Run one isolated CellPhenotyper workflow per image in a cohort folder.

Each sample receives its own top-level output and work directory. This keeps
large cohorts easy to inspect and permits one failed sample to be resumed
without disturbing completed samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline_source_digest(project_dir: Path) -> str:
    """Fingerprint runnable pipeline sources, excluding outputs and audits."""
    roots = ["bin", "lib", "modules", "subworkflows", "workflows"]
    files = [
        project_dir / "main.nf",
        project_dir / "nextflow.config",
        project_dir / "nextflow_schema.json",
    ]
    for name in roots:
        root = project_dir / name
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(project_dir)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def input_inventory(image: Path) -> list[dict]:
    """Record the image and VSI companions without hashing multi-GB payloads."""
    paths = [image]
    if image_suffix(image) == ".vsi":
        companion = image.parent / f"_{sample_id(image)}_"
        paths.extend(path for path in companion.rglob("*") if path.is_file())
    inventory = []
    for path in sorted(paths):
        stat_result = path.stat()
        inventory.append({
            "path": str(path.relative_to(image.parent)),
            "size_bytes": int(stat_result.st_size),
            "mtime_ns": int(stat_result.st_mtime_ns),
        })
    return inventory


def run_signature(
    *, image: Path, params_file: Path, profile: str, forwarded: list[str],
    project_digest: str,
) -> tuple[str, dict]:
    payload = {
        "image": str(image),
        "input_inventory": input_inventory(image),
        "resolved_params_sha256": file_sha256(params_file),
        "profile": profile,
        "pipeline_args": forwarded,
        "pipeline_source_sha256": project_digest,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def matching_completion(path: Path, signature: str) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") == 1
        and payload.get("status") == "completed"
        and payload.get("run_signature") == signature
    ):
        return payload
    return None


def remove_sample_work(sample_work: Path, work_root: Path) -> None:
    """Delete only a direct sample child, never the work root itself."""
    target = sample_work.resolve()
    root = work_root.resolve()
    if target == root or target.parent != root:
        raise RuntimeError(f"Refusing unsafe work cleanup outside direct work-root child: {target}")
    if target.is_dir():
        shutil.rmtree(target)


def run_command(command: list[str], *, cwd: Path, log_handle) -> tuple[int, int | None]:
    """Forward termination to Nextflow and report the interrupt signal."""
    child = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        start_new_session=True,
    )
    interrupted: list[int] = []

    def forward(signum, _frame) -> None:
        if not interrupted:
            interrupted.append(int(signum))
        if child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    watched = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous = {signum: signal.getsignal(signum) for signum in watched}
    for signum in watched:
        signal.signal(signum, forward)
    try:
        returncode = child.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return returncode, (interrupted[0] if interrupted else None)


def write_sample_params(
    source: Path, destination: Path, image: Path, extra: dict[str, object] | None = None,
) -> None:
    """Copy YAML parameters while forcing unambiguous single-image mode."""
    text = source.read_text(encoding="utf-8")
    replacements = {
        "folder_input": "null",
        "image_input": json.dumps(str(image)),
        "roi_geojson": "null",
    }
    for key, value in (extra or {}).items():
        replacements[key] = json.dumps(value)
    for key, value in replacements.items():
        pattern = rf"(?m)^{re.escape(key)}\s*:.*$"
        replacement = f"{key}: {value}"
        if re.search(pattern, text):
            text = re.sub(pattern, replacement, text, count=1)
        else:
            text = replacement + "\n" + text
    destination.write_text(text, encoding="utf-8")


def argument_value(arguments: list[str], key: str) -> str | None:
    flag = f"--{key}"
    for index, value in enumerate(arguments):
        if value == flag and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return None


def yaml_scalar(path: Path, key: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(key)}\s*:\s*([^#\n]*?)\s*$", path.read_text(encoding="utf-8"))
    if not match:
        return None
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value or None


def probe_vsi_series(
    image: Path,
    *,
    project_dir: Path,
    params_file: Path,
    profile: str,
    forwarded: list[str],
) -> dict:
    """Read the same unflattened Bio-Formats series used by conversion."""
    profiles = {value.strip().lower() for value in profile.split(",")}
    if "docker" not in profiles:
        raise RuntimeError(
            "Automatic VSI selected-series preflight currently requires the docker profile; "
            "provide storage_preflight_input_metadata explicitly for another runtime."
        )
    docker_image = (
        argument_value(forwarded, "docker_image")
        or yaml_scalar(params_file, "docker_image")
    )
    if not docker_image:
        raise RuntimeError("Cannot probe VSI metadata without a configured docker_image")
    raw_series = argument_value(forwarded, "vsi_series_index") or yaml_scalar(params_file, "vsi_series_index") or "1"
    try:
        series_index = int(raw_series)
    except ValueError as exc:
        raise RuntimeError(f"Invalid vsi_series_index: {raw_series}") from exc
    java_source = project_dir / "bin" / "BioFormatsSeriesProbe.java"
    if not java_source.is_file():
        raise RuntimeError(f"Bio-Formats series probe is missing: {java_source}")
    input_mount = "/cellphenotyper-input"
    bin_mount = "/cellphenotyper-bin"
    classpath = "/opt/micromamba/envs/stardist/share/raw2ometiff-0.7.1-0/lib/*"
    shell = (
        f'CP="{classpath}"; exec micromamba run -n stardist java '
        '--class-path "$CP" /cellphenotyper-bin/BioFormatsSeriesProbe.java "$1" "$2"'
    )
    command = [
        "docker", "run", "--rm",
        "-v", f"{image.parent}:{input_mount}:ro",
        "-v", f"{project_dir / 'bin'}:{bin_mount}:ro",
        docker_image,
        "sh", "-lc", shell, "probe",
        f"{input_mount}/{image.name}", str(series_index),
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    record = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "width_px" in candidate and "height_px" in candidate:
            record = candidate
            break
    if completed.returncode != 0 or record is None:
        tail = "\n".join(completed.stdout.splitlines()[-20:])
        raise RuntimeError(
            f"Bio-Formats VSI metadata probe failed for {image.name} "
            f"(exit {completed.returncode}):\n{tail}"
        )
    record["path"] = str(image.resolve())
    record["docker_image"] = docker_image
    return record


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
    cleanup = parser.add_mutually_exclusive_group()
    cleanup.add_argument(
        "--clean-work-on-success",
        dest="clean_work_on_success",
        action="store_true",
        help="Remove the completed sample's isolated work directory (default).",
    )
    cleanup.add_argument(
        "--keep-work-on-success",
        dest="clean_work_on_success",
        action="store_false",
        help="Retain successful sample work directories for debugging or manual resume.",
    )
    parser.set_defaults(clean_work_on_success=True)
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Ignore matching completion markers and run successful samples again.",
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
    project_digest = pipeline_source_digest(project_dir)

    for image in images:
        sid = sample_id(image)
        sample_out = output_root / sid
        sample_work = work_root / sid
        execution_dir = sample_out / "00_execution"
        execution_dir.mkdir(parents=True, exist_ok=True)
        log_path = execution_dir / "nextflow.console.log"
        sample_params = execution_dir / "resolved_params.yml"
        parameter_overrides: dict[str, object] = {}
        if image_suffix(image) == ".vsi":
            print(f"[BATCH] Probing selected VSI series for {sid}", flush=True)
            vsi_record = probe_vsi_series(
                image,
                project_dir=project_dir,
                params_file=params_file,
                profile=args.profile,
                forwarded=forwarded,
            )
            metadata_path = execution_dir / "vsi_preflight_metadata.json"
            write_status(metadata_path, {"schema_version": 1, "inputs": [vsi_record]})
            parameter_overrides["storage_preflight_input_metadata"] = str(metadata_path)
        write_sample_params(params_file, sample_params, image, parameter_overrides)
        signature, signature_payload = run_signature(
            image=image,
            params_file=sample_params,
            profile=args.profile,
            forwarded=forwarded,
            project_digest=project_digest,
        )
        completion_path = execution_dir / "batch_complete.json"
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

        prior_completion = None if args.rerun_completed else matching_completion(completion_path, signature)
        if prior_completion is not None:
            status["samples"][sid].update(
                status="completed",
                reused_completion=True,
                output_dir=str(sample_out),
                work_dir=str(sample_work),
                completion_marker=str(completion_path),
                run_signature=signature,
                finished_at=prior_completion.get("finished_at"),
            )
            write_status(manifest_path, status)
            print(f"[BATCH] Reusing verified completion for {sid}", flush=True)
            continue

        status["samples"][sid].update(
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            output_dir=str(sample_out),
            work_dir=str(sample_work),
            log=str(log_path),
            command=command,
            run_signature=signature,
        )
        write_status(manifest_path, status)
        print(f"[BATCH] Starting {sid}: {image}", flush=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[BATCH] command={json.dumps(command)}\n")
            log_handle.flush()
            try:
                returncode, interrupted_signal = run_command(
                    command, cwd=project_dir, log_handle=log_handle,
                )
            except OSError as exc:
                log_handle.write(f"[BATCH] launch error: {exc}\n")
                returncode, interrupted_signal = 127, None

        status["samples"][sid].update(
            status=("interrupted" if interrupted_signal is not None else
                    ("completed" if returncode == 0 else "failed")),
            finished_at=datetime.now(timezone.utc).isoformat(),
            returncode=returncode,
        )
        if interrupted_signal is not None:
            status["samples"][sid]["signal"] = interrupted_signal
        write_status(manifest_path, status)
        if returncode != 0:
            outcome = "interrupted" if interrupted_signal is not None else "failed"
            status["status"] = outcome
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_status(manifest_path, status)
            print(f"[BATCH] {sid} {outcome}; see {log_path}", file=sys.stderr, flush=True)
            return (128 + interrupted_signal) if interrupted_signal is not None else returncode

        completion = {
            "schema_version": 1,
            "status": "completed",
            "sample_id": sid,
            "input": str(image),
            "output_dir": str(sample_out),
            "run_signature": signature,
            "signature_inputs": signature_payload,
            "finished_at": status["samples"][sid]["finished_at"],
        }
        write_status(completion_path, completion)
        status["samples"][sid]["completion_marker"] = str(completion_path)
        if args.clean_work_on_success and sample_work.is_dir():
            remove_sample_work(sample_work, work_root)
            status["samples"][sid]["work_cleaned"] = True
            write_status(manifest_path, status)
        print(f"[BATCH] Completed {sid}", flush=True)

    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    status["status"] = "completed"
    write_status(manifest_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
