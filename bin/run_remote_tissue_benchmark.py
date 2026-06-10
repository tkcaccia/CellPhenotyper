#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from tissue_mask_benchmark_lib import BENCHMARK_METHODS, discover_input_manifest_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the remote tissue benchmark and write review-friendly reports.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--benchmark-script", default="bin/benchmark_tissue_mask_methods.py")
    ap.add_argument("--methods", default="", help="Comma-separated methods to run. Defaults to all benchmark methods.")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--runtime-label", default="remote_gpu_host")
    return ap.parse_args()


def run_capture(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_environment_report(repo_root: Path, outdir: Path) -> None:
    commands = {
        "pwd": ["pwd"],
        "git_status": ["git", "status", "--short"],
        "git_branch": ["git", "branch", "--show-current"],
        "python_version": [sys.executable, "--version"],
        "python_which": ["bash", "-lc", f"command -v {shlex.quote(sys.executable)} || true"],
        "nvidia_smi": ["bash", "-lc", "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo nvidia-smi-not-found"],
        "torch_cuda": [
            sys.executable,
            "-c",
            (
                "import json\n"
                "report = {}\n"
                "try:\n"
                " import torch\n"
                " report['torch_version'] = getattr(torch, '__version__', 'unknown')\n"
                " report['cuda_available'] = bool(torch.cuda.is_available())\n"
                " report['cuda_device_count'] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0\n"
                " report['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''\n"
                "except Exception as exc:\n"
                " report['error'] = str(exc)\n"
                "print(json.dumps(report, indent=2))\n"
            ),
        ],
        "singularity": ["bash", "-lc", "command -v singularity || true"],
        "docker": ["bash", "-lc", "command -v docker || true"],
        "nextflow": ["bash", "-lc", "command -v nextflow || true"],
    }

    lines = [
        "# Environment Report",
        "",
        f"- generated_at: {dt.datetime.now().isoformat()}",
        f"- repo_root: `{repo_root}`",
        f"- python: `{sys.executable}`",
        "",
    ]
    for title, cmd in commands.items():
        code, output = run_capture(cmd, repo_root)
        lines.extend([f"## {title}", "", f"- exit_code: `{code}`", "", "```text", output.rstrip(), "```", ""])
    write_text(outdir / "environment_report.md", "\n".join(lines).rstrip() + "\n")


def write_gpu_usage_report(outdir: Path, methods: list[str]) -> None:
    lines = [
        "# GPU Usage Report",
        "",
        "- This benchmark was launched on the remote NVIDIA host.",
        "- Classical methods run on CPU unless a dependency explicitly uses CUDA.",
        "- UNI-2 support remains container-available but is not benchmarked by this driver yet.",
        "- Path-SAM2 was probed and is not installed in the current GPU singularity runtime.",
        "",
        "## Methods in this run",
        "",
    ]
    for method in methods:
        if method.startswith("path_sam2"):
            usage = "unavailable in current runtime"
        elif method.startswith("uni2"):
            usage = "not run by this driver"
        else:
            usage = "cpu"
        lines.append(f"- `{method}`: {usage}")
    lines.append("")
    write_text(outdir / "gpu_usage_report.md", "\n".join(lines))


def write_path_sam2_status(repo_root: Path, outdir: Path) -> None:
    external_repo = repo_root.parent / "_external" / "SAM2PATH"
    lines = [
        "# Path-SAM2 Status",
        "",
        "- Real Path-SAM2 is not installed in the current GPU singularity runtime.",
        "- The public `SAM2PATH` repository was cloned for inspection on the remote host.",
        "- It appears to be training-oriented and does not expose a simple benchmark-ready semantic inference entrypoint plus suitable released pathology weights for these images.",
        "",
        f"- inspected_external_repo: `{external_repo}`",
        f"- current_singularity_runtime: `{repo_root / 'singularity' / 'cellphenotyper-2.2-gpu-amd64.sif'}`",
        "",
        "## Result",
        "",
        "- `path_sam2_semantic`: recorded as unavailable",
        "- `path_sam2_semantic_plus_cleanup`: recorded as unavailable",
        "",
        "A future integration step can add the real Path-SAM2 runtime once a stable inference recipe and weights are available.",
        "",
    ]
    write_text(outdir / "reports" / "path_sam2_status.md", "\n".join(lines))


def write_manifest(repo_root: Path, outdir: Path) -> None:
    rows = discover_input_manifest_rows(repo_root)
    path = outdir / "input_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample_id", "image_path", "roi_geojson_path", "image_shape", "notes"])
        writer.writeheader()
        writer.writerows(rows)


def write_commands_run(repo_root: Path, outdir: Path, benchmark_cmd: list[str]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(repo_root))}",
        " ".join(shlex.quote(part) for part in benchmark_cmd),
        "",
    ]
    path = outdir / "commands_run.sh"
    write_text(path, "\n".join(lines))
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    outdir = Path(args.outdir).resolve()
    if args.clean and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "reports").mkdir(parents=True, exist_ok=True)

    all_methods = [spec.name for spec in BENCHMARK_METHODS]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or all_methods

    write_environment_report(repo_root, outdir)
    write_manifest(repo_root, outdir)
    write_gpu_usage_report(outdir, methods)
    write_path_sam2_status(repo_root, outdir)

    benchmark_cmd = [
        sys.executable,
        str((repo_root / args.benchmark_script).resolve()),
        "--repo-root",
        str(repo_root),
        "--outdir",
        str(outdir),
        "--clean",
        "--methods",
        ",".join(methods),
    ]
    write_commands_run(repo_root, outdir, benchmark_cmd)

    print(f"[INFO] running benchmark with {len(methods)} methods", flush=True)
    subprocess.run(benchmark_cmd, check=True, cwd=str(repo_root))

    print(f"[OK] benchmark root: {outdir}", flush=True)
    print(f"[OK] environment report: {outdir / 'environment_report.md'}", flush=True)
    print(f"[OK] manifest: {outdir / 'input_manifest.csv'}", flush=True)
    print(f"[OK] global report: {outdir / 'reports' / 'global_report.md'}", flush=True)
    print(f"[OK] comparison panels: {outdir / 'comparison_panels'}", flush=True)
    print(f"[OK] gallery: {outdir / 'galleries' / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
