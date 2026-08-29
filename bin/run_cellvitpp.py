#!/usr/bin/env python3
"""Run official CellViT++ inference and normalize its WSI cell output."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


PANNUKE_TYPE_MAP = {
    "1": "neoplastic",
    "2": "inflammatory",
    "3": "connective",
    "4": "dead",
    "5": "epithelial",
}


def source_mpp(shift_path: Path, fallback: float) -> float:
    data = json.loads(shift_path.read_text())
    value = data.get("source_mpp", data.get("microns_per_pixel", fallback))
    if value is None or float(value) <= 0:
        raise RuntimeError("No valid MPP in shift.json; set --default-mpp explicitly")
    return float(value)


def auto_batch_size(requested: int, gpu: int) -> int:
    if requested > 0:
        return max(2, min(48, requested))
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        memory_mib = int(output.strip().splitlines()[0])
    except Exception:
        return 2
    if memory_mib >= 70 * 1024:
        return 32
    if memory_mib >= 40 * 1024:
        return 16
    if memory_mib >= 22 * 1024:
        return 8
    # CellViT++ 1.0.9 enforces a minimum batch size of two.
    return 2


def resolve_runtime_paths(image: str, shift: str, outdir: str) -> tuple[Path, Path, Path]:
    """Give CellViT and its Ray workers stable paths independent of their CWD."""
    return tuple(Path(value).resolve() for value in (image, shift, outdir))


def required_model_path(cache_dir: Path, model: str) -> Path:
    filename = "CellViT-256-x40-AMP.pth" if model == "HIPT" else "CellViT-SAM-H-x40-AMP.pth"
    return cache_dir / filename


def require_readable_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{label} is not readable by uid {os.geteuid()}: {path}")


def prepare_runtime_env(outdir: Path) -> tuple[dict[str, str], Path]:
    runtime_dir = outdir / "runtime"
    for child in (runtime_dir / "tmp", runtime_dir / "cache", runtime_dir / "matplotlib"):
        child.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(runtime_dir / "tmp")
    env["RAY_TMPDIR"] = str(runtime_dir / "tmp")
    env["XDG_CACHE_HOME"] = str(runtime_dir / "cache")
    env["MPLCONFIGDIR"] = str(runtime_dir / "matplotlib")
    return env, runtime_dir


def make_pyramid(src: Path, dst: Path) -> None:
    import pyvips

    image = pyvips.Image.new_from_file(str(src), access="sequential")
    if image.bands > 3:
        image = image[:3]
    if image.bands == 1:
        image = image.bandjoin([image, image])
    image.tiffsave(
        str(dst), tile=True, tile_width=512, tile_height=512, pyramid=True,
        compression="jpeg", Q=92, bigtiff=True,
    )


def normalize_type_map(raw_map: object, taxonomy: str) -> dict[str, str]:
    if isinstance(raw_map, dict) and raw_map:
        return {str(key): str(value).strip().lower() for key, value in raw_map.items()}
    if taxonomy.strip().lower() == "pannuke":
        return dict(PANNUKE_TYPE_MAP)
    return {}


def normalize_output(raw_json: Path, output_json: Path, taxonomy: str, metadata: dict) -> None:
    """Emit one stable schema with named classes and retained upstream IDs."""
    payload = json.loads(raw_json.read_text())
    cells = payload.get("cells", payload) if isinstance(payload, dict) else payload
    type_map = normalize_type_map(payload.get("type_map") if isinstance(payload, dict) else None, taxonomy)
    iterable = cells.items() if isinstance(cells, dict) else enumerate(cells)
    normalized = {
        "model": "CellViT++",
        "taxonomy": taxonomy,
        "metadata": metadata,
        "wsi_metadata": payload.get("wsi_metadata", {}) if isinstance(payload, dict) else {},
        "type_map": type_map,
        "cells": [],
    }
    for fallback_id, item in iterable:
        if not isinstance(item, dict) or "centroid" not in item:
            continue
        type_id = item.get("type")
        key = str(type_id)
        cell = dict(item)
        cell["id"] = str(item.get("id", fallback_id))
        cell["type_id"] = type_id
        cell["type"] = type_map.get(key, f"unknown_{key}")
        normalized["cells"].append(cell)
    output_json.write_text(json.dumps(normalized))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--shift", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--executable", default=os.environ.get("CELLVIT_EXECUTABLE", "cellvit-inference"))
    parser.add_argument("--model", choices=("SAM", "HIPT"), default="HIPT")
    parser.add_argument("--taxonomy", default="pannuke")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--memory-mb", type=int, default=32768)
    parser.add_argument("--ray-workers", type=int, default=1)
    parser.add_argument("--ray-worker-cpus", type=int, default=0)
    parser.add_argument("--default-mpp", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    image, shift, outdir = resolve_runtime_paths(args.image, args.shift, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(args.executable) if "/" not in args.executable else args.executable
    if not executable or not Path(executable).exists():
        raise FileNotFoundError(f"CellViT++ executable not found: {args.executable}")
    cache_dir = Path(os.environ.get("CELLVIT_CACHE", str(Path.home() / ".cache" / "cellvit")))
    require_readable_file(required_model_path(cache_dir, args.model), f"CellViT++ {args.model} checkpoint")
    prepared = outdir / "cellvit_input.tif"
    make_pyramid(image, prepared)
    mpp = source_mpp(shift, args.default_mpp)
    batch_size = auto_batch_size(args.batch_size, args.gpu)
    raw_dir = outdir / "raw"
    ray_workers = max(1, args.ray_workers)
    ray_worker_cpus = max(1, args.ray_worker_cpus or (args.cpus // ray_workers))
    cmd = [
        str(executable), "--model", args.model, "--nuclei_taxonomy", args.taxonomy,
        "--gpu", str(args.gpu), "--batch_size", str(batch_size), "--outdir", str(raw_dir),
        "--geojson", "--cpu_count", str(args.cpus), "--memory", str(args.memory_mb),
        "--ray_worker", str(ray_workers), "--ray_remote_cpus", str(ray_worker_cpus),
    ]
    if args.amp:
        cmd.append("--enforce_amp")
    cmd += ["process_wsi", "--wsi_path", str(prepared), "--wsi_mpp", str(mpp)]
    env, runtime_dir = prepare_runtime_env(outdir)
    subprocess.run(cmd, check=True, env=env)
    candidates = sorted(raw_dir.rglob("cells.json"))
    if not candidates:
        raise RuntimeError(f"CellViT++ produced no cells.json under {raw_dir}")
    metadata = {
        "model": args.model, "taxonomy": args.taxonomy, "source_mpp": mpp,
        "batch_size": batch_size, "ray_workers": ray_workers,
        "ray_worker_cpus": ray_worker_cpus,
    }
    normalize_output(candidates[0], outdir / "cellvit_cells.json", args.taxonomy, metadata)
    (outdir / "cellvit_metadata.json").write_text(json.dumps(metadata, indent=2))
    prepared.unlink(missing_ok=True)
    shutil.rmtree(runtime_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
