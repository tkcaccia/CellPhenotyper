#!/usr/bin/env python3
"""Run official CellViT++ inference and normalize its WSI cell output."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware_runtime import auto_batch_from_free_vram
from model_provenance import checkpoint_record
from cellvit_embedding_io import capture_inputs, check_inputs, file_record, sha256


PANNUKE_TYPE_MAP = {
    "1": "neoplastic",
    "2": "inflammatory",
    "3": "connective",
    "4": "dead",
    "5": "epithelial",
}

RAY_UNIX_SOCKET_MAX_BYTES = 107

# Runs in the configured entrypoint's interpreter, with no CellViT/Torch import.
# Package imports are resolved without executing cellvit/__init__.py.
_RUNTIME_PROBE = r'''
import ast, hashlib, importlib.metadata as md, importlib.util, json, platform, sys
from pathlib import Path
executable = Path(sys.argv[1]).resolve()
sys.path[0] = str(executable.parent)
distribution = md.distribution("cellvit")
entrypoints = [entry for entry in distribution.entry_points if entry.group == "console_scripts" and entry.name == "cellvit-inference"]
if len(entrypoints) != 1:
    raise ValueError("No unique CellViT console entrypoint")
entry = entrypoints[0]
module, attribute = entry.value.split(":", 1)
attribute = attribute.split("[", 1)[0].strip()
tree = ast.parse(executable.read_text())
imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == module]
aliases = {name.asname or name.name for node in imports for name in node.names if name.name == attribute}
called = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in aliases for node in ast.walk(tree))
if not aliases or not called or not module.startswith("cellvit."):
    raise ValueError("Configured executable does not call the installed CellViT entrypoint")
package = Path(distribution.locate_file("cellvit")).resolve()
spec = importlib.util.find_spec("cellvit")
if spec is None or spec.origin is None or Path(spec.origin).resolve().parent != package:
    raise ValueError("Imported CellViT package differs from installed distribution")
def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
parts = []
def walk(path, relative, ancestors):
    resolved = path.resolve()
    if not resolved.is_relative_to(package) or resolved in ancestors:
        raise ValueError("Package source path escape/cycle")
    if resolved.is_dir():
        for child in sorted(resolved.iterdir(), key=lambda item: item.name):
            if child.name != "__pycache__":
                walk(child, relative + "/" + child.name, ancestors | {resolved})
    elif resolved.is_file() and resolved.suffix not in (".pyc", ".pyo"):
        parts.append([relative, digest(resolved)])
walk(package, "cellvit", set())
if not parts:
    raise ValueError("No installed CellViT source payload")
for name in ("METADATA", "entry_points.txt", "WHEEL"):
    text = distribution.read_text(name)
    if text is not None:
        parts.append(["distribution/" + name, hashlib.sha256(text.encode()).hexdigest()])
versions = {}
for name in ("torch", "torchvision", "numpy", "timm", "ray", "pyvips", "openslide-python", "cupy-cuda12x"):
    try: versions[name] = md.version(name)
    except md.PackageNotFoundError: versions[name] = None
portable = {"package_version": distribution.version, "entry_point": entry.value,
    "package_source_sha256": hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest(),
    "executable_sha256": digest(executable), "interpreter_sha256": digest(sys.executable),
    "python_version": platform.python_version(), "python_implementation": platform.python_implementation(),
    "platform": sys.platform, "machine": platform.machine(), "dependency_versions": versions}
print(json.dumps({"package_version": distribution.version, "interpreter": sys.executable,
    "interpreter_prefix": sys.prefix, "package_directory": str(package), "portable_identity": portable}))
'''


def executable_runtime_identity(executable, env=None):
    """Observe the real configured Python CLI; unknown launchers stay unknown."""
    environment = os.environ.copy() if env is None else env
    configured = shutil.which(str(executable), path=environment.get("PATH")) if "/" not in str(executable) else str(executable)
    if not configured:
        raise FileNotFoundError(f"CellViT executable not found: {executable}")
    path = Path(configured).resolve(strict=True)
    record = {"status": "unverified", "executable": str(path), "executable_sha256": sha256(path),
              "package_version": None, "portable_identity": None}
    with path.open("rb") as handle:
        first_line = handle.readline(8192)
    try:
        if not first_line.startswith(b"#!"):
            raise ValueError()
        words = shlex.split(first_line[2:].decode().strip())
        if len(words) == 2 and Path(words[0]).name == "env":
            interpreter = shutil.which(words[1], path=environment.get("PATH"))
        elif len(words) == 1 and Path(words[0]).is_absolute():
            interpreter = words[0]
        else:
            raise ValueError()
        if not interpreter or not re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", Path(interpreter).name):
            raise ValueError()
        # Preserve the venv interpreter symlink when executing: resolving it
        # first would accidentally query the base/wrapper environment again.
        interpreter = str(Path(interpreter).absolute())
    except (UnicodeError, ValueError):
        return {**record, "reason": "unsupported_or_non_python_launcher"}
    try:
        result = subprocess.run([interpreter, "-c", _RUNTIME_PROBE, str(path)],
                                env=environment, capture_output=True, text=True, timeout=30, check=True)
        observed = json.loads(result.stdout)
        if not os.path.samefile(observed["interpreter"], interpreter):
            raise ValueError("Interpreter identity differs")
        if observed["portable_identity"]["executable_sha256"] != record["executable_sha256"] or sha256(path) != record["executable_sha256"]:
            raise RuntimeError("CellViT executable changed during runtime identity capture")
        return {**record, **observed, "status": "verified_configured_python_entrypoint"}
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return {**record, "interpreter": interpreter, "reason": "configured_entrypoint_identity_unverified"}


def source_mpp(shift_path: Path, fallback: float) -> float:
    data = json.loads(shift_path.read_text())
    value = data.get("source_mpp", data.get("microns_per_pixel", fallback))
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise RuntimeError("No valid MPP in shift.json; set --default-mpp explicitly")
    return float(value)


def auto_batch_size(requested: int, gpu: int) -> int:
    # CellViT++ 1.0.9 enforces a minimum batch size of two.
    return auto_batch_from_free_vram(
        requested,
        gpu,
        tiers=((68 * 1024, 32), (38 * 1024, 16), (20 * 1024, 8)),
        fallback=2,
        minimum=2,
        maximum=48,
    )


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


def validate_ray_temp_dir(temp_dir: Path) -> None:
    representative_socket = (
        temp_dir / "ray" / "session_2000-01-01_00-00-00_000000_0000000"
        / "sockets" / "plasma_store"
    )
    socket_bytes = len(os.fsencode(str(representative_socket)))
    if socket_bytes > RAY_UNIX_SOCKET_MAX_BYTES:
        raise RuntimeError(
            f"Ray temporary path is too long for an AF_UNIX socket ({socket_bytes} bytes): "
            f"{temp_dir}"
        )


def prepare_runtime_env(outdir: Path) -> tuple[dict[str, str], Path, Path]:
    runtime_dir = outdir / "runtime"
    for child in (runtime_dir / "cache", runtime_dir / "matplotlib"):
        child.mkdir(parents=True, exist_ok=True)
    temp_root = Path(os.environ.get("CELLVIT_TMP_ROOT", "/tmp"))
    temp_root.mkdir(parents=True, exist_ok=True)
    ray_temp_dir = Path(tempfile.mkdtemp(prefix="cellvit_", dir=str(temp_root)))
    try:
        validate_ray_temp_dir(ray_temp_dir)
    except Exception:
        shutil.rmtree(ray_temp_dir, ignore_errors=True)
        raise
    env = os.environ.copy()
    env["TMPDIR"] = str(ray_temp_dir)
    env["RAY_TMPDIR"] = str(ray_temp_dir)
    env["XDG_CACHE_HOME"] = str(runtime_dir / "cache")
    env["MPLCONFIGDIR"] = str(runtime_dir / "matplotlib")
    return env, runtime_dir, ray_temp_dir


def make_pyramid(src: Path, dst: Path) -> dict:
    import pyvips

    image = pyvips.Image.new_from_file(str(src), access="sequential")
    source_bands, source_format = image.bands, image.format
    conversion = "none"
    if image.bands > 3:
        image = image[:3]
        conversion = "first_three_channels"
    if image.bands == 1:
        image = image.bandjoin([image, image])
        conversion = "replicate_grayscale_to_rgb"
    if image.bands != 3:
        raise ValueError("CellViT preprocessing requires grayscale or at least three source channels")
    image.tiffsave(
        str(dst), tile=True, tile_width=512, tile_height=512, pyramid=True,
        compression="jpeg", Q=92, bigtiff=True,
    )
    return {"method": "libvips_tiffsave", "libvips_version": ".".join(str(pyvips.version(i)) for i in range(3)),
            "native_size_px": [image.width, image.height], "source_bands": source_bands,
            "source_pixel_format": source_format, "bands": 3, "channel_conversion": conversion,
            "tile_size_px": [512, 512], "pyramid": True, "compression": "jpeg", "jpeg_quality": 92, "bigtiff": True}


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
    parser.add_argument("--resolution-json", default="", help="Passed source-resolution report; required for source-bound embedding export")
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
    parser.add_argument("--export-embeddings", action="store_true", help="Export ID-aligned CellViT graph tokens as a numeric NPY matrix")
    parser.add_argument("--clean-tissue-mask", default="")
    args = parser.parse_args()

    image, shift, outdir = resolve_runtime_paths(args.image, args.shift, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    executable = shutil.which(args.executable) if "/" not in args.executable else args.executable
    if not executable or not Path(executable).exists():
        raise FileNotFoundError(f"CellViT++ executable not found: {args.executable}")
    cache_dir = Path(os.environ.get("CELLVIT_CACHE", str(Path.home() / ".cache" / "cellvit")))
    model_path = required_model_path(cache_dir, args.model)
    require_readable_file(model_path, f"CellViT++ {args.model} checkpoint")
    embedding_support = None
    source_binding, source_paths = (capture_inputs(image, shift, args.resolution_json, args.clean_tissue_mask)
                                    if args.resolution_json or args.export_embeddings else (None, None))
    checkpoint_before = checkpoint_record(model_path, logical_name=args.model)
    if args.export_embeddings:
        from cellvit_embedding_io import COMPLETION, FILES
        if any((outdir / name).exists() or (outdir / name).is_symlink()
               for name in [COMPLETION, *(FILES[key] for key in ("embeddings", "ids", "metadata", "raw_population"))]):
            raise FileExistsError("CellViT embedding artifacts already exist; choose a new output directory")
        from cellvit_embeddings import check_graph_support, require_graph_loader
        require_graph_loader()
        embedding_support = check_graph_support(str(executable))
    prepared = outdir / "cellvit_input.tif"
    preprocessing = make_pyramid(image, prepared)
    prepared_record = file_record(prepared) if args.export_embeddings else None
    mpp = source_binding["geometry"]["source_mpp"] if source_binding else source_mpp(shift, args.default_mpp)
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
    if args.export_embeddings:
        cmd.append("--graph")
    cmd += ["process_wsi", "--wsi_path", str(prepared), "--wsi_mpp", str(mpp)]
    env, runtime_dir, ray_temp_dir = prepare_runtime_env(outdir)
    try:
        runtime_before = executable_runtime_identity(executable, env)
        subprocess.run(cmd, check=True, env=env)
        candidates = sorted(raw_dir.rglob("cells.json"))
        if not candidates:
            raise RuntimeError(f"CellViT++ produced no cells.json under {raw_dir}")
        if args.export_embeddings and len(candidates) != 1:
            raise RuntimeError("CellViT embedding export requires exactly one cells.json for this image")
        runtime_after = executable_runtime_identity(executable, env)
        if runtime_after != runtime_before or checkpoint_record(model_path, logical_name=args.model) != checkpoint_before:
            raise ValueError("CellViT runtime/checkpoint changed during inference")
        if source_paths:
            check_inputs(source_paths, source_binding["inputs"])
        cellvit_version = runtime_before.get("package_version") if runtime_before["status"] == "verified_configured_python_entrypoint" else None
        metadata = {
            "model": args.model, "taxonomy": args.taxonomy, "source_mpp": mpp,
            "batch_size": batch_size, "ray_workers": ray_workers,
            "ray_worker_cpus": ray_worker_cpus,
            "runtime_identity": runtime_before, "preprocessing": preprocessing, "amp": args.amp,
            "model_provenance": {
                "source_repository": "https://github.com/TIO-IKIM/CellViT-plus-plus",
                "requested_revision": cellvit_version,
                "resolved_revision": (
                    f"package:cellvit=={cellvit_version}"
                    if cellvit_version else None
                ),
                "cache_path": str(cache_dir.resolve()),
                "checkpoints": [checkpoint_before],
            },
        }
        normalized_path = outdir / "cellvit_cells.json"
        normalize_output(candidates[0], normalized_path, args.taxonomy, metadata)
        if args.clean_tissue_mask:
            from grandqc_mask import filter_cell_payload
            payload, filter_summary = filter_cell_payload(
                json.loads(normalized_path.read_text()), args.clean_tissue_mask, shift,
            )
            normalized_path.write_text(json.dumps(payload))
            metadata["grandqc_filter"] = filter_summary
        if args.export_embeddings:
            from cellvit_embeddings import export_embeddings
            execution = {**source_binding, "runtime_identity": runtime_before,
                         "checkpoint": checkpoint_before, "model": args.model, "taxonomy": args.taxonomy,
                         "preprocessing": preprocessing, "prepared_image": prepared_record, "amp": args.amp}
            metadata["embeddings"] = export_embeddings(
                candidates[0], normalized_path, outdir,
                {**embedding_support, "model": args.model,
                 "model_provenance": metadata["model_provenance"]}, execution=execution,
            )
        (outdir / "cellvit_metadata.json").write_text(json.dumps(metadata, indent=2))
        if args.export_embeddings:
            from cellvit_embedding_io import complete_embedding_bundle
            check_inputs(source_paths, source_binding["inputs"])
            if file_record(prepared) != prepared_record or executable_runtime_identity(executable, env) != runtime_before or checkpoint_record(model_path, logical_name=args.model) != checkpoint_before:
                raise ValueError("CellViT prepared image/runtime/checkpoint changed before completion")
            complete_embedding_bundle(outdir, execution, source_paths=source_paths)
        prepared.unlink(missing_ok=True)
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)
        shutil.rmtree(ray_temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
