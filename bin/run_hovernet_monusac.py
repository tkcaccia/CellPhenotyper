#!/usr/bin/env python3
"""Run the official PyTorch HoVer-Net MoNuSAC model on a StarDist crop."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path


MONUSAC_TYPE_INFO = {
    "0": ["background", [255, 255, 255]],
    "1": ["epithelial", [255, 0, 0]],
    "2": ["lymphocyte", [0, 0, 255]],
    "3": ["macrophage", [0, 255, 0]],
    "4": ["neutrophil", [255, 255, 0]],
}


def monusac_type_name(type_id: object) -> str:
    """Return the checkpoint taxonomy label while preserving unknown IDs."""
    key = str(type_id)
    return MONUSAC_TYPE_INFO.get(key, [f"unknown_{key}"])[0]


def read_source_mpp(shift_path: Path, fallback: float) -> float:
    payload = json.loads(shift_path.read_text())
    for key in ("source_mpp", "microns_per_pixel"):
        value = payload.get(key)
        if value is not None and float(value) > 0:
            return float(value)
    if fallback > 0:
        return fallback
    raise RuntimeError("No valid MPP in shift.json; set --default-mpp explicitly")


def auto_batch_size(requested: int, gpu: str) -> int:
    if requested > 0:
        return requested
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        memory_mib = int(output.strip().splitlines()[0])
    except Exception:
        return 16
    if memory_mib >= 70 * 1024:
        return 128
    if memory_mib >= 40 * 1024:
        return 64
    if memory_mib >= 20 * 1024:
        return 32
    return 16


def objective_power_for_mpp(mpp: float) -> int:
    return max(1, int(round(10.0 / mpp)))


def resolve_runtime_paths(image: str, shift: str, repo: str, checkpoint: str, outdir: str) -> tuple[Path, ...]:
    """Resolve task-local paths before HoVer-Net changes its working directory."""
    return tuple(Path(value).resolve() for value in (image, shift, repo, checkpoint, outdir))


def require_readable_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"{label} is not readable by uid {os.geteuid()}: {path}")


def prepare_runtime_repo(repo: Path, outdir: Path) -> Path:
    """Copy the upstream runtime to a task-writable directory."""
    runtime_repo = outdir / "hovernet_runtime"
    shutil.rmtree(runtime_repo, ignore_errors=True)
    shutil.copytree(repo, runtime_repo)
    for path in (runtime_repo, *runtime_repo.rglob("*")):
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IWUSR | (stat.S_IXUSR if path.is_dir() else 0))
    return runtime_repo


def enable_cache_resume_runtime(runtime_repo: Path, cache_dir: Path, prediction_cache: Path) -> None:
    """Patch an isolated runtime copy to reuse a completed prediction mmap."""
    pred_map = prediction_cache / "pred_map.npy"
    if not pred_map.is_file():
        raise FileNotFoundError(f"HoVer-Net prediction cache is missing: {pred_map}")
    wsi_path = runtime_repo / "infer" / "wsi.py"
    source = wsi_path.read_text()
    allocation = '''        self.wsi_pred_map = np.lib.format.open_memmap(
            "%s/pred_map.npy" % self.cache_path,
            mode="w+",
            shape=tuple(self.wsi_proc_shape) + (out_ch,),
            dtype=np.float32,
        )'''
    resumed_allocation = '''        if os.environ.get("HOVERNET_RESUME_PRED_MAP") == "1":
            self.wsi_pred_map = np.load(
                "%s/pred_map.npy" % self.cache_path, mmap_mode="r"
            )
            expected_shape = tuple(self.wsi_proc_shape) + (out_ch,)
            if self.wsi_pred_map.shape != expected_shape:
                raise RuntimeError(
                    "Cached prediction shape %s does not match expected %s"
                    % (self.wsi_pred_map.shape, expected_shape)
                )
        else:
            self.wsi_pred_map = np.lib.format.open_memmap(
                "%s/pred_map.npy" % self.cache_path,
                mode="w+",
                shape=tuple(self.wsi_proc_shape) + (out_ch,),
                dtype=np.float32,
            )'''
    inference = "        self.__get_raw_prediction(chunk_info_list, patch_info_list)"
    resumed_inference = '''        if os.environ.get("HOVERNET_RESUME_PRED_MAP") == "1":
            log_info("Resume: using completed pred_map.npy; skipping raw inference")
        else:
            self.__get_raw_prediction(chunk_info_list, patch_info_list)'''
    if source.count(allocation) != 1 or source.count(inference) != 1:
        raise RuntimeError("Unsupported HoVer-Net wsi.py layout for prediction-cache resume")
    wsi_path.write_text(source.replace(allocation, resumed_allocation).replace(inference, resumed_inference))
    (cache_dir / "pred_map.npy").symlink_to(pred_map.resolve())


def prepare_cache_resume_runtime(repo: Path, outdir: Path, cache_dir: Path, prediction_cache: Path) -> Path:
    """Create and patch an isolated upstream copy for prediction-cache resume."""
    runtime_repo = prepare_runtime_repo(repo, outdir)
    enable_cache_resume_runtime(runtime_repo, cache_dir, prediction_cache)
    return runtime_repo


def make_pyramidal_input(src: Path, dst: Path, scale: float, target_mpp: float) -> tuple[int, int]:
    import pyvips

    image = pyvips.Image.new_from_file(str(src), access="sequential")
    if image.bands > 3:
        image = image[:3]
    if image.bands == 1:
        image = image.bandjoin([image, image])
    original = (image.width, image.height)
    if abs(scale - 1.0) > 1e-6:
        image = image.resize(scale, kernel="lanczos3")
    objective_power = objective_power_for_mpp(target_mpp)
    image = image.copy(xres=1000.0 / target_mpp, yres=1000.0 / target_mpp)
    description = (
        f"Aperio Image Library v12.4.0\n{image.width}x{image.height} "
        f"[0,0 {image.width}x{image.height}] (512x512) JPEG/RGB Q=92"
        f"|AppMag = {objective_power}|MPP = {target_mpp:.8g}"
    )
    image.set_type(pyvips.GValue.gstr_type, "image-description", description)
    image.tiffsave(
        str(dst), tile=True, tile_width=512, tile_height=512, pyramid=True,
        compression="jpeg", Q=92, bigtiff=True, resunit="cm",
    )
    return original


def normalize_output(raw_json: Path, output_json: Path, inverse_scale: float, metadata: dict) -> None:
    raw = json.loads(raw_json.read_text())
    instances = raw.get("nuclei", raw.get("nuc", raw)) if isinstance(raw, dict) else raw
    normalized = {
        "model": "HoVer-Net",
        "checkpoint": "MoNuSAC",
        "metadata": metadata,
        "type_map": {key: value[0] for key, value in MONUSAC_TYPE_INFO.items()},
        "cells": [],
    }
    iterable = instances.items() if isinstance(instances, dict) else enumerate(instances)
    for raw_id, item in iterable:
        if not isinstance(item, dict) or "centroid" not in item:
            continue
        centroid = [float(v) * inverse_scale for v in item["centroid"]]
        contour = [[float(p[0]) * inverse_scale, float(p[1]) * inverse_scale] for p in item.get("contour", [])]
        type_id = item.get("type")
        normalized["cells"].append({
            "id": str(raw_id), "centroid": centroid, "contour": contour,
            "type_id": type_id, "type": monusac_type_name(type_id),
            "type_prob": item.get("type_prob"),
        })
    output_json.write_text(json.dumps(normalized))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--shift", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--repo", default=os.environ.get("HOVERNET_REPO_DIR", "/opt/cellphenotyper/third_party/hover_net"))
    parser.add_argument("--checkpoint", default=os.environ.get("HOVERNET_MONUSAC_CHECKPOINT", "/opt/cellphenotyper/models/hovernet/hovernet_fast_monusac_type_tf2pytorch.tar"))
    parser.add_argument("--target-mpp", type=float, default=0.25)
    parser.add_argument("--default-mpp", type=float, default=0.0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--inference-workers", type=int, default=4)
    parser.add_argument("--postproc-workers", type=int, default=8)
    parser.add_argument("--chunk-shape", type=int, default=10000)
    parser.add_argument("--tile-shape", type=int, default=2048)
    parser.add_argument("--prediction-cache", default="")
    args = parser.parse_args()

    # HoVer-Net is launched from its repository, so every task-local path must
    # be absolute before changing the child process working directory.
    image, shift, repo, checkpoint, outdir = resolve_runtime_paths(
        args.image, args.shift, args.repo, args.checkpoint, args.outdir,
    )
    require_readable_file(repo / "run_infer.py", "Official HoVer-Net runtime")
    require_readable_file(checkpoint, "MoNuSAC checkpoint")
    outdir.mkdir(parents=True, exist_ok=True)
    input_dir, raw_dir, cache_dir = outdir / "input", outdir / "raw", outdir / "cache"
    input_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    source_mpp = read_source_mpp(shift, args.default_mpp)
    batch_size = auto_batch_size(args.batch_size, args.gpu)
    scale = source_mpp / args.target_mpp
    normalized_slide = input_dir / "hovernet_input.tif"
    width, height = make_pyramidal_input(image, normalized_slide, scale, args.target_mpp)
    # Upstream initializes debug.log in its current directory. The bundled
    # repository is read-only in Singularity, so always run an isolated copy.
    runtime_repo = prepare_runtime_repo(repo, outdir)
    if args.prediction_cache:
        prediction_cache = Path(args.prediction_cache).resolve()
        enable_cache_resume_runtime(runtime_repo, cache_dir, prediction_cache)
    type_info_path = outdir / "monusac_type_info.json"
    type_info_path.write_text(json.dumps(MONUSAC_TYPE_INFO, indent=2))
    compatibility_launcher = (
        "import runpy,sys,numpy as np;"
        "[np.__dict__.setdefault(k,v) for k,v in "
        "{'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'str':str}.items()];"
        "p=sys.argv.pop(1);sys.argv[0]=p;runpy.run_path(p,run_name='__main__')"
    )
    cmd = [
        "python", "-c", compatibility_launcher, str(runtime_repo / "run_infer.py"),
        f"--gpu={args.gpu}", "--nr_types=5",
        f"--type_info_path={type_info_path}",
        f"--model_path={checkpoint}", "--model_mode=fast",
        f"--nr_inference_workers={args.inference_workers}",
        f"--nr_post_proc_workers={args.postproc_workers}", f"--batch_size={batch_size}",
        "wsi", f"--input_dir={input_dir}", f"--output_dir={raw_dir}", f"--cache_path={cache_dir}",
        "--proc_mag=40", f"--chunk_shape={args.chunk_shape}", f"--tile_shape={args.tile_shape}",
    ]
    env = os.environ.copy()
    runtime_cache_dir = outdir / "runtime_cache"
    (runtime_cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(runtime_cache_dir / "matplotlib")
    env["XDG_CACHE_HOME"] = str(runtime_cache_dir)
    if args.prediction_cache:
        env["HOVERNET_RESUME_PRED_MAP"] = "1"
    subprocess.run(cmd, cwd=runtime_repo, env=env, check=True)
    candidates = sorted(raw_dir.rglob("*.json"))
    if not candidates:
        raise RuntimeError(f"HoVer-Net produced no instance JSON under {raw_dir}")
    raw_json = next((p for p in candidates if p.stem == normalized_slide.stem), candidates[0])
    metadata = {
        "source_mpp": source_mpp, "target_mpp": args.target_mpp, "coordinate_scale": scale,
        "source_width": width, "source_height": height, "batch_size": batch_size,
    }
    # Always emit one stable schema, even when no coordinate rescaling is needed.
    normalize_output(raw_json, outdir / "hovernet_cells.json", 1.0 / scale, metadata)
    (outdir / "hovernet_metadata.json").write_text(json.dumps(metadata, indent=2))
    shutil.rmtree(cache_dir, ignore_errors=True)
    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(runtime_cache_dir, ignore_errors=True)
    if runtime_repo != repo:
        shutil.rmtree(runtime_repo, ignore_errors=True)


if __name__ == "__main__":
    main()
