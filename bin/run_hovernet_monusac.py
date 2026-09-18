#!/usr/bin/env python3
"""Run the official PyTorch HoVer-Net MoNuSAC model on a StarDist crop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware_runtime import auto_batch_from_free_vram


MONUSAC_TYPE_INFO = {
    "0": ["background", [255, 255, 255]],
    "1": ["epithelial", [255, 0, 0]],
    "2": ["lymphocyte", [0, 0, 255]],
    "3": ["macrophage", [0, 255, 0]],
    "4": ["neutrophil", [255, 255, 0]],
}

MONUSAC_MODEL_SCOPE = {
    "target": "MoNuSAC-annotated epithelial, lymphocyte, macrophage, and neutrophil nuclei",
    "exhaustive_nuclei_detector": False,
    "known_omissions": [
        "fibroblasts and other nuclei not annotated as positive classes in MoNuSAC",
    ],
    "interpretation": (
        "This checkpoint is not expected to return the same total nucleus count as "
        "a general-purpose detector. Use its detections as class-specific supporting evidence."
    ),
    "reference": "https://github.com/simongraham/hovernet_inference#datasets",
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
    return auto_batch_from_free_vram(
        requested,
        gpu,
        tiers=((68 * 1024, 128), (38 * 1024, 64), (18 * 1024, 32)),
        fallback=16,
    )


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_storage(path: Path) -> dict[str, int]:
    """Return logical and allocated bytes for a transient cache tree."""
    logical = allocated = files = 0
    if not path.exists():
        return {"logical_bytes": 0, "allocated_bytes": 0, "files": 0}
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        stat_result = candidate.stat()
        logical += int(stat_result.st_size)
        allocated += int(getattr(stat_result, "st_blocks", 0)) * 512
        files += 1
    return {"logical_bytes": logical, "allocated_bytes": allocated, "files": files}


def git_revision(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


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


def enable_zarr_cache_runtime(runtime_repo: Path) -> None:
    """Patch an isolated upstream copy to use lossless chunk-compressed maps.

    Upstream allocates one float32 four-channel prediction map and one int32
    instance map over the complete processed WSI rectangle. Zarr preserves the
    exact dtypes and slice semantics while Blosc/Zstd compresses each tile on
    disk, especially background and spatially coherent instance labels.
    """
    wsi_path = runtime_repo / "infer" / "wsi.py"
    source = wsi_path.read_text()

    import_anchor = "import numpy as np\n"
    if source.count(import_anchor) != 1:
        raise RuntimeError("Unsupported HoVer-Net wsi.py imports for Zarr cache")
    source = source.replace(
        import_anchor,
        import_anchor + "import zarr\nfrom numcodecs import Blosc\n",
    )

    old_assemble = '''def _assemble_and_flush(wsi_pred_map_mmap_path, chunk_info, patch_output_list):
    """Assemble the results. Write to newly created holder for this wsi"""
    wsi_pred_map_ptr = np.load(wsi_pred_map_mmap_path, mmap_mode="r+")
    chunk_pred_map = wsi_pred_map_ptr[
        chunk_info[1][0][0] : chunk_info[1][1][0],
        chunk_info[1][0][1] : chunk_info[1][1][1],
    ]
    if patch_output_list is None:
        # chunk_pred_map[:] = 0 # zero flush when there is no-results
        # print(chunk_info.flatten(), 'flush 0')
        return

    for pinfo in patch_output_list:
        pcoord, pdata = pinfo
        pdata = np.squeeze(pdata)
        pcoord = np.squeeze(pcoord)[:2]
        chunk_pred_map[
            pcoord[0] : pcoord[0] + pdata.shape[0],
            pcoord[1] : pcoord[1] + pdata.shape[1],
        ] = pdata
    # print(chunk_info.flatten(), 'pass')
    return
'''
    new_assemble = '''def _assemble_and_flush(wsi_pred_map_mmap_path, chunk_info, patch_output_list):
    """Losslessly flush each storage block once with bounded memory."""
    if patch_output_list is None:
        return
    wsi_pred_map_ptr = zarr.open_array(wsi_pred_map_mmap_path, mode="r+")
    chunk_tl = chunk_info[1][0]
    storage_shape = np.asarray(wsi_pred_map_ptr.chunks[:2], dtype=np.int64)
    patches_by_storage_chunk = {}
    for pinfo in patch_output_list:
        pcoord, pdata = pinfo
        pdata = np.squeeze(pdata)
        pcoord = np.squeeze(pcoord)[:2]
        absolute_tl = chunk_tl + pcoord
        absolute_br = absolute_tl + np.asarray(pdata.shape[:2], dtype=np.int64)
        first_chunk = absolute_tl // storage_shape
        last_chunk = (absolute_br - 1) // storage_shape
        for row in range(int(first_chunk[0]), int(last_chunk[0]) + 1):
            for col in range(int(first_chunk[1]), int(last_chunk[1]) + 1):
                patches_by_storage_chunk.setdefault((row, col), []).append(
                    (absolute_tl, absolute_br, pdata)
                )
    array_shape = np.asarray(wsi_pred_map_ptr.shape[:2], dtype=np.int64)
    for storage_index, patches in patches_by_storage_chunk.items():
        storage_tl = np.asarray(storage_index, dtype=np.int64) * storage_shape
        storage_br = np.minimum(storage_tl + storage_shape, array_shape)
        storage_block = np.zeros(
            tuple(storage_br - storage_tl) + (wsi_pred_map_ptr.shape[-1],),
            dtype=np.float32,
        )
        for absolute_tl, absolute_br, pdata in patches:
            overlap_tl = np.maximum(storage_tl, absolute_tl)
            overlap_br = np.minimum(storage_br, absolute_br)
            dst_tl = overlap_tl - storage_tl
            dst_br = overlap_br - storage_tl
            src_tl = overlap_tl - absolute_tl
            src_br = overlap_br - absolute_tl
            storage_block[
                dst_tl[0] : dst_br[0], dst_tl[1] : dst_br[1]
            ] = pdata[src_tl[0] : src_br[0], src_tl[1] : src_br[1]]
        wsi_pred_map_ptr[
            storage_tl[0] : storage_br[0], storage_tl[1] : storage_br[1]
        ] = storage_block
    return
'''
    if source.count(old_assemble) != 1:
        raise RuntimeError("Unsupported HoVer-Net cache assembler for Zarr cache")
    source = source.replace(old_assemble, new_assemble)

    postproc_load = '    wsi_pred_map_ptr = np.load(pred_map_mmap_path, mmap_mode="r")'
    if source.count(postproc_load) != 1:
        raise RuntimeError("Unsupported HoVer-Net post-processing loader for Zarr cache")
    source = source.replace(
        postproc_load,
        '    wsi_pred_map_ptr = zarr.open_array(pred_map_mmap_path, mode="r")',
    )

    allocation = '''        self.wsi_inst_map = np.lib.format.open_memmap(
            "%s/pred_inst.npy" % self.cache_path,
            mode="w+",
            shape=tuple(self.wsi_proc_shape),
            dtype=np.int32,
        )
        # self.wsi_inst_map[:] = 0 # flush fill

        # warning, the value within this is uninitialized
        self.wsi_pred_map = np.lib.format.open_memmap(
            "%s/pred_map.npy" % self.cache_path,
            mode="w+",
            shape=tuple(self.wsi_proc_shape) + (out_ch,),
            dtype=np.float32,
        )'''
    compressed_allocation = '''        # Each storage block is assembled once, bounding prediction-map memory
        # to 16 MiB at float32x4 while avoiding repeated recompression.
        cache_chunks = tuple(
            max(1, min(1024, int(tile_shape[idx]), int(self.wsi_proc_shape[idx])))
            for idx in range(2)
        )
        cache_compressor = Blosc(
            cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE
        )
        self.wsi_inst_map = zarr.open_array(
            "%s/pred_inst.zarr" % self.cache_path,
            mode="w",
            shape=tuple(self.wsi_proc_shape),
            chunks=cache_chunks,
            dtype=np.int32,
            compressor=cache_compressor,
            fill_value=0,
        )
        self.wsi_pred_map = zarr.open_array(
            "%s/pred_map.zarr" % self.cache_path,
            mode="w",
            shape=tuple(self.wsi_proc_shape) + (out_ch,),
            chunks=cache_chunks + (out_ch,),
            dtype=np.float32,
            compressor=cache_compressor,
            fill_value=0.0,
        )'''
    if source.count(allocation) != 1:
        raise RuntimeError("Unsupported HoVer-Net WSI allocation for Zarr cache")
    source = source.replace(allocation, compressed_allocation)

    pred_path = 'wsi_pred_map_mmap_path = "%s/pred_map.npy" % self.cache_path'
    if source.count(pred_path) != 2:
        raise RuntimeError("Unsupported HoVer-Net prediction-map paths for Zarr cache")
    source = source.replace(
        pred_path,
        'wsi_pred_map_mmap_path = "%s/pred_map.zarr" % self.cache_path',
    )
    wsi_path.write_text(source)


def enable_cache_measurement_runtime(runtime_repo: Path) -> None:
    """Allow an explicit benchmark to inspect cache size before cleanup."""
    wsi_path = runtime_repo / "infer" / "wsi.py"
    source = wsi_path.read_text()
    cleanup = "        rm_n_mkdir(self.cache_path)  # clean up all cache"
    measured_cleanup = '''        if os.environ.get("HOVERNET_MEASURE_CACHE_STORAGE") != "1":
            rm_n_mkdir(self.cache_path)  # clean up all cache'''
    if source.count(cleanup) != 1:
        raise RuntimeError("Unsupported HoVer-Net cache cleanup for storage measurement")
    wsi_path.write_text(source.replace(cleanup, measured_cleanup))


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


def make_inference_mask(src: Path, dst: Path) -> tuple[int, int]:
    """Convert the shared clean-tissue mask to HoVer-Net's named PNG contract."""
    import pyvips

    mask = pyvips.Image.new_from_file(str(src), access="sequential")
    if mask.bands > 1:
        mask = mask[0]
    mask = (mask > 0).ifthenelse(255, 0).cast("uchar")
    mask.pngsave(str(dst), compression=6)
    return mask.width, mask.height


def inference_mask_args(mask_dir: Path | None) -> list[str]:
    if mask_dir is None:
        return []
    return [f"--input_mask_dir={mask_dir}", "--save_mask", "--save_thumb"]


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
    parser.add_argument("--cache-backend", choices=("numpy", "zarr"), default="zarr")
    parser.add_argument("--measure-cache-storage", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--clean-tissue-mask", default="")
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
    clean_tissue_mask = Path(args.clean_tissue_mask).resolve() if args.clean_tissue_mask else None
    inference_mask_dir = outdir / "input_mask"
    inference_mask_shape = None
    if clean_tissue_mask is not None:
        require_readable_file(clean_tissue_mask, "GrandQC clean-tissue mask")
        inference_mask_dir.mkdir(exist_ok=True)
        inference_mask_shape = make_inference_mask(
            clean_tissue_mask, inference_mask_dir / f"{normalized_slide.stem}.png",
        )
    # Upstream initializes debug.log in its current directory. The bundled
    # repository is read-only in Singularity, so always run an isolated copy.
    runtime_repo = prepare_runtime_repo(repo, outdir)
    if args.cache_backend == "zarr":
        if args.prediction_cache:
            raise ValueError("--prediction-cache currently requires --cache-backend numpy")
        enable_zarr_cache_runtime(runtime_repo)
    if args.prediction_cache:
        prediction_cache = Path(args.prediction_cache).resolve()
        enable_cache_resume_runtime(runtime_repo, cache_dir, prediction_cache)
    if args.measure_cache_storage:
        enable_cache_measurement_runtime(runtime_repo)
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
    cmd += inference_mask_args(inference_mask_dir if clean_tissue_mask is not None else None)
    env = os.environ.copy()
    runtime_cache_dir = outdir / "runtime_cache"
    (runtime_cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(runtime_cache_dir / "matplotlib")
    env["XDG_CACHE_HOME"] = str(runtime_cache_dir)
    if args.prediction_cache:
        env["HOVERNET_RESUME_PRED_MAP"] = "1"
    if args.measure_cache_storage:
        env["HOVERNET_MEASURE_CACHE_STORAGE"] = "1"
    subprocess.run(cmd, cwd=runtime_repo, env=env, check=True)
    transient_cache_storage = directory_storage(cache_dir) if args.measure_cache_storage else None
    candidates = sorted(raw_dir.rglob("*.json"))
    if not candidates:
        raise RuntimeError(f"HoVer-Net produced no instance JSON under {raw_dir}")
    raw_json = next((p for p in candidates if p.stem == normalized_slide.stem), candidates[0])
    checkpoint_sha256 = file_sha256(checkpoint)
    upstream_revision = git_revision(repo)
    metadata = {
        "source_mpp": source_mpp, "target_mpp": args.target_mpp, "coordinate_scale": scale,
        "source_width": width, "source_height": height, "batch_size": batch_size,
        "cache_backend": args.cache_backend,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "upstream_revision": upstream_revision,
        "model_provenance": {
            "source_repository": "https://github.com/vqdang/hover_net",
            "requested_revision": None,
            "resolved_revision": upstream_revision,
            "cache_path": str(checkpoint.parent),
            "checkpoints": [{
                "logical_name": "MoNuSAC",
                "cache_path": str(checkpoint),
                "filename": checkpoint.name,
                "size_bytes": int(checkpoint.stat().st_size),
                "sha256": checkpoint_sha256,
                "status": "verified",
            }],
        },
        "model_scope": MONUSAC_MODEL_SCOPE,
        "inference_tissue_mask": {
            "source": "grandqc_clean_tissue" if clean_tissue_mask is not None else "upstream_otsu",
            "path": str(clean_tissue_mask) if clean_tissue_mask is not None else None,
            "shape_xy": list(inference_mask_shape) if inference_mask_shape is not None else None,
        },
    }
    if transient_cache_storage is not None:
        metadata["transient_cache_storage"] = transient_cache_storage
    # Always emit one stable schema, even when no coordinate rescaling is needed.
    normalized_path = outdir / "hovernet_cells.json"
    normalize_output(raw_json, normalized_path, 1.0 / scale, metadata)
    if args.clean_tissue_mask:
        from grandqc_mask import filter_cell_payload
        payload, filter_summary = filter_cell_payload(
            json.loads(normalized_path.read_text()), args.clean_tissue_mask, shift,
        )
        normalized_path.write_text(json.dumps(payload))
        metadata["grandqc_filter"] = filter_summary
    (outdir / "hovernet_metadata.json").write_text(json.dumps(metadata, indent=2))
    shutil.rmtree(cache_dir, ignore_errors=True)
    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(inference_mask_dir, ignore_errors=True)
    shutil.rmtree(runtime_cache_dir, ignore_errors=True)
    if runtime_repo != repo:
        shutil.rmtree(runtime_repo, ignore_errors=True)


if __name__ == "__main__":
    main()
