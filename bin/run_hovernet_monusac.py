#!/usr/bin/env python3
"""Run the official PyTorch HoVer-Net MoNuSAC model on a StarDist crop."""

from __future__ import annotations

import argparse
import gzip
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
    read_bits = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    # os.access() returns true for uid 0 even when a file has no read bits.
    # Checking both makes the preflight deterministic in root-run containers
    # and still verifies the effective user's access on an HPC host.
    if not (path.stat().st_mode & read_bits) or not os.access(path, os.R_OK):
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


def enable_lightweight_tile_runtime(runtime_repo: Path) -> None:
    """Patch the pinned tile runner to retain JSON only, compressed per tile.

    The upstream tile route always writes an instance MAT file and RGB overlay
    for every tile.  CellPhenotyper consumes neither artifact.  Removing them
    avoids a second image-sized representation while retaining the official
    model and post-processing implementation.
    """
    tile_path = runtime_repo / "infer" / "tile.py"
    source = tile_path.read_text()
    source = source.replace('mat_dict.pop("inst_type", None) \n', 'mat_dict.pop("inst_type", None)\n')
    import_anchor = "import glob\n"
    if source.count(import_anchor) != 1:
        raise RuntimeError("Unsupported HoVer-Net tile imports")
    source = source.replace(import_anchor, import_anchor + "import gzip\n")

    directories = '''        rm_n_mkdir(self.output_dir + '/json/')
        rm_n_mkdir(self.output_dir + '/mat/')
        rm_n_mkdir(self.output_dir + '/overlay/')'''
    if source.count(directories) != 1:
        raise RuntimeError("Unsupported HoVer-Net tile output-directory layout")
    source = source.replace(directories, "        rm_n_mkdir(self.output_dir + '/json/')")

    bulky_outputs = '''            nuc_val_list = list(inst_info_dict.values())
            # need singleton to make matlab happy
            nuc_uid_list = np.array(list(inst_info_dict.keys()))[:,None]
            nuc_type_list = np.array([v["type"] for v in nuc_val_list])[:,None]
            nuc_coms_list = np.array([v["centroid"] for v in nuc_val_list])

            mat_dict = {
                "inst_map" : pred_inst,
                "inst_uid" : nuc_uid_list,
                "inst_type": nuc_type_list,
                "inst_centroid": nuc_coms_list
            }
            if self.nr_types is None: # matlab does not have None type array
                mat_dict.pop("inst_type", None)

            if self.save_raw_map:
                mat_dict["raw_map"] = pred_map
            save_path = "%s/mat/%s.mat" % (self.output_dir, img_name)
            sio.savemat(save_path, mat_dict)

            save_path = "%s/overlay/%s.png" % (self.output_dir, img_name)
            cv2.imwrite(save_path, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))
'''
    if source.count(bulky_outputs) != 1:
        raise RuntimeError("Unsupported HoVer-Net tile MAT/overlay writer")
    source = source.replace(bulky_outputs, "")

    json_writer = '''            save_path = "%s/json/%s.json" % (self.output_dir, img_name)
            self.__save_json(save_path, inst_info_dict, None)
            return img_name'''
    compressed_writer = '''            save_path = "%s/json/%s.json" % (self.output_dir, img_name)
            self.__save_json(save_path, inst_info_dict, None)
            with open(save_path, "rb") as source_handle:
                with gzip.open(save_path + ".gz", "wb", compresslevel=6) as target_handle:
                    while True:
                        block = source_handle.read(1024 * 1024)
                        if not block:
                            break
                        target_handle.write(block)
            os.unlink(save_path)
            return img_name'''
    if source.count(json_writer) != 1:
        raise RuntimeError("Unsupported HoVer-Net tile JSON writer")
    tile_path.write_text(source.replace(json_writer, compressed_writer))


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
    return [f"--input_mask_dir={mask_dir}"]


def _core_has_tissue(clean_mask, core: tuple[int, int, int, int], scale: float) -> bool:
    """Test a target-resolution core against the shared low-resolution mask."""
    x0, y0, x1, y1 = core
    sx0, sy0 = x0 / scale, y0 / scale
    sx1, sy1 = x1 / scale, y1 / scale
    mx0 = max(0, min(clean_mask.mask_width - 1, int(sx0 * clean_mask.mask_width / clean_mask.crop_width)))
    my0 = max(0, min(clean_mask.mask_height - 1, int(sy0 * clean_mask.mask_height / clean_mask.crop_height)))
    mx1 = max(mx0 + 1, min(clean_mask.mask_width, int((sx1 * clean_mask.mask_width / clean_mask.crop_width) + 1)))
    my1 = max(my0 + 1, min(clean_mask.mask_height, int((sy1 * clean_mask.mask_height / clean_mask.crop_height) + 1)))
    return bool(clean_mask.mask[my0:my1, mx0:mx1].any())


def plan_streaming_tiles(
    src: Path,
    scale: float,
    *,
    core_size: int,
    halo: int,
    clean_mask=None,
) -> tuple[tuple[int, int], tuple[int, int], list[dict]]:
    """Plan overlapping target-MPP tiles without decoding the complete WSI."""
    import pyvips

    if core_size < 512 or halo < 92 or core_size + (2 * halo) > 5000:
        raise ValueError("HoVer-Net streaming tiles require core>=512, halo>=92, and core+2*halo<=5000")
    image = pyvips.Image.new_from_file(str(src), access="random")
    if image.bands > 3:
        image = image[:3]
    if image.bands == 1:
        image = image.bandjoin([image, image])
    original = (int(image.width), int(image.height))
    if abs(scale - 1.0) > 1e-6:
        image = image.resize(scale, kernel="lanczos3")
    target = (int(image.width), int(image.height))
    records: list[dict] = []
    index = 0
    for core_y0 in range(0, target[1], core_size):
        core_y1 = min(target[1], core_y0 + core_size)
        for core_x0 in range(0, target[0], core_size):
            core_x1 = min(target[0], core_x0 + core_size)
            core = (core_x0, core_y0, core_x1, core_y1)
            if clean_mask is not None and not _core_has_tissue(clean_mask, core, scale):
                continue
            tile_x0, tile_y0 = max(0, core_x0 - halo), max(0, core_y0 - halo)
            tile_x1, tile_y1 = min(target[0], core_x1 + halo), min(target[1], core_y1 + halo)
            name = f"tile_{index:07d}"
            records.append({
                "name": name,
                "tile_origin_xy": [tile_x0, tile_y0],
                "core_xyxy": list(core),
                "tile_shape_xy": [tile_x1 - tile_x0, tile_y1 - tile_y0],
            })
            index += 1
    if not records:
        raise RuntimeError("GrandQC clean-tissue support selected no HoVer-Net streaming tiles")
    return original, target, records


def write_streaming_tile_batch(
    src: Path, input_dir: Path, scale: float, records: list[dict], jpeg_quality: int,
) -> None:
    """Decode only one bounded batch of planned tiles."""
    import pyvips

    if not 1 <= jpeg_quality <= 100:
        raise ValueError("HoVer-Net tile JPEG quality must be within 1..100")
    shutil.rmtree(input_dir, ignore_errors=True)
    input_dir.mkdir(parents=True)
    image = pyvips.Image.new_from_file(str(src), access="random")
    if image.bands > 3:
        image = image[:3]
    if image.bands == 1:
        image = image.bandjoin([image, image])
    if abs(scale - 1.0) > 1e-6:
        image = image.resize(scale, kernel="lanczos3")
    for record in records:
        x0, y0 = record["tile_origin_xy"]
        width, height = record["tile_shape_xy"]
        tile = image.crop(x0, y0, width, height)
        tile.jpegsave(
            str(input_dir / f"{record['name']}.jpg"),
            Q=jpeg_quality, optimize_coding=True, strip=True, subsample_mode="off",
        )


def _read_json(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def append_streaming_cells(
    records: list[dict], raw_dir: Path, handle, inverse_scale: float, *,
    clean_mask, include_contours: bool, tile_index_offset: int = 0, first: bool = True,
) -> tuple[int, int, bool]:
    """Append one inference batch and delete its raw tile records immediately."""
    input_cells = retained_cells = 0
    for local_tile_index, record in enumerate(records):
        raw_path = raw_dir / "json" / f"{record['name']}.json.gz"
        if not raw_path.is_file():
            raw_path = raw_dir / f"{record['name']}.json.gz"
        if not raw_path.is_file():
            raise RuntimeError(f"HoVer-Net produced no JSON for streaming tile: {record['name']}")
        raw = _read_json(raw_path)
        instances = raw.get("nuclei", raw.get("nuc", raw)) if isinstance(raw, dict) else raw
        iterable = instances.items() if isinstance(instances, dict) else enumerate(instances)
        ox, oy = record["tile_origin_xy"]
        cx0, cy0, cx1, cy1 = record["core_xyxy"]
        tile_index = tile_index_offset + local_tile_index
        for raw_id, item in iterable:
            if not isinstance(item, dict) or "centroid" not in item:
                continue
            input_cells += 1
            local_x, local_y = map(float, item["centroid"][:2])
            target_x, target_y = local_x + ox, local_y + oy
            if not (cx0 <= target_x < cx1 and cy0 <= target_y < cy1):
                continue
            source_x, source_y = target_x * inverse_scale, target_y * inverse_scale
            if clean_mask is not None and not clean_mask.contains_xy(source_x, source_y):
                continue
            type_id = item.get("type")
            cell = {
                "id": f"{tile_index}:{raw_id}",
                "centroid": [source_x, source_y],
                "type_id": type_id,
                "type": monusac_type_name(type_id),
                "type_prob": item.get("type_prob"),
            }
            if include_contours:
                cell["contour"] = [
                    [(float(point[0]) + ox) * inverse_scale,
                     (float(point[1]) + oy) * inverse_scale]
                    for point in item.get("contour", [])
                ]
            if not first:
                handle.write(",")
            handle.write(json.dumps(cell, separators=(",", ":")))
            first = False
            retained_cells += 1
        raw_path.unlink()
    return input_cells, retained_cells, first


def normalize_streaming_tiles(
    records: list[dict],
    raw_dir: Path,
    output_path: Path,
    inverse_scale: float,
    metadata: dict,
    *,
    clean_mask,
    include_contours: bool,
) -> dict:
    """Merge tile JSON incrementally while enforcing unique core ownership."""
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        prefix = {
            "model": "HoVer-Net", "checkpoint": "MoNuSAC", "metadata": metadata,
            "type_map": {key: value[0] for key, value in MONUSAC_TYPE_INFO.items()},
        }
        handle.write(json.dumps(prefix, separators=(",", ":"))[:-1])
        handle.write(',"cells":[')
        input_cells, retained_cells, _ = append_streaming_cells(
            records, raw_dir, handle, inverse_scale,
            clean_mask=clean_mask, include_contours=include_contours,
        )
        summary = {
            "input_tile_cells_including_halo_duplicates": int(input_cells),
            "retained_unique_core_cells": int(retained_cells),
            "removed_halo_or_outside_grandqc": int(input_cells - retained_cells),
        }
        handle.write('],"pipeline_metadata":{"grandqc_filter":')
        handle.write(json.dumps(summary, separators=(",", ":")))
        handle.write("}}")
    return summary


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
    parser.add_argument("--execution-mode", choices=("streaming_tiles", "wsi"), default="streaming_tiles")
    parser.add_argument("--stream-core-size", type=int, default=4096)
    parser.add_argument("--stream-halo", type=int, default=256)
    parser.add_argument("--stream-batch-tiles", type=int, default=64)
    parser.add_argument("--tile-jpeg-quality", type=int, default=92)
    parser.add_argument("--export-contours", action="store_true")
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
    clean_tissue_mask = Path(args.clean_tissue_mask).resolve() if args.clean_tissue_mask else None
    inference_mask_dir = outdir / "input_mask"
    inference_mask_shape = None
    if clean_tissue_mask is not None:
        require_readable_file(clean_tissue_mask, "GrandQC clean-tissue mask")
    clean_mask_reader = None
    if clean_tissue_mask is not None:
        from grandqc_mask import CropCleanTissueMask
        clean_mask_reader = CropCleanTissueMask(clean_tissue_mask, shift)
    # Upstream initializes debug.log in its current directory. The bundled
    # repository is read-only in Singularity, so always run an isolated copy.
    runtime_repo = prepare_runtime_repo(repo, outdir)
    tile_records = None
    target_shape = None
    if args.execution_mode == "streaming_tiles":
        if args.prediction_cache:
            raise ValueError("--prediction-cache requires --execution-mode wsi")
        if args.measure_cache_storage:
            raise ValueError("--measure-cache-storage is only valid with --execution-mode wsi")
        enable_lightweight_tile_runtime(runtime_repo)
        if args.stream_batch_tiles < 1:
            raise ValueError("--stream-batch-tiles must be positive")
        (width, height), target_shape, tile_records = plan_streaming_tiles(
            image, scale,
            core_size=args.stream_core_size, halo=args.stream_halo,
            clean_mask=clean_mask_reader,
        )
    else:
        normalized_slide = input_dir / "hovernet_input.tif"
        width, height = make_pyramidal_input(image, normalized_slide, scale, args.target_mpp)
        if clean_tissue_mask is not None:
            inference_mask_dir.mkdir(exist_ok=True)
            inference_mask_shape = make_inference_mask(
                clean_tissue_mask, inference_mask_dir / f"{normalized_slide.stem}.png",
            )
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
    base_cmd = [
        "python", "-c", compatibility_launcher, str(runtime_repo / "run_infer.py"),
        f"--gpu={args.gpu}", "--nr_types=5",
        f"--type_info_path={type_info_path}",
        f"--model_path={checkpoint}", "--model_mode=fast",
        f"--nr_inference_workers={args.inference_workers}",
        f"--nr_post_proc_workers={args.postproc_workers}", f"--batch_size={batch_size}",
    ]
    env = os.environ.copy()
    runtime_cache_dir = outdir / "runtime_cache"
    (runtime_cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(runtime_cache_dir / "matplotlib")
    env["XDG_CACHE_HOME"] = str(runtime_cache_dir)
    if args.prediction_cache:
        env["HOVERNET_RESUME_PRED_MAP"] = "1"
    if args.measure_cache_storage:
        env["HOVERNET_MEASURE_CACHE_STORAGE"] = "1"
    checkpoint_sha256 = file_sha256(checkpoint)
    upstream_revision = git_revision(repo)
    metadata = {
        "source_mpp": source_mpp, "target_mpp": args.target_mpp, "coordinate_scale": scale,
        "source_width": width, "source_height": height, "batch_size": batch_size,
        "execution_mode": args.execution_mode,
        "cache_backend": args.cache_backend if args.execution_mode == "wsi" else None,
        "streaming_tiles": ({
            "core_size_px_at_target_mpp": args.stream_core_size,
            "halo_px_at_target_mpp": args.stream_halo,
            "batch_tiles": args.stream_batch_tiles,
            "tile_count": len(tile_records),
            "target_shape_xy": list(target_shape),
            "tile_jpeg_quality": args.tile_jpeg_quality,
            "unique_ownership_rule": "centroid_in_half_open_core",
            "contours_retained": bool(args.export_contours),
        } if tile_records is not None else None),
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
    normalized_path = outdir / "hovernet_cells.json.gz"
    if args.execution_mode == "streaming_tiles":
        total_input_cells = total_retained_cells = 0
        first = True
        with gzip.open(normalized_path, "wt", encoding="utf-8", compresslevel=6) as handle:
            prefix = {
                "model": "HoVer-Net", "checkpoint": "MoNuSAC", "metadata": metadata,
                "type_map": {key: value[0] for key, value in MONUSAC_TYPE_INFO.items()},
            }
            handle.write(json.dumps(prefix, separators=(",", ":"))[:-1])
            handle.write(',"cells":[')
            for offset in range(0, len(tile_records), args.stream_batch_tiles):
                record_batch = tile_records[offset:offset + args.stream_batch_tiles]
                write_streaming_tile_batch(image, input_dir, scale, record_batch, args.tile_jpeg_quality)
                shutil.rmtree(raw_dir, ignore_errors=True)
                raw_dir.mkdir()
                cmd = base_cmd + [
                    "tile", f"--input_dir={input_dir}", f"--output_dir={raw_dir}", "--mem_usage=0.10",
                ]
                subprocess.run(cmd, cwd=runtime_repo, env=env, check=True)
                input_count, retained_count, first = append_streaming_cells(
                    record_batch, raw_dir, handle, 1.0 / scale,
                    clean_mask=clean_mask_reader, include_contours=args.export_contours,
                    tile_index_offset=offset, first=first,
                )
                total_input_cells += input_count
                total_retained_cells += retained_count
                # No tile input or raw per-tile cell record survives its batch.
                shutil.rmtree(input_dir, ignore_errors=True)
                shutil.rmtree(raw_dir, ignore_errors=True)
            filter_summary = {
                "input_tile_cells_including_halo_duplicates": int(total_input_cells),
                "retained_unique_core_cells": int(total_retained_cells),
                "removed_halo_or_outside_grandqc": int(total_input_cells - total_retained_cells),
            }
            handle.write('],"pipeline_metadata":{"grandqc_filter":')
            handle.write(json.dumps(filter_summary, separators=(",", ":")))
            handle.write("}}")
        metadata["grandqc_filter"] = filter_summary
    else:
        cmd = base_cmd + [
            "wsi", f"--input_dir={input_dir}", f"--output_dir={raw_dir}", f"--cache_path={cache_dir}",
            "--proc_mag=40", f"--chunk_shape={args.chunk_shape}", f"--tile_shape={args.tile_shape}",
        ]
        cmd += inference_mask_args(inference_mask_dir if clean_tissue_mask is not None else None)
        subprocess.run(cmd, cwd=runtime_repo, env=env, check=True)
        transient_cache_storage = directory_storage(cache_dir) if args.measure_cache_storage else None
        if transient_cache_storage is not None:
            metadata["transient_cache_storage"] = transient_cache_storage
        candidates = sorted(raw_dir.rglob("*.json"))
        if not candidates:
            raise RuntimeError(f"HoVer-Net produced no instance JSON under {raw_dir}")
        raw_json = next((p for p in candidates if p.stem == normalized_slide.stem), candidates[0])
        # Compatibility route: retain the historical in-memory normalization,
        # then gzip the sole consumed artifact and remove the raw duplicate.
        temporary_path = outdir / "hovernet_cells.json"
        normalize_output(raw_json, temporary_path, 1.0 / scale, metadata)
        payload = json.loads(temporary_path.read_text())
        if not args.export_contours:
            for cell in payload.get("cells", []):
                cell.pop("contour", None)
        if args.clean_tissue_mask:
            from grandqc_mask import filter_cell_payload
            payload, filter_summary = filter_cell_payload(payload, args.clean_tissue_mask, shift)
            metadata["grandqc_filter"] = filter_summary
        with gzip.open(normalized_path, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump(payload, handle, separators=(",", ":"))
        temporary_path.unlink()
    (outdir / "hovernet_metadata.json").write_text(json.dumps(metadata, indent=2))
    shutil.rmtree(raw_dir, ignore_errors=True)
    shutil.rmtree(cache_dir, ignore_errors=True)
    shutil.rmtree(input_dir, ignore_errors=True)
    shutil.rmtree(inference_mask_dir, ignore_errors=True)
    shutil.rmtree(runtime_cache_dir, ignore_errors=True)
    if runtime_repo != repo:
        shutil.rmtree(runtime_repo, ignore_errors=True)


if __name__ == "__main__":
    main()
