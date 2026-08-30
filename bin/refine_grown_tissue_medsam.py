#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tifffile import imwrite
import tifffile

from ome_tiff_metadata import (
    create_tiff_memmap,
    label_storage_dtype,
    read_mpp_json,
    tiff_resolution_kwargs,
    validate_ome_tiff,
)

from grow_to_tissue import (
    DEFAULT_PALETTE,
    ensure_2d,
    load_mask_2d,
    load_tiff,
    pyramidize_with_raw2ometiff,
    save_preview_png,
)
from grow_to_tissue_core import compute_boundary, to_float_rgb
from medsam_border_refine import DEFAULT_MEDSAM_CHECKPOINT, MedSAMConfig, MedSAMUnavailableError, run_medsam_border_refine


def sample_prefix(sample_id: str, out_path: str | Path) -> str:
    out_name = Path(out_path).name
    if out_name.endswith(".ome.tif"):
        out_name = out_name[:-8]
    if out_name.endswith(".tif"):
        out_name = out_name[:-4]
    return sample_id or out_name


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    return (to_float_rgb(image) * 255.0).astype(np.uint8)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    out = np.zeros(labels.shape + (3,), dtype=np.uint8)
    unique = np.unique(labels)
    unique = unique[unique > 0]
    for idx, lid in enumerate(unique):
        out[labels == lid] = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
    return out


def label_boundaries(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    b = np.zeros(labels.shape, dtype=bool)
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    b &= labels > 0
    return b


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    rgb = to_uint8_rgb(image).astype(np.float32)
    mask = np.asarray(mask).astype(bool)
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def overlay_labels(image: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = to_uint8_rgb(image).astype(np.float32)
    labels = np.asarray(labels).astype(np.int32)
    colors = colorize_labels(labels).astype(np.float32)
    mask = labels > 0
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * colors[mask]
    boundaries = label_boundaries(labels)
    rgb[boundaries] = colors[boundaries]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def boundary_compare(image: np.ndarray, raw_mask: np.ndarray, final_mask: np.ndarray) -> np.ndarray:
    rgb = to_uint8_rgb(image).copy()
    b0 = compute_boundary(raw_mask)
    b1 = compute_boundary(final_mask)
    rgb[b0 & ~b1] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[b1 & ~b0] = np.array([0, 255, 255], dtype=np.uint8)
    rgb[b0 & b1] = np.array([255, 255, 0], dtype=np.uint8)
    return rgb


def heatmap(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob).astype(np.float32), 0.0, 1.0)
    out = np.zeros(p.shape + (3,), dtype=np.uint8)
    out[..., 0] = np.clip(255.0 * p, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(255.0 * (1.0 - np.abs(p - 0.5) * 2.0), 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(255.0 * (1.0 - p), 0, 255).astype(np.uint8)
    return out


def fit_panel(arr: np.ndarray, width: int = 420) -> Image.Image:
    img = Image.fromarray(arr)
    h = int(round(img.height * (width / img.width)))
    return img.resize((width, h), Image.Resampling.BILINEAR)


def make_panel(items: List[Tuple[str, np.ndarray]], out_path: Path, columns: int = 3) -> None:
    font = ImageFont.load_default()
    panels = [(title, fit_panel(arr)) for title, arr in items]
    cell_w = max(img.width for _, img in panels)
    cell_h = max(img.height for _, img in panels)
    title_h = 22
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, rows * (cell_h + title_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, img) in enumerate(panels):
        r = idx // columns
        c = idx % columns
        x = c * cell_w
        y = r * (cell_h + title_h)
        canvas.paste(img, (x, y + title_h))
        draw.text((x + 6, y + 4), title, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_native_panel(items: List[Tuple[str, np.ndarray]], out_path: Path, columns: int = 2) -> None:
    """Create a QC panel without resizing the image crops."""
    font = ImageFont.load_default()
    panels = [(title, Image.fromarray(np.asarray(arr))) for title, arr in items]
    cell_w = max(img.width for _, img in panels)
    cell_h = max(img.height for _, img in panels)
    title_h = 24
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, rows * (cell_h + title_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, img) in enumerate(panels):
        r = idx // columns
        c = idx % columns
        x = c * cell_w
        y = r * (cell_h + title_h)
        canvas.paste(img, (x, y + title_h))
        draw.text((x + 6, y + 5), title, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def preview_step_for_shape(shape: Tuple[int, int], max_side: int = 2048) -> int:
    h, w = int(shape[0]), int(shape[1])
    return max(1, int(np.ceil(max(h, w) / float(max_side))))


def preview_subsample(arr: np.ndarray, step: int) -> np.ndarray:
    step = max(1, int(step))
    if step == 1:
        return np.asarray(arr)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[::step, ::step]
    return arr[::step, ::step, ...]


def load_tiff_memmap(path: str, name: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    try:
        return tifffile.memmap(path)
    except Exception:
        return load_tiff(path, name)


def first_zarr_array(root):
    if hasattr(root, "shape"):
        return root
    names = list(root.array_keys())
    if "0" in names:
        return root["0"]
    if names:
        return root[names[0]]
    for _, obj in root.items():
        if hasattr(obj, "shape"):
            return obj
    raise ValueError("TIFF zarr store did not expose an array.")


class TiffWindowReader:
    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name
        self.kind = "memmap"
        self._tf = None
        self._store = None
        self._arr = None
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
        try:
            self._arr = tifffile.memmap(path)
        except Exception:
            try:
                import zarr
            except Exception as exc:
                raise RuntimeError(f"{name} is not memmappable and zarr is unavailable: {path}") from exc
            self.kind = "zarr"
            self._tf = tifffile.TiffFile(path)
            self._store = self._tf.series[0].aszarr()
            self._arr = first_zarr_array(zarr.open(self._store, mode="r"))
        self.shape = tuple(int(x) for x in self._arr.shape)
        self.dtype = np.dtype(self._arr.dtype)

    def spatial_shape(self) -> Tuple[int, int]:
        if len(self.shape) == 2:
            return int(self.shape[0]), int(self.shape[1])
        if len(self.shape) == 3 and self.shape[-1] in (1, 3, 4):
            return int(self.shape[0]), int(self.shape[1])
        if len(self.shape) == 3 and self.shape[0] in (1, 3, 4):
            return int(self.shape[1]), int(self.shape[2])
        raise ValueError(f"{self.name} has unsupported TIFF shape: {self.shape}")

    def read(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        y0, y1, x0, x1 = map(int, (y0, y1, x0, x1))
        if len(self.shape) == 2:
            return np.asarray(self._arr[y0:y1, x0:x1])
        if len(self.shape) == 3 and self.shape[-1] in (1, 3, 4):
            return np.asarray(self._arr[y0:y1, x0:x1, ...])
        if len(self.shape) == 3 and self.shape[0] in (1, 3, 4):
            return np.asarray(self._arr[:, y0:y1, x0:x1]).transpose(1, 2, 0)
        raise ValueError(f"{self.name} has unsupported TIFF shape: {self.shape}")

    def read_stride(self, step: int) -> np.ndarray:
        step = max(1, int(step))
        if len(self.shape) == 2:
            return np.asarray(self._arr[::step, ::step])
        if len(self.shape) == 3 and self.shape[-1] in (1, 3, 4):
            return np.asarray(self._arr[::step, ::step, ...])
        if len(self.shape) == 3 and self.shape[0] in (1, 3, 4):
            return np.asarray(self._arr[:, ::step, ::step]).transpose(1, 2, 0)
        raise ValueError(f"{self.name} has unsupported TIFF shape: {self.shape}")

    def close(self) -> None:
        try:
            if self._store is not None and hasattr(self._store, "close"):
                self._store.close()
        finally:
            if self._tf is not None:
                self._tf.close()


def tiff_spatial_shape(path: str, name: str) -> Tuple[int, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    with tifffile.TiffFile(path) as tf:
        shape = tuple(int(x) for x in tf.series[0].shape)
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 3 and shape[-1] in (1, 3, 4):
        return shape[0], shape[1]
    if len(shape) == 3 and shape[0] in (1, 3, 4):
        return shape[1], shape[2]
    raise ValueError(f"{name} has unsupported TIFF shape: {shape}")


def tile_starts(n: int, tile_size: int, overlap: int) -> List[int]:
    n = int(n)
    tile_size = max(1, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size - 1))
    if n <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, n, stride))
    last = max(0, n - tile_size)
    starts = [s for s in starts if s <= last]
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def commit_bounds(y0: int, y1: int, x0: int, x1: int, shape: Tuple[int, int], overlap: int) -> Tuple[int, int, int, int]:
    h, w = int(shape[0]), int(shape[1])
    margin = max(0, int(overlap) // 2)
    cy0 = y0 if y0 <= 0 else min(y1, y0 + margin)
    cy1 = y1 if y1 >= h else max(y0, y1 - margin)
    cx0 = x0 if x0 <= 0 else min(x1, x0 + margin)
    cx1 = x1 if x1 >= w else max(x0, x1 - margin)
    return int(cy0), int(cy1), int(cx0), int(cx1)


def make_medsam_config(args) -> MedSAMConfig:
    return MedSAMConfig(
        checkpoint=str(args.medsam_checkpoint),
        device=str(args.medsam_device),
        bbox_margin=int(args.medsam_bbox_margin),
        component_min_area=int(args.medsam_component_min_area),
        component_merge_distance=int(args.medsam_component_merge_distance),
        seed_dilation_radius=int(args.medsam_seed_dilation_radius),
        core_erosion_radius=int(args.medsam_core_erosion_radius),
        outer_dilation_radius=int(args.medsam_outer_dilation_radius),
        min_object_size=int(args.medsam_min_object_size),
        smooth_radius=int(args.medsam_smooth_radius),
        force_core_preservation=bool(args.medsam_force_core_preservation),
        save_debug=bool(args.medsam_save_debug),
        cluster_tile_size=int(args.medsam_cluster_tile_size),
        cluster_tile_overlap=int(args.medsam_cluster_tile_overlap),
    )


def pyramidize_or_fallback(tmp_flat: Path, out_path: str, args) -> None:
    try:
        pyramidize_with_raw2ometiff(
            in_tif=str(tmp_flat),
            out_ome_tif=str(out_path),
            compression=args.pyr_compression,
            max_workers=args.max_workers,
            downsample=args.downsample,
            overwrite=args.overwrite,
            keep_tmp=args.keep_tmp,
            legacy=args.legacy,
        )
        ome_summary = validate_ome_tiff(
            out_path,
            expected_shape=args.output_spatial_shape,
            expected_mpp=(args.source_mpp_x, args.source_mpp_y),
        )
        print(f"[INFO] Validated MedSAM OME-TIFF: {json.dumps(ome_summary, sort_keys=True)}")
    except Exception:
        out_file = Path(out_path)
        if out_file.exists():
            out_file.unlink()
        raise


def count_nonzero_2d_blocks(arr: np.ndarray, block_rows: int = 512) -> int:
    h = int(arr.shape[0])
    block_rows = max(1, int(block_rows))
    total = 0
    for y0 in range(0, h, block_rows):
        y1 = min(h, y0 + block_rows)
        total += int(np.count_nonzero(np.asarray(arr[y0:y1, :])))
    return total


def read_stride_tiled(reader: TiffWindowReader, step: int, tile_size: int = 4096) -> np.ndarray:
    """Downsample a WSI reader without forcing zarr/tifffile to stride over the full image."""
    step = max(1, int(step))
    h, w = reader.spatial_shape()
    if step == 1:
        return reader.read(0, h, 0, w)
    tile_size = max(step, int(tile_size))
    out_h = (h + step - 1) // step
    out_w = (w + step - 1) // step
    sample = reader.read(0, min(h, 1), 0, min(w, 1))
    if sample.ndim == 2:
        out = np.zeros((out_h, out_w), dtype=sample.dtype)
    else:
        out = np.zeros((out_h, out_w) + sample.shape[2:], dtype=sample.dtype)
    del sample
    for y0 in range(0, h, tile_size):
        y1 = min(h, y0 + tile_size)
        sy0 = ((y0 + step - 1) // step) * step
        if sy0 >= y1:
            continue
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0 + tile_size)
            sx0 = ((x0 + step - 1) // step) * step
            if sx0 >= x1:
                continue
            block = reader.read(sy0, y1, sx0, x1)
            small = block[::step, ::step] if block.ndim == 2 else block[::step, ::step, ...]
            oy0 = sy0 // step
            ox0 = sx0 // step
            out[oy0:oy0 + small.shape[0], ox0:ox0 + small.shape[1], ...] = small
            del block, small
        gc.collect()
    return out


def array_stride_tiled(arr: np.ndarray, step: int, tile_size: int = 4096) -> np.ndarray:
    step = max(1, int(step))
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if step == 1:
        return np.asarray(arr)
    tile_size = max(step, int(tile_size))
    out = np.zeros(((h + step - 1) // step, (w + step - 1) // step), dtype=arr.dtype)
    for y0 in range(0, h, tile_size):
        y1 = min(h, y0 + tile_size)
        sy0 = ((y0 + step - 1) // step) * step
        if sy0 >= y1:
            continue
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0 + tile_size)
            sx0 = ((x0 + step - 1) // step) * step
            if sx0 >= x1:
                continue
            small = np.asarray(arr[sy0:y1:step, sx0:x1:step])
            oy0 = sy0 // step
            ox0 = sx0 // step
            out[oy0:oy0 + small.shape[0], ox0:ox0 + small.shape[1]] = small
            del small
        gc.collect()
    return out


def choose_random_tissue_crop(
    raw_preview: np.ndarray,
    refined_preview: np.ndarray,
    step: int,
    full_shape: Tuple[int, int],
    crop_size: int,
    seed: int,
) -> Tuple[int, int, int, int]:
    h, w = int(full_shape[0]), int(full_shape[1])
    crop_size = max(1, min(int(crop_size), h, w))
    tissue_preview = (np.asarray(raw_preview) > 0) | (np.asarray(refined_preview) > 0)
    coords = np.argwhere(tissue_preview)
    rng = np.random.default_rng(int(seed))
    if coords.size:
        py, px = coords[int(rng.integers(coords.shape[0]))]
        center_y = int(py) * int(step) + int(step) // 2
        center_x = int(px) * int(step) + int(step) // 2
    else:
        center_y = h // 2
        center_x = w // 2
    y0 = max(0, min(h - crop_size, center_y - crop_size // 2))
    x0 = max(0, min(w - crop_size, center_x - crop_size // 2))
    return int(y0), int(y0 + crop_size), int(x0), int(x0 + crop_size)


def medsam_change_map(image: np.ndarray, raw_labels: np.ndarray, refined_labels: np.ndarray) -> np.ndarray:
    rgb = to_uint8_rgb(image).astype(np.float32)
    raw = np.asarray(raw_labels)
    refined = np.asarray(refined_labels)
    raw_fg = raw > 0
    refined_fg = refined > 0
    raw_only = raw_fg & ~refined_fg
    final_only = refined_fg & ~raw_fg
    relabeled = (raw != refined) & raw_fg & refined_fg
    rgb[raw_only] = (0.50 * rgb[raw_only]) + 0.50 * np.array([255, 0, 0], dtype=np.float32)
    rgb[final_only] = (0.50 * rgb[final_only]) + 0.50 * np.array([0, 255, 255], dtype=np.float32)
    rgb[relabeled] = (0.45 * rgb[relabeled]) + 0.55 * np.array([255, 255, 0], dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def crop_change_summary(raw_labels: np.ndarray, refined_labels: np.ndarray) -> Dict[str, int]:
    raw = np.asarray(raw_labels)
    refined = np.asarray(refined_labels)
    raw_fg = raw > 0
    refined_fg = refined > 0
    return {
        "raw_pixels": int(np.count_nonzero(raw_fg)),
        "final_pixels": int(np.count_nonzero(refined_fg)),
        "removed_pixels": int(np.count_nonzero(raw_fg & ~refined_fg)),
        "added_pixels": int(np.count_nonzero(refined_fg & ~raw_fg)),
        "relabel_pixels": int(np.count_nonzero((raw != refined) & raw_fg & refined_fg)),
    }


def save_fullres_qc_crop(
    image_crop: np.ndarray,
    raw_crop: np.ndarray,
    refined_crop: np.ndarray,
    out_path: Path,
    crop_bounds: Tuple[int, int, int, int],
) -> Dict[str, object]:
    raw_crop = ensure_2d(np.asarray(raw_crop), "raw MedSAM QC crop")
    refined_crop = ensure_2d(np.asarray(refined_crop), "refined MedSAM QC crop")
    make_native_panel(
        [
            ("Original crop (native resolution)", to_uint8_rgb(image_crop)),
            ("Before MedSAM", overlay_labels(image_crop, raw_crop)),
            ("After MedSAM", overlay_labels(image_crop, refined_crop)),
            ("Changes: red removed, cyan added, yellow relabeled", medsam_change_map(image_crop, raw_crop, refined_crop)),
        ],
        out_path,
        columns=2,
    )
    y0, y1, x0, x1 = crop_bounds
    meta: Dict[str, object] = {
        "path": str(out_path),
        "y0": int(y0),
        "y1": int(y1),
        "x0": int(x0),
        "x1": int(x1),
        "height": int(y1 - y0),
        "width": int(x1 - x0),
    }
    meta.update(crop_change_summary(raw_crop, refined_crop))
    return meta


def write_stream_progress(path: Path | None, payload: Dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def run_large_image_streaming_medsam(args, med_cfg: MedSAMConfig) -> None:
    start = time.perf_counter()
    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = sample_prefix(args.sample_id, args.out)

    image_reader = TiffWindowReader(args.image, "image")
    seed_reader = TiffWindowReader(args.seed_mask, "seed mask")
    grown_reader = TiffWindowReader(args.grown_mask, "grown mask")
    try:
        shape = seed_reader.spatial_shape()
        if grown_reader.spatial_shape() != shape:
            raise ValueError(f"Step-15 and step-16 masks must share the same shape: {shape} vs {grown_reader.spatial_shape()}")
        if image_reader.spatial_shape() != shape:
            raise ValueError(f"Image and masks must share the same spatial shape: {image_reader.spatial_shape()} vs {shape}")

        label_dtype = seed_reader.dtype if np.issubdtype(seed_reader.dtype, np.integer) else np.uint16
        if np.dtype(label_dtype).itemsize > np.dtype(grown_reader.dtype).itemsize and np.issubdtype(grown_reader.dtype, np.integer):
            label_dtype = grown_reader.dtype
        if np.dtype(label_dtype).itemsize < 2:
            label_dtype = np.uint16
        label_dtype = np.dtype(label_dtype).newbyteorder("=")
        storage_dtype = label_dtype.newbyteorder(">")

        h, w = shape
        tile_size = max(512, int(args.medsam_cluster_tile_size))
        overlap = max(0, min(int(args.medsam_cluster_tile_overlap), tile_size - 1))
        block_rows = max(1, int(args.stream_block_rows))
        resume_tmp = Path(args.stream_resume_tmp).expanduser() if args.stream_resume_tmp else None
        resume_tiles = max(0, int(args.stream_resume_tiles))
        resume_grid_y = max(0, int(args.stream_resume_grid_y))
        resume_grid_x = max(0, int(args.stream_resume_grid_x))
        resume_by_grid = resume_grid_y > 0 and resume_grid_x > 0
        progress_path = Path(args.stream_progress_json).expanduser() if args.stream_progress_json else outdir / f"{prefix}_medsam_stream_progress.json"
        if resume_tmp is not None:
            if not resume_tmp.exists():
                raise FileNotFoundError(f"--stream-resume-tmp not found: {resume_tmp}")
            tmp_flat = resume_tmp
            refined_out = tifffile.memmap(tmp_flat)
            if tuple(refined_out.shape) != (h, w):
                raise ValueError(f"--stream-resume-tmp shape mismatch: {tuple(refined_out.shape)} vs {(h, w)}")
            storage_dtype = np.dtype(refined_out.dtype)
            label_dtype = storage_dtype.newbyteorder("=")
            print(
                f"[INFO] Resuming streaming MedSAM from checkpoint={tmp_flat} "
                f"skip_completed_foreground_tiles={resume_tiles}"
                + (
                    f" resume_after_grid=({resume_grid_y},{resume_grid_x})"
                    if resume_by_grid
                    else ""
                ),
                flush=True,
            )
        else:
            tmp_flat = outdir / f".tmp_refined_stream_{os.getpid()}_{int(time.time())}.tif"
            if tmp_flat.exists():
                tmp_flat.unlink()
            refined_out = create_tiff_memmap(
                tmp_flat,
                shape=(h, w),
                dtype=storage_dtype,
                mpp_x=args.source_mpp_x,
                mpp_y=args.source_mpp_y,
            )

        raw_pixels = 0
        print(
            f"[INFO] Large-image streaming MedSAM enabled: shape={shape}, tile_size={tile_size}, "
            f"overlap={overlap}, image_reader={image_reader.kind}, grown_reader={grown_reader.kind}, device={med_cfg.device}",
            flush=True,
        )
        for y0 in range(0, h, block_rows):
            y1 = min(h, y0 + block_rows)
            block = ensure_2d(grown_reader.read(y0, y1, 0, w), "grown mask block").astype(label_dtype, copy=False)
            if resume_tmp is None:
                refined_out[y0:y1, :] = block
            raw_pixels += int(np.count_nonzero(block))
            if y0 == 0 or y1 == h or ((y0 // block_rows) % 50 == 0):
                action = "Initialized" if resume_tmp is None else "Checked"
                print(f"[INFO] {action} refined output rows {y0}:{y1}", flush=True)
            del block
        refined_out.flush()

        ys = tile_starts(h, tile_size, overlap)
        xs = tile_starts(w, tile_size, overlap)
        total_tiles = len(ys) * len(xs)
        processed_tiles = resume_tiles if resume_by_grid else 0
        resumed_tiles_skipped = resume_tiles if resume_by_grid else 0
        skipped_no_seed = 0
        failed_tiles = 0
        cleanup_added_pixels = 0
        cleanup_removed_pixels = 0
        cleanup_relabel_pixels = 0
        tile_runtime_sec = 0.0
        tile_cfg = replace(
            med_cfg,
            save_debug=False,
            cluster_tile_size=tile_size,
            cluster_tile_overlap=overlap,
        )

        for yi, y0 in enumerate(ys, start=1):
            y1 = min(h, y0 + tile_size)
            for xi, x0 in enumerate(xs, start=1):
                x1 = min(w, x0 + tile_size)
                cy0, cy1, cx0, cx1 = commit_bounds(y0, y1, x0, x1, shape, overlap)
                if cy1 <= cy0 or cx1 <= cx0:
                    continue
                if resume_by_grid and (yi < resume_grid_y or (yi == resume_grid_y and xi <= resume_grid_x)):
                    if (yi == resume_grid_y and xi == resume_grid_x) or (xi == 1 and yi % 5 == 0):
                        print(
                            f"[INFO] Resume grid-skip through grid {yi}/{len(ys)}, {xi}/{len(xs)} "
                            f"(foreground_tiles={resume_tiles})",
                            flush=True,
                        )
                        write_stream_progress(
                            progress_path,
                            {
                                "sample_id": args.sample_id,
                                "mode": "large_image_streaming",
                                "status": "resume_grid_skip",
                                "processed_tiles": int(processed_tiles),
                                "resumed_tiles_skipped": int(resumed_tiles_skipped),
                                "grid_y": int(yi),
                                "grid_x": int(xi),
                                "grid_y_total": int(len(ys)),
                                "grid_x_total": int(len(xs)),
                                "total_grid": int(total_tiles),
                                "elapsed_sec": round(float(time.perf_counter() - start), 3),
                                "checkpoint": str(tmp_flat),
                            },
                        )
                    continue
                tile_seed = ensure_2d(seed_reader.read(y0, y1, x0, x1), "seed mask tile").astype(label_dtype, copy=False)
                if not np.any(tile_seed > 0):
                    skipped_no_seed += 1
                    continue
                tile_grown = ensure_2d(grown_reader.read(y0, y1, x0, x1), "grown mask tile").astype(label_dtype, copy=False)
                if not np.any(tile_grown > 0):
                    skipped_no_seed += 1
                    continue
                if processed_tiles < resume_tiles:
                    processed_tiles += 1
                    resumed_tiles_skipped += 1
                    if processed_tiles == resume_tiles or processed_tiles % 25 == 0:
                        print(
                            f"[INFO] Resume skip foreground tile {processed_tiles}/{resume_tiles} "
                            f"(grid {yi}/{len(ys)}, {xi}/{len(xs)})",
                            flush=True,
                        )
                    write_stream_progress(
                        progress_path,
                        {
                            "sample_id": args.sample_id,
                            "mode": "large_image_streaming",
                            "status": "resume_skip",
                            "processed_tiles": int(processed_tiles),
                            "resumed_tiles_skipped": int(resumed_tiles_skipped),
                            "grid_y": int(yi),
                            "grid_x": int(xi),
                            "grid_y_total": int(len(ys)),
                            "grid_x_total": int(len(xs)),
                            "total_grid": int(total_tiles),
                            "elapsed_sec": round(float(time.perf_counter() - start), 3),
                            "checkpoint": str(tmp_flat),
                        },
                    )
                    del tile_seed, tile_grown
                    continue
                tile_image = image_reader.read(y0, y1, x0, x1)
                tile_start = time.perf_counter()
                try:
                    _, _, runtime_sec, _, artifacts = run_medsam_border_refine(
                        image=tile_image,
                        seed_labels=tile_seed,
                        baseline_tissue_mask=tile_grown > 0,
                        config=tile_cfg,
                        baseline_label_map=tile_grown,
                    )
                except MedSAMUnavailableError:
                    raise
                except Exception:
                    failed_tiles += 1
                    raise
                tile_runtime_sec += float(runtime_sec)
                refined_tile = np.asarray(artifacts.get("label_map", tile_grown), dtype=label_dtype)

                ly0, ly1 = cy0 - y0, cy1 - y0
                lx0, lx1 = cx0 - x0, cx1 - x0
                raw_commit = tile_grown[ly0:ly1, lx0:lx1]
                new_commit = refined_tile[ly0:ly1, lx0:lx1]
                raw_fg = raw_commit > 0
                new_fg = new_commit > 0
                cleanup_removed_pixels += int(np.count_nonzero(raw_fg & ~new_fg))
                cleanup_added_pixels += int(np.count_nonzero(new_fg & ~raw_fg))
                cleanup_relabel_pixels += int(np.count_nonzero((raw_commit != new_commit) & raw_fg & new_fg))
                refined_out[cy0:cy1, cx0:cx1] = new_commit
                processed_tiles += 1
                if processed_tiles == 1 or processed_tiles % 10 == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        f"[INFO] Streaming MedSAM tile {processed_tiles} processed "
                        f"(grid {yi}/{len(ys)}, {xi}/{len(xs)}; total_grid={total_tiles}; elapsed_sec={elapsed:.1f})",
                        flush=True,
                    )
                write_stream_progress(
                    progress_path,
                    {
                        "sample_id": args.sample_id,
                        "mode": "large_image_streaming",
                        "status": "running",
                        "processed_tiles": int(processed_tiles),
                        "resumed_tiles_skipped": int(resumed_tiles_skipped),
                        "tiles_skipped_no_seed": int(skipped_no_seed),
                        "grid_y": int(yi),
                        "grid_x": int(xi),
                        "grid_y_total": int(len(ys)),
                        "grid_x_total": int(len(xs)),
                        "total_grid": int(total_tiles),
                        "elapsed_sec": round(float(time.perf_counter() - start), 3),
                        "checkpoint": str(tmp_flat),
                    },
                )
                del tile_image, tile_seed, tile_grown, refined_tile, raw_commit, new_commit
                gc.collect()

        refined_out.flush()
        print("[INFO] Streaming MedSAM loop complete; counting final refined pixels", flush=True)
        final_pixels = count_nonzero_2d_blocks(refined_out, block_rows=block_rows)
        diag_step = preview_step_for_shape(shape, max_side=2048)
        print(f"[INFO] Building tiled diagnostic previews with step={diag_step}", flush=True)
        image_preview = read_stride_tiled(image_reader, diag_step, tile_size=tile_size)
        seed_preview = ensure_2d(read_stride_tiled(seed_reader, diag_step, tile_size=tile_size), "seed mask preview")
        raw_preview = ensure_2d(read_stride_tiled(grown_reader, diag_step, tile_size=tile_size), "grown mask preview")
        refined_preview = array_stride_tiled(refined_out, diag_step, tile_size=tile_size)
        raw_vs_final_change = to_uint8_rgb(image_preview)
        raw_only = (raw_preview > 0) & ~(refined_preview > 0)
        final_only = (refined_preview > 0) & ~(raw_preview > 0)
        relabeled = (raw_preview != refined_preview) & (raw_preview > 0) & (refined_preview > 0)
        raw_vs_final_change[raw_only] = ((0.65 * raw_vs_final_change[raw_only]) + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
        raw_vs_final_change[final_only] = ((0.65 * raw_vs_final_change[final_only]) + 0.35 * np.array([0, 255, 255])).astype(np.uint8)
        raw_vs_final_change[relabeled] = ((0.55 * raw_vs_final_change[relabeled]) + 0.45 * np.array([255, 255, 0])).astype(np.uint8)

        save_png(outdir / f"{prefix}_medsam_seed_labels.png", overlay_labels(image_preview, seed_preview))
        save_png(outdir / f"{prefix}_medsam_raw_labels.png", overlay_labels(image_preview, raw_preview))
        save_png(outdir / f"{prefix}_medsam_refined_labels.png", overlay_labels(image_preview, refined_preview))
        save_png(outdir / f"{prefix}_medsam_boundary_compare.png", boundary_compare(image_preview, raw_preview > 0, refined_preview > 0))
        save_png(outdir / f"{prefix}_medsam_streaming_change_map.png", raw_vs_final_change)
        save_png(Path(args.preview), overlay_labels(image_preview, refined_preview, alpha=float(args.preview_alpha)))

        panel_path = outdir / f"{prefix}_medsam_raw_vs_final_panel.png"
        make_panel(
            [
                ("Original image", to_uint8_rgb(image_preview)),
                ("Step-15 labels", overlay_labels(image_preview, seed_preview)),
                ("Grown labels before MedSAM", overlay_labels(image_preview, raw_preview)),
                ("Streaming MedSAM refined labels", overlay_labels(image_preview, refined_preview)),
                ("What refinements changed", raw_vs_final_change),
            ],
            panel_path,
            columns=3,
        )

        random_fullres_qc = None
        if int(args.medsam_qc_crop_size) > 0:
            y0, y1, x0, x1 = choose_random_tissue_crop(
                raw_preview=raw_preview,
                refined_preview=refined_preview,
                step=diag_step,
                full_shape=shape,
                crop_size=int(args.medsam_qc_crop_size),
                seed=int(args.medsam_qc_random_seed),
            )
            qc_path = outdir / f"{prefix}_medsam_random_fullres_qc.png"
            print(
                f"[INFO] Writing native-resolution random MedSAM QC crop: "
                f"y={y0}:{y1}, x={x0}:{x1}, path={qc_path}",
                flush=True,
            )
            image_crop = image_reader.read(y0, y1, x0, x1)
            raw_crop = ensure_2d(grown_reader.read(y0, y1, x0, x1), "grown mask MedSAM QC crop")
            refined_crop = np.asarray(refined_out[y0:y1, x0:x1])
            random_fullres_qc = save_fullres_qc_crop(
                image_crop=image_crop,
                raw_crop=raw_crop,
                refined_crop=refined_crop,
                out_path=qc_path,
                crop_bounds=(y0, y1, x0, x1),
            )
            del image_crop, raw_crop, refined_crop

        summary = {
            "sample_id": args.sample_id,
            "mode": "large_image_streaming",
            "image_shape": [int(h), int(w)],
            "medsam_device": str(med_cfg.device),
            "source_mpp_x": float(args.source_mpp_x),
            "source_mpp_y": float(args.source_mpp_y),
            "medsam_runtime_sec": round(float(time.perf_counter() - start), 3),
            "medsam_tile_inference_sec": round(float(tile_runtime_sec), 3),
            "tiles_total": int(total_tiles),
            "tiles_processed": int(processed_tiles),
            "tiles_resumed_skipped": int(resumed_tiles_skipped),
            "tiles_skipped_no_seed": int(skipped_no_seed),
            "tiles_failed": int(failed_tiles),
            "tile_size": int(tile_size),
            "tile_overlap": int(overlap),
            "raw_pixels": int(raw_pixels),
            "final_pixels": int(final_pixels),
            "cleanup_removed_pixels": int(cleanup_removed_pixels),
            "cleanup_added_pixels": int(cleanup_added_pixels),
            "cleanup_relabel_pixels": int(cleanup_relabel_pixels),
            "diagnostic_preview_step": int(diag_step),
            "image_reader": image_reader.kind,
            "grown_reader": grown_reader.kind,
            "seed_reader": seed_reader.kind,
        }
        if random_fullres_qc is not None:
            summary["random_fullres_qc"] = random_fullres_qc
        (outdir / f"{prefix}_medsam_summary.json").write_text(json.dumps(summary, indent=2))
        write_stream_progress(progress_path, {**summary, "status": "complete", "checkpoint": str(tmp_flat)})

        del refined_out
        gc.collect()
        try:
            print(f"[INFO] Pyramidizing refined checkpoint to {args.out}", flush=True)
            pyramidize_or_fallback(tmp_flat, args.out, args)
        finally:
            if not args.keep_tmp and resume_tmp is None:
                try:
                    tmp_flat.unlink()
                except Exception:
                    pass

        print(json.dumps(summary, indent=2))
        print(f"[OK] wrote streaming refined mask: {args.out}")
        print(f"[OK] wrote streaming refinement panel: {panel_path}")
    finally:
        image_reader.close()
        seed_reader.close()
        grown_reader.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refine the step-16 grown tissue mask with MedSAM while preserving the protected core and seeded labels."
    )
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--image", required=True, help="ROI image TIFF used by the pipeline")
    ap.add_argument("--seed-mask", required=True, help="Step-15 cluster mask TIFF")
    ap.add_argument("--grown-mask", required=True, help="Step-16 grown tissue mask TIFF")
    ap.add_argument("--out", required=True, help="Output refined OME-TIFF path")
    ap.add_argument("--resolution-json", default="",
                    help="Pipeline shift/resolution JSON containing authoritative source_mpp metadata.")
    ap.add_argument("--default-mpp", type=float, default=0.0,
                    help="Fallback MPP used only when --resolution-json has no physical size.")
    ap.add_argument("--preview", required=True, help="Output preview PNG path")
    ap.add_argument("--preview-factor", type=int, default=10)
    ap.add_argument("--preview-threshold-mb", type=float, default=100.0)
    ap.add_argument("--preview-alpha", type=float, default=0.45)
    ap.add_argument("--pyr-compression", default="LZW")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--downsample", default="GAUSSIAN")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--large-image-mode", choices=("auto", "full", "stream"), default="auto",
                    help="Use streamed tiled MedSAM for WSI-scale inputs; 'auto' switches above --large-image-max-pixels.")
    ap.add_argument("--large-image-max-pixels", type=int, default=50_000_000,
                    help="Pixel threshold for automatic streamed MedSAM mode.")
    ap.add_argument("--stream-block-rows", type=int, default=512,
                    help="Rows per block when initializing streamed large-image outputs.")
    ap.add_argument("--stream-resume-tmp", default="",
                    help="Existing flat checkpoint TIFF from an interrupted streamed MedSAM run.")
    ap.add_argument("--stream-resume-tiles", type=int, default=0,
                    help="Number of previously completed foreground tiles to skip when resuming from --stream-resume-tmp.")
    ap.add_argument("--stream-resume-grid-y", type=int, default=0,
                    help="1-based grid row of the last completed streamed tile; skips earlier grid cells without re-reading them.")
    ap.add_argument("--stream-resume-grid-x", type=int, default=0,
                    help="1-based grid column of the last completed streamed tile; used with --stream-resume-grid-y.")
    ap.add_argument("--stream-progress-json", default="",
                    help="Optional progress JSON path written during streamed MedSAM.")
    ap.add_argument("--medsam-qc-crop-size", type=int, default=1024,
                    help="Native-resolution square crop size for random tissue MedSAM QC preview; set 0 to disable.")
    ap.add_argument("--medsam-qc-random-seed", type=int, default=1729,
                    help="Random seed used to choose the tissue location for the MedSAM QC crop.")

    ap.add_argument("--medsam-checkpoint", default=str(DEFAULT_MEDSAM_CHECKPOINT))
    ap.add_argument("--medsam-device", default="cuda")
    ap.add_argument("--medsam-bbox-margin", type=int, default=144)
    ap.add_argument("--medsam-component-min-area", type=int, default=200)
    ap.add_argument("--medsam-component-merge-distance", type=int, default=24)
    ap.add_argument("--medsam-seed-dilation-radius", type=int, default=8)
    ap.add_argument("--medsam-core-erosion-radius", type=int, default=43)
    ap.add_argument("--medsam-outer-dilation-radius", type=int, default=53)
    ap.add_argument("--medsam-min-object-size", type=int, default=5000)
    ap.add_argument("--medsam-smooth-radius", type=int, default=5)
    ap.add_argument("--medsam-cluster-tile-size", type=int, default=4096)
    ap.add_argument("--medsam-cluster-tile-overlap", type=int, default=512)
    ap.add_argument("--medsam-force-core-preservation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--medsam-save-debug", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    args.source_mpp_x, args.source_mpp_y = read_mpp_json(args.resolution_json, args.default_mpp)

    med_cfg = make_medsam_config(args)
    image_shape = tiff_spatial_shape(args.image, "image")
    seed_shape = tiff_spatial_shape(args.seed_mask, "seed mask")
    grown_shape = tiff_spatial_shape(args.grown_mask, "grown mask")
    if seed_shape != grown_shape:
        raise ValueError(f"Step-15 and step-16 masks must share the same shape: {seed_shape} vs {grown_shape}")
    if image_shape != seed_shape:
        raise ValueError(f"Image and masks must share the same spatial shape: {image_shape} vs {seed_shape}")
    args.output_spatial_shape = seed_shape
    print(
        f"[INFO] Output physical resolution: mpp_x={args.source_mpp_x:.9g}, "
        f"mpp_y={args.source_mpp_y:.9g}"
    )
    pixel_count = int(seed_shape[0]) * int(seed_shape[1])
    if args.large_image_mode == "stream" or (
        args.large_image_mode == "auto" and pixel_count > int(args.large_image_max_pixels)
    ):
        run_large_image_streaming_medsam(args, med_cfg)
        return

    image_rgb = load_tiff_memmap(args.image, "image")
    seed_labels = ensure_2d(load_tiff_memmap(args.seed_mask, "seed mask"), "seed mask")
    grown_labels = ensure_2d(load_tiff_memmap(args.grown_mask, "grown mask"), "grown mask")
    label_dtype = seed_labels.dtype
    if np.issubdtype(seed_labels.dtype, np.integer):
        min_label = int(np.min(seed_labels, initial=0))
        max_label = int(np.max(seed_labels, initial=0))
        if min_label >= 0 and max_label <= np.iinfo(np.uint16).max:
            label_dtype = np.uint16
    seed_labels = seed_labels.astype(label_dtype, copy=False)
    grown_labels = grown_labels.astype(label_dtype, copy=False)

    if grown_labels.shape != seed_labels.shape:
        raise ValueError(f"Step-15 and step-16 masks must share the same shape: {seed_labels.shape} vs {grown_labels.shape}")

    if image_rgb.ndim == 3 and image_rgb.shape[:2] != seed_labels.shape:
        raise ValueError(f"Image and masks must share the same spatial shape: {image_rgb.shape[:2]} vs {seed_labels.shape}")
    if image_rgb.ndim == 2 and image_rgb.shape != seed_labels.shape:
        raise ValueError(f"Image and masks must share the same spatial shape: {image_rgb.shape} vs {seed_labels.shape}")

    try:
        refined_tissue, probability_map, runtime_sec, med_meta, artifacts = run_medsam_border_refine(
            image=image_rgb,
            seed_labels=seed_labels,
            baseline_tissue_mask=grown_labels > 0,
            config=med_cfg,
            baseline_label_map=grown_labels,
        )
    except MedSAMUnavailableError as exc:
        raise RuntimeError(f"MedSAM unavailable: {exc}") from exc

    refined_labels = np.asarray(artifacts.get("label_map", grown_labels), dtype=label_dtype)
    raw_labels = np.asarray(artifacts.get("raw_medsam_label_map", refined_labels), dtype=label_dtype)
    protected_core = np.asarray(artifacts.get("protected_core", np.zeros_like(grown_labels)), dtype=bool)
    editable_band = np.asarray(artifacts.get("editable_band", np.zeros_like(grown_labels)), dtype=bool)

    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = sample_prefix(args.sample_id, args.out)
    debug_dir = outdir / f"{prefix}_medsam_debug"

    if med_cfg.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(med_meta)
        meta["sample_id"] = args.sample_id
        meta["raw_pixels"] = int((raw_labels > 0).sum())
        meta["final_pixels"] = int((refined_labels > 0).sum())
        (debug_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        for name, arr in artifacts.items():
            np.save(debug_dir / f"{name}.npy", np.asarray(arr))
        if probability_map is not None:
            np.save(debug_dir / "probability_map.npy", np.asarray(probability_map, dtype=np.float32))

    summary = {
        "sample_id": args.sample_id,
        "source_mpp_x": float(args.source_mpp_x),
        "source_mpp_y": float(args.source_mpp_y),
        "medsam_runtime_sec": round(float(runtime_sec), 3),
        "raw_pixels": int((raw_labels > 0).sum()),
        "final_pixels": int((refined_labels > 0).sum()),
        "cleanup_removed_pixels": int(((raw_labels > 0) & ~(refined_labels > 0)).sum()),
        "cleanup_added_pixels": int(((refined_labels > 0) & ~(raw_labels > 0)).sum()),
        "cleanup_relabel_pixels": int(((raw_labels != refined_labels) & (raw_labels > 0) & (refined_labels > 0)).sum()),
        "protected_core_pixels": int(protected_core.sum()),
        "editable_band_pixels": int(editable_band.sum()),
    }
    diag_step = preview_step_for_shape(seed_labels.shape, max_side=2048)
    summary["diagnostic_preview_step"] = int(diag_step)

    image_preview = preview_subsample(image_rgb, diag_step)
    seed_preview = preview_subsample(seed_labels, diag_step)
    raw_preview = preview_subsample(raw_labels, diag_step)
    refined_preview = preview_subsample(refined_labels, diag_step)
    protected_preview = preview_subsample(protected_core, diag_step)
    editable_preview = preview_subsample(editable_band, diag_step)
    prob_preview = preview_subsample(probability_map, diag_step) if probability_map is not None else None

    raw_vs_final_change = to_uint8_rgb(image_preview)
    raw_only = (raw_preview > 0) & ~(refined_preview > 0)
    final_only = (refined_preview > 0) & ~(raw_preview > 0)
    relabeled = (raw_preview != refined_preview) & (raw_preview > 0) & (refined_preview > 0)
    raw_vs_final_change[raw_only] = ((0.65 * raw_vs_final_change[raw_only]) + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
    raw_vs_final_change[final_only] = ((0.65 * raw_vs_final_change[final_only]) + 0.35 * np.array([0, 255, 255])).astype(np.uint8)
    raw_vs_final_change[relabeled] = ((0.55 * raw_vs_final_change[relabeled]) + 0.45 * np.array([255, 255, 0])).astype(np.uint8)

    save_png(outdir / f"{prefix}_medsam_seed_labels.png", overlay_labels(image_preview, seed_preview))
    save_png(outdir / f"{prefix}_medsam_protected_core.png", overlay_mask(image_preview, protected_preview, (255, 200, 0)))
    save_png(outdir / f"{prefix}_medsam_editable_band.png", overlay_mask(image_preview, editable_preview, (180, 0, 255)))
    save_png(outdir / f"{prefix}_medsam_raw_labels.png", overlay_labels(image_preview, raw_preview))
    save_png(outdir / f"{prefix}_medsam_refined_labels.png", overlay_labels(image_preview, refined_preview))
    if prob_preview is not None:
        save_png(outdir / f"{prefix}_medsam_probability_heatmap.png", heatmap(prob_preview))
    save_png(outdir / f"{prefix}_medsam_boundary_compare.png", boundary_compare(image_preview, raw_preview > 0, refined_preview > 0))

    panel_path = outdir / f"{prefix}_medsam_raw_vs_final_panel.png"
    make_panel(
        [
            ("Original image", to_uint8_rgb(image_preview)),
            ("Step-15 labels", overlay_labels(image_preview, seed_preview)),
            ("Protected core", overlay_mask(image_preview, protected_preview, (255, 200, 0))),
            ("Editable band", overlay_mask(image_preview, editable_preview, (180, 0, 255))),
            ("Immediately after MedSAM", overlay_labels(image_preview, raw_preview)),
            ("After refinements", overlay_labels(image_preview, refined_preview)),
            ("What refinements changed", raw_vs_final_change),
        ],
        panel_path,
        columns=3,
    )

    if int(args.medsam_qc_crop_size) > 0:
        y0, y1, x0, x1 = choose_random_tissue_crop(
            raw_preview=raw_preview,
            refined_preview=refined_preview,
            step=diag_step,
            full_shape=seed_labels.shape,
            crop_size=int(args.medsam_qc_crop_size),
            seed=int(args.medsam_qc_random_seed),
        )
        image_crop = np.asarray(image_rgb[y0:y1, x0:x1, ...]) if image_rgb.ndim == 3 else np.asarray(image_rgb[y0:y1, x0:x1])
        raw_crop = np.asarray(raw_labels[y0:y1, x0:x1])
        refined_crop = np.asarray(refined_labels[y0:y1, x0:x1])
        summary["random_fullres_qc"] = save_fullres_qc_crop(
            image_crop=image_crop,
            raw_crop=raw_crop,
            refined_crop=refined_crop,
            out_path=outdir / f"{prefix}_medsam_random_fullres_qc.png",
            crop_bounds=(y0, y1, x0, x1),
        )
        del image_crop, raw_crop, refined_crop
    (outdir / f"{prefix}_medsam_summary.json").write_text(json.dumps(summary, indent=2))

    tmp_flat = outdir / f".tmp_refined_flat_{os.getpid()}_{int(time.time())}.tif"
    maxlab = int(refined_labels.max()) if refined_labels.size else 0
    flat = refined_labels.astype(label_storage_dtype(maxlab))
    imwrite(
        tmp_flat,
        flat,
        bigtiff=True,
        byteorder=flat.dtype.byteorder if flat.dtype.itemsize > 1 else None,
        **tiff_resolution_kwargs(args.source_mpp_x, args.source_mpp_y, "YX"),
    )
    try:
        pyramidize_or_fallback(tmp_flat, args.out, args)
    finally:
        try:
            tmp_flat.unlink()
        except Exception:
            pass

    save_preview_png(
        image_path=args.image,
        grown=refined_labels,
        out_png=args.preview,
        factor=args.preview_factor,
        size_threshold_mb=args.preview_threshold_mb,
        alpha=args.preview_alpha,
        default_value=0,
    )

    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote refined mask: {args.out}")
    print(f"[OK] wrote refinement panel: {panel_path}")


if __name__ == "__main__":
    main()
