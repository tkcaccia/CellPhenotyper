#!/usr/bin/env python3
"""Build tissue-filtered, calibrated UNI2 grid observations without a dense label mask."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image, ImageDraw

from ome_tiff_metadata import read_mpp_json
from uni2_grid import centered_axis_lattice, resolve_spatial_grid_geometry


GRID_FIELDS = [
    "label",
    "x",
    "y",
    "polygon_label",
    "grid_row",
    "grid_col",
    "core_x0",
    "core_y0",
    "core_x1",
    "core_y1",
    "tile_x0",
    "tile_y0",
    "tile_x1",
    "tile_y1",
    "tissue_px",
    "tissue_fraction",
    "area_px",
]


class MaskReader:
    """Windowed 2D TIFF access with a disk-backed fallback."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self._tif = tifffile.TiffFile(str(self.path))
        series = self._tif.series[0]
        if len(series.shape) != 2:
            raise RuntimeError(f"Tissue mask must be 2D, got shape={series.shape}")
        self.shape = tuple(map(int, series.shape))
        self._zarr = None
        self._array = None
        try:
            import zarr

            self._zarr = zarr.open(series.aszarr(), mode="r")
            if not hasattr(self._zarr, "shape"):
                keys = list(self._zarr.array_keys())
                if not keys:
                    raise RuntimeError("Tissue-mask Zarr store contains no arrays")
                self._zarr = self._zarr["0" if "0" in keys else sorted(keys)[0]]
        except Exception:
            self._array = series.asarray(out="memmap")

    def read_rows(self, y0: int, y1: int) -> np.ndarray:
        source = self._zarr if self._zarr is not None else self._array
        return np.asarray(source[int(y0) : int(y1), :])

    def close(self) -> None:
        self._tif.close()


def read_image_shape(path: Path) -> tuple[int, int]:
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes = str(series.axes)
        shape = tuple(map(int, series.shape))
        if "Y" in axes and "X" in axes:
            return shape[axes.index("Y")], shape[axes.index("X")]
        page = tif.pages[0]
        return int(page.imagelength), int(page.imagewidth)


def build_records(
    mask: MaskReader,
    *,
    image_shape: tuple[int, int] | None = None,
    context_size: int,
    stride: int,
    min_tissue_fraction: float,
    record_callback=None,
    collect_records: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray, np.ndarray]:
    mask_height, mask_width = mask.shape
    height, width = image_shape or mask.shape
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    scale_y = mask_height / height
    scale_x = mask_width / width
    if abs(scale_x - scale_y) / max(scale_x, scale_y) > 0.02:
        raise RuntimeError(
            f"Tissue mask aspect ratio is not aligned with the image: image={(height, width)}, "
            f"mask={mask.shape}, scale_x={scale_x:.6g}, scale_y={scale_y:.6g}"
        )
    x_centers, x_starts, x_ends = centered_axis_lattice(width, stride)
    y_centers, y_starts, y_ends = centered_axis_lattice(height, stride)
    occupancy = np.zeros((len(y_centers), len(x_centers)), dtype=np.float32)
    selected = np.zeros_like(occupancy, dtype=bool)
    records: list[dict[str, Any]] = []
    retained_units = 0
    half_context = context_size // 2

    for row, (center_y, core_y0, core_y1) in enumerate(zip(y_centers, y_starts, y_ends)):
        source_y0 = max(0, int(core_y0))
        source_y1 = min(height, int(core_y1))
        if source_y1 <= source_y0:
            continue
        mask_y0 = max(0, min(mask_height - 1, int(np.floor(source_y0 * scale_y))))
        mask_y1 = max(mask_y0 + 1, min(mask_height, int(np.ceil(source_y1 * scale_y))))
        strip = mask.read_rows(mask_y0, mask_y1) != 0
        column_counts = np.count_nonzero(strip, axis=0).astype(np.int64, copy=False)
        prefix = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(column_counts)))

        for col, (center_x, core_x0, core_x1) in enumerate(zip(x_centers, x_starts, x_ends)):
            source_x0 = max(0, int(core_x0))
            source_x1 = min(width, int(core_x1))
            visible_area = (source_x1 - source_x0) * (source_y1 - source_y0)
            if visible_area <= 0:
                continue
            mask_x0 = max(0, min(mask_width - 1, int(np.floor(source_x0 * scale_x))))
            mask_x1 = max(mask_x0 + 1, min(mask_width, int(np.ceil(source_x1 * scale_x))))
            tissue_px = int(prefix[mask_x1] - prefix[mask_x0])
            mask_area = (mask_x1 - mask_x0) * (mask_y1 - mask_y0)
            tissue_fraction = tissue_px / mask_area
            occupancy[row, col] = tissue_fraction
            if tissue_fraction < min_tissue_fraction:
                continue

            retained_units += 1
            label = retained_units
            selected[row, col] = True
            record = {
                "label": label,
                "x": int(center_x),
                "y": int(center_y),
                "polygon_label": "grid",
                "grid_row": row,
                "grid_col": col,
                "core_x0": int(core_x0),
                "core_y0": int(core_y0),
                "core_x1": int(core_x1),
                "core_y1": int(core_y1),
                "tile_x0": int(center_x) - half_context,
                "tile_y0": int(center_y) - half_context,
                "tile_x1": int(center_x) - half_context + context_size,
                "tile_y1": int(center_y) - half_context + context_size,
                "tissue_px": tissue_px,
                "tissue_fraction": tissue_fraction,
                "area_px": visible_area,
            }
            if record_callback is not None:
                record_callback(record)
            if collect_records:
                records.append(record)

    geometry = {
        "grid_rows": int(len(y_centers)),
        "grid_cols": int(len(x_centers)),
        "grid_origin_x": int(x_starts[0]),
        "grid_origin_y": int(y_starts[0]),
        "candidate_units": int(occupancy.size),
        "retained_units": int(retained_units),
        "tissue_mask_height_px": int(mask_height),
        "tissue_mask_width_px": int(mask_width),
        "tissue_mask_scale_x": float(scale_x),
        "tissue_mask_scale_y": float(scale_y),
    }
    return records, geometry, occupancy, selected


def write_preview(path: Path, occupancy: np.ndarray, selected: np.ndarray, max_side: int) -> None:
    intensity = np.clip(np.rint(occupancy * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack((intensity, intensity, intensity), axis=-1)
    rgb[selected, 0] = np.maximum(rgb[selected, 0], 30)
    rgb[selected, 1] = np.maximum(rgb[selected, 1], 180)
    rgb[selected, 2] = np.maximum(rgb[selected, 2], 80)

    rows, cols = occupancy.shape
    scale = max(1, min(8, int(max_side) // max(1, rows, cols)))
    preview = Image.fromarray(rgb).resize(
        (max(1, cols * scale), max(1, rows * scale)), resample=Image.Resampling.NEAREST
    )
    if scale >= 4 and rows * cols <= 1_000_000:
        draw = ImageDraw.Draw(preview)
        for x in range(0, preview.width, scale):
            draw.line((x, 0, x, preview.height), fill=(25, 25, 25), width=1)
        for y in range(0, preview.height, scale):
            draw.line((0, y, preview.width, y), fill=(25, 25, 25), width=1)
    preview.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--tissue-mask", required=True)
    parser.add_argument("--resolution-json", required=True)
    parser.add_argument("--objects-out", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--preview-out", required=True)
    parser.add_argument("--model-tile-size", type=int, default=224)
    parser.add_argument("--inner-square-size", type=int, default=90)
    parser.add_argument(
        "--grid-stride-size", type=int, default=0,
        help="Grid stride in model-input pixels; zero uses the inner-square size",
    )
    parser.add_argument("--target-mpp", type=float, default=0.25)
    parser.add_argument("--default-source-mpp", type=float, default=0.25)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.05)
    parser.add_argument("--preview-max-side", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_tissue_fraction <= 1.0:
        raise ValueError("--min-tissue-fraction must be between 0 and 1")

    image_path = Path(args.image)
    mask_path = Path(args.tissue_mask)
    mpp_x, mpp_y = read_mpp_json(args.resolution_json, fallback=args.default_source_mpp)
    source_mpp = (mpp_x + mpp_y) / 2.0
    if abs(mpp_x - mpp_y) / source_mpp > 0.05:
        raise RuntimeError(f"Grid mode requires nearly isotropic pixels, got mpp_x={mpp_x}, mpp_y={mpp_y}")
    context_size, inner_square_source_size, stride = resolve_spatial_grid_geometry(
        model_tile_size=args.model_tile_size,
        inner_square_size=args.inner_square_size,
        grid_stride_size=args.grid_stride_size,
        source_mpp=source_mpp,
        target_mpp=args.target_mpp,
    )

    image_shape = read_image_shape(image_path)
    mask = MaskReader(mask_path)
    with Path(args.objects_out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GRID_FIELDS)
        writer.writeheader()
        try:
            _, geometry, occupancy, selected = build_records(
                mask,
                image_shape=image_shape,
                context_size=context_size,
                stride=stride,
                min_tissue_fraction=args.min_tissue_fraction,
                record_callback=writer.writerow,
                collect_records=False,
            )
        finally:
            mask.close()

    if geometry["retained_units"] == 0:
        raise RuntimeError("No grid units passed the tissue-occupancy threshold")

    with Path(args.objects_out).open("r", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    if header != GRID_FIELDS:
        raise RuntimeError("Grid observation CSV header validation failed")

    metadata = {
        "schema_version": 1,
        "observation_type": "spatial_grid",
        "coordinate_space": "crop_roi_level0_pixels",
        "image": str(image_path.resolve()),
        "tissue_mask": str(mask_path.resolve()),
        "image_height_px": image_shape[0],
        "image_width_px": image_shape[1],
        "source_mpp_x": mpp_x,
        "source_mpp_y": mpp_y,
        "target_mpp": args.target_mpp,
        "model_tile_size_px": args.model_tile_size,
        "extraction_tile_size_source_px": context_size,
        "inner_square_size_model_px": args.inner_square_size,
        "inner_square_size_source_px": inner_square_source_size,
        "grid_stride_size_model_px": (
            args.grid_stride_size if args.grid_stride_size > 0 else args.inner_square_size
        ),
        "grid_stride_source_px": stride,
        "context_overlap_source_px": context_size - stride,
        "context_overlap_fraction": 1.0 - (stride / context_size),
        "min_tissue_fraction": args.min_tissue_fraction,
        **geometry,
    }
    Path(args.metadata_out).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    write_preview(Path(args.preview_out), occupancy, selected, args.preview_max_side)
    print(
        "[INFO] UNI2 spatial grid: "
        f"shape={image_shape[0]}x{image_shape[1]} context={context_size}px stride={stride}px "
        f"overlap={context_size - stride}px retained={geometry['retained_units']}/{occupancy.size}"
    )


if __name__ == "__main__":
    main()
