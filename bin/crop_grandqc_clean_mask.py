#!/usr/bin/env python3
"""Crop a low-resolution GrandQC clean-tissue mask to the analysis ROI."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tifffile

from prepare_analysis_crop import iter_polygon_rings


def crop_aligned_support(mask: np.ndarray, shift: dict) -> tuple[np.ndarray, dict]:
    """Sample a crop-aligned grid at pixel centres in the original mask frame.

    A floor/ceil source-mask slice usually extends beyond the requested native
    crop. Stretching that slice to the crop would shift tissue boundaries. The
    output instead spans exactly the crop and samples the original mask there.
    This is categorical resampling, not recovery of sub-GrandQC tissue gaps.
    """
    if mask.ndim != 2 or not mask.size or mask.dtype.kind not in "buif":
        raise ValueError("GrandQC clean-tissue mask must be a nonempty 2D numeric raster")
    if not np.isfinite(mask).all() or np.any(mask < 0):
        raise ValueError("GrandQC clean-tissue mask must have finite nonnegative values")

    def integer(value, field):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
            raise ValueError(f"{field} must be an integer pixel coordinate or size")
        return int(value)

    full = shift["full_size"]
    crop = shift["crop_size"]
    bbox = shift["crop_bbox_xyxy"]
    fw, fh = (integer(full[key], f"full_size.{key}") for key in ("width", "height"))
    cw, ch = (integer(crop[key], f"crop_size.{key}") for key in ("width", "height"))
    x0, y0, x1, y1 = (integer(bbox[key], f"crop_bbox_xyxy.{key}") for key in ("x0", "y0", "x1", "y1"))
    if min(fw, fh, cw, ch) <= 0 or not (0 <= x0 < x1 <= fw and 0 <= y0 < y1 <= fh):
        raise ValueError("Crop bounds must lie inside positive full-image dimensions")
    if (x1 - x0, y1 - y0) != (cw, ch):
        raise ValueError("crop_size must match crop_bbox_xyxy exactly")
    offset = shift.get("offset_crop_to_original")
    if offset is not None and (offset.get("dx"), offset.get("dy")) != (x0, y0):
        raise ValueError("Crop offset and bounding box disagree")
    mh, mw = mask.shape
    # Preserve approximately the original GrandQC sampling scale, including a
    # single output pixel for crops smaller than one source-mask pixel.
    ow, oh = max(1, (cw * mw + fw - 1) // fw), max(1, (ch * mh + fh - 1) // fh)
    xs = np.floor((x0 + (np.arange(ow) + .5) * cw / ow) * mw / fw).astype(np.int64)
    ys = np.floor((y0 + (np.arange(oh) + .5) * ch / oh) * mh / fh).astype(np.int64)
    output = mask[np.ix_(ys, xs)] != 0
    geometry = {
        "source_mask_shape_yx": [mh, mw],
        "source_mask_bbox_xyxy": [math.floor(x0 * mw / fw), math.floor(y0 * mh / fh),
                                   math.ceil(x1 * mw / fw), math.ceil(y1 * mh / fh)],
        "source_pixel_size_original_pixels_xy": [fw / mw, fh / mh],
        "output_origin_original_pixels_xy": [x0, y0],
        "output_pixel_size_original_pixels_xy": [cw / ow, ch / oh],
        "sampling": "crop_aligned_pixel_centres_into_original_grandqc_mask",
        "gap_precision": "GrandQC sampling resolution; finer gaps are not reconstructed",
        "footprint_precision": "Categorical pixel-centre approximation, not exact preservation of every source-mask boundary",
        "output_half_pixel_original_pixels_xy": [cw / ow / 2, ch / oh / 2],
    }
    return output, geometry


def rasterize_crop_roi_centres(roi: dict, shape: tuple[int, int], coordinate_size: tuple[int, int]) -> np.ndarray:
    """Union polygon interiors at crop-aligned pixel centres, retaining holes.

    Exterior-boundary centres are included; hole-boundary centres are excluded.
    Rings use even-odd scanline filling and must be finite, closed and nonzero-
    area. This checks ring structure, not arbitrary polygon topology. Coordinates
    are never clamped to image edges. Beyond the low-resolution output mask,
    working memory is O(output width + current polygon vertex count), not a
    native-resolution image or one mask per polygon.
    """
    dimensions = (*shape, *coordinate_size)
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in dimensions):
        raise ValueError("ROI raster and coordinate dimensions must be positive integers")
    height, width = map(int, shape)
    coordinate_width, coordinate_height = map(int, coordinate_size)
    x_centres = (np.arange(width, dtype=np.float64) + .5) * coordinate_width / width
    output = np.zeros((height, width), dtype=bool)

    def validate_ring(raw):
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            raise ValueError("ROI rings must contain at least four coordinates including closure")
        for point in raw:
            if not isinstance(point, (list, tuple)) or len(point) < 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point[:2]
            ):
                raise ValueError("ROI coordinates must be finite numeric x,y pairs")
        ring = np.asarray([point[:2] for point in raw], dtype=np.float64)
        if not np.array_equal(ring[0], ring[-1]):
            raise ValueError("ROI rings must be explicitly closed")
        # Translate before the shoelace sum to avoid cancellation for offset ROIs.
        shifted = ring - ring[0]
        area2 = np.sum(shifted[:-1, 0] * shifted[1:, 1] - shifted[1:, 0] * shifted[:-1, 1])
        if not np.isfinite(area2) or area2 == 0:
            raise ValueError("ROI rings must have finite nonzero area")
        return ring

    def ring_scanline(ring, y):
        row = np.zeros(width, dtype=bool)
        a, b = ring[:-1], ring[1:]
        crosses = (a[:, 1] > y) != (b[:, 1] > y)
        left, right = a[crosses], b[crosses]
        intersections = np.sort(left[:, 0] + (y - left[:, 1]) * (right[:, 0] - left[:, 0]) / (right[:, 1] - left[:, 1]))
        if len(intersections) % 2:
            raise ValueError("ROI ring has an invalid scanline intersection count")

        def fill_interval(x0, x1):
            start = np.searchsorted(x_centres, min(x0, x1), side="left")
            stop = np.searchsorted(x_centres, max(x0, x1), side="right")
            row[start:stop] = True

        for x0, x1 in zip(intersections[::2], intersections[1::2]):
            fill_interval(x0, x1)
        # The half-open edge crossing rule omits upper extrema and horizontal
        # segments. Explicitly include their boundary centres as well.
        on_line = a[:, 1] == y
        for point in a[on_line]:
            fill_interval(point[0], point[0])
        for start, end in zip(a[on_line & (b[:, 1] == y)], b[on_line & (b[:, 1] == y)]):
            fill_interval(start[0], end[0])
        return row

    polygon_count = 0
    for exterior, holes in iter_polygon_rings(roi):
        outer = validate_ring(exterior)
        interiors = [validate_ring(hole) for hole in holes]
        polygon_count += 1
        ymin, ymax = float(outer[:, 1].min()), float(outer[:, 1].max())
        first_row = max(0, math.ceil(ymin * height / coordinate_height - .5))
        stop_row = min(height, math.floor(ymax * height / coordinate_height - .5) + 1)
        for row_id in range(first_row, stop_row):
            y = (row_id + .5) * coordinate_height / height
            row = ring_scanline(outer, y)
            for hole in interiors:
                row &= ~ring_scanline(hole, y)
            output[row_id] |= row
    if not polygon_count:
        raise ValueError("ROI contains no polygon geometry")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True)
    parser.add_argument("--shift", required=True)
    parser.add_argument("--roi", required=True, help="Crop-coordinate ROI GeoJSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    shift = json.loads(Path(args.shift).read_text())
    try:
        mask = np.asarray(tifffile.memmap(args.mask))
    except ValueError:
        mask = np.asarray(tifffile.imread(args.mask))
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise RuntimeError(f"GrandQC clean-tissue mask must be 2D, got {mask.shape}")
    cropped, geometry = crop_aligned_support(mask, shift)
    crop_size = shift["crop_size"]
    roi = json.loads(Path(args.roi).read_text())
    roi_mask = rasterize_crop_roi_centres(
        roi,
        cropped.shape,
        (int(crop_size["width"]), int(crop_size["height"])),
    )
    cropped = np.where(cropped & roi_mask, 255, 0).astype(np.uint8)
    if not np.any(cropped):
        raise RuntimeError("Cropped GrandQC clean tissue and ROI do not overlap")
    tifffile.imwrite(args.output, cropped, compression="zlib")

    summary = {
        "source_mask": str(Path(args.mask).resolve()),
        **geometry,
        "output_shape_yx": list(map(int, cropped.shape)),
        "analysis_crop_shape_yx": [int(crop_size["height"]), int(crop_size["width"])],
        "scale_mask_per_crop_x": float(cropped.shape[1] / int(crop_size["width"])),
        "scale_mask_per_crop_y": float(cropped.shape[0] / int(crop_size["height"])),
        "clean_tissue_fraction": float(np.count_nonzero(cropped) / max(1, cropped.size)),
        "mask_contract": "grandqc_clean_tissue_intersection_roi",
        "roi_sampling": "crop_pixel_centres; polygon_union; exterior_boundary_included; hole_boundary_excluded",
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"[OK] GrandQC ROI mask={cropped.shape[1]}x{cropped.shape[0]} clean={summary['clean_tissue_fraction']:.4f}")


if __name__ == "__main__":
    main()
