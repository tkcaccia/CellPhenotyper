#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from PIL import Image, ImageDraw
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_holes, remove_small_objects

try:
    import pyvips
except Exception:  # pragma: no cover - optional runtime accelerator
    pyvips = None

try:
    import zarr
except Exception:  # pragma: no cover - optional fallback
    zarr = None


COORD_X_CANDIDATES = ("x", "centroid_x", "center_x", "x_centroid", "x_px", "X")
COORD_Y_CANDIDATES = ("y", "centroid_y", "center_y", "y_centroid", "y_px", "Y")
FULL_X_CANDIDATES = ("x_orig", "x_full", "global_x", "x_global")
FULL_Y_CANDIDATES = ("y_orig", "y_full", "global_y", "y_global")


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        arr_float = arr.astype(np.float32, copy=False)
        lo = float(np.nanpercentile(arr_float, 0.5))
        hi = float(np.nanpercentile(arr_float, 99.5))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(arr_float)) if arr_float.size else 0.0
            hi = float(np.nanmax(arr_float)) if arr_float.size else 1.0
        denom = max(hi - lo, 1e-6)
        arr = ((arr_float - lo) / denom * 255.0).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def _select_zarr_array(obj):
    if hasattr(obj, "shape") and hasattr(obj, "ndim"):
        return obj
    if hasattr(obj, "__getitem__"):
        for key in ("0", 0):
            try:
                sub = obj[key]
                if hasattr(sub, "shape"):
                    return sub
            except Exception:
                pass
        try:
            keys = list(obj.keys()) if hasattr(obj, "keys") else []
        except Exception:
            keys = []
        for key in keys:
            try:
                sub = obj[key]
                if hasattr(sub, "shape"):
                    return sub
            except Exception:
                continue
    return None


def read_thumbnail(image_path: Path, max_side: int) -> tuple[np.ndarray, int, int, str]:
    if pyvips is not None:
        try:
            image_vips = pyvips.Image.new_from_file(str(image_path), access="sequential")
            width = int(image_vips.width)
            height = int(image_vips.height)
            target_width = max(1, int(round(width * min(1.0, max_side / max(width, height)))))
            thumb = pyvips.Image.thumbnail(str(image_path), target_width)
            arr = np.ndarray(
                buffer=thumb.write_to_memory(),
                dtype=np.uint8,
                shape=(thumb.height, thumb.width, thumb.bands),
            ).copy()
            return _to_uint8_rgb(arr), height, width, "pyvips-thumbnail"
        except Exception:
            pass

    with tifffile.TiffFile(str(image_path)) as tf:
        if not tf.series:
            raise RuntimeError(f"No TIFF series found in {image_path}")
        series = tf.series[0]
        levels = list(getattr(series, "levels", []) or [series])
        level = levels[0]
        axes = (getattr(level, "axes", None) or getattr(series, "axes", None) or "")
        shape0 = tuple(levels[0].shape)
        if axes and "Y" in axes and "X" in axes:
            ydim0 = axes.index("Y")
            xdim0 = axes.index("X")
        else:
            ydim0, xdim0 = 0, 1
        full_h = int(shape0[ydim0])
        full_w = int(shape0[xdim0])

        for cand in reversed(levels):
            c_axes = (getattr(cand, "axes", None) or axes or "")
            c_shape = tuple(cand.shape)
            if c_axes and "Y" in c_axes and "X" in c_axes:
                cy = c_axes.index("Y")
                cx = c_axes.index("X")
            else:
                cy, cx = 0, 1
            if max(int(c_shape[cy]), int(c_shape[cx])) <= max_side * 2 or cand is levels[-1]:
                level = cand
                axes = c_axes
                break

        try:
            zobj = level.aszarr()
            arr_obj = _select_zarr_array(zobj)
            if arr_obj is None and zarr is not None:
                arr_obj = _select_zarr_array(zarr.open(zobj, mode="r"))
            if arr_obj is None:
                raise RuntimeError("could not open zarr view")
            shape = tuple(level.shape)
            if axes and "Y" in axes and "X" in axes:
                ydim = axes.index("Y")
                xdim = axes.index("X")
            else:
                ydim, xdim = 0, 1
            h = int(shape[ydim])
            w = int(shape[xdim])
            step = max(1, int(math.ceil(max(h, w) / max_side)))
            slicer = [slice(None)] * len(shape)
            slicer[ydim] = slice(0, h, step)
            slicer[xdim] = slice(0, w, step)
            small = np.asarray(arr_obj[tuple(slicer)])
        except Exception:
            arr = level.asarray()
            rgb = _to_uint8_rgb(arr)
            step = max(1, int(math.ceil(max(rgb.shape[:2]) / max_side)))
            small = rgb[::step, ::step]

    if axes and len(axes) == small.ndim and "Y" in axes and "X" in axes:
        order = [axes.index("Y"), axes.index("X")] + [i for i, a in enumerate(axes) if a not in ("Y", "X")]
        small = np.transpose(small, axes=order)
    return _to_uint8_rgb(small), full_h, full_w, "tifffile"


def build_tissue_mask(rgb: np.ndarray, close_radius: int, min_obj_area: int, hole_area: int) -> np.ndarray:
    rgb = _to_uint8_rgb(rgb)
    lab = rgb2lab(rgb)
    chroma = np.sqrt(lab[..., 1] * lab[..., 1] + lab[..., 2] * lab[..., 2])
    darkness = 1.0 - (rgb.astype(np.float32).mean(axis=2) / 255.0)
    try:
        chroma_thr = threshold_otsu(chroma)
    except Exception:
        chroma_thr = float(np.percentile(chroma, 70))
    try:
        dark_thr = threshold_otsu(darkness)
    except Exception:
        dark_thr = float(np.percentile(darkness, 70))
    mask = (chroma > chroma_thr) | (darkness > dark_thr)
    if close_radius > 0:
        mask = binary_closing(mask, disk(close_radius))
        mask = binary_opening(mask, disk(max(1, close_radius // 3)))
    if hole_area > 0:
        mask = remove_small_holes(mask, area_threshold=hole_area)
    if min_obj_area > 0:
        mask = remove_small_objects(mask, min_size=min_obj_area)
    return mask.astype(bool)


def cluster_axis(values: list[float], tolerance: float) -> list[int]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    groups: list[dict] = []
    out = [0] * len(values)
    for idx, value in indexed:
        if not groups or abs(value - groups[-1]["mean"]) > tolerance:
            groups.append({"mean": float(value), "items": [idx]})
        else:
            group = groups[-1]
            group["items"].append(idx)
            group["mean"] = float(np.mean([values[i] for i in group["items"]]))
    for group_idx, group in enumerate(groups, start=1):
        for idx in group["items"]:
            out[idx] = group_idx
    return out


def contour_to_ring(mask: np.ndarray, scale_x: float, scale_y: float, shift_x: float, shift_y: float, max_points: int) -> list[list[float]]:
    contours = find_contours(mask.astype(np.uint8), 0.5)
    if not contours:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        pts = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0], [y0, x0]], dtype=np.float32)
    else:
        pts = max(contours, key=len)
    if len(pts) > max_points:
        step = int(math.ceil(len(pts) / max_points))
        pts = pts[::step]
    ring = [[float(x * scale_x + shift_x), float(y * scale_y + shift_y)] for y, x in pts]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def bbox_from_ring(ring: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 4:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        intersects = ((yi > y) != (yj > y)) and (x < ((xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi))
        if intersects:
            inside = not inside
        j = i
    return inside


def load_shift(path: Path) -> tuple[float, float]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return 0.0, 0.0
    for x_key, y_key in (("x0", "y0"), ("shift_x", "shift_y"), ("x_offset", "y_offset"), ("crop_x", "crop_y")):
        if x_key in data and y_key in data:
            return float(data.get(x_key) or 0.0), float(data.get(y_key) or 0.0)
    bbox = data.get("bbox") or data.get("crop_bbox")
    if isinstance(bbox, list) and len(bbox) >= 2:
        return float(bbox[0] or 0.0), float(bbox[1] or 0.0)
    return 0.0, 0.0


def detect_spots(args) -> tuple[list[dict], dict, np.ndarray, np.ndarray]:
    thumb, full_h, full_w, reader = read_thumbnail(args.image, args.thumbnail_max_side)
    scale_y = full_h / max(1, thumb.shape[0])
    scale_x = full_w / max(1, thumb.shape[1])
    scale_area = scale_x * scale_y
    scaled_close = max(1, int(round(args.close_radius / max(scale_x, scale_y))))
    min_area_thumb = max(1, int(round((full_h * full_w * args.min_spot_area_fraction) / scale_area)))
    hole_area_thumb = max(1, int(round(min_area_thumb * 0.25)))
    mask = build_tissue_mask(thumb, scaled_close, min_area_thumb, hole_area_thumb)
    labelled = label(mask)
    props = regionprops(labelled)

    candidates = []
    max_area_crop = full_h * full_w * args.max_spot_area_fraction
    for prop in props:
        area_crop = float(prop.area * scale_area)
        area_frac = area_crop / max(1.0, full_h * full_w)
        if area_frac < args.min_spot_area_fraction or area_frac > args.max_spot_area_fraction:
            continue
        perimeter = max(float(prop.perimeter), 1.0)
        circularity = float(4.0 * math.pi * prop.area / (perimeter * perimeter))
        solidity = float(getattr(prop, "solidity", 0.0) or 0.0)
        eccentricity = float(getattr(prop, "eccentricity", 1.0) or 1.0)
        if circularity < args.min_circularity and solidity < args.min_solidity:
            continue
        if eccentricity > args.max_eccentricity and circularity < args.min_circularity * 1.25:
            continue
        y_thumb, x_thumb = prop.centroid
        y0, x0, y1, x1 = prop.bbox
        candidates.append(
            {
                "label_thumb": int(prop.label),
                "area_px_crop_est": area_crop,
                "area_fraction": area_frac,
                "centroid_x_crop": float(x_thumb * scale_x),
                "centroid_y_crop": float(y_thumb * scale_y),
                "bbox_crop": [float(x0 * scale_x), float(y0 * scale_y), float(x1 * scale_x), float(y1 * scale_y)],
                "circularity": circularity,
                "solidity": solidity,
                "eccentricity": eccentricity,
                "equivalent_diameter_crop": float(prop.equivalent_diameter * math.sqrt(scale_area)),
            }
        )

    candidates.sort(key=lambda s: (s["centroid_y_crop"], s["centroid_x_crop"]))
    areas = [c["area_px_crop_est"] for c in candidates]
    area_cv = float(np.std(areas) / max(np.mean(areas), 1.0)) if areas else 999.0
    median_diam = float(np.median([c["equivalent_diameter_crop"] for c in candidates])) if candidates else 0.0
    tol = max(1.0, median_diam * args.grid_tolerance_fraction)
    rows = cluster_axis([c["centroid_y_crop"] for c in candidates], tol) if candidates else []
    cols = cluster_axis([c["centroid_x_crop"] for c in candidates], tol) if candidates else []
    row_count = len(set(rows)) if rows else 0
    col_count = len(set(cols)) if cols else 0
    grid_like = (row_count >= 2 and col_count >= 2) or (len(candidates) >= args.min_spots and max(row_count, col_count) >= 2)
    is_tma = len(candidates) >= args.min_spots and area_cv <= args.max_area_cv and grid_like

    shift_x, shift_y = load_shift(args.shift)
    spots = []
    for idx, cand in enumerate(candidates, start=1):
        spot_mask = labelled == cand["label_thumb"]
        ring_full = contour_to_ring(spot_mask, scale_x, scale_y, shift_x, shift_y, args.max_polygon_points)
        if not ring_full:
            continue
        ring_crop = [[p[0] - shift_x, p[1] - shift_y] for p in ring_full]
        row = rows[idx - 1] if idx - 1 < len(rows) else 0
        col = cols[idx - 1] if idx - 1 < len(cols) else 0
        spot_id = f"{args.sample_id}_TMA_{idx:03d}"
        spot_label = f"R{row:02d}C{col:02d}" if row and col else f"spot_{idx:03d}"
        bbox_full = bbox_from_ring(ring_full)
        spot = {
            **cand,
            "spot_id": spot_id,
            "spot_label": spot_label,
            "row": int(row),
            "column": int(col),
            "centroid_x_full": float(cand["centroid_x_crop"] + shift_x),
            "centroid_y_full": float(cand["centroid_y_crop"] + shift_y),
            "bbox_full": [float(v) for v in bbox_full],
            "ring_full": ring_full,
            "ring_crop": ring_crop,
        }
        spots.append(spot)

    if not is_tma:
        spots = []

    parameter_summary = vars(args).copy()
    parameter_summary.update({"image": str(args.image), "objects": str(args.objects), "shift": str(args.shift), "outdir": str(args.outdir)})
    summary = {
        "sample_id": args.sample_id,
        "is_tma": bool(is_tma),
        "candidate_spot_count": int(len(candidates)),
        "spot_count": int(len(spots)),
        "row_count": int(row_count if is_tma else 0),
        "column_count": int(col_count if is_tma else 0),
        "area_cv": area_cv,
        "median_equivalent_diameter_px_crop": median_diam,
        "thumbnail_shape_yx": [int(thumb.shape[0]), int(thumb.shape[1])],
        "image_shape_yx": [int(full_h), int(full_w)],
        "reader": reader,
        "shift_xy": [shift_x, shift_y],
        "parameters": parameter_summary,
    }
    return spots, summary, thumb, mask


def write_geojson(spots: list[dict], summary: dict, out_path: Path) -> None:
    features = []
    for spot in spots:
        props = {k: v for k, v in spot.items() if k not in {"ring_full", "ring_crop"}}
        props["is_tma"] = True
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [spot["ring_full"]]},
                "properties": props,
            }
        )
    payload = {
        "type": "FeatureCollection",
        "name": f"{summary['sample_id']}_tma_spots",
        "features": features,
        "properties": {k: v for k, v in summary.items() if k != "parameters"},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_spot_csv(spots: list[dict], out_path: Path) -> None:
    fields = [
        "spot_id",
        "spot_label",
        "row",
        "column",
        "area_px_crop_est",
        "area_fraction",
        "centroid_x_crop",
        "centroid_y_crop",
        "centroid_x_full",
        "centroid_y_full",
        "equivalent_diameter_crop",
        "circularity",
        "solidity",
        "eccentricity",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for spot in spots:
            writer.writerow({field: spot.get(field, "") for field in fields})


def pick_column(fieldnames: list[str], candidates: Iterable[str]) -> str | None:
    lower_map = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def assign_objects(args, spots: list[dict], summary: dict, out_path: Path) -> dict:
    with args.objects.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    x_col = pick_column(fieldnames, COORD_X_CANDIDATES)
    y_col = pick_column(fieldnames, COORD_Y_CANDIDATES)
    full_x_col = pick_column(fieldnames, FULL_X_CANDIDATES)
    full_y_col = pick_column(fieldnames, FULL_Y_CANDIDATES)
    shift_x, shift_y = summary.get("shift_xy", [0.0, 0.0])

    assignment_fields = [
        "tma_is_tma",
        "tma_spot_id",
        "tma_spot_label",
        "tma_spot_row",
        "tma_spot_column",
        "tma_assignment_method",
    ]
    out_fields = fieldnames + [f for f in assignment_fields if f not in fieldnames]
    assigned = 0
    spot_bboxes = [(spot, bbox_from_ring(spot["ring_crop"])) for spot in spots]
    for row in rows:
        row["tma_is_tma"] = str(bool(summary["is_tma"])).lower()
        row["tma_spot_id"] = ""
        row["tma_spot_label"] = ""
        row["tma_spot_row"] = ""
        row["tma_spot_column"] = ""
        row["tma_assignment_method"] = "none"
        if not spots:
            continue
        try:
            if x_col and y_col:
                x_crop = float(row[x_col])
                y_crop = float(row[y_col])
            elif full_x_col and full_y_col:
                x_crop = float(row[full_x_col]) - float(shift_x)
                y_crop = float(row[full_y_col]) - float(shift_y)
            else:
                continue
        except Exception:
            continue
        for spot, bbox in spot_bboxes:
            x0, y0, x1, y1 = bbox
            if x_crop < x0 or x_crop > x1 or y_crop < y0 or y_crop > y1:
                continue
            if point_in_ring(x_crop, y_crop, spot["ring_crop"]):
                row["tma_spot_id"] = spot["spot_id"]
                row["tma_spot_label"] = spot["spot_label"]
                row["tma_spot_row"] = str(spot["row"])
                row["tma_spot_column"] = str(spot["column"])
                row["tma_assignment_method"] = "point_in_spot_polygon"
                assigned += 1
                break
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "object_count": len(rows),
        "assigned_object_count": assigned,
        "coordinate_columns": {"x": x_col, "y": y_col, "full_x": full_x_col, "full_y": full_y_col},
    }


def save_preview(thumb: np.ndarray, mask: np.ndarray, spots: list[dict], summary: dict, out_path: Path) -> None:
    rgb = _to_uint8_rgb(thumb).copy()
    overlay = rgb.copy()
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([255, 180, 0], dtype=np.float32)).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    full_h, full_w = summary["image_shape_yx"]
    shift_x, shift_y = summary["shift_xy"]
    scale_x = image.width / max(1, full_w)
    scale_y = image.height / max(1, full_h)
    for spot in spots:
        pts = [((x - shift_x) * scale_x, (y - shift_y) * scale_y) for x, y in spot["ring_full"]]
        draw.line(pts, fill=(255, 0, 0), width=3)
        cx = (spot["centroid_x_full"] - shift_x) * scale_x
        cy = (spot["centroid_y_full"] - shift_y) * scale_y
        draw.text((cx + 4, cy + 4), spot["spot_label"], fill=(255, 0, 0))
    image.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect TMA cores and assign StarDist cells to spots.")
    parser.add_argument("--image", type=Path, required=True, help="StarDist crop ROI TIFF.")
    parser.add_argument("--objects", type=Path, required=True, help="StarDist objects.csv.")
    parser.add_argument("--shift", type=Path, required=True, help="StarDist shift.json.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--thumbnail-max-side", type=int, default=2048)
    parser.add_argument("--min-spots", type=int, default=4)
    parser.add_argument("--min-spot-area-fraction", type=float, default=0.001)
    parser.add_argument("--max-spot-area-fraction", type=float, default=0.25)
    parser.add_argument("--max-area-cv", type=float, default=0.75)
    parser.add_argument("--min-circularity", type=float, default=0.25)
    parser.add_argument("--min-solidity", type=float, default=0.50)
    parser.add_argument("--max-eccentricity", type=float, default=0.98)
    parser.add_argument("--grid-tolerance-fraction", type=float, default=0.75)
    parser.add_argument("--close-radius", type=int, default=48)
    parser.add_argument("--max-polygon-points", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    spots, summary, thumb, mask = detect_spots(args)
    assignment = assign_objects(args, spots, summary, args.outdir / f"{args.sample_id}_objects_tma_assigned.csv")
    summary.update(assignment)
    write_geojson(spots, summary, args.outdir / f"{args.sample_id}_tma_spots.geojson")
    write_spot_csv(spots, args.outdir / f"{args.sample_id}_tma_spots.csv")
    save_preview(thumb, mask, spots, summary, args.outdir / f"{args.sample_id}_tma_spots_preview.png")
    (args.outdir / f"{args.sample_id}_tma_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"sample_id": args.sample_id, "is_tma": summary["is_tma"], "spot_count": summary["spot_count"], "assigned_object_count": summary["assigned_object_count"]}))


if __name__ == "__main__":
    main()
