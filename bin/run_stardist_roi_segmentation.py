#!/usr/bin/env python3
"""
roi_crop_and_stardist_segment.py

MERGED PIPELINE (from your two scripts):
1) Crop a pyramidal OME-TIFF (level 0) to the *union bounding box* across ALL polygons in an ROI GeoJSON.
2) Run StarDist cell segmentation over the entire cropped ROI rectangle (includes cells not inside any polygon).
3) Write:
   - crop_roi.tif
   - ROI GeoJSON in crop coordinates (shifted)
   - shift.json (offset info to map crop<->original)
   - labels.tif + objects.csv
   - segmentation polygons GeoJSON in crop coords AND in original-image coords
   - QC overlays

Usage example:
  python roi_crop_and_stardist_segment.py \
    --in Data/Visium_HD_Human_Colon_Cancer_tissue_image.ome.tif \
    --roi Data/ROI.geojson \
    --outdir out_stardist_roi \
    --model 2D_versatile_he \
    --prob 0.48 --nms 0.30

Notes:
- ROI GeoJSON coordinates must be in the SAME pixel coordinate system as the input image (x,y).
- Crop is always the ROI bbox (fast to read from WSI). Segmentation is restricted to the actual ROI polygon.
"""

import argparse
import json
import math
import io
import contextlib
import os
import subprocess
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
import tifffile

def write_full_labels_tif(full_out: Path, labels_crop: np.ndarray, x0: int, y0: int, full_w: int, full_h: int) -> None:
    """Write a full-resolution label image (BigTIFF) without allocating the full array in RAM."""
    import tifffile
    full_out.parent.mkdir(parents=True, exist_ok=True)
    # BigTIFF is required for very large images; memmap writes uncompressed but avoids huge RAM usage.
    mm = tifffile.memmap(str(full_out), shape=(full_h, full_w), dtype=np.uint32, bigtiff=True)
    mm[:] = 0
    h, w = labels_crop.shape
    mm[y0:y0+h, x0:x0+w] = labels_crop.astype(np.uint32, copy=False)
    mm.flush()


def write_full_labels_zarr(full_out: Path, labels_crop: np.ndarray, x0: int, y0: int, full_w: int, full_h: int, chunks=(4096, 4096)) -> None:
    """Write a full-resolution label image as chunked Zarr (sparse-friendly)."""
    import zarr
    full_out.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(zarr, "DirectoryStore"):
        store = zarr.DirectoryStore(str(full_out))
    else:
        store = zarr.storage.DirectoryStore(str(full_out))
    root = zarr.group(store=store, overwrite=True)
    compressor = None
    try:
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
    except Exception:
        compressor = None
    arr = root.create_dataset(
        "labels",
        shape=(full_h, full_w),
        chunks=chunks,
        dtype="uint32",
        compressor=compressor,
        overwrite=True,
    )
    h, w = labels_crop.shape
    arr[y0:y0+h, x0:x0+w] = labels_crop.astype(np.uint32, copy=False)

import zarr

STARDIST_AVAILABLE = True
STARDIST_IMPORT_ERROR = None
try:
    # IMPORTANT: import StarDist early (per Squidpy tutorial)
    from stardist.models import StarDist2D
except Exception as e:
    STARDIST_AVAILABLE = False
    STARDIST_IMPORT_ERROR = e
    StarDist2D = None

def _local_stardist_candidates(model_name: str) -> List[Tuple[str, str]]:
    """
    Return local StarDist model candidates as (basedir, name).
    StarDist typically stores pretrained models under:
      $KERAS_HOME/models/StarDist2D/<model_name>
    """
    candidates: List[Tuple[str, str]] = []
    keras_home = (os.environ.get("KERAS_HOME") or "").strip()
    if not keras_home:
        return candidates

    root = Path(keras_home) / "models" / "StarDist2D"
    names = [model_name]
    if not model_name.startswith("python_"):
        names.append(f"python_{model_name}")

    for name in names:
        model_dir = root / name
        if model_dir.is_dir():
            candidates.append((str(root), name))
    return candidates


def load_stardist_model_filtered(model_name: str):
    """
    Load a pretrained StarDist model but suppress the line:
      'Using default values: prob_thresh=..., nms_thresh=...'
    Those values are the model's recommended defaults from thresholds.json and are NOT used
    if you pass your own prob_thresh/nms_thresh to predict_instances().
    """
    if not STARDIST_AVAILABLE:
        raise RuntimeError(f"StarDist import failed: {STARDIST_IMPORT_ERROR}")

    # First try explicit local cache paths (offline-safe).
    local_errors: List[str] = []
    for basedir, name in _local_stardist_candidates(model_name):
        try:
            log(f"Trying local StarDist model: basedir={basedir} name={name}")
            return StarDist2D(None, name=name, basedir=basedir)
        except Exception as exc:
            local_errors.append(f"{basedir}/{name}: {exc}")

    # Fallback to StarDist pretrained resolver (may download if cache is incomplete).
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        model = StarDist2D.from_pretrained(model_name)

    captured = (buf_out.getvalue() + "\n" + buf_err.getvalue()).splitlines()
    for line in captured:
        if "Using default values:" in line:
            continue
        if line.strip() == "":
            continue
        print(line, flush=True)
    if local_errors:
        log("Local StarDist candidates were found but could not be loaded:")
        for err in local_errors:
            log(f"  - {err}")
    return model

if STARDIST_AVAILABLE:
    from stardist.plot import render_label
else:
    def render_label(labels: np.ndarray, img: Optional[np.ndarray] = None) -> np.ndarray:
        """Fallback overlay when stardist.plot is unavailable."""
        if img is None:
            base = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.float32)
        else:
            arr = np.asarray(img)
            if arr.ndim == 2:
                base = np.repeat(arr[..., None], 3, axis=2)
            elif arr.ndim == 3 and arr.shape[-1] == 1:
                base = np.repeat(arr, 3, axis=2)
            else:
                base = arr[..., :3]
            base = base.astype(np.float32, copy=False)
            vmax = float(np.max(base)) if base.size else 1.0
            if vmax > 1.0:
                base = base / vmax
            base = np.clip(base, 0.0, 1.0)
        b = find_boundaries(labels, mode="outer")
        out = base.copy()
        out[b] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return out

import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from skimage.draw import polygon as sk_polygon
from skimage.measure import regionprops_table, find_contours
from skimage.segmentation import find_boundaries

try:
    from skimage.segmentation import relabel_sequential
except Exception:  # fallback
    relabel_sequential = None


# ------------------------- tiny utils -------------------------

def die(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)

def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


# ------------------------- GeoJSON helpers -------------------------

def _iter_coords(geo: Dict[str, Any]) -> Iterable[Tuple[float, float]]:
    """Yield (x,y) coordinates from Polygon/MultiPolygon GeoJSON, Feature, or FeatureCollection."""
    if not geo:
        return
    gtype = geo.get("type", "")
    if gtype == "FeatureCollection":
        for f in geo.get("features", []):
            yield from _iter_coords(f.get("geometry", {}))
    elif gtype == "Feature":
        yield from _iter_coords(geo.get("geometry", {}))
    elif gtype == "GeometryCollection":
        for g in geo.get("geometries", []):
            yield from _iter_coords(g)
    elif gtype == "Polygon":
        for ring in geo.get("coordinates", []):
            for x, y in ring:
                yield float(x), float(y)
    elif gtype == "MultiPolygon":
        for poly in geo.get("coordinates", []):
            for ring in poly:
                for x, y in ring:
                    yield float(x), float(y)

def roi_bbox(geo: Dict[str, Any], W: int, H: int) -> Tuple[int, int, int, int]:
    xs, ys = [], []
    for x, y in _iter_coords(geo):
        xs.append(x)
        ys.append(y)
    if not xs:
        die("No coordinates found in ROI GeoJSON.")
    x0 = int(np.floor(min(xs))); x1 = int(np.ceil(max(xs)))
    y0 = int(np.floor(min(ys))); y1 = int(np.ceil(max(ys)))

    # clamp
    x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
    if x1 <= x0 or y1 <= y0:
        die(f"ROI bbox is empty after clamping: ({x0},{y0})-({x1},{y1})")
    return x0, y0, x1, y1

def _shift_geometry(geom: Dict[str, Any], dx: float, dy: float) -> Dict[str, Any]:
    """Shift a geometry dict by (dx, dy) in-place style but returns a new dict."""
    gtype = geom.get("type", "")
    if gtype == "Polygon":
        out = {"type": "Polygon", "coordinates": []}
        for ring in geom.get("coordinates", []):
            out["coordinates"].append([[float(x) + dx, float(y) + dy] for x, y in ring])
        return out
    if gtype == "MultiPolygon":
        out = {"type": "MultiPolygon", "coordinates": []}
        for poly in geom.get("coordinates", []):
            poly_out = []
            for ring in poly:
                poly_out.append([[float(x) + dx, float(y) + dy] for x, y in ring])
            out["coordinates"].append(poly_out)
        return out
    if gtype == "GeometryCollection":
        return {
            "type": "GeometryCollection",
            "geometries": [_shift_geometry(g, dx, dy) for g in geom.get("geometries", [])],
        }
    # pass-through for unsupported
    return geom

def shift_geojson(geo: Dict[str, Any], dx: float, dy: float) -> Dict[str, Any]:
    """Shift a GeoJSON Feature/FeatureCollection/Geometry by (dx, dy)."""
    gtype = geo.get("type", "")
    if gtype == "FeatureCollection":
        feats = []
        for f in geo.get("features", []):
            feats.append(shift_geojson(f, dx, dy))
        return {"type": "FeatureCollection", "features": feats}
    if gtype == "Feature":
        out = dict(geo)
        out["geometry"] = _shift_geometry(geo.get("geometry", {}), dx, dy)
        return out
    # geometry
    return _shift_geometry(geo, dx, dy)

def _iter_polygons_rings(geo: Dict[str, Any]) -> Iterable[List[List[List[float]]]]:
    """
    Yield polygon ring lists:
      - Polygon: yields [ring0, ring1, ...]
      - MultiPolygon: yields for each poly [ring0, ring1, ...]
    """
    if not geo:
        return
    gtype = geo.get("type", "")
    if gtype == "FeatureCollection":
        for f in geo.get("features", []):
            yield from _iter_polygons_rings(f.get("geometry", {}))
    elif gtype == "Feature":
        yield from _iter_polygons_rings(geo.get("geometry", {}))
    elif gtype == "GeometryCollection":
        for g in geo.get("geometries", []):
            yield from _iter_polygons_rings(g)
    elif gtype == "Polygon":
        rings = geo.get("coordinates", [])
        if rings:
            yield rings
    elif gtype == "MultiPolygon":
        for poly in geo.get("coordinates", []):
            if poly:
                yield poly

def rasterize_roi_mask(geo_polygon_like: Dict[str, Any], H: int, W: int) -> np.ndarray:
    """
    Rasterize GeoJSON polygons into a boolean mask (YX) of shape (H, W).
    Supports holes (inner rings) by subtracting them.
    """
    mask = np.zeros((H, W), dtype=bool)
    for rings in _iter_polygons_rings(geo_polygon_like):
        if not rings:
            continue
        # exterior
        ext = np.asarray(rings[0], dtype=float)
        if ext.ndim != 2 or ext.shape[1] != 2:
            continue
        rr, cc = sk_polygon(ext[:, 1], ext[:, 0], shape=mask.shape)  # (row=y, col=x)
        mask[rr, cc] = True
        # holes
        for hole in rings[1:]:
            h = np.asarray(hole, dtype=float)
            if h.ndim != 2 or h.shape[1] != 2:
                continue
            rr, cc = sk_polygon(h[:, 1], h[:, 0], shape=mask.shape)
            mask[rr, cc] = False
    return mask

# ------------------------- GeoJSON feature/polygon extraction (with properties) -------------------------

def _polygon_area_xy(coords: List[List[float]]) -> float:
    """
    Shoelace area of a ring (x,y). Positive/negative depends on orientation; we return abs.
    Ring may be closed or open.
    """
    if len(coords) < 3:
        return 0.0
    xs = np.array([p[0] for p in coords], dtype=float)
    ys = np.array([p[1] for p in coords], dtype=float)
    # ensure closed
    if xs[0] != xs[-1] or ys[0] != ys[-1]:
        xs = np.r_[xs, xs[0]]
        ys = np.r_[ys, ys[0]]
    return float(0.5 * abs(np.dot(xs[:-1], ys[1:]) - np.dot(xs[1:], ys[:-1])))

def polygon_area_with_holes(rings: List[List[List[float]]]) -> float:
    """Area(exterior) - sum(Area(holes))."""
    if not rings:
        return 0.0
    area = _polygon_area_xy(rings[0])
    for hole in rings[1:]:
        area -= _polygon_area_xy(hole)
    return float(max(area, 0.0))

def polygon_bbox(rings: List[List[List[float]]]) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) over all points in rings."""
    xs, ys = [], []
    for ring in rings:
        for x, y in ring:
            xs.append(float(x)); ys.append(float(y))
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))

def extract_polygons_with_properties(geo: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten a GeoJSON (FeatureCollection/Feature/GeometryCollection/Polygon/MultiPolygon)
    into a list of polygon dicts:
      {"rings": <list of rings>, "properties": <dict>, "feature_index": <int>}
    MultiPolygon becomes multiple entries (one per polygon) sharing the same properties.
    """
    polys: List[Dict[str, Any]] = []

    gtype = geo.get("type", "")
    if gtype == "FeatureCollection":
        for i, f in enumerate(geo.get("features", [])):
            polys.extend(extract_polygons_with_properties(f))
        return polys

    if gtype == "Feature":
        props = geo.get("properties", {}) or {}
        geom = geo.get("geometry", {}) or {}
        gtype2 = geom.get("type", "")
        if gtype2 == "Polygon":
            rings = geom.get("coordinates", []) or []
            if rings:
                polys.append({"rings": rings, "properties": props, "feature_index": geo.get("id", None)})
        elif gtype2 == "MultiPolygon":
            for poly in geom.get("coordinates", []) or []:
                if poly:
                    polys.append({"rings": poly, "properties": props, "feature_index": geo.get("id", None)})
        elif gtype2 == "GeometryCollection":
            # keep same properties for all geometries inside this feature
            for g in geom.get("geometries", []) or []:
                sub = {"type": g.get("type", ""), "coordinates": g.get("coordinates", []), "geometries": g.get("geometries", [])}
                if g.get("type") == "Polygon":
                    polys.append({"rings": g.get("coordinates", []) or [], "properties": props, "feature_index": geo.get("id", None)})
                elif g.get("type") == "MultiPolygon":
                    for poly in g.get("coordinates", []) or []:
                        if poly:
                            polys.append({"rings": poly, "properties": props, "feature_index": geo.get("id", None)})
        return polys

    if gtype == "Polygon":
        rings = geo.get("coordinates", []) or []
        if rings:
            polys.append({"rings": rings, "properties": {}, "feature_index": None})
        return polys

    if gtype == "MultiPolygon":
        for poly in geo.get("coordinates", []) or []:
            if poly:
                polys.append({"rings": poly, "properties": {}, "feature_index": None})
        return polys

    if gtype == "GeometryCollection":
        for g in geo.get("geometries", []) or []:
            polys.extend(extract_polygons_with_properties(g))
        return polys

    return polys

def get_class_name(props: Dict[str, Any]) -> Optional[str]:
    """
    Try to extract a human-readable class label from common GeoJSON annotation schemas.
    (Works with QuPath-like: properties.classification.name)
    """
    c = props.get("classification")
    if isinstance(c, dict):
        name = c.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    for k in ("class", "label", "name", "type"):
        v = props.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def get_class_color(props: Dict[str, Any]) -> Optional[str]:
    c = props.get("classification")
    if isinstance(c, dict):
        col = c.get("color")
        if isinstance(col, (list, tuple)) and len(col) >= 3:
            try:
                r, g, b = int(col[0]), int(col[1]), int(col[2])
                return f"{r},{g},{b}"
            except Exception:
                return None
    return None

def compute_crop_bbox_from_largest_polygon(geo: Dict[str, Any], W: int, H: int, pad_pixels: int = 200) -> Tuple[int, int, int, int, int]:
    """
    Pick the largest polygon by area, then crop to its bbox expanded by a fixed padding.
    Padding is exactly `pad_pixels` on each side (clamped to image bounds).
    Returns (x0, y0, x1, y1, pad_used).
    """
    polys = extract_polygons_with_properties(geo)
    if not polys:
        die("No Polygon/MultiPolygon geometry found in ROI GeoJSON.")
    areas = []
    bbs = []
    for p in polys:
        area = polygon_area_with_holes(p["rings"])
        bb = polygon_bbox(p["rings"])
        areas.append(area)
        bbs.append(bb)

    imax = int(np.argmax(np.array(areas, dtype=float)))
    minx, miny, maxx, maxy = bbs[imax]
    pad_used = int(max(0, pad_pixels))

    x0 = int(math.floor(minx - pad_used))
    y0 = int(math.floor(miny - pad_used))
    x1 = int(math.ceil(maxx + pad_used))
    y1 = int(math.ceil(maxy + pad_used))

    # clamp
    x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))

    if x1 <= x0 or y1 <= y0:
        die(f"Computed crop bbox is empty after clamping: ({x0},{y0})-({x1},{y1})")
    return x0, y0, x1, y1, pad_used


def compute_crop_bbox_from_all_polygons(geo: Dict[str, Any], W: int, H: int, pad_pixels: int = 200) -> Tuple[int, int, int, int, int]:
    """
    Crop bbox that encloses ALL polygons in the GeoJSON, expanded by a fixed padding.
    Padding is exactly `pad_pixels` on each side (clamped to image bounds).
    Returns (x0, y0, x1, y1, pad_used).
    """
    polys = extract_polygons_with_properties(geo)
    if not polys:
        die("No Polygon/MultiPolygon geometry found in ROI GeoJSON.")
    minx, miny = float("inf"), float("inf")
    maxx, maxy = float("-inf"), float("-inf")
    for p in polys:
        bb = polygon_bbox(p["rings"])
        minx = min(minx, bb[0]); miny = min(miny, bb[1])
        maxx = max(maxx, bb[2]); maxy = max(maxy, bb[3])

    pad_used = int(max(0, pad_pixels))

    x0 = int(math.floor(minx - pad_used))
    y0 = int(math.floor(miny - pad_used))
    x1 = int(math.ceil(maxx + pad_used))
    y1 = int(math.ceil(maxy + pad_used))

    # clamp
    x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))

    if x1 <= x0 or y1 <= y0:
        die(f"Computed crop bbox is empty after clamping: ({x0},{y0})-({x1},{y1})")
    return x0, y0, x1, y1, pad_used


# ------------------------- OME-TIFF crop helpers -------------------------

def _aszarr_level0(tf: tifffile.TiffFile):
    """Return (zarr_array, axes) for level-0 of series[0] from an OME-TIFF."""
    if not tf.series:
        raise RuntimeError("No TIFF series found")

    series = tf.series[0]
    base = series.levels[0] if getattr(series, "levels", None) else series

    axes = getattr(base, "axes", None) or getattr(series, "axes", None) or ""
    axes = axes.replace("S", "C")

    zobj = base.aszarr()

    def _select_array(obj):
        if hasattr(obj, "shape") and hasattr(obj, "ndim"):
            return obj
        try:
            if hasattr(obj, "__getitem__"):
                try:
                    sub = obj["0"]
                    if hasattr(sub, "shape"):
                        return sub
                except Exception:
                    pass
                keys = None
                if hasattr(obj, "keys"):
                    try:
                        keys = list(obj.keys())
                    except Exception:
                        keys = None
                if keys:
                    for k in keys:
                        try:
                            sub = obj[k]
                            if hasattr(sub, "shape"):
                                return sub
                        except Exception:
                            continue
        except Exception:
            pass
        return None

    arr = _select_array(zobj)
    if arr is None:
        root = zarr.open(zobj, mode="r")
        arr = _select_array(root)

    if arr is None:
        raise RuntimeError("Could not resolve a zarr array from aszarr() output")

    return arr, axes

def _to_yxc(arr: np.ndarray, axes: str) -> np.ndarray:
    axes = (axes or "").replace("S", "C")
    if not axes:
        if arr.ndim == 2:
            arr = arr[:, :, None]
            axes = "YXC"
        elif arr.ndim == 3:
            axes = "YXC"
        else:
            die(f"Cannot infer axes for ndim={arr.ndim}")

    if len(axes) != arr.ndim:
        while arr.ndim > 3:
            arr = arr[0]
        axes = "YXC" if arr.ndim == 3 else "YX"

    if set(axes) - set("YXC"):
        slicer = []
        kept = []
        for ax in axes:
            if ax in "YXC":
                slicer.append(slice(None))
                kept.append(ax)
            else:
                slicer.append(0)
        arr = arr[tuple(slicer)]
        axes = "".join(kept)

    if axes == "YX":
        arr = arr[:, :, None]
        axes = "YXC"

    if axes != "YXC":
        perm = [axes.index("Y"), axes.index("X"), axes.index("C")]
        arr = np.transpose(arr, perm)

    return arr

def _ensure_rgb(yxc: np.ndarray) -> np.ndarray:
    if yxc.ndim != 3:
        die("Expected YXC array.")
    C = yxc.shape[2]
    if C == 3:
        return yxc
    if C == 1:
        return np.repeat(yxc, 3, axis=2)
    return yxc[:, :, :3]


# ------------------------- StarDist helpers -------------------------

def choose_model(img_yxc: np.ndarray, model: str) -> str:
    if model != "auto":
        return model
    # H&E expects RGB; fluo expects single channel
    return "2D_versatile_he" if img_yxc.shape[-1] == 3 else "2D_versatile_fluo"

def _masked_normalize(img_yxc: np.ndarray, model_name: str, roi_mask: np.ndarray) -> np.ndarray:
    """
    Percentile normalize using only ROI pixels, then zero outside ROI.
    Matches StarDist's percentile-normalization spirit, but ROI-aware.
    """
    x = img_yxc.astype(np.float32)

    if model_name == "2D_versatile_fluo":
        x2 = x[..., 0]
        vals = x2[roi_mask]
        if vals.size == 0:
            die("ROI mask is empty; cannot normalize.")
        p1, p998 = np.percentile(vals, [1, 99.8])
        x_in = (x2 - p1) / max(p998 - p1, 1e-6)
        x_in = np.clip(x_in, 0.0, 1.0)
        x_in[~roi_mask] = 0.0
        return x_in
    else:
        # RGB
        x_in = x.copy()
        for c in range(min(3, x_in.shape[-1])):
            vals = x_in[..., c][roi_mask]
            if vals.size == 0:
                die("ROI mask is empty; cannot normalize.")
            p1, p998 = np.percentile(vals, [1, 99.8])
            x_in[..., c] = (x_in[..., c] - p1) / max(p998 - p1, 1e-6)
        x_in = np.clip(x_in, 0.0, 1.0)
        x_in[~roi_mask, :] = 0.0
        return x_in

def stardist_segment_roi(img_yxc: np.ndarray, roi_mask: np.ndarray,
                         model_name: str, prob_thresh: float, nms_thresh: float,
                         min_area: int = 0,
                         tiles: tuple[int,int] | None = None,
                         no_tiles: bool = False,
                         tile_progress: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    model = load_stardist_model_filtered(model_name)
    # Normalize using ROI-only pixels, then zero outside ROI
    x_in = _masked_normalize(img_yxc, model_name, roi_mask)

    # Use StarDist's built-in tiling (n_tiles). We do not implement tiling ourselves.
    # If user does not specify --tiles/--no-tiles, we ask StarDist to choose via model._guess_n_tiles().
    if bool(no_tiles):
        tiles_use = None
        tiles_source = 'disabled'
    elif tiles is not None:
        tiles_use = (int(tiles[0]), int(tiles[1]))
        tiles_source = 'user'
    else:
        try:
            t = model._guess_n_tiles(x_in)
            tiles_use = tuple(int(v) for v in t) if t is not None else None
        except Exception:
            tiles_use = None
        tiles_source = 'stardist-auto'

    # StarDist expects n_tiles length == img.ndim (e.g. YXC -> 3 values).
    if tiles_use is not None:
        if not isinstance(tiles_use, (tuple, list)):
            tiles_use = tuple(tiles_use)
        if len(tiles_use) == 2 and x_in.ndim == 3:
            # User provided NY NX; keep channels un-tiled.
            tiles_use = (int(tiles_use[0]), int(tiles_use[1]), 1)
        elif len(tiles_use) != x_in.ndim:
            raise ValueError(f"n_tiles must have length {x_in.ndim} for input ndim={x_in.ndim}; got {tiles_use}")
    log(f"Running predict_instances with prob_thresh={prob_thresh}, nms_thresh={nms_thresh}, n_tiles={tiles_use} (source={tiles_source}), tile_progress={tile_progress}")
    labels, details = model.predict_instances(
        x_in,
        prob_thresh=float(prob_thresh),
        nms_thresh=float(nms_thresh),
        n_tiles=tiles_use,
        show_tile_progress=bool(tile_progress),
    )

    labels = labels.astype(np.uint32)

    # hard-filter: keep only objects whose centroid lies in ROI
    props = regionprops_table(labels, properties=("label", "centroid"))
    if len(props.get("label", [])) > 0:
        keep = set()
        for lab, cy, cx in zip(props["label"], props["centroid-0"], props["centroid-1"]):
            iy = int(round(cy))
            ix = int(round(cx))
            if 0 <= iy < roi_mask.shape[0] and 0 <= ix < roi_mask.shape[1] and roi_mask[iy, ix]:
                keep.add(int(lab))
        if keep:
            m = np.isin(labels, list(keep))
            labels = (labels * m).astype(np.uint32)
        else:
            labels[:] = 0

    # Remove very small objects after ROI filtering.
    min_area_use = int(max(0, min_area))
    if min_area_use > 0:
        aprops = regionprops_table(labels, properties=("label", "area"))
        if len(aprops.get("label", [])) > 0:
            keep = set()
            removed = 0
            for lab, area in zip(aprops["label"], aprops["area"]):
                if float(area) >= float(min_area_use):
                    keep.add(int(lab))
                else:
                    removed += 1
            if keep:
                m = np.isin(labels, list(keep))
                labels = (labels * m).astype(np.uint32)
            else:
                labels[:] = 0
            log(f"Applied min-area filter: min_area={min_area_use}px, removed_objects={removed}")

    # relabel sequentially for nicer IDs
    if relabel_sequential is not None:
        labels, _, _ = relabel_sequential(labels)

    return labels.astype(np.uint32), details


def load_precomputed_labels_crop(full_labels_path: str, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Load a crop from a precomputed full-size labels map (TIFF or Zarr)."""
    p = Path(full_labels_path)
    if not p.exists():
        die(f"Precomputed labels not found: {p}")

    if p.is_dir() or str(p).lower().endswith(".zarr"):
        z = zarr.open(str(p), mode="r")
        arr = np.asarray(z[int(y0):int(y1), int(x0):int(x1)])
    else:
        try:
            mm = tifffile.memmap(str(p), mode="r")
            arr = np.asarray(mm[int(y0):int(y1), int(x0):int(x1)])
        except Exception:
            # Some TIFFs are compressed/non-mappable; fallback to regular read.
            full = tifffile.imread(str(p))
            arr = np.asarray(full[int(y0):int(y1), int(x0):int(x1)])

    if arr.ndim != 2:
        die(f"Precomputed labels must be 2D, got {arr.shape}")
    return arr.astype(np.uint32, copy=False)

def labels_to_geojson_polygons(labels: np.ndarray, out_geojson: Path,
                              max_objects: int = 0, offset_x: float = 0.0, offset_y: float = 0.0) -> None:
    """
    Convert each labeled object to GeoJSON polygons (contours).
    Outputs pixel coordinates (x, y). Add (offset_x, offset_y) to shift to original image.
    """
    features = []
    ids = np.unique(labels)
    ids = ids[ids != 0]

    if max_objects and len(ids) > max_objects:
        ids = ids[:max_objects]

    for obj_id in ids:
        mask = (labels == obj_id).astype(np.uint8)
        contours = find_contours(mask, level=0.5)
        if not contours:
            continue

        contours.sort(key=lambda a: a.shape[0], reverse=True)
        c = contours[0]  # (n, 2) in (row=y, col=x)

        ring = [[float(pt[1] + offset_x), float(pt[0] + offset_y)] for pt in c]
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        features.append({
            "type": "Feature",
            "properties": {"id": int(obj_id)},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    out_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features}))

def _downsample_preview_arrays(img_yxc: np.ndarray,
                               labels: np.ndarray,
                               roi_mask: Optional[np.ndarray],
                               max_side: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    h, w = labels.shape
    max_dim = max(h, w)
    if max_side <= 0 or max_dim <= max_side:
        return img_yxc, labels, roi_mask

    step = max(1, int(math.ceil(max_dim / float(max_side))))
    img_ds = img_yxc[::step, ::step] if img_yxc.ndim >= 2 else img_yxc
    labels_ds = labels[::step, ::step]
    roi_ds = roi_mask[::step, ::step] if roi_mask is not None else None
    return img_ds, labels_ds, roi_ds


def save_qc_plots(img_yxc: np.ndarray,
                  labels: np.ndarray,
                  roi_mask: Optional[np.ndarray],
                  outdir: Path,
                  max_side: int = 1600) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    img_show, labels_show, roi_mask_show = _downsample_preview_arrays(img_yxc, labels, roi_mask, int(max_side))
    if img_show.shape[-1] == 1:
        img_show = img_show[..., 0]


    # Avoid matplotlib 'Clipping input data' warnings
    if isinstance(img_show, np.ndarray) and img_show.dtype.kind == 'f':
        img_show = np.clip(img_show, 0.0, 1.0)
    overlay = render_label(labels_show, img=img_show)

    plt.figure()
    plt.imshow(overlay)
    if roi_mask_show is not None:
        # outline ROI mask (cheap): show boundaries in white-ish overlay using alpha
        roi_b = find_boundaries(roi_mask_show.astype(np.uint8), mode="outer")
        plt.imshow(roi_b, alpha=0.4)
    plt.axis("off")
    plt.title("StarDist labels (ROI only)")
    plt.tight_layout()
    plt.savefig(outdir / "overlay.png", dpi=200)
    plt.close()

    b = find_boundaries(labels_show, mode="outer")
    plt.figure()
    plt.imshow(img_show, cmap=None if img_show.ndim == 3 else "gray")
    plt.imshow(b, alpha=0.6)
    if roi_mask_show is not None:
        roi_b = find_boundaries(roi_mask_show.astype(np.uint8), mode="outer")
        plt.imshow(roi_b, alpha=0.4)
    plt.axis("off")
    plt.title("Boundaries + ROI outline")
    plt.tight_layout()
    plt.savefig(outdir / "boundaries.png", dpi=200)
    plt.close()



def write_full_labels_from_crop(labels_crop: Any,
                               full_h: int,
                               full_w: int,
                               x0: int, y0: int, x1: int, y1: int,
                               out_path: Path,
                               fmt: str = "zarr",
                               chunk: int = 2048,
                               compression: str = "zlib",
                               allow_huge_tif: bool = False) -> None:
    """
    Paste crop labels back into a full-size canvas (original image coordinate space).

    For large WSIs, writing a *dense* full TIFF can be impractical. Use fmt='zarr' to
    write a chunked on-disk array and only materialize the crop region.
    """
    crop_shape = getattr(labels_crop, "shape", None)
    if crop_shape is None or len(crop_shape) != 2:
        die(f"labels_crop must be 2D, got shape={crop_shape}")
    if (y1 - y0) != int(crop_shape[0]) or (x1 - x0) != int(crop_shape[1]):
        die("Crop bbox does not match labels shape. "
            f"bbox size={(y1-y0)}x{(x1-x0)} labels={crop_shape}")

    dtype_out = np.uint32
    if isinstance(labels_crop, np.ndarray):
        max_lab = int(labels_crop.max()) if labels_crop.size else 0
        dtype_out = np.uint16 if max_lab <= np.iinfo(np.uint16).max else np.uint32

    full_pixels = int(full_h) * int(full_w)

    fmt = (fmt or "").lower()
    if fmt == "zarr":
        try:
            import zarr
        except Exception as e:
            die(f"zarr is required for --full-format zarr, but import failed: {e}")

        # Create a chunked array with implicit fill_value=0 (unwritten chunks are treated as zero).
        out_path = Path(out_path)
        log(f"[FULL] Writing Zarr full labels: {out_path} (shape={full_h}x{full_w}, chunk={chunk}, dtype={dtype_out})")

        # zarr.open(path) works for both zarr v2/v3; overwrite=True will delete existing store.
        z = zarr.open(
            str(out_path),
            mode="w",
            shape=(int(full_h), int(full_w)),
            chunks=(int(chunk), int(chunk)),
            dtype=dtype_out,
            overwrite=True,
        )
        if isinstance(labels_crop, np.ndarray):
            z[int(y0):int(y1), int(x0):int(x1)] = labels_crop.astype(dtype_out, copy=False)
        else:
            z[int(y0):int(y1), int(x0):int(x1)] = labels_crop
        log("[FULL] Done writing Zarr full labels (only crop region materialized).")
        return

    if fmt == "tif":
        if not isinstance(labels_crop, np.ndarray):
            die("Dense TIFF full-label export requires an in-memory NumPy labels array; use --full-format zarr for large-image mode.")
        # Guard: dense full TIFF can be enormous (both disk and RAM). Refuse by default beyond a threshold.
        # Threshold is conservative; override with --allow-huge-tif if you know what you're doing.
        thresh = 500_000_000  # 0.5 billion pixels
        if full_pixels > thresh and not allow_huge_tif:
            die(
                "Refusing to write dense full TIFF because the full image is very large "
                f"({full_w}x{full_h}={full_pixels:,} pixels). "
                "Use --full-format zarr (recommended), or add --allow-huge-tif to force TIFF."
            )

        log(f"[FULL] Writing dense TIFF full labels: {out_path} (shape={full_h}x{full_w}, dtype={dtype_out})")
        full = np.zeros((int(full_h), int(full_w)), dtype=dtype_out)
        full[int(y0):int(y1), int(x0):int(x1)] = labels_crop.astype(dtype_out, copy=False)

        # BigTIFF is safer; compression reduces disk use but increases CPU.
        tifffile.imwrite(
            out_path,
            full,
            compression=compression,
            bigtiff=True,
        )
        del full
        log("[FULL] Done writing full TIFF labels.")
        return

    die(f"Unknown full label format: {fmt}. Use 'zarr' or 'tif'.")


# ------------------------- large-image helpers -------------------------

def iter_tiff_tiles_2d(arr_like: Any, shape: Tuple[int, int], tile_hw: int) -> Iterable[np.ndarray]:
    h, w = shape
    for y0 in range(0, h, tile_hw):
        y1 = min(h, y0 + tile_hw)
        for x0 in range(0, w, tile_hw):
            x1 = min(w, x0 + tile_hw)
            yield np.asarray(arr_like[y0:y1, x0:x1])


def iter_tiff_tiles_rgb(reader: "CropSourceReader", tile_hw: int) -> Iterable[np.ndarray]:
    h, w, _ = reader.shape
    for y0 in range(0, h, tile_hw):
        y1 = min(h, y0 + tile_hw)
        for x0 in range(0, w, tile_hw):
            x1 = min(w, x0 + tile_hw)
            yield reader.read_raw(y0, y1, x0, x1)


def write_tiled_tiff_2d(out_path: Path, arr_like: Any, shape: Tuple[int, int],
                        dtype: np.dtype, tile_hw: int, compression: str = "zlib") -> None:
    tifffile.imwrite(
        str(out_path),
        data=iter_tiff_tiles_2d(arr_like, shape, tile_hw),
        shape=shape,
        dtype=dtype,
        tile=(int(tile_hw), int(tile_hw)),
        compression=compression,
        bigtiff=True,
        photometric="minisblack",
        metadata=None,
    )


def write_tiled_tiff_rgb(out_path: Path, reader: "CropSourceReader", tile_hw: int,
                         compression: str = "zlib") -> None:
    tifffile.imwrite(
        str(out_path),
        data=iter_tiff_tiles_rgb(reader, tile_hw),
        shape=reader.shape,
        dtype=np.uint8,
        tile=(int(tile_hw), int(tile_hw)),
        compression=compression,
        bigtiff=True,
        photometric="rgb",
        metadata={"axes": "YXC"},
    )


def compute_centroids_streaming_array(mask_arr: Any, shape: Tuple[int, int], block: int = 4096) -> pd.DataFrame:
    h, w = shape
    counts = np.zeros((1,), dtype=np.int64)
    sumx = np.zeros((1,), dtype=np.float64)
    sumy = np.zeros((1,), dtype=np.float64)

    def _ensure_len(arr: np.ndarray, n: int) -> np.ndarray:
        if arr.shape[0] >= n:
            return arr
        out = np.zeros((n,), dtype=arr.dtype)
        out[:arr.shape[0]] = arr
        return out

    for y0 in range(0, h, block):
        y1 = min(h, y0 + block)
        yy = (np.arange(y1 - y0, dtype=np.float64) + y0)[:, None]
        for x0 in range(0, w, block):
            x1 = min(w, x0 + block)
            m = np.asarray(mask_arr[y0:y1, x0:x1])
            if m.size == 0:
                continue
            m = m.astype(np.int64, copy=False)
            mx = int(m.max())
            if mx <= 0:
                continue
            newn = mx + 1
            counts = _ensure_len(counts, newn)
            sumx = _ensure_len(sumx, newn)
            sumy = _ensure_len(sumy, newn)

            flat = m.ravel()
            counts[:newn] += np.bincount(flat, minlength=newn)

            xx = (np.arange(x1 - x0, dtype=np.float64) + x0)[None, :]
            sumx[:newn] += np.bincount(flat, weights=np.broadcast_to(xx, m.shape).ravel(), minlength=newn)
            sumy[:newn] += np.bincount(flat, weights=np.broadcast_to(yy, m.shape).ravel(), minlength=newn)

    labels = np.nonzero(counts[1:] > 0)[0] + 1
    area = counts[labels].astype(np.int64)
    x = (sumx[labels] / counts[labels]).astype(np.float64)
    y = (sumy[labels] / counts[labels]).astype(np.float64)

    df = pd.DataFrame({
        "label": labels.astype(np.int64),
        "area": area,
        "y": y,
        "x": x,
        "xmin": np.nan,
        "ymin": np.nan,
        "xmax": np.nan,
        "ymax": np.nan,
        "eccentricity": np.nan,
        "solidity": np.nan,
    })
    return df


def write_preview_big(reader: "CropSourceReader", labels_arr: Any, shape: Tuple[int, int],
                      outdir: Path, max_side: int = 1600) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    h, w = shape
    step = max(1, int(math.ceil(max(h, w) / float(max_side))))
    img_show = reader.read_raw(0, h, 0, w, y_step=step, x_step=step)
    labels = np.asarray(labels_arr[::step, ::step])
    img_show_f = img_show.astype(np.float32, copy=False) / 255.0

    overlay = render_label(labels, img=img_show_f)
    plt.figure()
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("StarDist labels (large-image preview)")
    plt.tight_layout()
    plt.savefig(outdir / "overlay.png", dpi=200)
    plt.close()

    b = find_boundaries(labels, mode="outer")
    plt.figure()
    plt.imshow(img_show)
    plt.imshow(b, alpha=0.6)
    plt.axis("off")
    plt.title("Boundaries")
    plt.tight_layout()
    plt.savefig(outdir / "boundaries.png", dpi=200)
    plt.close()


class CropSourceReader:
    """Lazy level-0 crop reader that exposes YXC tiles without loading the full crop."""

    def __init__(self, src_arr: Any, src_axes: str, x0: int, y0: int, x1: int, y1: int):
        self.src_arr = src_arr
        self.src_axes = (src_axes or "").replace("S", "C")
        self.x0 = int(x0)
        self.y0 = int(y0)
        self.x1 = int(x1)
        self.y1 = int(y1)

        if self.src_axes:
            try:
                self.ydim = self.src_axes.index("Y")
                self.xdim = self.src_axes.index("X")
            except ValueError:
                self.ydim, self.xdim = 0, 1
        else:
            self.ydim, self.xdim = 0, 1
        self.shape = (self.y1 - self.y0, self.x1 - self.x0, 3)

    def _read_src(self, ys: slice, xs: slice) -> np.ndarray:
        slicer = [slice(None)] * len(self.src_arr.shape)
        slicer[self.ydim] = slice(self.y0 + ys.start, self.y0 + ys.stop, ys.step)
        slicer[self.xdim] = slice(self.x0 + xs.start, self.x0 + xs.stop, xs.step)
        arr = np.asarray(self.src_arr[tuple(slicer)])
        return _ensure_rgb(_to_yxc(arr, self.src_axes))

    def read_raw(self, y0: int, y1: int, x0: int, x1: int,
                 y_step: int = 1, x_step: int = 1) -> np.ndarray:
        return self._read_src(
            slice(int(y0), int(y1), int(max(1, y_step))),
            slice(int(x0), int(x1), int(max(1, x_step))),
        )

    def sample(self, max_side: int = 2048) -> np.ndarray:
        h, w, _ = self.shape
        step = max(1, int(math.ceil(max(h, w) / float(max_side))))
        return self.read_raw(0, h, 0, w, y_step=step, x_step=step)


class NormalizedCropArray:
    """Array-like YXC view for StarDist big prediction."""

    def __init__(self, reader: CropSourceReader, lo: np.ndarray, hi: np.ndarray):
        self.reader = reader
        self.shape = reader.shape
        self.ndim = len(self.shape)
        self.dtype = np.dtype(np.float32)
        self.lo = lo.astype(np.float32, copy=False)
        self.hi = hi.astype(np.float32, copy=False)

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        key = list(key) + [slice(None)] * (3 - len(key))
        ys, xs, cs = key[:3]

        def _norm_slice(s: Any, size: int) -> slice:
            if isinstance(s, slice):
                return slice(
                    0 if s.start is None else int(s.start),
                    size if s.stop is None else int(s.stop),
                    1 if s.step is None else int(s.step),
                )
            i = int(s)
            if i < 0:
                i += size
            return slice(i, i + 1, 1)

        ys = _norm_slice(ys, self.shape[0])
        xs = _norm_slice(xs, self.shape[1])
        tile = self.reader.read_raw(ys.start, ys.stop, xs.start, xs.stop, ys.step, xs.step).astype(np.float32, copy=False)
        for c in range(tile.shape[-1]):
            denom = float(self.hi[c] - self.lo[c]) if float(self.hi[c]) > float(self.lo[c]) else 1.0
            tile[..., c] = np.clip((tile[..., c] - self.lo[c]) / denom, 0.0, 1.0)
        return tile[..., cs]


def estimate_rgb_percentiles(reader: CropSourceReader, p_lo: float = 1.0, p_hi: float = 99.8,
                             max_side: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    sample = reader.sample(max_side=max_side).astype(np.float32, copy=False)
    lo = np.zeros((3,), dtype=np.float32)
    hi = np.zeros((3,), dtype=np.float32)
    for c in range(3):
        vals = sample[..., c].ravel()
        lo[c] = np.percentile(vals, p_lo)
        hi[c] = np.percentile(vals, p_hi)
        if not np.isfinite(lo[c]) or not np.isfinite(hi[c]) or hi[c] <= lo[c]:
            lo[c] = float(vals.min()) if vals.size else 0.0
            hi[c] = float(vals.max()) if vals.size else 255.0
            if hi[c] <= lo[c]:
                hi[c] = lo[c] + 1.0
    return lo, hi


def detect_gpu_memory_gib() -> Optional[float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    first = out.splitlines()[0].strip()
    if not first:
        return None
    try:
        return float(first) / 1024.0
    except Exception:
        return None


def build_big_block_candidates(requested_block_size: int,
                               memory_budget_gb: float = 0.0,
                               gpu_memory_gib: Optional[float] = None) -> List[int]:
    block = int(max(512, requested_block_size))
    if memory_budget_gb > 0:
        if memory_budget_gb <= 8.0:
            block = min(block, 2048)
        elif memory_budget_gb <= 12.0:
            block = min(block, 3072)
    if gpu_memory_gib is not None:
        if gpu_memory_gib <= 12.0:
            block = min(block, 1536)
        elif gpu_memory_gib <= 16.5:
            block = min(block, 2048)
        elif gpu_memory_gib <= 24.0:
            block = min(block, 3072)

    ladder = [block, 3072, 2048, 1536, 1024, 768, 512]
    seen: set[int] = set()
    candidates: List[int] = []
    for value in ladder:
        ivalue = int(max(512, value))
        if ivalue > requested_block_size:
            continue
        if ivalue in seen:
            continue
        seen.add(ivalue)
        candidates.append(ivalue)
    if requested_block_size not in seen:
        candidates.insert(0, int(max(512, requested_block_size)))
    return candidates


def exception_looks_like_oom(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    oom_tokens = (
        "resourceexhausted",
        "resource exhausted",
        "out of memory",
        "oom",
        "allocator",
        "failed to allocate",
        "cuda_error_out_of_memory",
    )
    return any(token in msg for token in oom_tokens)


def stardist_segment_big(reader: CropSourceReader, model_name: str,
                         prob_thresh: float, nms_thresh: float,
                         outdir: Path, block_size: int, min_overlap: int,
                         context: int, n_tiles_big: Optional[Tuple[int, int]],
                         show_progress: bool = True,
                         memory_budget_gb: float = 0.0) -> Tuple[Any, Dict[str, Any]]:
    lo, hi = estimate_rgb_percentiles(reader)
    log(f"Large-image normalization p1/p99.8: lo={lo.tolist()} hi={hi.tolist()}")
    x_in = NormalizedCropArray(reader, lo=lo, hi=hi)
    gpu_memory_gib = detect_gpu_memory_gib()
    block_candidates = build_big_block_candidates(
        requested_block_size=int(block_size),
        memory_budget_gb=float(max(0.0, memory_budget_gb)),
        gpu_memory_gib=gpu_memory_gib,
    )
    if block_candidates and block_candidates[0] != int(block_size):
        log(
            "Auto-adjusted large-image StarDist block size from "
            f"{int(block_size)} to {block_candidates[0]} "
            f"(task_memory_gb={float(max(0.0, memory_budget_gb)):.1f}, gpu_memory_gib={gpu_memory_gib if gpu_memory_gib is not None else 'unknown'})."
        )

    last_exc: Optional[BaseException] = None
    for idx, block_size_try in enumerate(block_candidates):
        model = load_stardist_model_filtered(model_name)
        if not hasattr(model, "predict_instances_big"):
            die("Installed StarDist runtime does not provide predict_instances_big(), which is required for very large crops.")

        labels_tmp = Path(tempfile.mkdtemp(prefix="labels_big_", dir=str(outdir)))
        z = zarr.open(
            str(labels_tmp),
            mode="w",
            shape=(reader.shape[0], reader.shape[1]),
            chunks=(1024, 1024),
            dtype=np.uint32,
        )

        kwargs: Dict[str, Any] = {
            "axes": "YXC",
            "block_size": int(block_size_try),
            "min_overlap": int(min_overlap),
            "context": int(context),
            "labels_out": z,
            "labels_out_dtype": np.uint32,
            "show_progress": bool(show_progress),
            "prob_thresh": float(prob_thresh),
            "nms_thresh": float(nms_thresh),
            "show_tile_progress": False,
        }
        if n_tiles_big is not None:
            kwargs["n_tiles"] = (int(n_tiles_big[0]), int(n_tiles_big[1]), 1)

        try:
            labels_out, details = model.predict_instances_big(x_in, **kwargs)
            return labels_out, {"tmp_labels_store": str(labels_tmp), "details": details, "block_size": int(block_size_try)}
        except Exception as exc:
            last_exc = exc
            shutil.rmtree(labels_tmp, ignore_errors=True)
            if not exception_looks_like_oom(exc) or idx == (len(block_candidates) - 1):
                raise
            next_block = block_candidates[idx + 1]
            log(
                "StarDist large-image inference hit an allocation failure with "
                f"block_size={int(block_size_try)}; retrying with smaller block_size={int(next_block)}."
            )
            try:
                import tensorflow as tf  # type: ignore
                tf.keras.backend.clear_session()
            except Exception:
                pass

    assert last_exc is not None
    raise last_exc


def estimate_in_memory_stardist_gib(crop_h: int,
                                    crop_w: int,
                                    full_h: int,
                                    full_w: int,
                                    write_full_labels: bool,
                                    full_format: str) -> float:
    pixels = float(crop_h) * float(crop_w)
    total_bytes = (
        pixels * 3.0 +   # RGB crop uint8
        pixels * 4.0 +   # labels uint32
        pixels * 1.0 +   # ROI mask / booleans
        pixels * 12.0    # StarDist / QC headroom
    )
    if write_full_labels and str(full_format).strip().lower() == "tif":
        total_bytes += float(full_h) * float(full_w) * 4.0
    return total_bytes / (1024.0 ** 3)


# ------------------------- main -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Crop OME-TIFF to the LARGEST ROI polygon bbox, then StarDist-segment only inside ROI mask.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input OME-TIFF (can be pyramidal)")
    ap.add_argument("--roi", required=True, help="ROI GeoJSON in original image pixel coords (x,y)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--crop-name", default="crop_roi.tif", help="Name for the cropped TIFF (inside outdir)")

    ap.add_argument("--model", default="auto",
                    help="auto | 2D_versatile_he | 2D_versatile_fluo | 2D_paper_dsb2018 | 2D_demo")
    ap.add_argument("--prob", type=float, default=0.48, help="StarDist prob_thresh (higher = fewer objects)")
    ap.add_argument("--nms", type=float, default=0.30, help="StarDist nms_thresh")
    ap.add_argument("--min-area", type=int, default=0,
                    help="Remove segmented objects with area < min-area pixels after ROI filtering.")
    # StarDist built-in tiling controls (handled internally by StarDist)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--tiles", type=int, nargs=2, metavar=("NY","NX"), default=None,
                   help="Let StarDist tile the image using n_tiles=(NY,NX,1) for RGB (YXC). Example: --tiles 8 8")
    g.add_argument("--no-tiles", action="store_true", default=False,
                   help="Disable tiling (pass n_tiles=None). Note: StarDist's built-in progress bar requires tiling.")
    ap.add_argument("--tile-progress", dest="tile_progress", action="store_true", default=True,
                    help="Show StarDist tile progress bar (when tiling is enabled). Default: on.")
    ap.add_argument("--no-tile-progress", dest="tile_progress", action="store_false",
                    help="Disable StarDist tile progress bar.")
    ap.add_argument("--write-full-labels", dest="write_full_labels", action="store_true", default=False,
                    help="Also write a full-size label map in original image coordinates by pasting the crop labels "
                         "into a (H0,W0) canvas. Default off (recommended for huge WSIs).")
    ap.add_argument("--full-format", choices=("zarr", "tif"), default="zarr",
                    help="Format for full labels when --write-full-labels. 'zarr' is strongly recommended for large images.")
    ap.add_argument("--full-out", default=None,
                    help="Output path for full labels when --write-full-labels. "
                         "Default: <outdir>/labels_full.(zarr|tif) depending on --full-format.")
    ap.add_argument("--full-chunk", type=int, default=2048,
                    help="Chunk size for Zarr full labels (pixels). Larger chunks = fewer files but higher write bursts. Default 2048.")
    ap.add_argument("--allow-huge-tif", action="store_true", default=False,
                    help="Allow writing a huge full-size TIFF (may require enormous disk/RAM). If not set, the script will refuse very large TIFFs.")
    ap.add_argument("--roi-mask-max-mpix", type=float, default=250.0,
                    help="Rasterize the true ROI mask only when the crop area is at or below this size in megapixels. Larger crops use a rectangular ROI for memory safety.")
    ap.add_argument("--big-mode", choices=("auto", "never", "always"), default="auto",
                    help="Use StarDist's blockwise large-image inference path. 'auto' enables it only for very large crops.")
    ap.add_argument("--big-threshold-mpix", type=float, default=300.0,
                    help="In --big-mode auto, switch to large-image inference at or above this crop size (MPix).")
    ap.add_argument("--big-block-size", type=int, default=4096,
                    help="Block size passed to StarDist predict_instances_big.")
    ap.add_argument("--big-min-overlap", type=int, default=128,
                    help="Guaranteed overlap between neighboring large-image blocks. Should exceed cell size.")
    ap.add_argument("--big-context", type=int, default=128,
                    help="Context margin discarded from each large-image block.")
    ap.add_argument("--big-tiles", type=int, nargs=2, metavar=("NY", "NX"), default=(4, 4),
                    help="Per-block StarDist tiling used inside large-image inference.")
    ap.add_argument("--big-preview-max-side", type=int, default=1600,
                    help="Maximum side length for QC previews in large-image mode.")
    ap.add_argument("--memory-budget-gb", type=float, default=0.0,
                    help="Optional task memory budget. In --big-mode auto, switch to blockwise inference when the in-memory estimate would consume too much of this budget.")
    ap.add_argument("--big-tiff-tile", type=int, default=1024,
                    help="Tile size for streamed TIFF writing in large-image mode (multiple of 16 recommended).")
    ap.add_argument("--precomputed-labels-full", default="",
                    help="Optional full-size labels map used as fallback when StarDist/TensorFlow is unavailable.")
    ap.add_argument("--write-polygons", action="store_true", default=False,
                    help="Write segmentation polygons GeoJSON (VERY slow on large WSIs). Default: off.")
    ap.add_argument("--max-polygons", type=int, default=0,
                    help="Limit polygons written when --write-polygons (0 = no limit)")
    ap.add_argument("--pad", type=int, default=200, help="Fixed padding (pixels) added on each side of the ROI crop bbox (default 200).")

    args = ap.parse_args()

    in_path = Path(args.in_path)
    roi_path = Path(args.roi)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        die(f"Input not found: {in_path}")
    if not roi_path.exists():
        die(f"ROI not found: {roi_path}")

    roi_geo = json.loads(roi_path.read_text())

    # ---- 1) crop bbox from level0 ----
    with tifffile.TiffFile(in_path) as tf:
        zarr_arr, axes = _aszarr_level0(tf)
        shape = zarr_arr.shape

        if axes:
            axes2 = axes.replace("S", "C")
            try:
                ydim = axes2.index("Y")
                xdim = axes2.index("X")
            except ValueError:
                ydim, xdim = 0, 1
        else:
            ydim, xdim = 0, 1

        H0 = int(shape[ydim]); W0 = int(shape[xdim])
        log(f"Level0 size: {W0}x{H0}")

        x0, y0, x1, y1, pad_used = compute_crop_bbox_from_all_polygons(roi_geo, W0, H0, pad_pixels=args.pad)
        crop_h = int(y1 - y0)
        crop_w = int(x1 - x0)
        log(f"Crop bbox from ALL polygons: origin=({x0},{y0}) size={crop_w}x{crop_h} pad={pad_used}px")

        crop_reader = CropSourceReader(zarr_arr, axes, x0, y0, x1, y1)
        crop_tif = outdir / args.crop_name

        # ---- 2) create ROI GeoJSON in crop coords + mask policy ----
        roi_all_crop_geo = shift_geojson(roi_geo, dx=-x0, dy=-y0)
        (outdir / "roi_all_crop.geojson").write_text(json.dumps(roi_all_crop_geo))
        (outdir / "roi_all_original.geojson").write_text(json.dumps(roi_geo))

        crop_mpix = (float(crop_h) * float(crop_w)) / 1_000_000.0
        big_mode = str(args.big_mode).strip().lower()
        inmem_est_gib = estimate_in_memory_stardist_gib(
            crop_h=crop_h,
            crop_w=crop_w,
            full_h=H0,
            full_w=W0,
            write_full_labels=bool(args.write_full_labels),
            full_format=str(args.full_format),
        )
        budget_gib = float(max(0.0, args.memory_budget_gb))
        budget_forces_big = budget_gib > 0.0 and inmem_est_gib >= (budget_gib * 0.25)
        use_big = big_mode == "always" or (
            big_mode == "auto" and (
                crop_mpix >= float(args.big_threshold_mpix) or budget_forces_big
            )
        )

        roi_mask: Optional[np.ndarray] = None
        img_yxc: Optional[np.ndarray] = None
        labels_arr: Any = None
        labels_np: Optional[np.ndarray] = None
        details: Dict[str, Any] = {}
        labels_tmp_store: Optional[Path] = None

        if use_big:
            reasons = []
            if crop_mpix >= float(args.big_threshold_mpix):
                reasons.append(f"crop_mpix={crop_mpix:.1f} >= threshold={float(args.big_threshold_mpix):.1f}")
            if budget_forces_big:
                reasons.append(f"inmem_estimate={inmem_est_gib:.1f} GiB exceeds 25% of budget={budget_gib:.1f} GiB")
            if not reasons:
                reasons.append(f"big_mode={big_mode}")
            log(
                "Using blockwise large-image StarDist path for crop "
                f"{crop_mpix:.1f} MPix ({'; '.join(reasons)})."
            )
        elif crop_mpix > float(args.roi_mask_max_mpix):
            log(
                f"Crop is large ({crop_mpix:.1f} MPix) but large-image mode is disabled; "
                f"continuing with the in-memory path may require substantial RAM (estimate {inmem_est_gib:.1f} GiB)."
            )

        if not use_big:
            slicer = [slice(None)] * len(shape)
            slicer[ydim] = slice(y0, y1)
            slicer[xdim] = slice(x0, x1)
            arr = np.asarray(zarr_arr[tuple(slicer)])
            img_yxc = _ensure_rgb(_to_yxc(arr, axes))
            tifffile.imwrite(crop_tif, img_yxc, photometric="rgb", metadata={"axes": "YXC"})
            log(f"Wrote crop TIFF: {crop_tif}")

            if crop_mpix <= float(args.roi_mask_max_mpix):
                roi_mask = rasterize_roi_mask(roi_all_crop_geo, crop_h, crop_w)
                if not roi_mask.any():
                    log("Rasterized ROI mask is empty after crop shift; falling back to rectangular crop ROI.")
                    roi_mask = np.ones((crop_h, crop_w), dtype=bool)
                else:
                    log(f"Rasterized ROI mask for crop ({crop_mpix:.1f} MPix).")
            else:
                log(
                    f"Skipping ROI rasterization for large crop ({crop_mpix:.1f} MPix > {float(args.roi_mask_max_mpix):.1f} MPix); "
                    "using rectangular crop ROI for memory safety."
                )
                roi_mask = np.ones((crop_h, crop_w), dtype=bool)
            tifffile.imwrite(outdir / "roi_mask.tif", roi_mask.astype(np.uint8) * 255, compression="zlib")
        else:
            log("Deferring crop TIFF creation until after blockwise StarDist inference.")

        # ---- 3) save shift info for future mapping ----
        shift_info = {
            "input_image": str(in_path),
            "roi_input_geojson": str(roi_path),
            "crop_tif": str(crop_tif),
            "crop_bbox_xyxy": {"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1)},
            "pad_pixels_used": int(pad_used),
            "offset_crop_to_original": {"dx": int(x0), "dy": int(y0)},
            "crop_size": {"width": int(crop_w), "height": int(crop_h)},
            "full_size": {"width": int(W0), "height": int(H0)},
        }
        (outdir / "shift.json").write_text(json.dumps(shift_info, indent=2))
        log(f"Wrote shift info: {outdir / 'shift.json'}")

        # ---- 4) StarDist segmentation ----
        model_name = "2D_versatile_he" if str(args.model) == "auto" else str(args.model)
        log(f"Using StarDist model: {model_name}")

        if STARDIST_AVAILABLE and use_big:
            labels_arr, big_meta = stardist_segment_big(
                reader=crop_reader,
                model_name=model_name,
                prob_thresh=args.prob,
                nms_thresh=args.nms,
                outdir=outdir,
                block_size=args.big_block_size,
                min_overlap=args.big_min_overlap,
                context=args.big_context,
                n_tiles_big=None if args.no_tiles else tuple(args.big_tiles),
                show_progress=True,
                memory_budget_gb=budget_gib,
            )
            labels_tmp_store = Path(big_meta["tmp_labels_store"])
            details = big_meta["details"] if isinstance(big_meta.get("details"), dict) else {"details": str(big_meta.get("details"))}
        elif STARDIST_AVAILABLE:
            assert img_yxc is not None and roi_mask is not None
            labels_np, details = stardist_segment_roi(
                img_yxc=img_yxc,
                roi_mask=roi_mask,
                model_name=args.model,
                prob_thresh=args.prob,
                nms_thresh=args.nms,
                min_area=args.min_area,
                tiles=args.tiles,
                no_tiles=args.no_tiles,
                tile_progress=args.tile_progress,
            )
            labels_arr = labels_np
        else:
            if not args.precomputed_labels_full:
                die(
                    "StarDist is unavailable (TensorFlow missing) and no fallback labels were provided. "
                    "Set --precomputed-labels-full to a full labels TIFF/Zarr."
                )
            log(
                f"StarDist unavailable ({STARDIST_IMPORT_ERROR}); "
                f"loading fallback labels from {args.precomputed_labels_full}"
            )
            labels_np = load_precomputed_labels_crop(args.precomputed_labels_full, x0, y0, x1, y1)
            if roi_mask is None:
                roi_mask = np.ones((crop_h, crop_w), dtype=bool)
            labels_np[~roi_mask] = 0
            if int(max(0, args.min_area)) > 0:
                aprops = regionprops_table(labels_np, properties=("label", "area"))
                if len(aprops.get("label", [])) > 0:
                    keep = set()
                    removed = 0
                    for lab, area in zip(aprops["label"], aprops["area"]):
                        if float(area) >= float(int(max(0, args.min_area))):
                            keep.add(int(lab))
                        else:
                            removed += 1
                    if keep:
                        m = np.isin(labels_np, list(keep))
                        labels_np = (labels_np * m).astype(np.uint32)
                    else:
                        labels_np[:] = 0
                    log(f"Applied min-area filter on precomputed labels: min_area={int(max(0, args.min_area))}px, removed_objects={removed}")
            if relabel_sequential is not None:
                labels_np, _, _ = relabel_sequential(labels_np)
            labels_arr = labels_np
            details = {
                "fallback": "precomputed_labels_full",
                "source": str(Path(args.precomputed_labels_full).resolve())
            }

        # ---- 5) materialize outputs in pipeline-compatible formats ----
        if use_big:
            assert labels_arr is not None
            write_tiled_tiff_rgb(crop_tif, crop_reader, tile_hw=int(args.big_tiff_tile))
            log(f"Wrote streamed crop TIFF: {crop_tif}")
            write_tiled_tiff_2d(
                outdir / "labels.tif",
                labels_arr,
                shape=(crop_h, crop_w),
                dtype=np.uint32,
                tile_hw=int(args.big_tiff_tile),
                compression="zlib",
            )
            log(f"Wrote streamed labels TIFF: {outdir / 'labels.tif'}")
            df = compute_centroids_streaming_array(labels_arr, shape=(crop_h, crop_w), block=max(1024, int(args.big_tiff_tile)))
            write_preview_big(crop_reader, labels_arr, (crop_h, crop_w), outdir, max_side=int(args.big_preview_max_side))
            if args.write_full_labels:
                fmt = str(args.full_format).strip().lower()
                out_full = Path(args.full_out) if args.full_out else (outdir / f"labels_full.{fmt}")
                if fmt == "zarr":
                    write_full_labels_from_crop(
                        labels_crop=labels_arr,
                        full_h=H0,
                        full_w=W0,
                        x0=x0, y0=y0, x1=x1, y1=y1,
                        out_path=out_full,
                        fmt=fmt,
                        chunk=args.full_chunk,
                        compression="zlib",
                        allow_huge_tif=args.allow_huge_tif,
                    )
                else:
                    log("[WARN] Skipping dense full-size TIFF labels export in large-image mode. Use --full-format zarr if a later stage requires full-canvas labels.")
        else:
            assert labels_np is not None and img_yxc is not None and roi_mask is not None
            tifffile.imwrite(outdir / "labels.tif", labels_np, compression="zlib")
            if args.write_full_labels:
                fmt = args.full_format
                out_full = Path(args.full_out) if args.full_out else (outdir / f"labels_full.{fmt}")
                write_full_labels_from_crop(
                    labels_crop=labels_np,
                    full_h=H0,
                    full_w=W0,
                    x0=x0, y0=y0, x1=x1, y1=y1,
                    out_path=out_full,
                    fmt=fmt,
                    chunk=args.full_chunk,
                    compression="zlib",
                    allow_huge_tif=args.allow_huge_tif,
                )

            props = regionprops_table(
                labels_np,
                properties=("label", "area", "centroid", "bbox", "eccentricity", "solidity"),
            )
            df = pd.DataFrame(props).rename(columns={
                "centroid-0": "y",
                "centroid-1": "x",
                "bbox-0": "ymin",
                "bbox-1": "xmin",
                "bbox-2": "ymax",
                "bbox-3": "xmax",
            })
            save_qc_plots(img_yxc, labels_np, roi_mask, outdir, max_side=int(args.big_preview_max_side))

        # ---- 6) details / objects / optional polygons ----
        def _jsonable(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {str(k): _jsonable(v) for k, v in obj.items() if str(k) not in {"points", "coord"}}
            if isinstance(obj, (list, tuple)):
                return [_jsonable(v) for v in obj]
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            return str(obj)

        (outdir / "stardist_details.json").write_text(json.dumps(_jsonable(details), indent=2))

        if not df.empty:
            df["x_orig"] = df["x"] + x0
            df["y_orig"] = df["y"] + y0
            if "xmin" in df.columns:
                df["xmin_orig"] = df["xmin"] + x0
                df["xmax_orig"] = df["xmax"] + x0
                df["ymin_orig"] = df["ymin"] + y0
                df["ymax_orig"] = df["ymax"] + y0
        df.to_csv(outdir / "objects.csv", index=False)

        if args.write_polygons:
            if use_big:
                log("[WARN] Polygon GeoJSON export is disabled in large-image mode to avoid loading the full label image into RAM.")
            else:
                assert labels_np is not None
                labels_to_geojson_polygons(labels_np, outdir / "segmentation_polygons_crop.geojson",
                                          max_objects=args.max_polygons, offset_x=0.0, offset_y=0.0)
                labels_to_geojson_polygons(labels_np, outdir / "segmentation_polygons_original.geojson",
                                          max_objects=args.max_polygons, offset_x=float(x0), offset_y=float(y0))
        else:
            log("Skipping polygon GeoJSON export (disabled).")

        if labels_tmp_store is not None:
            shutil.rmtree(labels_tmp_store, ignore_errors=True)
    log(f"[DONE] Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
