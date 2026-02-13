#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import tifffile

from shapely.geometry import shape as shp_shape
from shapely.geometry import mapping
from shapely.geometry import Polygon
from shapely.affinity import scale as shp_scale
from shapely.ops import unary_union
from skimage.measure import find_contours

try:
    import rasterio
    from rasterio.features import shapes
except Exception:
    rasterio = None
    shapes = None


def normalize_mask(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Mask must be 2D. Got shape={arr.shape}")


def read_tiff_page(path: str, page: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found: {path}")
    with tifffile.TiffFile(path) as tf:
        if page < 0 or page >= len(tf.pages):
            raise ValueError(f"--page {page} out of range. This TIFF has {len(tf.pages)} pages.")
        arr = tf.pages[page].asarray()
    return normalize_mask(arr)


def cast_for_rasterio(data: np.ndarray, binary: bool) -> np.ndarray:
    # rasterio supports: int8, uint8, int16, uint16, int32, float32, float64
    if binary:
        return (data > 0).astype(np.uint8)

    mx = int(data.max()) if data.size else 0
    mn = int(data.min()) if data.size else 0
    if mn >= 0 and mx <= 65535:
        return data.astype(np.uint16, copy=False)
    return data.astype(np.int32, copy=False)


def load_group_map(path: str | None) -> dict[int, str]:
    """
    JSON mapping file, value -> classification string.
    Example:
      {"1": "Tumor", "2": "Stroma", "3": "Immune"}
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Group map not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out = {}
    for k, v in obj.items():
        out[int(k)] = str(v)
    return out


def classification_for_value(v: int, group_map: dict[int, str], prefix: str) -> str:
    # Always return something
    return group_map.get(v, f"{prefix}{v}")


def smooth_geom(g, smooth_buffer: float, smooth_passes: int,
                simplify: float, preserve_topology: bool):
    """
    Strong smoothing:
      - repeat buffer(+r) then buffer(-r) multiple times
      - simplify afterwards to reduce vertices / file size
    """
    if g.is_empty:
        return g

    out = g

    # multiple smoothing passes -> heavier smoothing
    if smooth_buffer and smooth_buffer > 0:
        passes = max(1, int(smooth_passes))
        for _ in range(passes):
            out = out.buffer(smooth_buffer, join_style=1, cap_style=1).buffer(
                -smooth_buffer, join_style=1, cap_style=1
            )

    if simplify and simplify > 0:
        out = out.simplify(simplify, preserve_topology=preserve_topology)

    if not out.is_valid:
        out = out.buffer(0)

    return out


def drop_holes(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return type(geom)(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return type(geom)([type(p)(p.exterior) for p in geom.geoms])
    return geom


def apply_scale(geom, scale_x: float, scale_y: float):
    if geom.is_empty:
        return geom
    if scale_x == 1.0 and scale_y == 1.0:
        return geom
    return shp_scale(geom, xfact=scale_x, yfact=scale_y, origin=(0.0, 0.0))


def iter_polygons_from_mask(mask2d: np.ndarray, binary: bool):
    data = cast_for_rasterio(mask2d, binary=binary)

    if rasterio is not None and shapes is not None:
        # Pixel coords: x=col, y=row
        transform = rasterio.Affine(1, 0, 0, 0, 1, 0)

        for geom, val in shapes(data, mask=(data > 0), transform=transform, connectivity=8):
            yield geom, int(val)
        return

    # Fallback when rasterio or its shared libraries are not available:
    # contour extraction with scikit-image.
    if binary:
        values = [1]
        masks = [(data > 0)]
    else:
        values = [int(v) for v in np.unique(data) if int(v) > 0]
        masks = [(data == v) for v in values]

    for val, bin_mask in zip(values, masks):
        contours = find_contours(bin_mask.astype(np.uint8), level=0.5)
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            coords = [(float(p[1]), float(p[0])) for p in contour]
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if poly.geom_type == "Polygon":
                yield mapping(poly), int(val)
            elif poly.geom_type == "MultiPolygon":
                for part in poly.geoms:
                    if not part.is_empty:
                        yield mapping(part), int(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, help="Input mask TIFF (binary or labeled; can be pyramidal/OME-TIFF)")
    ap.add_argument("--page", type=int, default=0, help="TIFF page to read (0 = full-res)")
    ap.add_argument("--out", required=True, help="Output GeoJSON file")
    ap.add_argument("--scale-x", type=float, default=1.0,
                    help="Scale factor for X coordinates (to map from mask space to source-image space).")
    ap.add_argument("--scale-y", type=float, default=1.0,
                    help="Scale factor for Y coordinates (to map from mask space to source-image space).")

    ap.add_argument("--binary", action="store_true", help="Treat mask as binary foreground (mask>0)")
    ap.add_argument("--dissolve", action="store_true",
                    help="Binary only: merge all polygons into one MultiPolygon feature")
    ap.add_argument("--dissolve-by-value", action="store_true",
                    help="Labeled: merge polygons per value into one MultiPolygon per value (best for small GeoJSON).")
    ap.add_argument("--min-area", type=float, default=0.0, help="Drop polygons with area < this (pixel^2)")

    # Strong smoothing controls
    ap.add_argument("--smooth-buffer", type=float, default=0.0,
                    help="Smoothing radius in pixels via buffer+/- (try 4–12 for strong smoothing)")
    ap.add_argument("--smooth-passes", type=int, default=1,
                    help="Repeat smoothing passes (try 2–4 for more smoothing)")
    ap.add_argument("--simplify", type=float, default=0.0,
                    help="Simplify tolerance in pixels (try 2–8 for strong simplification)")
    ap.add_argument("--preserve-topology", action="store_true",
                    help="Topology-preserving simplify (safer)")

    ap.add_argument("--fill-holes", action="store_true",
                    help="Remove holes by keeping only exterior rings (after smoothing)")

    # Classification/grouping
    ap.add_argument("--group-map", default=None,
                    help="Optional JSON: value -> classification label (e.g. {'1':'Tumor'})")
    ap.add_argument("--group-prefix", default="group_",
                    help="Prefix used when group-map is not provided (default group_)")

    args = ap.parse_args()

    mask2d = read_tiff_page(args.mask, args.page)
    group_map = load_group_map(args.group_map)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    features = []

    # ---------- Binary ----------
    if args.binary:
        if args.dissolve:
            geoms = []
            for geom, val in iter_polygons_from_mask(mask2d, binary=True):
                g = shp_shape(geom)
                if args.min_area > 0 and g.area < args.min_area:
                    continue
                g = smooth_geom(g, args.smooth_buffer, args.smooth_passes,
                                args.simplify, args.preserve_topology)
                if args.fill_holes:
                    g = drop_holes(g)
                g = apply_scale(g, args.scale_x, args.scale_y)
                if not g.is_empty:
                    geoms.append(g)

            if geoms:
                merged = unary_union(geoms)
                features.append({
                    "type": "Feature",
                    "geometry": mapping(merged),
                    "properties": {
                        "value": 1,
                        "classification": "foreground"
                    }
                })
        else:
            for geom, val in iter_polygons_from_mask(mask2d, binary=True):
                g = shp_shape(geom)
                if args.min_area > 0 and g.area < args.min_area:
                    continue
                g = smooth_geom(g, args.smooth_buffer, args.smooth_passes,
                                args.simplify, args.preserve_topology)
                if args.fill_holes:
                    g = drop_holes(g)
                g = apply_scale(g, args.scale_x, args.scale_y)
                if g.is_empty:
                    continue
                features.append({
                    "type": "Feature",
                    "geometry": mapping(g),
                    "properties": {
                        "value": 1,
                        "classification": "foreground"
                    }
                })

        gj = {"type": "FeatureCollection", "features": features}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(gj, f)
        print(f"[OK] wrote {len(features)} feature(s) -> {args.out}")
        return

    # ---------- Labeled ----------
    if args.dissolve_by_value:
        # buckets: value -> list of geometries; then unary_union into one feature per value
        buckets: dict[int, list] = {}
        for geom, val in iter_polygons_from_mask(mask2d, binary=False):
            g = shp_shape(geom)
            if args.min_area > 0 and g.area < args.min_area:
                continue
            g = smooth_geom(g, args.smooth_buffer, args.smooth_passes,
                            args.simplify, args.preserve_topology)
            if args.fill_holes:
                g = drop_holes(g)
            g = apply_scale(g, args.scale_x, args.scale_y)
            if g.is_empty:
                continue
            buckets.setdefault(val, []).append(g)

        for val, geoms in sorted(buckets.items(), key=lambda x: x[0]):
            merged = unary_union(geoms)
            features.append({
                "type": "Feature",
                "geometry": mapping(merged),
                "properties": {
                    "value": int(val),
                    "classification": classification_for_value(int(val), group_map, args.group_prefix)
                }
            })
    else:
        # one feature per connected region (each still gets classification based on its value)
        for geom, val in iter_polygons_from_mask(mask2d, binary=False):
            g = shp_shape(geom)
            if args.min_area > 0 and g.area < args.min_area:
                continue
            g = smooth_geom(g, args.smooth_buffer, args.smooth_passes,
                            args.simplify, args.preserve_topology)
            if args.fill_holes:
                g = drop_holes(g)
            g = apply_scale(g, args.scale_x, args.scale_y)
            if g.is_empty:
                continue
            features.append({
                "type": "Feature",
                "geometry": mapping(g),
                "properties": {
                    "value": int(val),
                    "classification": classification_for_value(int(val), group_map, args.group_prefix)
                }
            })

    gj = {"type": "FeatureCollection", "features": features}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(gj, f)

    print(f"[OK] read: {args.mask} (page {args.page})")
    print(f"[OK] wrote {len(features)} feature(s) -> {args.out}")


if __name__ == "__main__":
    main()
