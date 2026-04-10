#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.features import rasterize
from shapely.geometry import GeometryCollection
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry
import tifffile


def read_reference_shape(path: str, page: int) -> tuple[int, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference image not found: {path}")
    with tifffile.TiffFile(path) as tf:
        if page < 0 or page >= len(tf.pages):
            raise ValueError(f"--reference-page {page} out of range for {path} ({len(tf.pages)} pages)")
        arr = tf.pages[page].asarray()
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):
            return int(arr.shape[-2]), int(arr.shape[-1])
        return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError(f"Unsupported reference image shape: {arr.shape}")


def get_nested_property(props: dict[str, Any], key: str, default: Any) -> Any:
    if not key:
        return default
    cur: Any = props
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def iter_polygon_geoms(geom: BaseGeometry):
    if geom.is_empty:
        return
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        yield geom
        return
    if isinstance(geom, GeometryCollection):
        for subgeom in geom.geoms:
            yield from iter_polygon_geoms(subgeom)


def normalize_value(raw: Any, default_value: int) -> int:
    if raw is None:
        return default_value
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, float):
        return int(round(raw))
    text = str(raw).strip()
    if not text:
        return default_value
    try:
        return int(float(text))
    except ValueError:
        return default_value


def choose_dtype(values: list[int]) -> np.dtype:
    mn = min(values) if values else 0
    mx = max(values) if values else 0
    if mn >= 0:
        if mx <= np.iinfo(np.uint8).max:
            return np.uint8
        if mx <= np.iinfo(np.uint16).max:
            return np.uint16
        if mx <= np.iinfo(np.uint32).max:
            return np.uint32
    if mn >= np.iinfo(np.int16).min and mx <= np.iinfo(np.int16).max:
        return np.int16
    return np.int32


def load_geojson_shapes(
    path: str,
    *,
    binary: bool,
    default_value: int,
    value_prop: str,
) -> list[tuple[dict[str, Any], int]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    features = payload.get("features", [])
    shapes_with_values: list[tuple[dict[str, Any], int]] = []

    for feature in features:
        geom_payload = feature.get("geometry")
        if not geom_payload:
            continue
        geom = shapely_shape(geom_payload)
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue

        props = feature.get("properties") or {}
        value = 1 if binary else normalize_value(
            get_nested_property(props, value_prop, default_value),
            default_value,
        )

        for poly_geom in iter_polygon_geoms(geom):
            if poly_geom.is_empty:
                continue
            shapes_with_values.append((poly_geom.__geo_interface__, value))

    return shapes_with_values


def write_mask(path: str, mask: np.ndarray, compression: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tifffile.imwrite(
        path,
        mask,
        photometric="minisblack",
        compression=compression,
        bigtiff=mask.nbytes >= (4 * 1024 * 1024 * 1024),
        metadata=None,
    )


def main():
    ap = argparse.ArgumentParser(description="Rasterize GeoJSON polygons to a mask aligned to a reference image.")
    ap.add_argument("--geojson", required=True, help="Input GeoJSON feature collection")
    ap.add_argument("--reference", required=True, help="Reference TIFF defining mask width/height")
    ap.add_argument("--reference-page", type=int, default=0, help="Reference TIFF page to read")
    ap.add_argument("--out", required=True, help="Output mask TIFF path")
    ap.add_argument("--binary", action="store_true", help="Rasterize all polygons with value 1")
    ap.add_argument("--value-prop", default="value", help="Feature property to use for raster values")
    ap.add_argument("--default-value", type=int, default=1, help="Fallback value when the property is absent")
    ap.add_argument("--fill-value", type=int, default=0, help="Background value")
    ap.add_argument("--all-touched", action="store_true", help="Rasterize all touched pixels instead of center-only")
    ap.add_argument("--compression", default="deflate", help="TIFF compression (default: deflate)")
    args = ap.parse_args()

    height, width = read_reference_shape(args.reference, args.reference_page)
    shapes_with_values = load_geojson_shapes(
        args.geojson,
        binary=args.binary,
        default_value=args.default_value,
        value_prop=args.value_prop,
    )

    if shapes_with_values:
        dtype = choose_dtype([int(v) for _, v in shapes_with_values] + [int(args.fill_value)])
        mask = rasterize(
            shapes=shapes_with_values,
            out_shape=(height, width),
            fill=int(args.fill_value),
            transform=Affine.identity(),
            all_touched=args.all_touched,
            dtype=dtype,
        )
    else:
        dtype = choose_dtype([int(args.fill_value)])
        mask = np.full((height, width), int(args.fill_value), dtype=dtype)

    write_mask(args.out, mask, args.compression)
    print(
        f"[OK] wrote mask {args.out} shape={mask.shape[0]}x{mask.shape[1]} "
        f"dtype={mask.dtype} features={len(shapes_with_values)}"
    )


if __name__ == "__main__":
    main()
