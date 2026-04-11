#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Optional

import numpy as np
from PIL import Image
from rasterio import Affine
from rasterio.features import rasterize
from shapely.geometry import GeometryCollection
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry
import tifffile


DEFAULT_PALETTE = np.array([
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
], dtype=np.uint8)


def read_reference_page(path: str, page: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference image not found: {path}")
    with tifffile.TiffFile(path) as tf:
        if page < 0 or page >= len(tf.pages):
            raise ValueError(f"--reference-page {page} out of range for {path} ({len(tf.pages)} pages)")
        return tf.pages[page].asarray()


def read_reference_shape(path: str, page: int) -> tuple[int, int]:
    arr = read_reference_page(path, page)
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):
            return int(arr.shape[-2]), int(arr.shape[-1])
        return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError(f"Unsupported reference image shape: {arr.shape}")


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] not in (3, 4):
            raise ValueError(f"Unsupported preview image shape: {arr.shape}")
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported preview image shape: {arr.shape}")

    if arr.dtype == np.uint8:
        return arr

    arr_f = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr_f)
    if not finite.any():
        return np.zeros(arr.shape[:2] + (3,), dtype=np.uint8)
    lo = float(np.percentile(arr_f[finite], 1.0))
    hi = float(np.percentile(arr_f[finite], 99.0))
    if hi <= lo:
        hi = lo + 1.0
    arr_f = (arr_f - lo) * (255.0 / (hi - lo))
    return np.clip(arr_f, 0, 255).astype(np.uint8)


def downsample_nearest(arr: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    if factor == 1:
        return arr
    if arr.ndim == 2:
        return arr[::factor, ::factor]
    return arr[::factor, ::factor, ...]


def colorize_label_mask(label_mask: np.ndarray, default_value: int = 0) -> np.ndarray:
    out = np.zeros(label_mask.shape + (3,), dtype=np.uint8)
    label_ids = np.unique(label_mask)
    label_ids = label_ids[label_ids != default_value]
    for idx, lid in enumerate(label_ids):
        out[label_mask == lid] = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
    return out


def save_preview_png(
    reference_path: str,
    reference_page: int,
    mask: np.ndarray,
    out_png: str,
    *,
    factor: int,
    size_threshold_mb: float,
    alpha: float,
    default_value: int,
) -> None:
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    bg = to_uint8_rgb(read_reference_page(reference_path, reference_page))
    if bg.shape[:2] != mask.shape:
        raise ValueError(
            f"Preview background shape {bg.shape[:2]} does not match mask shape {mask.shape}. "
            "Expected a crop-aligned reference image."
        )

    est_bytes = int(bg.nbytes + mask.nbytes)
    threshold_bytes = int(float(size_threshold_mb) * 1024 * 1024)
    use_factor = max(1, int(factor)) if est_bytes > threshold_bytes else 1

    bg_small = downsample_nearest(bg, use_factor)
    mask_small = downsample_nearest(mask, use_factor)
    overlay = colorize_label_mask(mask_small, default_value=default_value)
    fg = mask_small != default_value

    out = bg_small.astype(np.float32, copy=True)
    blend = float(max(0.0, min(1.0, alpha)))
    out[fg] = (1.0 - blend) * out[fg] + blend * overlay[fg].astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    Image.fromarray(out).save(out_png)
    print(
        f"[INFO] wrote preview {out_png} "
        f"(factor={use_factor}, threshold_mb={size_threshold_mb}, estimated_mb={est_bytes / (1024.0 * 1024.0):.1f})"
    )


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


def parse_intlike(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, float):
        return int(round(raw))
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


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


def parse_annotation_props(raw: str) -> list[str]:
    props = [part.strip() for part in (raw or "").split(",")]
    return [part for part in props if part]


def get_annotation_label(props: dict[str, Any], annotation_props: list[str]) -> Optional[str]:
    for key in annotation_props:
        value = get_nested_property(props, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    classification = props.get("classification")
    if isinstance(classification, str) and classification.strip():
        return classification.strip()
    return None


def next_available_value(used_values: set[int]) -> int:
    candidate = 1
    while candidate in used_values:
        candidate += 1
    return candidate


def load_geojson_shapes(
    path: str,
    *,
    binary: bool,
    default_value: int,
    value_prop: str,
    label_mode: str,
    annotation_props: list[str],
    fill_value: int,
) -> tuple[list[tuple[dict[str, Any], int]], list[dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    features = payload.get("features", [])
    shapes_with_values: list[tuple[dict[str, Any], int]] = []
    label_entries: list[dict[str, Any]] = []
    label_to_value: dict[str, int] = {}
    value_to_label: dict[int, str] = {}
    feature_records: list[dict[str, Any]] = []
    explicit_values: set[int] = set()
    unlabeled_used = False

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
        explicit_value = parse_intlike(get_nested_property(props, value_prop, None))
        annotation_label = get_annotation_label(props, annotation_props)
        explicit_values.add(int(explicit_value)) if explicit_value is not None else None
        if explicit_value is None and not annotation_label:
            unlabeled_used = True
        feature_records.append(
            {
                "geom": geom,
                "explicit_value": explicit_value,
                "annotation_label": annotation_label,
            }
        )

    used_values: set[int] = {int(fill_value), *explicit_values}
    if unlabeled_used and int(default_value) != int(fill_value):
        used_values.add(int(default_value))

    for record in feature_records:
        geom = record["geom"]
        explicit_value = record["explicit_value"]
        annotation_label = record["annotation_label"]
        if binary:
            value = 1
        elif label_mode == "property":
            value = explicit_value if explicit_value is not None else int(default_value)
        elif label_mode == "annotation":
            if annotation_label:
                if annotation_label not in label_to_value:
                    label_to_value[annotation_label] = next_available_value(used_values)
                    used_values.add(label_to_value[annotation_label])
                value = label_to_value[annotation_label]
            else:
                value = int(default_value)
                unlabeled_used = True
        else:  # auto
            if explicit_value is not None:
                value = explicit_value
                used_values.add(int(value))
            elif annotation_label:
                if annotation_label not in label_to_value:
                    label_to_value[annotation_label] = next_available_value(used_values)
                    used_values.add(label_to_value[annotation_label])
                value = label_to_value[annotation_label]
            else:
                value = int(default_value)
                unlabeled_used = True

        value = int(value)
        if annotation_label and value not in value_to_label:
            value_to_label[value] = annotation_label

        for poly_geom in iter_polygon_geoms(geom):
            if poly_geom.is_empty:
                continue
            shapes_with_values.append((poly_geom.__geo_interface__, value))

    if unlabeled_used and int(default_value) != int(fill_value) and int(default_value) not in value_to_label:
        value_to_label[int(default_value)] = "unlabeled"

    for value, label in sorted(value_to_label.items(), key=lambda item: item[0]):
        label_entries.append({"value": int(value), "label": label})

    return shapes_with_values, label_entries


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


def write_label_map(
    path: str,
    *,
    fill_value: int,
    default_value: int,
    label_mode: str,
    value_prop: str,
    annotation_props: list[str],
    label_entries: list[dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "fill_value": int(fill_value),
        "default_value": int(default_value),
        "label_mode": label_mode,
        "value_prop": value_prop,
        "annotation_props": annotation_props,
        "labels": label_entries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="Rasterize GeoJSON polygons to a mask aligned to a reference image.")
    ap.add_argument("--geojson", required=True, help="Input GeoJSON feature collection")
    ap.add_argument("--reference", required=True, help="Reference TIFF defining mask width/height and preview background")
    ap.add_argument("--reference-page", type=int, default=0, help="Reference TIFF page to read")
    ap.add_argument("--out", required=True, help="Output mask TIFF path")
    ap.add_argument("--binary", action="store_true", help="Rasterize all polygons with value 1")
    ap.add_argument("--label-mode", choices=["auto", "annotation", "property"], default="auto",
                    help="How polygon values are assigned when --binary is not used")
    ap.add_argument("--value-prop", default="value", help="Feature property to use for raster values when numeric IDs are available")
    ap.add_argument("--annotation-props", default="classification.name,classification.label,class,label,type,name",
                    help="Comma-separated fallback properties used to derive per-annotation class labels")
    ap.add_argument("--default-value", type=int, default=1, help="Fallback value when numeric/class properties are absent")
    ap.add_argument("--fill-value", type=int, default=0, help="Background value")
    ap.add_argument("--all-touched", action="store_true", help="Rasterize all touched pixels instead of center-only")
    ap.add_argument("--compression", default="deflate", help="TIFF compression (default: deflate)")
    ap.add_argument("--preview", default="", help="Optional preview PNG overlay path")
    ap.add_argument("--preview-factor", type=int, default=10,
                    help="Downsample factor used only when preview image is larger than threshold")
    ap.add_argument("--preview-threshold-mb", type=float, default=100.0,
                    help="Downsample preview only when estimated image+mask size exceeds this threshold (MB)")
    ap.add_argument("--preview-alpha", type=float, default=0.45,
                    help="Overlay alpha for colored mask preview (0..1)")
    ap.add_argument("--label-map-out", default="", help="Optional JSON file recording value-to-label mapping")
    args = ap.parse_args()

    annotation_props = parse_annotation_props(args.annotation_props)
    height, width = read_reference_shape(args.reference, args.reference_page)
    shapes_with_values, label_entries = load_geojson_shapes(
        args.geojson,
        binary=args.binary,
        default_value=int(args.default_value),
        value_prop=args.value_prop,
        label_mode=args.label_mode,
        annotation_props=annotation_props,
        fill_value=int(args.fill_value),
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

    if args.label_map_out:
        write_label_map(
            args.label_map_out,
            fill_value=int(args.fill_value),
            default_value=int(args.default_value),
            label_mode="binary" if args.binary else args.label_mode,
            value_prop=args.value_prop,
            annotation_props=annotation_props,
            label_entries=label_entries,
        )
        print(f"[INFO] wrote label map {args.label_map_out}")

    if args.preview:
        save_preview_png(
            args.reference,
            args.reference_page,
            mask,
            args.preview,
            factor=int(args.preview_factor),
            size_threshold_mb=float(args.preview_threshold_mb),
            alpha=float(args.preview_alpha),
            default_value=int(args.fill_value),
        )

    print(
        f"[OK] wrote mask {args.out} shape={mask.shape[0]}x{mask.shape[1]} "
        f"dtype={mask.dtype} features={len(shapes_with_values)} labels={len(label_entries)}"
    )


if __name__ == "__main__":
    main()
