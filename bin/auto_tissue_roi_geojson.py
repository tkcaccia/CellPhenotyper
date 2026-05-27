#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import zarr
from PIL import Image
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects

from convert_tissue_mask_to_geojson import (
    apply_scale,
    drop_holes,
    iter_polygons_from_mask,
    smooth_geom,
)
from shapely.geometry import mapping
from shapely.geometry import shape as shp_shape

try:
    import pyvips
except Exception:
    pyvips = None


def scaled_cleanup_params(close_radius: int, min_obj_area: int, hole_area: int, scale: int):
    if scale <= 1:
        return close_radius, min_obj_area, hole_area
    scale2 = scale * scale
    close_radius_s = 0 if close_radius <= 0 else max(1, int(round(close_radius / scale)))
    min_obj_area_s = 0 if min_obj_area <= 0 else max(1, int(round(min_obj_area / scale2)))
    hole_area_s = 0 if hole_area <= 0 else max(1, int(round(hole_area / scale2)))
    return close_radius_s, min_obj_area_s, hole_area_s


def infer_hw_and_downsample(image_path: Path, step: int) -> tuple[np.ndarray, int, int]:
    if pyvips is not None:
        try:
            image_vips = pyvips.Image.new_from_file(str(image_path), access="sequential")
            width = int(image_vips.width)
            height = int(image_vips.height)
            target_width = max(1, int(round(width / max(1, step))))
            thumb = pyvips.Image.thumbnail(str(image_path), target_width)
            bands = min(thumb.bands, 3)
            if bands <= 0:
                raise RuntimeError("thumbnail has no bands")
            if bands == 1:
                thumb = thumb.colourspace("b-w")
            arr = np.ndarray(
                buffer=thumb.write_to_memory(),
                dtype=np.uint8,
                shape=(thumb.height, thumb.width, thumb.bands),
            ).copy()
            if thumb.bands == 1:
                arr = arr[..., 0]
            elif thumb.bands > 3:
                arr = arr[..., :3]
            return arr, height, width
        except Exception:
            pass

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
                    for key in keys:
                        try:
                            sub = obj[key]
                            if hasattr(sub, "shape"):
                                return sub
                        except Exception:
                            continue
        except Exception:
            pass
        return None

    with tifffile.TiffFile(str(image_path)) as tf:
        if not tf.series:
            raise RuntimeError(f"No TIFF series found in {image_path}")
        series = tf.series[0]
        level0 = series.levels[0] if getattr(series, "levels", None) else series
        zobj = level0.aszarr()
        arr = _select_array(zobj)
        if arr is None:
            arr = _select_array(zarr.open(zobj, mode="r"))
        if arr is None:
            raise RuntimeError(f"Could not resolve a zarr array from {image_path}")
        shape = tuple(level0.shape)
        axes = (getattr(level0, "axes", None) or getattr(series, "axes", None) or "").replace("S", "C")

        if axes and "Y" in axes and "X" in axes:
            ydim = axes.index("Y")
            xdim = axes.index("X")
        else:
            ydim, xdim = 0, 1

        h0 = int(shape[ydim])
        w0 = int(shape[xdim])
        slicer = [slice(None)] * len(shape)
        slicer[ydim] = slice(0, h0, step)
        slicer[xdim] = slice(0, w0, step)
        small = np.asarray(arr[tuple(slicer)])

    if axes and len(axes) == small.ndim and "Y" in axes and "X" in axes:
        order = [axes.index("Y"), axes.index("X")] + [i for i, a in enumerate(axes) if a not in ("Y", "X")]
        small = np.transpose(small, axes=order)

    if small.ndim == 2:
        return small, h0, w0
    if small.ndim == 3 and small.shape[-1] in (1, 3, 4):
        return small[..., :3], h0, w0
    if small.ndim == 3 and small.shape[0] in (1, 3, 4):
        return np.moveaxis(small, 0, -1)[..., :3], h0, w0
    if small.ndim >= 2:
        return small[..., :3] if small.shape[-1] >= 3 else small[..., 0], h0, w0
    raise RuntimeError(f"Could not interpret image shape {small.shape}")


def build_tissue_mask(image: np.ndarray, close_radius: int, min_obj_area: int, hole_area: int, keep_largest: bool) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(np.float32, copy=False)
        gray -= gray.min()
        mx = gray.max()
        if mx > 0:
            gray /= mx
        inv = 1.0 - gray
        thresh = threshold_otsu(inv)
        mask = inv > thresh
    else:
        rgb = image
        if rgb.dtype != np.uint8:
            mx = rgb.max()
            rgb = (rgb.astype(np.float32) / (mx if mx else 1.0) * 255.0).clip(0, 255).astype(np.uint8)
        lab = rgb2lab(rgb[..., :3])
        chroma = np.sqrt(lab[..., 1] * lab[..., 1] + lab[..., 2] * lab[..., 2])
        thresh = threshold_otsu(chroma)
        mask = chroma > thresh

    if close_radius > 0:
        mask = binary_closing(mask, disk(close_radius))
    if hole_area > 0:
        mask = remove_small_holes(mask, area_threshold=hole_area)
    if min_obj_area > 0:
        mask = remove_small_objects(mask, min_size=min_obj_area)
    if keep_largest:
        lab = label(mask)
        props = regionprops(lab)
        if props:
            largest = max(props, key=lambda p: p.area)
            mask = lab == largest.label
    return mask.astype(bool)


def save_preview(image: np.ndarray, mask: np.ndarray, out_png: Path) -> None:
    if image.ndim == 2:
        rgb = np.repeat(image[..., None], 3, axis=2)
    else:
        rgb = image[..., :3]
    if rgb.dtype != np.uint8:
        mx = rgb.max()
        rgb = (rgb.astype(np.float32) / (mx if mx else 1.0) * 255.0).clip(0, 255).astype(np.uint8)
    overlay = rgb.copy()
    overlay[mask] = (0.65 * overlay[mask] + 0.35 * np.array([255, 140, 0], dtype=np.float32)).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_png)


def full_image_geojson(width: int, height: int, source_image: str, reason: str) -> dict:
    polygon = [
        [0.0, 0.0],
        [float(width), 0.0],
        [float(width), float(height)],
        [0.0, float(height)],
        [0.0, 0.0],
    ]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "full_image_roi",
                    "source_image": source_image,
                    "source": reason,
                    "width_px": int(width),
                    "height_px": int(height),
                },
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
            }
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Infer a tissue-driven ROI GeoJSON from an image without requiring a manual ROI.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", default="")
    ap.add_argument("--downsample", type=int, default=32)
    ap.add_argument("--close-radius", type=int, default=12)
    ap.add_argument("--min-obj-area", type=int, default=30000)
    ap.add_argument("--hole-area", type=int, default=30000)
    ap.add_argument("--keep-largest", action="store_true")
    ap.add_argument("--smooth-buffer", type=float, default=4.0)
    ap.add_argument("--smooth-passes", type=int, default=2)
    ap.add_argument("--simplify", type=float, default=4.0)
    ap.add_argument("--fill-holes", action="store_true")
    args = ap.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    preview_path = Path(args.preview).expanduser().resolve() if args.preview else None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    downsample = max(1, int(args.downsample))
    image_small, h0, w0 = infer_hw_and_downsample(image_path, downsample)
    close_radius, min_obj_area, hole_area = scaled_cleanup_params(
        args.close_radius,
        args.min_obj_area,
        args.hole_area,
        downsample,
    )
    mask = build_tissue_mask(
        image_small,
        close_radius=close_radius,
        min_obj_area=min_obj_area,
        hole_area=hole_area,
        keep_largest=args.keep_largest,
    )

    if preview_path is not None:
        save_preview(image_small, mask, preview_path)

    features = []
    for geom, _ in iter_polygons_from_mask(mask.astype(np.uint8), binary=True):
        shp = shp_shape(geom)
        if shp.is_empty:
            continue
        shp = smooth_geom(
            shp,
            smooth_buffer=args.smooth_buffer,
            smooth_passes=args.smooth_passes,
            simplify=args.simplify,
            preserve_topology=True,
        )
        if args.fill_holes:
            shp = drop_holes(shp)
        shp = apply_scale(shp, downsample, downsample)
        if shp.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(shp),
                "properties": {
                    "classification": "tissue",
                    "source": "auto_tissue_roi",
                    "downsample": downsample,
                },
            }
        )

    if not features:
        geo = full_image_geojson(w0, h0, image_path.name, "auto_tissue_roi_fallback")
    else:
        geo = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "source_image": image_path.name,
                "source": "auto_tissue_roi",
                "width_px": int(w0),
                "height_px": int(h0),
                "downsample": downsample,
            },
        }
    out_path.write_text(json.dumps(geo), encoding="utf-8")
    print(f"[OK] wrote {len(features)} tissue ROI feature(s) to {out_path}")


if __name__ == "__main__":
    main()
