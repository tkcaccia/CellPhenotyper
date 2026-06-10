#!/usr/bin/env python3
import argparse
import os
import numpy as np

import tifffile
import zarr
from PIL import Image

from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    disk,
)
from skimage.measure import label, regionprops


def _normalize_axes(axes: str | None, ndim: int) -> str:
    axes = (axes or "").replace("S", "C")
    if axes and len(axes) == ndim:
        return axes
    if ndim == 2:
        return "YX"
    if ndim == 3:
        return "YXC"
    return axes or "".join(str(i) for i in range(ndim))


def _to_yxc(arr: np.ndarray, axes: str) -> np.ndarray:
    axes = _normalize_axes(axes, arr.ndim)
    if arr.ndim == 2:
        return arr
    if "Y" in axes and "X" in axes:
        ydim = axes.index("Y")
        xdim = axes.index("X")
        cdim = axes.index("C") if "C" in axes else None
        if cdim is None:
            arr = np.moveaxis(arr, [ydim, xdim], [0, 1])
            return arr
        return np.moveaxis(arr, [ydim, xdim, cdim], [0, 1, 2])
    if arr.shape[-1] in (3, 4):
        return arr
    if arr.shape[0] in (3, 4):
        return np.moveaxis(arr, 0, -1)
    return arr


def read_image_lazy_downsample(path: str, factor: int, tile_out: int = 512) -> np.ndarray:
    """
    Read a downsampled RGB/grayscale working image without loading the full WSI.

    ``tifffile.imread()[::factor]`` still materializes the full input first. For
    huge StarDist crops this can be tens of GiB, so we read bounded contiguous
    windows and downsample each window locally.
    """
    factor = int(max(1, factor))
    if factor <= 1:
        image = tifffile.imread(path)
        if isinstance(image, list):
            image = image[0]
        return image

    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = _normalize_axes(getattr(series, "axes", None), len(series.shape))
        arr = zarr.open(series.aszarr(), mode="r")

        if "Y" in axes and "X" in axes:
            h = int(series.shape[axes.index("Y")])
            w = int(series.shape[axes.index("X")])
            cdim = axes.index("C") if "C" in axes else None
            channels = int(series.shape[cdim]) if cdim is not None else 0
        else:
            yxc_shape = _to_yxc(np.asarray(arr[tuple(slice(0, min(1, s)) for s in arr.shape)]), axes).shape
            h, w = int(series.shape[0]), int(series.shape[1])
            channels = yxc_shape[2] if len(yxc_shape) == 3 else 0

        out_h = int(np.ceil(h / float(factor)))
        out_w = int(np.ceil(w / float(factor)))
        if channels:
            out = np.zeros((out_h, out_w, min(channels, 4)), dtype=np.dtype(arr.dtype))
        else:
            out = np.zeros((out_h, out_w), dtype=np.dtype(arr.dtype))

        tile_out = int(max(64, tile_out))
        for oy0 in range(0, out_h, tile_out):
            oy1 = min(out_h, oy0 + tile_out)
            sy0 = oy0 * factor
            sy1 = min(h, (oy1 - 1) * factor + 1)
            for ox0 in range(0, out_w, tile_out):
                ox1 = min(out_w, ox0 + tile_out)
                sx0 = ox0 * factor
                sx1 = min(w, (ox1 - 1) * factor + 1)

                slicer = [slice(None)] * len(series.shape)
                if "Y" in axes and "X" in axes:
                    slicer[axes.index("Y")] = slice(sy0, sy1)
                    slicer[axes.index("X")] = slice(sx0, sx1)
                else:
                    slicer[0] = slice(sy0, sy1)
                    slicer[1] = slice(sx0, sx1)

                patch = _to_yxc(np.asarray(arr[tuple(slicer)]), axes)
                if patch.ndim == 2:
                    patch_ds = patch[::factor, ::factor]
                    out[oy0:oy0 + patch_ds.shape[0], ox0:ox0 + patch_ds.shape[1]] = patch_ds
                else:
                    patch_ds = patch[::factor, ::factor, ...]
                    out[oy0:oy0 + patch_ds.shape[0], ox0:ox0 + patch_ds.shape[1], :patch_ds.shape[2]] = patch_ds
        return out


def read_image_shape(path: str) -> tuple[int, int]:
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = _normalize_axes(getattr(series, "axes", None), len(series.shape))
        if "Y" in axes and "X" in axes:
            return int(series.shape[axes.index("Y")]), int(series.shape[axes.index("X")])
        return int(series.shape[0]), int(series.shape[1])


def scaled_cleanup_params(close_radius: int, min_obj_area: int, hole_area: int, scale: int):
    """
    Scale morphology parameters when mask computation runs on a downsampled image.
    """
    if scale <= 1:
        return close_radius, min_obj_area, hole_area

    scale2 = scale * scale
    close_radius_s = 0 if close_radius <= 0 else max(1, int(round(close_radius / scale)))
    min_obj_area_s = 0 if min_obj_area <= 0 else max(1, int(round(min_obj_area / scale2)))
    hole_area_s = 0 if hole_area <= 0 else max(1, int(round(hole_area / scale2)))
    return close_radius_s, min_obj_area_s, hole_area_s

def round_up_power_of_two(value: int) -> int:
    value = int(max(1, value))
    return 1 << (value - 1).bit_length()

def cap_work_downsample(height: int, width: int, requested: int, max_work_pixels: int) -> int:
    """
    Keep the tissue-mask working image bounded for WSI-scale inputs.

    Downstream steps resize this mask to the clustering grid, so a coarser
    working mask is preferable to letting LAB-sized temporaries exhaust RAM.
    """
    requested = int(max(1, requested))
    max_work_pixels = int(max(0, max_work_pixels))
    if max_work_pixels <= 0:
        return requested

    out_h = int(np.ceil(height / float(requested)))
    out_w = int(np.ceil(width / float(requested)))
    if out_h * out_w <= max_work_pixels:
        return requested

    needed = int(np.ceil(np.sqrt((float(height) * float(width)) / float(max_work_pixels))))
    return max(requested, round_up_power_of_two(needed))

def rgb_tissue_score(rgb: np.ndarray) -> np.ndarray:
    """
    Memory-light tissue score for RGB slides.

    The older LAB chroma path allocates several large float arrays. This uses
    uint8 channel spread plus darkness, which captures stained tissue against
    bright background while keeping only one float32 score image.
    """
    maxc = rgb.max(axis=2).astype(np.int16, copy=False)
    minc = rgb.min(axis=2).astype(np.int16, copy=False)
    saturation = maxc - minc
    darkness = 255 - maxc
    score = np.maximum(saturation * 2, darkness).astype(np.float32, copy=False)
    return score

def save_preview(mask: np.ndarray, out_png: str, factor: int = 10):
    # Downsample for quick QC
    h, w = mask.shape
    hh, ww = max(1, h // factor), max(1, w // factor)
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    img = img.resize((ww, hh), resample=Image.NEAREST)
    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    img.save(out_png)

def keep_largest_component(mask: np.ndarray, min_area: int = 0):
    lab = label(mask)
    if lab.max() == 0:
        return mask
    props = regionprops(lab)
    props = [p for p in props if p.area >= min_area]
    if not props:
        return np.zeros_like(mask, dtype=bool)
    biggest = max(props, key=lambda p: p.area)
    out = (lab == biggest.label)
    return out

def build_tissue_mask(image: np.ndarray,
                      close_radius: int,
                      min_obj_area: int,
                      hole_area: int,
                      keep_largest: bool) -> np.ndarray:
    # 2D inputs are handled as grayscale directly (no RGB conversion).
    if image.ndim == 2:
        gray = image.astype(np.float32, copy=False)
        gray -= gray.min()
        mx = gray.max()
        if mx > 0:
            gray /= mx
        inv = 1.0 - gray  # tissue is usually darker than bright background
        t = threshold_otsu(inv)
        mask = inv > t
    else:
        # RGB branch
        rgb = image
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            mx = rgb.max()
            rgb = (rgb.astype(np.float32) / (mx if mx else 1.0) * 255.0).clip(0, 255).astype(np.uint8)

        score = rgb_tissue_score(rgb)
        t = threshold_otsu(score)
        mask = score > t

    # Clean up
    if close_radius > 0:
        mask = binary_closing(mask, disk(close_radius))
    if hole_area > 0:
        mask = remove_small_holes(mask, area_threshold=hole_area)
    if min_obj_area > 0:
        mask = remove_small_objects(mask, min_size=min_obj_area)

    # Optional: keep only the largest connected tissue region
    if keep_largest:
        mask = keep_largest_component(mask, min_area=min_obj_area)

    return mask.astype(bool)

def write_mask_tiff(mask: np.ndarray,
                    out_tif: str,
                    tile: int,
                    compression: str,
                    bigtiff: bool):
    """
    Write a compressed TIFF that tifffile can reopen reliably.

    This mask is an internal pipeline artifact that is consumed by later
    tifffile-based steps. Reliability matters more than pyramidal storage.
    Keep ``tile`` in the signature for CLI compatibility even though it is not
    used by the plain TIFF writer.
    """
    out_dir = os.path.dirname(out_tif)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    u8 = (mask.astype(np.uint8) * 255)
    _ = tile

    from tifffile import imread, imwrite

    imwrite(
        out_tif,
        u8,
        compression=compression,
        bigtiff=bool(bigtiff),
        photometric="minisblack",
        metadata=None,
    )

    check = imread(out_tif)
    if isinstance(check, list):
        check = check[0]
    if tuple(check.shape[:2]) != tuple(u8.shape[:2]):
        raise RuntimeError(
            f"Tissue mask sanity check failed for {out_tif}: "
            f"expected {u8.shape}, got {getattr(check, 'shape', None)}"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input RGB TIFF (cropped ROI)")
    ap.add_argument("--out-mask", required=True, help="Output tissue mask TIFF")
    ap.add_argument("--preview", default=None, help="Optional preview PNG")
    ap.add_argument("--preview-factor", type=int, default=10)
    ap.add_argument("--work-downsample", type=int, default=8,
                    help="Downsample factor used during tissue-mask computation (1 = full resolution).")
    ap.add_argument("--auto-no-downsample-max-side", type=int, default=1024,
                    help="If max(image_height,image_width) <= this value, force work_downsample=1. Set 0 to disable.")
    ap.add_argument("--max-work-pixels", type=int, default=40000000,
                    help="Increase work_downsample automatically when the working image would exceed this many pixels. Set 0 to disable.")

    # Mask cleanup knobs
    ap.add_argument("--close-radius", type=int, default=10,
                    help="Binary closing radius in pixels (fill small gaps)")
    ap.add_argument("--min-obj-area", type=int, default=20000,
                    help="Remove objects smaller than this area (px)")
    ap.add_argument("--hole-area", type=int, default=20000,
                    help="Fill holes smaller than this area (px)")
    ap.add_argument("--keep-largest", action="store_true",
                    help="Keep only the largest connected tissue region")

    # TIFF writing knobs
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--compression", default="deflate",
                    help='e.g. "deflate" or "lzw"')
    ap.add_argument("--bigtiff", action="store_true")

    args = ap.parse_args()

    work_downsample = max(1, int(args.work_downsample))
    h0, w0 = read_image_shape(args.image)
    if args.auto_no_downsample_max_side > 0:
        if max(h0, w0) <= int(args.auto_no_downsample_max_side):
            work_downsample = 1
    work_downsample = cap_work_downsample(h0, w0, work_downsample, args.max_work_pixels)

    image = read_image_lazy_downsample(args.image, work_downsample)

    close_radius, min_obj_area, hole_area = scaled_cleanup_params(
        args.close_radius,
        args.min_obj_area,
        args.hole_area,
        work_downsample,
    )

    mask = build_tissue_mask(
        image,
        close_radius=close_radius,
        min_obj_area=min_obj_area,
        hole_area=hole_area,
        keep_largest=args.keep_largest,
    )

    if args.preview:
        save_preview(mask, args.preview, factor=args.preview_factor)

    # Write a compressed TIFF that downstream tifffile readers can reopen.
    write_mask_tiff(
        mask,
        args.out_mask,
        tile=args.tile,
        compression=args.compression,
        bigtiff=args.bigtiff,
    )

    # Simple sanity print
    frac = float(mask.mean()) if mask.size else 0.0
    print(f"[INFO] work_downsample={work_downsample}")
    print(f"[OK] Tissue fraction: {frac:.4f}")
    print(f"[OK] Wrote: {args.out_mask}")

if __name__ == "__main__":
    main()
