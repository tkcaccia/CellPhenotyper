#!/usr/bin/env python3
import argparse
import os
import numpy as np

from tifffile import imread
from PIL import Image

from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    disk,
)
from skimage.measure import label, regionprops


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

def build_tissue_mask(rgb: np.ndarray,
                      close_radius: int,
                      min_obj_area: int,
                      hole_area: int,
                      keep_largest: bool) -> np.ndarray:
    # Ensure RGB uint8
    if rgb.ndim == 2:
        rgb = np.stack([rgb]*3, axis=-1)
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        # safe-ish scaling if it's 16-bit etc.
        mx = rgb.max()
        rgb = (rgb.astype(np.float32) / (mx if mx else 1.0) * 255.0).clip(0, 255).astype(np.uint8)

    # LAB: tissue tends to deviate from white background in a*/b*
    lab = rgb2lab(rgb)
    a = lab[..., 1]
    b = lab[..., 2]
    chroma = np.sqrt(a*a + b*b)

    # Otsu threshold on chroma
    t = threshold_otsu(chroma)
    mask = chroma > t

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

def write_pyramidal_tiff(mask: np.ndarray,
                         out_tif: str,
                         tile: int,
                         compression: str,
                         bigtiff: bool):
    """
    Writes a *tiled pyramidal* TIFF using pyvips (best for WSI-style pyramids).
    """
    out_dir = os.path.dirname(out_tif)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    u8 = (mask.astype(np.uint8) * 255)

    try:
        import pyvips

        h, w = u8.shape
        img = pyvips.Image.new_from_memory(u8.tobytes(), w, h, 1, format="uchar")
        img.tiffsave(
            out_tif,
            tile=True,
            tile_width=tile,
            tile_height=tile,
            pyramid=True,
            bigtiff=bigtiff,
            compression=compression,
            predictor=True,
        )
    except Exception:
        # Fallback for environments without libvips/pyvips support.
        from tifffile import imwrite
        imwrite(out_tif, u8, compression=compression)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input RGB TIFF (cropped ROI)")
    ap.add_argument("--out-mask", required=True, help="Output pyramidal mask TIFF")
    ap.add_argument("--preview", default=None, help="Optional preview PNG")
    ap.add_argument("--preview-factor", type=int, default=10)
    ap.add_argument("--work-downsample", type=int, default=8,
                    help="Downsample factor used during tissue-mask computation (1 = full resolution).")

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

    rgb = imread(args.image)  # supports multi-page too; for normal TIFF returns array
    if isinstance(rgb, list):
        rgb = rgb[0]

    work_downsample = max(1, int(args.work_downsample))
    close_radius, min_obj_area, hole_area = scaled_cleanup_params(
        args.close_radius,
        args.min_obj_area,
        args.hole_area,
        work_downsample,
    )

    if work_downsample > 1:
        if rgb.ndim == 2:
            rgb = rgb[::work_downsample, ::work_downsample].copy()
        else:
            rgb = rgb[::work_downsample, ::work_downsample, ...].copy()

    mask = build_tissue_mask(
        rgb,
        close_radius=close_radius,
        min_obj_area=min_obj_area,
        hole_area=hole_area,
        keep_largest=args.keep_largest,
    )

    if args.preview:
        save_preview(mask, args.preview, factor=args.preview_factor)

    # Write pyramidal, tiled, compressed TIFF
    write_pyramidal_tiff(
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
