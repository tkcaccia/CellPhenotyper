#!/usr/bin/env python3
"""
grow_to_tissue.py

Grow/extend a cluster-labeled seed mask within a tissue mask, merging all
cells that belong to the same cluster (growth is per cluster label), and
export the FINAL output as a pyramidal + compressed OME-TIFF using
bioformats2raw + raw2ometiff (same approach as btf_to_ometiff.sh).

Requires:
  - tifffile, numpy, scipy
  - scikit-image (for hole filling / nuclei add-in / components)
  - bioformats2raw and raw2ometiff available in PATH inside container

Typical:
singularity exec --bind "$PWD":"$PWD" --pwd "$PWD" "${SINGULARITY}" \
python grow_to_tissue.py \
  --image out_stardist_roi/crop_roi.tif \
  --mask output/KODAMA/cluster_mask.tif \
  --tissue-mask output/KODAMA/tissue_mask.tif \
  --out output/KODAMA/grown_mask.ome.tif \
  --preview output/KODAMA/qc_preview_10x.png \
  --preview-factor 10 \
  --sigma 1.0 \
  --restrict-to-seeded-components \
  --min-seed-area 200 \
  --pyr-compression LZW \
  --max-workers 16 \
  --downsample GAUSSIAN \
  --overwrite
"""

import argparse
import os
import shutil
import subprocess
import time
import numpy as np

from tifffile import imread, imwrite
from scipy.ndimage import distance_transform_edt


# Optional preview deps
try:
    from PIL import Image
except Exception:
    Image = None


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


def ensure_2d(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"{name} must be 2D. Got shape={arr.shape}")


def load_tiff(path: str, name: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    arr = imread(path)
    return arr


def load_mask_2d(path: str, name: str) -> np.ndarray:
    return ensure_2d(load_tiff(path, name), name)

def resize_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize for label/binary masks to match target (H,W)."""
    if mask.shape == target_shape:
        return mask
    from skimage.transform import resize
    out = resize(
        mask.astype(np.float32),
        target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False
    )
    return out.astype(mask.dtype, copy=False)


def require_tool(exe: str):
    if shutil.which(exe) is None:
        raise RuntimeError(f"Required tool not found on PATH: {exe}")


def remove_small_seed_components_per_label(seed_labels: np.ndarray, min_area: int) -> np.ndarray:
    """Remove tiny connected components per label (>0)."""
    if min_area <= 1:
        return seed_labels

    from skimage.measure import label as cc_label

    out = seed_labels.copy()
    labs = np.unique(out)
    labs = labs[labs != 0]

    for lab in labs:
        m = (out == lab)
        cc = cc_label(m, connectivity=1)
        if cc.max() == 0:
            continue
        areas = np.bincount(cc.ravel())
        small = np.where(areas < min_area)[0]
        small = small[small != 0]
        if small.size:
            out[np.isin(cc, small)] = 0

    return out


def restrict_tissue_to_seeded_components(tissue: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Keep only tissue connected components that contain at least one seed pixel."""
    from skimage.measure import label as cc_label

    cc = cc_label(tissue.astype(bool), connectivity=1)
    if cc.max() == 0:
        return tissue.astype(bool)

    keep = np.zeros(cc.max() + 1, dtype=bool)
    seeded_ids = np.unique(cc[seeds > 0])
    seeded_ids = seeded_ids[seeded_ids != 0]
    keep[seeded_ids] = True
    return keep[cc]


def improve_tissue_mask(tissue_bool: np.ndarray,
                        image_rgb: np.ndarray | None,
                        fill_holes_area: int,
                        close_radius: int,
                        add_nuclei: bool,
                        nuclei_thresh: int,
                        nuclei_dilate: int) -> np.ndarray:
    """
    Fix 'holes' in tissue and optionally ensure nuclei are included by OR-ing
    dark-pixel nuclei candidates from the RGB image.
    """
    from skimage.morphology import remove_small_holes, binary_closing, disk, binary_dilation

    t = tissue_bool.astype(bool)

    if fill_holes_area > 0:
        t = remove_small_holes(t, area_threshold=fill_holes_area)

    if close_radius > 0:
        t = binary_closing(t, disk(close_radius))

    if add_nuclei and image_rgb is not None:
        # Heuristic nuclei detection: nuclei are dark in brightfield.
        img = image_rgb
        if img.ndim == 2:
            gray = img.astype(np.float32)
        elif img.ndim == 3 and img.shape[-1] >= 3:
            # luminance-like grayscale
            gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
        else:
            gray = ensure_2d(img, "image").astype(np.float32)

        # Normalize to 0..255
        if gray.max() <= 1.5:
            g8 = (gray * 255.0).clip(0, 255).astype(np.uint8)
        else:
            # handle 16-bit etc
            g = gray - gray.min()
            g = g / (g.max() + 1e-8)
            g8 = (g * 255.0).astype(np.uint8)

        nuclei = (g8 < nuclei_thresh)
        if nuclei_dilate > 0:
            nuclei = binary_dilation(nuclei, disk(nuclei_dilate))

        # Add nuclei pixels to tissue
        t = t | nuclei

        # Fill holes again after adding nuclei
        if fill_holes_area > 0:
            t = remove_small_holes(t, area_threshold=fill_holes_area)

    return t.astype(bool)


def grow_clusters_with_nearest_seed(seeds: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    """
    Voronoi growth within tissue: each tissue pixel gets the label of nearest seed pixel.
    """
    if not np.any(seeds > 0):
        return np.zeros_like(seeds, dtype=np.uint32)

    bg = (seeds == 0).astype(np.uint8)
    _, inds = distance_transform_edt(bg, return_indices=True)
    rr, cc = inds[0], inds[1]

    grown = np.zeros_like(seeds, dtype=np.uint32)
    grown[tissue] = seeds[rr[tissue], cc[tissue]].astype(np.uint32)
    return grown


def downsample_nearest(arr: np.ndarray, factor: int) -> np.ndarray:
    f = int(max(1, factor))
    if f <= 1:
        return arr
    if arr.ndim == 2:
        return arr[::f, ::f]
    return arr[::f, ::f, ...]


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


def colorize_label_mask(label_mask: np.ndarray, default_value: int = 0) -> np.ndarray:
    out = np.zeros(label_mask.shape + (3,), dtype=np.uint8)
    label_ids = np.unique(label_mask)
    label_ids = label_ids[label_ids != default_value]
    for idx, lid in enumerate(label_ids):
        out[label_mask == lid] = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
    return out


def save_preview_png(
    image_path: str | None,
    grown: np.ndarray,
    out_png: str,
    factor: int = 10,
    size_threshold_mb: float = 100.0,
    alpha: float = 0.45,
    default_value: int = 0,
):
    if Image is None:
        print("[WARN] Preview disabled: pillow not available.")
        return
    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if image_path and os.path.exists(image_path):
        bg = to_uint8_rgb(imread(image_path))
        if bg.shape[:2] != grown.shape:
            raise ValueError(
                f"Preview background shape {bg.shape[:2]} does not match grown mask shape {grown.shape}. "
                "Expected crop_roi.tif aligned with mask."
            )
    else:
        bg = np.full(grown.shape + (3,), 255, dtype=np.uint8)

    est_bytes = int(bg.nbytes + grown.nbytes)
    threshold_bytes = int(float(size_threshold_mb) * 1024 * 1024)
    use_factor = int(max(1, factor)) if est_bytes > threshold_bytes else 1

    bg_small = downsample_nearest(bg, use_factor)
    grown_small = downsample_nearest(grown, use_factor)
    overlay = colorize_label_mask(grown_small, default_value=default_value)
    fg = grown_small != default_value

    out = bg_small.astype(np.float32, copy=True)
    a = float(max(0.0, min(1.0, alpha)))
    out[fg] = (1.0 - a) * out[fg] + a * overlay[fg].astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    Image.fromarray(out).save(out_png)
    print(
        "[INFO] wrote grown overlay preview: "
        f"{out_png} (factor={use_factor}, "
        f"threshold_mb={size_threshold_mb}, "
        f"estimated_mb={est_bytes / (1024.0 * 1024.0):.1f})"
    )


def pyramidize_with_raw2ometiff(in_tif: str,
                                out_ome_tif: str,
                                compression: str,
                                max_workers: int,
                                downsample: str,
                                overwrite: bool,
                                keep_tmp: bool,
                                legacy: bool):
    """
    Create a pyramidal + compressed OME-TIFF using:
      bioformats2raw --downsample-type <...> in_tif tmp.rawdir
      raw2ometiff --compression=<...> --max_workers=<...> -p [--legacy] tmp.rawdir out_ome_tif

    Mirrors btf_to_ometiff.sh logic :contentReference[oaicite:2]{index=2}.
    """
    require_tool("bioformats2raw")
    require_tool("raw2ometiff")

    if os.path.exists(out_ome_tif):
        if overwrite:
            os.remove(out_ome_tif)
        else:
            raise RuntimeError(f"Output exists (use --overwrite): {out_ome_tif}")

    outdir = os.path.dirname(out_ome_tif) or "."
    base = os.path.basename(out_ome_tif)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tmpdir = os.path.join(outdir, f"{base}.{stamp}.{os.getpid():06d}.rawdir")

    try:
        print(f"[INFO] bioformats2raw -> {tmpdir}")
        subprocess.run(
            ["bioformats2raw", "--downsample-type", downsample, in_tif, tmpdir],
            check=True
        )

        r2o = ["raw2ometiff", f"--compression={compression}", f"--max_workers={max_workers}"]
        if legacy:
            r2o.append("--legacy")

        print(f"[INFO] raw2ometiff -> {out_ome_tif}")
        subprocess.run(r2o + [tmpdir, out_ome_tif], check=True)

        print(f"[OK] pyramidal OME-TIFF written: {out_ome_tif}")

    finally:
        if not keep_tmp:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(
        description="Grow cluster-labeled seeds inside tissue and export pyramidal + compressed mask via raw2ometiff."
    )
    ap.add_argument("--image", default=None, help="RGB image for preview + optional nuclei inclusion")
    ap.add_argument("--mask", required=True, help="Seed mask TIFF: integer cluster labels (0=background)")
    ap.add_argument("--tissue-mask", required=True, help="Binary tissue mask TIFF (nonzero=tissue)")
    ap.add_argument("--out", required=True, help="FINAL output pyramidal OME-TIFF (e.g. grown_mask.ome.tif)")
    ap.add_argument("--preview", default=None, help="Optional preview PNG")
    ap.add_argument("--preview-factor", type=int, default=10)
    ap.add_argument("--preview-threshold-mb", type=float, default=100.0,
                    help="Downsample preview only when estimated image+mask size exceeds this threshold (MB).")
    ap.add_argument("--preview-alpha", type=float, default=0.45,
                    help="Overlay alpha for colored grown mask preview (0..1).")

    # kept for CLI compatibility
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="Kept for compatibility; not used (growth is nearest-seed within tissue).")

    ap.add_argument("--restrict-to-seeded-components", action="store_true",
                    help="Only keep tissue CCs that contain at least one seed.")
    ap.add_argument("--min-seed-area", type=int, default=200,
                    help="Remove seed connected components smaller than this (per label).")

    # Tissue-mask “hole” fixes
    ap.add_argument("--fill-holes-area", type=int, default=50000,
                    help="Fill holes in tissue mask smaller than this area (px).")
    ap.add_argument("--close-radius", type=int, default=12,
                    help="Binary closing radius to connect tissue gaps (px).")

    # Nuclei inclusion (helps prevent nuclei being excluded by a weak tissue mask)
    ap.add_argument("--no-add-nuclei", action="store_true",
                    help="Disable nuclei add-in from the RGB image.")
    ap.add_argument("--nuclei-thresh", type=int, default=170,
                    help="Grayscale threshold (0-255): pixels darker than this are considered nuclei.")
    ap.add_argument("--nuclei-dilate", type=int, default=2,
                    help="Dilate nuclei candidates by this radius (px) before OR into tissue.")

    # Pyramid/compression (raw2ometiff)
    ap.add_argument("--pyr-compression", default="LZW",
                    help="UNCOMPRESSED|LZW|JPEG|JPEG_2000|JPEG_2000_LOSSY (default LZW).")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--downsample", default="GAUSSIAN",
                    help="SIMPLE|GAUSSIAN|AREA|LINEAR|CUBIC|LANCZOS (default GAUSSIAN).")
    ap.add_argument("--legacy", action="store_true", help="Write Bio-Formats 5.9.x-compatible pyramid.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    ap.add_argument("--keep-tmp", action="store_true", help="Keep intermediate .rawdir.")

    args = ap.parse_args()

    seeds = load_mask_2d(args.mask, "seed mask").astype(np.int32)
    tissue_raw = load_mask_2d(args.tissue_mask, "tissue mask")
    if tissue_raw.shape != seeds.shape:
        print(f"[WARN] Resizing tissue mask from {tissue_raw.shape} to {seeds.shape} to match seed/image grid.")
        tissue_raw = resize_mask_to_shape(tissue_raw, seeds.shape)
    tissue = (tissue_raw > 0)

    image_rgb = None
    if args.image and os.path.exists(args.image):
        image_rgb = load_tiff(args.image, "image")

    # Improve tissue mask to remove holes + ensure nuclei are included
    tissue = improve_tissue_mask(
        tissue_bool=tissue,
        image_rgb=image_rgb,
        fill_holes_area=args.fill_holes_area,
        close_radius=args.close_radius,
        add_nuclei=(not args.no_add_nuclei and image_rgb is not None),
        nuclei_thresh=args.nuclei_thresh,
        nuclei_dilate=args.nuclei_dilate,
    )

    # Restrict seeds to tissue
    seeds[~tissue] = 0

    # Remove tiny seed specks per label
    if args.min_seed_area and args.min_seed_area > 1:
        seeds = remove_small_seed_components_per_label(seeds, args.min_seed_area)

    # Optionally restrict tissue to only seeded components
    if args.restrict_to_seeded_components:
        tissue = restrict_tissue_to_seeded_components(tissue, seeds)
        seeds[~tissue] = 0

    # Grow clusters within tissue (fills everything inside tissue, so no holes remain in grown mask)
    grown = grow_clusters_with_nearest_seed(seeds, tissue)

    # Write a temporary single-level label TIFF (raw2ometiff will pyramidize/compress)
    outdir = os.path.dirname(args.out) or "."
    os.makedirs(outdir, exist_ok=True)
    tmp_flat = os.path.join(outdir, f".tmp_flat_{os.getpid()}_{int(time.time())}.tif")

    maxlab = int(grown.max()) if grown.size else 0
    flat = grown.astype(np.uint16) if maxlab <= 65535 else grown.astype(np.uint32)

    imwrite(tmp_flat, flat)  # temp only; final compression is handled by raw2ometiff

    # Pyramidize + compress using the same approach as your script :contentReference[oaicite:3]{index=3}
    try:
        pyramidize_with_raw2ometiff(
            in_tif=tmp_flat,
            out_ome_tif=args.out,
            compression=args.pyr_compression,
            max_workers=args.max_workers,
            downsample=args.downsample,
            overwrite=args.overwrite,
            keep_tmp=args.keep_tmp,
            legacy=args.legacy,
        )
    except Exception as e:
        # Robust fallback for environments where raw2ometiff/bioformats2raw are present
        # but incompatible with the generated temporary pyramid source.
        print(f"[WARN] Pyramidal conversion failed ({e}). Writing flat TIFF fallback to: {args.out}")
        if os.path.exists(args.out):
            if args.overwrite:
                os.remove(args.out)
            else:
                raise
        shutil.copyfile(tmp_flat, args.out)

    # Preview
    if args.preview:
        save_preview_png(
            image_path=args.image,
            grown=grown,
            out_png=args.preview,
            factor=args.preview_factor,
            size_threshold_mb=args.preview_threshold_mb,
            alpha=args.preview_alpha,
            default_value=0,
        )

    # Cleanup temp
    try:
        os.remove(tmp_flat)
    except Exception:
        pass

    frac = float((grown > 0).mean()) if grown.size else 0.0
    print(f"[OK] labeled tissue fraction: {frac:.4f}")
    print(f"[OK] done: {args.out}")


if __name__ == "__main__":
    main()
