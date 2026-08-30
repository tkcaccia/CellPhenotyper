 
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
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np

import tifffile
from tifffile import imread, imwrite
from scipy.ndimage import distance_transform_edt

from ome_tiff_metadata import (
    create_tiff_memmap,
    label_storage_dtype,
    read_mpp_json,
    tiff_resolution_kwargs,
    validate_ome_tiff,
)


# Optional preview deps
try:
    from PIL import Image
except Exception:
    Image = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MEDSAM_REPO = Path(os.environ.get("MEDSAM_REPO_DIR", str(PROJECT_ROOT / "third_party" / "MedSAM")))
DEFAULT_MEDSAM_CHECKPOINT = Path(
    os.environ.get(
        "MEDSAM_CHECKPOINT",
        str(DEFAULT_MEDSAM_REPO / "work_dir" / "MedSAM" / "medsam_vit_b.pth"),
    )
)


def load_medsam_runtime():
    from medsam_border_refine import (
        MedSAMConfig,
        MedSAMUnavailableError,
        run_medsam_border_refine,
    )
    return MedSAMConfig, MedSAMUnavailableError, run_medsam_border_refine


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


def tiff_2d_shape_dtype(path: str, name: str) -> tuple[tuple[int, int], np.dtype]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        shape = tuple(page.shape)
        dtype = np.dtype(page.dtype)
    if len(shape) == 3 and shape[0] == 1:
        shape = shape[1:]
    elif len(shape) == 3 and shape[-1] == 1:
        shape = shape[:2]
    if len(shape) != 2:
        raise ValueError(f"{name} must be 2D. Got shape={shape}")
    return (int(shape[0]), int(shape[1])), dtype


def open_tiff_2d(path: str, name: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    try:
        arr = tifffile.memmap(path)
    except Exception:
        with tifffile.TiffFile(path) as tf:
            arr = tf.pages[0].asarray(out="memmap")
    arr = ensure_2d(arr, name)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int32)
    return arr

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


def row_blocks(n_rows: int, block_rows: int):
    block_rows = max(1, int(block_rows))
    for y0 in range(0, int(n_rows), block_rows):
        yield y0, min(int(n_rows), y0 + block_rows)


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


def downsample_seed_labels_mode(labels: np.ndarray, factor: int, block_rows_out: int) -> np.ndarray:
    f = max(1, int(factor))
    h, w = labels.shape
    out_h = int(np.ceil(h / f))
    out_w = int(np.ceil(w / f))
    out = np.zeros((out_h, out_w), dtype=labels.dtype)
    rows_out = max(1, int(block_rows_out))

    for oy0 in range(0, out_h, rows_out):
        oy1 = min(out_h, oy0 + rows_out)
        y0 = oy0 * f
        y1 = min(h, oy1 * f)
        block = np.asarray(labels[y0:y1, :])
        pad_h = (oy1 - oy0) * f - block.shape[0]
        pad_w = out_w * f - block.shape[1]
        if pad_h or pad_w:
            block = np.pad(block, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        reshaped = block.reshape((oy1 - oy0), f, out_w, f)
        ids = np.unique(block)
        ids = ids[ids != 0]
        if ids.size == 0:
            continue
        best = np.zeros((oy1 - oy0, out_w), dtype=labels.dtype)
        best_count = np.zeros((oy1 - oy0, out_w), dtype=np.uint16)
        for lab in ids:
            counts = (reshaped == lab).sum(axis=(1, 3)).astype(np.uint16, copy=False)
            take = counts > best_count
            best[take] = lab
            best_count[take] = counts[take]
        out[oy0:oy1, :] = best
    return out


def downsample_tissue_any(tissue: np.ndarray, factor: int, block_rows_out: int) -> np.ndarray:
    f = max(1, int(factor))
    h, w = tissue.shape
    out_h = int(np.ceil(h / f))
    out_w = int(np.ceil(w / f))
    out = np.zeros((out_h, out_w), dtype=bool)
    rows_out = max(1, int(block_rows_out))

    for oy0 in range(0, out_h, rows_out):
        oy1 = min(out_h, oy0 + rows_out)
        y0 = oy0 * f
        y1 = min(h, oy1 * f)
        block = np.asarray(tissue[y0:y1, :]) > 0
        pad_h = (oy1 - oy0) * f - block.shape[0]
        pad_w = out_w * f - block.shape[1]
        if pad_h or pad_w:
            block = np.pad(block, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
        out[oy0:oy1, :] = block.reshape((oy1 - oy0), f, out_w, f).any(axis=(1, 3))
    return out


def write_upsampled_lowres_mask(
    low_mask: np.ndarray,
    tissue_full: np.ndarray | None,
    out_path: str,
    full_shape: tuple[int, int],
    factor: int,
    dtype: np.dtype,
    block_rows: int,
    mpp_x: float,
    mpp_y: float,
) -> tuple[np.ndarray, int]:
    f = max(1, int(factor))
    h, w = full_shape
    out = create_tiff_memmap(
        out_path,
        shape=(h, w),
        dtype=dtype,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )
    nonzero = 0
    for y0, y1 in row_blocks(h, block_rows):
        ly0 = y0 // f
        ly1 = int(np.ceil(y1 / f))
        low_block = low_mask[ly0:ly1, :]
        up = np.repeat(np.repeat(low_block, f, axis=0), f, axis=1)
        y_offset = y0 - ly0 * f
        up = up[y_offset:y_offset + (y1 - y0), :w]
        up = up.astype(dtype, copy=False)
        if tissue_full is not None:
            tissue_block = np.asarray(tissue_full[y0:y1, :]) > 0
            up[~tissue_block] = 0
        out[y0:y1, :] = up
        nonzero += int(np.count_nonzero(up))
    out.flush()
    return out, nonzero


def grow_downsampled_to_fullres(
    seed_path: str,
    tissue_path: str,
    out_flat_tif: str,
    factor: int,
    block_rows: int,
    min_seed_area: int,
    restrict_to_seeded_components_flag: bool,
    fill_holes_area: int,
    close_radius: int,
    mpp_x: float,
    mpp_y: float,
) -> tuple[np.ndarray, float]:
    seeds_full = open_tiff_2d(seed_path, "seed mask")
    f = max(1, int(factor))
    tissue_arr = open_tiff_2d(tissue_path, "tissue mask")
    low_shape = (int(np.ceil(seeds_full.shape[0] / f)), int(np.ceil(seeds_full.shape[1] / f)))
    if tissue_arr.shape == seeds_full.shape:
        tissue_full_for_write = tissue_arr
        tissue_low = downsample_tissue_any(tissue_arr, f, max(1, int(np.ceil(block_rows / f))))
    elif tissue_arr.shape == low_shape:
        print(f"[INFO] Tissue mask is already on the downsampled grow grid: {tissue_arr.shape}")
        tissue_full_for_write = None
        tissue_low = np.asarray(tissue_arr) > 0
    else:
        raise ValueError(
            f"Downsampled grow requires tissue mask on the seed grid or requested low-res grid. "
            f"Got tissue={tissue_arr.shape}, seeds={seeds_full.shape}, expected_low={low_shape}."
        )

    rows_out = max(1, int(np.ceil(block_rows / f)))
    print(f"[INFO] Downsampled grow: full_shape={seeds_full.shape}, factor={f}, block_rows={block_rows}")
    seeds_low = downsample_seed_labels_mode(seeds_full, f, rows_out)
    print(f"[INFO] Downsampled grow grid: {seeds_low.shape}, seed_labels={np.unique(seeds_low[seeds_low > 0]).tolist()}")

    low_fill_holes_area = max(1, int(np.ceil(fill_holes_area / float(f * f)))) if fill_holes_area > 0 else 0
    low_close_radius = max(1, int(np.ceil(close_radius / float(f)))) if close_radius > 0 else 0
    tissue_low = improve_tissue_mask(
        tissue_bool=tissue_low,
        image_rgb=None,
        fill_holes_area=low_fill_holes_area,
        close_radius=low_close_radius,
        add_nuclei=False,
        nuclei_thresh=0,
        nuclei_dilate=0,
    )

    low_min_seed_area = max(1, int(np.ceil(min_seed_area / float(f * f)))) if min_seed_area > 1 else 0
    if low_min_seed_area > 1:
        seeds_low = remove_small_seed_components_per_label(seeds_low, low_min_seed_area)
    seeds_low[~tissue_low] = 0
    if restrict_to_seeded_components_flag:
        tissue_low = restrict_tissue_to_seeded_components(tissue_low, seeds_low)
        seeds_low[~tissue_low] = 0

    grown_low = grow_clusters_with_nearest_seed(seeds_low, tissue_low)
    maxlab = int(grown_low.max()) if grown_low.size else 0
    out_dtype = label_storage_dtype(maxlab)
    grown_full, nonzero = write_upsampled_lowres_mask(
        grown_low,
        tissue_full_for_write,
        out_flat_tif,
        full_shape=seeds_full.shape,
        factor=f,
        dtype=out_dtype,
        block_rows=block_rows,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )
    frac = float(nonzero / max(1, seeds_full.shape[0] * seeds_full.shape[1]))
    return grown_full, frac


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


def preview_metadata(path: str) -> tuple[tuple[int, ...], np.dtype, int]:
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        shape = tuple(page.shape)
        dtype = np.dtype(page.dtype)
    samples = 1
    if len(shape) == 3:
        if shape[-1] in (3, 4):
            samples = shape[-1]
        elif shape[0] in (3, 4):
            samples = shape[0]
    return shape, dtype, samples


def read_preview_background_small(path: str, expected_shape: tuple[int, int], factor: int) -> np.ndarray | None:
    f = max(1, int(factor))
    try:
        bg = tifffile.memmap(path)
        if bg.ndim > 3:
            bg = bg[0]
        if bg.ndim == 2:
            bg = np.asarray(bg[::f, ::f])
        elif bg.ndim == 3 and bg.shape[0] in (3, 4) and bg.shape[-1] not in (3, 4):
            bg = np.asarray(bg[:, ::f, ::f])
        elif bg.ndim == 3:
            bg = np.asarray(bg[::f, ::f, :])
        else:
            return None
    except Exception:
        if f > 1:
            return None
        bg = imread(path)
    if bg.ndim > 3:
        bg = bg[0]
    bg_rgb = to_uint8_rgb(bg)
    expected_small = ((expected_shape[0] + f - 1) // f, (expected_shape[1] + f - 1) // f)
    if bg_rgb.shape[:2] not in (expected_shape, expected_small):
        raise ValueError(
            f"Preview background shape {bg_rgb.shape[:2]} does not match grown mask shape {expected_shape} "
            f"or downsampled shape {expected_small}."
        )
    return bg_rgb


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

    bg_est_bytes = grown.shape[0] * grown.shape[1] * 3
    if image_path and os.path.exists(image_path):
        bg_shape, bg_dtype, bg_samples = preview_metadata(image_path)
        bg_est_bytes = int(np.prod(bg_shape[:2])) * bg_samples * int(bg_dtype.itemsize)
    est_bytes = int(bg_est_bytes + grown.nbytes)
    threshold_bytes = int(float(size_threshold_mb) * 1024 * 1024)
    use_factor = int(max(1, factor)) if est_bytes > threshold_bytes else 1

    grown_small = downsample_nearest(grown, use_factor)
    if image_path and os.path.exists(image_path):
        bg_small = read_preview_background_small(image_path, grown.shape, use_factor)
    else:
        bg_small = None
    if bg_small is None:
        bg_small = np.full(grown_small.shape + (3,), 255, dtype=np.uint8)
    elif bg_small.shape[:2] != grown_small.shape:
        bg_small = downsample_nearest(bg_small, use_factor)
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

    tool_env = os.environ.copy()
    if tool_env.get("JAVA_HOME") and not os.path.isdir(tool_env["JAVA_HOME"]):
        tool_env.pop("JAVA_HOME", None)

    try:
        print(f"[INFO] bioformats2raw -> {tmpdir}")
        subprocess.run(
            ["bioformats2raw", "--log-level=OFF", "--downsample-type", downsample, in_tif, tmpdir],
            check=True,
            env=tool_env,
        )

        r2o = ["raw2ometiff", f"--compression={compression}", f"--max_workers={max_workers}"]
        if legacy:
            r2o.append("--legacy")

        print(f"[INFO] raw2ometiff -> {out_ome_tif}")
        subprocess.run(r2o + [tmpdir, out_ome_tif], check=True, env=tool_env)

        print(f"[OK] pyramidal OME-TIFF written: {out_ome_tif}")

    finally:
        if not keep_tmp:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass


def _debug_dir_for(out_path: str | Path, suffix: str = 'medsam_debug') -> Path:
    out_path = Path(out_path)
    return out_path.parent / f"{out_path.stem}_{suffix}"


def main():
    ap = argparse.ArgumentParser(
        description="Grow cluster-labeled seeds inside tissue and export pyramidal + compressed mask via raw2ometiff."
    )
    ap.add_argument("--image", default=None, help="RGB image for preview + optional nuclei inclusion")
    ap.add_argument("--mask", required=True, help="Seed mask TIFF: integer cluster labels (0=background)")
    ap.add_argument("--tissue-mask", required=True, help="Binary tissue mask TIFF (nonzero=tissue)")
    ap.add_argument("--out", required=True, help="FINAL output pyramidal OME-TIFF (e.g. grown_mask.ome.tif)")
    ap.add_argument("--resolution-json", default="",
                    help="Pipeline shift/resolution JSON containing authoritative source_mpp metadata.")
    ap.add_argument("--default-mpp", type=float, default=0.0,
                    help="Fallback MPP used only when --resolution-json has no physical size.")
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
    ap.add_argument("--downsample", default="SIMPLE",
                    help="SIMPLE|GAUSSIAN|AREA|LINEAR|CUBIC|LANCZOS (default SIMPLE for categorical labels).")
    ap.add_argument("--legacy", action="store_true", help="Write Bio-Formats 5.9.x-compatible pyramid.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output.")
    ap.add_argument("--keep-tmp", action="store_true", help="Keep intermediate .rawdir.")
    ap.add_argument("--work-downsample", type=int, default=1,
                    help="For very large masks, grow on this downsampled grid and stream back to full resolution.")
    ap.add_argument("--fullres-max-pixels", type=int, default=50_000_000,
                    help="Use full-resolution EDT only up to this many pixels when --work-downsample > 1.")
    ap.add_argument("--block-rows", type=int, default=512,
                    help="Rows per block for memory-safe full-resolution writes.")

    ap.add_argument("--method", default="classic_existing", choices=["classic_existing", "medsam_border_refine"],
                    help="Step-16 method: original baseline or MedSAM border refinement.")
    ap.add_argument("--medsam-checkpoint", default=str(DEFAULT_MEDSAM_CHECKPOINT))
    ap.add_argument("--medsam-device", default="cuda")
    ap.add_argument("--medsam-bbox-margin", type=int, default=144)
    ap.add_argument("--medsam-component-min-area", type=int, default=200)
    ap.add_argument("--medsam-component-merge-distance", type=int, default=24)
    ap.add_argument("--medsam-seed-dilation-radius", type=int, default=8)
    ap.add_argument("--medsam-core-erosion-radius", type=int, default=36)
    ap.add_argument("--medsam-outer-dilation-radius", type=int, default=44)
    ap.add_argument("--medsam-min-object-size", type=int, default=5000)
    ap.add_argument("--medsam-smooth-radius", type=int, default=2)
    ap.add_argument("--medsam-force-core-preservation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--medsam-save-debug", action=argparse.BooleanOptionalAction, default=False)

    args = ap.parse_args()

    mpp_x, mpp_y = read_mpp_json(args.resolution_json, args.default_mpp)
    print(f"[INFO] Output physical resolution: mpp_x={mpp_x:.9g}, mpp_y={mpp_y:.9g}")

    seed_shape, _ = tiff_2d_shape_dtype(args.mask, "seed mask")
    seed_pixels = int(seed_shape[0] * seed_shape[1])
    use_downsampled_classic = (
        args.method == "classic_existing"
        and int(args.work_downsample) > 1
        and seed_pixels > int(args.fullres_max_pixels)
    )
    if use_downsampled_classic:
        print(
            f"[INFO] Large mask detected ({seed_pixels:,} px > {int(args.fullres_max_pixels):,}); "
            f"using downsampled classic grow with factor={int(args.work_downsample)}."
        )
        outdir = os.path.dirname(args.out) or "."
        os.makedirs(outdir, exist_ok=True)
        tmp_flat = os.path.join(outdir, f".tmp_flat_{os.getpid()}_{int(time.time())}.tif")
        grown, frac = grow_downsampled_to_fullres(
            seed_path=args.mask,
            tissue_path=args.tissue_mask,
            out_flat_tif=tmp_flat,
            factor=int(args.work_downsample),
            block_rows=int(args.block_rows),
            min_seed_area=int(args.min_seed_area),
            restrict_to_seeded_components_flag=bool(args.restrict_to_seeded_components),
            fill_holes_area=int(args.fill_holes_area),
            close_radius=int(args.close_radius),
            mpp_x=mpp_x,
            mpp_y=mpp_y,
        )
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
            ome_summary = validate_ome_tiff(
                args.out,
                expected_shape=seed_shape,
                expected_mpp=(mpp_x, mpp_y),
            )
            print(f"[INFO] Validated grown OME-TIFF: {json.dumps(ome_summary, sort_keys=True)}")
        except Exception:
            if os.path.exists(args.out):
                os.remove(args.out)
            raise
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
        try:
            os.remove(tmp_flat)
        except Exception:
            pass
        print(f"[OK] method={args.method} downsampled_factor={int(args.work_downsample)} labeled tissue fraction: {frac:.4f}")
        print(f"[OK] done: {args.out}")
        return

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

    # Original step-16 baseline grown mask from the current tissue mask.
    classic_grown = grow_clusters_with_nearest_seed(seeds, tissue)

    if args.method == "classic_existing":
        grown = classic_grown
        print(f"[INFO] step16 method={args.method}")
    elif args.method == "medsam_border_refine":
        if image_rgb is None:
            raise RuntimeError("MedSAM border refinement requires --image so the border can be refined from the ROI crop.")
        MedSAMConfig, MedSAMUnavailableError, run_medsam_border_refine = load_medsam_runtime()
        print(f"[INFO] step16 method={args.method}")
        med_cfg = MedSAMConfig(
            checkpoint=str(args.medsam_checkpoint),
            device=str(args.medsam_device),
            bbox_margin=int(args.medsam_bbox_margin),
            component_min_area=int(args.medsam_component_min_area),
            component_merge_distance=int(args.medsam_component_merge_distance),
            seed_dilation_radius=int(args.medsam_seed_dilation_radius),
            core_erosion_radius=int(args.medsam_core_erosion_radius),
            outer_dilation_radius=int(args.medsam_outer_dilation_radius),
            min_object_size=int(args.medsam_min_object_size),
            smooth_radius=int(args.medsam_smooth_radius),
            force_core_preservation=bool(args.medsam_force_core_preservation),
            save_debug=bool(args.medsam_save_debug),
        )
        debug_dir = _debug_dir_for(args.out, "medsam_debug")
        baseline_tissue = classic_grown > 0
        try:
            refined_tissue, probability_map, runtime_sec, med_meta, artifacts = run_medsam_border_refine(
                image=image_rgb,
                seed_labels=seeds,
                baseline_tissue_mask=baseline_tissue,
                config=med_cfg,
                baseline_label_map=classic_grown,
            )
        except MedSAMUnavailableError as exc:
            if med_cfg.save_debug:
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / "ERROR.txt").write_text(f"{exc}\n")
            raise RuntimeError(f"MedSAM unavailable: {exc}") from exc
        label_map = np.asarray(artifacts.get("label_map")) if isinstance(artifacts, dict) and "label_map" in artifacts else None
        if label_map is not None and label_map.shape == seeds.shape:
            grown = label_map.astype(seeds.dtype, copy=False)
        else:
            grown = grow_clusters_with_nearest_seed(seeds, refined_tissue)
        if med_cfg.save_debug:
            debug_dir.mkdir(parents=True, exist_ok=True)
            meta = dict(med_meta)
            meta["classic_existing_pixels"] = int(baseline_tissue.sum())
            meta["grown_pixels"] = int((grown > 0).sum())
            (debug_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            np.save(debug_dir / "probability_map.npy", probability_map.astype(np.float32))
            for name, arr in artifacts.items():
                np.save(debug_dir / f"{name}.npy", np.asarray(arr))
        print(f"[INFO] medsam runtime={runtime_sec:.2f}s baseline_pixels={int(baseline_tissue.sum())} refined_pixels={int((grown > 0).sum())}")
    else:
        raise RuntimeError(f"Unsupported method: {args.method}")

    # Write a temporary single-level label TIFF (raw2ometiff will pyramidize/compress)
    outdir = os.path.dirname(args.out) or "."
    os.makedirs(outdir, exist_ok=True)
    tmp_flat = os.path.join(outdir, f".tmp_flat_{os.getpid()}_{int(time.time())}.tif")

    maxlab = int(grown.max()) if grown.size else 0
    flat = grown.astype(label_storage_dtype(maxlab))

    imwrite(
        tmp_flat,
        flat,
        bigtiff=True,
        byteorder=flat.dtype.byteorder if flat.dtype.itemsize > 1 else None,
        **tiff_resolution_kwargs(mpp_x, mpp_y, "YX"),
    )

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
        ome_summary = validate_ome_tiff(
            args.out,
            expected_shape=seeds.shape,
            expected_mpp=(mpp_x, mpp_y),
        )
        print(f"[INFO] Validated grown OME-TIFF: {json.dumps(ome_summary, sort_keys=True)}")
    except Exception:
        if os.path.exists(args.out):
            os.remove(args.out)
        raise

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
    print(f"[OK] method={args.method} labeled tissue fraction: {frac:.4f}")
    print(f"[OK] done: {args.out}")


if __name__ == "__main__":
    main()
