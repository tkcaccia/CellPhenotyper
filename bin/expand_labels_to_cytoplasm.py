#!/usr/bin/env python3

"""
expand_cells_to_cytoplasm_v3.py

Expand (dilate) cell label masks to approximate cytoplasmic area, while guaranteeing:
- labels NEVER overlap (each pixel is assigned to at most one cell)
- expansion is bounded by a user-specified distance (in pixels)

Method
------
We compute an Euclidean distance transform (EDT) of the background (label==0) and
retrieve, for each background pixel, the nearest labeled pixel. Background pixels within
--expand-px of any cell are reassigned to that cell's label. This produces a Voronoi-like
non-overlapping expansion.

Input
-----
- A 2D integer label image (e.g., StarDist output 'labels.tif')

Output
------
- Expanded label image (same label IDs, larger regions)

Example
-------
python expand_cells_to_cytoplasm_v3.py \
  --labels out_stardist_roi/labels.tif \
  --out out_stardist_roi/labels_cyto.tif \
  --expand-px 12
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import tifffile

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None

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


def die(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def read_label_image(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))

    # Accept common shapes: (H,W), (1,H,W), (H,W,1)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            die(f"Expected 2D label image; got shape {arr.shape} from {path}")

    if arr.ndim != 2:
        die(f"Expected 2D label image; got shape {arr.shape} from {path}")

    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int32)

    return arr


def expand_labels_nonoverlap(labels: np.ndarray, expand_px: int) -> np.ndarray:
    if distance_transform_edt is None:
        die("SciPy is required (scipy.ndimage.distance_transform_edt). It is missing in your environment.")

    if expand_px <= 0:
        return labels.copy()

    lab = labels.copy()
    bg = (lab == 0)

    # If there is no background, nothing to fill
    if not bg.any():
        return lab

    # If there are no labels at all, nothing to expand
    if lab.max() == 0:
        return lab

    # Distance to nearest labeled pixel + indices of that nearest labeled pixel
    dist, (iy, ix) = distance_transform_edt(bg, return_indices=True)

    nearest = lab[iy, ix]
    out = lab.copy()

    # Fill only background pixels within expansion distance
    to_fill = bg & (dist <= float(expand_px))
    out[to_fill] = nearest[to_fill]

    return out


def expand_labels_nonoverlap_tiled(labels: np.ndarray, expand_px: int, tile_size: int) -> np.ndarray:
    """
    Memory-bounded expansion for large label images.

    Each core tile is processed with a halo == expand_px. This guarantees exact
    assignments for pixels in the core tile because only labels within expand_px
    can influence the final fill decision.
    """
    if distance_transform_edt is None:
        die("SciPy is required (scipy.ndimage.distance_transform_edt). It is missing in your environment.")

    if expand_px <= 0:
        return labels.copy()
    if tile_size <= 0:
        die("--tile-size must be > 0")

    lab = labels
    out = lab.copy()
    h, w = lab.shape
    radius = int(expand_px)
    radius_f = float(expand_px)

    n_tiles_y = math.ceil(h / tile_size)
    n_tiles_x = math.ceil(w / tile_size)
    n_tiles = n_tiles_y * n_tiles_x
    idx = 0

    for y0 in range(0, h, tile_size):
        y1 = min(h, y0 + tile_size)
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0 + tile_size)
            idx += 1
            log(f"Tiled expansion: tile {idx}/{n_tiles} at y={y0}:{y1}, x={x0}:{x1}")

            core = lab[y0:y1, x0:x1]
            core_bg = (core == 0)
            if not core_bg.any():
                continue

            hy0 = max(0, y0 - radius)
            hy1 = min(h, y1 + radius)
            hx0 = max(0, x0 - radius)
            hx1 = min(w, x1 + radius)

            win = lab[hy0:hy1, hx0:hx1]
            if win.max() == 0:
                continue

            win_bg = (win == 0)
            dist, (iy, ix) = distance_transform_edt(win_bg, return_indices=True)

            cy0 = y0 - hy0
            cy1 = cy0 + (y1 - y0)
            cx0 = x0 - hx0
            cx1 = cx0 + (x1 - x0)

            core_within = dist[cy0:cy1, cx0:cx1] <= radius_f
            fill = core_bg & core_within
            if not fill.any():
                continue

            iyy = iy[cy0:cy1, cx0:cx1]
            ixx = ix[cy0:cy1, cx0:cx1]
            nearest_core = win[iyy, ixx]

            out_tile = out[y0:y1, x0:x1]
            out_tile[fill] = nearest_core[fill]

    return out


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


def write_preview_overlay_png(
    label_mask: np.ndarray,
    out_png: Path,
    factor_if_large: int,
    size_threshold_mb: float,
    default_value: int,
    preview_background_path: Path | None,
    alpha: float,
) -> None:
    if Image is None:
        log("Preview requested but pillow is unavailable; skipping preview generation.")
        return

    threshold_bytes = int(float(size_threshold_mb) * 1024 * 1024)
    use_factor = 1

    if preview_background_path is not None:
        bg = tifffile.imread(str(preview_background_path))
        if bg.ndim > 3:
            bg = bg[0]
        bg_rgb = to_uint8_rgb(bg)
        if bg_rgb.shape[:2] != label_mask.shape:
            die(
                f"Preview background shape {bg_rgb.shape[:2]} does not match label mask shape {label_mask.shape}. "
                "Expected aligned background image."
            )

        est_bytes = int(bg_rgb.nbytes + label_mask.nbytes)
        use_factor = int(max(1, factor_if_large)) if est_bytes > threshold_bytes else 1
        bg_small = downsample_nearest(bg_rgb, use_factor)
        mask_small = downsample_nearest(label_mask, use_factor)
    else:
        # Memory-safe fallback for huge masks: estimate full preview memory and
        # decide downsample factor before allocating any RGB background.
        est_bytes = int(label_mask.nbytes * 4)
        use_factor = int(max(1, factor_if_large)) if est_bytes > threshold_bytes else 1
        mask_small = downsample_nearest(label_mask, use_factor)
        bg_small = np.full(mask_small.shape + (3,), 255, dtype=np.uint8)

    overlay = colorize_label_mask(mask_small, default_value=default_value)
    fg = mask_small != default_value

    out = bg_small.astype(np.float32, copy=True)
    a = float(max(0.0, min(1.0, alpha)))
    out[fg] = (1.0 - a) * out[fg] + a * overlay[fg].astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(str(out_png))
    log(
        f"Wrote cytoplasm overlay preview: {out_png} "
        f"(factor={use_factor}, threshold_mb={size_threshold_mb}, "
        f"estimated_mb={est_bytes / (1024.0 * 1024.0):.1f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand cell label masks to approximate cytoplasm without overlaps.")
    ap.add_argument("--labels", required=True, help="Input label image (2D), e.g. StarDist labels.tif")
    ap.add_argument("--out", required=True, help="Output expanded label image (.tif)")
    ap.add_argument("--expand-px", type=int, required=True, help="Expansion radius in pixels (e.g. 8-20)")
    ap.add_argument("--mode", choices=["auto", "full", "tiled"], default="auto",
                    help="Expansion mode (default: auto).")
    ap.add_argument("--tile-size", type=int, default=2048,
                    help="Core tile size for tiled mode (default: 2048).")
    ap.add_argument("--auto-threshold-mpix", type=float, default=25.0,
                    help="In auto mode, switch to tiled if image > this megapixel threshold (default: 25).")
    ap.add_argument("--compression", default="zlib",
                    help="TIFF compression (default: zlib; use 'none' for no compression)")
    ap.add_argument("--preview", default=None, help="Optional output preview PNG path.")
    ap.add_argument("--preview-background", default=None,
                    help="Optional background TIFF for overlay preview (must align with labels).")
    ap.add_argument("--preview-factor", type=int, default=10,
                    help="Downsample factor used only when preview image is larger than threshold.")
    ap.add_argument("--preview-threshold-mb", type=float, default=100.0,
                    help="Downsample preview only when estimated image+mask size exceeds this threshold (MB).")
    ap.add_argument("--preview-alpha", type=float, default=0.45,
                    help="Overlay alpha for colored cytoplasm mask preview (0..1).")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    out_path = Path(args.out)

    if not labels_path.exists():
        die(f"Labels file not found: {labels_path}")

    labels = read_label_image(labels_path)
    log(f"Loaded labels: {labels.shape}, dtype={labels.dtype}, n_labels={int(labels.max())}")

    h, w = labels.shape
    mpix = (h * w) / 1_000_000.0

    mode = args.mode
    if mode == "auto":
        mode = "tiled" if mpix > float(args.auto_threshold_mpix) else "full"
    log(f"Expansion mode: {mode} (image={mpix:.2f} MP)")

    if mode == "tiled":
        out = expand_labels_nonoverlap_tiled(labels, args.expand_px, args.tile_size)
    else:
        if h * w >= 100_000_000:
            log("Note: full EDT mode can use a lot of RAM on very large images.")
        out = expand_labels_nonoverlap(labels, args.expand_px)

    if out.dtype != labels.dtype:
        out = out.astype(labels.dtype, copy=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    comp = args.compression
    if isinstance(comp, str) and comp.lower() == "none":
        comp = None

    bigtiff = out.nbytes >= 4_000_000_000
    tifffile.imwrite(str(out_path), out, compression=comp, bigtiff=bigtiff)
    log(f"Wrote expanded labels: {out_path}")
    log(f"Output n_labels={int(out.max())}; filled_pixels={(out>0).sum()}")

    if args.preview:
        bg_path = None
        if args.preview_background:
            bg_path = Path(args.preview_background)
            if not bg_path.exists():
                die(f"Preview background file not found: {bg_path}")

        write_preview_overlay_png(
            out,
            Path(args.preview),
            factor_if_large=int(args.preview_factor),
            size_threshold_mb=float(args.preview_threshold_mb),
            default_value=0,
            preview_background_path=bg_path,
            alpha=float(args.preview_alpha),
        )

if __name__ == "__main__":
    main()
