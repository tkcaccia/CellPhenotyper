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

if __name__ == "__main__":
    main()
