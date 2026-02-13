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


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand cell label masks to approximate cytoplasm without overlaps.")
    ap.add_argument("--labels", required=True, help="Input label image (2D), e.g. StarDist labels.tif")
    ap.add_argument("--out", required=True, help="Output expanded label image (.tif)")
    ap.add_argument("--expand-px", type=int, required=True, help="Expansion radius in pixels (e.g. 8-20)")
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
    if h * w >= 100_000_000:
        log("Note: EDT-based expansion can use a lot of RAM on very large images (often multiple GB).")

    out = expand_labels_nonoverlap(labels, args.expand_px)

    if out.dtype != labels.dtype:
        out = out.astype(labels.dtype, copy=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    comp = args.compression
    if isinstance(comp, str) and comp.lower() == "none":
        comp = None

    tifffile.imwrite(str(out_path), out, compression=comp)
    log(f"Wrote expanded labels: {out_path}")
    log(f"Output n_labels={int(out.max())}; filled_pixels={(out>0).sum()}")

if __name__ == "__main__":
    main()
