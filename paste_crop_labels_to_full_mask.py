#!/usr/bin/env python3
"""
paste_crop_labels_to_full_mask.py

Paste a cropped label image (e.g. out_stardist_roi/labels.tif) back into the
original image coordinate space using the shift.json produced by
roi_crop_and_stardist_segment_only_v9.py.

This is useful if a downstream tool expects labels in the original (level-0)
coordinate system.

For large WSIs, writing a dense full-size TIFF can be impractical. Prefer Zarr.

Examples:
  # Recommended (chunked, sparse on disk outside the crop region)
  python paste_crop_labels_to_full_mask.py \
    --labels out_stardist_roi/labels.tif \
    --shift  out_stardist_roi/shift.json \
    --out    out_stardist_roi/labels_full.zarr \
    --format zarr

  # Dense TIFF (only for smaller images, or add --allow-huge-tif)
  python paste_crop_labels_to_full_mask.py \
    --labels out_stardist_roi/labels.tif \
    --shift  out_stardist_roi/shift.json \
    --out    out_stardist_roi/labels_full.tif \
    --format tif
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile


def die(msg: str, code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def write_full_labels_from_crop(labels_crop: np.ndarray,
                               full_h: int,
                               full_w: int,
                               x0: int, y0: int, x1: int, y1: int,
                               out_path: Path,
                               fmt: str = "zarr",
                               chunk: int = 2048,
                               compression: str = "zlib",
                               allow_huge_tif: bool = False) -> None:
    if labels_crop.ndim != 2:
        die(f"labels_crop must be 2D, got shape={labels_crop.shape}")
    if (y1 - y0) != labels_crop.shape[0] or (x1 - x0) != labels_crop.shape[1]:
        die("Crop bbox does not match labels shape. "
            f"bbox size={(y1-y0)}x{(x1-x0)} labels={labels_crop.shape}")

    max_lab = int(labels_crop.max()) if labels_crop.size else 0
    dtype_out = np.uint16 if max_lab <= np.iinfo(np.uint16).max else np.uint32
    full_pixels = int(full_h) * int(full_w)

    fmt = (fmt or "").lower()
    if fmt == "zarr":
        try:
            import zarr
        except Exception as e:
            die(f"zarr is required for --format zarr, but import failed: {e}")

        out_path = Path(out_path)
        log(f"[FULL] Writing Zarr full labels: {out_path} (shape={full_h}x{full_w}, chunk={chunk}, dtype={dtype_out})")

        z = zarr.open(
            str(out_path),
            mode="w",
            shape=(int(full_h), int(full_w)),
            chunks=(int(chunk), int(chunk)),
            dtype=dtype_out,
            overwrite=True,
        )
        z[int(y0):int(y1), int(x0):int(x1)] = labels_crop.astype(dtype_out, copy=False)
        log("[FULL] Done writing Zarr full labels (only crop region materialized).")
        return

    if fmt == "tif":
        thresh = 500_000_000  # 0.5 billion pixels
        if full_pixels > thresh and not allow_huge_tif:
            die(
                "Refusing to write dense full TIFF because the full image is very large "
                f"({full_w}x{full_h}={full_pixels:,} pixels). "
                "Use --format zarr (recommended), or add --allow-huge-tif to force TIFF."
            )

        log(f"[FULL] Writing dense TIFF full labels: {out_path} (shape={full_h}x{full_w}, dtype={dtype_out})")
        full = np.zeros((int(full_h), int(full_w)), dtype=dtype_out)
        full[int(y0):int(y1), int(x0):int(x1)] = labels_crop.astype(dtype_out, copy=False)

        tifffile.imwrite(out_path, full, compression=compression, bigtiff=True)
        del full
        log("[FULL] Done writing full TIFF labels.")
        return

    die(f"Unknown format: {fmt}. Use 'zarr' or 'tif'.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Paste crop labels back into full image coordinate space using shift.json.")
    ap.add_argument("--labels", required=True, help="Cropped label image (e.g. outdir/labels.tif)")
    ap.add_argument("--shift", required=True, help="shift.json produced by roi_crop_and_stardist_segment_only_v9.py")
    ap.add_argument("--out", required=True, help="Output path (.zarr directory or .tif file)")
    ap.add_argument("--format", choices=("zarr", "tif"), default="zarr", help="Output format. Default zarr.")
    ap.add_argument("--chunk", type=int, default=2048, help="Chunk size for Zarr output (pixels). Default 2048.")
    ap.add_argument("--allow-huge-tif", action="store_true", default=False, help="Allow creating very large dense TIFFs.")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    shift_path = Path(args.shift)
    out_path = Path(args.out)

    if not labels_path.exists():
        die(f"Labels not found: {labels_path}")
    if not shift_path.exists():
        die(f"shift.json not found: {shift_path}")

    labels = tifffile.imread(labels_path)
    if labels.ndim != 2:
        die(f"Expected 2D labels, got shape={labels.shape}")
    log(f"Loaded labels: shape={labels.shape}, dtype={labels.dtype}, max={int(labels.max()) if labels.size else 0}")

    shift = json.loads(shift_path.read_text())
    bbox = shift.get("crop_bbox_xyxy", {})
    fs = shift.get("full_size", None)
    if not fs:
        die("shift.json does not contain 'full_size'. Re-run roi_crop_and_stardist_segment_only_v9.py (newer) "
            "or manually provide full dimensions in shift.json.")
    try:
        x0 = int(bbox["x0"]); y0 = int(bbox["y0"]); x1 = int(bbox["x1"]); y1 = int(bbox["y1"])
        full_w = int(fs["width"]); full_h = int(fs["height"])
    except Exception as e:
        die(f"shift.json missing required keys (crop_bbox_xyxy and full_size): {e}")

    write_full_labels_from_crop(
        labels_crop=labels,
        full_h=full_h, full_w=full_w,
        x0=x0, y0=y0, x1=x1, y1=y1,
        out_path=out_path,
        fmt=args.format,
        chunk=args.chunk,
        allow_huge_tif=args.allow_huge_tif,
    )


if __name__ == "__main__":
    main()
