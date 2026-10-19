#!/usr/bin/env python3
"""Crop a low-resolution GrandQC clean-tissue mask to the analysis ROI."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tifffile

from prepare_analysis_crop import rasterize_roi_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True)
    parser.add_argument("--shift", required=True)
    parser.add_argument("--roi", required=True, help="Crop-coordinate ROI GeoJSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    shift = json.loads(Path(args.shift).read_text())
    bbox = shift["crop_bbox_xyxy"]
    full = shift["full_size"]
    try:
        mask = np.asarray(tifffile.memmap(args.mask))
    except ValueError:
        mask = np.asarray(tifffile.imread(args.mask))
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise RuntimeError(f"GrandQC clean-tissue mask must be 2D, got {mask.shape}")
    mask_h, mask_w = map(int, mask.shape)
    full_w, full_h = int(full["width"]), int(full["height"])
    qx0 = max(0, min(mask_w - 1, int(math.floor(int(bbox["x0"]) * mask_w / full_w))))
    qy0 = max(0, min(mask_h - 1, int(math.floor(int(bbox["y0"]) * mask_h / full_h))))
    qx1 = max(qx0 + 1, min(mask_w, int(math.ceil(int(bbox["x1"]) * mask_w / full_w))))
    qy1 = max(qy0 + 1, min(mask_h, int(math.ceil(int(bbox["y1"]) * mask_h / full_h))))
    cropped = mask[qy0:qy1, qx0:qx1] != 0
    crop_size = shift["crop_size"]
    roi = json.loads(Path(args.roi).read_text())
    roi_mask = rasterize_roi_mask(
        roi,
        cropped.shape,
        (int(crop_size["width"]), int(crop_size["height"])),
    )
    cropped = np.where(cropped & roi_mask, 255, 0).astype(np.uint8)
    if not np.any(cropped):
        raise RuntimeError("Cropped GrandQC clean tissue and ROI do not overlap")
    tifffile.imwrite(args.output, cropped, compression="zlib")

    summary = {
        "source_mask": str(Path(args.mask).resolve()),
        "source_mask_shape_yx": [mask_h, mask_w],
        "source_mask_bbox_xyxy": [qx0, qy0, qx1, qy1],
        "output_shape_yx": list(map(int, cropped.shape)),
        "analysis_crop_shape_yx": [int(crop_size["height"]), int(crop_size["width"])],
        "scale_mask_per_crop_x": float(cropped.shape[1] / int(crop_size["width"])),
        "scale_mask_per_crop_y": float(cropped.shape[0] / int(crop_size["height"])),
        "clean_tissue_fraction": float(np.count_nonzero(cropped) / max(1, cropped.size)),
        "mask_contract": "grandqc_clean_tissue_intersection_roi",
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"[OK] GrandQC ROI mask={cropped.shape[1]}x{cropped.shape[0]} clean={summary['clean_tissue_fraction']:.4f}")


if __name__ == "__main__":
    main()
