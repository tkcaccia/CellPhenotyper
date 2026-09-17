"""Utilities for filtering crop-coordinate detections with a GrandQC mask."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile


def read_mask(path: str | Path) -> np.ndarray:
    try:
        return np.asarray(tifffile.memmap(str(path)))
    except ValueError:
        # GrandQC masks are low-resolution compressed TIFFs and therefore are
        # commonly not directly memory-mappable.
        return np.asarray(tifffile.imread(str(path)))


class CropCleanTissueMask:
    def __init__(self, mask_path: str | Path, shift_path: str | Path):
        self.mask = read_mask(mask_path)
        self.mask = np.squeeze(self.mask)
        if self.mask.ndim != 2:
            raise RuntimeError(f"Clean-tissue mask must be 2D, got {self.mask.shape}")
        shift = json.loads(Path(shift_path).read_text())
        crop = shift["crop_size"]
        self.crop_width = int(crop["width"])
        self.crop_height = int(crop["height"])
        self.mask_height, self.mask_width = map(int, self.mask.shape)

    def contains_xy(self, x: float, y: float) -> bool:
        if x < 0 or y < 0 or x >= self.crop_width or y >= self.crop_height:
            return False
        mx = min(self.mask_width - 1, int(float(x) * self.mask_width / self.crop_width))
        my = min(self.mask_height - 1, int(float(y) * self.mask_height / self.crop_height))
        return bool(self.mask[my, mx] != 0)

    def keep_xy(self, coordinates) -> np.ndarray:
        return np.fromiter(
            (self.contains_xy(float(x), float(y)) for x, y in coordinates),
            dtype=bool,
        )


def filter_cell_payload(payload: dict, mask_path: str | Path, shift_path: str | Path) -> tuple[dict, dict]:
    clean_mask = CropCleanTissueMask(mask_path, shift_path)
    cells = payload.get("cells", [])
    kept = []
    for cell in cells:
        centroid = cell.get("centroid") if isinstance(cell, dict) else None
        if centroid and len(centroid) >= 2 and clean_mask.contains_xy(centroid[0], centroid[1]):
            kept.append(cell)
    result = dict(payload)
    result["cells"] = kept
    summary = {
        "input_cells": int(len(cells)),
        "retained_cells": int(len(kept)),
        "removed_outside_grandqc_clean_tissue": int(len(cells) - len(kept)),
    }
    result.setdefault("pipeline_metadata", {})["grandqc_filter"] = summary
    return result, summary


def filter_labels_in_place(labels, keep_ids, shape: tuple[int, int], block_size: int = 1024) -> None:
    max_id = max((int(value) for value in keep_ids), default=0)
    lookup = np.zeros(max_id + 1, dtype=bool)
    if max_id:
        lookup[np.fromiter((int(value) for value in keep_ids), dtype=np.int64)] = True
    height, width = map(int, shape)
    for y0 in range(0, height, block_size):
        y1 = min(height, y0 + block_size)
        for x0 in range(0, width, block_size):
            x1 = min(width, x0 + block_size)
            block = np.asarray(labels[y0:y1, x0:x1]).copy()
            valid = block <= max_id
            keep = np.zeros(block.shape, dtype=bool)
            keep[valid] = lookup[block[valid]]
            block[~keep] = 0
            labels[y0:y1, x0:x1] = block
