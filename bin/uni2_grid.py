#!/usr/bin/env python3
"""Grid assignment helpers shared by UNI2 extraction and tests."""

from __future__ import annotations

import math

import numpy as np


def assign_rounded_centers_to_grid(
    cx: np.ndarray,
    cy: np.ndarray,
    *,
    height: int,
    width: int,
    grid_rows: int,
    grid_cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    if height <= 0 or width <= 0 or grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("Image and grid dimensions must be positive")
    center_x = np.clip(np.rint(np.asarray(cx)).astype(np.int64), 0, width - 1)
    center_y = np.clip(np.rint(np.asarray(cy)).astype(np.int64), 0, height - 1)
    tile_w = int(math.ceil(width / grid_cols))
    tile_h = int(math.ceil(height / grid_rows))
    grid_r = np.clip(center_y // tile_h, 0, grid_rows - 1)
    grid_c = np.clip(center_x // tile_w, 0, grid_cols - 1)
    tile_id = grid_r * grid_cols + grid_c
    return center_x, center_y, grid_r, grid_c, tile_id, tile_w, tile_h
