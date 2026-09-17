#!/usr/bin/env python3
"""Grid assignment helpers shared by UNI2 extraction and tests."""

from __future__ import annotations

import math

import numpy as np


def resolve_spatial_grid_geometry(
    *,
    model_tile_size: int,
    inner_square_size: int,
    grid_stride_size: int = 0,
    source_mpp: float,
    target_mpp: float,
) -> tuple[int, int, int]:
    """Return source-pixel context, inner-square size and independent grid stride."""
    if model_tile_size <= 0 or inner_square_size <= 0:
        raise ValueError("Model tile and inner-square sizes must be positive")
    if inner_square_size > model_tile_size:
        raise ValueError("Inner-square size cannot exceed the model tile size")
    if grid_stride_size < 0 or grid_stride_size > model_tile_size:
        raise ValueError("Grid stride must be zero (automatic) or no larger than the model tile size")
    if source_mpp <= 0 or target_mpp <= 0:
        raise ValueError("Source and target MPP must be positive")

    extraction_tile_size = max(1, int(round(model_tile_size * target_mpp / source_mpp)))
    inner_square_source_size = max(
        1, int(round(inner_square_size * extraction_tile_size / model_tile_size))
    )
    effective_stride_size = grid_stride_size if grid_stride_size > 0 else inner_square_size
    stride = max(1, int(round(effective_stride_size * extraction_tile_size / model_tile_size)))
    return extraction_tile_size, min(inner_square_source_size, extraction_tile_size), min(stride, extraction_tile_size)


def centered_axis_lattice(length: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build adjacent logical cores with centres kept inside the image."""
    if length <= 0 or stride <= 0:
        raise ValueError("Axis length and stride must be positive")

    count = max(1, int(math.ceil(length / stride)))
    first_center = int(math.ceil((length - (count - 1) * stride) / 2.0))
    centers = first_center + np.arange(count, dtype=np.int64) * int(stride)
    if centers[-1] >= length:
        centers -= int(centers[-1] - (length - 1))
    if centers[0] < 0:
        centers -= int(centers[0])

    starts = centers - int(stride // 2)
    ends = starts + int(stride)
    if count > 1 and not np.array_equal(starts[1:], ends[:-1]):
        raise RuntimeError("Grid core construction produced a gap or overlap")
    if starts[0] > 0 or ends[-1] < length:
        raise RuntimeError("Grid cores do not cover the complete image axis")
    return centers, starts, ends


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
