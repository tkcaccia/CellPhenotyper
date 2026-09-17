#!/usr/bin/env python3
"""Physical, tissue-constrained cell neighbourhoods and exploratory cellular niches.

Cell coordinates and mask calibration MUST be in the same specimen coordinate
frame. Pixel (0, 0) occupies [origin_x, origin_x + mpp_x) in x, likewise y.
Unknown phenotypes are retained as an explicit category. Niche IDs are discovery
labels, independent of tissue-domain IDs; neither stability nor centroid margin
is a calibrated biological probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from cell_profile_io import RasterReader


KEYS = ["sample_id", "cell_id"]
UNKNOWN = "__unknown__"
FEATURE_REPRESENTATION_VERSION = "2.0.0"


@dataclass
class SupportMask:
    """Boolean tissue support with physical pixel size and top-left pixel edge."""

    mask: np.ndarray
    mpp: tuple[float, float] = (1.0, 1.0)
    origin_um: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        raw = np.asarray(self.mask)
        if not np.all(np.isfinite(raw)) or np.any(raw < 0):
            raise ValueError("Support values must be finite and nonnegative; zero is background")
        self.mask = raw > 0
        self.mpp = tuple(float(v) for v in self.mpp)
        self.origin_um = tuple(float(v) for v in self.origin_um)
        if self.mask.ndim != 2 or min(self.mask.shape) == 0:
            raise ValueError("Support must be a non-empty 2D mask")
        if len(self.mpp) != 2 or not np.all(np.isfinite(self.mpp)) or min(self.mpp) <= 0:
            raise ValueError("Support mpp must contain positive finite x,y values")
        if len(self.origin_um) != 2 or not np.all(np.isfinite(self.origin_um)):
            raise ValueError("Support origin must contain finite x,y values")
        # Four-connectivity prevents joining pieces that touch only at a corner.
        self.components, _ = ndimage.label(self.mask)
        self._row_prefix: np.ndarray | None = None

    def pixels(self, xy_um: np.ndarray) -> np.ndarray:
        return (np.asarray(xy_um, dtype=float) - self.origin_um) / self.mpp

    @property
    def shape(self):
        return self.mask.shape

    def is_tissue(self, x, y):
        return 0 <= y < self.shape[0] and 0 <= x < self.shape[1] and bool(self.mask[y, x])

    def all_tissue_in_box(self, x0, y0, x1, y1):
        return bool(np.all(self.mask[y0:y1, x0:x1]))

    def grid_tissue(self, shape):
        y = np.floor((np.arange(shape[0]) + .5) * self.shape[0] / shape[0]).astype(int)
        x = np.floor((np.arange(shape[1]) + .5) * self.shape[1] / shape[1]).astype(int)
        return self.mask[np.ix_(y, x)]

    def component_at(self, pixel: np.ndarray) -> int:
        x, y = np.floor(pixel).astype(np.int64)
        if 0 <= y < self.mask.shape[0] and 0 <= x < self.mask.shape[1]:
            return int(self.components[y, x])
        return 0

    def tissue_area_in_disk(self, xy_um: np.ndarray, radius_um: float) -> float:
        """Tissue area in a disk, counting pixel centres (units: square micrometres).

        This corrects the density denominator for slide edges/background holes.
        It is NOT geodesic visibility area: the graph also rejects line-of-sight
        occlusions, so density close to convoluted boundaries is conservative.
        """
        if self._row_prefix is None:
            dtype = np.int64 if self.mask.shape[1] > np.iinfo(np.int32).max else np.int32
            self._row_prefix = np.pad(
                np.cumsum(self.mask, axis=1, dtype=dtype), ((0, 0), (1, 0))
            )
        px, py = self.pixels(xy_um)
        mx, my = self.mpp
        y0 = max(0, math.ceil(py - radius_um / my - 0.5))
        y1 = min(self.mask.shape[0] - 1, math.floor(py + radius_um / my - 0.5))
        if y0 > y1:
            return 0.0
        ys = np.arange(y0, y1 + 1)
        delta_y = (ys + 0.5 - py) * my
        halfwidth = np.sqrt(np.maximum(0, radius_um**2 - delta_y**2)) / mx
        x0 = np.maximum(0, np.ceil(px - halfwidth - 0.5).astype(int))
        x1 = np.minimum(self.mask.shape[1], np.floor(px + halfwidth - 0.5).astype(int) + 1)
        valid = x1 > x0
        count = np.sum(self._row_prefix[ys[valid], x1[valid]] - self._row_prefix[ys[valid], x0[valid]])
        return float(count) * mx * my


class RasterSupportMask:
    """Exact source-raster support through bounded windows, never downsampled.

    An optional view selects an integer pixel-edge crop of a full-slide mask.
    ``origin_um`` locates that view, not the original full raster. Connected
    components are not materialized: exact line-of-sight traversal itself proves
    each accepted edge stays in one four-connected tissue component.
    """

    def __init__(self, path, mpp=(1., 1.), origin_um=(0., 0.), *,
                 window_xyxy=None, tile_size=256, cache_tiles=16):
        geometry = SupportMask(np.ones((1, 1), bool), mpp, origin_um)
        self.mpp, self.origin_um = geometry.mpp, geometry.origin_um
        if isinstance(tile_size, bool) or int(tile_size) != tile_size or tile_size < 1:
            raise ValueError("Raster support tile_size must be a positive integer")
        if isinstance(cache_tiles, bool) or int(cache_tiles) != cache_tiles or cache_tiles < 1:
            raise ValueError("Raster support cache_tiles must be a positive integer")
        self.tile_size, self.cache_tiles = int(tile_size), int(cache_tiles)
        self.reader = RasterReader(path)
        self.path = Path(path)
        self._cache = OrderedDict()
        self.max_window_pixels_seen = 0
        try:
            if len(self.reader.reader.shape) != 2:
                raise ValueError("Support mask must be two dimensional")
            bounds = (0, 0, self.reader.width, self.reader.height) if window_xyxy is None else tuple(window_xyxy)
            if len(bounds) != 4 or any(not np.isfinite(v) or int(v) != v for v in bounds):
                raise ValueError("Support view requires four finite integer pixel-edge coordinates")
            x0, y0, x1, y1 = map(int, bounds)
            if not (0 <= x0 < x1 <= self.reader.width and 0 <= y0 < y1 <= self.reader.height):
                raise ValueError("Support view must lie within the source raster")
            self.window_xyxy = (x0, y0, x1, y1)
            self.shape = (y1 - y0, x1 - x0)
        except Exception:
            self.close()
            raise

    def pixels(self, xy_um):
        return (np.asarray(xy_um, dtype=float) - self.origin_um) / self.mpp

    def _tile(self, tx, ty):
        key = (tx, ty)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        x0, y0 = tx * self.tile_size, ty * self.tile_size
        x1, y1 = min(x0 + self.tile_size, self.shape[1]), min(y0 + self.tile_size, self.shape[0])
        ox, oy = self.window_xyxy[:2]
        raw = self.reader.window(x0 + ox, y0 + oy, x1 + ox, y1 + oy)
        self.max_window_pixels_seen = max(self.max_window_pixels_seen, raw.size)
        if not np.all(np.isfinite(raw)) or np.any(raw < 0):
            raise ValueError("Support values must be finite and nonnegative; zero is background")
        mask = raw > 0
        entry = {"mask": mask, "all": bool(mask.all()), "prefix": None}
        if len(self._cache) >= self.cache_tiles:
            self._cache.popitem(last=False)
        self._cache[key] = entry
        return entry

    def is_tissue(self, x, y):
        if not (0 <= x < self.shape[1] and 0 <= y < self.shape[0]):
            return False
        tile = self._tile(x // self.tile_size, y // self.tile_size)
        return bool(tile["mask"][y % self.tile_size, x % self.tile_size])

    def component_at(self, pixel):
        x, y = np.floor(pixel).astype(np.int64)
        return int(self.is_tissue(int(x), int(y)))

    def all_tissue_in_box(self, x0, y0, x1, y1):
        for ty in range(y0 // self.tile_size, (y1 - 1) // self.tile_size + 1):
            for tx in range(x0 // self.tile_size, (x1 - 1) // self.tile_size + 1):
                tile = self._tile(tx, ty)
                if tile["all"]:
                    continue
                ox, oy = tx * self.tile_size, ty * self.tile_size
                if not np.all(tile["mask"][max(0, y0-oy):y1-oy, max(0, x0-ox):x1-ox]):
                    return False
        return True

    def grid_tissue(self, shape):
        """Nearest-centre analysis grid, NEVER used to approve graph edges."""
        result = np.empty(shape, bool)
        ox, oy = self.window_xyxy[:2]
        for y0 in range(0, shape[0], self.tile_size):
            for x0 in range(0, shape[1], self.tile_size):
                y1, x1 = min(y0 + self.tile_size, shape[0]), min(x0 + self.tile_size, shape[1])
                y = np.floor((np.arange(y0, y1) + .5) * self.shape[0] / shape[0]).astype(int) + oy
                x = np.floor((np.arange(x0, x1) + .5) * self.shape[1] / shape[1]).astype(int) + ox
                xx, yy = np.meshgrid(x, y)
                values = self.reader.sample(xx.ravel(), yy.ravel(), tile_size=self.tile_size)
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise ValueError("Support values must be finite and nonnegative; zero is background")
                result[y0:y1, x0:x1] = (values > 0).reshape(y1-y0, x1-x0)
        return result

    def tissue_area_in_disk(self, xy_um, radius_um):
        px, py = self.pixels(xy_um)
        mx, my = self.mpp
        y0 = max(0, math.ceil(py - radius_um / my - .5))
        y1 = min(self.shape[0] - 1, math.floor(py + radius_um / my - .5))
        if y0 > y1:
            return 0.
        # Process at most tile_size rows per batch, even for unusually large radii.
        count = 0
        for ty in range(y0 // self.tile_size, y1 // self.tile_size + 1):
            ys = np.arange(max(y0, ty * self.tile_size), min(y1 + 1, (ty + 1) * self.tile_size))
            halfwidth = np.sqrt(np.maximum(0., radius_um**2 - ((ys + .5 - py) * my)**2)) / mx
            left = np.maximum(0, np.ceil(px - halfwidth - .5).astype(np.int64))
            right = np.minimum(self.shape[1], np.floor(px + halfwidth - .5).astype(np.int64) + 1)
            valid = right > left
            if not valid.any():
                continue
            for tx in range(int(left[valid].min()) // self.tile_size, (int(right[valid].max()) - 1) // self.tile_size + 1):
                entry = self._tile(tx, ty)
                if entry["prefix"] is None:
                    entry["prefix"] = np.pad(np.cumsum(entry["mask"], axis=1, dtype=np.int64), ((0, 0), (1, 0)))
                ox, oy = tx * self.tile_size, ty * self.tile_size
                a = np.maximum(left, ox) - ox
                b = np.minimum(right, ox + entry["mask"].shape[1]) - ox
                keep = valid & (b > a)
                count += int(np.sum(entry["prefix"][ys[keep]-oy, b[keep]] - entry["prefix"][ys[keep]-oy, a[keep]]))
        return float(count) * mx * my

    def close(self):
        self._cache.clear()
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass
class DomainGrid:
    """Separate bounded analysis grid for approximate domain-boundary distances."""
    labels: np.ndarray
    support: SupportMask


def bounded_grid_shape(shape, max_pixels):
    if isinstance(max_pixels, bool) or int(max_pixels) != max_pixels or max_pixels < 1:
        raise ValueError("max_support_pixels must be a positive integer")
    if math.prod(shape) <= max_pixels:
        return tuple(shape)
    scale = math.sqrt(max_pixels / math.prod(shape))
    height = max(1, min(int(shape[0] * scale), int(max_pixels)))
    width = max(1, min(int(shape[1] * scale), int(max_pixels) // height))
    return height, width


def sample_domain_grid(path, shape, block_rows=64, block_columns=256):
    """Categorical nearest-centre registration in bounded two-axis batches."""
    height, width = shape
    if min(height, width, block_rows, block_columns) < 1:
        raise ValueError("Domain grid and sampling blocks must have positive dimensions")
    with RasterReader(path) as reader:
        if len(reader.reader.shape) != 2 or reader.dtype.kind not in "ui":
            raise ValueError("Domain labels must be a two-dimensional integer image")
        result = np.empty(shape, dtype=reader.dtype)
        for y0 in range(0, height, block_rows):
            y1 = min(height, y0 + block_rows)
            y = np.floor((np.arange(y0, y1) + .5) * reader.height / height).astype(int)
            for x0 in range(0, width, block_columns):
                x1 = min(width, x0 + block_columns)
                x = np.floor((np.arange(x0, x1) + .5) * reader.width / width).astype(int)
                xx, yy = np.meshgrid(x, y)
                result[y0:y1, x0:x1] = reader.sample(xx.ravel(), yy.ravel()).reshape(y1-y0, x1-x0)
    return result


def segment_in_support(start: np.ndarray, end: np.ndarray, support: SupportMask) -> bool:
    """Conservative supercover grid traversal, in floating mask-pixel coordinates.

    Every crossed pixel must be tissue. At exact grid corners both side pixels
    must also be tissue; this prevents diagonal bridges across one-pixel gaps.
    """
    a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if not np.all(np.isfinite([a, b])):
        return False
    ax, ay, bx, by = float(a[0]), float(a[1]), float(b[0]), float(b[1])
    x, y = math.floor(ax), math.floor(ay)
    last_x, last_y = math.floor(bx), math.floor(by)
    h, w = support.shape
    inside = support.is_tissue
    if not inside(x, y) or not inside(last_x, last_y):
        return False
    dx, dy = bx - ax, by - ay
    step_x, step_y = (1 if dx > 0 else -1 if dx < 0 else 0), (1 if dy > 0 else -1 if dy < 0 else 0)
    boundary_x = dx == 0 and abs(ax - round(ax)) < 1e-12
    boundary_y = dy == 0 and abs(ay - round(ay)) < 1e-12
    # Most intra-tissue edges have an entirely tissue-filled bounding rectangle.
    # This exact fast path avoids Python pixel traversal away from boundaries;
    # holes, thin gaps and diagonal edge cases still use the supercover below.
    lo_x, lo_y = min(x, last_x) - int(boundary_x), min(y, last_y) - int(boundary_y)
    hi_x, hi_y = max(x, last_x), max(y, last_y)
    if lo_x >= 0 and lo_y >= 0 and hi_x < w and hi_y < h:
        if support.all_tissue_in_box(lo_x, lo_y, hi_x+1, hi_y+1):
            return True
    delta_t_x = 1.0 / abs(dx) if dx != 0 else math.inf
    delta_t_y = 1.0 / abs(dy) if dy != 0 else math.inf
    max_t_x = (x + int(step_x > 0) - ax) / dx if dx != 0 else math.inf
    max_t_y = (y + int(step_y > 0) - ay) / dy if dy != 0 else math.inf
    # A segment along a grid line touches both rows/columns. Require both.
    def covered(px: int, py: int) -> bool:
        if not inside(px, py):
            return False
        if boundary_x and not inside(px-1, py):
            return False
        if boundary_y and not inside(px, py-1):
            return False
        return True

    if not covered(x, y):
        return False
    # IEEE float64 scalar arithmetic keeps the same step order and absolute
    # corner tolerance as the former NumPy DDA, without tiny arrays per pixel.
    # Explicit equality also retains np.isclose(inf, inf) semantics.
    while x != last_x or y != last_y:
        if max_t_x == max_t_y or abs(max_t_x - max_t_y) <= 1e-12:
            if not covered(x + step_x, y) or not covered(x, y + step_y):
                return False
            x += step_x
            y += step_y
            max_t_x += delta_t_x
            max_t_y += delta_t_y
        elif max_t_x < max_t_y:
            x += step_x
            max_t_x += delta_t_x
        else:
            y += step_y
            max_t_y += delta_t_y
        if not covered(x, y):
            return False
    return True


def validate_cells(cells: pd.DataFrame) -> pd.DataFrame:
    required = KEYS + ["x_um", "y_um"]
    missing = sorted(set(required) - set(cells.columns))
    if missing:
        raise ValueError(f"Missing canonical cell fields: {missing}")
    cells = cells.copy()
    if cells[KEYS].isna().any().any() or cells.empty:
        raise ValueError("Cells must be non-empty and identifiers must not be missing")
    for key in KEYS:
        cells[key] = cells[key].astype(str)
        if cells[key].str.strip().eq("").any():
            raise ValueError(f"Empty {key}")
    if cells.duplicated(KEYS).any():
        raise ValueError("Duplicate (sample_id, cell_id) identifiers")
    for coord in ["x_um", "y_um"]:
        cells[coord] = pd.to_numeric(cells[coord], errors="raise")
    if not np.all(np.isfinite(cells[["x_um", "y_um"]])):
        raise ValueError("Cell coordinates must be finite micrometre values")
    if "phenotype" not in cells:
        cells["phenotype"] = UNKNOWN
    cells["phenotype"] = cells["phenotype"].fillna(UNKNOWN).astype(str)
    cells.loc[cells.phenotype.str.strip().eq(""), "phenotype"] = UNKNOWN
    if "orientation_rad" in cells:
        cells["orientation_rad"] = pd.to_numeric(cells.orientation_rad, errors="raise")
        if np.isinf(cells.orientation_rad).any():
            raise ValueError("Orientation must be radians or missing")
    return cells.sort_values(KEYS, kind="stable").reset_index(drop=True)


def build_radius_graph(
    cells: pd.DataFrame, supports: dict[str, SupportMask], radius_um: float,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Sparse symmetric physical-distance graph in input row order, without self edges.

    Even coincident cells have explicit stored zero-distance edges. Use the CSR
    structure (indptr/indices), not distance > 0, when constructing adjacency.
    Cells outside tissue are retained as isolated and marked unsupported.
    """
    if not math.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("Graph radius must be positive and finite")
    if set(cells.sample_id.astype(str)) != set(supports):
        raise ValueError("Provide exactly one explicitly mapped support per sample_id")
    rows: list[int] = []
    cols: list[int] = []
    distances: list[float] = []
    supported = np.zeros(len(cells), dtype=bool)
    xy = cells[["x_um", "y_um"]].to_numpy(dtype=float)
    for sample, indices in cells.groupby("sample_id", sort=True).indices.items():
        support = supports[str(sample)]
        indices = np.asarray(indices)
        local_xy = xy[indices]
        pixel_xy = support.pixels(local_xy)
        components = np.asarray([support.component_at(p) for p in pixel_xy])
        supported[indices] = components > 0
        tree = cKDTree(local_xy)
        # Iterating query_ball_point avoids materializing all candidate pairs.
        for i, point in enumerate(local_xy):
            if not components[i]:
                continue
            for j in tree.query_ball_point(point, radius_um):
                if j <= i or components[i] != components[j]:
                    continue
                if not segment_in_support(pixel_xy[i], pixel_xy[j], support):
                    continue
                rows.extend([int(indices[i]), int(indices[j])])
                cols.extend([int(indices[j]), int(indices[i])])
                distance = float(np.linalg.norm(point - local_xy[j]))
                distances.extend([distance, distance])
    graph = sparse.csr_matrix((distances, (rows, cols)), shape=(len(cells), len(cells)))
    graph.sort_indices()
    return graph, supported


def graph_at_radius(graph: sparse.csr_matrix, radius_um: float) -> sparse.csr_matrix:
    coo = graph.tocoo()
    keep = coo.data <= radius_um
    return sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=graph.shape)


def _domain_distances(cells: pd.DataFrame, supports: dict[str, SupportMask], domain_masks: dict[str, np.ndarray]) -> np.ndarray:
    distances = np.full(len(cells), np.nan)
    for sample, indices in cells.groupby("sample_id", sort=True).indices.items():
        if sample not in domain_masks:
            continue
        entry = domain_masks[sample]
        support = entry.support if isinstance(entry, DomainGrid) else supports[sample]
        labels = np.asarray(entry.labels if isinstance(entry, DomainGrid) else entry)
        if not isinstance(support, SupportMask):
            raise ValueError("Raster support requires an explicit bounded DomainGrid for domain distances")
        if labels.shape != support.mask.shape or labels.ndim != 2:
            raise ValueError(f"Domain mask must share support geometry for {sample}")
        if not np.all(np.isfinite(labels)) or np.any(labels < 0) or not np.all(labels == np.floor(labels)):
            raise ValueError("Domain masks must contain finite nonnegative integer labels")
        valid = support.mask & (labels > 0)
        boundary = np.zeros(labels.shape, dtype=bool)
        for axis in (0, 1):
            first, second = [slice(None)] * 2, [slice(None)] * 2
            first[axis], second[axis] = slice(None, -1), slice(1, None)
            a, b = tuple(first), tuple(second)
            different = valid[a] & valid[b] & (labels[a] != labels[b])
            boundary[a] |= different
            boundary[b] |= different
        if not boundary.any():
            continue  # A single domain has no observed internal domain boundary.
        edt = ndimage.distance_transform_edt(~boundary, sampling=(support.mpp[1], support.mpp[0]))
        for row in indices:
            x, y = np.floor(support.pixels(cells.loc[row, ["x_um", "y_um"]].to_numpy(float))).astype(int)
            if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] and valid[y, x]:
                distances[row] = edt[y, x]
    return distances


def _align_feature_blocks(cells: pd.DataFrame, blocks: dict[str, pd.DataFrame]) -> dict[str, tuple[np.ndarray, list[str]]]:
    aligned = {}
    keys = pd.MultiIndex.from_frame(cells[KEYS])
    for name, block in sorted(blocks.items()):
        if name in {"density", "composition", "orientation"}:
            raise ValueError(f"Feature block {name} uses a reserved group name")
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Feature block names must use letters, digits, _ or -")
        if not block.columns.is_unique or not set(KEYS).issubset(block.columns) or block.duplicated(KEYS).any():
            raise ValueError(f"Feature block {name} needs unique sample_id,cell_id keys")
        block = block.copy()
        if block[KEYS].isna().any().any():
            raise ValueError(f"Feature block {name} has missing identifiers")
        block[KEYS] = block[KEYS].astype(str)
        if block.duplicated(KEYS).any():
            raise ValueError(f"Feature block {name} has duplicate identifiers after conversion")
        indexed = block.set_index(KEYS)
        if not indexed.index.isin(keys).all():
            raise ValueError(f"Feature block {name} contains cells absent from canonical table")
        if indexed.shape[1] == 0:
            raise ValueError(f"Feature block {name} has no numerical features")
        if len({str(column) for column in indexed.columns}) != len(indexed.columns):
            raise ValueError(f"Feature block {name} has colliding serialized feature names")
        array = indexed.reindex(keys).apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if np.isinf(array).any():
            raise ValueError(f"Feature block {name} has infinite values")
        aligned[name] = (array, list(indexed.columns))
    return aligned


def neighborhood_features(
    cells: pd.DataFrame, graphs: dict[float, sparse.csr_matrix],
    supports: dict[str, SupportMask], supported: np.ndarray,
    feature_blocks: dict[str, pd.DataFrame] | None = None,
    domain_masks: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return retained cells and physical/contextual features grouped for scaling."""
    profiles = cells.copy()
    profiles["in_tissue_support"] = supported
    profiles["domain_boundary_distance_um"] = _domain_distances(cells, supports, domain_masks or {})
    profiles["domain_boundary_distance_available"] = profiles.domain_boundary_distance_um.notna()
    phenotypes = sorted(set(cells.phenotype) | {UNKNOWN})
    ph_index = {p: i for i, p in enumerate(phenotypes)}
    onehot = sparse.csr_matrix((np.ones(len(cells)), (np.arange(len(cells)), [ph_index[p] for p in cells.phenotype])), shape=(len(cells), len(phenotypes)))
    orientation = cells.get("orientation_rad", pd.Series(np.nan, index=cells.index)).to_numpy(float)
    valid_orientation = np.isfinite(orientation)
    cos2 = np.where(valid_orientation, np.cos(2 * orientation), 0)
    sin2 = np.where(valid_orientation, np.sin(2 * orientation), 0)
    blocks = _align_feature_blocks(cells, feature_blocks or {})
    groups: dict[str, list[str]] = {"density": [], "composition": [], "orientation": []}
    fields: dict[str, np.ndarray] = {}
    for name, (array, columns) in blocks.items():
        # Own-cell features enter once, separately from each-radius neighbour
        # means. ':' cannot occur in an input block name, so own group keys
        # cannot collide with the existing neighbour group names.
        own_group = f"own:{name}"
        groups[own_group] = []
        for col, label in enumerate(columns):
            field = f"own_{name}:{quote(str(label), safe='')}"
            fields[field] = array[:, col]
            groups[own_group].append(field)
        fields[f"own_{name}_observed_fraction"] = np.isfinite(array).mean(axis=1)
    for radius, graph in sorted(graphs.items()):
        prefix = f"r{radius:g}um"
        adjacency = graph.copy()
        adjacency.data = np.ones_like(adjacency.data)
        counts = np.diff(adjacency.indptr).astype(int)
        nonempty = counts > 0
        denominator = np.maximum(counts, 1)
        areas = np.asarray([
            supports[str(row.sample_id)].tissue_area_in_disk(np.array([row.x_um, row.y_um]), radius) if supported[i] else np.nan
            for i, row in enumerate(cells.itertuples())
        ])
        fields[f"{prefix}_neighbor_count"] = counts
        fields[f"{prefix}_tissue_area_um2"] = areas
        density = np.divide(counts * 1e6, areas, out=np.full(len(cells), np.nan), where=areas > 0)
        fields[f"{prefix}_neighbor_density_per_mm2"] = density
        groups["density"].append(f"{prefix}_neighbor_density_per_mm2")
        fractions = (adjacency @ onehot).toarray() / denominator[:, None]
        # A cell without neighbours has no composition estimate, not 0% unknown.
        fractions[~nonempty] = np.nan
        for i, phenotype in enumerate(phenotypes):
            field = f"{prefix}_phenotype_fraction:{quote(phenotype, safe='')}"
            fields[field] = fractions[:, i]
            groups["composition"].append(field)
        logf = np.zeros_like(fractions)
        np.log(fractions, out=logf, where=fractions > 0)
        entropy = -np.sum(fractions * logf, axis=1)
        fields[f"{prefix}_phenotype_entropy_nats"] = entropy
        fields[f"{prefix}_phenotype_mixing_normalized"] = entropy / np.log(len(phenotypes)) if len(phenotypes) > 1 else np.where(nonempty, 0.0, np.nan)
        valid_count = adjacency @ valid_orientation.astype(float)
        mean_cos = np.divide(adjacency @ cos2, valid_count, out=np.full(len(cells), np.nan), where=valid_count > 0)
        mean_sin = np.divide(adjacency @ sin2, valid_count, out=np.full(len(cells), np.nan), where=valid_count > 0)
        fields[f"{prefix}_orientation_count"] = valid_count
        fields[f"{prefix}_axial_coherence"] = np.hypot(mean_cos, mean_sin)
        fields[f"{prefix}_alignment_to_focal_axis"] = np.where(valid_orientation, mean_cos * cos2 + mean_sin * sin2, np.nan)
        groups["orientation"].extend([f"{prefix}_axial_coherence", f"{prefix}_alignment_to_focal_axis"])
        for name, (array, columns) in blocks.items():
            finite = np.isfinite(array)
            observed = adjacency @ finite.astype(float)
            sums = adjacency @ np.where(finite, array, 0)
            means = np.divide(sums, observed, out=np.full_like(sums, np.nan), where=observed > 0)
            for col, label in enumerate(columns):
                field = f"{prefix}_{name}_mean:{quote(str(label), safe='')}"
                fields[field] = means[:, col]
                groups.setdefault(name, []).append(field)
            fields[f"{prefix}_{name}_observed_fraction"] = np.mean(observed / denominator[:, None], axis=1)
    if set(fields) & set(profiles.columns):
        raise ValueError("Derived niche feature columns collide with supplied cell columns")
    return pd.concat([profiles, pd.DataFrame(fields, index=profiles.index)], axis=1), groups


def feature_group_definitions(groups, feature_blocks, radii_um, phenotypes):
    """Portable column roles for separately assembled/cohort-pooled profiles.

    Model/checkpoint definitions live in the source profile's feature blocks;
    this contract identifies which source block and aggregation each column
    uses. Coverage columns remain QC, not additional clustering coordinates.
    """
    radii = sorted(set(map(float, radii_um)))
    definitions = {
        "density": {"role": "neighborhood_density", "aggregation": "visible_neighbors_per_tissue_disk_area_mm2"},
        "composition": {"role": "neighbor_phenotype_composition", "aggregation": "neighbor_fraction",
            "phenotypes": sorted(set(map(str, phenotypes)) | {UNKNOWN}),
            "absent_category": "zero_only_for_nonempty_neighborhood; otherwise_missing"},
        "orientation": {"role": "neighbor_axial_orientation", "aggregation": "axial_coherence_and_alignment_to_focal_axis"},
    }
    for name, definition in definitions.items():
        definition.update({"columns": list(groups[name]), "radii_um": radii, "source_feature_block": None})
    for name, block in sorted(feature_blocks.items()):
        source_features = [str(column) for column in block if column not in KEYS]
        definitions[name] = {"role": "neighbor_feature_mean", "aggregation": "mean_of_finite_neighbors_per_feature",
            "source_feature_block": name, "source_features": source_features, "columns": list(groups[name]),
            "radii_um": radii, "focal_cell_included": False,
            "coverage_columns": [f"r{radius:g}um_{name}_observed_fraction" for radius in radii]}
        definitions[f"own:{name}"] = {"role": "own_cell_feature", "aggregation": "identity_no_imputation",
            "source_feature_block": name, "source_features": source_features, "columns": list(groups[f"own:{name}"]),
            "radii_um": [], "focal_cell_included": True, "coverage_columns": [f"own_{name}_observed_fraction"]}
    return definitions


def _niche_matrix(profiles: pd.DataFrame, groups: dict[str, list[str]], weights: dict[str, float], eligible: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    arrays, metadata = [], {}
    for name, columns in sorted(groups.items()):
        if not columns:
            continue
        if name in weights and (not math.isfinite(weights[name]) or weights[name] <= 0):
            raise ValueError("Feature group weights must be positive finite values")
        raw = profiles[columns].to_numpy(float)
        available = np.isfinite(raw)
        count = available[eligible].sum(axis=0)
        means = np.divide(np.where(available[eligible], raw[eligible], 0).sum(axis=0), count, out=np.zeros(len(columns)), where=count > 0)
        centered = np.where(available, raw - means, 0)
        std = np.sqrt(np.divide((centered[eligible] ** 2).sum(axis=0), count, out=np.zeros(len(columns)), where=count > 0))
        keep = (std > 1e-12) & (count > 1)
        # Availability is an independent dimension: constant observed values
        # must not erase a varying missingness pattern. Entirely absent or
        # entirely observed columns need no missingness indicator.
        missing = ~available
        variable_missing = missing[eligible].any(axis=0) & ~missing[eligible].all(axis=0)
        if not keep.any() and not variable_missing.any():
            continue
        z = centered[:, keep] / std[keep]
        if variable_missing.any():
            z = np.column_stack([z, missing[:, variable_missing].astype(float)])
        weight = weights.get(name, 1.0)
        arrays.append(z * math.sqrt(weight / z.shape[1]))
        retained_columns = [c for c, ok in zip(columns, keep) if ok]
        missing_columns = [c for c, ok in zip(columns, variable_missing) if ok]
        metadata[name] = {"columns": retained_columns, "means": means[keep].tolist(), "std": std[keep].tolist(),
            "missing_indicator_columns": missing_columns, "weight": weight, "dimensions": z.shape[1],
            "matrix_columns": [{"kind": "standardized_value", "source_column": c} for c in retained_columns]
                + [{"kind": "missing_indicator", "source_column": c} for c in missing_columns]}
    matrix = np.column_stack(arrays) if arrays else np.zeros((len(profiles), 0))
    return matrix, metadata


def _canonical_labels(labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(range(len(centers)), key=lambda i: tuple(centers[i]))
    inverse = np.argsort(order)
    return inverse[labels], centers[order]


def discover_niches(
    profiles: pd.DataFrame, groups: dict[str, list[str]], *, fixed_k: int | None = None,
    max_k: int = 8, seed: int = 17, repeats: int = 5,
    feature_weights: dict[str, float] | None = None, fit_limit: int = 20000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reproducible k-means niches, selected by subsample stability and separation.

    Fits are bounded to fit_limit cells; all cells are assigned to fitted centers.
    Auto-K requires mean ARI >= .75 and mean centroid silhouette >= .45, then
    maximizes stability * separation - .01*K. This is a disclosed heuristic for
    exploratory discovery, not a test of the true biological domain count.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from scipy.spatial.distance import cdist

    if max_k < 2 or repeats < 2 or fit_limit < 10:
        raise ValueError("max_k >= 2, repeats >= 2 and fit_limit >= 10 are required")
    if fixed_k is not None and fixed_k < 1:
        raise ValueError("fixed_k must be positive")
    unknown_weights = set(feature_weights or {}) - set(groups)
    if unknown_weights:
        raise ValueError(f"Unknown feature weight groups: {sorted(unknown_weights)}")
    count_columns = [c for c in profiles if c.endswith("_neighbor_count")]
    eligible = profiles.in_tissue_support.to_numpy(bool) & (profiles[count_columns].max(axis=1).to_numpy() > 0)
    output = profiles[KEYS].copy()
    output["niche_id"] = pd.Series([pd.NA] * len(profiles), dtype="Int64")
    output["niche_status"] = np.where(profiles.in_tissue_support, "isolated_no_neighborhood", "outside_tissue_support")
    output["niche_centroid_margin"] = np.nan
    output["niche_stability"] = np.nan
    matrix, scaling = _niche_matrix(profiles, groups, feature_weights or {}, eligible)
    meta: dict[str, Any] = {"feature_representation_version": FEATURE_REPRESENTATION_VERSION,
        "missing_feature_policy": "raw_NaNs_preserved; centered_missing_values_zero_in_fit_only; independent_variable_missingness_indicators",
        "missingness_interpretation": "Availability can separate exploratory niches; missingness-driven clusters are not evidence of biological differences.",
        "seed": seed, "repeats": repeats, "fit_limit": fit_limit, "eligible_cells": int(eligible.sum()), "fixed_k": fixed_k, "scaling": scaling, "candidate_diagnostics": [], "confidence_is_calibrated_probability": False, "selection_rule": "ARI>=0.75 and centroid_silhouette>=0.45; maximize ARI*centroid_silhouette-0.01*K; otherwise K=1"}
    if not eligible.any():
        if fixed_k is not None:
            raise ValueError("Cannot force niches when no cell has a supported neighborhood")
        meta["selected_k"] = 0
        return output, meta
    data = matrix[eligible]
    rng = np.random.default_rng(seed)
    train_indices = np.sort(rng.choice(len(data), min(len(data), fit_limit), replace=False))
    train = data[train_indices]
    distinct = len(np.unique(train, axis=0)) if train.shape[1] else 1
    if fixed_k and fixed_k > distinct:
        raise ValueError(f"Requested {fixed_k} niches but only {distinct} distinct feature profiles exist")
    candidates = [fixed_k] if fixed_k is not None else list(range(2, min(max_k, distinct, len(train) // 2) + 1))
    selected = None
    best_score = -np.inf
    for k in candidates:
        if k == 1:
            break
        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(train)
        labels, centers = _canonical_labels(model.predict(data), model.cluster_centers_)
        baseline_train = labels[train_indices]
        # Only an N x K matrix; no cell-by-cell dense distance matrix is built.
        distances = cdist(data, centers)
        own = distances[np.arange(len(data)), labels]
        other = np.partition(distances, 1, axis=1)[:, 1]
        margin = (other - own) / np.maximum(other, 1e-12)
        stability_count = np.zeros(len(data))
        ari_values = []
        for repeat in range(repeats):
            subset = np.sort(rng.choice(len(train), max(k, int(len(train) * .8)), replace=False))
            bootstrap = KMeans(n_clusters=k, random_state=seed + repeat + 1, n_init=5).fit(train[subset])
            pred = bootstrap.predict(data)
            contingency = np.zeros((k, k), dtype=int)
            np.add.at(contingency, (pred[train_indices], baseline_train), 1)
            row, col = linear_sum_assignment(-contingency)
            mapping = np.empty(k, dtype=int)
            mapping[row] = col
            aligned = mapping[pred]
            stability_count += aligned == labels
            ari_values.append(float(adjusted_rand_score(baseline_train, aligned[train_indices])))
        stability = float(np.mean(ari_values))
        separation = float(margin[train_indices].mean())
        min_count = int(np.bincount(baseline_train, minlength=k).min())
        accepted = fixed_k is not None or (stability >= .75 and separation >= .45 and min_count >= max(2, int(len(train) * .01)))
        score = stability * separation - .01 * k
        meta["candidate_diagnostics"].append({"k": int(k), "subsample_ari": stability, "centroid_silhouette": separation, "smallest_training_cluster": min_count, "score": score, "passes_selection": bool(accepted)})
        if accepted and score > best_score:
            selected = (labels, centers, margin, stability_count / repeats)
            best_score = score
    if selected is None:
        labels = np.zeros(len(data), dtype=int)
        centers = np.mean(data, axis=0, keepdims=True)
        margin, stability = np.full(len(data), np.nan), np.full(len(data), np.nan)
        status = "assigned_fixed_k" if fixed_k == 1 else "single_niche_no_supported_subdivision"
    else:
        labels, centers, margin, stability = selected
        status = "assigned_fixed_k" if fixed_k is not None else "assigned_exploratory"
    output.loc[eligible, "niche_id"] = labels + 1
    output.loc[eligible, "niche_status"] = status
    output.loc[eligible, "niche_centroid_margin"] = margin
    output.loc[eligible, "niche_stability"] = stability
    meta.update({"selected_k": int(len(centers)), "training_cells": int(len(train)), "feature_dimensions": int(matrix.shape[1]), "centroids_scaled": centers.tolist(), "training_scope": "pooled specimens; graphs remain specimen-specific; sample identity excluded from features"})
    return output, meta


def analyze_cells(
    cells: pd.DataFrame, supports: dict[str, SupportMask], *, radii_um: tuple[float, ...] = (25, 50, 100),
    domain_masks: dict[str, np.ndarray] | None = None,
    feature_blocks: dict[str, pd.DataFrame] | None = None,
    fixed_k: int | None = None, seed: int = 17, max_k: int = 8, repeats: int = 5,
    feature_weights: dict[str, float] | None = None,
    defer_niches: bool = False,
    fit_limit: int = 20000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[float, sparse.csr_matrix], dict[str, Any]]:
    cells = validate_cells(cells)
    radii = sorted(set(float(v) for v in radii_um))
    if not radii or any(not math.isfinite(r) or r <= 0 for r in radii):
        raise ValueError("Provide positive finite neighbourhood radii in micrometres")
    if domain_masks and not set(domain_masks).issubset(supports):
        raise ValueError("Domain masks contain specimen IDs absent from supports")
    max_graph, supported = build_radius_graph(cells, supports, radii[-1])
    graphs = {radius: graph_at_radius(max_graph, radius) for radius in radii}
    profiles, groups = neighborhood_features(cells, graphs, supports, supported, feature_blocks, domain_masks)
    if defer_niches:
        # Assembly can persist high-dimensional groups outside this scalar
        # table, then fit through bounded reads in the same sorted cell order.
        niches, niche_metadata = profiles[KEYS].copy(), {"status": "deferred"}
    else:
        niches, niche_metadata = discover_niches(profiles, groups, fixed_k=fixed_k, seed=seed, max_k=max_k, repeats=repeats, feature_weights=feature_weights, fit_limit=fit_limit)
    summary = {
        "schema_version": "1.1.0", "cell_count": len(cells), "specimens": sorted(supports),
        "radii_um": radii, "outside_support_count": int((~supported).sum()),
        "graph": {f"{radius:g}": {"undirected_edges": graph.nnz // 2, "isolated_cells": int(np.sum(np.diff(graph.indptr) == 0))} for radius, graph in graphs.items()},
        "graph_contract": "CSR values are Euclidean distances in um, including stored zero-distance edges; rows follow graph_cells.csv; no self edges; per-specimen radius with conservative pixel-supercover tissue traversal",
        "density_contract": "neighbors (excluding focal cell) per mm2 of tissue pixels within Euclidean disk; area estimated by pixel centres; visibility-constrained neighbors; denominator includes occluded tissue and is not geodesic area",
        "phenotype_unknown_category": UNKNOWN, "empty_neighborhood_fractions": "missing, not zero",
        "orientation_contract": "axial radians modulo pi; coherence in [0,1], focal alignment in [-1,1]; missing when no valid orientation",
        "domain_boundary_contract": "distance in um to nearest raster boundary pixel between two positive tissue domains; requires domain mask registered to support; missing for no internal boundary or unknown domain",
        "mask_geometry": {sample: {"shape_yx": list(s.shape), "mpp_xy": list(s.mpp), "origin_um_xy": list(s.origin_um),
            "graph_verification_grid": "exact_supplied_raster_pixels",
            "backend": "bounded_raster_windows" if isinstance(s, RasterSupportMask) else "in_memory",
            "gap_precision_limit": "Gaps absent from the supplied raster cannot be recovered or certified."} for sample, s in supports.items()},
        "domain_analysis_geometry": {sample: {"shape_yx": list(entry.support.shape),
            "mpp_xy": list(entry.support.mpp), "origin_um_xy": list(entry.support.origin_um)}
            for sample, entry in (domain_masks or {}).items() if isinstance(entry, DomainGrid)},
        "feature_representation_version": FEATURE_REPRESENTATION_VERSION,
        "feature_groups": groups,
        "feature_group_definitions": feature_group_definitions(groups, feature_blocks or {}, radii, cells.phenotype),
        "niche_discovery": niche_metadata,
    }
    return profiles, niches, graphs, summary


def _pair(value: Any) -> tuple[float, float]:
    values = [float(v) for v in value.split(",")] if isinstance(value, str) else ([float(value)] if np.isscalar(value) else list(map(float, value)))
    if len(values) == 1:
        values *= 2
    if len(values) != 2:
        raise ValueError("Expected one value or x,y pair")
    return tuple(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    masks = parser.add_mutually_exclusive_group(required=True)
    masks.add_argument("--support-mask", type=Path)
    masks.add_argument("--support-manifest", type=Path, help="JSON mapping sample_id to {path,mpp,origin_um?,domain_path?}; paths relative to manifest")
    parser.add_argument("--support-mpp", help="Micrometres per support pixel: scalar or x,y")
    parser.add_argument("--support-origin-um", default="0,0")
    parser.add_argument("--domain-mask", type=Path, help="Registered integer tissue-domain labels; 0=unknown/background")
    parser.add_argument("--support-tile-size", type=int, default=256)
    parser.add_argument("--support-cache-tiles", type=int, default=16)
    parser.add_argument("--max-domain-pixels", type=int, default=50_000_000, help="Bound approximate domain distances only; graph/density support is never resampled")
    parser.add_argument("--radii-um", default="25,50,100")
    parser.add_argument("--fixed-k", type=int)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--stability-repeats", type=int, default=5)
    parser.add_argument("--feature-block", action="append", default=[], metavar="NAME=CSV", help="Numeric features keyed by sample_id,cell_id; repeatable")
    parser.add_argument("--feature-weight", action="append", default=[], metavar="NAME=WEIGHT", help="Separate group weights: block name for neighbour means, own:NAME for focal features; built-ins density,composition,orientation; defaults equal")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    cells = pd.read_csv(args.cells, dtype={key: str for key in KEYS})
    samples = sorted(set(cells.sample_id.dropna().astype(str)))
    if args.support_mask:
        if len(samples) != 1 or args.support_mpp is None:
            parser.error("--support-mask requires exactly one specimen and explicit --support-mpp; use --support-manifest for multiple specimens")
        entries = {samples[0]: {"path": str(args.support_mask.resolve()), "mpp": _pair(args.support_mpp), "origin_um": _pair(args.support_origin_um)}}
        if args.domain_mask:
            entries[samples[0]]["domain_path"] = str(args.domain_mask.resolve())
        base = Path.cwd()
    else:
        if args.support_mpp or args.domain_mask or args.support_origin_um != "0,0":
            parser.error("With a support manifest, put all geometry/domain paths inside the manifest")
        entries = json.loads(args.support_manifest.read_text())
        base = args.support_manifest.resolve().parent
    feature_blocks = {}
    inputs = {"cells": {"path": str(args.cells.resolve()), "sha256": _sha256(args.cells)}}
    for spec in args.feature_block:
        name, filename = spec.split("=", 1)
        if name in feature_blocks:
            parser.error(f"Repeated feature block {name}")
        path = Path(filename)
        feature_blocks[name] = pd.read_csv(path, dtype={key: str for key in KEYS})
        inputs[f"features:{name}"] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    weights = {name: float(value) for name, value in (item.split("=", 1) for item in args.feature_weight)}
    with ExitStack() as stack:
        supports, domains = {}, {}
        for sample, entry in entries.items():
            path = (base / entry["path"]).resolve()
            support = stack.enter_context(RasterSupportMask(path, _pair(entry["mpp"]),
                _pair(entry.get("origin_um", (0, 0))), tile_size=args.support_tile_size,
                cache_tiles=args.support_cache_tiles))
            supports[sample] = support
            inputs[f"support:{sample}"] = {"path": str(path), "sha256": _sha256(path)}
            if entry.get("domain_path"):
                path = (base / entry["domain_path"]).resolve()
                with RasterReader(path) as reader:
                    if (reader.height, reader.width) != support.shape:
                        raise ValueError(f"Domain mask must share support geometry for {sample}")
                shape = bounded_grid_shape(support.shape, args.max_domain_pixels)
                mpp = (support.mpp[0] * support.shape[1] / shape[1], support.mpp[1] * support.shape[0] / shape[0])
                domains[sample] = DomainGrid(sample_domain_grid(path, shape),
                    SupportMask(support.grid_tissue(shape), mpp, support.origin_um))
                inputs[f"domains:{sample}"] = {"path": str(path), "sha256": _sha256(path)}
        profiles, niches, graphs, summary = analyze_cells(cells, supports, radii_um=tuple(float(v) for v in args.radii_um.split(",")), domain_masks=domains, feature_blocks=feature_blocks, fixed_k=args.fixed_k, max_k=args.max_k, seed=args.seed, repeats=args.stability_repeats, feature_weights=weights)
    summary["inputs"] = inputs
    import scipy
    import sklearn
    summary["software"] = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "script_sha256": _sha256(Path(__file__))}
    args.outdir.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.outdir / "neighborhood_profiles.csv", index=False)
    niches.to_csv(args.outdir / "cellular_niches.csv", index=False)
    profiles[KEYS].to_csv(args.outdir / "graph_cells.csv", index=False)
    for radius, graph in graphs.items():
        sparse.save_npz(args.outdir / f"neighborhood_graph_{radius:g}um.npz", graph)
    (args.outdir / "neighborhood_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cells": len(profiles), "niches": summary["niche_discovery"]["selected_k"], "outdir": str(args.outdir.resolve())}))


if __name__ == "__main__":
    main()
