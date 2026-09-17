"""Image-only, tissue-constrained cardinal adjacency for fixed observations.

This is a descriptive appearance graph, not a biological boundary classifier.
No observation is relabelled, excluded from the vertex table, or imputed. Source
identity/receipts belong to the caller; this pure function does not write files.
"""
from __future__ import annotations

from collections import OrderedDict
import math
from numbers import Integral, Real
import re

import numpy as np
import pandas as pd

from analyze_cell_neighborhoods import RasterSupportMask, segment_in_support
from cell_profile_io import RasterReader


FORMAT = "cellphenotyper_tissue_spatial_adjacency"
VERSION = "1.0.0"
OD_EPSILON = 1.0 / 255.0
EDGE_COLUMNS = ["source_index", "target_index", "distance_um", "boundary_strength", "boundary_weight"]
OBSERVATION_COLUMNS = ["label", "x", "y", "grid_row", "grid_col"]
DESCRIPTOR_CACHE_SIZE = 128


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real number")
    return value


def _observations(frame):
    if (not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique
            or not set(OBSERVATION_COLUMNS) <= set(frame.columns)):
        raise ValueError("Observations require a nonempty table with unique label,x,y,grid_row,grid_col columns")
    result = frame[OBSERVATION_COLUMNS].copy().reset_index(drop=True)
    labels = result["label"].tolist()
    if (any(not isinstance(label, str) or not label or label != label.strip()
            or re.search(r"[\x00-\x1f\x7f]", label) for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Observation labels must be unique nonempty literal strings without edge whitespace/control characters")
    for name in ("grid_row", "grid_col"):
        values = result[name].tolist()
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
               or value < 0 or value > np.iinfo(np.int64).max for value in values):
            raise ValueError(f"{name} must contain nonnegative int64 grid indices")
        result[name] = np.asarray(values, dtype=np.int64)
    if result.duplicated(["grid_row", "grid_col"]).any():
        raise ValueError("Observation grid_row/grid_col pairs must be unique")
    for name in ("x", "y"):
        values = result[name].tolist()
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
               or not math.isfinite(float(value)) for value in values):
            raise ValueError(f"Observation {name} coordinates must be finite real numbers")
        result[name] = np.asarray(values, dtype=np.float64)
    if result.duplicated(["x", "y"]).any():
        raise ValueError("Distinct grid observations must not claim duplicate x/y pixel centres")
    return result


class _Appearance:
    """Area-weighted square mean RGB OD, using bounded raster windows."""

    def __init__(self, reader, image_mpp, radius_um, tile_size):
        if len(reader.reader.shape) != 3 or reader.reader.shape[2] != 3:
            raise ValueError("Image must have exactly three RGB channels in YXC order")
        dtype = np.dtype(reader.dtype)
        if dtype.kind == "u" and dtype.itemsize in (1, 2):
            self.white_level = float(np.iinfo(dtype).max)
        elif dtype.kind == "f" and dtype.itemsize in (4, 8):
            self.white_level = 1.0
        else:
            raise ValueError("Image dtype must declare uint8/uint16 full-range RGB or float32/float64 normalized RGB in [0,1]")
        self.reader, self.tile_size = reader, tile_size
        self.radius_px = tuple(radius_um / value for value in image_mpp)
        if not np.isfinite(self.radius_px).all():
            raise ValueError("Descriptor radius must be representable in image pixels")
        self.cache = OrderedDict()
        self.max_window_pixels = 0
        self.descriptors_computed = 0
        self.cache_hits = 0

    def mean_od(self, xy):
        key = (float(xy[0]), float(xy[1]))
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache_hits += 1
            return self.cache[key]
        px, py = key
        rx, ry = self.radius_px
        left, right = max(0.0, px-rx), min(float(self.reader.width), px+rx)
        top, bottom = max(0.0, py-ry), min(float(self.reader.height), py+ry)
        if left >= right or top >= bottom:
            raise ValueError("Descriptor has no representable pixel area; check the physical radius")
        # Pixels are constant over [x,x+1) x [y,y+1). Fractional edge-pixel
        # overlap makes the square definition explicit even with anisotropic
        # calibration, subpixel coordinates, or clipping at an image edge.
        x0, x1, y0, y1 = math.floor(left), math.ceil(right), math.floor(top), math.ceil(bottom)
        total, area = np.zeros(3, dtype=np.float64), 0.0
        for by in range(y0, y1, self.tile_size):
            ey = min(y1, by+self.tile_size)
            wy = np.minimum(np.arange(by, ey, dtype=np.float64)+1, bottom) - np.maximum(np.arange(by, ey, dtype=np.float64), top)
            for bx in range(x0, x1, self.tile_size):
                ex = min(x1, bx+self.tile_size)
                wx = np.minimum(np.arange(bx, ex, dtype=np.float64)+1, right) - np.maximum(np.arange(bx, ex, dtype=np.float64), left)
                raw = self.reader.window(bx, by, ex, ey)
                self.max_window_pixels = max(self.max_window_pixels, (ey-by)*(ex-bx))
                values = raw.astype(np.float64) / self.white_level
                if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                    raise ValueError("Sampled RGB values must be finite normalized transmittance in [0,1]")
                od = -np.log((values+OD_EPSILON)/(1+OD_EPSILON))
                weights = wy[:, None] * wx[None, :]
                total += np.einsum("ij,ijk->k", weights, od)
                area += float(weights.sum())
        result = total / area
        if len(self.cache) >= DESCRIPTOR_CACHE_SIZE:
            self.cache.popitem(last=False)
        self.cache[key] = result
        self.descriptors_computed += 1
        return result


def build_adjacency(observations, image_path, support_path, *, image_mpp=(1., 1.),
                    radius_um, descriptor_radius_um=2.0, path_step_um=2.0,
                    boundary_sigma=0.15, tile_size=256):
    """Return ``(edges, vertices, metadata)`` without changing any source.

    ``x,y`` are level-zero crop pixel coordinates, with the top-left pixel edge
    at (0,0). Support is explicitly assumed to cover the *same full extent*;
    its physical x/y pixel sizes are calculated independently from dimensions.
    Edges are one-based source-row indices, always ``source_index < target_index``.
    Support validity is exact at the supplied mask resolution, not a claim that
    an upstream coarse mask represents every native gap.
    """
    if not isinstance(image_mpp, (tuple, list, np.ndarray)) or np.shape(image_mpp) != (2,):
        raise ValueError("image_mpp must contain exactly two positive finite x,y values")
    image_mpp = tuple(_positive(value, "image_mpp") for value in image_mpp)
    radius_um = _positive(radius_um, "radius_um")
    descriptor_radius_um = _positive(descriptor_radius_um, "descriptor_radius_um")
    path_step_um = _positive(path_step_um, "path_step_um")
    boundary_sigma = _positive(boundary_sigma, "boundary_sigma")
    if isinstance(tile_size, (bool, np.bool_)) or not isinstance(tile_size, Integral) or tile_size < 1:
        raise ValueError("tile_size must be a positive integer")
    tile_size = int(tile_size)
    vertices = _observations(observations)
    xy = vertices[["x", "y"]].to_numpy(dtype=np.float64)
    with RasterReader(image_path) as image:
        appearance = _Appearance(image, image_mpp, descriptor_radius_um, tile_size)
        if ((xy < 0).any() or (xy[:, 0] >= image.width).any() or (xy[:, 1] >= image.height).any()):
            raise ValueError("Observation coordinates must lie within the level-zero image bounds")
        physical = xy * np.asarray(image_mpp)
        extent = np.asarray([image.width, image.height], dtype=np.float64) * image_mpp
        if not np.isfinite(physical).all() or not np.isfinite(extent).all():
            raise ValueError("Image physical geometry must be finite")
        with RasterReader(support_path) as source:
            if len(source.reader.shape) != 2 or source.dtype.kind not in "buif":
                raise ValueError("Support must be a real numeric two-dimensional mask")
            support_shape = [source.height, source.width]
            mask_mpp = tuple((extent / [source.width, source.height]).tolist())
        with RasterSupportMask(support_path, mpp=mask_mpp, tile_size=tile_size) as support:
            supported = np.array([support.component_at(support.pixels(point)) > 0 for point in physical], dtype=bool)
            vertices.insert(0, "source_index", np.arange(1, len(vertices)+1, dtype=np.int64))
            vertices["in_tissue_support"] = supported
            vertices["status"] = np.where(supported, "supported", "unsupported")
            lookup = {(int(row), int(col)): i for i, (row, col) in
                      enumerate(vertices[["grid_row", "grid_col"]].itertuples(index=False, name=None))}
            edges = []
            counts = {"cardinal_candidates": 0, "unsupported_endpoint_rejections": 0,
                      "radius_rejections": 0, "support_segment_rejections": 0, "path_samples": 0}
            max_actual_step = 0.0
            for (row, col), i in lookup.items():
                for key in ((row+1, col), (row, col+1)):
                    if key not in lookup:
                        continue
                    j = lookup[key]
                    counts["cardinal_candidates"] += 1
                    if not (supported[i] and supported[j]):
                        counts["unsupported_endpoint_rejections"] += 1
                        continue
                    distance = math.hypot(*(physical[j]-physical[i]))
                    if distance == 0:
                        raise ValueError("Distinct grid centres collapse to zero physical distance; check calibration")
                    if distance > radius_um:
                        counts["radius_rejections"] += 1
                        continue
                    if not segment_in_support(support.pixels(physical[i]), support.pixels(physical[j]), support):
                        counts["support_segment_rejections"] += 1
                        continue
                    ratio = distance/path_step_um
                    if not math.isfinite(ratio):
                        raise ValueError("Path sampling count is not representable; increase path_step_um")
                    steps = max(1, math.ceil(ratio))
                    max_actual_step = max(max_actual_step, distance/steps)
                    previous, strength = appearance.mean_od(xy[i]), 0.0
                    for step in range(1, steps+1):
                        point = xy[j] if step == steps else xy[i] + (xy[j]-xy[i])*(step/steps)
                        current = appearance.mean_od(point)
                        strength = max(strength, float(np.linalg.norm(current-previous)))
                        previous = current
                    counts["path_samples"] += steps+1
                    scaled = strength/boundary_sigma
                    weight = math.exp(-0.5*scaled*scaled)
                    a, b = sorted((i+1, j+1))
                    edges.append((a, b, distance, strength, weight))
            result = pd.DataFrame(sorted(edges), columns=EDGE_COLUMNS).astype({
                "source_index": "int64", "target_index": "int64", "distance_um": "float64",
                "boundary_strength": "float64", "boundary_weight": "float64"})
            metadata = {"format": FORMAT, "schema_version": VERSION,
                "coordinate_system": "crop_level0_pixels_top_left_pixel_edge_origin",
                "image_shape_yx": [image.height, image.width], "image_mpp_xy": list(image_mpp),
                "support_shape_yx": support_shape, "support_mpp_xy": list(mask_mpp),
                "support_extent_assumption": "Support and image cover exactly the same full crop extent; caller must bind source geometry.",
                "support_semantics": "Finite nonnegative mask, >0 is tissue; exact conservative supercover at supplied mask resolution, including touched corner pixels.",
                "support_resolution_limit": "No recovery of native gaps absent from the supplied support raster.",
                "candidate_policy": "Existing grid_row/grid_col cardinal neighbours only; no diagonal or missing-tile bridges.",
                "appearance_sampling_limit": "Cardinal-only orientation bias and square/path sampling can smooth or miss thin appearance boundaries; low boundary_strength is not evidence that no biological boundary exists.",
                "index_policy": "One-based original observation row order; all vertices retained; undirected edges stored once with source_index < target_index.",
                "radius_um": radius_um, "descriptor_radius_um": descriptor_radius_um,
                "path_step_um": path_step_um, "max_actual_path_step_um": max_actual_step,
                "boundary_sigma": boundary_sigma,
                "boundary_strength": "Maximum L2 jump between adjacent sampled square-mean RGB optical-density vectors, including path endpoints.",
                "boundary_weight": "exp(-0.5*(boundary_strength/boundary_sigma)^2)",
                "descriptor": {"optical_density": "-ln((RGB/white_level+1/255)/(1+1/255))",
                    "image_dtype": str(image.dtype), "white_level": appearance.white_level,
                    "pixel_reduction": "Area-weighted mean of per-pixel RGB OD over a physical square of half-width descriptor_radius_um, clipped at image edges; background inside the square is not masked.",
                    "value_validation_scope": "All sampled descriptor-window RGB values; unsampled image pixels are not scanned.",
                    "normalization": "Declared uint8/uint16 full dtype range, or normalized float32/64 in [0,1]; no per-image rescaling."},
                "observation_count": len(vertices), "supported_count": int(supported.sum()),
                "unsupported_count": int((~supported).sum()), "edge_count": len(result), **counts,
                "memory": {"tile_size": tile_size, "max_rgb_window_pixels": appearance.max_window_pixels,
                    "max_support_window_pixels": support.max_window_pixels_seen,
                    "descriptor_cache_limit": DESCRIPTOR_CACHE_SIZE,
                    "descriptor_cache_hits": appearance.cache_hits,
                    "descriptors_computed": appearance.descriptors_computed,
                    "policy": "O(vertices+cardinal edges) graph memory; tile-bounded requested image/support windows and a fixed-size descriptor cache; path samples streamed.",
                    "measurement_scope": "Requested-window pixel counts, not peak RSS. Excludes backend decoding and OS mmap/page cache. Each RasterReader uses WindowReader defaults: 32 MiB decoded-tile LRU and 128 MiB compressed-segment ceiling."},
                "interpretation": "Image appearance heuristic, not biological confidence, calibrated probability, or validated tissue boundaries.",
                "expert_annotations_used": False, "model_inference_used": False}
    return result, vertices, metadata
