#!/usr/bin/env python3
"""Measure canonical nuclear masks and their H&E appearance with bounded image I/O.

Geometry describes the actual labelled pixel union, not the input detector polygon.
RGB optical density is an appearance/texture proxy; it is not a deconvolved stain,
measured protein, or calibrated chromatin quantity. Image and labels must share
the same level-zero crop coordinates. CSV bounding boxes are advisory only: the
mask is scanned in bounded tiles to verify IDs and recover exact raster bounds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_morphology_io import (GEOMETRY_FIELDS, APPEARANCE_FIELDS, FIELDS, CONTRACT,
    TABLE, SUMMARY, COMPLETION, OD_EPSILON, OD_MAX, TEXTURE_LEVELS, capture_sources,
    producer_identity, intensity_settings, texture_sampling, complete_morphology, json_bytes)


class WindowReader:
    """Level-zero TIFF/Zarr reader; never falls back to decoding a whole image.

    TIFF tile/strip decoding also works without Zarr, including deflate TIFFs.
    A small LRU avoids repeatedly decoding shared compressed tiles for nearby cells.
    An oversized compressed strip fails with an actionable error instead of risking
    a whole-slide allocation. Uncompressed TIFFs use a read-only memory map.
    """

    def __init__(self, path, *, cache_mb=32, max_segment_mb=128):
        self.path = Path(path)
        self._tf = None
        self._array = None
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = int(cache_mb * 1024**2)
        self._max_segment_bytes = int(max_segment_mb * 1024**2)
        if self.path.is_dir():
            try:
                import zarr
            except ImportError as exc:
                raise RuntimeError("Zarr input requires the zarr package") from exc
            arr = zarr.open(str(self.path), mode="r")
            if not hasattr(arr, "shape"):
                if "0" not in arr:
                    raise ValueError("Zarr group must contain its level-zero array at '0'")
                arr = arr["0"]
            if len(arr.shape) not in (2, 3) or (len(arr.shape) == 3 and arr.shape[2] not in (1, 3, 4)):
                raise ValueError("Zarr must be YX or YXS; RGB samples must be last")
            self._array = arr
            self.shape = tuple(arr.shape)
            self.dtype = np.dtype(arr.dtype)
            self.backend = "zarr_windows"
            return
        self._tf = tifffile.TiffFile(self.path)
        series = self._tf.series[0].levels[0]
        if len(series.pages) != 1:
            self.close()
            raise ValueError("Expected a single 2D level-zero TIFF page, optionally RGB")
        self._page = series.pages[0]
        page = self._page
        if int(page.imagedepth) != 1 or page.samplesperpixel not in (1, 3, 4):
            self.close()
            raise ValueError("Only 2D greyscale or RGB(A) TIFF pages are supported")
        self._height, self._width = int(page.imagelength), int(page.imagewidth)
        self._samples = int(page.samplesperpixel)
        self.shape = (self._height, self._width) + ((self._samples,) if self._samples > 1 else ())
        self.dtype = page.dtype
        try:
            self._array = tifffile.memmap(self.path, series=0, level=0, mode="r")
            self.backend = "tiff_memmap"
        except (ValueError, OSError):
            self.backend = "tiff_segment_windows"
        self._separate = int(page.planarconfig) == 2
        self._tile_w = int(page.tilewidth) if page.is_tiled else self._width
        self._tile_h = int(page.tilelength) if page.is_tiled else int(page.rowsperstrip)
        self._ncols = math.ceil(self._width / self._tile_w)
        self._nrows = math.ceil(self._height / self._tile_h)
        if self._array is None:
            segment_bytes = self._tile_h * self._tile_w * self.dtype.itemsize * (1 if self._separate else self._samples)
            if segment_bytes > self._max_segment_bytes:
                self.close()
                raise ValueError("Compressed TIFF segment exceeds memory limit; retile the input TIFF (e.g. 512x512)")

    def _segment(self, index):
        if index in self._cache:
            return self._cached(index)
        page = self._page
        fh = self._tf.filehandle
        fh.seek(page.dataoffsets[index])
        encoded = fh.read(page.databytecounts[index])
        kwargs = {"_fullsize": page.is_tiled}
        if int(page.compression) in (6, 7, 34892, 33007):
            kwargs.update(jpegtables=page.jpegtables, jpegheader=page.keyframe.jpegheader)
        decoded, position, _ = page.keyframe.decode(encoded, index, **kwargs)
        item = (decoded, position)
        if decoded is not None and decoded.nbytes <= self._cache_limit:
            while self._cache and self._cache_bytes + decoded.nbytes > self._cache_limit:
                old, _ = self._cache.popitem(last=False)[1]
                self._cache_bytes -= old.nbytes
            self._cache[index] = item
            self._cache_bytes += decoded.nbytes
        return item

    def _cached(self, index):
        item = self._cache.pop(index)
        self._cache[index] = item
        return item

    def read(self, x0, y0, x1, y1):
        """Read clipped, half-open pixel-edge bounds, returning YX or YXS."""
        height, width = self.shape[:2]
        x0, x1 = max(0, int(x0)), min(width, int(x1))
        y0, y1 = max(0, int(y0)), min(height, int(y1))
        if x1 < x0 or y1 < y0:
            raise ValueError("Inverted or out-of-image window")
        if self._array is not None:
            if self.backend == "tiff_memmap" and self._separate:
                return np.moveaxis(np.asarray(self._array[:, y0:y1, x0:x1]), 0, -1)
            return np.asarray(self._array[y0:y1, x0:x1])
        result = np.zeros((y1-y0, x1-x0, self._samples), dtype=self.dtype)
        for plane in range(self._samples if self._separate else 1):
            for ty in range(y0 // self._tile_h, math.ceil(y1 / self._tile_h)):
                for tx in range(x0 // self._tile_w, math.ceil(x1 / self._tile_w)):
                    index = plane * self._ncols * self._nrows + ty * self._ncols + tx
                    tile, pos = self._segment(index)
                    if tile is None:
                        continue
                    sy, sx, sample = int(pos[2]), int(pos[3]), int(pos[0] if self._separate else pos[4])
                    tile = tile[0]
                    ya, yb = max(y0, sy), min(y1, sy + tile.shape[0])
                    xa, xb = max(x0, sx), min(x1, sx + tile.shape[1])
                    result[ya-y0:yb-y0, xa-x0:xb-x0, sample:sample+tile.shape[2]] = tile[ya-sy:yb-sy, xa-sx:xb-sx]
        return result[..., 0] if self._samples == 1 else result

    def close(self):
        if self._tf is not None:
            self._tf.close()
            self._tf = None
        self._cache.clear()
        self._array = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_objects(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "x", "y", "xmin", "ymin", "xmax", "ymax"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Objects table requires columns {sorted(required)}")
        records = []
        seen = set()
        for row in reader:
            try:
                label = int(row["label"])
            except ValueError as exc:
                raise ValueError("Canonical labels must be positive integers") from exc
            if label <= 0 or label in seen:
                raise ValueError(f"Duplicate or nonpositive canonical label {label}")
            seen.add(label)
            for name in required - {"label"}:
                row[name] = float(row[name])
                if not math.isfinite(row[name]):
                    raise ValueError(f"Nonfinite object coordinate for label {label}: {name}")
            row["label"] = label
            records.append(row)
    return records


def read_calibration(path, resolution_json=None):
    shift = json.loads(Path(path).read_text())
    calibration = shift
    if resolution_json is not None:
        calibration = json.loads(Path(resolution_json).read_text())
        if calibration.get("status") != "pass":
            raise ValueError("Resolution report must have status 'pass'")
    scalar = calibration.get("source_mpp", calibration.get("microns_per_pixel", calibration.get("effective_mpp")))
    mx = calibration.get("mpp_x", calibration.get("source_mpp_x", scalar))
    my = calibration.get("mpp_y", calibration.get("source_mpp_y", scalar))
    if mx is None or my is None or not all(math.isfinite(float(v)) and float(v) > 0 for v in (mx, my)):
        raise ValueError("Shift metadata must supply positive source_mpp/microns_per_pixel or mpp_x,mpp_y")
    if not all(0.01 <= float(v) <= 10 for v in (mx, my)):
        raise ValueError("MPP outside plausible nuclear-image range [0.01, 10] um/px; provide a passed --resolution-json report with verified calibration")
    offset = shift.get("offset_crop_to_original", {})
    dx, dy = float(offset.get("dx", 0)), float(offset.get("dy", 0))
    if not all(math.isfinite(v) for v in (dx, dy)):
        raise ValueError("Crop offset must be finite")
    return float(mx), float(my), dx, dy


def scan_labels(reader, expected_ids, tile_size=1024):
    """Verify complete label coverage and collect exact raster bounds/counts."""
    if len(reader.shape) != 2 or reader.dtype.kind not in "ui":
        raise ValueError("Canonical mask must be a 2D integer label image")
    expected = set(expected_ids)
    found = {}
    h, w = reader.shape
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            tile = reader.read(x0, y0, min(w, x0+tile_size), min(h, y0+tile_size))
            yy, xx = np.nonzero(tile)
            if not len(xx):
                continue
            ids, inverse, counts = np.unique(tile[yy, xx], return_inverse=True, return_counts=True)
            extras = set(map(int, ids)) - expected
            if extras:
                raise ValueError(f"Mask labels absent from objects table: {sorted(extras)[:10]}")
            xmin = np.full(len(ids), tile.shape[1], dtype=np.int64)
            ymin = np.full(len(ids), tile.shape[0], dtype=np.int64)
            xmax, ymax = np.zeros(len(ids), dtype=np.int64), np.zeros(len(ids), dtype=np.int64)
            np.minimum.at(xmin, inverse, xx)
            np.minimum.at(ymin, inverse, yy)
            np.maximum.at(xmax, inverse, xx)
            np.maximum.at(ymax, inverse, yy)
            for i, raw in enumerate(ids):
                label = int(raw)
                box = [int(xmin[i]+x0), int(ymin[i]+y0), int(xmax[i]+x0+1), int(ymax[i]+y0+1), int(counts[i])]
                if label in found:
                    prev = found[label]
                    box = [min(prev[0], box[0]), min(prev[1], box[1]), max(prev[2], box[2]), max(prev[3], box[3]), prev[4]+box[4]]
                found[label] = box
    missing = expected - found.keys()
    if missing:
        raise ValueError(f"Canonical labels absent from mask: {sorted(missing)[:10]}")
    return found


def geometry_features(mask, mpp_x, mpp_y, *, x0=0, y0=0, dx=0, dy=0):
    """Exact pixel-union area/edge perimeter and pixel-area second moments."""
    mask = np.asarray(mask, dtype=bool)
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("Cannot profile an empty canonical mask")
    area = float(len(xx) * mpp_x * mpp_y)
    padded = np.pad(mask.astype(np.int8), 1)
    perimeter = float(np.abs(np.diff(padded, axis=0)).sum() * mpp_x + np.abs(np.diff(padded, axis=1)).sum() * mpp_y)
    points = np.column_stack((xx * mpp_x, yy * mpp_y))
    center = points.mean(axis=0)
    delta = points - center
    covariance = delta.T @ delta / len(points) + np.diag([mpp_x**2/12, mpp_y**2/12])
    eigenvalues, vectors = np.linalg.eigh(covariance)
    minor, major = np.maximum(eigenvalues, 0)
    axis = vectors[:, 1]
    orientation = float(math.atan2(axis[1], axis[0]) % math.pi) if major - minor > 1e-10 * major else float("nan")
    boundary = mask & ~ndimage.binary_erosion(mask)
    by, bx = np.nonzero(boundary)
    corners = np.concatenate([np.column_stack(((bx + ox)*mpp_x, (by + oy)*mpp_y)) for ox, oy in ((0,0), (1,0), (0,1), (1,1))])
    hull_area = float(ConvexHull(corners).volume)
    cx, cy = float(xx.mean()+x0+0.5), float(yy.mean()+y0+0.5)
    return {
        "mask_pixel_count": len(xx), "area_um2": area, "perimeter_um": perimeter,
        "circularity": 4 * math.pi * area / perimeter**2,
        "boundary_irregularity": perimeter**2 / (4 * math.pi * area),
        "solidity": min(1.0, area / hull_area), "eccentricity": math.sqrt(max(0.0, 1-minor/major)),
        "major_axis_um": 4*math.sqrt(major), "minor_axis_um": 4*math.sqrt(minor),
        "orientation_rad": orientation, "mask_centroid_x": cx, "mask_centroid_y": cy,
        "mask_centroid_x_um": (cx+dx)*mpp_x, "mask_centroid_y_um": (cy+dy)*mpp_y,
        "connected_components": int(ndimage.label(mask, structure=np.ones((3, 3)))[1]),
    }


def appearance_features(image, mask, mpp_x, mpp_y, *, white_level=None, levels=TEXTURE_LEVELS, texture_lag_um=.5):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4) or image.shape[:2] != mask.shape:
        raise ValueError("H&E must be RGB(A) with the same crop dimensions as labels")
    image = image[..., :3]
    white_level = intensity_settings(image.dtype, white_level)["white_level"]
    if type(levels) is not int or not 2 <= levels <= 256:
        raise ValueError("Texture levels must be an integer in [2,256]")
    sampling = texture_sampling(mpp_x, mpp_y, texture_lag_um)
    pixel_values = image[mask].astype(np.float64)
    if not np.isfinite(pixel_values).all() or (pixel_values < 0).any() or (pixel_values > white_level).any():
        raise ValueError("Nuclear RGB intensities are outside the specified white-level range")
    if not len(pixel_values):
        raise ValueError("Cannot profile empty nuclear appearance")
    od_values = -np.log((pixel_values / white_level + OD_EPSILON) / (1 + OD_EPSILON))
    mean_od = od_values.mean(axis=1)
    od = np.zeros(mask.shape, dtype=np.float64)
    od[mask] = mean_od
    result = {}
    for channel, values in zip(("red", "green", "blue", "mean"), [*od_values.T, mean_od]):
        for stat, value in zip(("mean", "std", "p10", "p90"), (values.mean(), values.std(), *np.percentile(values, [10, 90]))):
            result[f"rgb_od_{channel}_{stat}"] = float(value)
    quantized = np.clip(np.floor(od / OD_MAX * levels), 0, levels-1).astype(np.int32)
    histogram = np.zeros((levels, levels), dtype=np.int64)
    differences = 0.0
    pair_count = 0
    for sx, sy in sampling["offsets_px"]:
        if abs(sx) >= mask.shape[1] or sy >= mask.shape[0]:
            continue
        ys = slice(0, mask.shape[0]-sy)
        yt = slice(sy, None)
        xs, xt = (slice(0, -sx), slice(sx, None)) if sx > 0 else ((slice(-sx, None), slice(0, sx)) if sx < 0 else (slice(None), slice(None)))
        pairs = mask[ys, xs] & mask[yt, xt]
        a, b = quantized[ys, xs][pairs], quantized[yt, xt][pairs]
        histogram += np.bincount(a*levels+b, minlength=levels*levels).reshape(levels, levels)
        differences += float(np.abs(od[ys, xs][pairs]-od[yt, xt][pairs]).sum() / math.hypot(sx*mpp_x, sy*mpp_y))
        pair_count += len(a)
    if pair_count:
        probabilities = (histogram + histogram.T) / (2*pair_count)
        grid = np.arange(levels) / (levels-1)
        contrast = (grid[:, None]-grid[None, :])**2
        nonzero = probabilities[probabilities > 0]
        result.update(od_glcm_contrast=float((probabilities*contrast).sum()),
                      od_glcm_homogeneity=float((probabilities/(1+contrast)).sum()),
                      od_glcm_entropy_bits=float(-(nonzero*np.log2(nonzero)).sum()),
                      od_gradient_mean_per_um=differences/pair_count)
    else:
        result.update({field: float("nan") for field in ("od_glcm_contrast", "od_glcm_homogeneity", "od_glcm_entropy_bits", "od_gradient_mean_per_um")})
    result["od_texture_pair_count"] = pair_count
    result["image_saturated_fraction"] = float(np.any((pixel_values == 0) | (pixel_values == white_level), axis=1).mean())
    return result


def profile_cells(labels, image, objects, shift, outdir, *, tile_size=1024, max_cell_pixels=4_000_000, white_level=None, resolution_json=None, texture_lag_um=.5):
    source_paths = {key: path for key, path in {"labels": labels, "image": image, "objects": objects,
        "shift": shift, "resolution_json": resolution_json}.items() if path is not None}
    sources = capture_sources(source_paths)
    producer, producer_paths = producer_identity()
    outdir = Path(outdir)
    if any((outdir / name).exists() or (outdir / name).is_symlink() for name in (TABLE, SUMMARY, COMPLETION)):
        raise FileExistsError("Morphology output already exists; choose a new directory")
    records = read_objects(objects)
    mx, my, dx, dy = read_calibration(shift, resolution_json)
    with WindowReader(labels) as label_reader, WindowReader(image) as image_reader:
        if image_reader.shape[:2] != label_reader.shape or len(image_reader.shape) != 3:
            raise ValueError("RGB H&E and canonical mask must share level-zero crop dimensions")
        crop_size = json.loads(Path(shift).read_text()).get("crop_size", {})
        if crop_size and (crop_size.get("height"), crop_size.get("width")) != label_reader.shape:
            raise ValueError("Shift crop_size differs from the canonical mask dimensions")
        if resolution_json:
            report = json.loads(Path(resolution_json).read_text())
            if "width_px" in report and "height_px" in report:
                height, width = label_reader.shape
                if dx < 0 or dy < 0 or dx+width > report["width_px"] or dy+height > report["height_px"]:
                    raise ValueError("Crop-to-original transform lies outside the calibrated source image")
        settings = {"intensity": intensity_settings(image_reader.dtype, white_level),
                    "texture": texture_sampling(mx, my, texture_lag_um),
                    "scan_tile_size": tile_size, "max_cell_pixels": max_cell_pixels,
                    "source_channels": image_reader.shape[2]}
        geometry = {"mpp_xy": [mx, my], "crop_origin_px": [dx, dy],
                    "crop_size_px": [label_reader.shape[1], label_reader.shape[0]]}
        bounds = scan_labels(label_reader, [row["label"] for row in records], tile_size)
        oversized = [label for label, (x0, y0, x1, y1, _) in bounds.items() if (x1-x0)*(y1-y0) > max_cell_pixels]
        if oversized:
            raise ValueError(f"Cell bounding boxes exceed max-cell-pixels: {oversized[:10]}; inspect disconnected/corrupt labels")
        outdir.mkdir(parents=True, exist_ok=True)
        height, width = label_reader.shape
        with (outdir / TABLE).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for record in records:
                label = record["label"]
                x0, y0, x1, y1, count = bounds[label]
                mask = label_reader.read(x0, y0, x1, y1) == label
                if int(mask.sum()) != count:
                    raise RuntimeError(f"Mask changed while profiling canonical label {label}")
                rgb = image_reader.read(x0, y0, x1, y1)
                features = geometry_features(mask, mx, my, x0=x0, y0=y0, dx=dx, dy=dy)
                appearance = appearance_features(rgb, mask, mx, my, white_level=white_level, texture_lag_um=texture_lag_um)
                row = {"label": label, "x": record["x"], "y": record["y"],
                       "geometry_source": record.get("geometry_source", "unspecified"),
                       "geometry_representation": "canonical_label_pixel_union",
                       "morphology_status": "ok" if features["connected_components"] == 1 else "disconnected_label",
                       "morphology_contract": CONTRACT,
                       "texture_status": "ok" if appearance["od_texture_pair_count"] else "no_pairs_at_requested_physical_lag",
                       "mask_bbox_xmin": x0, "mask_bbox_ymin": y0, "mask_bbox_xmax": x1, "mask_bbox_ymax": y1,
                       "touches_image_edge": int(x0 == 0 or y0 == 0 or x1 == width or y1 == height),
                       **features, **appearance}
                writer.writerow({key: ("" if isinstance(value, float) and not math.isfinite(value) else value) for key, value in row.items()})
        summary = {
            "schema_version": "2.0", "binding_contract": CONTRACT,
            "cells": len(records), "mask_pixels": sum(box[4] for box in bounds.values()),
            "settings": settings, "geometry": geometry, "producer": producer, "source_identities": sources,
            "mpp_x": mx, "mpp_y": my, "crop_offset_xy": [dx, dy],
            "calibration_source": str(resolution_json) if resolution_json else str(shift),
            "coordinate_frame": "level-zero crop pixel edges; physical centroids include crop-to-original offset",
            "bbox_convention": "half-open pixel edges clipped to image; recovered from actual label mask",
            "geometry_representation": "canonical_label_pixel_union",
            "perimeter_definition": "exact exposed pixel-edge length, including holes; grid-dependent (not a rotation-invariant smooth-contour estimate)",
            "axis_definition": "4 sqrt(eigenvalue) of pixel-area second moments in physical coordinates",
            "orientation_definition": "major axis from +x toward +y modulo pi (clockwise in image coordinates); empty for isotropic masks",
            "solidity_definition": "pixel-union area divided by convex hull area of pixel corners",
            "optical_density_definition": "-ln((RGB/white_level+1/255)/(1+1/255)); normalized transmittance, not stain deconvolution or chromatin calibration",
            "texture_definition": "symmetric 32-level GLCM of mean RGB OD; recorded physical-lag offsets within each nucleus; fixed OD range [0,ln256]",
            "white_level": settings["intensity"]["white_level"],
            "missing_canonical_labels": 0, "unknown_mask_labels": 0,
            "io": {"labels": label_reader.backend, "image": image_reader.backend, "scan_tile_size": tile_size, "max_cell_pixels": max_cell_pixels, "whole_image_decode": False},
            "inputs": {"labels": str(labels), "image": str(image), "objects": str(objects), "shift": str(shift)},
            "feature_columns": FIELDS,
        }
        with (outdir / SUMMARY).open("xb") as handle:
            handle.write(json_bytes(summary))
        complete_morphology(outdir, sources=sources, source_paths=source_paths, producer=producer,
                            producer_paths=producer_paths, settings=settings, geometry=geometry)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("labels", "image", "objects", "shift", "outdir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--max-cell-pixels", type=int, default=4_000_000)
    parser.add_argument("--white-level", type=float)
    parser.add_argument("--texture-lag-um", type=float, default=.5, help="Target axial intranuclear texture lag in micrometres; effective rounded pixel offsets are recorded")
    parser.add_argument("--resolution-json", help="Passed input-resolution report; overrides stale MPP in shift metadata while retaining its crop offset")
    args = parser.parse_args()
    if args.tile_size <= 0 or args.max_cell_pixels <= 0:
        parser.error("Tile size and maximum cell pixels must be positive")
    summary = profile_cells(**vars(args))
    print(json.dumps({"cells": summary["cells"], "outdir": args.outdir}))


if __name__ == "__main__":
    main()
