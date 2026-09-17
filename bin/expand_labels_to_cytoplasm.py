#!/usr/bin/env python3

"""
expand_cells_to_cytoplasm_v3.py

The legacy --expand-px route below is retained for compatibility. The --expand-um
route uses verified physical calibration and a registered tissue support mask.
It writes separate canonical nuclei, a disjoint perinuclear ring, and a whole-cell
approximation, plus per-cell truncation and contact-risk flags. It does not infer
membranes or measure true cytoplasmic compartments.

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
import csv
import hashlib
import heapq
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_cell_morphology import WindowReader, read_calibration

try:
    from scipy.ndimage import distance_transform_edt
except Exception:
    distance_transform_edt = None

try:
    from PIL import Image
except Exception:
    Image = None


DEFAULT_PALETTE = np.array([
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
], dtype=np.uint8)


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


class LazyLabelReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._reader = WindowReader(path)
        if len(self._reader.shape) != 2 or self._reader.dtype.kind not in "ui":
            self.close()
            raise ValueError("Expected a 2D integer label image")
        self.shape = self._reader.shape
        self.dtype = self._reader.dtype

    def read(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        return self._reader.read(x0, y0, x1, y1)

    def downsample(self, factor: int) -> np.ndarray:
        f = int(max(1, factor))
        h, w = self.shape
        arr = np.empty((math.ceil(h/f), math.ceil(w/f)), dtype=self.dtype)
        for y in range(0, h, 512):
            for x in range(0, w, 512):
                yy, xx = min(y+512, h), min(x+512, w)
                first_y, first_x = (-y) % f, (-x) % f
                if first_y >= yy-y or first_x >= xx-x:
                    continue
                block = self.read(y, yy, x, xx)[first_y::f, first_x::f]
                oy, ox = (y+first_y)//f, (x+first_x)//f
                arr[oy:oy+block.shape[0], ox:ox+block.shape[1]] = block
        return arr

    def close(self) -> None:
        self._reader.close()


def _geodesic_expand_python(labels, support, radius, mx, my):
    """Multi-source Dijkstra; ties choose smaller ID; no diagonal corner cutting.

    The radius bounds physical length along the 8-neighbour pixel-centre graph,
    not a deconvolved cell membrane or smooth Euclidean contour. Nuclear pixels
    are immutable, including flagged pixels outside tissue support.
    """
    h, w = labels.shape
    owner = labels.copy()
    dist = np.full((h, w), np.inf, dtype=np.float64)
    queue = [(0.0, 0, 0, 0)]
    queue.pop()
    for y in range(h):
        for x in range(w):
            if labels[y, x] != 0:
                dist[y, x] = 0
                if support[y, x]:
                    for sy, sx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        yy, xx = y+sy, x+sx
                        if 0 <= yy < h and 0 <= xx < w and support[yy, xx] and labels[yy, xx] == 0:
                            heapq.heappush(queue, (0.0, np.int64(labels[y, x]), y, x))
                            break
    while queue:
        distance, label, y, x = heapq.heappop(queue)
        if distance > dist[y, x]+1e-10 or label != owner[y, x]:
            continue
        for sy, sx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            yy, xx = y+sy, x+sx
            if yy < 0 or yy >= h or xx < 0 or xx >= w or not support[yy, xx] or labels[yy, xx] != 0:
                continue
            if sy != 0 and sx != 0 and (not support[y, xx] or not support[yy, x]):
                continue
            candidate = distance+math.hypot(sx*mx, sy*my)
            if candidate > radius+1e-10:
                continue
            if candidate < dist[yy, xx]-1e-10 or (abs(candidate-dist[yy, xx]) <= 1e-10 and label < owner[yy, xx]):
                dist[yy, xx] = candidate
                owner[yy, xx] = label
                heapq.heappush(queue, (candidate, label, yy, xx))
    return owner


_geodesic_accelerated = None


def expand_labels_physical(labels, support, expand_um, mpp_x, mpp_y, *, accelerate=True):
    global _geodesic_accelerated
    labels = np.asarray(labels)
    support = np.asarray(support, dtype=bool)
    if labels.ndim != 2 or labels.dtype.kind not in "ui" or support.shape != labels.shape:
        raise ValueError("Labels must be 2D integers with aligned tissue support")
    if np.any(labels < 0):
        raise ValueError("Negative canonical labels are not valid")
    if labels.dtype.kind == "u" and np.any(labels > np.iinfo(np.int64).max):
        raise ValueError("Canonical IDs exceed signed 64-bit range")
    if not all(math.isfinite(v) and v > 0 for v in (mpp_x, mpp_y)) or not math.isfinite(expand_um) or expand_um < 0:
        raise ValueError("MPP must be positive and radius nonnegative")
    if expand_um == 0 or not np.any(labels):
        return labels.copy()
    implementation = _geodesic_expand_python
    if accelerate:
        if _geodesic_accelerated is None:
            try:
                from numba import njit
                _geodesic_accelerated = njit(cache=False)(_geodesic_expand_python)
            except ImportError:
                _geodesic_accelerated = False
        if _geodesic_accelerated:
            implementation = _geodesic_accelerated
    return implementation(np.ascontiguousarray(labels), np.ascontiguousarray(support), float(expand_um), float(mpp_x), float(mpp_y))


class TissueSupportReader:
    """Nearest-neighbour support sampling with an explicit crop/original frame."""

    def __init__(self, path, frame, crop_shape, source_shape, offset):
        self.reader = WindowReader(path)
        if len(self.reader.shape) != 2:
            self.reader.close()
            raise ValueError("Tissue support must be 2D")
        self.frame = frame
        self.height, self.width = crop_shape if frame == "crop" else source_shape
        self.dx, self.dy = (0, 0) if frame == "crop" else offset
        h, w = crop_shape
        if self.dx < 0 or self.dy < 0 or self.dx+w > self.width or self.dy+h > self.height:
            self.reader.close()
            raise ValueError("Crop transform lies outside the tissue-support coordinate frame")

    def read(self, x0, y0, x1, y1):
        th, tw = self.reader.shape
        xx = np.floor((np.arange(x0, x1)+self.dx+0.5)*tw/self.width).astype(np.int64)
        yy = np.floor((np.arange(y0, y1)+self.dy+0.5)*th/self.height).astype(np.int64)
        if not len(xx) or not len(yy):
            return np.zeros((len(yy), len(xx)), dtype=bool)
        arr = self.reader.read(int(xx[0]), int(yy[0]), int(xx[-1]+1), int(yy[-1]+1))
        if not np.isfinite(arr).all() or np.any(arr < 0):
            raise ValueError("Tissue support contains negative or nonfinite values")
        return arr[np.ix_(yy-yy[0], xx-xx[0])] > 0

    def close(self):
        self.reader.close()


def expand_physical_tiled(reader, support_reader, expand_um, mx, my, tile_size, out_mm, *, accelerate=True):
    if tile_size <= 0:
        raise ValueError("Tile size must be positive")
    h, w = reader.shape
    # One additional pixel permits exact neighbour-contact flags at core edges.
    hy, hx = math.ceil(expand_um/my)+1, math.ceil(expand_um/mx)+1
    stats = {}
    for y0 in range(0, h, tile_size):
        y1 = min(h, y0+tile_size)
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0+tile_size)
            ay, by = max(0, y0-hy), min(h, y1+hy)
            ax, bx = max(0, x0-hx), min(w, x1+hx)
            labels = reader.read(y0=ay, y1=by, x0=ax, x1=bx)
            tissue = support_reader.read(ax, ay, bx, by)
            whole = expand_labels_physical(labels, tissue, expand_um, mx, my, accelerate=accelerate)
            cy, cx = slice(y0-ay, y1-ay), slice(x0-ax, x1-ax)
            core = whole[cy, cx]
            nuclei = labels[cy, cx]
            if np.any(core[nuclei > 0] != nuclei[nuclei > 0]) or np.any((core > 0) & (nuclei == 0) & ~tissue[cy, cx]):
                raise RuntimeError("Compartment label/tissue invariant violated")
            out_mm[y0:y1, x0:x1] = core
            touched = np.zeros(whole.shape, dtype=bool)
            for sy, sx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                ys, yt = slice(0, whole.shape[0]-sy), slice(sy, None)
                xs, xt = (slice(None, -sx), slice(sx, None)) if sx > 0 else ((slice(-sx, None), slice(None, sx)) if sx < 0 else (slice(None), slice(None)))
                a, b = whole[ys, xs], whole[yt, xt]
                contacts = (a != 0) & (b != 0) & (a != b) & tissue[ys, xs] & tissue[yt, xt]
                if sy and sx:
                    contacts &= tissue[ys, xt] & tissue[yt, xs]
                touched[ys, xs] |= contacts
                touched[yt, xt] |= contacts
            # Border=True avoids treating crop edges as tissue gaps: those have a
            # separate flag. Distance is Euclidean only for this QC proximity flag.
            if np.all(tissue):
                near_tissue_gap = np.zeros(nuclei.shape, dtype=bool)
            else:
                near_tissue_gap = distance_transform_edt(tissue, sampling=(my, mx))[cy, cx] <= expand_um
            yy, xx = np.mgrid[y0:y1, x0:x1]
            near_image_edge = np.minimum.reduce(((xx+0.5)*mx, (w-xx-0.5)*mx, (yy+0.5)*my, (h-yy-0.5)*my)) <= expand_um
            def get_stats(label):
                return stats.setdefault(int(label), {"label": int(label), "nucleus_pixels": 0, "ring_pixels": 0,
                    "whole_cell_pixels": 0, "nucleus_outside_support_pixels": 0,
                    "tissue_truncation_flag": 0, "image_truncation_flag": 0,
                    "crowding_flag": 0, "possible_neighbour_contamination_flag": 0})
            for values, field in ((core, "whole_cell_pixels"), (nuclei, "nucleus_pixels"),
                                  (nuclei[~tissue[cy, cx]], "nucleus_outside_support_pixels")):
                ids, counts = np.unique(values[values > 0], return_counts=True)
                for raw, count in zip(ids, counts):
                    get_stats(raw)[field] += int(count)
            for values, field in ((nuclei[near_tissue_gap], "tissue_truncation_flag"),
                                  (nuclei[near_image_edge], "image_truncation_flag"),
                                  (core[touched[cy, cx]], "crowding_flag")):
                for raw in np.unique(values[values > 0]):
                    get_stats(raw)[field] = 1
                    if field == "crowding_flag":
                        get_stats(raw)["possible_neighbour_contamination_flag"] = 1
    for row in stats.values():
        row["ring_pixels"] = row["whole_cell_pixels"]-row["nucleus_pixels"]
    return stats


def write_compartments(labels_path, out_path, expand_um, resolution_json, shift, tissue_mask,
                       *, tissue_frame="original", compartments_dir=None, tile_size=1024,
                       compression="zlib", accelerate=True, label_frame="crop"):
    """Write a whole-cell approximation plus separate immutable nuclei and ring."""
    def artifact(path):
        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"path": str(path.resolve()), "sha256": digest.hexdigest()}

    input_paths = {"labels": labels_path, "tissue_mask": tissue_mask, "shift": shift, "resolution": resolution_json}
    input_artifacts = {name: artifact(path) for name, path in input_paths.items()}
    mx, my, dx, dy = read_calibration(shift, resolution_json)
    metadata = json.loads(Path(shift).read_text())
    report = json.loads(Path(resolution_json).read_text())
    if not all(key in report for key in ("width_px", "height_px")):
        raise ValueError("Physical mode requires source width_px/height_px in the passed resolution report")
    out_path = Path(out_path)
    destination = Path(compartments_dir) if compartments_dir else out_path.parent / f"{out_path.stem}_compartments"
    whole_bundle = destination / "labels_whole_cell.tif"
    if whole_bundle.exists() or whole_bundle.is_symlink():
        raise FileExistsError("Compartment whole-cell bundle already exists; use a new output directory")
    outputs = [out_path, destination / "labels_nucleus.tif", destination / "labels_perinuclear_ring.tif"]
    input_targets = {Path(path).resolve() for path in input_paths.values()}
    if (len(set(p.resolve() for p in outputs)) != len(outputs)
            or input_targets & {p.resolve() for p in outputs + [whole_bundle]}):
        raise ValueError("Compartment output paths must be distinct from each other and all source artifacts")
    if not math.isfinite(expand_um) or expand_um < 0 or tile_size <= 0:
        raise ValueError("Expansion radius must be nonnegative and tile size positive")
    if (tile_size+2*(math.ceil(expand_um/min(mx, my))+1))**2 > 25_000_000:
        raise ValueError("Requested tile and physical halo exceed bounded-memory limit; reduce tile size/radius")
    reader = LazyLabelReader(Path(labels_path))
    support = None
    try:
        h, w = reader.shape
        if label_frame not in ("crop", "original"):
            raise ValueError("Label frame must be crop or original")
        source_shape = (int(report["height_px"]), int(report["width_px"]))
        if label_frame == "original":
            if reader.shape != source_shape or tissue_frame != "original":
                raise ValueError("Original-frame labels require full source dimensions and original-frame support")
            dx, dy = 0, 0
        elif metadata.get("crop_size") and metadata["crop_size"] != {"width": w, "height": h}:
            raise ValueError("Shift crop_size does not match canonical labels")
        support = TissueSupportReader(tissue_mask, tissue_frame, reader.shape, source_shape, (dx, dy))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
        comp = None if str(compression).lower() == "none" else compression
        with tempfile.TemporaryDirectory(prefix="cell_compartments_", dir=out_path.parent) as temp:
            mm = np.memmap(Path(temp) / "whole.raw", mode="w+", dtype=reader.dtype, shape=reader.shape)
            stats = expand_physical_tiled(reader, support, expand_um, mx, my, tile_size, mm, accelerate=accelerate)
            mm.flush()
            tile = 512
            def blocks(kind):
                for y in range(0, h, tile):
                    for x in range(0, w, tile):
                        yy, xx = min(h, y+tile), min(w, x+tile)
                        if kind == "whole":
                            block = np.asarray(mm[y:yy, x:xx])
                        else:
                            original = reader.read(y, yy, x, xx)
                            block = original if kind == "nucleus" else np.where(original == 0, mm[y:yy, x:xx], 0).astype(reader.dtype)
                        yield block
            for target, kind in zip(outputs, ("whole", "nucleus", "ring")):
                tifffile.imwrite(target, data=blocks(kind), shape=reader.shape, dtype=reader.dtype,
                    tile=(tile, tile), compression=comp, bigtiff=True, photometric="minisblack", metadata=None)
            del mm
        fields = ["label", "nucleus_pixels", "ring_pixels", "whole_cell_pixels", "nucleus_area_um2", "ring_area_um2", "whole_cell_area_um2",
                  "nucleus_outside_support_pixels", "tissue_truncation_flag", "image_truncation_flag", "crowding_flag", "possible_neighbour_contamination_flag", "compartment_status"]
        with (destination / "compartment_qc.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for label in sorted(stats):
                row = stats[label]
                for name in ("nucleus", "ring", "whole_cell"):
                    row[f"{name}_area_um2"] = row[f"{name}_pixels"] * mx * my
                row["compartment_status"] = "nucleus_outside_support" if row["nucleus_outside_support_pixels"] else ("empty_ring" if row["ring_pixels"] == 0 else "approximation")
                writer.writerow(row)
        for name, path in input_paths.items():
            if artifact(path)["sha256"] != input_artifacts[name]["sha256"]:
                raise RuntimeError(f"Compartment input changed while processing: {name}; no completed lineage written")
        if whole_bundle.resolve() != out_path.resolve():
            try:
                os.link(out_path, whole_bundle)
            except FileExistsError:
                raise FileExistsError("Compartment whole-cell bundle appeared during processing; refusing overwrite") from None
            except OSError:
                # Exclusive create prevents copy fallback from overwriting a
                # user file or a target created concurrently by another task.
                with out_path.open("rb") as source, whole_bundle.open("xb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        output_paths = {"whole_cell": out_path, "nucleus": outputs[1], "perinuclear_ring": outputs[2],
                        "qc": destination / "compartment_qc.csv", "whole_cell_bundle": whole_bundle}
        output_artifacts = {name: artifact(path) for name, path in output_paths.items()}
        summary = {"schema_version": "1.0", "cells": len(stats), "expand_um": expand_um, "mpp_x": mx, "mpp_y": my,
            "calibration_source": str(resolution_json), "crop_offset_xy": [dx, dy], "tissue_frame": tissue_frame, "label_frame": label_frame,
            "distance_metric": "nearest 8-neighbour tissue-constrained physical path; orthogonal mpp and diagonal hypot(mpp_x,mpp_y) lengths; no diagonal corner cutting; lower ID wins ties",
            "semantics": {"nucleus": "unchanged canonical nuclear pixels", "perinuclear_ring": "whole-cell approximation minus all canonical nuclear pixels", "whole_cell": "nucleus plus a bounded tissue-supported expansion; no inferred membrane"},
            "flags": {"tissue_truncation_flag": "nuclear pixel within radius of tissue exclusion", "image_truncation_flag": "nuclear pixel centre within radius of image edge", "crowding_flag": "grown cell region contacts another cell region (8 neighbours)", "possible_neighbour_contamination_flag": "same contact risk as crowding; not measured contamination", "nucleus_outside_support_pixels": "retained to preserve canonical segmentation; never used to seed expansion outside support"},
            "outputs": {name: str(path) for name, path in output_paths.items()},
            "provenance_schema_version": "cellphenotyper.compartments.v1",
            "inputs": input_artifacts, "output_artifacts": output_artifacts,
            "io": {"mode": "tiled_halo", "tile_size": tile_size, "halo_yx": [math.ceil(expand_um/my)+1, math.ceil(expand_um/mx)+1], "whole_image_decode": False, "numba_accelerated": bool(accelerate and _geodesic_accelerated)},
            "qc": {
                "tissue_truncated_cells": sum(bool(row["tissue_truncation_flag"]) for row in stats.values()),
                "image_truncated_cells": sum(bool(row["image_truncation_flag"]) for row in stats.values()),
                "crowded_cells": sum(bool(row["crowding_flag"]) for row in stats.values()),
                "nuclei_outside_support": sum(bool(row["nucleus_outside_support_pixels"]) for row in stats.values()),
                "nuclear_pixels_outside_support": sum(row["nucleus_outside_support_pixels"] for row in stats.values())}}
        (destination / "compartment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        reader.close()
        if support:
            support.close()


def expand_labels_nonoverlap_tiled_lazy(reader: LazyLabelReader, expand_px: int, tile_size: int, out_mm: np.ndarray) -> None:
    if distance_transform_edt is None:
        die("SciPy is required (scipy.ndimage.distance_transform_edt). It is missing in your environment.")
    if expand_px <= 0:
        h, w = reader.shape
        for y0 in range(0, h, tile_size):
            y1 = min(h, y0 + tile_size)
            for x0 in range(0, w, tile_size):
                x1 = min(w, x0 + tile_size)
                out_mm[y0:y1, x0:x1] = reader.read(y0, y1, x0, x1)
        return
    if tile_size <= 0:
        die("--tile-size must be > 0")

    h, w = reader.shape
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

            core = reader.read(y0, y1, x0, x1)
            core_bg = (core == 0)
            if not core_bg.any():
                out_mm[y0:y1, x0:x1] = core
                continue

            hy0 = max(0, y0 - radius)
            hy1 = min(h, y1 + radius)
            hx0 = max(0, x0 - radius)
            hx1 = min(w, x1 + radius)

            win = reader.read(hy0, hy1, hx0, hx1)
            if win.max() == 0:
                out_mm[y0:y1, x0:x1] = core
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
                out_mm[y0:y1, x0:x1] = core
                continue

            iyy = iy[cy0:cy1, cx0:cx1]
            ixx = ix[cy0:cy1, cx0:cx1]
            nearest_core = win[iyy, ixx]

            out_tile = core.copy()
            out_tile[fill] = nearest_core[fill]
            out_mm[y0:y1, x0:x1] = out_tile


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


def downsample_nearest(arr: np.ndarray, factor: int) -> np.ndarray:
    f = int(max(1, factor))
    if f <= 1:
        return arr
    if arr.ndim == 2:
        return arr[::f, ::f]
    return arr[::f, ::f, ...]


def downsample_source_nearest(arr_like, factor: int) -> np.ndarray:
    f = int(max(1, factor))
    if f <= 1:
        return np.asarray(arr_like)
    return np.asarray(arr_like[::f, ::f])


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] not in (3, 4):
            raise ValueError(f"Unsupported preview image shape: {arr.shape}")
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported preview image shape: {arr.shape}")

    if arr.dtype == np.uint8:
        return arr

    arr_f = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr_f)
    if not finite.any():
        return np.zeros(arr.shape[:2] + (3,), dtype=np.uint8)
    lo = float(np.percentile(arr_f[finite], 1.0))
    hi = float(np.percentile(arr_f[finite], 99.0))
    if hi <= lo:
        hi = lo + 1.0
    arr_f = (arr_f - lo) * (255.0 / (hi - lo))
    return np.clip(arr_f, 0, 255).astype(np.uint8)


def colorize_label_mask(label_mask: np.ndarray, default_value: int = 0) -> np.ndarray:
    out = np.zeros(label_mask.shape + (3,), dtype=np.uint8)
    fg = label_mask != default_value
    if not np.any(fg):
        return out
    palette_idx = np.mod(label_mask[fg].astype(np.int64, copy=False), len(DEFAULT_PALETTE))
    out[fg] = DEFAULT_PALETTE[palette_idx]
    return out


def write_preview_overlay_png(
    label_mask,
    out_png: Path,
    factor_if_large: int,
    size_threshold_mb: float,
    default_value: int,
    preview_background_path: Path | None,
    alpha: float,
) -> None:
    if Image is None:
        log("Preview requested but pillow is unavailable; skipping preview generation.")
        return

    threshold_bytes = int(float(size_threshold_mb) * 1024 * 1024)
    use_factor = 1

    if preview_background_path is not None:
        est_bytes = int(np.prod(label_mask.shape, dtype=np.int64) * np.dtype(getattr(label_mask, "dtype", np.uint32)).itemsize * 4)
        use_factor = int(max(1, factor_if_large)) if est_bytes > threshold_bytes else 1
        if use_factor > 1:
            log(
                "Preview background is large; using a plain white background to avoid reloading the full crop "
                f"(factor={use_factor}, estimated_mb={est_bytes / (1024.0 * 1024.0):.1f})."
            )
            mask_small = downsample_source_nearest(label_mask, use_factor)
            bg_small = np.full(mask_small.shape + (3,), 255, dtype=np.uint8)
        else:
            bg = tifffile.imread(str(preview_background_path))
            if bg.ndim > 3:
                bg = bg[0]
            bg_rgb = to_uint8_rgb(bg)
            label_mask_arr = np.asarray(label_mask)
            if bg_rgb.shape[:2] != label_mask_arr.shape:
                die(
                    f"Preview background shape {bg_rgb.shape[:2]} does not match label mask shape {label_mask_arr.shape}. "
                    "Expected aligned background image."
                )
            bg_small = bg_rgb
            mask_small = label_mask_arr
    else:
        # Memory-safe fallback for huge masks: estimate full preview memory and
        # decide downsample factor before allocating any RGB background.
        est_bytes = int(np.prod(label_mask.shape, dtype=np.int64) * np.dtype(getattr(label_mask, "dtype", np.uint32)).itemsize * 4)
        use_factor = int(max(1, factor_if_large)) if est_bytes > threshold_bytes else 1
        mask_small = downsample_source_nearest(label_mask, use_factor)
        bg_small = np.full(mask_small.shape + (3,), 255, dtype=np.uint8)

    overlay = colorize_label_mask(mask_small, default_value=default_value)
    fg = mask_small != default_value

    out = bg_small.astype(np.float32, copy=True)
    a = float(max(0.0, min(1.0, alpha)))
    out[fg] = (1.0 - a) * out[fg] + a * overlay[fg].astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(str(out_png))
    log(
        f"Wrote cytoplasm overlay preview: {out_png} "
        f"(factor={use_factor}, threshold_mb={size_threshold_mb}, "
        f"estimated_mb={est_bytes / (1024.0 * 1024.0):.1f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand cell label masks to approximate cytoplasm without overlaps.")
    ap.add_argument("--labels", required=True, help="Input label image (2D), e.g. StarDist labels.tif")
    ap.add_argument("--out", required=True, help="Output expanded label image (.tif)")
    radius = ap.add_mutually_exclusive_group(required=True)
    radius.add_argument("--expand-px", type=int, help="Legacy Euclidean expansion radius in pixels (e.g. 8-20)")
    radius.add_argument("--expand-um", type=float, help="Tissue-constrained physical expansion radius in micrometres")
    ap.add_argument("--resolution-json", help="Passed source/converted resolution report with calibrated mpp_x/y and dimensions")
    ap.add_argument("--shift", help="Canonical crop-to-original coordinate transform")
    ap.add_argument("--tissue-mask", help="Binary tissue-support TIFF; only positive pixels permit expansion")
    ap.add_argument("--tissue-frame", choices=["original", "crop"], default="original", help="Tissue mask's spatial frame; downsampled masks are supported")
    ap.add_argument("--label-frame", choices=["original", "crop"], default="crop", help="Whether the label raster covers the analysis crop or full original image")
    ap.add_argument("--compartments-dir", help="Physical mode nucleus/ring masks, per-cell QC and metadata output directory")
    ap.add_argument("--no-accelerate", action="store_true", help="Use Python geodesic implementation even if Numba is installed")
    ap.add_argument("--mode", choices=["auto", "full", "tiled"], default="auto",
                    help="Expansion mode (default: auto).")
    ap.add_argument("--tile-size", type=int, default=2048,
                    help="Core tile size for tiled mode (default: 2048).")
    ap.add_argument("--auto-threshold-mpix", type=float, default=25.0,
                    help="In auto mode, switch to tiled if image > this megapixel threshold (default: 25).")
    ap.add_argument("--compression", default="zlib",
                    help="TIFF compression (default: zlib; use 'none' for no compression)")
    ap.add_argument("--preview", default=None, help="Optional output preview PNG path.")
    ap.add_argument("--preview-background", default=None,
                    help="Optional background TIFF for overlay preview (must align with labels).")
    ap.add_argument("--preview-factor", type=int, default=10,
                    help="Downsample factor used only when preview image is larger than threshold.")
    ap.add_argument("--preview-threshold-mb", type=float, default=100.0,
                    help="Downsample preview only when estimated image+mask size exceeds this threshold (MB).")
    ap.add_argument("--preview-alpha", type=float, default=0.45,
                    help="Overlay alpha for colored cytoplasm mask preview (0..1).")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    out_path = Path(args.out)

    if not labels_path.exists():
        die(f"Labels file not found: {labels_path}")

    if args.expand_um is not None:
        if not all((args.resolution_json, args.shift, args.tissue_mask)):
            ap.error("--expand-um requires --resolution-json, --shift and --tissue-mask")
        summary = write_compartments(labels_path, out_path, args.expand_um, args.resolution_json,
            args.shift, args.tissue_mask, tissue_frame=args.tissue_frame,
            compartments_dir=args.compartments_dir, tile_size=args.tile_size,
            compression=args.compression, accelerate=not args.no_accelerate, label_frame=args.label_frame)
        if args.preview:
            reader = LazyLabelReader(out_path)
            try:
                factor = max(1, int(args.preview_factor), math.ceil(max(reader.shape)/2048))
                preview = reader.downsample(factor)
            finally:
                reader.close()
            if Image is not None:
                Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(colorize_label_mask(preview)).save(args.preview)
        log(f"Wrote physical compartments for {summary['cells']} cells: {out_path}")
        return
    if args.expand_px < 0 or args.tile_size <= 0:
        ap.error("--expand-px must be nonnegative and --tile-size positive")
    if any((args.resolution_json, args.shift, args.tissue_mask, args.compartments_dir)):
        ap.error("Physical metadata and compartment outputs require --expand-um")

    lazy_reader = None
    labels = None
    try:
        lazy_reader = LazyLabelReader(labels_path)
        h, w = lazy_reader.shape
        dtype = lazy_reader.dtype
        n_labels = None
        log(f"Opened labels lazily: {(h, w)}, dtype={dtype}")
    except Exception as exc:
        log(f"Lazy label reader unavailable; falling back to eager load ({exc})")
        labels = read_label_image(labels_path)
        h, w = labels.shape
        dtype = labels.dtype
        n_labels = int(labels.max())
        log(f"Loaded labels: {labels.shape}, dtype={labels.dtype}, n_labels={n_labels}")

    mpix = (h * w) / 1_000_000.0

    mode = args.mode
    if mode == "auto":
        mode = "tiled" if mpix > float(args.auto_threshold_mpix) else "full"
    log(f"Expansion mode: {mode} (image={mpix:.2f} MP)")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    comp = args.compression
    if isinstance(comp, str) and comp.lower() == "none":
        comp = None

    output_for_preview = None
    if mode == "tiled" and lazy_reader is not None:
        tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{out_path.stem}_tmp_", suffix=".tif", dir=str(out_path.parent))
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        tmp_path.unlink(missing_ok=True)
        mm = tifffile.memmap(str(tmp_path), shape=(h, w), dtype=dtype, bigtiff=True)
        expand_labels_nonoverlap_tiled_lazy(lazy_reader, args.expand_px, args.tile_size, mm)
        mm.flush()
        output_for_preview = mm
        if comp is None:
            os.replace(tmp_path, out_path)
        else:
            tile_hw = int(max(256, min(args.tile_size, 2048)))
            tifffile.imwrite(
                str(out_path),
                data=(np.asarray(mm[y0:min(h, y0 + tile_hw), x0:min(w, x0 + tile_hw)])
                      for y0 in range(0, h, tile_hw)
                      for x0 in range(0, w, tile_hw)),
                shape=(h, w),
                dtype=dtype,
                tile=(tile_hw, tile_hw),
                compression=comp,
                bigtiff=True,
                photometric="minisblack",
                metadata=None,
            )
            try:
                tmp_path.unlink()
            except Exception:
                pass
        out_max = int(np.max(mm))
        filled_pixels = int(np.count_nonzero(mm))
    else:
        if labels is None:
            labels = read_label_image(labels_path)
            n_labels = int(labels.max())
        if mode == "tiled":
            out = expand_labels_nonoverlap_tiled(labels, args.expand_px, args.tile_size)
        else:
            if h * w >= 100_000_000:
                log("Note: full EDT mode can use a lot of RAM on very large images.")
            out = expand_labels_nonoverlap(labels, args.expand_px)

        if out.dtype != dtype:
            out = out.astype(dtype, copy=False)
        output_for_preview = out
        bigtiff = out.nbytes >= 4_000_000_000
        tifffile.imwrite(str(out_path), out, compression=comp, bigtiff=bigtiff)
        out_max = int(out.max())
        filled_pixels = int(np.count_nonzero(out))

    log(f"Wrote expanded labels: {out_path}")
    log(f"Output n_labels={out_max}; filled_pixels={filled_pixels}")

    if args.preview:
        bg_path = None
        if args.preview_background:
            bg_path = Path(args.preview_background)
            if not bg_path.exists():
                die(f"Preview background file not found: {bg_path}")

        write_preview_overlay_png(
            output_for_preview,
            Path(args.preview),
            factor_if_large=int(args.preview_factor),
            size_threshold_mb=float(args.preview_threshold_mb),
            default_value=0,
            preview_background_path=bg_path,
            alpha=float(args.preview_alpha),
        )

    if lazy_reader is not None:
        lazy_reader.close()

if __name__ == "__main__":
    main()
