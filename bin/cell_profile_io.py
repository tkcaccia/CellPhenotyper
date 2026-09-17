"""Shared identity, calibration and bounded raster I/O for cell atlas stages."""
from __future__ import annotations

import hashlib
import fnmatch
import json
from pathlib import Path

import numpy as np
import pandas as pd


def staged_files(source, pattern):
    """Find explicitly staged inputs, including Nextflow directory symlinks.

    Follow each resolved directory once to avoid symlink cycles/duplicate rows.
    This traverses only the caller-selected input roots, never the workspace.
    """
    source = Path(source)
    if source.is_file():
        return [source] if fnmatch.fnmatch(source.name, pattern) else []
    pending, visited, found = [source], set(), {}
    while pending:
        directory = pending.pop()
        resolved = directory.resolve()
        if resolved in visited or not directory.is_dir():
            continue
        visited.add(resolved)
        for child in directory.iterdir():
            if child.is_dir():
                pending.append(child)
            elif child.is_file() and fnmatch.fnmatch(child.name, pattern):
                found[child.resolve()] = child
    return sorted(found.values(), key=str)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_keyed_csv(path, key, *, allow_empty=True):
    frame = pd.read_csv(path, dtype={key: str, "cellvitpp_id": str, "stardist_id": str, "hovernet_id": str}, keep_default_na=False)
    if key not in frame:
        raise ValueError(f"{path}: missing key {key}")
    frame[key] = frame[key].astype(str).str.strip()
    if frame[key].eq("").any() or frame[key].duplicated().any():
        raise ValueError(f"{path}: empty or duplicate {key}")
    if not allow_empty and frame.empty:
        raise ValueError(f"{path}: no observations")
    return frame


def calibration(shift, resolution_json=None):
    data = json.loads(Path(shift).read_text()) if isinstance(shift, (str, Path)) else shift
    mpp = float(data.get("source_mpp", data.get("microns_per_pixel", 0)))
    if resolution_json:
        report = json.loads(Path(resolution_json).read_text())
        if report.get("status") != "pass":
            raise ValueError("The supplied physical-resolution report did not pass")
        mx, my = float(report["mpp_x"]), float(report["mpp_y"])
        if not np.isclose(mx, my, rtol=1e-4):
            raise ValueError("Anisotropic physical pixels are not supported for canonical cell profiles")
        mpp = mx
    if not np.isfinite(mpp) or not 0.01 <= mpp <= 10:
        raise ValueError("Cell profiles require verified source_mpp in [0.01, 10] um/px")
    offset = data.get("offset_crop_to_original")
    if offset is None:
        box = data.get("crop_bbox_xyxy", {})
        if not isinstance(box, dict) or not {"x0", "y0"} <= box.keys():
            raise ValueError("shift.json must explicitly specify the crop-to-original offset")
        offset = {"dx": box["x0"], "dy": box["y0"]}
    origin = np.array([float(offset["dx"]), float(offset["dy"])])
    if not np.isfinite(origin).all():
        raise ValueError("Nonfinite crop origin")
    size = data.get("crop_size", {})
    width, height = int(size.get("width", 0)), int(size.get("height", 0))
    if min(width, height) <= 0:
        raise ValueError("shift.json must provide positive crop_size width and height")
    return {"mpp": mpp, "origin_px": origin, "width": width, "height": height,
            "calibration_source": "passed_resolution_report" if resolution_json else "shift_metadata"}


class RasterReader:
    """Level-0 TIFF windows in YX or YXC order, without full-slide decoding."""
    def __init__(self, path):
        from profile_cell_morphology import WindowReader
        self.path = Path(path)
        self.reader = WindowReader(path)
        self.height, self.width = self.reader.shape[:2]
        self.dtype = self.reader.dtype

    def window(self, x0, y0, x1, y1):
        if not (0 <= x0 <= x1 <= self.width and 0 <= y0 <= y1 <= self.height):
            raise ValueError("Raster window out of bounds")
        return self.reader.read(x0, y0, x1, y1)

    def sample(self, x, y, tile_size=256):
        x, y = np.asarray(x, dtype=int), np.asarray(y, dtype=int)
        valid = (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        if not valid.all():
            raise ValueError("Observation coordinates are outside the raster")
        out = np.empty(len(x), dtype=self.dtype)
        tiles = (y // tile_size) * ((self.width + tile_size - 1) // tile_size) + x // tile_size
        order = np.argsort(tiles, kind="stable")
        cuts = np.r_[0, np.flatnonzero(np.diff(tiles[order])) + 1, len(order)]
        for begin, end in zip(cuts[:-1], cuts[1:]):
            indices = order[begin:end]
            if len(indices) == 0:
                continue
            x0, y0 = int(x[indices[0]] // tile_size * tile_size), int(y[indices[0]] // tile_size * tile_size)
            block = self.window(x0, y0, min(self.width, x0 + tile_size), min(self.height, y0 + tile_size))
            if block.ndim != 2:
                raise ValueError("Categorical mask must have one channel")
            out[indices] = block[y[indices] - y0, x[indices] - x0]
        return out

    def close(self):
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
