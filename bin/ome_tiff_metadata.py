#!/usr/bin/env python3
"""Physical-resolution helpers for pipeline OME-TIFF outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


MICRONS_PER_CENTIMETER = 10_000.0


def _positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def read_mpp_json(path: str | Path | None, fallback: float = 0.0) -> tuple[float, float]:
    """Read authoritative X/Y microns-per-pixel from a pipeline sidecar."""
    payload: dict[str, Any] = {}
    if path:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Resolution sidecar not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))

    scalar = next(
        (
            value
            for key in ("source_mpp", "microns_per_pixel", "mpp", "effective_mpp")
            if (value := _positive(payload.get(key))) is not None
        ),
        _positive(fallback),
    )
    mpp_x = next(
        (
            value
            for key in ("source_mpp_x", "microns_per_pixel_x", "mpp_x")
            if (value := _positive(payload.get(key))) is not None
        ),
        scalar,
    )
    mpp_y = next(
        (
            value
            for key in ("source_mpp_y", "microns_per_pixel_y", "mpp_y")
            if (value := _positive(payload.get(key))) is not None
        ),
        scalar,
    )
    if mpp_x is None or mpp_y is None:
        raise RuntimeError(
            "Physical pixel size is required for OME-TIFF output. "
            "Provide a shift/resolution JSON with source_mpp or a positive fallback."
        )
    return float(mpp_x), float(mpp_y)


def tiff_resolution_kwargs(mpp_x: float, mpp_y: float, axes: str) -> dict[str, Any]:
    """Return tifffile arguments encoding MPP as pixels per centimeter."""
    if _positive(mpp_x) is None or _positive(mpp_y) is None:
        raise ValueError(f"MPP must be positive, got x={mpp_x}, y={mpp_y}")
    return {
        "resolution": (
            MICRONS_PER_CENTIMETER / float(mpp_x),
            MICRONS_PER_CENTIMETER / float(mpp_y),
        ),
        "resolutionunit": "CENTIMETER",
        "metadata": {"axes": axes},
    }


def label_storage_dtype(max_label: int):
    """Choose a byte-order-safe integer dtype for Bio-Formats label staging."""
    import numpy as np

    value = int(max_label)
    if value < 0:
        raise ValueError(f"Label values must be non-negative, got {value}")
    if value <= np.iinfo(np.uint8).max:
        return np.dtype("u1")
    if value <= np.iinfo(np.uint16).max:
        return np.dtype(">u2")
    if value <= np.iinfo(np.uint32).max:
        return np.dtype(">u4")
    raise ValueError(f"Label value exceeds uint32: {value}")


def create_tiff_memmap(
    path: str | Path,
    *,
    shape: tuple[int, int],
    dtype,
    mpp_x: float,
    mpp_y: float,
):
    """Allocate a contiguous TIFF and map its pixels, including big-endian labels."""
    import numpy as np
    import tifffile

    target = Path(path)
    pixel_dtype = np.dtype(dtype)
    byteorder = pixel_dtype.byteorder if pixel_dtype.itemsize > 1 else None
    tifffile.imwrite(
        target,
        shape=tuple(map(int, shape)),
        dtype=pixel_dtype,
        bigtiff=True,
        byteorder=byteorder,
        contiguous=True,
        **tiff_resolution_kwargs(mpp_x, mpp_y, "YX"),
    )
    with tifffile.TiffFile(target) as tif:
        page = tif.pages[0]
        if not page.is_memmappable or len(page.dataoffsets) != 1:
            raise RuntimeError(f"Allocated TIFF is not contiguous and memory-mappable: {target}")
        offset = int(page.dataoffsets[0])
        byte_count = int(page.databytecounts[0])
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * pixel_dtype.itemsize
    if byte_count < expected_bytes:
        raise RuntimeError(
            f"Allocated TIFF pixel block is too small: {byte_count} < {expected_bytes} bytes"
        )
    return np.memmap(
        target,
        dtype=pixel_dtype,
        mode="r+",
        offset=offset,
        shape=tuple(map(int, shape)),
    )


def _spatial_shape(shape: tuple[int, ...], axes: str) -> tuple[int, int]:
    if "Y" not in axes or "X" not in axes:
        raise RuntimeError(f"OME series has no Y/X axes: axes={axes!r}, shape={shape}")
    return int(shape[axes.index("Y")]), int(shape[axes.index("X")])


def validate_ome_tiff(
    path: str | Path,
    *,
    expected_shape: tuple[int, int],
    expected_mpp: tuple[float, float],
    require_pyramid: bool = True,
) -> dict[str, Any]:
    """Fail if an output is not a readable, resolution-correct OME-TIFF."""
    import tifffile

    output = Path(path)
    with tifffile.TiffFile(output) as tif:
        if not tif.is_ome or not tif.ome_metadata:
            raise RuntimeError(f"Output is not a valid OME-TIFF: {output}")
        series = tif.series[0]
        observed_shape = _spatial_shape(tuple(series.shape), series.axes)
        if tuple(map(int, expected_shape)) != observed_shape:
            raise RuntimeError(
                f"OME spatial shape mismatch for {output}: "
                f"observed={observed_shape}, expected={tuple(expected_shape)}"
            )
        levels = [tuple(level.shape) for level in series.levels]
        if require_pyramid and max(observed_shape) > 512 and len(levels) < 2:
            raise RuntimeError(f"OME pyramid is missing for WSI-scale output: {output}")

        root = ET.fromstring(tif.ome_metadata)
        pixels = next((node for node in root.iter() if node.tag.endswith("Pixels")), None)
        if pixels is None:
            raise RuntimeError(f"OME Pixels metadata is missing: {output}")
        observed_mpp = (
            _positive(pixels.attrib.get("PhysicalSizeX")),
            _positive(pixels.attrib.get("PhysicalSizeY")),
        )
        if observed_mpp[0] is None or observed_mpp[1] is None:
            raise RuntimeError(f"OME physical pixel size is missing: {output}")
        for axis, observed, expected in zip("XY", observed_mpp, expected_mpp):
            if not math.isclose(float(observed), float(expected), rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError(
                    f"OME PhysicalSize{axis} mismatch for {output}: "
                    f"observed={observed}, expected={expected}"
                )
        return {
            "path": str(output),
            "axes": series.axes,
            "shape": list(series.shape),
            "spatial_shape": list(observed_shape),
            "pyramid_levels": [list(level) for level in levels],
            "physical_size_x_um": float(observed_mpp[0]),
            "physical_size_y_um": float(observed_mpp[1]),
        }
