"""Memory-bounded preview reads for tiled or compressed TIFF images."""

from __future__ import annotations

import numpy as np
import tifffile


def _resolve_zarr_array(root):
    if hasattr(root, "shape"):
        return root
    keys = list(root.array_keys()) if hasattr(root, "array_keys") else []
    if not keys and hasattr(root, "group_keys"):
        for key in root.group_keys():
            try:
                return _resolve_zarr_array(root[key])
            except ValueError:
                continue
    if keys:
        key = sorted(
            keys,
            key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)),
        )[0]
        return root[key]
    raise ValueError("TIFF Zarr store does not contain an image array")


def read_tiled_tiff_preview(path: str, factor: int) -> np.ndarray:
    """Read a strided preview without materializing a compressed tiled TIFF."""
    import zarr

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = getattr(series, "levels", None) or [series]
        level0 = levels[0]
        axes = str(getattr(level0, "axes", None) or series.axes)
        store = level0.aszarr()
        try:
            array = _resolve_zarr_array(zarr.open(store, mode="r"))
            slicer = []
            retained_axes = []
            for axis in axes:
                if axis in ("Y", "X"):
                    slicer.append(slice(None, None, max(1, int(factor))))
                    retained_axes.append(axis)
                elif axis in ("C", "S"):
                    slicer.append(slice(None))
                    retained_axes.append("C")
                else:
                    slicer.append(0)
            block = np.asarray(array[tuple(slicer)])
        finally:
            if hasattr(store, "close"):
                store.close()

    if "Y" not in retained_axes or "X" not in retained_axes:
        raise ValueError(f"Unsupported TIFF axes for preview: {axes}")
    order = [retained_axes.index("Y"), retained_axes.index("X")]
    if "C" in retained_axes:
        order.append(retained_axes.index("C"))
    return np.transpose(block, axes=order)
