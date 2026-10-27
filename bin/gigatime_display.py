"""Display-only tone mapping for virtual DAPI previews.

GigaTIME scores in the lossless OME-TIFF/Zarr remain untouched. The mapping is
computed from the whole preview, without using a nuclear mask or QC outcome.
"""

from __future__ import annotations

import numpy as np


def tone_map_dapi_preview(gray: np.ndarray) -> tuple[np.ndarray, dict]:
    """Darken diffuse midtones while retaining bright predicted nuclei.

    The same image-wide parameters apply at every pixel; local equalization
    would risk making tile seams or border shading look like genuine signal.
    """
    source = np.asarray(gray)
    if source.ndim != 2 or source.dtype != np.uint8:
        raise ValueError("DAPI preview must be a two-dimensional uint8 array")
    low, high = (float(v) for v in np.percentile(source, (2.0, 99.0)))
    info = {
        "kind": "display_only_global_percentile_gamma",
        "lower_percentile": 2.0,
        "upper_percentile": 99.0,
        "lower_value_uint8": low,
        "upper_value_uint8": high,
        "gamma": 2.0,
        "applied": high - low >= 8.0,
        "source": "8-bit linear preview of native GigaTIME scores",
        "native_scores_and_qc_unchanged": True,
    }
    if not info["applied"]:
        return source.copy(), info
    scaled = np.clip((source.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    return np.rint(255.0 * scaled**2.0).astype(np.uint8), info
