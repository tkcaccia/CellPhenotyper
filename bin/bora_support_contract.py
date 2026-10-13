"""Lightweight support checks shared by Bora integration and unit tests."""

from __future__ import annotations

import numpy as np


def verify_support(
    final_reader,
    baseline_reader,
    tissue_reader,
    uncertainty_reader,
    block_rows: int,
    hard_outlier_code: int = 5,
) -> dict[str, int]:
    """Verify that Bora did not expand the authoritative upstream footprint.

    The scaled GrandQC tissue raster is intentionally diagnostic: the upstream
    cluster footprint may contain GrandQC artifact candidates that policy keeps
    for analysis.  Only newly added foreground and KODAMA-confirmed artifact
    outliers are hard failures.
    """

    height, width = final_reader.spatial_shape()
    counts = {
        "foreground_pixels": 0,
        "baseline_foreground_pixels": 0,
        "new_foreground_outside_baseline_pixels": 0,
        "dropped_baseline_foreground_pixels": 0,
        "baseline_outside_scaled_tissue_pixels": 0,
        "final_outside_scaled_tissue_pixels": 0,
        "kodama_confirmed_artifact_outlier_pixels": 0,
        "foreground_on_confirmed_artifact_outlier_pixels": 0,
    }
    for y0 in range(0, height, max(1, block_rows)):
        y1 = min(height, y0 + max(1, block_rows))
        labels = np.squeeze(final_reader.read(y0, y1, 0, width))
        baseline = np.squeeze(baseline_reader.read(y0, y1, 0, width))
        tissue = tissue_reader.read(y0, y1, 0, width)
        uncertainty = np.squeeze(uncertainty_reader.read(y0, y1, 0, width))
        foreground = labels > 0
        baseline_foreground = baseline > 0
        hard_outlier = uncertainty == hard_outlier_code
        counts["foreground_pixels"] += int(np.count_nonzero(foreground))
        counts["baseline_foreground_pixels"] += int(
            np.count_nonzero(baseline_foreground)
        )
        counts["new_foreground_outside_baseline_pixels"] += int(
            np.count_nonzero(foreground & ~baseline_foreground)
        )
        counts["dropped_baseline_foreground_pixels"] += int(
            np.count_nonzero(baseline_foreground & ~foreground)
        )
        counts["baseline_outside_scaled_tissue_pixels"] += int(
            np.count_nonzero(baseline_foreground & ~tissue)
        )
        counts["final_outside_scaled_tissue_pixels"] += int(
            np.count_nonzero(foreground & ~tissue)
        )
        counts["kodama_confirmed_artifact_outlier_pixels"] += int(
            np.count_nonzero(hard_outlier)
        )
        counts["foreground_on_confirmed_artifact_outlier_pixels"] += int(
            np.count_nonzero(foreground & hard_outlier)
        )
    if counts["new_foreground_outside_baseline_pixels"]:
        raise RuntimeError(
            "Bora expanded foreground outside the upstream cluster-label footprint: "
            f"{counts['new_foreground_outside_baseline_pixels']} pixels"
        )
    if counts["foreground_on_confirmed_artifact_outlier_pixels"]:
        raise RuntimeError(
            "Bora produced labels on KODAMA-confirmed artifact outliers: "
            f"{counts['foreground_on_confirmed_artifact_outlier_pixels']} pixels"
        )
    return counts
