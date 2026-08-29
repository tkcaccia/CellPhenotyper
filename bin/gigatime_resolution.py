#!/usr/bin/env python3
"""Dependency-free physical-resolution selection for GigaTIME outputs."""

from __future__ import annotations

import math


def estimate_prediction_gib(
    height: int,
    width: int,
    num_channels: int = 23,
    *,
    bytes_per_sample: int = 4,
) -> float:
    bytes_total = int(num_channels) * int(height) * int(width) * int(bytes_per_sample)
    return float(bytes_total) / float(1024 ** 3)


def choose_downsample_factor(
    orig_h: int,
    orig_w: int,
    source_mpp: float | None,
    target_mpp: float,
    auto_threshold_mpix: float,
    max_side: int,
    max_output_gib: float,
    *,
    num_channels: int,
    bytes_per_sample: int,
    strict_target_mpp: bool,
    enable_max_side_fallback: bool = True,
) -> tuple[float, dict]:
    factor = 1.0
    reason = "native"

    if source_mpp and source_mpp > 0 and target_mpp > 0:
        factor = float(target_mpp) / float(source_mpp)
        reason = "target_mpp"
    else:
        if strict_target_mpp and target_mpp > 0:
            raise ValueError(
                "GigaTIME --strict-target-mpp was requested, but source MPP could not be resolved. "
                "Provide TIFF physical pixel-size metadata or stage/pass the StarDist shift.json so the "
                "original calibrated image can be found."
            )
        mpix = (orig_h * orig_w) / 1_000_000.0
        if enable_max_side_fallback and (mpix > float(auto_threshold_mpix) or max(orig_h, orig_w) > int(max_side)):
            factor = float(max(1, int(math.ceil(max(orig_h, orig_w) / float(max_side)))))
            reason = "max_side_fallback"

    if not (strict_target_mpp and source_mpp and source_mpp > 0 and target_mpp > 0):
        while True:
            height = int(math.ceil(orig_h / float(factor)))
            width = int(math.ceil(orig_w / float(factor)))
            estimated_gib = estimate_prediction_gib(
                height,
                width,
                num_channels=num_channels,
                bytes_per_sample=bytes_per_sample,
            )
            if estimated_gib <= float(max_output_gib):
                break
            factor = max(factor + 1.0, factor * math.sqrt(estimated_gib / float(max_output_gib)))
            reason = "disk_budget"

    final_h = int(math.ceil(orig_h / float(factor)))
    final_w = int(math.ceil(orig_w / float(factor)))
    metadata = {
        "selected_factor": float(factor),
        "selected_shape_yx": [final_h, final_w],
        "estimated_prediction_gib": float(
            estimate_prediction_gib(
                final_h,
                final_w,
                num_channels=num_channels,
                bytes_per_sample=bytes_per_sample,
            )
        ),
        "selection_reason": reason,
    }
    if source_mpp and source_mpp > 0:
        metadata["source_mpp"] = float(source_mpp)
        metadata["effective_mpp"] = float(source_mpp * factor)
        if target_mpp > 0:
            metadata["requested_downsample_factor"] = float(target_mpp) / float(source_mpp)
            metadata["mpp_error_fraction"] = float((source_mpp * factor - target_mpp) / target_mpp)
    return factor, metadata
