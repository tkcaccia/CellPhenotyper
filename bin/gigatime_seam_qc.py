#!/usr/bin/env python3
"""Measure and visualize block-boundary discontinuities in GigaTIME outputs."""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw
import tifffile

try:
    import zarr
except ImportError:  # Unit tests can still exercise in-memory metrics.
    zarr = None


def _probability_scale(dtype: np.dtype) -> float:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


@contextlib.contextmanager
def open_prediction_array(path: str | Path) -> Iterator[object]:
    if zarr is None:
        raise RuntimeError("zarr is required to inspect persisted GigaTIME outputs")
    path = Path(path)
    if path.is_dir():
        root = zarr.open(str(path), mode="r")
        yield root["0"] if hasattr(root, "keys") and "0" in root else root
        return

    tif = tifffile.TiffFile(path)
    store = None
    try:
        series = tif.series[0]
        level = series.levels[0] if getattr(series, "levels", None) else series
        store = level.aszarr()
        yield zarr.open(store, mode="r")
    finally:
        if store is not None and hasattr(store, "close"):
            store.close()
        tif.close()


def _as_cyx_shape(prediction: object) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in prediction.shape)
    if len(shape) == 2:
        return 1, shape[0], shape[1]
    if len(shape) != 3:
        raise ValueError(f"Expected a CYX prediction array; observed shape={shape}")
    return shape


def _read_boundary_strip(
    prediction: object,
    *,
    orientation: str,
    boundary: int,
    radius: int,
    sample_step: int,
    height: int,
    width: int,
    scale: float,
) -> tuple[np.ndarray, int]:
    if orientation == "horizontal":
        start = max(0, boundary - radius)
        end = min(height, boundary + radius + 1)
        strip = np.asarray(prediction[:, start:end, ::sample_step], dtype=np.float32)
        strip = np.transpose(strip, (0, 2, 1))
    else:
        start = max(0, boundary - radius)
        end = min(width, boundary + radius + 1)
        strip = np.asarray(prediction[:, ::sample_step, start:end], dtype=np.float32)
    return strip / max(scale, 1.0), boundary - start


def compute_seam_metrics(
    prediction: object,
    *,
    block_size: int,
    channel_names: list[str],
    max_p95_excess: float = 0.10,
    min_affected_fraction: float = 0.05,
    max_samples_per_boundary: int = 4096,
    neighborhood_radius: int = 4,
) -> dict:
    channels, height, width = _as_cyx_shape(prediction)
    if channels != len(channel_names):
        raise ValueError(
            f"Prediction has {channels} channels but {len(channel_names)} names were supplied"
        )
    if block_size < 2:
        raise ValueError("block_size must be at least 2")

    scale = _probability_scale(np.dtype(prediction.dtype))
    records: list[dict] = []
    for orientation, axis_length, sampled_span in (
        ("horizontal", height, width),
        ("vertical", width, height),
    ):
        sample_step = max(1, int(math.ceil(sampled_span / max(1, max_samples_per_boundary))))
        for boundary in range(block_size, axis_length, block_size):
            strip, center = _read_boundary_strip(
                prediction,
                orientation=orientation,
                boundary=boundary,
                radius=max(2, int(neighborhood_radius)),
                sample_step=sample_step,
                height=height,
                width=width,
                scale=scale,
            )
            if center <= 0 or center >= strip.shape[2]:
                continue
            boundary_diff = np.abs(strip[:, :, center] - strip[:, :, center - 1])
            neighboring_diff = np.abs(np.diff(strip, axis=2))
            baseline_parts = []
            if center - 1 > 0:
                baseline_parts.append(neighboring_diff[:, :, : center - 1])
            if center + 1 < neighboring_diff.shape[2]:
                baseline_parts.append(neighboring_diff[:, :, center + 1 :])
            if not baseline_parts:
                continue
            baseline = np.concatenate(baseline_parts, axis=2)
            left = strip[:, :, center - 1]
            right = strip[:, :, center]

            for channel_index, channel_name in enumerate(channel_names):
                observed = boundary_diff[channel_index]
                reference = baseline[channel_index].reshape(-1)
                boundary_p95 = float(np.quantile(observed, 0.95))
                baseline_p95 = float(np.quantile(reference, 0.95))
                p95_excess = max(0.0, boundary_p95 - baseline_p95)
                affected_fraction = float(
                    np.mean(observed > (baseline_p95 + float(max_p95_excess)))
                )
                zero_transition_fraction = float(
                    np.mean((left[channel_index] <= 0.0) != (right[channel_index] <= 0.0))
                )
                failed = bool(
                    p95_excess > float(max_p95_excess)
                    and affected_fraction >= float(min_affected_fraction)
                )
                records.append(
                    {
                        "orientation": orientation,
                        "boundary_px": int(boundary),
                        "channel_index": int(channel_index),
                        "channel": str(channel_name),
                        "sample_step": int(sample_step),
                        "sample_count": int(observed.size),
                        "boundary_mean_abs_diff": float(observed.mean()),
                        "baseline_mean_abs_diff": float(reference.mean()),
                        "boundary_p95_abs_diff": boundary_p95,
                        "baseline_p95_abs_diff": baseline_p95,
                        "p95_excess": p95_excess,
                        "affected_fraction": affected_fraction,
                        "zero_transition_fraction": zero_transition_fraction,
                        "failed": failed,
                    }
                )

    failed_records = [record for record in records if record["failed"]]
    worst = max(records, key=lambda record: record["p95_excess"], default=None)
    return {
        "schema_version": 1,
        "status": "fail" if failed_records else "pass",
        "interpretation": (
            "Block boundaries are compared with adjacent within-block gradients. "
            "This is an image-continuity QC metric, not a biological accuracy estimate."
        ),
        "prediction_shape_cyx": [channels, height, width],
        "prediction_dtype": str(np.dtype(prediction.dtype)),
        "probability_scale": scale,
        "block_size_px": int(block_size),
        "thresholds": {
            "max_p95_excess": float(max_p95_excess),
            "min_affected_fraction": float(min_affected_fraction),
            "max_samples_per_boundary": int(max_samples_per_boundary),
            "neighborhood_radius": int(neighborhood_radius),
        },
        "boundaries_evaluated": len({(r["orientation"], r["boundary_px"]) for r in records}),
        "channel_boundary_tests": len(records),
        "failed_channel_boundaries": len(failed_records),
        "worst_channel_boundary": worst,
        "records": records,
    }


def write_seam_heatmap(report: dict, path: str | Path) -> None:
    records = report.get("records", [])
    channels = list(dict.fromkeys(record["channel"] for record in records))
    orientations = ("horizontal", "vertical")
    boundaries = {}
    for orientation in orientations:
        all_boundaries = sorted(
            {record["boundary_px"] for record in records if record["orientation"] == orientation}
        )
        if len(all_boundaries) <= 80:
            boundaries[orientation] = all_boundaries
            continue
        worst_by_boundary = {
            boundary: max(
                record["p95_excess"]
                for record in records
                if record["orientation"] == orientation and record["boundary_px"] == boundary
            )
            for boundary in all_boundaries
        }
        worst = sorted(all_boundaries, key=lambda value: worst_by_boundary[value], reverse=True)[:20]
        evenly_spaced = [
            all_boundaries[index]
            for index in np.linspace(0, len(all_boundaries) - 1, 60, dtype=int)
        ]
        boundaries[orientation] = sorted(set(worst + evenly_spaced))
    cell = 24
    label_width = 130
    section_gap = 60
    section_widths = {key: max(cell, len(value) * cell) for key, value in boundaries.items()}
    width = label_width + sum(section_widths.values()) + section_gap + 20
    height = 75 + max(1, len(channels)) * cell + 45
    image = Image.new("RGB", (width, height), "#F6F1E7")
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), f"GigaTIME seam QC: {report.get('status', 'unknown').upper()}", fill="#1E292B")
    draw.text((12, 30), "color = p95 boundary excess; X = failed gate", fill="#46585B")
    threshold = max(1.0e-9, float(report.get("thresholds", {}).get("max_p95_excess", 0.10)))
    record_lookup = {
        (record["orientation"], record["boundary_px"], record["channel"]): record
        for record in records
    }
    for row, channel in enumerate(channels):
        y = 65 + row * cell
        draw.text((8, y + 5), channel, fill="#1E292B")
    x_start = label_width
    for orientation in orientations:
        draw.text((x_start, 50), orientation, fill="#1E292B")
        for col, boundary in enumerate(boundaries[orientation]):
            x = x_start + col * cell
            for row, channel in enumerate(channels):
                y = 65 + row * cell
                record = record_lookup[(orientation, boundary, channel)]
                ratio = min(1.0, float(record["p95_excess"]) / threshold)
                color = (
                    int(235 * ratio + 33 * (1.0 - ratio)),
                    int(70 * ratio + 134 * (1.0 - ratio)),
                    int(55 * ratio + 126 * (1.0 - ratio)),
                )
                draw.rectangle((x, y, x + cell - 2, y + cell - 2), fill=color)
                if record["failed"]:
                    draw.line((x + 4, y + 4, x + cell - 6, y + cell - 6), fill="white", width=2)
                    draw.line((x + cell - 6, y + 4, x + 4, y + cell - 6), fill="white", width=2)
        x_start += section_widths[orientation] + section_gap
    image.save(path)


def assess_prediction_seams(
    prediction_path: str | Path,
    *,
    outdir: str | Path,
    block_size: int,
    channel_names: list[str],
    max_p95_excess: float,
    min_affected_fraction: float,
    mode: str,
) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open_prediction_array(prediction_path) as prediction:
        report = compute_seam_metrics(
            prediction,
            block_size=block_size,
            channel_names=channel_names,
            max_p95_excess=max_p95_excess,
            min_affected_fraction=min_affected_fraction,
        )
    report["prediction_path"] = str(prediction_path)
    report["gate_mode"] = str(mode)
    json_path = outdir / "gigatime_seam_qc.json"
    png_path = outdir / "gigatime_seam_qc.png"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_seam_heatmap(report, png_path)
    print(
        "[INFO] GigaTIME seam QC "
        f"status={report['status']} failed={report['failed_channel_boundaries']} "
        f"tests={report['channel_boundary_tests']} report={json_path}",
        flush=True,
    )
    if report["status"] == "fail" and str(mode).lower() == "fail":
        worst = report.get("worst_channel_boundary") or {}
        raise RuntimeError(
            "GigaTIME seam QC failed: "
            f"orientation={worst.get('orientation')} boundary={worst.get('boundary_px')} "
            f"channel={worst.get('channel')} p95_excess={worst.get('p95_excess')}"
        )
    return report
