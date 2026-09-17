#!/usr/bin/env python3
"""Development-set sweep for label-only multiclass boundary regularization.

Candidate generation sees only the pipeline label raster. The expert annotation
is passed solely to the scoring functions imported from
``benchmark_k3_ground_truth``. This is a single-slide development experiment,
not an independent accuracy estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from benchmark_k3_ground_truth import internal_boundary, rasterize_geojson, score_candidate


def boundary_band(labels: np.ndarray, radius: int) -> np.ndarray:
    boundary = internal_boundary(labels)
    if radius <= 0:
        return boundary
    return ndi.binary_dilation(boundary, iterations=int(radius)) & (labels > 0)


def gaussian_label_diffusion(
    labels: np.ndarray,
    *,
    sigma: float,
    boundary_radius: int,
    iterations: int = 1,
) -> np.ndarray:
    """Round internal corners by normalized multiclass indicator diffusion."""
    source = np.asarray(labels)
    tissue = source > 0
    active = [int(value) for value in np.unique(source[tissue])]
    editable = boundary_band(source, boundary_radius)
    result = source.copy()
    for _ in range(max(1, int(iterations))):
        denominator = ndi.gaussian_filter(tissue.astype(np.float32), sigma=float(sigma))
        scores = []
        for label in active:
            numerator = ndi.gaussian_filter((result == label).astype(np.float32), sigma=float(sigma))
            scores.append(numerator / np.maximum(denominator, 1e-6))
        proposed = np.asarray(active, dtype=result.dtype)[np.argmax(np.stack(scores), axis=0)]
        result[editable] = proposed[editable]
        result[~tissue] = 0
    return result


def neighbour_majority(
    labels: np.ndarray,
    *,
    boundary_radius: int,
    iterations: int,
    minimum_fraction: float,
) -> np.ndarray:
    """Apply permutation-invariant 8-neighbour majority updates in a fixed band."""
    source = np.asarray(labels)
    tissue = source > 0
    active = [int(value) for value in np.unique(source[tissue])]
    editable = boundary_band(source, boundary_radius)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    result = source.copy()
    for _ in range(max(1, int(iterations))):
        counts = np.stack([
            ndi.convolve((result == label).astype(np.uint8), kernel, mode="constant", cval=0)
            for label in active
        ])
        winner_index = np.argmax(counts, axis=0)
        winner_count = np.max(counts, axis=0)
        proposed = np.asarray(active, dtype=result.dtype)[winner_index]
        change = editable & tissue & (proposed != result) & (winner_count >= int(np.ceil(8 * minimum_fraction)))
        result[change] = proposed[change]
    result[~tissue] = 0
    return result


def save_labels(path: Path, labels: np.ndarray) -> None:
    colors = np.zeros((*labels.shape, 3), dtype=np.uint8)
    palette = ((34, 197, 94), (239, 68, 68), (59, 130, 246), (234, 179, 8))
    for index, label in enumerate(int(value) for value in np.unique(labels) if value > 0):
        colors[labels == label] = palette[index % len(palette)]
    Image.fromarray(colors, mode="RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    reference, reference_meta = rasterize_geojson(args.reference, downsample=args.downsample)
    baseline, _ = rasterize_geojson(
        args.candidate, full_shape=tuple(reference_meta["full_shape"]), downsample=args.downsample,
    )
    candidates: dict[str, np.ndarray] = {"baseline": baseline}
    for radius_native in (32, 64, 96, 128):
        radius = max(1, int(round(radius_native / args.downsample)))
        for sigma_native in (8, 16, 24, 32, 48, 64):
            sigma = sigma_native / args.downsample
            name = f"gaussian_r{radius_native}_s{sigma_native}"
            candidates[name] = gaussian_label_diffusion(
                baseline, sigma=sigma, boundary_radius=radius, iterations=1,
            )
    for radius_native in (32, 64, 96):
        radius = max(1, int(round(radius_native / args.downsample)))
        for iterations in (1, 2, 4, 8):
            for fraction in (0.625, 0.75):
                name = f"majority_r{radius_native}_i{iterations}_f{fraction:g}"
                candidates[name] = neighbour_majority(
                    baseline, boundary_radius=radius, iterations=iterations,
                    minimum_fraction=fraction,
                )

    rows = []
    full = []
    for name, candidate in candidates.items():
        score = score_candidate(
            candidate, reference, downsample=args.downsample,
            tolerances_native_px=[32, 64, 128],
        )
        changed = int(np.count_nonzero(candidate != baseline))
        row = {
            "method": name,
            "changed_analysis_pixels": changed,
            "changed_reference_tissue_fraction": changed / max(1, int(np.count_nonzero(reference > 0))),
            "mean_class_iou": score["mean_class_iou"],
            "mean_class_dice": score["mean_class_dice"],
            "categorical_accuracy_on_reference_tissue": score["categorical_accuracy_on_reference_tissue"],
            "mean_boundary_distance_native_px": score["mean_symmetric_boundary_distance_native_px"],
            "p95_boundary_distance_native_px": score["p95_symmetric_boundary_distance_native_px"],
            "boundary_length_ratio": score["boundary_length_ratio"],
            "boundary_f1_32px": score["boundary_f1"][0]["f1"],
            "boundary_f1_64px": score["boundary_f1"][1]["f1"],
        }
        rows.append(row)
        full.append({"method": name, "metrics": score, **{key: row[key] for key in row if key.startswith("changed_")}})
    rows.sort(key=lambda row: (row["mean_class_iou"], row["boundary_f1_32px"]), reverse=True)
    full.sort(key=lambda item: next(index for index, row in enumerate(rows) if row["method"] == item["method"]))
    with (args.outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.outdir / "summary.json").write_text(json.dumps({
        "schema_version": "cellphenotyper.k3_boundary_smoothing_sweep.v1",
        "development_reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "downsample": args.downsample,
        "ground_truth_usage": "scoring_only; no candidate generator receives reference labels",
        "independence_limit": "single annotated slide used for parameter selection; external validation required",
        "ranking": full,
    }, indent=2) + "\n", encoding="utf-8")
    for row in rows[:6]:
        save_labels(args.outdir / f"{row['method']}.png", candidates[row["method"]])
    print(json.dumps(rows[:12], indent=2))


if __name__ == "__main__":
    main()
