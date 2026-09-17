#!/usr/bin/env python3
"""Sweep image-scale-aware multiclass watershed refinements for K=3.

Candidate generation receives only the H&E image and pipeline labels. The
expert GeoJSON is used exclusively after generation to score candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from benchmark_k3_ground_truth import (
    internal_boundary,
    prepare_boundary_reference,
    rasterize_geojson,
    score_candidate,
)


def read_vips_downsample(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Read a tiled WSI into a bounded RGB analysis frame with libvips."""
    import pyvips

    height, width = shape
    image = pyvips.Image.new_from_file(str(path), access="sequential")
    if image.bands > 3:
        image = image[:3]
    if image.bands == 1:
        image = image.bandjoin([image, image])
    scale = min(width / image.width, height / image.height)
    # libvips resize has no "area" enum; lanczos3 is its documented
    # anti-aliased reduction kernel and avoids nearest-neighbour stair steps.
    image = image.resize(scale, kernel="lanczos3")
    if image.width != width or image.height != height:
        image = image.resize(width / image.width, vscale=height / image.height, kernel="lanczos3")
    memory = image.cast("uchar").write_to_memory()
    return np.frombuffer(memory, dtype=np.uint8).reshape(image.height, image.width, image.bands).copy()


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    values = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    xyz = values @ np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32,
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        (116 * transformed[..., 1] - 16,
         500 * (transformed[..., 0] - transformed[..., 1]),
         200 * (transformed[..., 1] - transformed[..., 2])), axis=-1,
    ).astype(np.float32)


def feature_stack(image: np.ndarray, space: str) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    lab = rgb_to_lab(rgb)
    od = -np.log((rgb.astype(np.float32) + 1.0) / 256.0)
    if space == "luminance":
        return lab[..., :1]
    if space == "lab":
        return lab
    if space == "od":
        return od
    if space == "lab_od":
        return np.concatenate((lab, od), axis=-1)
    raise ValueError(f"Unknown feature space: {space}")


def robust_multichannel_gradient(
    image: np.ndarray,
    tissue: np.ndarray,
    *,
    space: str,
    sigma: float,
) -> np.ndarray:
    features = feature_stack(image, space)
    selected = features[tissue]
    center = np.median(selected, axis=0)
    q25, q75 = np.percentile(selected, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 0.1 * selected.std(axis=0))
    scale = np.where(scale > 1e-6, scale, 1.0)
    normalized = np.clip((features - center) / scale, -10.0, 10.0)
    if sigma > 0:
        normalized = np.stack(
            [ndi.gaussian_filter(normalized[..., index], sigma=float(sigma))
             for index in range(normalized.shape[-1])], axis=-1,
        )
    gradient = np.zeros(tissue.shape, dtype=np.float32)
    for index in range(normalized.shape[-1]):
        channel = normalized[..., index]
        gradient += ndi.sobel(channel, axis=0) ** 2 + ndi.sobel(channel, axis=1) ** 2
    gradient = np.sqrt(gradient / normalized.shape[-1])
    positive = gradient[tissue]
    low, high = np.percentile(positive, [1, 99])
    return np.clip((gradient - low) / max(1e-6, high - low), 0.0, 1.0).astype(np.float32)


def _ensure_component_markers(source: np.ndarray, markers: np.ndarray) -> np.ndarray:
    result = markers.copy()
    for label in (int(value) for value in np.unique(source) if value > 0):
        components, count = ndi.label(source == label, structure=np.ones((3, 3), dtype=np.uint8))
        for component in range(1, count + 1):
            region = components == component
            if np.any(result[region] == label):
                continue
            distance = ndi.distance_transform_edt(region)
            y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
            result[y, x] = label
    return result


def image_watershed_refine(
    labels: np.ndarray,
    gradient: np.ndarray,
    *,
    boundary_radius: int,
    compactness: float,
) -> np.ndarray:
    source = np.asarray(labels)
    tissue = source > 0
    boundary = internal_boundary(source)
    editable = ndi.binary_dilation(boundary, iterations=max(1, int(boundary_radius))) & tissue
    markers = source.astype(np.int32, copy=True)
    markers[editable] = 0
    markers = _ensure_component_markers(source, markers)
    result = watershed(
        np.asarray(gradient, dtype=np.float32), markers=markers, mask=tissue,
        connectivity=np.ones((3, 3), dtype=np.uint8), compactness=float(compactness),
    )
    output = source.copy()
    valid = editable & (result > 0)
    output[valid] = result[valid].astype(output.dtype, copy=False)
    output[~tissue] = 0
    return output


def save_overlay(path: Path, image: np.ndarray, labels: np.ndarray) -> None:
    colors = np.zeros((*labels.shape, 3), dtype=np.uint8)
    palette = ((34, 197, 94), (239, 68, 68), (59, 130, 246), (234, 179, 8))
    for index, label in enumerate(int(value) for value in np.unique(labels) if value > 0):
        colors[labels == label] = palette[index % len(palette)]
    overlay = image.copy()
    tissue = labels > 0
    overlay[tissue] = (0.55 * image[tissue] + 0.45 * colors[tissue]).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument(
        "--spaces", nargs="+", choices=("luminance", "lab", "od", "lab_od"),
        default=("luminance", "lab", "od", "lab_od"),
    )
    parser.add_argument(
        "--sigma-native", nargs="+", type=int,
        default=(0, 8, 16, 24, 32, 48, 64),
    )
    parser.add_argument(
        "--radius-native", nargs="+", type=int,
        default=(32, 64, 96),
    )
    parser.add_argument(
        "--compactness", nargs="+", type=float,
        default=(0.0, 0.0001, 0.001, 0.01),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    reference, reference_meta = rasterize_geojson(args.reference, downsample=args.downsample)
    baseline, _ = rasterize_geojson(
        args.candidate, full_shape=tuple(reference_meta["full_shape"]), downsample=args.downsample,
    )
    tissue = baseline > 0
    image = read_vips_downsample(args.image, baseline.shape)
    if image.shape[:2] != baseline.shape:
        raise RuntimeError(f"Image/label analysis shapes differ: {image.shape[:2]} vs {baseline.shape}")

    candidates: dict[str, np.ndarray] = {"baseline": baseline}
    gradients = {}
    if any(value < 0 for value in (*args.sigma_native, *args.radius_native, *args.compactness)):
        parser.error("Sigma, radius, and compactness values must be nonnegative")
    for space in args.spaces:
        for sigma_native in args.sigma_native:
            sigma = sigma_native / args.downsample
            gradients[(space, sigma_native)] = robust_multichannel_gradient(
                image, tissue, space=space, sigma=sigma,
            )
    for radius_native in args.radius_native:
        radius = max(1, int(round(radius_native / args.downsample)))
        for (space, sigma_native), gradient in gradients.items():
            for compactness in args.compactness:
                suffix = str(compactness).replace(".", "p")
                name = f"watershed_{space}_r{radius_native}_s{sigma_native}_c{suffix}"
                candidates[name] = image_watershed_refine(
                    baseline, gradient, boundary_radius=radius, compactness=compactness,
                )

    tolerances = [32, 64, 128]
    boundary_reference_cache = prepare_boundary_reference(
        reference,
        downsample=args.downsample,
        tolerances_native_px=tolerances,
    )
    rows, full = [], []
    for name, candidate in candidates.items():
        score = score_candidate(
            candidate, reference, downsample=args.downsample,
            tolerances_native_px=tolerances,
            boundary_reference_cache=boundary_reference_cache,
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
    rank = {row["method"]: index for index, row in enumerate(rows)}
    full.sort(key=lambda item: rank[item["method"]])
    with (args.outdir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.outdir / "summary.json").write_text(json.dumps({
        "schema_version": "cellphenotyper.k3_image_watershed_sweep.v1",
        "development_reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "image": str(args.image.resolve()),
        "downsample": args.downsample,
        "sweep": {
            "spaces": list(args.spaces),
            "sigma_native_px": list(args.sigma_native),
            "radius_native_px": list(args.radius_native),
            "compactness": list(args.compactness),
        },
        "ground_truth_usage": "scoring_only; no candidate generator receives reference labels",
        "independence_limit": "single annotated slide used for parameter selection; external validation required",
        "ranking": full,
    }, indent=2) + "\n", encoding="utf-8")
    for row in rows[:8]:
        save_overlay(args.outdir / f"{row['method']}.jpg", image, candidates[row["method"]])
    print(json.dumps(rows[:16], indent=2))


if __name__ == "__main__":
    main()
