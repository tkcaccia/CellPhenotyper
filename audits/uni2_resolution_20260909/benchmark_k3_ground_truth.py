#!/usr/bin/env python3
"""Benchmark three-class tissue-domain GeoJSONs against expert annotation.

The expert GeoJSON is used only by this audit program. It is never imported by
the production workflow or by a refinement process. Cluster identities are
matched to reference identities with a Hungarian assignment before categorical
metrics are calculated.

Raster evaluation is intentionally performed on a declared, bounded analysis
grid. This avoids trusting polygon validity for large hand-edited GeoJSONs and
keeps memory use predictable. All boundary distances are converted back to
native level-0 pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment


def iter_polygons(geometry: dict) -> Iterable[list]:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if kind == "Polygon":
        yield coordinates
    elif kind == "MultiPolygon":
        yield from coordinates
    elif kind == "GeometryCollection":
        for item in geometry.get("geometries", []):
            yield from iter_polygons(item)
    else:
        raise ValueError(f"Unsupported geometry type: {kind!r}")


def infer_slide_shape(payload: dict) -> tuple[int, int] | None:
    shapes = set()
    for feature in payload.get("features", []):
        wsi = (feature.get("properties") or {}).get("wsiTools") or {}
        width, height = wsi.get("slide_width"), wsi.get("slide_height")
        if width is not None and height is not None:
            shapes.add((int(height), int(width)))
    if not shapes:
        return None
    if len(shapes) != 1:
        raise ValueError(
            "GeoJSON must provide one consistent wsiTools slide_width/slide_height "
            f"pair; found {sorted(shapes)}"
        )
    return shapes.pop()


def _feature_value(feature: dict, fallback: int) -> int:
    value = (feature.get("properties") or {}).get("value", fallback)
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = int(fallback)
    if value <= 0:
        raise ValueError(f"Positive tissue labels are required; found {value}")
    return value


def _draw_feature(feature: dict, size: tuple[int, int], scale: float) -> np.ndarray:
    canvas = Image.new("1", size, 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in iter_polygons(feature.get("geometry") or {}):
        if not polygon:
            continue
        exterior = [(float(point[0]) * scale, float(point[1]) * scale) for point in polygon[0]]
        if len(exterior) >= 3:
            draw.polygon(exterior, fill=1)
        for ring in polygon[1:]:
            hole = [(float(point[0]) * scale, float(point[1]) * scale) for point in ring]
            if len(hole) >= 3:
                draw.polygon(hole, fill=0)
    return np.asarray(canvas, dtype=bool)


def angular_geometry_metrics(payload: dict, *, minimum_adjacent_length: float = 16.0) -> dict:
    """Measure conspicuous vector corners without requiring valid polygons."""
    segment_lengths = []
    turns = []
    long_turns = []
    ring_count = 0
    for feature in payload.get("features", []):
        for polygon in iter_polygons(feature.get("geometry") or {}):
            for ring in polygon:
                points = np.asarray([[float(point[0]), float(point[1])] for point in ring], dtype=np.float64)
                if len(points) < 4:
                    continue
                if np.allclose(points[0], points[-1]):
                    points = points[:-1]
                if len(points) < 3:
                    continue
                vectors = np.roll(points, -1, axis=0) - points
                lengths = np.linalg.norm(vectors, axis=1)
                valid = lengths > 1e-9
                if np.count_nonzero(valid) < 3:
                    continue
                ring_count += 1
                segment_lengths.extend(lengths[valid].tolist())
                unit = np.zeros_like(vectors)
                unit[valid] = vectors[valid] / lengths[valid, None]
                previous = np.roll(unit, 1, axis=0)
                previous_lengths = np.roll(lengths, 1)
                valid_turn = valid & np.roll(valid, 1)
                angle = np.degrees(np.arccos(np.clip(np.sum(previous * unit, axis=1), -1.0, 1.0)))
                turns.extend(angle[valid_turn].tolist())
                long = valid_turn & (lengths >= minimum_adjacent_length) & (previous_lengths >= minimum_adjacent_length)
                long_turns.extend(angle[long].tolist())
    length_values = np.asarray(segment_lengths, dtype=float)
    turn_values = np.asarray(turns, dtype=float)
    long_values = np.asarray(long_turns, dtype=float)
    total_length = float(length_values.sum()) if length_values.size else 0.0
    return {
        "ring_count": int(ring_count),
        "segment_count": int(length_values.size),
        "total_ring_length_native_px": total_length,
        "median_segment_length_native_px": float(np.median(length_values)) if length_values.size else None,
        "p95_turn_degrees": float(np.percentile(turn_values, 95)) if turn_values.size else None,
        "long_vertex_count": int(long_values.size),
        "long_sharp_turn_count_ge_45deg": int(np.count_nonzero(long_values >= 45.0)),
        "long_sharp_turn_count_ge_75deg": int(np.count_nonzero(long_values >= 75.0)),
        "long_sharp_turn_density_ge_45deg_per_100k_ring_px": _safe_ratio(
            100000.0 * np.count_nonzero(long_values >= 45.0), total_length, empty=0.0,
        ),
        "minimum_adjacent_segment_length_native_px": float(minimum_adjacent_length),
        "scope": "all_exterior_and_hole_rings; shared boundaries may be represented twice",
    }


def rasterize_geojson(
    path: Path,
    *,
    full_shape: tuple[int, int] | None = None,
    downsample: int = 16,
) -> tuple[np.ndarray, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    inferred_shape = infer_slide_shape(payload)
    if full_shape is None and inferred_shape is None:
        raise ValueError(
            "GeoJSON has no wsiTools slide_width/slide_height metadata; provide "
            "full_shape from the reference"
        )
    if full_shape is None:
        full_shape = inferred_shape
    if inferred_shape is not None and tuple(full_shape) != tuple(inferred_shape):
        raise ValueError(f"Declared slide shape {full_shape} != GeoJSON shape {inferred_shape} for {path}")
    if downsample < 1:
        raise ValueError("downsample must be positive")
    height, width = full_shape
    analysis_size = (int(np.ceil(width / downsample)), int(np.ceil(height / downsample)))
    scale = 1.0 / float(downsample)
    labels = np.zeros((analysis_size[1], analysis_size[0]), dtype=np.int16)
    occupancy = np.zeros(labels.shape, dtype=np.uint8)
    feature_rows = []
    for index, feature in enumerate(payload.get("features", []), start=1):
        value = _feature_value(feature, index)
        mask = _draw_feature(feature, analysis_size, scale)
        occupancy += mask.astype(np.uint8)
        labels[mask] = value
        feature_rows.append({"feature_index": index, "value": value, "pixels": int(mask.sum())})
    metadata = {
        "path": str(path.resolve()),
        "full_shape": [int(height), int(width)],
        "analysis_shape": [int(labels.shape[0]), int(labels.shape[1])],
        "downsample": int(downsample),
        "feature_count": int(len(payload.get("features", []))),
        "positive_labels": [int(value) for value in np.unique(labels) if value > 0],
        "overlap_pixels": int(np.count_nonzero(occupancy > 1)),
        "maximum_overlap_multiplicity": int(occupancy.max(initial=0)),
        "feature_rows": feature_rows,
        "overlap_resolution_rule": "later GeoJSON feature wins, matching display order",
        "angular_geometry": angular_geometry_metrics(payload),
    }
    return labels, metadata


def internal_boundary(labels: np.ndarray) -> np.ndarray:
    source = np.asarray(labels)
    foreground = source > 0
    result = np.zeros(source.shape, dtype=bool)
    vertical = (source[:-1] != source[1:]) & foreground[:-1] & foreground[1:]
    horizontal = (source[:, :-1] != source[:, 1:]) & foreground[:, :-1] & foreground[:, 1:]
    result[:-1] |= vertical
    result[1:] |= vertical
    result[:, :-1] |= horizontal
    result[:, 1:] |= horizontal
    return result


def optimal_label_mapping(prediction: np.ndarray, reference: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    pred_ids = np.asarray([int(value) for value in np.unique(prediction) if value > 0], dtype=int)
    ref_ids = np.asarray([int(value) for value in np.unique(reference) if value > 0], dtype=int)
    if len(pred_ids) != len(ref_ids):
        raise ValueError(f"Prediction/reference class counts differ: {pred_ids.tolist()} vs {ref_ids.tolist()}")
    common = (prediction > 0) & (reference > 0)
    contingency = np.zeros((len(pred_ids), len(ref_ids)), dtype=np.int64)
    for row, pred_id in enumerate(pred_ids):
        for column, ref_id in enumerate(ref_ids):
            contingency[row, column] = np.count_nonzero(common & (prediction == pred_id) & (reference == ref_id))
    rows, columns = linear_sum_assignment(-contingency)
    mapping = {int(pred_ids[row]): int(ref_ids[column]) for row, column in zip(rows, columns)}
    return mapping, contingency


def _safe_ratio(numerator: float, denominator: float, *, empty: float = 1.0) -> float:
    return float(numerator / denominator) if denominator else float(empty)


def _boundary_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    downsample: int,
    tolerances_native_px: list[int],
    reference_cache: dict | None = None,
) -> dict:
    pred = internal_boundary(prediction)
    ref = np.asarray(reference_cache["boundary"], dtype=bool) if reference_cache is not None else internal_boundary(reference)
    pred_count, ref_count = int(pred.sum()), int(ref.sum())
    result = {
        "predicted_internal_boundary_pixels": pred_count,
        "reference_internal_boundary_pixels": ref_count,
        "boundary_length_ratio": _safe_ratio(pred_count, ref_count),
    }
    if pred_count and ref_count:
        reference_distance = np.asarray(reference_cache["distance_to_boundary"]) if reference_cache is not None else ndi.distance_transform_edt(~ref)
        distance_to_ref = reference_distance[pred] * downsample
        distance_to_pred = ndi.distance_transform_edt(~pred)[ref] * downsample
        symmetric = np.concatenate((distance_to_ref, distance_to_pred))
        result.update({
            "mean_symmetric_boundary_distance_native_px": float(symmetric.mean()),
            "p95_symmetric_boundary_distance_native_px": float(np.percentile(symmetric, 95)),
            "max_symmetric_boundary_distance_native_px": float(symmetric.max()),
        })
    else:
        result.update({
            "mean_symmetric_boundary_distance_native_px": None,
            "p95_symmetric_boundary_distance_native_px": None,
            "max_symmetric_boundary_distance_native_px": None,
        })
    f1_rows = []
    cached_near = reference_cache.get("near_boundary", {}) if reference_cache is not None else {}
    for tolerance in tolerances_native_px:
        iterations = int(np.ceil(tolerance / downsample))
        near_ref = cached_near.get(int(tolerance))
        if near_ref is None:
            near_ref = ndi.binary_dilation(ref, iterations=iterations) if iterations else ref
        near_pred = ndi.binary_dilation(pred, iterations=iterations) if iterations else pred
        precision = _safe_ratio(np.count_nonzero(pred & near_ref), pred_count, empty=0.0)
        recall = _safe_ratio(np.count_nonzero(ref & near_pred), ref_count, empty=0.0)
        f1 = _safe_ratio(2 * precision * recall, precision + recall, empty=0.0)
        f1_rows.append({
            "tolerance_native_px": int(tolerance),
            "effective_tolerance_native_px": int(iterations * downsample),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    result["boundary_f1"] = f1_rows
    return result


def prepare_boundary_reference(
    reference: np.ndarray,
    *,
    downsample: int,
    tolerances_native_px: list[int],
) -> dict:
    """Cache reference-only boundary arrays for repeated candidate scoring."""
    boundary = internal_boundary(reference)
    near_boundary = {}
    for tolerance in tolerances_native_px:
        iterations = int(np.ceil(tolerance / downsample))
        near_boundary[int(tolerance)] = (
            ndi.binary_dilation(boundary, iterations=iterations)
            if iterations
            else boundary
        )
    return {
        "boundary": boundary,
        "distance_to_boundary": ndi.distance_transform_edt(~boundary),
        "near_boundary": near_boundary,
    }


def score_candidate(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    downsample: int,
    tolerances_native_px: list[int],
    boundary_reference_cache: dict | None = None,
) -> dict:
    mapping, contingency = optimal_label_mapping(prediction, reference)
    aligned = np.zeros_like(prediction)
    for source, target in mapping.items():
        aligned[prediction == source] = target
    pred_tissue, ref_tissue = aligned > 0, reference > 0
    common = pred_tissue & ref_tissue
    union = pred_tissue | ref_tissue
    per_class = []
    for label in sorted(int(value) for value in np.unique(reference) if value > 0):
        pred_class, ref_class = aligned == label, reference == label
        intersection = int(np.count_nonzero(pred_class & ref_class))
        pred_count, ref_count = int(pred_class.sum()), int(ref_class.sum())
        class_union = int(np.count_nonzero(pred_class | ref_class))
        per_class.append({
            "reference_label": label,
            "prediction_source_label": next(source for source, target in mapping.items() if target == label),
            "prediction_pixels": pred_count,
            "reference_pixels": ref_count,
            "intersection_pixels": intersection,
            "dice": _safe_ratio(2 * intersection, pred_count + ref_count),
            "iou": _safe_ratio(intersection, class_union),
        })
    metrics = {
        "label_mapping": {str(source): target for source, target in sorted(mapping.items())},
        "contingency_pred_rows_ref_columns": contingency.tolist(),
        "prediction_tissue_pixels": int(pred_tissue.sum()),
        "reference_tissue_pixels": int(ref_tissue.sum()),
        "tissue_dice": _safe_ratio(2 * np.count_nonzero(common), pred_tissue.sum() + ref_tissue.sum()),
        "tissue_iou": _safe_ratio(np.count_nonzero(common), np.count_nonzero(union)),
        "categorical_accuracy_on_reference_tissue": _safe_ratio(
            np.count_nonzero(aligned[ref_tissue] == reference[ref_tissue]), ref_tissue.sum()
        ),
        "categorical_accuracy_on_common_tissue": _safe_ratio(
            np.count_nonzero(aligned[common] == reference[common]), common.sum()
        ),
        "mean_class_dice": float(np.mean([row["dice"] for row in per_class])),
        "mean_class_iou": float(np.mean([row["iou"] for row in per_class])),
        "per_class": per_class,
    }
    metrics.update(_boundary_metrics(
        aligned,
        reference,
        downsample=downsample,
        tolerances_native_px=tolerances_native_px,
        reference_cache=boundary_reference_cache,
    ))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True,
                        help="Candidate GeoJSON; repeat for multiple routes.")
    parser.add_argument("--name", action="append", default=[],
                        help="Candidate name in the same order as --candidate.")
    parser.add_argument("--downsample", type=int, default=16)
    parser.add_argument("--boundary-tolerance-native-px", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()
    if args.name and len(args.name) != len(args.candidate):
        raise ValueError("Either omit --name or provide one name per --candidate")

    reference, reference_metadata = rasterize_geojson(args.reference, downsample=args.downsample)
    tolerances = sorted(set(args.boundary_tolerance_native_px))
    boundary_reference_cache = prepare_boundary_reference(
        reference,
        downsample=args.downsample,
        tolerances_native_px=tolerances,
    )
    results = []
    for index, path in enumerate(args.candidate):
        prediction, prediction_metadata = rasterize_geojson(
            path, full_shape=tuple(reference_metadata["full_shape"]), downsample=args.downsample,
        )
        result = {
            "name": args.name[index] if args.name else path.stem,
            "candidate": str(path.resolve()),
            "rasterization": prediction_metadata,
            "metrics": score_candidate(
                prediction, reference, downsample=args.downsample,
                tolerances_native_px=tolerances,
                boundary_reference_cache=boundary_reference_cache,
            ),
        }
        results.append(result)

    results.sort(
        key=lambda item: (
            item["metrics"]["mean_class_iou"],
            -float(item["metrics"]["mean_symmetric_boundary_distance_native_px"] or np.inf),
        ),
        reverse=True,
    )
    payload = {
        "schema_version": "cellphenotyper.k3_ground_truth_benchmark.v1",
        "reference": reference_metadata,
        "ground_truth_usage": "evaluation_and_parameter_selection_only; never an inference input",
        "class_identity": "Hungarian assignment maximizing common-tissue overlap",
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    rows = []
    for rank, item in enumerate(results, start=1):
        metrics = item["metrics"]
        row = {
            "rank": rank,
            "name": item["name"],
            "mean_class_iou": metrics["mean_class_iou"],
            "mean_class_dice": metrics["mean_class_dice"],
            "categorical_accuracy_on_reference_tissue": metrics["categorical_accuracy_on_reference_tissue"],
            "tissue_iou": metrics["tissue_iou"],
            "mean_symmetric_boundary_distance_native_px": metrics["mean_symmetric_boundary_distance_native_px"],
            "p95_symmetric_boundary_distance_native_px": metrics["p95_symmetric_boundary_distance_native_px"],
            "boundary_length_ratio": metrics["boundary_length_ratio"],
        }
        for entry in metrics["boundary_f1"]:
            row[f"boundary_f1_{entry['tolerance_native_px']}px"] = entry["f1"]
        rows.append(row)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
