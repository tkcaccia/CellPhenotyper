#!/usr/bin/env python3
"""Compare cell-centred and grid UNI-2 tissue partitions to adjudicated domains."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import tifffile


REQUIRED_COLUMNS = {
    "study_id", "split", "patient_id", "slide_id", "roi_id", "site", "scanner",
    "tissue", "compartment", "quality_stratum", "route", "condition", "analysis_intent",
    "reference_endpoint", "reference_status", "reference_path", "prediction_path",
    "tissue_mask_path", "mask_stage", "mpp_x", "mpp_y", "image_width_px",
    "image_height_px", "pipeline_commit", "container_digest", "uni2_model_revision",
    "uni2_feature_definition", "clustering_definition", "primary_metric",
    "comparison_direction", "acceptance_margin", "decision_rule",
}
ALLOWED_SPLITS = {"development", "internal_test", "external_test"}
ALLOWED_ROUTES = {"cells", "grid"}
ALLOWED_DIRECTIONS = {"grid_minus_cells", "cells_minus_grid"}
ALLOWED_PRIMARY_METRICS = {
    "ari_including_abstention", "ari_accepted", "nmi_including_abstention",
    "boundary_f1", "coverage_fraction",
}
PAIR_METRICS = [
    "ari_including_abstention", "ari_accepted", "nmi_including_abstention",
    "nmi_accepted", "homogeneity_including_abstention", "completeness_including_abstention",
    "variation_of_information_including_abstention", "coverage_fraction", "boundary_f1",
    "mean_symmetric_boundary_error_um",
]
HIGHER_IS_BETTER = set(PAIR_METRICS).difference(
    {"variation_of_information_including_abstention", "mean_symmetric_boundary_error_um"}
)
ROUTE_COLORS = np.asarray(
    [
        [31, 44, 51], [0, 114, 178], [213, 94, 0], [0, 158, 115], [204, 121, 167],
        [230, 159, 0], [86, 180, 233], [240, 228, 66], [148, 103, 189], [102, 166, 30],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cell-centred and grid UNI-2 tissue-domain masks with an adjudicated "
            "reference using label-invariant, abstention-aware, patient-level metrics."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-mask-pixels", type=int, default=16_000_000)
    parser.add_argument("--max-evaluation-pixels", type=int, default=1_000_000)
    parser.add_argument("--boundary-tolerance-um", type=float, default=8.0)
    parser.add_argument("--max-boundary-points", type=int, default=200_000)
    parser.add_argument("--max-review-points", type=int, default=200)
    parser.add_argument("--preview-max-side", type=int, default=1000)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def validate_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"UNI-2 route manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("UNI-2 route manifest contains no route/ROI rows")
    for column in REQUIRED_COLUMNS:
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Manifest column '{column}' contains empty values")
    invalid_splits = sorted(set(frame["split"]) - ALLOWED_SPLITS)
    if invalid_splits:
        raise ValueError(f"Unsupported split values: {invalid_splits}")
    invalid_routes = sorted(set(frame["route"]) - ALLOWED_ROUTES)
    if invalid_routes:
        raise ValueError(f"Unsupported routes: {invalid_routes}")
    if set(frame["analysis_intent"]) != {"tissue_domain_discovery"}:
        raise ValueError("Every row must declare analysis_intent=tissue_domain_discovery")
    if frame["reference_status"].str.lower().ne("adjudicated").any():
        raise ValueError("Every reference must have reference_status=adjudicated")
    invalid_directions = sorted(set(frame["comparison_direction"]) - ALLOWED_DIRECTIONS)
    if invalid_directions:
        raise ValueError(f"Unsupported comparison directions: {invalid_directions}")
    invalid_primary = sorted(set(frame["primary_metric"]) - ALLOWED_PRIMARY_METRICS)
    if invalid_primary:
        raise ValueError(f"Unsupported primary metrics: {invalid_primary}")
    if set(frame["decision_rule"]) != {"bootstrap_ci_lower_bound"}:
        raise ValueError("decision_rule must be bootstrap_ci_lower_bound")

    for column in ("mpp_x", "mpp_y", "acceptance_margin"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (~np.isfinite(frame[column])).any():
            raise ValueError(f"Manifest {column} must contain finite values")
    if ((frame["mpp_x"] <= 0) | (frame["mpp_y"] <= 0)).any():
        raise ValueError("Manifest MPP values must be positive")
    for column in ("image_width_px", "image_height_px"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
        if (frame[column] <= 0).any():
            raise ValueError(f"Manifest {column} must contain positive integers")

    frame["patient_key"] = frame["study_id"] + "::" + frame["patient_id"]
    frame["slide_key"] = frame["study_id"] + "::" + frame["slide_id"]
    frame["roi_key"] = frame["patient_key"] + "::" + frame["slide_id"] + "::" + frame["roi_id"]
    frame["comparison_key"] = frame["roi_key"] + "::" + frame["condition"] + "::" + frame["mask_stage"]
    patient_splits = frame.groupby("patient_key")["split"].nunique()
    if (patient_splits > 1).any():
        raise ValueError(
            f"Patient leakage across splits: {patient_splits[patient_splits > 1].index[:10].tolist()}"
        )
    slide_patients = frame.groupby("slide_key")["patient_key"].nunique()
    if (slide_patients > 1).any():
        raise ValueError(
            f"Slide IDs assigned to multiple patients: {slide_patients[slide_patients > 1].index[:10].tolist()}"
        )
    if frame.duplicated(["comparison_key", "route"]).any():
        duplicates = frame.loc[
            frame.duplicated(["comparison_key", "route"], keep=False), ["comparison_key", "route"]
        ]
        raise ValueError(f"Duplicate route/ROI rows: {duplicates.head(10).to_dict('records')}")
    route_sets = frame.groupby("comparison_key")["route"].agg(lambda values: set(values))
    incomplete = route_sets[route_sets != ALLOWED_ROUTES]
    if len(incomplete):
        raise ValueError(f"Every comparison requires cells and grid routes: {incomplete.index[:10].tolist()}")

    invariant_columns = [
        "study_id", "split", "patient_id", "slide_id", "roi_id", "site", "scanner", "tissue",
        "compartment", "quality_stratum", "analysis_intent", "reference_endpoint",
        "reference_status", "reference_path", "tissue_mask_path", "mask_stage", "mpp_x", "mpp_y",
        "image_width_px", "image_height_px", "pipeline_commit", "container_digest",
        "uni2_model_revision", "clustering_definition", "primary_metric", "comparison_direction",
        "acceptance_margin", "decision_rule",
    ]
    for comparison_key, group in frame.groupby("comparison_key", sort=False):
        changed = [column for column in invariant_columns if group[column].nunique(dropna=False) != 1]
        if changed:
            raise ValueError(f"Comparison metadata differ within {comparison_key}: {changed}")

    for column in ("reference_path", "prediction_path", "tissue_mask_path"):
        frame[column] = frame[column].map(lambda value: str(resolve_path(path, value)))
        absent = [value for value in frame[column] if not Path(value).is_file()]
        if absent:
            raise FileNotFoundError(f"Missing {column} files: {absent[:10]}")
    return frame


def load_label_mask(path: Path, shape: tuple[int, int], max_pixels: int, role: str) -> np.ndarray:
    array = tifffile.imread(path)
    array = np.asarray(array)
    while array.ndim > 2 and 1 in array.shape:
        array = np.squeeze(array, axis=next(index for index, size in enumerate(array.shape) if size == 1))
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{role} must be a 2D integer TIFF: {path} shape={array.shape}")
    if array.size > max_pixels:
        raise ValueError(f"{role} has {array.size:,} pixels; validation limit={max_pixels:,}: {path}")
    if tuple(array.shape) != shape:
        raise ValueError(f"{role} shape {array.shape} differs from declared shape {shape}: {path}")
    if np.any(array < 0):
        raise ValueError(f"{role} contains negative labels: {path}")
    return array.astype(np.int64, copy=False)


def contingency_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if len(reference) == 0:
        return {
            "adjusted_rand_index": float("nan"), "normalized_mutual_information": float("nan"),
            "homogeneity": float("nan"), "completeness": float("nan"),
            "variation_of_information": float("nan"),
        }
    _, ref_inverse = np.unique(reference, return_inverse=True)
    _, pred_inverse = np.unique(prediction, return_inverse=True)
    n_ref = int(ref_inverse.max()) + 1
    n_pred = int(pred_inverse.max()) + 1
    contingency = np.bincount(
        ref_inverse * n_pred + pred_inverse, minlength=n_ref * n_pred
    ).reshape(n_ref, n_pred).astype(np.float64)
    total = float(contingency.sum())
    rows = contingency.sum(axis=1)
    columns = contingency.sum(axis=0)

    def choose_two(values: np.ndarray) -> float:
        return float(np.sum(values * (values - 1) / 2))

    cell_pairs = choose_two(contingency)
    row_pairs = choose_two(rows)
    column_pairs = choose_two(columns)
    total_pairs = total * (total - 1) / 2
    expected = row_pairs * column_pairs / total_pairs if total_pairs > 0 else 0.0
    denominator = 0.5 * (row_pairs + column_pairs) - expected
    ari = (cell_pairs - expected) / denominator if denominator else float(reference.tolist() == prediction.tolist())

    probabilities = contingency / total
    row_prob = rows / total
    column_prob = columns / total
    nonzero = probabilities > 0
    expected_prob = row_prob[:, None] * column_prob[None, :]
    mutual_information = float(np.sum(probabilities[nonzero] * np.log(probabilities[nonzero] / expected_prob[nonzero])))
    ref_entropy = float(-np.sum(row_prob[row_prob > 0] * np.log(row_prob[row_prob > 0])))
    pred_entropy = float(-np.sum(column_prob[column_prob > 0] * np.log(column_prob[column_prob > 0])))
    nmi = 2 * mutual_information / (ref_entropy + pred_entropy) if ref_entropy + pred_entropy else 1.0
    homogeneity = 1.0 if ref_entropy == 0 else mutual_information / ref_entropy
    completeness = 1.0 if pred_entropy == 0 else mutual_information / pred_entropy
    return {
        "adjusted_rand_index": float(ari),
        "normalized_mutual_information": float(nmi),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "variation_of_information": float(ref_entropy + pred_entropy - 2 * mutual_information),
    }


def label_boundaries(labels: np.ndarray, support: np.ndarray, require_nonzero: bool) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = support[:, 1:] & support[:, :-1] & (labels[:, 1:] != labels[:, :-1])
    vertical = support[1:, :] & support[:-1, :] & (labels[1:, :] != labels[:-1, :])
    if require_nonzero:
        horizontal &= (labels[:, 1:] > 0) & (labels[:, :-1] > 0)
        vertical &= (labels[1:, :] > 0) & (labels[:-1, :] > 0)
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def sampled_coordinates(mask: np.ndarray, maximum: int) -> tuple[np.ndarray, int]:
    coordinates = np.column_stack(np.nonzero(mask))
    total = len(coordinates)
    if total > maximum:
        indices = np.linspace(0, total - 1, maximum, dtype=np.int64)
        coordinates = coordinates[indices]
    return coordinates, total


def boundary_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    support: np.ndarray,
    mpp_x: float,
    mpp_y: float,
    tolerance_um: float,
    max_points: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    reference_boundary = label_boundaries(reference, support, require_nonzero=True)
    prediction_boundary = label_boundaries(prediction, support, require_nonzero=True)
    ref_coords, ref_total = sampled_coordinates(reference_boundary, max_points)
    pred_coords, pred_total = sampled_coordinates(prediction_boundary, max_points)
    empty = np.empty(0, dtype=float)
    if len(ref_coords) == 0 or len(pred_coords) == 0:
        metrics = {
            "reference_boundary_pixels": ref_total,
            "prediction_boundary_pixels": pred_total,
            "reference_boundary_points_evaluated": len(ref_coords),
            "prediction_boundary_points_evaluated": len(pred_coords),
            "boundary_precision": float("nan"),
            "boundary_recall": float("nan"),
            "boundary_f1": float("nan"),
            "mean_symmetric_boundary_error_um": float("nan"),
        }
        return metrics, {"reference": ref_coords, "prediction": pred_coords, "ref_distance": empty, "pred_distance": empty}
    scale = np.asarray([mpp_y, mpp_x], dtype=float)
    ref_physical = ref_coords * scale
    pred_physical = pred_coords * scale
    ref_distance = cKDTree(pred_physical).query(ref_physical, k=1)[0]
    pred_distance = cKDTree(ref_physical).query(pred_physical, k=1)[0]
    precision = float(np.mean(pred_distance <= tolerance_um))
    recall = float(np.mean(ref_distance <= tolerance_um))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "reference_boundary_pixels": ref_total,
        "prediction_boundary_pixels": pred_total,
        "reference_boundary_points_evaluated": len(ref_coords),
        "prediction_boundary_points_evaluated": len(pred_coords),
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": float(f1),
        "mean_symmetric_boundary_error_um": float((ref_distance.mean() + pred_distance.mean()) / 2),
    }
    details = {
        "reference": ref_coords, "prediction": pred_coords,
        "ref_distance": ref_distance, "pred_distance": pred_distance,
    }
    return metrics, details


def evaluate_partition(
    reference: np.ndarray,
    prediction: np.ndarray,
    tissue: np.ndarray,
    sample_indices: np.ndarray,
    *,
    mpp_x: float,
    mpp_y: float,
    boundary_tolerance_um: float,
    max_boundary_points: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    support = (reference > 0) & (tissue > 0)
    support_flat = np.flatnonzero(support)
    sampled_flat = support_flat[sample_indices]
    reference_values = reference.ravel()[sampled_flat]
    prediction_values = prediction.ravel()[sampled_flat]
    accepted = prediction_values > 0
    all_metrics = contingency_metrics(reference_values, prediction_values)
    accepted_metrics = contingency_metrics(reference_values[accepted], prediction_values[accepted])
    boundary, details = boundary_metrics(
        reference, prediction, support, mpp_x, mpp_y, boundary_tolerance_um, max_boundary_points
    )
    accepted_values = prediction_values[accepted]
    cluster_fraction = (
        float(np.bincount(np.unique(accepted_values, return_inverse=True)[1]).max() / len(accepted_values))
        if len(accepted_values)
        else float("nan")
    )
    metrics = {
        "reference_support_pixels": int(len(support_flat)),
        "evaluation_pixels": int(len(sampled_flat)),
        "coverage_fraction": float(np.mean(accepted)),
        "abstention_fraction": float(np.mean(~accepted)),
        "reference_domain_count": int(len(np.unique(reference_values))),
        "predicted_cluster_count": int(len(np.unique(accepted_values))) if len(accepted_values) else 0,
        "largest_predicted_cluster_fraction": cluster_fraction,
        "ari_including_abstention": all_metrics["adjusted_rand_index"],
        "nmi_including_abstention": all_metrics["normalized_mutual_information"],
        "homogeneity_including_abstention": all_metrics["homogeneity"],
        "completeness_including_abstention": all_metrics["completeness"],
        "variation_of_information_including_abstention": all_metrics["variation_of_information"],
        "ari_accepted": accepted_metrics["adjusted_rand_index"],
        "nmi_accepted": accepted_metrics["normalized_mutual_information"],
        **boundary,
    }
    return metrics, details


def review_features(details: dict[str, np.ndarray], metadata: dict, tolerance: float, maximum: int) -> list[dict]:
    candidates = []
    for kind, coordinates, distances in (
        ("missed_reference_boundary", details["reference"], details["ref_distance"]),
        ("unsupported_predicted_boundary", details["prediction"], details["pred_distance"]),
    ):
        outside = np.flatnonzero(distances > tolerance)
        if len(outside) > maximum:
            ranked = outside[np.argsort(-distances[outside], kind="mergesort")[:maximum]]
        else:
            ranked = outside
        for index in ranked:
            y, x = coordinates[index]
            candidates.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [int(x), int(y)]},
                    "properties": {
                        **metadata,
                        "disagreement_type": kind,
                        "nearest_boundary_distance_um": float(distances[index]),
                        "selection": "largest_boundary_disagreements_for_review_not_exclusion",
                    },
                }
            )
    return candidates


def colorize(labels: np.ndarray, palette_offset: int) -> np.ndarray:
    output = ROUTE_COLORS[np.mod(labels + palette_offset, len(ROUTE_COLORS))]
    output = output.copy()
    output[labels == 0] = np.asarray([45, 48, 50], dtype=np.uint8)
    return output


def write_preview(
    reference: np.ndarray,
    predictions: dict[str, np.ndarray],
    tissue: np.ndarray,
    output: Path,
    max_side: int,
    title: str,
) -> None:
    height, width = reference.shape
    scale = min(1.0, max_side / max(height, width))
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    y_index = np.linspace(0, height - 1, target_height, dtype=np.int64)
    x_index = np.linspace(0, width - 1, target_width, dtype=np.int64)
    arrays = {"Reference": reference, "Cell-centred": predictions["cells"], "Grid": predictions["grid"]}
    header = 70
    canvas = Image.new("RGB", (target_width * 3, target_height + header), "#f3efe5")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="#18323a")
    draw.text((10, 26), "Within-panel IDs only; labels are not mapped.", fill="#5b6669")
    for panel_index, (label, array) in enumerate(arrays.items()):
        sampled = array[np.ix_(y_index, x_index)]
        sampled_tissue = tissue[np.ix_(y_index, x_index)] > 0
        colored = colorize(sampled, palette_offset=panel_index * 3)
        colored[~sampled_tissue] = np.asarray([238, 235, 225], dtype=np.uint8)
        panel = Image.fromarray(colored)
        x0 = panel_index * target_width
        canvas.paste(panel, (x0, header))
        draw.text((x0 + 8, 48), label, fill="#18323a")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def patient_metrics(per_roi: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "study_id", "split", "patient_id", "patient_key", "route", "condition", "mask_stage",
        "reference_endpoint", "pipeline_commit", "container_digest", "uni2_model_revision",
        "uni2_feature_definition", "clustering_definition", "primary_metric",
        "comparison_direction", "acceptance_margin", "decision_rule",
    ]
    rows = []
    numeric = PAIR_METRICS + [
        "abstention_fraction", "predicted_cluster_count", "largest_predicted_cluster_fraction",
        "boundary_precision", "boundary_recall",
    ]
    for keys, group in per_roi.groupby(identity, sort=True):
        row = dict(zip(identity, keys))
        row["slides"] = int(group["slide_id"].nunique())
        row["rois"] = int(group["roi_key"].nunique())
        row["reference_support_pixels"] = int(group["reference_support_pixels"].sum())
        for metric in numeric:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[metric] = float(values.mean()) if values.notna().any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_route_metrics(patient: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = [
        "split", "route", "condition", "mask_stage", "reference_endpoint", "pipeline_commit",
        "container_digest", "uni2_model_revision", "uni2_feature_definition", "clustering_definition",
    ]
    rows = []
    for keys, group in patient.groupby(groups, sort=True):
        row = dict(zip(groups, keys))
        row["patients"] = int(group["patient_key"].nunique())
        for metric in PAIR_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_patients"] = int(len(values))
            row[metric] = float(values.mean()) if len(values) else float("nan")
            bootstrap = []
            if len(values) >= 2 and replicates > 0:
                for _ in range(replicates):
                    bootstrap.append(float(rng.choice(values, size=len(values), replace=True).mean()))
            row[f"{metric}_ci_low"] = float(np.percentile(bootstrap, 2.5)) if bootstrap else np.nan
            row[f"{metric}_ci_high"] = float(np.percentile(bootstrap, 97.5)) if bootstrap else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_route_differences(patient: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    fixed = [
        "study_id", "split", "condition", "mask_stage", "reference_endpoint", "pipeline_commit",
        "container_digest", "uni2_model_revision", "clustering_definition", "primary_metric",
        "comparison_direction", "acceptance_margin", "decision_rule",
    ]
    rows = []
    rng = np.random.default_rng(seed + 1103)
    for keys, group in patient.groupby(fixed, sort=True):
        metadata = dict(zip(fixed, keys))
        cells = group[group["route"] == "cells"]
        grid = group[group["route"] == "grid"]
        paired = cells.merge(grid, on="patient_key", suffixes=("_cells", "_grid"), validate="one_to_one")
        if paired.empty:
            continue
        direction = metadata["comparison_direction"]
        for metric in PAIR_METRICS:
            cell_values = pd.to_numeric(paired[f"{metric}_cells"], errors="coerce")
            grid_values = pd.to_numeric(paired[f"{metric}_grid"], errors="coerce")
            valid = cell_values.notna() & grid_values.notna()
            if direction == "grid_minus_cells":
                differences = (grid_values[valid] - cell_values[valid]).to_numpy(dtype=float)
            else:
                differences = (cell_values[valid] - grid_values[valid]).to_numpy(dtype=float)
            point = float(differences.mean()) if len(differences) else float("nan")
            bootstrap = []
            if len(differences) >= 2 and replicates > 0:
                for _ in range(replicates):
                    bootstrap.append(
                        float(rng.choice(differences, size=len(differences), replace=True).mean())
                    )
            low = float(np.percentile(bootstrap, 2.5)) if bootstrap else np.nan
            high = float(np.percentile(bootstrap, 97.5)) if bootstrap else np.nan
            oriented_point = point if metric in HIGHER_IS_BETTER else -point
            oriented_low = low if metric in HIGHER_IS_BETTER else -high
            is_primary = metric == metadata["primary_metric"]
            if not is_primary:
                decision = "secondary_metric_no_decision"
            elif metadata["split"] == "development":
                decision = "development_only_no_confirmatory_decision"
            elif not math.isfinite(oriented_low):
                decision = "insufficient_patients_for_ci_decision"
            elif oriented_low >= float(metadata["acceptance_margin"]):
                decision = "passes_prespecified_margin"
            else:
                decision = "does_not_pass_prespecified_margin"
            rows.append(
                {
                    **metadata,
                    "metric": metric,
                    "metric_direction": "higher_is_better" if metric in HIGHER_IS_BETTER else "lower_is_better",
                    "paired_patients": int(len(differences)),
                    "route_difference": point,
                    "ci_low": low,
                    "ci_high": high,
                    "oriented_improvement": oriented_point,
                    "oriented_ci_lower_bound": oriented_low,
                    "is_primary": is_primary,
                    "decision": decision,
                }
            )
    return pd.DataFrame(rows)


def stratified_route_metrics(per_roi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stratum in ("site", "scanner", "tissue", "compartment", "quality_stratum"):
        columns = ["split", "route", "condition", "mask_stage", "reference_endpoint", stratum]
        patient_columns = [*columns, "patient_key"]
        patient_rows = []
        for keys, group in per_roi.groupby(patient_columns, sort=True):
            row = dict(zip(patient_columns, keys))
            for metric in PAIR_METRICS:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[metric] = float(values.mean()) if values.notna().any() else float("nan")
            patient_rows.append(row)
        patient_frame = pd.DataFrame(patient_rows)
        for keys, group in patient_frame.groupby(columns, sort=True):
            row = {
                "split": keys[0], "route": keys[1], "condition": keys[2],
                "mask_stage": keys[3], "reference_endpoint": keys[4],
                "stratum_type": stratum, "stratum_value": keys[5],
                "patients": int(group["patient_key"].nunique()),
            }
            for metric in PAIR_METRICS:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[metric] = float(values.mean()) if values.notna().any() else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "comparison"


def format_ci(row: dict, metric: str) -> str:
    value = float(row[metric])
    low = float(row[f"{metric}_ci_low"])
    high = float(row[f"{metric}_ci_high"])
    if not math.isfinite(value):
        return "NA"
    if math.isfinite(low) and math.isfinite(high):
        return f"{value:.3f} [{low:.3f}, {high:.3f}]"
    return f"{value:.3f} [CI unavailable]"


def render_html(summary: dict, aggregate: pd.DataFrame, paired: pd.DataFrame) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    aggregate_rows = []
    for row in aggregate.to_dict("records"):
        aggregate_rows.append(
            "<tr>"
            f"<td>{esc(row['split'])}</td><td>{esc(row['route'])}</td>"
            f"<td>{esc(row['condition'])}</td><td>{row['patients']}</td>"
            f"<td>{format_ci(row, 'ari_including_abstention')}</td>"
            f"<td>{format_ci(row, 'ari_accepted')}</td>"
            f"<td>{format_ci(row, 'coverage_fraction')}</td>"
            f"<td>{format_ci(row, 'boundary_f1')}</td>"
            "</tr>"
        )
    primary = paired[paired["is_primary"] == True] if not paired.empty else paired
    decision_rows = []
    for row in primary.to_dict("records"):
        decision_rows.append(
            "<tr>"
            f"<td>{esc(row['split'])}</td><td>{esc(row['condition'])}</td>"
            f"<td>{esc(row['comparison_direction'])}</td><td>{esc(row['metric'])}</td>"
            f"<td>{row['route_difference']:.3f}</td><td>{row['ci_low']:.3f}</td>"
            f"<td>{row['ci_high']:.3f}</td><td>{esc(row['decision'])}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UNI-2 route reference benchmark</title><style>
body{{font-family:Georgia,serif;margin:0;background:#f3efe5;color:#18323a}}main{{max-width:1250px;margin:auto;padding:42px 24px}}
h1{{font-size:3rem;line-height:1;margin:.2em 0}}.notice{{background:#fffaf0;border-left:8px solid #b84a2b;padding:18px;margin:24px 0}}
table{{width:100%;border-collapse:collapse;background:white;font-family:Arial,sans-serif;font-size:.86rem;margin-bottom:28px}}th,td{{padding:9px;border-bottom:1px solid #ccc;text-align:right}}th{{background:#18323a;color:white}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
</style></head><body><main><p>CellPhenotyper validation evidence</p><h1>UNI-2 tissue-domain routes</h1>
<div class="notice"><strong>Scope:</strong> This report compares two distinct scientific observation routes with adjudicated tissue domains. Label-invariant agreement is not biological naming, and accepted-pixel scores must be read with coverage and abstention.</div>
<p>Study: <strong>{esc(summary['study_id'])}</strong>. Patients: {summary['patients']}. ROIs: {summary['rois']}. Manifest SHA-256: <code>{esc(summary['manifest_sha256'])}</code>.</p>
<h2>Route estimates</h2><table><thead><tr><th>Split</th><th>Route</th><th>Condition</th><th>Patients</th><th>ARI incl. abstention [95% CI]</th><th>ARI accepted [95% CI]</th><th>Coverage [95% CI]</th><th>Boundary F1 [95% CI]</th></tr></thead><tbody>{''.join(aggregate_rows)}</tbody></table>
<h2>Prespecified paired decision</h2><table><thead><tr><th>Split</th><th>Condition</th><th>Direction</th><th>Metric</th><th>Difference</th><th>CI low</th><th>CI high</th><th>Decision</th></tr></thead><tbody>{''.join(decision_rows)}</tbody></table>
<h2>Required review</h2><p>Inspect every QC panel and <code>route_boundary_disagreements.geojson</code>. Do not choose the route from a test-set image, remap clusters to the reference, or discard abstained pixels after viewing the result.</p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be nonnegative")
    if args.max_evaluation_pixels < 100:
        raise ValueError("--max-evaluation-pixels must be at least 100")
    if args.boundary_tolerance_um <= 0:
        raise ValueError("--boundary-tolerance-um must be positive")
    manifest_path = Path(args.manifest).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    preview_dir = outdir / "route_qc"
    manifest = validate_manifest(manifest_path)
    rng = np.random.default_rng(args.seed)
    per_roi_rows = []
    provenance_rows = []
    review_geojson = []
    hash_cache: dict[str, str] = {}

    for comparison_key, group in manifest.groupby("comparison_key", sort=True):
        first = group.iloc[0]
        shape = (int(first.image_height_px), int(first.image_width_px))
        reference = load_label_mask(Path(first.reference_path), shape, args.max_mask_pixels, "Reference mask")
        tissue = load_label_mask(Path(first.tissue_mask_path), shape, args.max_mask_pixels, "Tissue mask")
        support_count = int(np.sum((reference > 0) & (tissue > 0)))
        if support_count < 100:
            raise ValueError(f"Reference support has fewer than 100 pixels: {comparison_key}")
        evaluation_count = min(support_count, args.max_evaluation_pixels)
        sample_indices = (
            np.sort(rng.choice(support_count, size=evaluation_count, replace=False))
            if evaluation_count < support_count
            else np.arange(support_count, dtype=np.int64)
        )
        predictions = {}
        for row in group.itertuples(index=False):
            prediction = load_label_mask(
                Path(row.prediction_path), shape, args.max_mask_pixels, f"{row.route} prediction mask"
            )
            predictions[row.route] = prediction
            metrics, details = evaluate_partition(
                reference,
                prediction,
                tissue,
                sample_indices,
                mpp_x=float(row.mpp_x),
                mpp_y=float(row.mpp_y),
                boundary_tolerance_um=args.boundary_tolerance_um,
                max_boundary_points=args.max_boundary_points,
            )
            metadata_columns = [
                "study_id", "split", "patient_id", "patient_key", "slide_id", "slide_key",
                "roi_id", "roi_key", "site", "scanner", "tissue", "compartment",
                "quality_stratum", "route", "condition", "mask_stage", "reference_endpoint",
                "pipeline_commit", "container_digest", "uni2_model_revision",
                "uni2_feature_definition", "clustering_definition", "primary_metric",
                "comparison_direction", "acceptance_margin", "decision_rule",
            ]
            metadata = {column: getattr(row, column) for column in metadata_columns}
            per_roi_rows.append({**metadata, **metrics})
            review_geojson.extend(
                review_features(
                    details, metadata, args.boundary_tolerance_um, args.max_review_points
                )
            )
            provenance = {**metadata}
            for name in ("reference_path", "prediction_path", "tissue_mask_path"):
                path = str(getattr(row, name))
                if path not in hash_cache:
                    hash_cache[path] = sha256_file(Path(path))
                provenance[name] = path
                provenance[f"{name}_sha256"] = hash_cache[path]
            provenance_rows.append(provenance)
        preview_name = safe_name(comparison_key) + ".png"
        write_preview(
            reference,
            predictions,
            tissue,
            preview_dir / preview_name,
            args.preview_max_side,
            f"{first.slide_id} / {first.roi_id} / {first.condition}",
        )

    per_roi = pd.DataFrame(per_roi_rows)
    patient = patient_metrics(per_roi)
    aggregate = aggregate_route_metrics(patient, args.bootstrap_replicates, args.seed)
    paired = paired_route_differences(patient, args.bootstrap_replicates, args.seed)
    strata = stratified_route_metrics(per_roi)
    per_roi.to_csv(outdir / "uni2_route_metrics_per_roi.csv", index=False)
    patient.to_csv(outdir / "uni2_route_metrics_per_patient.csv", index=False)
    aggregate.to_csv(outdir / "uni2_route_metrics_aggregate.csv", index=False)
    paired.to_csv(outdir / "uni2_route_paired_differences.csv", index=False)
    strata.to_csv(outdir / "uni2_route_metrics_by_stratum.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(outdir / "uni2_route_benchmark_provenance.csv", index=False)
    (outdir / "route_boundary_disagreements.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "metadata": {
                    "coordinate_space": "crop_level0_pixels",
                    "boundary_tolerance_um": args.boundary_tolerance_um,
                    "interpretation": "Blinded review sample; never an automatic exclusion list.",
                },
                "features": review_geojson,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    primary_decisions = (
        paired[paired["is_primary"] == True][
            ["split", "condition", "metric", "comparison_direction", "acceptance_margin", "decision"]
        ].to_dict("records")
        if not paired.empty
        else []
    )
    summary = {
        "schema_version": 1,
        "study_id": ";".join(sorted(manifest["study_id"].unique())),
        "analysis_intent": "tissue_domain_discovery",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "patients": int(manifest["patient_key"].nunique()),
        "slides": int(manifest["slide_key"].nunique()),
        "rois": int(manifest["roi_key"].nunique()),
        "routes": sorted(manifest["route"].unique()),
        "conditions": sorted(manifest["condition"].unique()),
        "evaluation_sampling": "identical deterministic reference-support pixel sample for both routes",
        "primary_statistical_unit": "patient_macro_average",
        "boundary_tolerance_um": float(args.boundary_tolerance_um),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "primary_decisions": primary_decisions,
        "claim_limit": (
            "Results compare route partitions only for the declared adjudicated domains, cohort, masks, "
            "model revision, feature definitions and clustering policy. Agreement does not name a domain, "
            "validate cell phenotypes, or establish external or clinical generalization."
        ),
    }
    (outdir / "uni2_route_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (outdir / "uni2_route_benchmark_report.html").write_text(
        render_html(summary, aggregate, paired), encoding="utf-8"
    )
    print(
        f"[INFO] UNI-2 route benchmark complete: patients={summary['patients']} "
        f"rois={summary['rois']} output={outdir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
