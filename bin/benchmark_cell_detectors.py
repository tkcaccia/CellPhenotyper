#!/usr/bin/env python3
"""Evaluate detector instance geometry against adjudicated ROI annotations."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.affinity import scale as scale_geometry
from shapely.geometry import Polygon, mapping, shape
from shapely.strtree import STRtree


REQUIRED_MANIFEST_COLUMNS = {
    "study_id",
    "split",
    "patient_id",
    "slide_id",
    "roi_id",
    "site",
    "scanner",
    "tissue",
    "compartment",
    "quality_stratum",
    "mpp_x",
    "mpp_y",
    "image_width_px",
    "image_height_px",
    "coordinate_space",
    "reference_path",
    "detector",
    "condition",
    "prediction_path",
    "prediction_format",
}
ALLOWED_SPLITS = {"development", "internal_test", "external_test"}
ALLOWED_FORMATS = {"geojson", "cell_json", "labels_tif"}


@dataclass
class Instance:
    instance_id: str
    geometry: object
    class_label: str | None


@dataclass
class Pair:
    reference_index: int
    prediction_index: int
    intersection: float
    union: float
    iou: float
    dice: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark StarDist, HoVer-Net, CellViT++, or consensus instance geometry "
            "against expert reference polygons. Phenotype taxonomies are retained but not scored."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--iou-thresholds", default="0.50,0.75")
    parser.add_argument("--primary-iou-threshold", type=float, default=0.50)
    parser.add_argument("--split-merge-overlap", type=float, default=0.10)
    parser.add_argument("--boundary-samples", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-unit", choices=("patient_id", "slide_id"), default="slide_id")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-mask-pixels", type=int, default=64_000_000)
    parser.add_argument("--prediction-invalid-policy", choices=("fail", "repair"), default="fail")
    parser.add_argument("--require-adjudicated", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_thresholds(raw: str, primary: float) -> list[float]:
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--iou-thresholds must be comma-separated numbers") from exc
    values.append(float(primary))
    thresholds = sorted(set(values))
    if not thresholds or any(not 0 < value <= 1 for value in thresholds):
        raise ValueError("All IoU thresholds must be in (0, 1]")
    return thresholds


def resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def validate_manifest(path: Path, require_adjudicated: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Validation manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Validation manifest contains no detector/ROI rows")
    for column in REQUIRED_MANIFEST_COLUMNS:
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Validation manifest column '{column}' contains empty values")
    invalid_splits = sorted(set(frame["split"]) - ALLOWED_SPLITS)
    if invalid_splits:
        raise ValueError(f"Unsupported split values: {invalid_splits}")
    invalid_formats = sorted(set(frame["prediction_format"]) - ALLOWED_FORMATS)
    if invalid_formats:
        raise ValueError(f"Unsupported prediction formats: {invalid_formats}")
    if set(frame["coordinate_space"]) != {"crop_level0_pixels"}:
        raise ValueError("Every row must declare coordinate_space=crop_level0_pixels")
    for column in ("mpp_x", "mpp_y"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (~np.isfinite(frame[column]) | (frame[column] <= 0)).any():
            raise ValueError(f"Manifest {column} must contain finite positive values")
    for column in ("image_width_px", "image_height_px"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
        if (frame[column] <= 0).any():
            raise ValueError(f"Manifest {column} must contain positive integers")

    frame["patient_key"] = frame["study_id"] + "::" + frame["patient_id"]
    frame["slide_key"] = frame["study_id"] + "::" + frame["slide_id"]
    patient_splits = frame.groupby("patient_key")["split"].nunique()
    if (patient_splits > 1).any():
        offenders = patient_splits[patient_splits > 1].index[:10].tolist()
        raise ValueError(f"Patient leakage across splits: {offenders}")
    slide_patients = frame.groupby("slide_key")["patient_key"].nunique()
    if (slide_patients > 1).any():
        offenders = slide_patients[slide_patients > 1].index[:10].tolist()
        raise ValueError(f"Slide IDs assigned to multiple patients: {offenders}")

    frame["roi_key"] = (
        frame["study_id"] + "::" + frame["patient_id"] + "::" + frame["slide_id"] + "::" + frame["roi_id"]
    )
    identity_columns = ["roi_key", "detector", "condition"]
    if frame.duplicated(identity_columns).any():
        duplicate = frame.loc[frame.duplicated(identity_columns, keep=False), identity_columns]
        raise ValueError(f"Duplicate detector/condition/ROI rows: {duplicate.head(10).to_dict('records')}")
    invariant_columns = [
        "split", "patient_id", "slide_id", "site", "scanner", "tissue", "compartment",
        "quality_stratum", "mpp_x", "mpp_y", "image_width_px", "image_height_px",
        "coordinate_space", "reference_path",
    ]
    for roi_key, group in frame.groupby("roi_key", sort=False):
        changed = [column for column in invariant_columns if group[column].nunique(dropna=False) != 1]
        if changed:
            raise ValueError(f"ROI metadata differ across detectors for {roi_key}: {changed}")

    if require_adjudicated:
        if "reference_status" not in frame.columns:
            raise ValueError("--require-adjudicated requires manifest column reference_status")
        invalid = frame["reference_status"].str.strip().str.lower().ne("adjudicated")
        if invalid.any():
            raise ValueError("Every evaluated reference must have reference_status=adjudicated")

    frame["reference_path"] = frame["reference_path"].map(lambda value: str(resolve_manifest_path(path, value)))
    frame["prediction_path"] = frame["prediction_path"].map(lambda value: str(resolve_manifest_path(path, value)))
    for column in ("reference_path", "prediction_path"):
        missing_paths = [value for value in frame[column] if not Path(value).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Missing {column} files: {missing_paths[:10]}")
    return frame


def feature_class(properties: dict) -> str | None:
    for key in ("cell_class", "class", "type", "label"):
        value = properties.get(key)
        if value not in (None, ""):
            return str(value)
    classification = properties.get("classification")
    if isinstance(classification, dict) and classification.get("name"):
        return str(classification["name"])
    if isinstance(classification, str) and classification:
        return classification
    return None


def normalize_geometry(geometry, instance_id: str, invalid_policy: str, bounds: tuple[int, int]):
    if geometry.is_empty:
        raise ValueError(f"Instance {instance_id} has empty geometry")
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"Instance {instance_id} must be Polygon/MultiPolygon, got {geometry.geom_type}")
    if not geometry.is_valid:
        if invalid_policy == "repair":
            geometry = geometry.buffer(0)
        else:
            raise ValueError(f"Instance {instance_id} has invalid polygon geometry")
    if geometry.is_empty or geometry.area <= 0 or not geometry.is_valid:
        raise ValueError(f"Instance {instance_id} has unusable polygon geometry")
    width, height = bounds
    xmin, ymin, xmax, ymax = geometry.bounds
    if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        raise ValueError(f"Instance {instance_id} has non-finite coordinates")
    tolerance = 1.0
    if xmin < -tolerance or ymin < -tolerance or xmax > width + tolerance or ymax > height + tolerance:
        raise ValueError(
            f"Instance {instance_id} is outside declared crop bounds {width}x{height}: {geometry.bounds}"
        )
    return geometry


def load_geojson_instances(
    path: Path,
    *,
    invalid_policy: str,
    bounds: tuple[int, int],
    require_adjudicated: bool,
) -> list[Instance]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        raise ValueError(f"Expected GeoJSON FeatureCollection: {path}")
    instances = []
    for index, feature in enumerate(payload["features"], start=1):
        properties = feature.get("properties") or {}
        if require_adjudicated:
            status = str(properties.get("reference_status", properties.get("adjudication_status", ""))).lower()
            if status != "adjudicated":
                raise ValueError(f"Reference feature {index} is not marked adjudicated")
            if not str(properties.get("instance_id", "")).strip():
                raise ValueError(f"Reference feature {index} has no explicit instance_id")
        instance_id = str(
            properties.get("instance_id", properties.get("id", properties.get("object_id", index)))
        )
        geometry = normalize_geometry(
            shape(feature.get("geometry")), instance_id, invalid_policy, bounds
        )
        instances.append(Instance(instance_id, geometry, feature_class(properties)))
    validate_instance_ids(instances, path)
    return instances


def load_cell_json_instances(
    path: Path, *, invalid_policy: str, bounds: tuple[int, int]
) -> list[Instance]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells") if isinstance(payload, dict) else payload
    if isinstance(cells, dict):
        cells = list(cells.values())
    if not isinstance(cells, list):
        raise ValueError(f"Cell JSON must contain a cells list: {path}")
    instances = []
    for index, cell in enumerate(cells, start=1):
        contour = cell.get("contour") or []
        if len(contour) < 3:
            raise ValueError(f"Cell JSON instance {index} has fewer than three contour points")
        instance_id = str(cell.get("id", index))
        geometry = normalize_geometry(Polygon(contour), instance_id, invalid_policy, bounds)
        instances.append(Instance(instance_id, geometry, feature_class(cell)))
    validate_instance_ids(instances, path)
    return instances


def load_label_tiff_instances(path: Path, *, bounds: tuple[int, int], max_pixels: int) -> list[Instance]:
    from skimage.measure import find_contours, regionprops

    labels = tifffile.imread(path)
    if labels.ndim > 2:
        labels = labels[0]
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Label TIFF must be a 2D integer array: {path} shape={labels.shape}")
    if labels.size > int(max_pixels):
        raise ValueError(
            f"Label TIFF has {labels.size:,} pixels, exceeding validation limit {max_pixels:,}. "
            "Use bounded benchmark ROIs or export detector polygons."
        )
    if tuple(labels.shape) != (bounds[1], bounds[0]):
        raise ValueError(
            f"Label TIFF shape {labels.shape} differs from manifest height/width {(bounds[1], bounds[0])}"
        )
    instances = []
    for region in regionprops(labels):
        padded = np.pad(region.image.astype(np.uint8), 1)
        contours = find_contours(padded, 0.5, fully_connected="high")
        if not contours:
            continue
        contour = max(contours, key=len)
        min_row, min_col, _, _ = region.bbox
        coordinates = [
            (float(point[1] - 1 + min_col), float(point[0] - 1 + min_row))
            for point in contour
        ]
        instance_id = str(region.label)
        geometry = normalize_geometry(Polygon(coordinates), instance_id, "fail", bounds)
        instances.append(Instance(instance_id, geometry, None))
    validate_instance_ids(instances, path)
    return instances


def validate_instance_ids(instances: list[Instance], path: Path) -> None:
    ids = [instance.instance_id for instance in instances]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate instance IDs in {path}")


def load_prediction_instances(
    path: Path,
    prediction_format: str,
    *,
    invalid_policy: str,
    bounds: tuple[int, int],
    max_mask_pixels: int,
) -> list[Instance]:
    if prediction_format == "geojson":
        return load_geojson_instances(
            path, invalid_policy=invalid_policy, bounds=bounds, require_adjudicated=False
        )
    if prediction_format == "cell_json":
        return load_cell_json_instances(path, invalid_policy=invalid_policy, bounds=bounds)
    return load_label_tiff_instances(path, bounds=bounds, max_pixels=max_mask_pixels)


def candidate_pairs(reference: list[Instance], prediction: list[Instance]) -> list[Pair]:
    if not reference or not prediction:
        return []
    prediction_geometries = [instance.geometry for instance in prediction]
    tree = STRtree(prediction_geometries)
    object_index = {id(geometry): index for index, geometry in enumerate(prediction_geometries)}
    pairs = []
    for reference_index, reference_instance in enumerate(reference):
        hits = tree.query(reference_instance.geometry)
        for hit in hits:
            prediction_index = int(hit) if isinstance(hit, (int, np.integer)) else object_index[id(hit)]
            prediction_geometry = prediction[prediction_index].geometry
            intersection = float(reference_instance.geometry.intersection(prediction_geometry).area)
            if intersection <= 0:
                continue
            union = float(reference_instance.geometry.area + prediction_geometry.area - intersection)
            iou = intersection / union if union > 0 else 0.0
            dice = 2.0 * intersection / (
                float(reference_instance.geometry.area) + float(prediction_geometry.area)
            )
            pairs.append(Pair(reference_index, prediction_index, intersection, union, iou, dice))
    return pairs


def optimal_matches(pairs: list[Pair], threshold: float) -> list[Pair]:
    eligible = [pair for pair in pairs if pair.iou >= threshold]
    if not eligible:
        return []
    by_reference: dict[int, list[Pair]] = {}
    by_prediction: dict[int, list[Pair]] = {}
    for pair in eligible:
        by_reference.setdefault(pair.reference_index, []).append(pair)
        by_prediction.setdefault(pair.prediction_index, []).append(pair)
    unvisited = set(by_reference)
    matches = []
    while unvisited:
        reference_nodes = {unvisited.pop()}
        prediction_nodes: set[int] = set()
        pending_reference = list(reference_nodes)
        pending_prediction: list[int] = []
        while pending_reference or pending_prediction:
            while pending_reference:
                reference_index = pending_reference.pop()
                for pair in by_reference.get(reference_index, []):
                    if pair.prediction_index not in prediction_nodes:
                        prediction_nodes.add(pair.prediction_index)
                        pending_prediction.append(pair.prediction_index)
            while pending_prediction:
                prediction_index = pending_prediction.pop()
                for pair in by_prediction.get(prediction_index, []):
                    if pair.reference_index not in reference_nodes:
                        reference_nodes.add(pair.reference_index)
                        unvisited.discard(pair.reference_index)
                        pending_reference.append(pair.reference_index)
        reference_list = sorted(reference_nodes)
        prediction_list = sorted(prediction_nodes)
        reference_position = {value: index for index, value in enumerate(reference_list)}
        prediction_position = {value: index for index, value in enumerate(prediction_list)}
        score = np.zeros((len(reference_list), len(prediction_list)), dtype=float)
        pair_lookup = {}
        cardinality_weight = max(len(reference_list), len(prediction_list)) + 1.0
        for pair in eligible:
            if pair.reference_index in reference_position and pair.prediction_index in prediction_position:
                row = reference_position[pair.reference_index]
                column = prediction_position[pair.prediction_index]
                score[row, column] = cardinality_weight + pair.iou
                pair_lookup[(row, column)] = pair
        rows, columns = linear_sum_assignment(-score)
        for row, column in zip(rows, columns):
            if score[row, column] > 0:
                matches.append(pair_lookup[(int(row), int(column))])
    return matches


def boundary_points(geometry, count: int) -> np.ndarray:
    boundary = geometry.boundary
    if boundary.length <= 0:
        return np.asarray([[geometry.centroid.x, geometry.centroid.y]], dtype=float)
    distances = np.linspace(0, boundary.length, max(8, int(count)), endpoint=False)
    return np.asarray([(boundary.interpolate(float(value)).x, boundary.interpolate(float(value)).y) for value in distances])


def boundary_metrics(
    reference_geometry,
    prediction_geometry,
    samples: int,
    mpp_x: float,
    mpp_y: float,
) -> tuple[float, float]:
    reference_points = boundary_points(reference_geometry, samples)
    prediction_points = boundary_points(prediction_geometry, samples)
    scale = np.asarray([mpp_x, mpp_y], dtype=float)
    reference_points *= scale
    prediction_points *= scale
    ref_to_pred = cKDTree(prediction_points).query(reference_points, k=1)[0]
    pred_to_ref = cKDTree(reference_points).query(prediction_points, k=1)[0]
    average_symmetric = float((ref_to_pred.mean() + pred_to_ref.mean()) / 2.0)
    reference_physical = scale_geometry(reference_geometry, xfact=mpp_x, yfact=mpp_y, origin=(0, 0))
    prediction_physical = scale_geometry(prediction_geometry, xfact=mpp_x, yfact=mpp_y, origin=(0, 0))
    return float(reference_physical.hausdorff_distance(prediction_physical)), average_symmetric


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def f1_score(precision: float, recall: float) -> float:
    if precision == 0 or recall == 0:
        return 0.0
    if not math.isfinite(precision) or not math.isfinite(recall):
        return float("nan")
    return float(2 * precision * recall / (precision + recall))


def evaluate_threshold(
    reference: list[Instance],
    prediction: list[Instance],
    pairs: list[Pair],
    threshold: float,
    mpp_x: float,
    mpp_y: float,
    boundary_samples: int,
) -> tuple[dict, list[dict]]:
    matches = optimal_matches(pairs, threshold)
    any_overlap_matches = optimal_matches(pairs, np.nextafter(0.0, 1.0))
    matched_reference = {pair.reference_index for pair in matches}
    matched_prediction = {pair.prediction_index for pair in matches}
    tp = len(matches)
    fp = len(prediction) - tp
    fn = len(reference) - tp
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = f1_score(precision, recall)
    sum_iou = float(sum(pair.iou for pair in matches))
    denominator = tp + 0.5 * fp + 0.5 * fn
    reference_area = float(sum(instance.geometry.area for instance in reference))
    prediction_area = float(sum(instance.geometry.area for instance in prediction))
    aji_intersection = float(sum(pair.intersection for pair in any_overlap_matches))
    aji_union = float(sum(pair.union for pair in any_overlap_matches))
    any_ref = {pair.reference_index for pair in any_overlap_matches}
    any_pred = {pair.prediction_index for pair in any_overlap_matches}
    aji_union += sum(reference[index].geometry.area for index in range(len(reference)) if index not in any_ref)
    aji_union += sum(prediction[index].geometry.area for index in range(len(prediction)) if index not in any_pred)
    centroid_errors = []
    hausdorff_errors = []
    average_boundary_errors = []
    outcomes = []
    for pair in matches:
        reference_instance = reference[pair.reference_index]
        prediction_instance = prediction[pair.prediction_index]
        delta_x = reference_instance.geometry.centroid.x - prediction_instance.geometry.centroid.x
        delta_y = reference_instance.geometry.centroid.y - prediction_instance.geometry.centroid.y
        centroid_um = float(math.hypot(delta_x * mpp_x, delta_y * mpp_y))
        hausdorff_um, average_boundary_um = boundary_metrics(
            reference_instance.geometry,
            prediction_instance.geometry,
            boundary_samples,
            mpp_x,
            mpp_y,
        )
        centroid_errors.append(centroid_um)
        hausdorff_errors.append(hausdorff_um)
        average_boundary_errors.append(average_boundary_um)
        outcomes.append(
            {
                "outcome": "true_positive",
                "reference_id": reference_instance.instance_id,
                "prediction_id": prediction_instance.instance_id,
                "reference_class_unscored": reference_instance.class_label,
                "prediction_class_unscored": prediction_instance.class_label,
                "iou": pair.iou,
                "dice": pair.dice,
                "centroid_error_um": centroid_um,
                "hausdorff_error_um": hausdorff_um,
                "average_symmetric_boundary_error_um": average_boundary_um,
            }
        )
    for index, instance in enumerate(reference):
        if index not in matched_reference:
            outcomes.append(
                {
                    "outcome": "false_negative", "reference_id": instance.instance_id,
                    "prediction_id": None, "reference_class_unscored": instance.class_label,
                    "prediction_class_unscored": None,
                }
            )
    for index, instance in enumerate(prediction):
        if index not in matched_prediction:
            outcomes.append(
                {
                    "outcome": "false_positive", "reference_id": None,
                    "prediction_id": instance.instance_id, "reference_class_unscored": None,
                    "prediction_class_unscored": instance.class_label,
                }
            )
    metrics = {
        "reference_instances": len(reference),
        "predicted_instances": len(prediction),
        "count_error": len(prediction) - len(reference),
        "count_ratio": safe_ratio(len(prediction), len(reference)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection_quality": safe_ratio(tp, denominator),
        "segmentation_quality": safe_ratio(sum_iou, tp),
        "panoptic_quality": safe_ratio(sum_iou, denominator),
        "mean_matched_iou": safe_ratio(sum_iou, tp),
        "mean_matched_dice": safe_ratio(sum(pair.dice for pair in matches), tp),
        "aggregated_jaccard_index": safe_ratio(aji_intersection, aji_union),
        "aggregated_instance_dice": safe_ratio(2 * aji_intersection, reference_area + prediction_area),
        "mean_centroid_error_um": float(np.mean(centroid_errors)) if centroid_errors else float("nan"),
        "median_centroid_error_um": float(np.median(centroid_errors)) if centroid_errors else float("nan"),
        "mean_hausdorff_error_um": float(np.mean(hausdorff_errors)) if hausdorff_errors else float("nan"),
        "mean_average_symmetric_boundary_error_um": (
            float(np.mean(average_boundary_errors)) if average_boundary_errors else float("nan")
        ),
        "sum_matched_iou": sum_iou,
        "sum_aji_intersection": aji_intersection,
        "reference_area_total_px2": reference_area,
        "prediction_area_total_px2": prediction_area,
    }
    return metrics, outcomes


def split_merge_metrics(
    reference: list[Instance], prediction: list[Instance], pairs: list[Pair], overlap_threshold: float
) -> dict:
    reference_degree = np.zeros(len(reference), dtype=np.int64)
    prediction_degree = np.zeros(len(prediction), dtype=np.int64)
    for pair in pairs:
        minimum_area = min(
            float(reference[pair.reference_index].geometry.area),
            float(prediction[pair.prediction_index].geometry.area),
        )
        if safe_ratio(pair.intersection, minimum_area) >= overlap_threshold:
            reference_degree[pair.reference_index] += 1
            prediction_degree[pair.prediction_index] += 1
    split_count = int((reference_degree > 1).sum())
    merge_count = int((prediction_degree > 1).sum())
    return {
        "split_reference_instances": split_count,
        "split_rate": safe_ratio(split_count, len(reference)),
        "merged_prediction_instances": merge_count,
        "merge_rate": safe_ratio(merge_count, len(prediction)),
    }


def metric_from_totals(frame: pd.DataFrame) -> dict[str, float]:
    tp, fp, fn = (float(frame[column].sum()) for column in ("tp", "fp", "fn"))
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = f1_score(precision, recall)
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "panoptic_quality": safe_ratio(float(frame["sum_matched_iou"].sum()), denominator),
        "aggregated_jaccard_index": safe_ratio(
            float(frame["sum_aji_intersection"].sum()),
            float(
                frame["reference_area_total_px2"].sum()
                + frame["prediction_area_total_px2"].sum()
                - frame["sum_aji_intersection"].sum()
            ),
        ),
        "aggregated_instance_dice": safe_ratio(
            2 * float(frame["sum_aji_intersection"].sum()),
            float(frame["reference_area_total_px2"].sum() + frame["prediction_area_total_px2"].sum()),
        ),
    }


def aggregate_metrics(
    per_roi: pd.DataFrame, bootstrap_unit: str, replicates: int, seed: int
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    metrics = [
        "precision", "recall", "f1", "panoptic_quality",
        "aggregated_jaccard_index", "aggregated_instance_dice",
    ]
    for (split, detector, condition, threshold), group in per_roi.groupby(
        ["split", "detector", "condition", "iou_threshold"], sort=True
    ):
        point = metric_from_totals(group)
        units = sorted(group[bootstrap_unit].unique())
        distributions = {metric: [] for metric in metrics}
        if len(units) >= 2 and replicates > 0:
            for _ in range(int(replicates)):
                selected = rng.choice(units, size=len(units), replace=True)
                counts = pd.Series(selected).value_counts()
                parts = []
                for unit, weight in counts.items():
                    part = group[group[bootstrap_unit] == unit].copy()
                    numeric = [
                        "tp", "fp", "fn", "sum_matched_iou", "sum_aji_intersection",
                        "reference_area_total_px2", "prediction_area_total_px2",
                    ]
                    part[numeric] = part[numeric] * int(weight)
                    parts.append(part)
                sampled = pd.concat(parts, ignore_index=True)
                values = metric_from_totals(sampled)
                for metric in metrics:
                    distributions[metric].append(values[metric])
        row = {
            "split": split,
            "detector": detector,
            "condition": condition,
            "iou_threshold": threshold,
            "bootstrap_unit": bootstrap_unit,
            "independent_units": len(units),
            "rois": len(group),
            "reference_instances": int(group["reference_instances"].sum()),
            "predicted_instances": int(group["predicted_instances"].sum()),
            "tp": int(group["tp"].sum()),
            "fp": int(group["fp"].sum()),
            "fn": int(group["fn"].sum()),
        }
        for metric in metrics:
            row[metric] = point[metric]
            finite = np.asarray(distributions[metric], dtype=float)
            finite = finite[np.isfinite(finite)]
            row[f"{metric}_ci_low"] = float(np.percentile(finite, 2.5)) if len(finite) else np.nan
            row[f"{metric}_ci_high"] = float(np.percentile(finite, 97.5)) if len(finite) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def stratified_metrics(per_roi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stratum in ("tissue", "compartment", "quality_stratum", "site", "scanner"):
        group_columns = ["split", "detector", "condition", "iou_threshold", stratum]
        for keys, group in per_roi.groupby(group_columns, sort=True):
            point = metric_from_totals(group)
            rows.append(
                {
                    "split": keys[0],
                    "detector": keys[1],
                    "condition": keys[2],
                    "iou_threshold": keys[3],
                    "stratum_type": stratum,
                    "stratum_value": keys[4],
                    "patients": int(group["patient_id"].nunique()),
                    "slides": int(group["slide_id"].nunique()),
                    "rois": int(group["roi_key"].nunique()),
                    "reference_instances": int(group["reference_instances"].sum()),
                    "predicted_instances": int(group["predicted_instances"].sum()),
                    **point,
                }
            )
    return pd.DataFrame(rows)


def paired_detector_differences(
    per_roi: pd.DataFrame,
    bootstrap_unit: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    metrics = [
        "precision", "recall", "f1", "panoptic_quality",
        "aggregated_jaccard_index", "aggregated_instance_dice",
    ]
    rows = []
    rng = np.random.default_rng(seed + 1009)
    total_columns = [
        "tp", "fp", "fn", "sum_matched_iou", "sum_aji_intersection",
        "reference_area_total_px2", "prediction_area_total_px2",
    ]

    def detector_totals(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return pd.DataFrame({column: frame[f"{column}{suffix}"] for column in total_columns})

    for (split, condition, threshold), split_frame in per_roi.groupby(
        ["split", "condition", "iou_threshold"], sort=True
    ):
        detectors = sorted(split_frame["detector"].unique())
        for left_position, detector_a in enumerate(detectors):
            for detector_b in detectors[left_position + 1 :]:
                left = split_frame[split_frame["detector"] == detector_a]
                right = split_frame[split_frame["detector"] == detector_b]
                paired = left.merge(
                    right,
                    on=["roi_key", bootstrap_unit],
                    suffixes=("_a", "_b"),
                    validate="one_to_one",
                )
                if paired.empty:
                    continue
                totals_a = detector_totals(paired, "_a")
                totals_b = detector_totals(paired, "_b")
                point_a = metric_from_totals(totals_a)
                point_b = metric_from_totals(totals_b)
                units = sorted(paired[bootstrap_unit].unique())
                bootstrap = {metric: [] for metric in metrics}
                if len(units) >= 2 and replicates > 0:
                    for _ in range(int(replicates)):
                        selected = rng.choice(units, size=len(units), replace=True)
                        counts = pd.Series(selected).value_counts()
                        sampled_parts = []
                        for unit, weight in counts.items():
                            part = paired[paired[bootstrap_unit] == unit].copy()
                            for suffix in ("_a", "_b"):
                                columns = [f"{column}{suffix}" for column in total_columns]
                                part[columns] = part[columns] * int(weight)
                            sampled_parts.append(part)
                        sampled = pd.concat(sampled_parts, ignore_index=True)
                        sampled_a = metric_from_totals(detector_totals(sampled, "_a"))
                        sampled_b = metric_from_totals(detector_totals(sampled, "_b"))
                        for metric in metrics:
                            bootstrap[metric].append(sampled_a[metric] - sampled_b[metric])
                for metric in metrics:
                    point = point_a[metric] - point_b[metric]
                    finite = np.asarray(bootstrap[metric], dtype=float)
                    finite = finite[np.isfinite(finite)]
                    rows.append(
                        {
                            "split": split,
                            "condition": condition,
                            "iou_threshold": threshold,
                            "detector_a": detector_a,
                            "detector_b": detector_b,
                            "metric": metric,
                            "direction": "detector_a_minus_detector_b",
                            "paired_rois": int(len(paired)),
                            "detector_a_rois": int(len(left)),
                            "detector_b_rois": int(len(right)),
                            "independent_units": int(len(units)),
                            "detector_a_estimate": point_a[metric],
                            "detector_b_estimate": point_b[metric],
                            "mean_paired_difference": point,
                            "ci_low": float(np.percentile(finite, 2.5)) if len(finite) else np.nan,
                            "ci_high": float(np.percentile(finite, 97.5)) if len(finite) else np.nan,
                        }
                    )
    columns = [
        "split", "condition", "iou_threshold", "detector_a", "detector_b", "metric",
        "direction", "paired_rois", "detector_a_rois", "detector_b_rois",
        "independent_units", "detector_a_estimate", "detector_b_estimate",
        "mean_paired_difference", "ci_low", "ci_high",
    ]
    return pd.DataFrame(rows, columns=columns)


def paired_condition_differences(
    per_roi: pd.DataFrame,
    bootstrap_unit: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    swapped = per_roi.rename(columns={"detector": "_detector", "condition": "detector"})
    swapped = swapped.rename(columns={"_detector": "condition"})
    paired = paired_detector_differences(swapped, bootstrap_unit, replicates, seed + 7919)
    if paired.empty:
        return paired
    paired = paired.rename(
        columns={
            "condition": "detector",
            "detector_a": "condition_a",
            "detector_b": "condition_b",
            "detector_a_estimate": "condition_a_estimate",
            "detector_b_estimate": "condition_b_estimate",
            "detector_a_rois": "condition_a_rois",
            "detector_b_rois": "condition_b_rois",
        }
    )
    paired["direction"] = "condition_a_minus_condition_b"
    return paired


def error_features(
    reference: list[Instance], prediction: list[Instance], outcomes: list[dict], metadata: dict
) -> Iterable[dict]:
    reference_by_id = {instance.instance_id: instance for instance in reference}
    prediction_by_id = {instance.instance_id: instance for instance in prediction}
    for outcome in outcomes:
        if outcome["outcome"] == "false_negative":
            instance = reference_by_id[outcome["reference_id"]]
        elif outcome["outcome"] == "false_positive":
            instance = prediction_by_id[outcome["prediction_id"]]
        else:
            continue
        properties = dict(metadata)
        properties.update(outcome)
        yield {"type": "Feature", "geometry": mapping(instance.geometry), "properties": properties}


def render_html(summary: dict, aggregate: pd.DataFrame) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    def estimate_ci(row: dict, metric: str) -> str:
        value = float(row[metric])
        low = float(row[f"{metric}_ci_low"])
        high = float(row[f"{metric}_ci_high"])
        if math.isfinite(low) and math.isfinite(high):
            return f"{value:.3f} [{low:.3f}, {high:.3f}]"
        return f"{value:.3f} [CI unavailable]"

    table_rows = []
    for row in aggregate.to_dict("records"):
        table_rows.append(
            "<tr>"
            f"<td>{esc(row['split'])}</td><td>{esc(row['detector'])}</td>"
            f"<td>{esc(row['condition'])}</td><td>{row['iou_threshold']:.2f}</td>"
            f"<td>{row['independent_units']}</td>"
            f"<td>{row['precision']:.3f}</td><td>{row['recall']:.3f}</td>"
            f"<td>{estimate_ci(row, 'f1')}</td><td>{estimate_ci(row, 'panoptic_quality')}</td>"
            f"<td>{row['aggregated_jaccard_index']:.3f}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detector reference-standard benchmark</title><style>
body{{font-family:Georgia,serif;margin:0;background:#f3efe5;color:#18323a}}main{{max-width:1180px;margin:auto;padding:42px 24px}}
h1{{font-size:3rem;line-height:1;margin:.2em 0}}.notice{{background:#fffaf0;border-left:8px solid #b84a2b;padding:18px;margin:24px 0}}
table{{width:100%;border-collapse:collapse;background:white;font-family:Arial,sans-serif;font-size:.9rem}}th,td{{padding:10px;border-bottom:1px solid #ccc;text-align:right}}th{{background:#18323a;color:white}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
</style></head><body><main><p>CellPhenotyper validation evidence</p><h1>Detector geometry benchmark</h1>
<div class="notice"><strong>Scope:</strong> This report evaluates instance geometry against adjudicated annotations. Detector phenotype labels are retained for audit but are not reconciled or scored. A successful result is not clinical validation.</div>
	<p>Study: <strong>{esc(summary['study_id'])}</strong>. Detector-condition-ROI rows: {summary['roi_detector_rows']}. Manifest SHA-256: <code>{esc(summary['manifest_sha256'])}</code>.</p>
<table><thead><tr><th>Split</th><th>Detector</th><th>Condition</th><th>IoU</th><th>Units</th><th>Precision</th><th>Recall</th><th>F1 [95% CI]</th><th>PQ [95% CI]</th><th>AJI</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>Required review</h2><p>Inspect <code>error_cases.geojson</code>, confirm reference blinding and adjudication, and interpret confidence intervals at the declared patient/slide unit. Do not tune thresholds on internal or external test splits.</p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.iou_thresholds, args.primary_iou_threshold)
    frame = validate_manifest(manifest_path, args.require_adjudicated)
    reference_cache = {}
    per_roi_rows = []
    outcome_rows = []
    error_geojson_features = []
    provenance_rows = []

    for row in frame.itertuples(index=False):
        bounds = (int(row.image_width_px), int(row.image_height_px))
        reference_path = Path(row.reference_path)
        prediction_path = Path(row.prediction_path)
        reference_key = (str(reference_path), bounds)
        if reference_key not in reference_cache:
            reference_cache[reference_key] = load_geojson_instances(
                reference_path,
                invalid_policy="fail",
                bounds=bounds,
                require_adjudicated=args.require_adjudicated,
            )
        reference = reference_cache[reference_key]
        prediction = load_prediction_instances(
            prediction_path,
            row.prediction_format,
            invalid_policy=args.prediction_invalid_policy,
            bounds=bounds,
            max_mask_pixels=args.max_mask_pixels,
        )
        pairs = candidate_pairs(reference, prediction)
        split_merge = split_merge_metrics(
            reference, prediction, pairs, args.split_merge_overlap
        )
        row_metadata = {
            column: getattr(row, column)
            for column in (
                "study_id", "split", "patient_id", "slide_id", "roi_id", "site", "scanner",
                "tissue", "compartment", "quality_stratum", "detector", "condition",
            )
        }
        row_metadata["roi_key"] = row.roi_key
        row_metadata["patient_key"] = row.patient_key
        row_metadata["slide_key"] = row.slide_key
        for threshold in thresholds:
            metrics, outcomes = evaluate_threshold(
                reference,
                prediction,
                pairs,
                threshold,
                float(row.mpp_x),
                float(row.mpp_y),
                args.boundary_samples,
            )
            per_roi_rows.append(
                {
                    **row_metadata,
                    "iou_threshold": threshold,
                    **metrics,
                    **split_merge,
                }
            )
            if math.isclose(threshold, args.primary_iou_threshold):
                for outcome in outcomes:
                    outcome_rows.append({**row_metadata, "iou_threshold": threshold, **outcome})
                error_geojson_features.extend(
                    error_features(reference, prediction, outcomes, row_metadata)
                )
        provenance_rows.append(
            {
                **row_metadata,
                "reference_path": str(reference_path),
                "reference_sha256": sha256_file(reference_path),
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256_file(prediction_path),
                "prediction_format": row.prediction_format,
                "reference_instances": len(reference),
                "prediction_instances": len(prediction),
            }
        )

    per_roi = pd.DataFrame(per_roi_rows)
    bootstrap_column = "patient_key" if args.bootstrap_unit == "patient_id" else "slide_key"
    aggregate = aggregate_metrics(per_roi, bootstrap_column, args.bootstrap_replicates, args.seed)
    aggregate["bootstrap_unit"] = args.bootstrap_unit
    strata = stratified_metrics(per_roi)
    paired = paired_detector_differences(
        per_roi, bootstrap_column, args.bootstrap_replicates, args.seed
    )
    paired["bootstrap_unit"] = args.bootstrap_unit
    scale_sensitivity = paired_condition_differences(
        per_roi, bootstrap_column, args.bootstrap_replicates, args.seed
    )
    scale_sensitivity["bootstrap_unit"] = args.bootstrap_unit
    per_roi.to_csv(outdir / "detector_metrics_per_roi.csv", index=False)
    aggregate.to_csv(outdir / "detector_metrics_aggregate.csv", index=False)
    strata.to_csv(outdir / "detector_metrics_by_stratum.csv", index=False)
    paired.to_csv(outdir / "detector_paired_differences.csv", index=False)
    scale_sensitivity.to_csv(outdir / "detector_scale_sensitivity.csv", index=False)
    pd.DataFrame(outcome_rows).to_csv(outdir / "instance_outcomes.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(outdir / "detector_benchmark_provenance.csv", index=False)
    (outdir / "error_cases.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "metadata": {
                    "coordinate_space": "crop_level0_pixels",
                    "primary_iou_threshold": args.primary_iou_threshold,
                    "contents": "False-negative reference polygons and false-positive prediction polygons",
                },
                "features": error_geojson_features,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "study_id": ";".join(sorted(frame["study_id"].unique())),
        "intended_evaluation": "reference_standard_instance_segmentation_geometry",
        "phenotype_evaluation": "not_performed_detector_taxonomies_are_not_reconciled",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "roi_detector_rows": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()),
        "slides": int(frame["slide_id"].nunique()),
        "rois": int(frame["roi_key"].nunique()),
        "sites": int(frame["site"].nunique()),
        "scanners": int(frame["scanner"].nunique()),
        "conditions": sorted(frame["condition"].unique()),
        "iou_thresholds": thresholds,
        "primary_iou_threshold": float(args.primary_iou_threshold),
        "matching": "maximum-cardinality_then-IoU one-to-one assignment within overlap components",
        "physical_distance_calculation": "coordinate-wise scaling by mpp_x and mpp_y",
        "split_merge_overlap": float(args.split_merge_overlap),
        "bootstrap_unit": args.bootstrap_unit,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "reference_requirement": "feature-level and manifest-level adjudicated" if args.require_adjudicated else "not enforced",
        "claim_limit": (
            "Metrics establish performance only for the declared reference sample, splits, annotation protocol, "
            "tissues, sites, scanners, and physical resolution. They do not validate phenotype labels or clinical use."
        ),
    }
    (outdir / "detector_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (outdir / "detector_benchmark_report.html").write_text(
        render_html(summary, aggregate), encoding="utf-8"
    )
    print(
        f"[INFO] Detector benchmark complete: rows={len(frame)} rois={summary['rois']} "
        f"detectors={frame['detector'].nunique()} output={outdir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
