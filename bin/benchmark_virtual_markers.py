#!/usr/bin/env python3
"""Benchmark GigaTIME virtual markers against registered measured protein values."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


REQUIRED_COLUMNS = {
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
    "marker",
    "observation_unit",
    "condition",
    "model_id",
    "model_revision",
    "reference_assay",
    "reference_panel_version",
    "reference_design",
    "registration_status",
    "registration_method",
    "registration_landmarks",
    "registration_median_error_um",
    "registration_p95_error_um",
    "registration_acceptance_p95_um",
    "matched_fraction",
    "minimum_matched_fraction",
    "mpp_x",
    "mpp_y",
    "image_width_px",
    "image_height_px",
    "paired_table_path",
    "pair_id_column",
    "x_column",
    "y_column",
    "predicted_column",
    "reference_column",
    "control_columns",
    "reference_positive_threshold",
    "prediction_positive_threshold",
    "threshold_source",
    "threshold_locked_before_test",
}
NONEMPTY_COLUMNS = REQUIRED_COLUMNS.difference(
    {
        "control_columns",
        "reference_positive_threshold",
        "prediction_positive_threshold",
        "threshold_source",
    }
)
ALLOWED_SPLITS = {"development", "internal_test", "external_test"}
ALLOWED_UNITS = {"cell", "spatial_bin"}
ALLOWED_REFERENCE_DESIGNS = {"same_section_registered", "serial_section_region_level"}
ANALYSIS_COLUMNS = [
    "split", "marker", "observation_unit", "condition", "model_id", "model_revision",
    "reference_assay", "reference_panel_version",
]
METRIC_COLUMNS = [
    "pearson_r",
    "spearman_rho",
    "partial_pearson_r",
    "partial_spearman_rho",
    "rank_mae",
    "z_rmse",
    "auroc",
    "average_precision",
    "sensitivity",
    "specificity",
    "positive_predictive_value",
    "negative_predictive_value",
    "binary_f1",
    "spatial_pearson_r",
    "spatial_spearman_rho",
    "spatial_hotspot_jaccard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GigaTIME marker values against registered measured protein. "
            "The tool reports patient-level uncertainty and does not establish clinical validity."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-observations-per-roi", type=int, default=20)
    parser.add_argument("--max-observations-per-table", type=int, default=5_000_000)
    parser.add_argument("--max-missing-fraction", type=float, default=0.05)
    parser.add_argument("--spatial-bin-um", type=float, default=64.0)
    parser.add_argument("--hotspot-quantile", type=float, default=0.90)
    parser.add_argument("--discordant-fraction", type=float, default=0.01)
    parser.add_argument("--max-discordant-per-row", type=int, default=200)
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


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_control_columns(value: object) -> list[str]:
    raw = str(value).strip()
    if not raw:
        return []
    return [item.strip() for item in raw.replace(",", ";").split(";") if item.strip()]


def validate_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Virtual-marker manifest is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Virtual-marker manifest contains no marker/ROI rows")
    for column in NONEMPTY_COLUMNS:
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Manifest column '{column}' contains empty values")

    invalid_splits = sorted(set(frame["split"]) - ALLOWED_SPLITS)
    if invalid_splits:
        raise ValueError(f"Unsupported split values: {invalid_splits}")
    invalid_units = sorted(set(frame["observation_unit"]) - ALLOWED_UNITS)
    if invalid_units:
        raise ValueError(f"Unsupported observation units: {invalid_units}")
    invalid_designs = sorted(set(frame["reference_design"]) - ALLOWED_REFERENCE_DESIGNS)
    if invalid_designs:
        raise ValueError(f"Unsupported reference designs: {invalid_designs}")
    if frame["registration_status"].str.lower().ne("passed").any():
        raise ValueError("Every evaluated row must have registration_status=passed")
    if (~frame["threshold_locked_before_test"].map(as_bool)).any():
        raise ValueError("Every marker threshold decision must be locked before testing")
    invalid_cell_design = (
        frame["observation_unit"].eq("cell")
        & frame["reference_design"].ne("same_section_registered")
    )
    if invalid_cell_design.any():
        raise ValueError("Cell-level evaluation requires reference_design=same_section_registered")
    non_dapi_without_controls = (
        ~frame["marker"].str.upper().eq("DAPI")
        & frame["control_columns"].map(parse_control_columns).map(len).eq(0)
    )
    if non_dapi_without_controls.any():
        markers = sorted(frame.loc[non_dapi_without_controls, "marker"].unique())
        raise ValueError(f"Non-DAPI markers require cellularity control columns: {markers}")

    numeric_positive = ["mpp_x", "mpp_y"]
    numeric_nonnegative = [
        "registration_landmarks", "registration_median_error_um", "registration_p95_error_um",
        "registration_acceptance_p95_um",
    ]
    for column in numeric_positive + numeric_nonnegative + ["matched_fraction", "minimum_matched_fraction"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (~np.isfinite(frame[column])).any():
            raise ValueError(f"Manifest {column} must contain finite values")
    for column in numeric_positive:
        if (frame[column] <= 0).any():
            raise ValueError(f"Manifest {column} must be positive")
    for column in numeric_nonnegative:
        if (frame[column] < 0).any():
            raise ValueError(f"Manifest {column} must be nonnegative")
    if (frame["registration_landmarks"] < 3).any():
        raise ValueError("At least three independent registration landmarks are required")
    if (frame["registration_p95_error_um"] < frame["registration_median_error_um"]).any():
        raise ValueError("Registration p95 error cannot be lower than median error")
    if (frame["registration_acceptance_p95_um"] <= 0).any():
        raise ValueError("registration_acceptance_p95_um must be positive")
    if (frame["registration_p95_error_um"] > frame["registration_acceptance_p95_um"]).any():
        raise ValueError("Registration p95 error exceeds the prespecified acceptance threshold")
    for column in ("matched_fraction", "minimum_matched_fraction"):
        if ((frame[column] <= 0) | (frame[column] > 1)).any():
            raise ValueError(f"{column} must be in (0, 1]")
    if (frame["matched_fraction"] < frame["minimum_matched_fraction"]).any():
        raise ValueError("Matched fraction is below the prespecified minimum")
    for column in ("image_width_px", "image_height_px"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
        if (frame[column] <= 0).any():
            raise ValueError(f"Manifest {column} must contain positive integers")

    threshold_columns = [
        "reference_positive_threshold", "prediction_positive_threshold", "threshold_source"
    ]
    supplied = frame[threshold_columns].apply(lambda column: column.astype(str).str.strip().ne(""))
    partial_thresholds = supplied.any(axis=1) & ~supplied.all(axis=1)
    if partial_thresholds.any():
        raise ValueError("Binary thresholds require reference, prediction, and source together")
    frame["binary_evaluation"] = supplied.all(axis=1)
    for column in threshold_columns[:2]:
        values = pd.to_numeric(frame.loc[frame["binary_evaluation"], column], errors="raise")
        if (~np.isfinite(values)).any():
            raise ValueError(f"Manifest {column} must be finite when supplied")
    prediction_thresholds = pd.to_numeric(
        frame.loc[frame["binary_evaluation"], "prediction_positive_threshold"], errors="raise"
    )
    if ((prediction_thresholds < 0) | (prediction_thresholds > 1)).any():
        raise ValueError("GigaTIME prediction thresholds must be in [0, 1]")

    frame["patient_key"] = frame["study_id"] + "::" + frame["patient_id"]
    frame["slide_key"] = frame["study_id"] + "::" + frame["slide_id"]
    frame["roi_key"] = (
        frame["patient_key"] + "::" + frame["slide_id"] + "::" + frame["roi_id"]
    )
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
    identity = ["roi_key", "marker", "observation_unit", "condition"]
    if frame.duplicated(identity).any():
        duplicates = frame.loc[frame.duplicated(identity, keep=False), identity]
        raise ValueError(f"Duplicate marker/condition/ROI rows: {duplicates.head(10).to_dict('records')}")
    invariant_columns = [
        "split", "patient_id", "slide_id", "site", "scanner", "tissue", "compartment",
        "quality_stratum",
        "reference_assay", "reference_panel_version", "reference_design", "registration_status",
        "registration_method", "registration_landmarks", "registration_median_error_um",
        "registration_p95_error_um", "registration_acceptance_p95_um", "matched_fraction",
        "minimum_matched_fraction", "mpp_x", "mpp_y", "image_width_px", "image_height_px",
    ]
    for roi_key, group in frame.groupby("roi_key", sort=False):
        changed = [column for column in invariant_columns if group[column].nunique(dropna=False) != 1]
        if changed:
            raise ValueError(f"Reference or acquisition metadata differ within ROI {roi_key}: {changed}")

    frame["paired_table_path"] = frame["paired_table_path"].map(
        lambda value: str(resolve_path(path, value))
    )
    absent = [value for value in frame["paired_table_path"] if not Path(value).is_file()]
    if absent:
        raise FileNotFoundError(f"Missing paired tables: {absent[:10]}")
    return frame


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    result = pearsonr(x, y) if method == "pearson" else spearmanr(x, y)
    return float(result.statistic)


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    if controls.ndim == 1:
        controls = controls[:, None]
    keep = np.ptp(controls, axis=0) > 0
    controls = controls[:, keep]
    if controls.shape[1] == 0:
        return values - np.mean(values)
    design = np.column_stack([np.ones(len(values)), controls])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def partial_correlation(
    prediction: np.ndarray, reference: np.ndarray, controls: np.ndarray, method: str
) -> float:
    if controls.size == 0:
        return float("nan")
    if method == "spearman":
        prediction = rankdata(prediction)
        reference = rankdata(reference)
        controls = np.column_stack([rankdata(controls[:, index]) for index in range(controls.shape[1])])
    pred_residual = residualize(prediction, controls)
    ref_residual = residualize(reference, controls)
    return safe_correlation(pred_residual, ref_residual, "pearson")


def binary_metrics(prediction: np.ndarray, truth: np.ndarray, calls: np.ndarray) -> dict[str, float]:
    positives = truth == 1
    negatives = ~positives
    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())
    if n_positive and n_negative:
        ranks = rankdata(prediction, method="average")
        auroc = float(
            (ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
            / (n_positive * n_negative)
        )
        order = np.argsort(-prediction, kind="mergesort")
        sorted_truth = truth[order]
        distinct_end = np.r_[prediction[order][1:] != prediction[order][:-1], True]
        tp_curve = np.cumsum(sorted_truth)[distinct_end]
        fp_curve = np.cumsum(1 - sorted_truth)[distinct_end]
        recall_curve = tp_curve / n_positive
        precision_curve = tp_curve / np.maximum(tp_curve + fp_curve, 1)
        average_precision = float(np.sum(np.diff(np.r_[0.0, recall_curve]) * precision_curve))
    else:
        auroc = float("nan")
        average_precision = float("nan")
    tp = int(np.sum(calls & positives))
    fp = int(np.sum(calls & negatives))
    fn = int(np.sum(~calls & positives))
    tn = int(np.sum(~calls & negatives))

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "reference_positive_fraction": float(np.mean(positives)),
        "predicted_positive_fraction": float(np.mean(calls)),
        "auroc": auroc,
        "average_precision": average_precision,
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "positive_predictive_value": ratio(tp, tp + fp),
        "negative_predictive_value": ratio(tn, tn + fn),
        "binary_f1": ratio(2 * tp, 2 * tp + fp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def spatial_metrics(
    frame: pd.DataFrame,
    mpp_x: float,
    mpp_y: float,
    bin_um: float,
    hotspot_quantile: float,
) -> dict[str, float]:
    binned = frame.assign(
        _bin_x=np.floor(frame["_x"].to_numpy() * mpp_x / bin_um).astype(np.int64),
        _bin_y=np.floor(frame["_y"].to_numpy() * mpp_y / bin_um).astype(np.int64),
    ).groupby(["_bin_x", "_bin_y"], sort=False)[["_prediction", "_reference"]].mean()
    prediction = binned["_prediction"].to_numpy(dtype=float)
    reference = binned["_reference"].to_numpy(dtype=float)
    if len(binned) < 3:
        return {
            "spatial_bins": int(len(binned)),
            "spatial_pearson_r": float("nan"),
            "spatial_spearman_rho": float("nan"),
            "spatial_hotspot_jaccard": float("nan"),
        }
    pred_hot = prediction >= np.quantile(prediction, hotspot_quantile)
    ref_hot = reference >= np.quantile(reference, hotspot_quantile)
    union = int(np.sum(pred_hot | ref_hot))
    return {
        "spatial_bins": int(len(binned)),
        "spatial_pearson_r": safe_correlation(prediction, reference, "pearson"),
        "spatial_spearman_rho": safe_correlation(prediction, reference, "spearman"),
        "spatial_hotspot_jaccard": float(np.sum(pred_hot & ref_hot) / union) if union else float("nan"),
    }


def evaluate_values(
    frame: pd.DataFrame,
    controls: list[str],
    *,
    binary: bool,
    reference_threshold: float | None,
    prediction_threshold: float | None,
    mpp_x: float,
    mpp_y: float,
    spatial_bin_um: float,
    hotspot_quantile: float,
) -> dict[str, float]:
    prediction = frame["_prediction"].to_numpy(dtype=float)
    reference = frame["_reference"].to_numpy(dtype=float)
    if np.any((prediction < 0) | (prediction > 1)):
        raise ValueError("GigaTIME prediction values must be in [0, 1]")
    control_values = (
        frame[[f"_control_{index}" for index in range(len(controls))]].to_numpy(dtype=float)
        if controls
        else np.empty((len(frame), 0), dtype=float)
    )
    pred_z = (prediction - prediction.mean()) / max(prediction.std(ddof=0), np.finfo(float).eps)
    ref_z = (reference - reference.mean()) / max(reference.std(ddof=0), np.finfo(float).eps)
    pred_rank = rankdata(prediction, method="average") / len(prediction)
    ref_rank = rankdata(reference, method="average") / len(reference)
    metrics = {
        "observations": int(len(frame)),
        "pearson_r": safe_correlation(prediction, reference, "pearson"),
        "spearman_rho": safe_correlation(prediction, reference, "spearman"),
        "partial_pearson_r": partial_correlation(prediction, reference, control_values, "pearson"),
        "partial_spearman_rho": partial_correlation(prediction, reference, control_values, "spearman"),
        "rank_mae": float(np.mean(np.abs(pred_rank - ref_rank))),
        "z_rmse": float(np.sqrt(np.mean((pred_z - ref_z) ** 2))),
        "predicted_min": float(np.min(prediction)),
        "predicted_q01": float(np.quantile(prediction, 0.01)),
        "predicted_median": float(np.median(prediction)),
        "predicted_q99": float(np.quantile(prediction, 0.99)),
        "predicted_max": float(np.max(prediction)),
        "predicted_zero_fraction": float(np.mean(prediction <= 0)),
        "predicted_saturation_fraction": float(np.mean(prediction >= 1)),
        "reference_min": float(np.min(reference)),
        "reference_median": float(np.median(reference)),
        "reference_max": float(np.max(reference)),
        "reference_positive_fraction": float("nan"),
        "predicted_positive_fraction": float("nan"),
        "auroc": float("nan"),
        "average_precision": float("nan"),
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "positive_predictive_value": float("nan"),
        "negative_predictive_value": float("nan"),
        "binary_f1": float("nan"),
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
    }
    if binary:
        truth = (reference >= float(reference_threshold)).astype(np.int8)
        calls = prediction >= float(prediction_threshold)
        metrics.update(binary_metrics(prediction, truth, calls))
    metrics.update(spatial_metrics(frame, mpp_x, mpp_y, spatial_bin_um, hotspot_quantile))
    return metrics


def read_paired_values(
    row,
    *,
    max_rows: int,
    max_missing_fraction: float,
    minimum_observations: int,
) -> tuple[pd.DataFrame, list[str], dict]:
    path = Path(row.paired_table_path)
    controls = parse_control_columns(row.control_columns)
    columns = [
        row.pair_id_column,
        row.x_column,
        row.y_column,
        row.predicted_column,
        row.reference_column,
        *controls,
    ]
    if len(columns) != len(set(columns)):
        raise ValueError(f"Paired-table column roles must be distinct for {row.roi_key}/{row.marker}")
    header = pd.read_csv(path, nrows=0)
    missing_columns = sorted(set(columns).difference(header.columns))
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")
    frame = pd.read_csv(path, usecols=columns)
    if len(frame) > max_rows:
        raise ValueError(f"Paired table {path} has {len(frame):,} rows; limit={max_rows:,}")
    if frame[row.pair_id_column].astype(str).str.strip().eq("").any():
        raise ValueError(f"Paired table {path} contains empty pair IDs")
    if frame[row.pair_id_column].duplicated().any():
        raise ValueError(f"Paired table {path} contains duplicate pair IDs")
    numeric_columns = [row.x_column, row.y_column, row.predicted_column, row.reference_column, *controls]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(numeric.to_numpy()).all(axis=1)
    missing_fraction = float(1.0 - np.mean(complete)) if len(frame) else 1.0
    if missing_fraction > max_missing_fraction:
        raise ValueError(
            f"Paired table {path} missing/nonfinite fraction {missing_fraction:.4f} "
            f"exceeds {max_missing_fraction:.4f}"
        )
    frame = frame.loc[complete].copy()
    numeric = numeric.loc[complete]
    if len(frame) < minimum_observations:
        raise ValueError(
            f"Paired table {path} has {len(frame)} complete observations; minimum={minimum_observations}"
        )
    normalized = pd.DataFrame(
        {
            "_pair_id": frame[row.pair_id_column].astype(str).to_numpy(),
            "_x": numeric[row.x_column].to_numpy(dtype=float),
            "_y": numeric[row.y_column].to_numpy(dtype=float),
            "_prediction": numeric[row.predicted_column].to_numpy(dtype=float),
            "_reference": numeric[row.reference_column].to_numpy(dtype=float),
        }
    )
    for index, column in enumerate(controls):
        normalized[f"_control_{index}"] = numeric[column].to_numpy(dtype=float)
    if (
        (normalized["_x"] < 0).any()
        or (normalized["_x"] >= int(row.image_width_px)).any()
        or (normalized["_y"] < 0).any()
        or (normalized["_y"] >= int(row.image_height_px)).any()
    ):
        raise ValueError(f"Paired-table coordinates are outside declared crop bounds: {path}")
    audit = {
        "source_rows": int(len(complete)),
        "complete_rows": int(len(normalized)),
        "missing_fraction": missing_fraction,
        "control_columns": ";".join(controls),
    }
    return normalized, controls, audit


def patient_metrics(per_roi: pd.DataFrame) -> pd.DataFrame:
    identity = ["study_id", "patient_id", "patient_key", *ANALYSIS_COLUMNS]
    rows = []
    for keys, group in per_roi.groupby(identity, sort=True):
        row = dict(zip(identity, keys))
        row["slides"] = int(group["slide_id"].nunique())
        row["rois"] = int(group["roi_key"].nunique())
        row["observations"] = int(group["observations"].sum())
        row["missing_fraction"] = float(
            np.average(group["missing_fraction"], weights=group["source_rows"])
        )
        for metric in METRIC_COLUMNS + ["reference_positive_fraction", "predicted_positive_fraction"]:
            finite = pd.to_numeric(group[metric], errors="coerce")
            row[metric] = float(finite.mean()) if finite.notna().any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_patient_metrics(patient: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_columns = ANALYSIS_COLUMNS
    rows = []
    for keys, group in patient.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row["patients"] = int(group["patient_key"].nunique())
        row["observations"] = int(group["observations"].sum())
        for metric in METRIC_COLUMNS + ["reference_positive_fraction", "predicted_positive_fraction"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_patients"] = int(len(values))
            row[metric] = float(values.mean()) if len(values) else float("nan")
            bootstrap = []
            if len(values) >= 2 and replicates > 0:
                for _ in range(replicates):
                    bootstrap.append(float(rng.choice(values, size=len(values), replace=True).mean()))
            finite = np.asarray(bootstrap, dtype=float)
            row[f"{metric}_ci_low"] = float(np.percentile(finite, 2.5)) if len(finite) else np.nan
            row[f"{metric}_ci_high"] = float(np.percentile(finite, 97.5)) if len(finite) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def stratified_patient_metrics(per_roi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stratum in ("site", "scanner", "tissue", "compartment", "quality_stratum"):
        columns = [*ANALYSIS_COLUMNS, stratum]
        patient_columns = [*columns, "patient_key"]
        patient_rows = []
        for keys, group in per_roi.groupby(patient_columns, sort=True):
            patient_row = dict(zip(patient_columns, keys))
            patient_row["observations"] = int(group["observations"].sum())
            for metric in METRIC_COLUMNS:
                values = pd.to_numeric(group[metric], errors="coerce")
                patient_row[metric] = float(values.mean()) if values.notna().any() else float("nan")
            patient_rows.append(patient_row)
        stratum_patients = pd.DataFrame(patient_rows)
        for keys, group in stratum_patients.groupby(columns, sort=True):
            row = {
                "split": keys[0],
                "marker": keys[1],
                "observation_unit": keys[2],
                "condition": keys[3],
                "model_id": keys[4],
                "model_revision": keys[5],
                "reference_assay": keys[6],
                "reference_panel_version": keys[7],
                "stratum_type": stratum,
                "stratum_value": keys[8],
                "patients": int(group["patient_key"].nunique()),
                "observations": int(group["observations"].sum()),
            }
            for metric in METRIC_COLUMNS:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[metric] = float(values.mean()) if values.notna().any() else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def discordant_features(
    frame: pd.DataFrame,
    metadata: dict,
    fraction: float,
    maximum: int,
) -> list[dict]:
    prediction_rank = rankdata(frame["_prediction"], method="average") / len(frame)
    reference_rank = rankdata(frame["_reference"], method="average") / len(frame)
    disagreement = np.abs(prediction_rank - reference_rank)
    count = min(maximum, max(1, int(math.ceil(len(frame) * fraction))))
    selected = np.argsort(-disagreement, kind="mergesort")[:count]
    features = []
    for index in selected:
        properties = {
            **metadata,
            "pair_id": str(frame.iloc[index]["_pair_id"]),
            "predicted_value": float(frame.iloc[index]["_prediction"]),
            "reference_value": float(frame.iloc[index]["_reference"]),
            "absolute_percentile_rank_disagreement": float(disagreement[index]),
            "selection": "largest_within_roi_marker_rank_disagreement",
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(frame.iloc[index]["_x"]), float(frame.iloc[index]["_y"])],
                },
                "properties": properties,
            }
        )
    return features


def format_ci(row: dict, metric: str) -> str:
    value = float(row[metric])
    low = float(row[f"{metric}_ci_low"])
    high = float(row[f"{metric}_ci_high"])
    if not math.isfinite(value):
        return "NA"
    if math.isfinite(low) and math.isfinite(high):
        return f"{value:.3f} [{low:.3f}, {high:.3f}]"
    return f"{value:.3f} [CI unavailable]"


def render_html(summary: dict, aggregate: pd.DataFrame) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for row in aggregate.to_dict("records"):
        rows.append(
            "<tr>"
            f"<td>{esc(row['split'])}</td><td>{esc(row['marker'])}</td>"
            f"<td>{esc(row['observation_unit'])}</td><td>{esc(row['condition'])}</td>"
            f"<td>{row['patients']}</td><td>{format_ci(row, 'spearman_rho')}</td>"
            f"<td>{format_ci(row, 'partial_spearman_rho')}</td>"
            f"<td>{format_ci(row, 'spatial_spearman_rho')}</td>"
            f"<td>{format_ci(row, 'auroc')}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Virtual-marker reference benchmark</title><style>
body{{font-family:Georgia,serif;margin:0;background:#f3efe5;color:#18323a}}main{{max-width:1250px;margin:auto;padding:42px 24px}}
h1{{font-size:3rem;line-height:1;margin:.2em 0}}.notice{{background:#fffaf0;border-left:8px solid #b84a2b;padding:18px;margin:24px 0}}
table{{width:100%;border-collapse:collapse;background:white;font-family:Arial,sans-serif;font-size:.86rem}}th,td{{padding:9px;border-bottom:1px solid #ccc;text-align:right}}th{{background:#18323a;color:white}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
</style></head><body><main><p>CellPhenotyper validation evidence</p><h1>Measured-marker concordance</h1>
<div class="notice"><strong>Scope:</strong> This report compares GigaTIME outputs with registered measured protein in the declared sample. Predictions remain virtual markers. Correlation, including spatial correlation, does not establish interchangeable measurement or clinical validity.</div>
<p>Study: <strong>{esc(summary['study_id'])}</strong>. Patients: {summary['patients']}. Marker-condition-ROI rows: {summary['manifest_rows']}. Manifest SHA-256: <code>{esc(summary['manifest_sha256'])}</code>.</p>
<table><thead><tr><th>Split</th><th>Marker</th><th>Unit</th><th>Condition</th><th>Patients</th><th>Spearman [95% CI]</th><th>Cellularity-adjusted Spearman [95% CI]</th><th>Spatial Spearman [95% CI]</th><th>AUROC [95% CI]</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Required review</h2><p>Inspect <code>discordant_observations.geojson</code>, registration records, missingness, marker-specific prevalence and the difference between unadjusted and cellularity-adjusted results. Do not tune thresholds or exclude discordant regions after viewing locked-test results.</p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be nonnegative")
    if args.min_observations_per_roi < 3:
        raise ValueError("--min-observations-per-roi must be at least 3")
    if not 0 <= args.max_missing_fraction < 1:
        raise ValueError("--max-missing-fraction must be in [0, 1)")
    if args.spatial_bin_um <= 0:
        raise ValueError("--spatial-bin-um must be positive")
    if not 0 < args.hotspot_quantile < 1:
        raise ValueError("--hotspot-quantile must be in (0, 1)")
    if not 0 < args.discordant_fraction <= 1:
        raise ValueError("--discordant-fraction must be in (0, 1]")

    manifest_path = Path(args.manifest).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = validate_manifest(manifest_path)
    per_roi_rows = []
    provenance_rows = []
    discordant = []
    hash_cache: dict[str, str] = {}

    for row in manifest.itertuples(index=False):
        paired, controls, audit = read_paired_values(
            row,
            max_rows=args.max_observations_per_table,
            max_missing_fraction=args.max_missing_fraction,
            minimum_observations=args.min_observations_per_roi,
        )
        reference_threshold = (
            float(row.reference_positive_threshold) if row.binary_evaluation else None
        )
        prediction_threshold = (
            float(row.prediction_positive_threshold) if row.binary_evaluation else None
        )
        metrics = evaluate_values(
            paired,
            controls,
            binary=bool(row.binary_evaluation),
            reference_threshold=reference_threshold,
            prediction_threshold=prediction_threshold,
            mpp_x=float(row.mpp_x),
            mpp_y=float(row.mpp_y),
            spatial_bin_um=args.spatial_bin_um,
            hotspot_quantile=args.hotspot_quantile,
        )
        metadata_columns = [
            "study_id", "split", "patient_id", "patient_key", "slide_id", "slide_key",
            "roi_id", "roi_key", "site", "scanner", "tissue", "compartment",
            "quality_stratum", "marker", "observation_unit", "condition", "model_id",
            "model_revision", "reference_assay", "reference_panel_version", "reference_design",
        ]
        metadata = {column: getattr(row, column) for column in metadata_columns}
        per_roi_rows.append(
            {
                **metadata,
                "control_columns": ";".join(controls),
                "binary_evaluation": bool(row.binary_evaluation),
                "registration_median_error_um": float(row.registration_median_error_um),
                "registration_p95_error_um": float(row.registration_p95_error_um),
                "matched_fraction": float(row.matched_fraction),
                **audit,
                **metrics,
            }
        )
        discordant.extend(
            discordant_features(
                paired,
                metadata,
                args.discordant_fraction,
                args.max_discordant_per_row,
            )
        )
        table_path = str(row.paired_table_path)
        if table_path not in hash_cache:
            hash_cache[table_path] = sha256_file(Path(table_path))
        provenance_rows.append(
            {
                **metadata,
                "paired_table_path": table_path,
                "paired_table_sha256": hash_cache[table_path],
                "predicted_column": row.predicted_column,
                "reference_column": row.reference_column,
                "control_columns": ";".join(controls),
                "source_rows": audit["source_rows"],
                "complete_rows": audit["complete_rows"],
            }
        )

    per_roi = pd.DataFrame(per_roi_rows)
    patient = patient_metrics(per_roi)
    aggregate = aggregate_patient_metrics(patient, args.bootstrap_replicates, args.seed)
    strata = stratified_patient_metrics(per_roi)
    per_roi.to_csv(outdir / "virtual_marker_metrics_per_roi.csv", index=False)
    patient.to_csv(outdir / "virtual_marker_metrics_per_patient.csv", index=False)
    aggregate.to_csv(outdir / "virtual_marker_metrics_aggregate.csv", index=False)
    strata.to_csv(outdir / "virtual_marker_metrics_by_stratum.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(outdir / "virtual_marker_provenance.csv", index=False)
    (outdir / "discordant_observations.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "metadata": {
                    "coordinate_space": "crop_level0_pixels",
                    "selection_fraction_per_marker_roi": args.discordant_fraction,
                    "maximum_per_marker_roi": args.max_discordant_per_row,
                    "interpretation": "Review sample only; not an exclusion list.",
                },
                "features": discordant,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "study_id": ";".join(sorted(manifest["study_id"].unique())),
        "intended_evaluation": "registered_measured_protein_concordance",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_rows": int(len(manifest)),
        "patients": int(manifest["patient_key"].nunique()),
        "slides": int(manifest["slide_key"].nunique()),
        "rois": int(manifest["roi_key"].nunique()),
        "markers": sorted(manifest["marker"].unique()),
        "conditions": sorted(manifest["condition"].unique()),
        "primary_statistical_unit": "patient_macro_average",
        "spatial_bin_um": float(args.spatial_bin_um),
        "hotspot_quantile": float(args.hotspot_quantile),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "seed": int(args.seed),
        "cellularity_adjustment": "required for every non-DAPI marker",
        "claim_limit": (
            "Results establish concordance only for the declared paired sample, registration protocol, "
            "assay, markers, tissues, sites, scanners, model revision, and thresholds. Virtual markers "
            "are not interchangeable with measured protein and are not clinically validated."
        ),
    }
    (outdir / "virtual_marker_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (outdir / "virtual_marker_benchmark_report.html").write_text(
        render_html(summary, aggregate), encoding="utf-8"
    )
    print(
        f"[INFO] Virtual-marker benchmark complete: rows={len(manifest)} "
        f"patients={summary['patients']} markers={len(summary['markers'])} output={outdir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
