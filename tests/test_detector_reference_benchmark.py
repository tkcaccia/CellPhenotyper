import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "benchmark_cell_detectors.py"


def load_module():
    pytest.importorskip("shapely")
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location("benchmark_cell_detectors", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def square(x0, y0, x1, y1):
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def write_geojson(path: Path, geometries, *, reference: bool) -> None:
    features = []
    for index, geometry in enumerate(geometries, start=1):
        properties = {"instance_id": f"i{index}", "cell_class": "nucleus"}
        if reference:
            properties["reference_status"] = "adjudicated"
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def make_manifest(tmp_path: Path) -> Path:
    rows = []
    reference_geometries = [square(1.5, 1.5, 7.5, 7.5), square(11.5, 1.5, 17.5, 7.5)]
    for patient_number in (1, 2):
        patient = f"P{patient_number}"
        slide = f"S{patient_number}"
        roi = f"R{patient_number}"
        reference_path = tmp_path / f"{roi}_reference.geojson"
        perfect_path = tmp_path / f"{roi}_perfect.geojson"
        cell_path = tmp_path / f"{roi}_cells.json"
        labels_path = tmp_path / f"{roi}_labels.tif"
        write_geojson(reference_path, reference_geometries, reference=True)
        write_geojson(perfect_path, reference_geometries, reference=False)
        cell_path.write_text(
            json.dumps(
                {
                    "cells": [
                        {"id": "c1", "contour": reference_geometries[0]["coordinates"][0], "type": "epithelial"},
                        {"id": "c2", "contour": reference_geometries[1]["coordinates"][0], "type": "connective"},
                        {"id": "fp", "contour": square(19, 3, 22, 6)["coordinates"][0], "type": "other"},
                    ]
                }
            )
        )
        labels = np.zeros((12, 24), dtype=np.uint16)
        labels[2:8, 2:8] = 1
        labels[2:8, 12:18] = 2
        tifffile.imwrite(labels_path, labels)
        common = {
            "study_id": "detector_test",
            "split": "internal_test",
            "patient_id": patient,
            "slide_id": slide,
            "roi_id": roi,
            "site": "site_a",
            "scanner": "scanner_a",
            "tissue": "breast",
            "compartment": "tumor",
            "quality_stratum": "typical",
            "mpp_x": "0.25",
            "mpp_y": "0.25",
            "image_width_px": "24",
            "image_height_px": "12",
            "coordinate_space": "crop_level0_pixels",
            "reference_status": "adjudicated",
            "reference_path": str(reference_path),
            "condition": "fixed_0.25_mpp",
        }
        rows.extend(
            [
                {**common, "detector": "perfect", "prediction_path": str(perfect_path), "prediction_format": "geojson"},
                {**common, "detector": "cell_json", "prediction_path": str(cell_path), "prediction_format": "cell_json"},
                {**common, "detector": "labels", "prediction_path": str(labels_path), "prediction_format": "labels_tif"},
                {
                    **common,
                    "detector": "perfect",
                    "condition": "recommended_scale",
                    "prediction_path": str(perfect_path),
                    "prediction_format": "geojson",
                },
            ]
        )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_optimal_matching_prioritizes_cardinality_before_iou() -> None:
    module = load_module()
    pairs = [
        module.Pair(0, 0, 1, 1, 0.90, 0.95),
        module.Pair(0, 1, 1, 1, 0.80, 0.89),
        module.Pair(1, 0, 1, 1, 0.80, 0.89),
    ]

    matches = module.optimal_matches(pairs, threshold=0.50)

    assert {(pair.reference_index, pair.prediction_index) for pair in matches} == {(0, 1), (1, 0)}


def test_f1_is_zero_when_reference_cells_have_no_predictions() -> None:
    module = load_module()

    metrics, outcomes = module.evaluate_threshold(
        [module.Instance("reference", module.Polygon([(1, 1), (4, 1), (4, 4), (1, 4)]), None)],
        [],
        [],
        threshold=0.50,
        mpp_x=0.25,
        mpp_y=0.25,
        boundary_samples=16,
    )

    assert metrics["precision"] != metrics["precision"]
    assert metrics["recall"] == 0
    assert metrics["f1"] == 0
    assert [outcome["outcome"] for outcome in outcomes] == ["false_negative"]


def test_physical_errors_use_anisotropic_pixel_sizes() -> None:
    module = load_module()
    reference = module.Instance(
        "reference", module.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)]), None
    )
    prediction = module.Instance(
        "prediction", module.Polygon([(2, 1), (6, 1), (6, 5), (2, 5)]), None
    )
    pairs = module.candidate_pairs([reference], [prediction])

    metrics, _ = module.evaluate_threshold(
        [reference],
        [prediction],
        pairs,
        threshold=0.50,
        mpp_x=2.0,
        mpp_y=0.5,
        boundary_samples=64,
    )

    assert metrics["mean_centroid_error_um"] == pytest.approx(2.0)
    assert metrics["mean_hausdorff_error_um"] == pytest.approx(2.0)


def test_benchmark_cli_scores_all_supported_formats_and_writes_evidence(tmp_path: Path) -> None:
    module = load_module()
    manifest_path = make_manifest(tmp_path)
    outdir = tmp_path / "results"

    old_argv = sys.argv
    try:
        sys.argv = [
            str(SCRIPT),
            "--manifest",
            str(manifest_path),
            "--outdir",
            str(outdir),
            "--iou-thresholds",
            "0.50",
            "--bootstrap-unit",
            "patient_id",
            "--bootstrap-replicates",
            "40",
            "--seed",
            "11",
        ]
        module.main()
    finally:
        sys.argv = old_argv

    per_roi = pd.read_csv(outdir / "detector_metrics_per_roi.csv")
    perfect = per_roi[(per_roi["detector"] == "perfect") & (per_roi["iou_threshold"] == 0.5)]
    cells = per_roi[(per_roi["detector"] == "cell_json") & (per_roi["iou_threshold"] == 0.5)]
    labels = per_roi[(per_roi["detector"] == "labels") & (per_roi["iou_threshold"] == 0.5)]
    assert np.allclose(perfect["panoptic_quality"], 1.0)
    assert set(cells["tp"]) == {2}
    assert set(cells["fp"]) == {1}
    assert set(cells["fn"]) == {0}
    assert set(labels["tp"]) == {2}

    aggregate = pd.read_csv(outdir / "detector_metrics_aggregate.csv")
    assert set(aggregate["independent_units"]) == {2}
    assert aggregate["f1_ci_low"].notna().all()
    strata = pd.read_csv(outdir / "detector_metrics_by_stratum.csv")
    assert set(strata["stratum_type"]) == {
        "tissue", "compartment", "quality_stratum", "site", "scanner"
    }
    paired = pd.read_csv(outdir / "detector_paired_differences.csv")
    comparison = paired[
        (paired["detector_a"] == "cell_json")
        & (paired["detector_b"] == "perfect")
        & (paired["metric"] == "f1")
    ].iloc[0]
    assert comparison["paired_rois"] == 2
    assert comparison["detector_a_rois"] == 2
    assert comparison["detector_b_rois"] == 2
    assert comparison["independent_units"] == 2
    assert comparison["detector_a_estimate"] < comparison["detector_b_estimate"]
    assert comparison["mean_paired_difference"] < 0
    assert np.isfinite(comparison["ci_low"])
    assert np.isfinite(comparison["ci_high"])
    scale = pd.read_csv(outdir / "detector_scale_sensitivity.csv")
    scale_comparison = scale[
        (scale["detector"] == "perfect")
        & (scale["metric"] == "f1")
    ].iloc[0]
    assert scale_comparison["condition_a_rois"] == 2
    assert scale_comparison["condition_b_rois"] == 2
    assert scale_comparison["mean_paired_difference"] == pytest.approx(0)
    outcomes = pd.read_csv(outdir / "instance_outcomes.csv")
    assert "reference_class_unscored" in outcomes
    assert "prediction_class_unscored" in outcomes
    errors = json.loads((outdir / "error_cases.geojson").read_text())
    assert sum(feature["properties"]["outcome"] == "false_positive" for feature in errors["features"]) == 2
    summary = json.loads((outdir / "detector_benchmark_summary.json").read_text())
    assert summary["patients"] == 2
    assert summary["conditions"] == ["fixed_0.25_mpp", "recommended_scale"]
    assert summary["phenotype_evaluation"] == "not_performed_detector_taxonomies_are_not_reconciled"
    assert "clinical" in summary["claim_limit"]
    report = (outdir / "detector_benchmark_report.html").read_text()
    assert len(report) > 1000
    assert "F1 [95% CI]" in report
    provenance = pd.read_csv(outdir / "detector_benchmark_provenance.csv")
    assert provenance["reference_sha256"].str.len().eq(64).all()
    assert provenance["prediction_sha256"].str.len().eq(64).all()


def test_manifest_rejects_patient_leakage(tmp_path: Path) -> None:
    module = load_module()
    manifest_path = make_manifest(tmp_path)
    frame = pd.read_csv(manifest_path)
    patient_rows = frame.index[frame["patient_id"] == "P1"]
    frame.loc[patient_rows, "split"] = "internal_test"
    frame.loc[patient_rows[0], "split"] = "development"
    frame.to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="Patient leakage"):
        module.validate_manifest(manifest_path, require_adjudicated=True)


def test_manifest_namespaces_reused_patient_ids_by_study(tmp_path: Path) -> None:
    module = load_module()
    manifest_path = make_manifest(tmp_path)
    frame = pd.read_csv(manifest_path)
    second_study = frame["patient_id"] == "P2"
    frame.loc[second_study, "study_id"] = "detector_test_external"
    frame.loc[second_study, "patient_id"] = "P1"
    frame.loc[second_study, "split"] = "external_test"
    frame.to_csv(manifest_path, index=False)

    validated = module.validate_manifest(manifest_path, require_adjudicated=True)

    assert validated["patient_key"].nunique() == 2
    assert validated.groupby("patient_key")["split"].nunique().max() == 1


def test_reference_requires_feature_level_adjudication(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "reference.geojson"
    write_geojson(path, [square(1, 1, 4, 4)], reference=False)

    with pytest.raises(ValueError, match="not marked adjudicated"):
        module.load_geojson_instances(
            path,
            invalid_policy="fail",
            bounds=(10, 10),
            require_adjudicated=True,
        )
