import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "benchmark_virtual_markers.py"


def load_module():
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location("benchmark_virtual_markers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_manifest(tmp_path: Path) -> Path:
    rows = []
    for patient_number in (1, 2, 3):
        rng = np.random.default_rng(100 + patient_number)
        patient = f"P{patient_number}"
        slide = f"S{patient_number}"
        roi = f"R{patient_number}"
        count = 64
        control = np.linspace(0.05, 0.95, count)
        latent = np.tile(np.linspace(-0.25, 0.25, 8), 8)
        measured_cd3 = np.clip(0.45 * control + latent + 0.25, 0, 1)
        predicted_cd3 = np.clip(0.45 * control + 0.92 * latent + 0.25 + rng.normal(0, 0.01, count), 0, 1)
        measured_dapi = np.clip(control + rng.normal(0, 0.01, count), 0, 1)
        predicted_dapi = np.clip(control + rng.normal(0, 0.015, count), 0, 1)
        paired_path = tmp_path / f"{roi}_paired.csv"
        pd.DataFrame(
            {
                "pair_id": [f"{roi}_cell_{index}" for index in range(count)],
                "x_px": np.tile(np.arange(8) * 100 + 50, 8),
                "y_px": np.repeat(np.arange(8) * 100 + 50, 8),
                "predicted_DAPI": predicted_dapi,
                "measured_DAPI": measured_dapi,
                "predicted_CD3": predicted_cd3,
                "measured_CD3": measured_cd3,
                "measured_DAPI_control": measured_dapi,
                "nucleus_area_px": 80 + 20 * control,
            }
        ).to_csv(paired_path, index=False)
        common = {
            "study_id": "virtual_marker_test",
            "split": "external_test",
            "patient_id": patient,
            "slide_id": slide,
            "roi_id": roi,
            "site": "external_site",
            "scanner": "scanner_b",
            "tissue": "breast",
            "compartment": "tumor",
            "quality_stratum": "typical",
            "observation_unit": "cell",
            "condition": "released_model",
            "model_id": "prov-gigatime/GigaTIME",
            "model_revision": "synthetic-test-revision",
            "reference_assay": "registered_mIF",
            "reference_panel_version": "panel_v1",
            "reference_design": "same_section_registered",
            "registration_status": "passed",
            "registration_method": "locked_affine_plus_deformable",
            "registration_landmarks": "20",
            "registration_median_error_um": "0.4",
            "registration_p95_error_um": "0.9",
            "registration_acceptance_p95_um": "1.0",
            "matched_fraction": "0.95",
            "minimum_matched_fraction": "0.90",
            "mpp_x": "0.25",
            "mpp_y": "0.25",
            "image_width_px": "800",
            "image_height_px": "800",
            "paired_table_path": str(paired_path),
            "pair_id_column": "pair_id",
            "x_column": "x_px",
            "y_column": "y_px",
            "threshold_locked_before_test": "true",
        }
        rows.extend(
            [
                {
                    **common,
                    "marker": "DAPI",
                    "predicted_column": "predicted_DAPI",
                    "reference_column": "measured_DAPI",
                    "control_columns": "",
                    "reference_positive_threshold": "",
                    "prediction_positive_threshold": "",
                    "threshold_source": "",
                },
                {
                    **common,
                    "marker": "CD3",
                    "predicted_column": "predicted_CD3",
                    "reference_column": "measured_CD3",
                    "control_columns": "measured_DAPI_control;nucleus_area_px",
                    "reference_positive_threshold": "0.5",
                    "prediction_positive_threshold": "0.5",
                    "threshold_source": "development_protocol_v1",
                },
            ]
        )
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_benchmark(module, manifest_path: Path, outdir: Path) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [
            str(SCRIPT),
            "--manifest",
            str(manifest_path),
            "--outdir",
            str(outdir),
            "--bootstrap-replicates",
            "80",
            "--min-observations-per-roi",
            "20",
            "--spatial-bin-um",
            "50",
            "--discordant-fraction",
            "0.05",
            "--seed",
            "17",
        ]
        module.main()
    finally:
        sys.argv = old_argv


def test_virtual_marker_cli_writes_patient_level_evidence(tmp_path: Path) -> None:
    module = load_module()
    manifest_path = make_manifest(tmp_path)
    outdir = tmp_path / "results"

    run_benchmark(module, manifest_path, outdir)

    per_roi = pd.read_csv(outdir / "virtual_marker_metrics_per_roi.csv")
    cd3 = per_roi[per_roi["marker"] == "CD3"]
    assert len(cd3) == 3
    assert (cd3["partial_spearman_rho"] > 0.85).all()
    assert (cd3["auroc"] > 0.90).all()
    assert (cd3["spatial_bins"] >= 10).all()
    patients = pd.read_csv(outdir / "virtual_marker_metrics_per_patient.csv")
    assert len(patients) == 6
    assert patients.groupby(["marker", "patient_key"]).size().max() == 1
    aggregate = pd.read_csv(outdir / "virtual_marker_metrics_aggregate.csv")
    cd3_aggregate = aggregate[aggregate["marker"] == "CD3"].iloc[0]
    assert cd3_aggregate["patients"] == 3
    assert np.isfinite(cd3_aggregate["partial_spearman_rho_ci_low"])
    assert np.isfinite(cd3_aggregate["partial_spearman_rho_ci_high"])
    strata = pd.read_csv(outdir / "virtual_marker_metrics_by_stratum.csv")
    assert set(strata["stratum_type"]) == {
        "site", "scanner", "tissue", "compartment", "quality_stratum"
    }
    discordant = json.loads((outdir / "discordant_observations.geojson").read_text())
    assert len(discordant["features"]) == 24
    assert discordant["metadata"]["interpretation"] == "Review sample only; not an exclusion list."
    provenance = pd.read_csv(outdir / "virtual_marker_provenance.csv")
    assert provenance["paired_table_sha256"].str.len().eq(64).all()
    summary = json.loads((outdir / "virtual_marker_benchmark_summary.json").read_text())
    assert summary["patients"] == 3
    assert summary["primary_statistical_unit"] == "patient_macro_average"
    assert "not interchangeable" in summary["claim_limit"]
    report = (outdir / "virtual_marker_benchmark_report.html").read_text()
    assert "Cellularity-adjusted Spearman [95% CI]" in report
    assert "Predictions remain virtual markers" in report


def test_manifest_rejects_patient_leakage(tmp_path: Path) -> None:
    module = load_module()
    path = make_manifest(tmp_path)
    frame = pd.read_csv(path)
    first_patient = frame.index[frame["patient_id"] == "P1"]
    frame.loc[first_patient[0], "split"] = "development"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Patient leakage"):
        module.validate_manifest(path)


def test_manifest_rejects_cell_level_serial_sections(tmp_path: Path) -> None:
    module = load_module()
    path = make_manifest(tmp_path)
    frame = pd.read_csv(path)
    frame["reference_design"] = "serial_section_region_level"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Cell-level evaluation requires"):
        module.validate_manifest(path)


def test_manifest_requires_non_dapi_cellularity_controls(tmp_path: Path) -> None:
    module = load_module()
    path = make_manifest(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[frame["marker"] == "CD3", "control_columns"] = ""
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="require cellularity control"):
        module.validate_manifest(path)


def test_manifest_enforces_registration_and_matching_gates(tmp_path: Path) -> None:
    module = load_module()
    path = make_manifest(tmp_path)
    frame = pd.read_csv(path)
    frame["registration_p95_error_um"] = 1.1
    frame["matched_fraction"] = 0.80
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Registration p95 error exceeds"):
        module.validate_manifest(path)

    frame["registration_p95_error_um"] = 0.9
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Matched fraction is below"):
        module.validate_manifest(path)


def test_paired_table_rejects_out_of_bounds_coordinates(tmp_path: Path) -> None:
    module = load_module()
    path = make_manifest(tmp_path)
    manifest = module.validate_manifest(path)
    paired_path = Path(manifest.iloc[0]["paired_table_path"])
    paired = pd.read_csv(paired_path)
    paired.loc[0, "x_px"] = 900
    paired.to_csv(paired_path, index=False)

    with pytest.raises(ValueError, match="outside declared crop bounds"):
        module.read_paired_values(
            next(manifest.itertuples(index=False)),
            max_rows=1000,
            max_missing_fraction=0.05,
            minimum_observations=20,
        )
