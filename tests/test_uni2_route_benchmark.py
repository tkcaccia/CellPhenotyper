import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "benchmark_uni2_routes.py"


def load_module():
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location("benchmark_uni2_routes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_manifest(tmp_path: Path) -> Path:
    rows = []
    height = width = 100
    reference = np.ones((height, width), dtype=np.uint16)
    reference[:, 50:] = 2
    tissue = np.ones((height, width), dtype=np.uint8)
    grid = reference.copy()
    cells = np.ones((height, width), dtype=np.uint16)
    cells[:, 42:] = 2
    cells[:10, :10] = 0
    for patient_number in (1, 2, 3):
        patient = f"P{patient_number}"
        slide = f"S{patient_number}"
        roi = f"R{patient_number}"
        reference_path = tmp_path / f"{roi}_reference.tif"
        tissue_path = tmp_path / f"{roi}_tissue.tif"
        grid_path = tmp_path / f"{roi}_grid.tif"
        cells_path = tmp_path / f"{roi}_cells.tif"
        tifffile.imwrite(reference_path, reference)
        tifffile.imwrite(tissue_path, tissue)
        tifffile.imwrite(grid_path, grid)
        tifffile.imwrite(cells_path, cells)
        common = {
            "study_id": "route_test",
            "split": "external_test",
            "patient_id": patient,
            "slide_id": slide,
            "roi_id": roi,
            "site": "external_site",
            "scanner": "scanner_b",
            "tissue": "breast",
            "compartment": "tumor",
            "quality_stratum": "typical",
            "condition": "locked_default",
            "analysis_intent": "tissue_domain_discovery",
            "reference_endpoint": "adjudicated_morphological_domains",
            "reference_status": "adjudicated",
            "reference_path": str(reference_path),
            "tissue_mask_path": str(tissue_path),
            "mask_stage": "medsam_refined",
            "mpp_x": "1.0",
            "mpp_y": "1.0",
            "image_width_px": str(width),
            "image_height_px": str(height),
            "pipeline_commit": "synthetic-test-commit",
            "container_digest": "sha256:synthetic-test-container",
            "uni2_model_revision": "synthetic-test-uni2",
            "clustering_definition": "locked_leiden_v1",
            "primary_metric": "ari_including_abstention",
            "comparison_direction": "grid_minus_cells",
            "acceptance_margin": "0.0",
            "decision_rule": "bootstrap_ci_lower_bound",
        }
        rows.extend(
            [
                {
                    **common,
                    "route": "cells",
                    "prediction_path": str(cells_path),
                    "uni2_feature_definition": "cell_centred_tile_plus_inner_square",
                },
                {
                    **common,
                    "route": "grid",
                    "prediction_path": str(grid_path),
                    "uni2_feature_definition": "overlapping_grid_context_side_by_side_cores",
                },
            ]
        )
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_benchmark(module, manifest: Path, outdir: Path) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [
            str(SCRIPT),
            "--manifest", str(manifest),
            "--outdir", str(outdir),
            "--bootstrap-replicates", "80",
            "--boundary-tolerance-um", "2",
            "--max-evaluation-pixels", "10000",
            "--seed", "19",
        ]
        module.main()
    finally:
        sys.argv = old_argv


def test_route_benchmark_writes_paired_patient_evidence(tmp_path: Path) -> None:
    module = load_module()
    manifest = make_manifest(tmp_path)
    outdir = tmp_path / "results"

    run_benchmark(module, manifest, outdir)

    per_roi = pd.read_csv(outdir / "uni2_route_metrics_per_roi.csv")
    grid = per_roi[per_roi["route"] == "grid"]
    cells = per_roi[per_roi["route"] == "cells"]
    assert np.allclose(grid["ari_including_abstention"], 1.0)
    assert (cells["ari_including_abstention"] < 0.8).all()
    assert np.allclose(grid["coverage_fraction"], 1.0)
    assert (cells["coverage_fraction"] < 1.0).all()
    assert np.allclose(grid["boundary_f1"], 1.0)
    assert np.allclose(cells["boundary_f1"], 0.0)
    patient = pd.read_csv(outdir / "uni2_route_metrics_per_patient.csv")
    assert len(patient) == 6
    aggregate = pd.read_csv(outdir / "uni2_route_metrics_aggregate.csv")
    assert set(aggregate["patients"]) == {3}
    paired = pd.read_csv(outdir / "uni2_route_paired_differences.csv")
    primary = paired[paired["is_primary"] == True].iloc[0]
    assert primary["paired_patients"] == 3
    assert primary["route_difference"] > 0
    assert primary["ci_low"] > 0
    assert primary["decision"] == "passes_prespecified_margin"
    strata = pd.read_csv(outdir / "uni2_route_metrics_by_stratum.csv")
    assert set(strata["stratum_type"]) == {
        "site", "scanner", "tissue", "compartment", "quality_stratum"
    }
    previews = sorted((outdir / "route_qc").glob("*.png"))
    assert len(previews) == 3
    with Image.open(previews[0]) as preview:
        assert preview.width == 300
        assert preview.height == 170
    disagreements = json.loads((outdir / "route_boundary_disagreements.geojson").read_text())
    assert disagreements["features"]
    assert {feature["properties"]["route"] for feature in disagreements["features"]} == {"cells"}
    provenance = pd.read_csv(outdir / "uni2_route_benchmark_provenance.csv")
    for column in ("reference_path_sha256", "prediction_path_sha256", "tissue_mask_path_sha256"):
        assert provenance[column].str.len().eq(64).all()
    summary = json.loads((outdir / "uni2_route_benchmark_summary.json").read_text())
    assert summary["primary_statistical_unit"] == "patient_macro_average"
    assert summary["routes"] == ["cells", "grid"]
    assert "does not name a domain" in summary["claim_limit"]
    report = (outdir / "uni2_route_benchmark_report.html").read_text()
    assert "ARI incl. abstention [95% CI]" in report
    assert "Do not choose the route from a test-set image" in report


def test_manifest_rejects_incomplete_route_pairs(tmp_path: Path) -> None:
    module = load_module()
    path = make_manifest(tmp_path)
    frame = pd.read_csv(path)
    frame = frame[~((frame["patient_id"] == "P1") & (frame["route"] == "cells"))]
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="requires cells and grid"):
        module.validate_manifest(path)


def test_manifest_rejects_patient_leakage(tmp_path: Path) -> None:
    module = load_module()
    path = make_manifest(tmp_path)
    frame = pd.read_csv(path)
    index = frame.index[(frame["patient_id"] == "P1") & (frame["route"] == "cells")][0]
    frame.loc[index, "split"] = "development"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Patient leakage"):
        module.validate_manifest(path)


def test_manifest_rejects_route_metadata_drift(tmp_path: Path) -> None:
    module = load_module()
    path = make_manifest(tmp_path)
    frame = pd.read_csv(path)
    index = frame.index[(frame["patient_id"] == "P1") & (frame["route"] == "cells")][0]
    frame.loc[index, "mpp_x"] = 0.5
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Comparison metadata differ"):
        module.validate_manifest(path)


def test_partition_metrics_are_label_permutation_invariant() -> None:
    module = load_module()
    reference = np.asarray([1, 1, 2, 2, 3, 3])
    prediction = np.asarray([9, 9, 4, 4, 7, 7])

    metrics = module.contingency_metrics(reference, prediction)

    assert metrics["adjusted_rand_index"] == pytest.approx(1.0)
    assert metrics["normalized_mutual_information"] == pytest.approx(1.0)
    assert metrics["variation_of_information"] == pytest.approx(0.0)
