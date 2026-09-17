import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "bin" / "assess_cluster_interpretation.py"


def load_module():
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location("cluster_interpretation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_spatial_coherence_detects_compact_domains() -> None:
    module = load_module()
    rng = np.random.default_rng(1)
    first = rng.normal(loc=(0, 0), scale=0.2, size=(60, 2))
    second = rng.normal(loc=(10, 10), scale=0.2, size=(60, 2))
    frame = pd.DataFrame(np.vstack([first, second]), columns=["x", "y"])
    frame["assessment_cluster"] = np.repeat([1, 2], 60)
    frame["label_key"] = np.arange(len(frame)).astype(str)

    summary, rows = module.spatial_coherence(frame, k=10, maximum=1000, seed=1)

    assert summary["status"] == "descriptive_only"
    assert summary["mean_same_cluster_neighbor_fraction"] > 0.95
    assert summary["excess_over_prevalence_chance"] > 0.4
    assert set(rows["cluster"]) == {1, 2}


def test_grid_marker_mapping_and_review_packet_keep_key_separate(tmp_path: Path) -> None:
    module = load_module()
    objects = pd.DataFrame(
        {
            "label": ["g0", "g1"],
            "x": [50, 150],
            "y": [50, 50],
            "core_x0": [0, 100],
            "core_y0": [0, 0],
            "core_x1": [100, 200],
            "core_y1": [100, 100],
        }
    )
    clusters = pd.DataFrame(
        {"label": ["g0", "g1"], "cluster": [1, 2], "interpretable_cluster": [1, 2]}
    )
    object_path = tmp_path / "objects.csv"
    cluster_path = tmp_path / "clusters.csv"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    objects.to_csv(object_path, index=False)
    clusters.to_csv(cluster_path, index=False)
    pd.DataFrame(
        {
            "label_id": [1, 2, 3, 4],
            "centroid_x_px": [25, 75, 125, 175],
            "centroid_y_px": [50, 50, 50, 50],
            "DAPI": [0.1, 0.2, 0.8, 0.9],
        }
    ).to_csv(marker_dir / "sample_nuclei_gigatime_mean_intensity.csv", index=False)

    observations, observation_type = module.load_observations(cluster_path, object_path)
    marker_summary, enrichment = module.marker_enrichment(
        observations, observation_type, marker_dir
    )
    review_summary = module.write_blinded_review_packet(
        observations, tmp_path, per_cluster=1, radius=20, seed=1
    )

    assert observation_type == "spatial_grid"
    assert marker_summary["mapped_marker_objects"] == 4
    assert set(enrichment["cluster"]) == {1, 2}
    assert review_summary["status"] == "ready_for_independent_review"
    regions = json.loads((tmp_path / "blinded_review_regions.geojson").read_text())
    assert all("cluster" not in feature["properties"] for feature in regions["features"])
    assert "cluster" not in pd.read_csv(tmp_path / "blinded_review_form.csv").columns
    assert "cluster" in pd.read_csv(tmp_path / "blinded_review_key.csv").columns


def test_nextflow_wires_assessment_after_each_cluster_variant() -> None:
    main = (ROOT / "main.nf").read_text(encoding="utf-8")
    module = (ROOT / "modules" / "assess_cluster_interpretation.nf").read_text(
        encoding="utf-8"
    )

    assert "include { ASSESS_CLUSTER_INTERPRETATION }" in main
    assert "ASSESS_CLUSTER_INTERPRETATION(cluster_assessment_input_ch)" in main
    assert "cluster_variant_defs.collect" in main
    assert "--marker-quant-dir" in module
    assert "cluster_interpretation_summary.json" in module
    assert "cluster_abstentions.geojson" in module
    assert "cluster_spatial_uncertainty.png" in module


def test_abstention_packet_preserves_grid_geometry_and_uncertainty(tmp_path: Path) -> None:
    module = load_module()
    frame = pd.DataFrame(
        {
            "label_key": ["g1", "g2", "g3"],
            "x": [5.0, 15.0, 25.0],
            "y": [5.0, 5.0, 5.0],
            "core_x0": [0, 10, 20],
            "core_y0": [0, 0, 0],
            "core_x1": [10, 20, 30],
            "core_y1": [10, 10, 10],
            "cluster": [1, 1, 2],
            "assessment_cluster": [1, np.nan, np.nan],
            "assignment_vote_fraction": [1.0, 0.6, 0.8],
            "assignment_vote_margin": [1.0, 0.1, 0.5],
            "stability_fraction": [1.0, 1.0, 0.33],
            "interpretation_status": [
                "accepted",
                "abstained_ambiguous_assignment",
                "abstained_seed_instability",
            ],
        }
    )

    geojson_summary = module.write_abstention_geojson(frame, "spatial_grid", tmp_path)
    plot_summary = module.write_spatial_uncertainty_plot(
        frame, "spatial_grid", tmp_path, seed=7
    )

    payload = json.loads((tmp_path / "cluster_abstentions.geojson").read_text())
    assert geojson_summary["abstained_observations"] == 2
    assert len(payload["features"]) == 2
    assert {feature["geometry"]["type"] for feature in payload["features"]} == {"Polygon"}
    assert {
        feature["properties"]["interpretation_status"] for feature in payload["features"]
    } == {"abstained_ambiguous_assignment", "abstained_seed_instability"}
    assert plot_summary["abstained_points_rendered"] == 2
    assert (tmp_path / "cluster_spatial_uncertainty.png").stat().st_size > 0
