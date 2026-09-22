import sys
import subprocess
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from labels_to_cluster_mask import (  # noqa: E402
    DEFAULT_PALETTE,
    colorize_cluster_mask,
    load_map,
    load_uncertainty_map,
)


def test_cluster_process_exposes_stability_and_abstention_contract():
    module_text = (ROOT / "modules" / "run_rcode_clustering.nf").read_text()
    config_text = (ROOT / "nextflow.config").read_text()
    script_text = (ROOT / "bin" / "Rcode_Clustering.R").read_text()

    assert "cluster_stability.csv" in module_text
    assert "--stability-runs ${params.cluster_stability_runs}" in module_text
    assert "--auto-selection ${params.cluster_auto_selection ?: 'minimum_abstention'}" in module_text
    assert "--abstain-uncertain ${params.cluster_abstain_uncertain}" in module_text
    assert "cluster_stability_runs         = 3" in config_text
    assert "cluster_auto_selection         = 'minimum_abstention'" in config_text
    assert "cluster_abstain_uncertain      = true" in config_text
    assert "adjusted_rand_index" in script_text
    assert "assignment_vote_margin" in script_text
    assert "interpretable_cluster" in script_text
    assert "cluster_kodama_uncertainty.png" in module_text
    assert "is_abstained = is_abstained" in script_text
    assert "uncertainty_reason = uncertainty_reason" in script_text
    assert "cluster_analysis_role = cluster_analysis_role" in script_text
    assert "forced_cluster_count_requested = forced_cluster_count_requested" in script_text
    assert "claim_status = if (forced_cluster_count_requested) \"sensitivity_only\"" in script_text
    assert "estimated_abstained_count" in script_text
    assert "cluster_resolution_candidates.csv" in module_text
    assert "--observations \"${objects_assigned_csv}\"" in module_text
    assert "grandqc_artifact_kodama_outlier" in script_text


def test_interpretable_cluster_abstentions_map_to_background(tmp_path):
    mapping_path = tmp_path / "clusters.csv"
    pd.DataFrame(
        {
            "label": [1, 2, 3],
            "cluster": [1, 1, 2],
            "interpretable_cluster": [1, None, 2],
        }
    ).to_csv(mapping_path, index=False)

    observed = load_map(str(mapping_path), default_value=0)

    assert observed.to_dict("records") == [
        {"label": 1, "cluster": 1},
        {"label": 2, "cluster": 0},
        {"label": 3, "cluster": 2},
    ]


def test_legacy_cluster_map_remains_supported(tmp_path):
    mapping_path = tmp_path / "legacy.csv"
    pd.DataFrame({"label": [1, 2], "cluster": [3, 4]}).to_csv(mapping_path, index=False)

    observed = load_map(str(mapping_path), default_value=0)

    assert observed.to_dict("records") == [
        {"label": 1, "cluster": 3},
        {"label": 2, "cluster": 4},
    ]


def test_uncertainty_status_codes_are_explicit_and_legacy_rows_are_accepted(tmp_path):
    mapping_path = tmp_path / "uncertainty.csv"
    pd.DataFrame(
        {
            "label": [1, 2, 3, 4, 5],
            "cluster": [1, 1, 2, 2, 3],
            "interpretation_status": [
                "accepted",
                "abstained_ambiguous_assignment",
                "abstained_seed_instability",
                "abstained_ambiguous_assignment_and_seed_instability",
                "unexpected_abstention_reason",
            ],
        }
    ).to_csv(mapping_path, index=False)

    observed = load_uncertainty_map(str(mapping_path))

    assert observed["uncertainty_code"].tolist() == [0, 1, 2, 3, 4]
    legacy_path = tmp_path / "legacy_uncertainty.csv"
    pd.DataFrame({"label": [1, 2], "cluster": [1, 2]}).to_csv(legacy_path, index=False)
    assert load_uncertainty_map(str(legacy_path))["uncertainty_code"].tolist() == [0, 0]

    policy_off_path = tmp_path / "policy_off.csv"
    pd.DataFrame(
        {
            "label": [1],
            "cluster": [2],
            "interpretable_cluster": [2],
            "uncertainty_reason": ["ambiguous_assignment"],
            "is_abstained": [False],
            "interpretation_status": ["abstained_ambiguous_assignment"],
        }
    ).to_csv(policy_off_path, index=False)
    assert load_uncertainty_map(str(policy_off_path))["uncertainty_code"].tolist() == [0]


def test_cluster_palette_is_stable_by_cluster_id_not_observed_subset():
    first = colorize_cluster_mask(np.array([[1, 3]], dtype=np.uint16), default_value=0)
    second = colorize_cluster_mask(np.array([[3]], dtype=np.uint16), default_value=0)

    np.testing.assert_array_equal(first[0, 0], DEFAULT_PALETTE[0])
    np.testing.assert_array_equal(first[0, 1], DEFAULT_PALETTE[2])
    np.testing.assert_array_equal(second[0, 0], DEFAULT_PALETTE[2])


def test_cell_cluster_mask_publishes_separate_abstention_raster(tmp_path):
    labels_path = tmp_path / "labels.tif"
    background_path = tmp_path / "background.tif"
    clusters_path = tmp_path / "clusters.csv"
    cluster_mask_path = tmp_path / "cluster_mask.tif"
    uncertainty_mask_path = tmp_path / "uncertainty_mask.tif"
    cluster_preview_path = tmp_path / "cluster_preview.png"
    uncertainty_preview_path = tmp_path / "uncertainty_preview.png"
    summary_path = tmp_path / "summary.json"
    tifffile.imwrite(
        labels_path,
        np.array([[1, 1, 0, 2], [1, 1, 0, 2]], dtype=np.uint16),
    )
    tifffile.imwrite(
        background_path,
        np.full((2, 4, 3), 190, dtype=np.uint8),
        photometric="rgb",
    )
    pd.DataFrame(
        {
            "label": [1, 2],
            "cluster": [3, 4],
            "interpretable_cluster": [3, None],
            "interpretation_status": ["accepted", "abstained_ambiguous_assignment"],
        }
    ).to_csv(clusters_path, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "bin" / "labels_to_cluster_mask.py"),
            "--mask",
            str(labels_path),
            "--map",
            str(clusters_path),
            "--out",
            str(cluster_mask_path),
            "--uncertainty-out",
            str(uncertainty_mask_path),
            "--summary",
            str(summary_path),
            "--preview",
            str(cluster_preview_path),
            "--uncertainty-preview",
            str(uncertainty_preview_path),
            "--preview-background",
            str(background_path),
        ],
        check=True,
    )

    np.testing.assert_array_equal(
        tifffile.imread(cluster_mask_path),
        np.array([[3, 3, 0, 0], [3, 3, 0, 0]], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        tifffile.imread(uncertainty_mask_path),
        np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.uint8),
    )
    summary = json.loads(summary_path.read_text())
    assert summary["accepted_observations"] == 1
    assert summary["abstained_observations"] == 1
    assert uncertainty_preview_path.stat().st_size > 0
