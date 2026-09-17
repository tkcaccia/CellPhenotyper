from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from summarize_pathsegmentor_semantics import cluster_summary, summarize  # noqa: E402


def manifest() -> dict:
    return {
        "source_shape_yx": [100, 100],
        "source_mpp": 0.5,
        "output_mpp": 1.0,
        "prompt_panel": {"prompts": [{"id": "tumor"}, {"id": "stroma"}]},
    }


def test_grid_core_semantics_are_spatial_averages() -> None:
    probabilities = np.zeros((2, 50, 50), np.float32)
    probabilities[0, :, :25] = 1.0
    probabilities[1, :, 25:] = 0.5
    observations = pd.DataFrame([
        {"label": 1, "x": 25, "y": 50, "core_x0": 0, "core_y0": 0, "core_x1": 50, "core_y1": 100},
        {"label": 2, "x": 75, "y": 50, "core_x0": 50, "core_y0": 0, "core_x1": 100, "core_y1": 100},
    ])
    result = summarize(probabilities, manifest(), observations, cell_window_um=20)
    assert result.loc[0, "pathsegmentor_tumor_mean"] == pytest.approx(1.0)
    assert result.loc[1, "pathsegmentor_stroma_mean"] == pytest.approx(0.5)


def test_cluster_summary_preserves_kodama_identity() -> None:
    semantic = pd.DataFrame({
        "label": ["1", "2", "3"],
        "pathsegmentor_tumor_mean": [1.0, 0.8, 0.0],
        "pathsegmentor_stroma_mean": [0.0, 0.2, 1.0],
    })
    clusters = pd.DataFrame({"label": [1, 2, 3], "cluster": [1, 1, 2]})
    result = cluster_summary(semantic, clusters)
    assert result["cluster"].tolist() == [1, 2]
    assert result.loc[0, "pathsegmentor_tumor_mean"] == pytest.approx(0.9)
    assert set(result["semantic_role"]) == {"descriptive_supervised_evidence_not_cluster_ground_truth"}
