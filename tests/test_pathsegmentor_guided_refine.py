from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from pathsegmentor_guided_refine import refine_labels, semantic_prototypes  # noqa: E402


def test_refinement_changes_only_boundary_band_and_preserves_cluster_count() -> None:
    labels = np.ones((20, 20), np.int32)
    labels[:, 10:] = 2
    tissue = np.ones_like(labels, bool)
    probabilities = np.zeros((2, 20, 20), np.float32)
    probabilities[0, :, :12] = 1.0
    probabilities[1, :, 12:] = 1.0
    refined, changed, metadata = refine_labels(
        labels, probabilities, tissue,
        boundary_band_px=3, core_erosion_px=2,
        min_distance_improvement=0.01, assign_unlabeled=False,
    )
    assert np.all(refined[:, :7] == 1)
    assert np.all(refined[:, 13:] == 2)
    assert changed.any()
    assert metadata["cluster_count_before"] == metadata["cluster_count_after"] == 2


def test_grandqc_is_a_hard_support_mask() -> None:
    labels = np.ones((12, 12), np.int32)
    labels[:, 6:] = 2
    tissue = np.ones_like(labels, bool)
    tissue[:2] = False
    probabilities = np.stack([np.ones_like(labels), np.zeros_like(labels)]).astype(np.float32)
    refined, _, _ = refine_labels(
        labels, probabilities, tissue,
        boundary_band_px=2, core_erosion_px=1,
        min_distance_improvement=0.0, assign_unlabeled=True,
    )
    assert np.all(refined[:2] == 0)


def test_unassigned_tissue_is_optional_not_silently_filled() -> None:
    labels = np.ones((15, 15), np.int32)
    labels[:, 8:] = 2
    labels[7, 7] = 0
    tissue = np.ones_like(labels, bool)
    probabilities = np.zeros((2, 15, 15), np.float32)
    probabilities[0, :, :8] = 1
    probabilities[1, :, 8:] = 1
    untouched, _, _ = refine_labels(
        labels, probabilities, tissue,
        boundary_band_px=0, core_erosion_px=1,
        min_distance_improvement=0.01, assign_unlabeled=False,
    )
    filled, _, _ = refine_labels(
        labels, probabilities, tissue,
        boundary_band_px=0, core_erosion_px=1,
        min_distance_improvement=0.01, assign_unlabeled=True,
    )
    assert untouched[7, 7] == 0
    assert filled[7, 7] in {1, 2}


def test_zero_core_erosion_uses_full_cluster_support() -> None:
    labels = np.array([[1, 1], [2, 2]], dtype=np.int32)
    tissue = np.ones_like(labels, dtype=bool)
    probabilities = np.stack((labels == 1, labels == 2)).astype(np.float32)
    cluster_ids, prototypes = semantic_prototypes(labels, probabilities, tissue, 0)
    assert cluster_ids.tolist() == [1, 2]
    assert np.allclose(prototypes, np.eye(2, dtype=np.float32))
