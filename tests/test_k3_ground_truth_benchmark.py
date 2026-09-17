import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "audits" / "uni2_resolution_20260909" / "benchmark_k3_ground_truth.py"
SPEC = importlib.util.spec_from_file_location("benchmark_k3_ground_truth", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_optimal_label_mapping_recovers_permutation():
    reference = np.array([[1, 1, 2], [1, 3, 3]], dtype=np.int16)
    prediction = np.array([[8, 8, 4], [8, 7, 7]], dtype=np.int16)
    mapping, contingency = MODULE.optimal_label_mapping(prediction, reference)
    assert mapping == {4: 2, 7: 3, 8: 1}
    assert int(contingency.sum()) == 6


def test_score_candidate_separates_tissue_and_class_error():
    reference = np.array([[1, 1, 0], [2, 3, 0]], dtype=np.int16)
    prediction = np.array([[3, 3, 0], [1, 2, 2]], dtype=np.int16)
    score = MODULE.score_candidate(
        prediction, reference, downsample=4, tolerances_native_px=[0, 4],
    )
    assert score["mean_class_dice"] < 1
    assert score["categorical_accuracy_on_common_tissue"] == 1
    assert score["tissue_iou"] == 0.8


def test_internal_boundary_excludes_tissue_background_edge():
    labels = np.array([[0, 1, 1], [0, 1, 2], [0, 2, 2]], dtype=np.int16)
    boundary = MODULE.internal_boundary(labels)
    assert not boundary[0, 0]
    assert not boundary[1, 0]
    assert boundary[0, 2]
    assert boundary[2, 1]


def test_cached_reference_boundary_metrics_match_uncached():
    reference = np.array([
        [1, 1, 1, 2],
        [1, 1, 2, 2],
        [3, 3, 2, 2],
    ], dtype=np.int16)
    prediction = np.array([
        [7, 7, 4, 4],
        [7, 7, 4, 4],
        [9, 9, 4, 4],
    ], dtype=np.int16)
    tolerances = [0, 4, 8]
    uncached = MODULE.score_candidate(
        prediction, reference, downsample=4, tolerances_native_px=tolerances,
    )
    cache = MODULE.prepare_boundary_reference(
        reference, downsample=4, tolerances_native_px=tolerances,
    )
    cached = MODULE.score_candidate(
        prediction,
        reference,
        downsample=4,
        tolerances_native_px=tolerances,
        boundary_reference_cache=cache,
    )
    assert cached == uncached


def test_angular_geometry_metrics_detects_square_corners():
    payload = {"features": [{"geometry": {"type": "Polygon", "coordinates": [[
        [0, 0], [20, 0], [20, 20], [0, 20], [0, 0],
    ]]}}]}
    metrics = MODULE.angular_geometry_metrics(payload)
    assert metrics["ring_count"] == 1
    assert metrics["segment_count"] == 4
    assert metrics["long_sharp_turn_count_ge_75deg"] == 4
