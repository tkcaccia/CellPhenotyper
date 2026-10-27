"""Actual tiny source-bound semantic/H&E pipeline and fail-latched CPU stages."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_export_bora_semantic_project import _fixture, SOURCE_ROOT

pytestmark = pytest.mark.skipif(
    not (SOURCE_ROOT / "bora/wsi_semantic_refinement.py").is_file(),
    reason="Optional Bora semantic integration source unavailable; set BORA_SEMANTIC_SOURCE_ROOT")

from bin import bora_refined_semantic_fullslide_action as action
from bin.export_experiment import validate_experiment


def _inputs(tmp_path: Path):
    pytest.importorskip("cv2")
    pytest.importorskip("shapely")
    item = _fixture(tmp_path, regional_signal=True)
    action._cpu_environment()
    return item, dict(roi_spec_path=item["spec"],
        evaluation_folder=item["evaluation"],
        materialization_folder=item["materialization"],
        registration_path=item["registration"],
        registered_support_tiff=item["support"],
        base_project=item["base"], bora_source=SOURCE_ROOT.parent,
        expected_bora_sha256=action.bora_source_identity(SOURCE_ROOT.parent)["sha256"],
        expected_runtime_sha256=action.identity_digest(action.semantic_runtime_identity()),
        output_folder=tmp_path / "one-slide-result",
        core_pixels=700, source_halo_pixels=96, output_halo_pixels=3,
        memory_budget_bytes=512 * 1024**2,
        disk_budget_bytes=3 * 1024**3, deadline_seconds=600)


def test_actual_full_chain_in_one_folder_opens_as_project(tmp_path: Path):
    item, request = _inputs(tmp_path)
    result = action.run(**request)
    root = request["output_folder"]
    assert result["complete"] is True
    assert result["shard_count"] == 1
    assert result["algorithm_id"] == "bora-wsi-native-he-refined-semantic-chain-v1"
    assert (root / "RUN_COMPLETE.json").is_file()
    assert not (root / "RUN_FAILED.json").exists()
    preflight = json.loads((root / "PREFLIGHT.json").read_text())
    assert preflight["refinement_spec"] == {
        "format": "bora-local-he-boundary-v1",
        "uncertainty_margin": .30,
        "maximum_distance_um": 12.0,
        "he_blur_um": .5,
        "he_edge_sigma": .12,
        "smoothness": 1.5,
        "iterations": 4,
    }
    project = validate_experiment(root / "experiment.cellphenotyper.json")
    artifact = project["tissues"][0]["artifacts"][0]
    assert artifact["role"] == "bora_exploratory_boundary_review"
    assert artifact["review_only"] is True
    assert artifact["training_eligible"] is False
    assert artifact["provenance"]["native_he_boundary_refinement_applied"] is True
    assert "H&E-refined Bora" in project["settings"]["bora_status"]
    mask = np.fromfile(root / "06_refined_global/vectors/exact/mask.uint8.dat",
                       dtype=np.uint8)
    assert set(np.unique(mask)) == {0, 1, 2}
    with pytest.raises(ValueError, match="output folder|overwrite"):
        action.run(**request)


def test_refinement_failure_latches_and_never_admits_next_shard(tmp_path: Path,
                                                                 monkeypatch):
    _, request = _inputs(tmp_path)
    request["core_pixels"] = 100  # Four cores; first failure must stop the rest.
    refiner = action._load_bora(SOURCE_ROOT.parent)[4]
    def fail_first_refinement(*_args, **_kwargs):
        raise RuntimeError("synthetic native H&E stage failure")
    monkeypatch.setattr(refiner, "produce_native_he_refined_semantic_shard",
                        fail_first_refinement)
    with pytest.raises(RuntimeError, match="synthetic native H&E"):
        action.run(**request)
    root = request["output_folder"]
    failure = json.loads((root / "RUN_FAILED.json").read_text())
    assert failure["failed_stage"] == "he_refined_shard_0000"
    assert failure["completed_shards"] == 0
    assert failure["next_stage_blocked"] is True
    assert (root / "03_native_source_shards/0000/SHARD_COMPLETE.json").is_file()
    assert not (root / "03_native_source_shards/0001").exists()
    assert not (root / "06_refined_global").exists()
    assert not (root / "experiment.cellphenotyper.json").exists()
    assert not (root / "RUN_COMPLETE.json").exists()


def test_bad_pin_or_insufficient_he_halo_rejected_before_any_output(tmp_path: Path):
    _, request = _inputs(tmp_path)
    wrong = {**request, "expected_bora_sha256": "0"*64}
    with pytest.raises(ValueError, match="source tree differs"):
        action.run(**wrong)
    too_small = {**request, "source_halo_pixels": 3}
    with pytest.raises(ValueError, match="Source halo"):
        action.run(**too_small)
    assert not request["output_folder"].exists()


def test_failed_or_unsupported_tissue_holdout_cannot_publish_pm016_like_labels(tmp_path: Path):
    item, request = _inputs(tmp_path)
    report_path = item["evaluation"] / "EVALUATION.json"
    report = json.loads(report_path.read_text())
    report["status"] = "same_slide_holdout_failed"
    report["gates"]["no_single_class_collapse"] = False
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="Same-slide expert evaluation"):
        action.run(**request)
    assert not request["output_folder"].exists()
