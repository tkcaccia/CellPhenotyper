"""Only a source-bound native H&E-refined Bora chain can enter project review."""

from __future__ import annotations

import json
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from bin import export_bora_refined_semantic_project as module
from bin.export_experiment import sha256, validate_experiment
from test_export_bora_semantic_project import _fixture, SOURCE_ROOT

pytestmark = pytest.mark.skipif(
    not (SOURCE_ROOT / "bora/wsi_semantic_refinement.py").is_file(),
    reason="Optional Bora semantic integration source unavailable; set BORA_SEMANTIC_SOURCE_ROOT")
from test_export_bora_continuous_project import _write_json


def _case(root: Path) -> dict:
    item = _fixture(root)
    source_folder = item["shards"][0]
    source_receipt_path = source_folder / "SHARD_COMPLETE.json"
    source_receipt = json.loads(source_receipt_path.read_text())
    refined_folder = root / "refined-shard-0"
    refined_folder.mkdir()
    h, w = item["native"]["height"], item["native"]["width"]
    scores = np.full((h, w, 2), .5, np.float32)
    support = np.load(item["output"] / module.OUTPUT_PATHS["support"]).astype(bool)
    scores[support] = [1, 0]
    scores[30:90, 60:90] = [0, 1]
    np.save(refined_folder / "probability.npy", scores)
    spec = {"format": "bora-local-he-boundary-v1", "test": True}
    receipt = {
        "format": module.SHARD_FORMAT,
        "source_shard_receipt_sha256": sha256(source_receipt_path),
        "source_shard_probability_sha256": source_receipt["probability_sha256"],
        "source_shard_path": str(source_folder),
        "source_input_origin_yx": source_receipt["origin_yx"],
        "source_input_shape_yxc": source_receipt["probability_shape_yxc"],
        "token_model_binding_sha256": source_receipt["token_model_binding_sha256"],
        "source_semantic_module_sha256": sha256(SOURCE_ROOT / "bora/wsi_semantic_probability.py"),
        "source_module_sha256": sha256(SOURCE_ROOT / "bora/wsi_semantic_refinement.py"),
        "registration_sha256": sha256(item["registration"]),
        "native_he_source_sha256": item["native"]["sha256"],
        "tissue_support_sha256": sha256(item["output"] / module.OUTPUT_PATHS["support"]),
        "model_sha256": source_receipt["model_sha256"],
        "class_ids": [1, 2], "class_names": {"1": "stroma", "2": "tumor"},
        "refinement_spec": spec,
        "source_frame_id": item["native"]["frame_id"],
        "source_image_sha256": item["native"]["sha256"],
        "native_shape_yx": [h, w], "native_mpp_xy": [.25, .25],
        "output_halo_pixels": 3,
        "native_he_boundary_refinement_applied": True,
        "expert_annotation_used_at_inference": False,
        "biological_accuracy_qualified": False,
        "partition_invariant_by_finite_halo": True,
        "categorical_scores": "one-hot-inside-support/uniform-outside",
        "core_yxhw": [0, 0, h, w], "origin_yx": [0, 0],
        "required_context_yx": [0, 0],
        "probability_shape_yxc": [h, w, 2],
        "probability_sha256": sha256(refined_folder / "probability.npy"),
    }
    receipt["boundary_refinement"] = {
        "method": "bora-local-he-boundary-v1",
        "class_ids": [1, 2], "mpp_xy": [.25, .25],
        "parameters": spec, "processing_core_size": 512,
        "changed_outside_movable_pixels": 0,
        "partition_invariant_by_finite_halo": True,
        "expert_annotation_used": False,
        "biological_accuracy_validated": False,
    }
    binding = module._binding({field: receipt[field]
                               for field in module.REFINED_BINDING_FIELDS})
    receipt["global_model_binding_sha256"] = binding
    receipt["window"] = {
        "probability_path": str(refined_folder / "probability.npy"),
        "probability_sha256": receipt["probability_sha256"],
        "global_model_binding_sha256": binding,
        "core_yxhw": receipt["core_yxhw"], "origin_yx": [0, 0],
        "independently_fitted": False,
    }
    _write_json(refined_folder / "SHARD_COMPLETE.json", receipt)
    refined_output = root / "refined-global"
    shutil.copytree(item["output"], refined_output)
    for key in ("assembly_receipt", "vector_receipt", "exact_geojson", "review_geojson"):
        path = refined_output / module.OUTPUT_PATHS[key]
        value = json.loads(path.read_text())
        value["global_model_binding_sha256"] = binding
        _write_json(path, value)
    vector_path = refined_output / module.OUTPUT_PATHS["vector_receipt"]
    vectors = json.loads(vector_path.read_text())
    vectors["output_sha256"] = {
        "exact/mask.uint8.dat": sha256(refined_output / module.OUTPUT_PATHS["exact_mask"]),
        "exact/mask.geojson": sha256(refined_output / module.OUTPUT_PATHS["exact_geojson"]),
        "review/mask.geojson": sha256(refined_output / module.OUTPUT_PATHS["review_geojson"]),
    }
    _write_json(vector_path, vectors)
    source_chain = json.loads((item["output"] / "SEMANTIC_CHAIN_COMPLETE.json").read_text())
    chain = dict(source_chain)
    chain.update(format=module.CHAIN_FORMAT,
        native_he_boundary_refinement_applied=True,
        native_he_source_sha256=item["native"]["sha256"],
        global_model_binding_sha256=binding,
        refinement_spec=spec,
        source_shard_receipt_sha256=[sha256(source_receipt_path)],
        refined_shard_receipt_sha256=[sha256(refined_folder / "SHARD_COMPLETE.json")],
        output_sha256={key: sha256(refined_output / relative)
                       for key, relative in module.OUTPUT_PATHS.items()})
    _write_json(refined_output / "REFINED_SEMANTIC_CHAIN_COMPLETE.json", chain)
    return {**item, "refined_folder": refined_folder, "refined_output": refined_output}


def _export(item: dict, destination: Path, *, canonical=False):
    return module.export(item["base"], item["spec"], item["evaluation"],
        item["registration"], item["support"], item["materialization"],
        item["token_folder"], item["token_source"], item["he_token_source"],
        item["shards"], [item["refined_folder"]], item["output"],
        item["refined_output"], SOURCE_ROOT, destination,
        canonical_tissue_annotations=canonical)


@pytest.mark.skipif(not (SOURCE_ROOT / "bora/wsi_semantic_refined_chain.py").is_file(),
                    reason="Staged Bora refined semantic source unavailable")
def test_refined_result_publishes_locked_project_review(tmp_path: Path):
    item = _case(tmp_path)
    destination = tmp_path / "refined-review.cellphenotyper.json"
    report = _export(item, destination)
    artifact = validate_experiment(destination)["tissues"][0]["artifacts"][0]
    assert report["complete"] is True
    assert artifact["role"] == module.ROLE
    assert artifact["review_only"] is True
    assert artifact["training_eligible"] is False
    assert artifact["provenance"]["producer"] == module.CHAIN_FORMAT
    assert artifact["provenance"]["native_he_boundary_refinement_applied"] is True
    assert artifact["provenance"]["biological_accuracy_qualified"] is False
    assert artifact["sha256"] == sha256(item["refined_output"] /
                                        module.OUTPUT_PATHS["review_geojson"])
    assert not any(a["role"] == "tissue_annotations" for a in
                   validate_experiment(destination)["tissues"][0]["artifacts"])


@pytest.mark.parametrize("tamper", ["holdout", "native_he", "materialization",
                                    "refined_scores", "refined_receipt", "chain",
                                    "geojson", "source_shard", "class_collapse"])
def test_refined_result_rejects_wrong_or_changed_sources(tmp_path: Path, tamper: str):
    item = _case(tmp_path)
    if tamper == "holdout":
        path = item["evaluation"] / "EVALUATION.json"
        value = json.loads(path.read_text())
        value["gates"]["per_class_iou"] = False
        _write_json(path, value)
    elif tamper == "native_he":
        with item["native_image"].open("ab") as stream:
            stream.write(b"changed")
    elif tamper == "materialization":
        path = item["materialization"] / "MATERIALIZATION_COMPLETE.json"
        value = json.loads(path.read_text())
        value["no_chunk_gap_or_overlap"] = False
        _write_json(path, value)
    elif tamper == "refined_scores":
        with (item["refined_folder"] / "probability.npy").open("ab") as stream:
            stream.write(b"changed")
    elif tamper == "refined_receipt":
        path = item["refined_folder"] / "SHARD_COMPLETE.json"
        value = json.loads(path.read_text())
        value["native_he_boundary_refinement_applied"] = False
        _write_json(path, value)
    elif tamper == "chain":
        path = item["refined_output"] / "REFINED_SEMANTIC_CHAIN_COMPLETE.json"
        value = json.loads(path.read_text())
        value["format"] = "bora-wsi-continuous-chain-v1"
        _write_json(path, value)
    elif tamper == "geojson":
        with (item["refined_output"] / module.OUTPUT_PATHS["review_geojson"]).open("ab") as stream:
            stream.write(b"changed")
    elif tamper == "source_shard":
        with (item["shards"][0] / "SHARD_COMPLETE.json").open("ab") as stream:
            stream.write(b"changed")
    else:
        mask_path = item["refined_output"] / module.OUTPUT_PATHS["exact_mask"]
        mask = bytearray(mask_path.read_bytes())
        mask[:] = bytes(1 if value else 0 for value in mask)
        mask_path.write_bytes(mask)
        path = item["refined_output"] / "REFINED_SEMANTIC_CHAIN_COMPLETE.json"
        value = json.loads(path.read_text())
        value["output_sha256"]["exact_mask"] = sha256(mask_path)
        _write_json(path, value)
    destination = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError):
        _export(item, destination)
    assert not destination.exists()


def test_canonical_request_is_refused(tmp_path: Path):
    item = _case(tmp_path)
    with pytest.raises(ValueError, match="Independent-patient"):
        _export(item, tmp_path / "canonical.json", canonical=True)


def test_real_bora_refinement_chain_exports_and_reopens(tmp_path: Path):
    """Cross-repository contract: actual Bora shard/global producers, not mock receipts."""
    pytest.importorskip("cv2")
    pytest.importorskip("shapely")
    package = importlib.util.spec_from_file_location(
        "bora", SOURCE_ROOT / "bora/__init__.py",
        submodule_search_locations=[str(SOURCE_ROOT / "bora")])
    bora = importlib.util.module_from_spec(package)
    sys.modules["bora"] = bora
    package.loader.exec_module(bora)
    from bora.wsi_semantic_refinement import produce_native_he_refined_semantic_shard
    from bora.wsi_semantic_refined_chain import run_semantic_refined_global_chain

    item = _fixture(tmp_path)
    probability_path = item["shards"][0] / "probability.npy"
    scores = np.load(probability_path)
    scores[:] = [.99, .01]
    scores[30:90, 60:90] = [.01, .99]
    np.save(probability_path, scores)
    source_receipt_path = item["shards"][0] / "SHARD_COMPLETE.json"
    source_receipt = json.loads(source_receipt_path.read_text())
    source_receipt["probability_sha256"] = sha256(probability_path)
    source_receipt["window"]["probability_sha256"] = sha256(probability_path)
    _write_json(source_receipt_path, source_receipt)
    source_chain_path = item["output"] / "SEMANTIC_CHAIN_COMPLETE.json"
    source_chain = json.loads(source_chain_path.read_text())
    source_chain["shard_receipt_sha256"] = [sha256(source_receipt_path)]
    _write_json(source_chain_path, source_chain)

    refinement_spec = {"format": "bora-local-he-boundary-v1",
        "uncertainty_margin": .9, "maximum_distance_um": 1.,
        "he_blur_um": .25, "he_edge_sigma": .2,
        "smoothness": 2., "iterations": 2}
    refined_shard = tmp_path / "actual-refined-shard"
    produce_native_he_refined_semantic_shard(
        item["shards"][0], item["token_folder"], item["registration"],
        item["native_image"], item["output"] / module.OUTPUT_PATHS["support"],
        sha256(item["output"] / module.OUTPUT_PATHS["support"]),
        refined_shard, refinement_spec, output_halo_pixels=3,
        memory_budget_bytes=512 * 1024**2, disk_budget_bytes=8 * 1024**2)
    refined_global = tmp_path / "actual-refined-global"
    run_semantic_refined_global_chain(
        [refined_shard], item["shards"], item["token_folder"],
        item["evaluation"] / "local_semantic_model.npz",
        item["token_source"], item["he_token_source"],
        item["registration"], item["native_image"], item["support"],
        item["output"] / module.OUTPUT_PATHS["support"], refined_global,
        disk_budget_bytes=8 * 1024**2, memory_budget_bytes=8 * 1024**2)
    destination = tmp_path / "actual-refined.cellphenotyper.json"
    module.export(item["base"], item["spec"], item["evaluation"],
        item["registration"], item["support"], item["materialization"],
        item["token_folder"], item["token_source"], item["he_token_source"],
        item["shards"], [refined_shard], item["output"], refined_global,
        SOURCE_ROOT, destination)
    project = validate_experiment(destination)
    artifact = project["tissues"][0]["artifacts"][0]
    assert artifact["role"] == module.ROLE
    assert artifact["provenance"]["producer"] == module.CHAIN_FORMAT
    assert artifact["provenance"]["native_he_boundary_refinement_applied"] is True
    mask = np.fromfile(refined_global / module.OUTPUT_PATHS["exact_mask"],
                       dtype=np.uint8)
    assert set(np.unique(mask)) == {0, 1, 2}
