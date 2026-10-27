"""Frozen expert/model/shard provenance gates for a review-only project revision."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_export_bora_continuous_project import _case, _write_json

from bin import export_bora_semantic_project as module
from bin import materialize_bora_semantic_tokens as materializer
from bin.export_experiment import sha256, validate_experiment


SOURCE_ROOT = Path(os.environ.get("BORA_SEMANTIC_SOURCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "Bora" / "src")))
pytestmark = pytest.mark.skipif(
    not (SOURCE_ROOT / "bora" / "wsi_semantic_chain.py").is_file(),
    reason="Optional Bora semantic integration source unavailable; set BORA_SEMANTIC_SOURCE_ROOT")


def _upgrade_frames(item):
    native, working = item["native"], item["working"]
    working["mpp_x"] = native["width"] * native["mpp_x"] / working["width"]
    working["mpp_y"] = native["height"] * native["mpp_y"] / working["height"]
    for frame in (native, working):
        identity = {key: frame[key] for key in (
            "sha256", "width", "height", "mpp_x", "mpp_y")}
        identity.update(origin_x=0, origin_y=0,
            coordinate_space="crop-native-pixels",
            coordinate_convention="pixel-edge-origin_centres-in-native-pixels")
        frame["frame_id"] = "cellvit-frame-" + hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        frame.update(identity)
    registration = json.loads(item["registration"].read_text())
    registration["cell_full_image"] = native
    registration["image"] = working
    registration["tissue_to_cell_full_affine"] = [
        [native["width"] / working["width"], 0, 0],
        [0, native["height"] / working["height"], 0], [0, 0, 1]]
    _write_json(item["registration"], registration)
    base = json.loads(item["base"].read_text())
    base["tissues"][0]["image"]["frame_id"] = native["frame_id"]
    _write_json(item["base"], base)
    for relative in ("CONTINUOUS_CHAIN_COMPLETE.json",
                     *module.OUTPUT_PATHS.values()):
        path = item["output"] / relative
        if not path.name.endswith(".json") and not path.name.endswith(".geojson"):
            continue
        value = json.loads(path.read_text())
        value["source_frame_id"] = native["frame_id"]
        if path.name == "CONTINUOUS_CHAIN_COMPLETE.json":
            value["registration_sha256"] = sha256(item["registration"])
        _write_json(path, value)
    return registration


def _materialized_inputs(root, item, source_hashes, *, regional_signal=False):
    package = sys.modules.get("bora")
    if package is None or Path(package.__file__).resolve() != \
            (SOURCE_ROOT / "bora/__init__.py").resolve():
        for name in tuple(sys.modules):
            if name == "bora" or name.startswith("bora."):
                sys.modules.pop(name)
        spec = importlib.util.spec_from_file_location(
            "bora", SOURCE_ROOT / "bora/__init__.py",
            submodule_search_locations=[str(SOURCE_ROOT / "bora")])
        package = importlib.util.module_from_spec(spec)
        sys.modules["bora"] = package
        spec.loader.exec_module(package)
    from bora.chunk_plan import make_chunk_plan
    from bora.chunk_store import relative_name

    working = item["working"]
    wh, ww = working["height"], working["width"]
    from PIL import Image
    with Image.open(item["working_image"]) as image:
        he = np.asarray(image.convert("RGB"))
    he_path = root / "working-he.npy"
    np.save(he_path, he)
    evidence_path = root / "he-evidence.json"
    _write_json(evidence_path, {
        "format": "coordinator-bora-global-evidence-v1",
        "source_sha256": {"registration": sha256(item["registration"])},
        "output_sha256": {"working_he": sha256(he_path)},
        "working_frame_id": working["frame_id"],
        "working_image_sha256": working["sha256"],
        "working_shape_yx": [wh, ww]})
    checkpoint = root / "virchow2-checkpoint.bin"
    checkpoint.write_bytes(b"synthetic frozen checkpoint")
    plan = make_chunk_plan(wh, ww)
    chunk_root = root / "chunks"
    chunk_root.mkdir()
    hashes = {}
    for case in plan["cases"]:
        chunk = np.full(case["chunk_shape_yxc"], case["scale_factor"], np.float32)
        if regional_signal and case["scale_factor"] == 1:
            # A frozen synthetic left/right token signal for a real producer
            # chain, with all three chunk scales still materialized/hashed.
            chunk[..., 0] = 0
            chunk[:, 2:, 0] = 2
        path = chunk_root / relative_name(case)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, chunk)
        hashes[str(case["case_id"])] = sha256(path)
    inventory_path = chunk_root / "CHUNK_INVENTORY.json"
    _write_json(inventory_path, {
        "format": "coordinator-bora-token-chunk-inventory-v1",
        "plan": plan, "sha256_by_case": hashes,
        "working_frame_id": working["frame_id"],
        "working_image_sha256": working["sha256"],
        "checkpoint_sha256": sha256(checkpoint),
        "completed_case_count": plan["case_count"],
        "all_cleanup_verified": True, "one_global_absolute_field": True})
    material_spec_path = root / "materialization-spec.json"
    material_inputs = {"registration": item["registration"],
        "native_image": item["native_image"],
        "working_image": item["working_image"],
        "working_he": he_path, "he_evidence": evidence_path,
        "token_inventory": inventory_path,
        "virchow2_checkpoint": checkpoint}
    _write_json(material_spec_path, {"format": materializer.SPEC_FORMAT,
        "bora_source_root": str(SOURCE_ROOT),
        "bora_source_sha256": source_hashes,
        "inputs": {key: {"path": str(path), "sha256": sha256(path)}
                   for key, path in material_inputs.items()}})
    folder = root / "materialized"
    materializer.materialize(material_spec_path, folder,
        memory_budget_bytes=16 * 1024**2, disk_budget_bytes=10 * 1024**2)
    return material_inputs, folder


def _fixture(root: Path, *, regional_signal=False) -> dict:
    if not (SOURCE_ROOT / "bora/wsi_semantic_chain.py").is_file():
        pytest.skip("Optional Bora integration source is unavailable")
    # The arithmetic holdout fixture declares 320 tokens; give it a physical
    # token grid large enough for separate training and holdout observations.
    item = _case(root, visible_geometry=True, native_size=(700, 700))
    registration = _upgrade_frames(item)
    output = item["output"]
    classes = {"1": "stroma", "2": "tumor"}
    source_hashes = {name: sha256(SOURCE_ROOT / "bora" / name)
                     for name in module.SOURCE_MODULES}
    material_inputs, material_folder = _materialized_inputs(
        root, item, source_hashes, regional_signal=regional_signal)
    frame = {"format": "virchow2-spatial-tokens-v1",
             "encoder_sha256": sha256(material_inputs["virchow2_checkpoint"]),
             "working_mpp_xy": [item["working"]["mpp_x"], item["working"]["mpp_y"]],
             "token_stride_yx": [14, 14],
             "token_origin_working_yx": [0.0, 0.0],
             "channel_order": "YXC-1280"}
    evaluation_folder = root / "evaluation"
    evaluation_folder.mkdir()
    model = evaluation_folder / "local_semantic_model.npz"
    coefficient = np.zeros((1, 1283))
    coefficient[0, 0] = 10 if regional_signal else 0
    np.savez_compressed(model, metadata=json.dumps({
        "format": "bora-local-semantic-token-v2", "token_channels": 1280,
        "classifier": "balanced-l2-logistic-regression-cpu",
        "training_labels_included": False, "he_enabled": True,
        "composition_schema": None, "class_names": classes,
        "frame": frame,
        "training_specimen_groups": ["synthetic-training"],
        "reserved_reference_groups": ["synthetic-holdout"],
        "training_patient_groups": ["synthetic-patient"],
        "reserved_reference_patient_groups": [],
        "training_counts": {"1": 100, "2": 100},
        "exclusions": {}, "solver_iterations": 1}),
        mean=np.zeros(1283), scale=np.ones(1283), coef=coefficient,
        intercept=np.array([-18.6 if regional_signal else 0.]),
        classes=np.array([1, 2]))
    model_sha = sha256(model)
    inputs = {"registration": item["registration"],
              "native_image": item["native_image"],
              "working_image": item["working_image"],
              **{name: material_inputs[name] for name in (
                  "working_he", "he_evidence", "token_inventory",
                  "virchow2_checkpoint")}}
    roi_rows = [
        ("train-stroma", 1, "train", "stroma-train", 100),
        ("train-tumor", 2, "train", "tumor-train", 100),
        ("holdout-stroma-a", 1, "holdout", "stroma-a", 80),
        ("holdout-stroma-b", 1, "holdout", "stroma-b", 80),
        ("holdout-tumor-a", 2, "holdout", "tumor-a", 80),
        ("holdout-tumor-b", 2, "holdout", "tumor-b", 80),
    ]
    expert_features = [{"type": "Feature", "properties": {
        "annotation_id": aid, "class_id": cid, "split": split,
        "roi_id": roi, "annotator_id": "synthetic-expert",
        "annotation_source": "independent_expert"},
        "geometry": {"type": "Polygon", "coordinates": [[
            [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}}
        for aid, cid, split, roi, _count in roi_rows]
    for name in module.REQUIRED_INPUTS:
        if name in inputs:
            continue
        path = root / (name + ".json")
        if name == "expert_geojson":
            _write_json(path, {"type": "FeatureCollection",
                "schema": "cellphenotyper-expert-tissue-v1",
                "class_names": classes,
                "source": {"source_frame_id": item["native"]["frame_id"],
                    "image_sha256": item["native"]["sha256"],
                    "native_shape_yx": [item["native"]["height"], item["native"]["width"]],
                    "native_mpp_xy": [.25, .25],
                    "token_inventory_sha256": sha256(material_inputs["token_inventory"]),
                    "coordinate_space": "native_image_pixel_edges"},
                "features": expert_features})
        else:
            path.write_text(name)
        inputs[name] = path
    spec = {"format": "cellphenotyper-bora-semantic-roi-intake-v1",
            "bora_source_root": str(SOURCE_ROOT),
            "bora_source_sha256": source_hashes,
            "inputs": {name: {"path": str(path), "sha256": sha256(path)}
                       for name, path in inputs.items()}}
    spec_path = root / "roi-spec.json"
    _write_json(spec_path, spec)
    evaluation = {"format": module.EVALUATION_FORMAT,
        "complete": True, "status": module.PASS_STATUS,
        "same_slide_holdout": True, "independent_patient_validation": False,
        "probability_calibrated": False,
        "annotation_source": "declared_independent_expert",
        "fullslide_projection_started": False,
        "semantic_annotation_published": False,
        "source_sha256": {"spec": sha256(spec_path),
            "bora_source_modules": source_hashes,
            **{name: ref["sha256"] for name, ref in spec["inputs"].items()}},
        "gates": {key: True for key in (
            "holdout_material", "macro_iou", "per_class_iou",
            "per_class_recall", "no_single_class_collapse")},
        "gate_thresholds": {"holdout_tokens_per_class": 100,
            "holdout_rois_per_class": 2, "macro_iou": .7,
            "class_iou": .6, "class_recall": .7},
        "macro_iou": 150 / 170, "model_sha256": model_sha,
        "source_frame_id": item["native"]["frame_id"],
        "working_frame_id": item["working"]["frame_id"],
        "class_ids": [1, 2], "class_names": classes,
        "token_shape_yx": [(item["working"]["height"] + 13) // 14,
                            (item["working"]["width"] + 13) // 14],
        "token_stride_working_yx": [14, 14],
        "token_origin_working_yx": [0, 0],
        "token_label_rule": "full_14x14_working_pixel_interior_then_one_token_erosion_and_full_support",
        "he_feature_rule": "registered_working_RGB_white_padded_to_token_grid_mean_14x14_divide_255",
        "train_holdout_minimum_gap_um": 224.0,
        "training_sample_seed": 314159,
        "training_sample_cap_per_class": 1024,
        "selected_train_tokens_by_class": {"1": 100, "2": 100},
        "selected_train_tokens_by_polygon": {
            "train-stroma": 100, "train-tumor": 100},
        "per_polygon": [{"annotation_id": aid, "class_id": cid,
            "split": split, "roi_id": roi, "annotator_id": "synthetic-expert",
            "retained_supported_interior_tokens": count}
            for aid, cid, split, roi, count in roi_rows],
        "holdout_tokens": 320, "confusion": [[150, 10], [10, 150]],
        "per_class": {
            "1": {"name": "stroma", "support": 160,
                "holdout_roi_counts": {"stroma-a": 80, "stroma-b": 80},
                "precision": 150 / 160, "recall": 150 / 160, "iou": 150 / 170},
            "2": {"name": "tumor", "support": 160,
                "holdout_roi_counts": {"tumor-a": 80, "tumor-b": 80},
                "precision": 150 / 160, "recall": 150 / 160, "iou": 150 / 170}}}
    eval_path = evaluation_folder / "EVALUATION.json"
    _write_json(eval_path, evaluation)
    work_h, work_w = item["working"]["height"], item["working"]["width"]
    ty, tx = (work_h + 13) // 14, (work_w + 13) // 14
    token_source = material_folder / materializer.OUTPUTS["virchow2_tokens"]
    he_token_source = material_folder / materializer.OUTPUTS["he_tokens"]
    token_folder = root / "semantic-token-field"
    token_folder.mkdir()
    np.save(token_folder / "probability.npy",
            np.full((ty, tx, 2), .5, np.float32))
    probability_source_sha = sha256(SOURCE_ROOT / "bora" / "wsi_semantic_probability.py")
    token_binding = {"format": module.TOKEN_FORMAT,
        "model_sha256": model_sha,
        "token_source_sha256": sha256(token_source),
        "he_token_source_sha256": sha256(he_token_source),
        "registration_sha256": sha256(item["registration"]),
        "source_module_sha256": probability_source_sha,
        "materialization_receipt_sha256": sha256(
            material_folder / "MATERIALIZATION_COMPLETE.json"),
        "materialization_checkpoint_sha256": sha256(
            material_inputs["virchow2_checkpoint"]),
        "materialization_inventory_sha256": sha256(
            material_inputs["token_inventory"]),
        "native_frame_id": item["native"]["frame_id"],
        "native_image_sha256": item["native"]["sha256"],
        "working_frame_id": item["working"]["frame_id"],
        "working_image_sha256": item["working"]["sha256"],
        "frame": frame,
        "class_ids": [1, 2],
        "class_names": classes, "he_token_features": True,
        "cell_composition": False}
    token_binding_sha = module._binding(token_binding)
    token_receipt = {**token_binding,
        "global_model_binding_sha256": token_binding_sha,
        "probability_sha256": sha256(token_folder / "probability.npy"),
        "probability_shape_yxc": [ty, tx, 2],
        "working_shape_yx": [work_h, work_w],
        "token_shape_yx": [ty, tx],
        "independently_fitted_windows": False,
        "expert_annotation_used_at_inference": False,
        "native_he_boundary_refinement_applied": False}
    token_receipt_path = token_folder / "TOKEN_SEMANTIC_COMPLETE.json"
    _write_json(token_receipt_path, token_receipt)
    shard_folder = root / "shard-0"
    shard_folder.mkdir()
    h, w = item["native"]["height"], item["native"]["width"]
    np.save(shard_folder / "probability.npy", np.full((h, w, 2), .5, np.float32))
    old_chain = json.loads((output / "CONTINUOUS_CHAIN_COMPLETE.json").read_text())
    for key in ("semantic_mask", "exact_mask"):
        mask_path = output / module.OUTPUT_PATHS[key]
        mask = np.frombuffer(mask_path.read_bytes(), np.uint8).copy().reshape(h, w)
        mask[30:90, 60:90] = 2
        mask_path.write_bytes(mask.tobytes())
    semantic_sha = sha256(output / module.OUTPUT_PATHS["semantic_mask"])
    shard_binding = {"format": module.SHARD_FORMAT,
        "token_receipt_sha256": sha256(token_receipt_path),
        "token_probability_sha256": token_receipt["probability_sha256"],
        "model_sha256": model_sha,
        "token_source_sha256": token_binding["token_source_sha256"],
        "he_token_source_sha256": token_binding["he_token_source_sha256"],
        "token_model_binding_sha256": token_binding_sha,
        "registration_sha256": sha256(item["registration"]),
        "registered_support_source_sha256": sha256(item["support"]),
        "tissue_support_sha256": old_chain["tissue_support_sha256"],
        "source_frame_id": item["native"]["frame_id"],
        "source_image_sha256": item["native"]["sha256"],
        "native_shape_yx": [h, w], "native_mpp_xy": [.25, .25],
        "class_ids": [1, 2], "class_names": classes,
        "interpolation": "absolute-native-centre-to-14-working-pixel-token-centre/bilinear-nearest-edge",
        "source_module_sha256": probability_source_sha,
        "native_he_boundary_refinement_applied": False}
    shard_binding_sha = module._binding(shard_binding)
    window = {"probability_path": str(shard_folder / "probability.npy"),
        "probability_sha256": sha256(shard_folder / "probability.npy"),
        "origin_yx": [0, 0], "core_yxhw": [0, 0, h, w],
        "global_model_binding_sha256": shard_binding_sha,
        "independently_fitted": False}
    shard_receipt = {**shard_binding,
        "global_model_binding_sha256": shard_binding_sha,
        "probability_sha256": window["probability_sha256"],
        "probability_shape_yxc": [h, w, 2],
        "origin_yx": [0, 0], "core_yxhw": [0, 0, h, w],
        "window": window, "partition_independent_projection": True,
        "independently_fitted": False,
        "expert_annotation_used_at_inference": False,
        "biological_accuracy_qualified": False}
    shard_receipt_path = shard_folder / "SHARD_COMPLETE.json"
    _write_json(shard_receipt_path, shard_receipt)
    old_chain.update(format=module.CHAIN_FORMAT,
        identity_status="exploratory_semantic_transfer",
        model_sha256=model_sha,
        token_source_sha256=token_binding["token_source_sha256"],
        he_token_source_sha256=token_binding["he_token_source_sha256"],
        token_receipt_sha256=sha256(token_receipt_path),
        token_probability_sha256=token_receipt["probability_sha256"],
        global_model_binding_sha256=shard_binding_sha,
        class_names=classes, native_he_boundary_refinement_applied=False,
        expert_annotation_used_at_inference=False,
        shard_receipt_sha256=[sha256(shard_receipt_path)],
        output_paths=module.OUTPUT_PATHS,
        output_sha256={key: sha256(output / relative)
                       for key, relative in module.OUTPUT_PATHS.items()})
    for key, relative in module.OUTPUT_PATHS.items():
        if key not in ("assembly_receipt", "vector_receipt", "exact_geojson", "review_geojson"):
            continue
        path = output / relative
        value = json.loads(path.read_text())
        value["global_model_binding_sha256"] = shard_binding_sha
        if key == "assembly_receipt":
            value["semantic_mask_sha256"] = semantic_sha
        if key == "vector_receipt":
            value["source_semantic_mask_sha256"] = semantic_sha
            value["review_component_count"] = 2
            value["exact_component_count"] = 2
        if key in ("exact_geojson", "review_geojson"):
            value["source_semantic_mask_sha256"] = semantic_sha
            first = value["features"][0]
            first["properties"]["classification"]["name"] = "stroma"
            first["properties"]["area_um2"] = 112.5
            first["geometry"]["coordinates"] = [[
                [30, 30], [60, 30], [60, 90], [30, 90], [30, 30]]]
            second = json.loads(json.dumps(first))
            second["properties"]["classification"]["name"] = "tumor"
            second["properties"]["class_id"] = 2
            second["properties"]["component_id"] = "synthetic-second-component"
            second["geometry"]["coordinates"] = [[
                [60, 30], [90, 30], [90, 90], [60, 90], [60, 30]]]
            value["features"].append(second)
        _write_json(path, value)
    vector_path = output / module.OUTPUT_PATHS["vector_receipt"]
    vectors = json.loads(vector_path.read_text())
    vectors["exact_mask_sha256"] = sha256(output / module.OUTPUT_PATHS["exact_mask"])
    vectors["output_sha256"] = {
        "exact/mask.uint8.dat": sha256(output / module.OUTPUT_PATHS["exact_mask"]),
        "exact/mask.geojson": sha256(output / module.OUTPUT_PATHS["exact_geojson"]),
        "review/mask.geojson": sha256(output / module.OUTPUT_PATHS["review_geojson"])}
    _write_json(vector_path, vectors)
    old_chain["output_sha256"] = {key: sha256(output / relative)
                                  for key, relative in module.OUTPUT_PATHS.items()}
    _write_json(output / "SEMANTIC_CHAIN_COMPLETE.json", old_chain)
    return {**item, "spec": spec_path, "evaluation": evaluation_folder,
        "materialization": material_folder,
        "token_folder": token_folder, "token_source": token_source,
        "he_token_source": he_token_source, "shards": [shard_folder]}


def _export(item: dict, dest: Path, *, canonical: bool = False):
    return module.export(item["base"], item["spec"], item["evaluation"],
        item["registration"], item["support"], item["materialization"],
        item["token_folder"],
        item["token_source"], item["he_token_source"], item["shards"],
        item["output"], SOURCE_ROOT, dest,
        canonical_tissue_annotations=canonical)


@pytest.mark.skipif(not (SOURCE_ROOT / "bora" / "wsi_semantic_chain.py").is_file(),
                    reason="Staged Bora semantic source unavailable")
def test_passed_same_slide_export_is_predicted_review_only(tmp_path: Path):
    item = _fixture(tmp_path)
    dest = tmp_path / "semantic-review.cellphenotyper.json"
    report = _export(item, dest)
    project = validate_experiment(dest)
    artifact = project["tissues"][0]["artifacts"][0]
    assert report["complete"] is True
    assert artifact["role"] == module.ROLE
    assert artifact["review_only"] is True
    assert artifact["training_eligible"] is False
    assert artifact["predicted_not_expert_annotation"] is True
    assert artifact["provenance"]["identity_status"] == "exploratory_same_slide_holdout"
    assert artifact["provenance"]["native_he_boundary_refinement_applied"] is False
    assert not any(a["role"] == "tissue_annotations" for a in project["tissues"][0]["artifacts"])


@pytest.mark.parametrize("tamper", [
    "missing_confusion", "missing_per_class", "forged_macro_iou",
    "stale_class_support", "stale_roi_count", "forged_precision",
    "stale_polygon_count", "stale_class_ids", "missing_train_counts",
    "forged_holdout_total", "nonfinite_macro_iou",
])
def test_evaluation_metrics_are_recomputed_before_semantic_export(tmp_path: Path, tamper: str):
    item = _fixture(tmp_path)
    evaluation_path = item["evaluation"] / "EVALUATION.json"
    evaluation = json.loads(evaluation_path.read_text())
    if tamper == "missing_confusion":
        del evaluation["confusion"]
    elif tamper == "missing_per_class":
        del evaluation["per_class"]
    elif tamper == "forged_macro_iou":
        evaluation["macro_iou"] = .99
    elif tamper == "stale_class_support":
        evaluation["per_class"]["1"]["support"] += 1
    elif tamper == "stale_roi_count":
        evaluation["per_class"]["2"]["holdout_roi_counts"]["tumor-a"] += 1
    elif tamper == "forged_precision":
        evaluation["per_class"]["1"]["precision"] = 1.0
    elif tamper == "stale_polygon_count":
        evaluation["per_polygon"][2]["retained_supported_interior_tokens"] += 1
    elif tamper == "stale_class_ids":
        evaluation["class_ids"] = [1, 3]
    elif tamper == "missing_train_counts":
        del evaluation["selected_train_tokens_by_class"]
    elif tamper == "forged_holdout_total":
        evaluation["holdout_tokens"] += 1
    elif tamper == "nonfinite_macro_iou":
        evaluation["macro_iou"] = float("nan")
    # Deliberately retain the old booleans/model hash to demonstrate that
    # their presence cannot turn inconsistent held-out counts into evidence.
    evaluation_path.write_text(json.dumps(evaluation, allow_nan=True) + "\n")
    with pytest.raises(ValueError):
        module._evaluation(item["spec"], item["evaluation"], SOURCE_ROOT,
                           item["registration"])


@pytest.mark.parametrize("tamper", ["failed_holdout", "alter_model", "alter_token",
                                    "alter_shard", "alter_material_receipt",
                                    "alter_scale2_chunk", "anonymous_chain", "hard_group",
                                    "class_collapse", "alter_vector"])
def test_incomplete_or_unbound_semantic_result_is_rejected(tmp_path: Path, tamper: str):
    item = _fixture(tmp_path)
    if tamper == "failed_holdout":
        path = item["evaluation"] / "EVALUATION.json"
        value = json.loads(path.read_text())
        value["gates"]["per_class_recall"] = False
        _write_json(path, value)
    elif tamper == "alter_model":
        with (item["evaluation"] / "local_semantic_model.npz").open("ab") as stream:
            stream.write(b"changed")
    elif tamper == "alter_token":
        with item["token_source"].open("ab") as stream:
            stream.write(b"changed")
    elif tamper == "alter_shard":
        with (item["shards"][0] / "probability.npy").open("ab") as stream:
            stream.write(b"changed")
    elif tamper == "alter_material_receipt":
        path = item["materialization"] / "MATERIALIZATION_COMPLETE.json"
        value = json.loads(path.read_text())
        value["no_chunk_gap_or_overlap"] = False
        _write_json(path, value)
    elif tamper == "alter_scale2_chunk":
        path = next((item["materialization"].parent / "chunks" / "scale-2").glob("*.npy"))
        with path.open("ab") as stream:
            stream.write(b"changed")
    elif tamper == "anonymous_chain":
        path = item["output"] / "SEMANTIC_CHAIN_COMPLETE.json"
        value = json.loads(path.read_text())
        value["class_names"] = {"1": "anonymous group 1", "2": "anonymous group 2"}
        _write_json(path, value)
    elif tamper == "hard_group":
        path = item["output"] / "SEMANTIC_CHAIN_COMPLETE.json"
        value = json.loads(path.read_text())
        value["format"] = "bora-wsi-continuous-chain-v1"
        _write_json(path, value)
    elif tamper == "class_collapse":
        exact = item["output"] / module.OUTPUT_PATHS["exact_mask"]
        mask = bytearray(exact.read_bytes())
        mask[:] = bytes(1 if value else 0 for value in mask)
        exact.write_bytes(mask)
        chain_path = item["output"] / "SEMANTIC_CHAIN_COMPLETE.json"
        chain = json.loads(chain_path.read_text())
        chain["output_sha256"]["exact_mask"] = sha256(exact)
        _write_json(chain_path, chain)
    else:
        with (item["output"] / module.OUTPUT_PATHS["review_geojson"]).open("ab") as stream:
            stream.write(b"changed")
    match = "collapsed" if tamper == "class_collapse" else None
    with pytest.raises(ValueError, match=match):
        _export(item, tmp_path / "rejected.json")


def test_canonical_annotation_request_is_refused_without_he_refinement(tmp_path: Path):
    item = _fixture(tmp_path)
    dest = tmp_path / "not-published.json"
    with pytest.raises(ValueError, match="native H&E refinement"):
        _export(item, dest, canonical=True)
    assert not dest.exists()


def test_existing_project_is_not_overwritten(tmp_path: Path):
    item = _fixture(tmp_path)
    dest = tmp_path / "preserved.json"
    dest.write_text("preserve")
    with pytest.raises(ValueError, match="new project JSON path"):
        _export(item, dest)
    assert dest.read_text() == "preserve"
