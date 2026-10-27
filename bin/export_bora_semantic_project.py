"""Export a verified same-slide Bora semantic result as a locked review layer.

The current Bora semantic chain has no native H&E boundary refinement and
declares biological accuracy unqualified. Even a passing same-slide holdout
therefore cannot become canonical ``tissue_annotations`` through this route.
This command performs no inference and never edits its input experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from .bora_semantic_roi_intake import (
        REQUIRED_INPUTS, SOURCE_MODULES, MIN_HOLDOUT_TOKENS,
        MIN_HOLDOUT_ROIS, MIN_MACRO_IOU, MIN_CLASS_IOU,
        MIN_CLASS_RECALL, MIN_GAP_UM, STRIDE,
    )
    from .export_bora_continuous_project import (
        _frame, _geojson_header, _hash, _json, _same_native,
    )
    from .export_experiment import export_experiment, sha256, validate_experiment
    from .materialize_bora_semantic_tokens import (
        OUTPUTS as MATERIALIZED_OUTPUTS, verify_materialization,
    )
else:
    from bora_semantic_roi_intake import (
        REQUIRED_INPUTS, SOURCE_MODULES, MIN_HOLDOUT_TOKENS,
        MIN_HOLDOUT_ROIS, MIN_MACRO_IOU, MIN_CLASS_IOU,
        MIN_CLASS_RECALL, MIN_GAP_UM, STRIDE,
    )
    from export_bora_continuous_project import (
        _frame, _geojson_header, _hash, _json, _same_native,
    )
    from export_experiment import export_experiment, sha256, validate_experiment
    from materialize_bora_semantic_tokens import (
        OUTPUTS as MATERIALIZED_OUTPUTS, verify_materialization,
    )


CHAIN_FORMAT = "bora-wsi-semantic-chain-v1"
EVALUATION_FORMAT = "cellphenotyper-bora-semantic-roi-evaluation-v1"
TOKEN_FORMAT = "bora-wsi-semantic-token-field-v1"
SHARD_FORMAT = "bora-wsi-semantic-probability-shard-v1"
OUTPUT_PATHS = {
    "support": "support.npy",
    "assembly_receipt": "assembly/ASSEMBLY_COMPLETE.json",
    "semantic_mask": "assembly/semantic-mask.uint8.dat",
    "vector_receipt": "vectors/GLOBAL_VECTOR_COMPLETE.json",
    "exact_geojson": "vectors/exact/mask.geojson",
    "review_geojson": "vectors/review/mask.geojson",
    "exact_mask": "vectors/exact/mask.uint8.dat",
}
PASS_STATUS = "same_slide_holdout_passed_independent_patient_validation_pending"
ROLE = "bora_exploratory_boundary_review"
SHARD_BINDING_FIELDS = (
    "format", "token_receipt_sha256", "token_probability_sha256",
    "model_sha256", "token_source_sha256", "he_token_source_sha256",
    "token_model_binding_sha256", "registration_sha256",
    "registered_support_source_sha256", "tissue_support_sha256",
    "source_frame_id", "source_image_sha256", "native_shape_yx",
    "native_mpp_xy", "class_ids", "class_names", "interpolation",
    "source_module_sha256", "native_he_boundary_refinement_applied",
)


def _binding(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _ordinary(path: Path, name: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or symlinked {name}: {path}")
    return path


def _source_ref(ref: dict, name: str) -> Path:
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError(f"{name} needs a frozen path and SHA-256")
    path = Path(ref["path"])
    if not path.is_absolute() or sha256(_ordinary(path, name)) != _hash(ref["sha256"], name):
        raise ValueError(f"{name} differs from its frozen source hash")
    return path.resolve(strict=True)


def _reject_nonfinite(value: str):
    raise ValueError(f"Nonfinite scientific metadata value: {value}")


def _strict_evaluation(path: Path) -> dict:
    """JSON's NaN/Infinity extension must never satisfy a scientific gate."""
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Expert evaluation is oversized")
    try:
        report = json.loads(path.read_text(), parse_constant=_reject_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Expert evaluation is not strict JSON") from exc
    if not isinstance(report, dict):
        raise ValueError("Expert evaluation must be an object")
    return report


def _verify_evaluation_metrics(evaluation: dict, spec: dict, expert: dict,
                               registration: dict, model_metadata: dict) -> None:
    """Recompute every declared holdout gate from source-bound count evidence."""
    names = expert["class_names"]
    ids = evaluation.get("class_ids")
    if (not isinstance(names, dict) or not isinstance(ids, list) or len(ids) < 2 or
            any(type(cid) is not int or not 1 <= cid <= 253 for cid in ids) or
            ids != sorted(set(ids)) or set(names) != {str(cid) for cid in ids} or
            evaluation.get("class_names") != names):
        raise ValueError("Expert evaluation class IDs/names differ from annotations")
    working = registration["image"]
    expected_shape = [(working["height"] + STRIDE - 1) // STRIDE,
                      (working["width"] + STRIDE - 1) // STRIDE]
    seed = spec.get("seed", 314159)
    cap = spec.get("max_samples_per_class", 1024)
    expected_frame = {"format": "virchow2-spatial-tokens-v1",
        "encoder_sha256": spec["inputs"]["virchow2_checkpoint"]["sha256"],
        "working_mpp_xy": [working["mpp_x"], working["mpp_y"]],
        "token_stride_yx": [STRIDE, STRIDE],
        "token_origin_working_yx": [0, 0], "channel_order": "YXC-1280"}
    if (evaluation.get("source_frame_id") != registration["cell_full_image"]["frame_id"] or
            evaluation.get("working_frame_id") != working["frame_id"] or
            evaluation.get("token_shape_yx") != expected_shape or
            evaluation.get("token_stride_working_yx") != [STRIDE, STRIDE] or
            evaluation.get("token_origin_working_yx") != [0, 0] or
            evaluation.get("token_label_rule") !=
                "full_14x14_working_pixel_interior_then_one_token_erosion_and_full_support" or
            evaluation.get("he_feature_rule") !=
                "registered_working_RGB_white_padded_to_token_grid_mean_14x14_divide_255" or
            evaluation.get("train_holdout_minimum_gap_um") != MIN_GAP_UM or
            type(seed) is not int or seed < 0 or type(cap) is not int or not 2 <= cap <= 1024 or
            evaluation.get("training_sample_seed") != seed or
            evaluation.get("training_sample_cap_per_class") != cap or
            model_metadata.get("frame") != expected_frame):
        raise ValueError("Expert evaluation token frame or training specification differs")

    features = expert["features"]
    polygon_rows = evaluation.get("per_polygon")
    if (not isinstance(features, list) or not isinstance(polygon_rows, list) or
            len(polygon_rows) != len(features)):
        raise ValueError("Expert evaluation lacks complete per-polygon evidence")
    expected_features = {}
    for feature in features:
        props = feature.get("properties") if isinstance(feature, dict) else None
        if (not isinstance(props, dict) or not isinstance(props.get("annotation_id"), str) or
                not props["annotation_id"].strip() or
                not isinstance(props.get("roi_id"), str) or not props["roi_id"].strip() or
                not isinstance(props.get("annotator_id"), str) or
                not props["annotator_id"].strip() or
                props.get("annotation_source") != "independent_expert"):
            raise ValueError("Expert polygon provenance is incomplete")
        aid = props["annotation_id"]
        if aid in expected_features or props.get("class_id") not in ids or \
                props.get("split") not in ("train", "holdout"):
            raise ValueError("Expert polygon IDs/classes are ambiguous")
        expected_features[aid] = props
    retained = {}
    roi_counts = {cid: {} for cid in ids}
    for row in polygon_rows:
        if not isinstance(row, dict) or set(row) != {
                "annotation_id", "split", "class_id", "roi_id", "annotator_id",
                "retained_supported_interior_tokens"}:
            raise ValueError("Expert per-polygon evidence is incomplete")
        aid = row["annotation_id"]
        props = expected_features.get(aid)
        count = row["retained_supported_interior_tokens"]
        if (props is None or aid in retained or type(count) is not int or
                not 0 <= count <= expected_shape[0] * expected_shape[1] or
                any(row.get(key) != props.get(key)
                    for key in ("split", "class_id", "roi_id", "annotator_id"))):
            raise ValueError("Expert per-polygon counts or provenance differ")
        retained[aid] = count
        if row["split"] == "holdout":
            class_rois = roi_counts[row["class_id"]]
            roi = row["roi_id"]
            class_rois[roi] = class_rois.get(roi, 0) + count
    if set(retained) != set(expected_features):
        raise ValueError("Expert evaluation omitted annotation polygons")
    if sum(retained.values()) > expected_shape[0] * expected_shape[1]:
        raise ValueError("Expert polygon token counts exceed registered token grid")

    selected = evaluation.get("selected_train_tokens_by_polygon")
    selected_class = evaluation.get("selected_train_tokens_by_class")
    train_ids = {aid for aid, props in expected_features.items() if props["split"] == "train"}
    if (not isinstance(selected, dict) or set(selected) != train_ids or
            not isinstance(selected_class, dict) or set(selected_class) != set(names) or
            model_metadata.get("training_counts") != selected_class):
        raise ValueError("Training token counts differ from saved model or expert ROIs")
    for cid in ids:
        count = selected_class[str(cid)]
        if (type(count) is not int or not 1 <= count <= cap or
                any(type(selected[aid]) is not int or not 0 <= selected[aid] <= retained[aid]
                    for aid in train_ids if expected_features[aid]["class_id"] == cid) or
                sum(selected[aid] for aid in train_ids
                    if expected_features[aid]["class_id"] == cid) != count):
            raise ValueError("Saved model training counts are inconsistent")

    confusion = evaluation.get("confusion")
    n = len(ids)
    if (not isinstance(confusion, list) or len(confusion) != n or
            any(not isinstance(row, list) or len(row) != n or
                any(type(value) is not int or value < 0 for value in row)
                for row in confusion)):
        raise ValueError("Expert holdout confusion must be a nonnegative integer matrix")
    total = sum(map(sum, confusion))
    if (type(evaluation.get("holdout_tokens")) is not int or total == 0 or
            total > expected_shape[0] * expected_shape[1] or
            evaluation["holdout_tokens"] != total):
        raise ValueError("Expert holdout token count differs from confusion")
    classes = evaluation.get("per_class")
    if not isinstance(classes, dict) or set(classes) != set(names):
        raise ValueError("Expert holdout lacks complete per-class evidence")
    expected_metrics = {}
    for i, cid in enumerate(ids):
        row = classes[str(cid)]
        if not isinstance(row, dict) or set(row) != {
                "name", "support", "holdout_roi_counts", "precision", "recall", "iou"}:
            raise ValueError("Expert holdout class metrics are incomplete")
        actual = sum(confusion[i])
        predicted = sum(confusion[k][i] for k in range(n))
        tp = confusion[i][i]
        union = actual + predicted - tp
        expected = {"precision": tp / predicted if predicted else None,
                    "recall": tp / actual if actual else None,
                    "iou": tp / union if union else None}
        if (row["name"] != names[str(cid)] or type(row["support"]) is not int or
                row["support"] != actual or row["holdout_roi_counts"] != roi_counts[cid] or
                sum(roi_counts[cid].values()) != actual):
            raise ValueError("Expert ROI/token support differs from confusion")
        for key, computed in expected.items():
            claimed = row[key]
            if computed is None:
                if claimed is not None:
                    raise ValueError("Expert holdout class metric differs from confusion")
            elif (type(claimed) not in (int, float) or not math.isfinite(claimed) or
                  not math.isclose(claimed, computed, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("Expert holdout class metric differs from confusion")
        expected_metrics[cid] = expected
    if any(expected_metrics[cid]["iou"] is None for cid in ids):
        raise ValueError("Expert holdout has a class without IoU support")
    macro = sum(expected_metrics[cid]["iou"] for cid in ids) / n
    claimed_macro = evaluation.get("macro_iou")
    if (type(claimed_macro) not in (int, float) or not math.isfinite(claimed_macro) or
            not math.isclose(claimed_macro, macro, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("Expert macro IoU differs from confusion")
    thresholds = {"holdout_tokens_per_class": MIN_HOLDOUT_TOKENS,
        "holdout_rois_per_class": MIN_HOLDOUT_ROIS, "macro_iou": MIN_MACRO_IOU,
        "class_iou": MIN_CLASS_IOU, "class_recall": MIN_CLASS_RECALL}
    gates = {"holdout_material": all(
            sum(confusion[i]) >= MIN_HOLDOUT_TOKENS and
            sum(count > 0 for count in roi_counts[cid].values()) >= MIN_HOLDOUT_ROIS
            for i, cid in enumerate(ids)),
        "macro_iou": macro >= MIN_MACRO_IOU,
        "per_class_iou": all(expected_metrics[cid]["iou"] is not None and
            expected_metrics[cid]["iou"] >= MIN_CLASS_IOU for cid in ids),
        "per_class_recall": all(expected_metrics[cid]["recall"] is not None and
            expected_metrics[cid]["recall"] >= MIN_CLASS_RECALL for cid in ids),
        "no_single_class_collapse": sum(
            sum(confusion[k][i] for k in range(n)) > 0 for i in range(n)) >= 2}
    if (evaluation.get("gate_thresholds") != thresholds or
            evaluation.get("gates") != gates or any(value is not True for value in gates.values())):
        raise ValueError("Expert holdout gates do not follow verified counts")


def _evaluation(spec_path: Path, evaluation_folder: Path, source_root: Path,
                registration_path: Path) -> tuple[dict, dict, Path]:
    spec = _json(_ordinary(spec_path, "semantic ROI specification"), maximum=16 * 1024 * 1024)
    if (spec.get("format") != "cellphenotyper-bora-semantic-roi-intake-v1" or
            not isinstance(spec.get("inputs"), dict) or
            set(spec["inputs"]) != set(REQUIRED_INPUTS) or
            Path(spec.get("bora_source_root", "")) != source_root):
        raise ValueError("A complete frozen semantic expert ROI specification is required")
    inputs = {key: _source_ref(spec["inputs"][key], key) for key in REQUIRED_INPUTS}
    if inputs["registration"] != registration_path.resolve(strict=True):
        raise ValueError("Expert ROI evaluation uses a different registration")
    source_hashes = spec.get("bora_source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_MODULES):
        raise ValueError("Expert ROI Bora source inventory is incomplete")
    for name in SOURCE_MODULES:
        if sha256(_ordinary(source_root / "bora" / name, name)) != _hash(source_hashes[name], name):
            raise ValueError(f"Expert ROI Bora source changed: {name}")
    evaluation_path = _ordinary(evaluation_folder / "EVALUATION.json", "ROI evaluation")
    evaluation = _strict_evaluation(evaluation_path)
    sources = evaluation.get("source_sha256")
    if (evaluation.get("format") != EVALUATION_FORMAT or
            evaluation.get("complete") is not True or
            evaluation.get("status") != PASS_STATUS or
            evaluation.get("same_slide_holdout") is not True or
            evaluation.get("independent_patient_validation") is not False or
            evaluation.get("probability_calibrated") is not False or
            evaluation.get("annotation_source") != "declared_independent_expert" or
            evaluation.get("fullslide_projection_started") is not False or
            evaluation.get("semantic_annotation_published") is not False or
            not isinstance(sources, dict) or
            sources.get("spec") != sha256(spec_path) or
            sources.get("bora_source_modules") != source_hashes or
            any(sources.get(name) != spec["inputs"][name]["sha256"]
                for name in REQUIRED_INPUTS)):
        raise ValueError("Same-slide expert evaluation is failed, incomplete or source-mismatched")
    gates = evaluation.get("gates")
    required_gates = {"holdout_material", "macro_iou", "per_class_iou",
                      "per_class_recall", "no_single_class_collapse"}
    if (not isinstance(gates, dict) or set(gates) != required_gates or
            any(value is not True for value in gates.values()) or
            evaluation.get("gate_thresholds") != {
                "holdout_tokens_per_class": 100, "holdout_rois_per_class": 2,
                "macro_iou": .7, "class_iou": .6, "class_recall": .7} or
            not isinstance(evaluation.get("macro_iou"), (int, float)) or
            not math.isfinite(evaluation["macro_iou"]) or
            evaluation["macro_iou"] < .7):
        raise ValueError("Expert same-slide holdout gates did not all pass")
    model_path = _ordinary(evaluation_folder / "local_semantic_model.npz", "semantic model")
    if sha256(model_path) != _hash(evaluation.get("model_sha256"), "expert model"):
        raise ValueError("Expert-trained model differs from passing evaluation")
    try:
        with np.load(model_path, allow_pickle=False) as model:
            if set(model.files) != {"metadata", "mean", "scale", "coef", "intercept", "classes"}:
                raise ValueError("Not a Bora semantic model")
            metadata = json.loads(str(model["metadata"].item()),
                                  parse_constant=_reject_nonfinite)
            model_classes = model["classes"].tolist()
            if (not isinstance(metadata, dict) or
                    metadata.get("format") != "bora-local-semantic-token-v2" or
                    metadata.get("classifier") != "balanced-l2-logistic-regression-cpu" or
                    metadata.get("training_labels_included") is not False or
                    metadata.get("token_channels") != 1280 or
                    metadata.get("he_enabled") is not True or
                    metadata.get("composition_schema") is not None or
                    metadata.get("class_names") != evaluation.get("class_names") or
                    model_classes != evaluation.get("class_ids")):
                raise ValueError("Bora model metadata differs from expert evaluation")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot verify the saved Bora semantic model") from exc
    expert = _json(inputs["expert_geojson"], maximum=16 * 1024 * 1024)
    expert_source = expert.get("source")
    registration = _json(inputs["registration"])
    native = registration["cell_full_image"]
    if (expert.get("type") != "FeatureCollection" or
            expert.get("schema") not in ("cellphenotyper-expert-tissue-v1",
                                        "pm016-expert-tissue-v1") or
            not expert.get("features") or
            expert.get("class_names") != evaluation.get("class_names") or
            not isinstance(expert_source, dict) or
            expert_source.get("source_frame_id") != native["frame_id"] or
            expert_source.get("image_sha256") != native["sha256"] or
            expert_source.get("native_shape_yx") != [native["height"], native["width"]] or
            expert_source.get("native_mpp_xy") != [native["mpp_x"], native["mpp_y"]] or
            expert_source.get("token_inventory_sha256") !=
                spec["inputs"]["token_inventory"]["sha256"] or
            expert_source.get("coordinate_space") != "native_image_pixel_edges"):
        raise ValueError("Expert polygons or class names differ from passing evaluation")
    _verify_evaluation_metrics(evaluation, spec, expert, registration, metadata)
    return evaluation, spec, model_path


def _token_receipt(token_folder: Path, token_source: Path, he_token_source: Path,
                   registration: dict, chain: dict, evaluation: dict,
                   source_root: Path, materialization_folder: Path,
                   materialized: dict) -> dict:
    path = _ordinary(token_folder / "TOKEN_SEMANTIC_COMPLETE.json", "semantic token receipt")
    token = _json(path)
    working, native = registration["image"], registration["cell_full_image"]
    field = _ordinary(token_folder / "probability.npy", "semantic token probability")
    source_hash = sha256(_ordinary(source_root / "bora" / "wsi_semantic_probability.py",
                                   "semantic probability source"))
    binding_keys = ("format", "model_sha256", "token_source_sha256",
        "he_token_source_sha256", "registration_sha256", "source_module_sha256",
        "materialization_receipt_sha256", "materialization_checkpoint_sha256",
        "materialization_inventory_sha256",
        "native_frame_id", "native_image_sha256", "working_frame_id",
        "working_image_sha256", "frame", "class_ids", "class_names",
        "he_token_features", "cell_composition")
    if (token.get("format") != TOKEN_FORMAT or
            any(key not in token for key in binding_keys) or
            token.get("global_model_binding_sha256") !=
                _binding({key: token[key] for key in binding_keys}) or
            token.get("source_module_sha256") != source_hash or
            token.get("materialization_receipt_sha256") !=
                sha256(materialization_folder / "MATERIALIZATION_COMPLETE.json") or
            token.get("materialization_checkpoint_sha256") !=
                materialized["checkpoint_sha256"] or
            token.get("materialization_inventory_sha256") !=
                materialized["inventory_sha256"] or
            token.get("registration_sha256") != chain["registration_sha256"] or
            token.get("native_frame_id") != native["frame_id"] or
            token.get("native_image_sha256") != native["sha256"] or
            token.get("working_frame_id") != working["frame_id"] or
            token.get("working_image_sha256") != working["sha256"] or
            token.get("model_sha256") != evaluation["model_sha256"] or
            token.get("class_ids") != chain["class_ids"] or
            token.get("class_names") != chain["class_names"] or
            token.get("he_token_features") is not True or
            token.get("cell_composition") is not False or
            token.get("independently_fitted_windows") is not False or
            token.get("expert_annotation_used_at_inference") is not False or
            token.get("native_he_boundary_refinement_applied") is not False or
            token.get("probability_sha256") != sha256(field) or
            sha256(path) != chain["token_receipt_sha256"] or
            token.get("probability_sha256") != chain["token_probability_sha256"] or
            token.get("token_source_sha256") != chain["token_source_sha256"] or
            token.get("he_token_source_sha256") != chain["he_token_source_sha256"]):
        raise ValueError("Semantic token receipt, trained model or registered sources differ")
    for name, source_path, digest in (
            ("Virchow2 token field", token_source, token["token_source_sha256"]),
            ("registered H&E token field", he_token_source,
             token["he_token_source_sha256"])):
        if sha256(_ordinary(source_path, name)) != _hash(digest, name):
            raise ValueError(f"{name} differs from semantic token receipt")
    expected_yx = [(working["height"] + 13) // 14, (working["width"] + 13) // 14]
    for name, source_path, channels in (
            ("Virchow2 token field", token_source, 1280),
            ("registered H&E token field", he_token_source, 3)):
        source_array = np.load(source_path, mmap_mode="r", allow_pickle=False)
        if source_array.dtype != np.float32 or list(source_array.shape) != [*expected_yx, channels]:
            raise ValueError(f"{name} has the wrong registered shape/dtype")
    array = np.load(field, mmap_mode="r", allow_pickle=False)
    expected_shape = [*expected_yx, len(chain["class_ids"])]
    if array.dtype != np.float32 or list(array.shape) != expected_shape or \
            token.get("probability_shape_yxc") != expected_shape:
        raise ValueError("Semantic token field has the wrong registered shape")
    return token


def _shards(shard_folders: list[Path], chain: dict, token: dict,
            registration: dict, source_root: Path) -> None:
    if not isinstance(shard_folders, list) or len(shard_folders) != chain["shard_count"]:
        raise ValueError("Every semantic shard folder is required")
    source_sha = sha256(_ordinary(source_root / "bora" / "wsi_semantic_probability.py",
                                 "semantic shard producer"))
    rows = []
    seen = set()
    for folder in shard_folders:
        folder = folder.resolve(strict=True)
        if folder in seen:
            raise ValueError("Duplicate semantic shard folder")
        seen.add(folder)
        path = _ordinary(folder / "SHARD_COMPLETE.json", "semantic shard receipt")
        receipt = _json(path)
        probability = _ordinary(folder / "probability.npy", "semantic shard probability")
        window = receipt.get("window")
        if (receipt.get("format") != SHARD_FORMAT or
                receipt.get("token_receipt_sha256") != chain["token_receipt_sha256"] or
                receipt.get("token_probability_sha256") != token["probability_sha256"] or
                receipt.get("model_sha256") != chain["model_sha256"] or
                receipt.get("token_model_binding_sha256") != token["global_model_binding_sha256"] or
                receipt.get("registration_sha256") != chain["registration_sha256"] or
                receipt.get("source_module_sha256") != source_sha or
                receipt.get("global_model_binding_sha256") != chain["global_model_binding_sha256"] or
                any(key not in receipt for key in SHARD_BINDING_FIELDS) or
                receipt.get("global_model_binding_sha256") !=
                    _binding({key: receipt[key] for key in SHARD_BINDING_FIELDS}) or
                receipt.get("source_frame_id") != registration["cell_full_image"]["frame_id"] or
                receipt.get("source_image_sha256") != registration["cell_full_image"]["sha256"] or
                receipt.get("native_shape_yx") != chain["native_shape_yx"] or
                receipt.get("native_mpp_xy") != chain["native_mpp_xy"] or
                receipt.get("class_ids") != chain["class_ids"] or
                receipt.get("class_names") != chain["class_names"] or
                receipt.get("tissue_support_sha256") != chain["tissue_support_sha256"] or
                receipt.get("registered_support_source_sha256") !=
                    chain["registered_support_source_sha256"] or
                receipt.get("partition_independent_projection") is not True or
                receipt.get("independently_fitted") is not False or
                receipt.get("expert_annotation_used_at_inference") is not False or
                receipt.get("native_he_boundary_refinement_applied") is not False or
                not isinstance(window, dict) or
                Path(window.get("probability_path", "")).resolve() != probability or
                window.get("probability_sha256") != receipt.get("probability_sha256") or
                receipt.get("probability_sha256") != sha256(probability) or
                window.get("origin_yx") != receipt.get("origin_yx") or
                window.get("core_yxhw") != receipt.get("core_yxhw") or
                window.get("global_model_binding_sha256") != chain["global_model_binding_sha256"]):
            raise ValueError("Semantic shard source, probability or model binding differs")
        shape = receipt.get("probability_shape_yxc")
        array = np.load(probability, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or list(array.shape) != shape or \
                array.shape[-1] != len(chain["class_ids"]):
            raise ValueError("Semantic shard array shape/dtype differs")
        core = receipt.get("core_yxhw")
        if not isinstance(core, list) or len(core) != 4 or \
                any(type(value) is not int or value < 0 for value in core):
            raise ValueError("Semantic shard core is invalid")
        rows.append((tuple(core[:2]), sha256(path)))
    rows.sort()
    if [sha for _, sha in rows] != chain["shard_receipt_sha256"]:
        raise ValueError("Semantic shard receipt inventory differs from global chain")


def _native_mask_coverage(output: Path, native: dict, class_ids: list[int]) -> dict[int, int]:
    """Check the cleaned native mask in bounded row blocks for class collapse."""
    h, w = native["height"], native["width"]
    for key in ("semantic_mask", "exact_mask"):
        path = output / OUTPUT_PATHS[key]
        if path.stat().st_size != h * w:
            raise ValueError(f"Semantic {key} does not cover the native frame")
    support = np.load(output / OUTPUT_PATHS["support"], mmap_mode="r", allow_pickle=False)
    if support.dtype != np.uint8 or support.shape != (h, w):
        raise ValueError("Semantic support array has the wrong native shape/dtype")
    exact = np.memmap(output / OUTPUT_PATHS["exact_mask"], mode="r",
                      dtype=np.uint8, shape=(h, w))
    counts = {value: 0 for value in class_ids}
    valid = np.zeros(256, dtype=bool)
    valid[[0, *class_ids]] = True
    for y in range(0, h, 128):
        block = exact[y:y + 128]
        support_block = support[y:y + 128]
        if not np.isin(support_block, (0, 1)).all() or \
                not valid[block].all() or np.any(block[~support_block.astype(bool)] != 0):
            raise ValueError("Semantic exact mask has undeclared/out-of-support labels")
        for value in class_ids:
            counts[value] += int(np.count_nonzero(block == value))
    minimum_pixels = math.ceil(100.0 / (native["mpp_x"] * native["mpp_y"]))
    if any(count < minimum_pixels for count in counts.values()):
        raise ValueError("Semantic full-slide output collapsed or lacks a class above the 100 um2 area floor")
    return counts


def verify_semantic_result(roi_spec_path: Path, evaluation_folder: Path,
                           registration_path: Path, support_path: Path,
                           materialization_folder: Path, token_folder: Path,
                           token_source: Path,
                           he_token_source: Path, shard_folders: list[Path],
                           bora_output: Path, source_root: Path) -> tuple[dict, dict, dict]:
    registration = _json(_ordinary(registration_path, "dual-MPP registration"))
    if (registration.get("format") != "cellphenotyper-dual-mpp-registration-v1" or
            registration.get("verified") is not True or
            registration.get("support_sampling") !=
                "absolute pixel centre, nearest, full-frame GrandQC mask"):
        raise ValueError("Verified native/working registration is required")
    native = _frame(registration.get("cell_full_image"), "native")
    working = _frame(registration.get("image"), "working")
    support_ref = registration.get("full_support_mask")
    if (not isinstance(support_ref, dict) or
            [support_ref.get("height"), support_ref.get("width")] !=
                [native["height"], native["width"]] or
            sha256(_ordinary(support_path, "registered GrandQC support")) !=
                _hash(support_ref.get("sha256"), "registered GrandQC support")):
        raise ValueError("Registered GrandQC support hash/shape differs")
    evaluation, spec, _ = _evaluation(roi_spec_path, evaluation_folder,
                                      source_root, registration_path)
    if (spec["inputs"]["native_image"]["sha256"] != native["sha256"] or
            spec["inputs"]["working_image"]["sha256"] != working["sha256"]):
        raise ValueError("Expert ROI image bytes differ from dual-MPP registration")
    chain_path = _ordinary(bora_output / "SEMANTIC_CHAIN_COMPLETE.json", "semantic chain")
    chain = _json(chain_path)
    _same_native(chain, native, "semantic completion")
    class_ids = chain.get("class_ids")
    class_names = chain.get("class_names")
    if (chain.get("format") != CHAIN_FORMAT or chain.get("review_only") is not True or
            chain.get("identity_status") != "exploratory_semantic_transfer" or
            chain.get("tissue_identity_status") != "exploratory_transfer" or
            chain.get("biological_accuracy_qualified") is not False or
            chain.get("expert_annotation_used_at_inference") is not False or
            chain.get("native_he_boundary_refinement_applied") is not False or
            chain.get("registration_sha256") != sha256(registration_path) or
            chain.get("registered_support_source_sha256") != support_ref["sha256"] or
            chain.get("minimum_area_um2") != 100.0 or
            chain.get("exact_vector_matches_cleaned_mask") is not True or
            chain.get("no_macroregion_cuts") is not True or
            not isinstance(class_ids, list) or not 2 <= len(class_ids) <= 16 or
            any(type(value) is not int or value < 1 or value > 253 for value in class_ids) or
            len(set(class_ids)) != len(class_ids) or
            not isinstance(class_names, dict) or
            class_names != evaluation.get("class_names") or
            set(class_names) != {str(value) for value in class_ids} or
            any(not isinstance(name, str) or not name.strip() or
                name.lower().startswith("anonymous") for name in class_names.values()) or
            chain.get("model_sha256") != evaluation["model_sha256"] or
            type(chain.get("shard_count")) is not int or chain["shard_count"] < 1 or
            not isinstance(chain.get("shard_receipt_sha256"), list) or
            len(chain["shard_receipt_sha256"]) != chain["shard_count"] or
            not isinstance(chain.get("output_paths"), dict) or
            chain["output_paths"] != OUTPUT_PATHS or
            not isinstance(chain.get("output_sha256"), dict) or
            set(chain["output_sha256"]) != set(OUTPUT_PATHS)):
        raise ValueError("Semantic chain is incomplete, anonymous, unregistered or non-review-only")
    for digest in chain["shard_receipt_sha256"]:
        _hash(digest, "semantic shard receipt")
    for key, relative in OUTPUT_PATHS.items():
        path = _ordinary(bora_output / relative, key)
        if sha256(path) != _hash(chain["output_sha256"][key], key):
            raise ValueError(f"Semantic output hash differs: {key}")
    if chain["output_sha256"]["support"] != chain.get("tissue_support_sha256"):
        raise ValueError("Semantic native support differs from completion")
    _native_mask_coverage(bora_output, native, class_ids)
    materialized, material_spec = verify_materialization(materialization_folder)
    if (materialized.get("registration_sha256") != chain["registration_sha256"] or
            materialized.get("inventory_sha256") !=
                spec["inputs"]["token_inventory"]["sha256"] or
            materialized.get("checkpoint_sha256") !=
                spec["inputs"]["virchow2_checkpoint"]["sha256"] or
            materialized["source_sha256"].get("working_he") !=
                spec["inputs"]["working_he"]["sha256"] or
            materialized["source_sha256"].get("he_evidence") !=
                spec["inputs"]["he_evidence"]["sha256"] or
            material_spec.get("bora_source_root") != str(source_root) or
            token_source.resolve(strict=True) !=
                (materialization_folder / MATERIALIZED_OUTPUTS["virchow2_tokens"]).resolve(strict=True) or
            he_token_source.resolve(strict=True) !=
                (materialization_folder / MATERIALIZED_OUTPUTS["he_tokens"]).resolve(strict=True)):
        raise ValueError("Semantic token arrays lack the matching frozen inventory/H&E materialization")
    token = _token_receipt(token_folder, token_source, he_token_source,
                           registration, chain, evaluation, source_root,
                           materialization_folder, materialized)
    _shards(shard_folders, chain, token, registration, source_root)
    assembly = _json(bora_output / OUTPUT_PATHS["assembly_receipt"])
    vectors = _json(bora_output / OUTPUT_PATHS["vector_receipt"])
    for name, record in (("assembly", assembly), ("vectors", vectors)):
        _same_native(record, native, name)
        if record.get("global_model_binding_sha256") != chain["global_model_binding_sha256"]:
            raise ValueError(f"Semantic {name} model binding differs")
    if (assembly.get("format") != "bora-wsi-probability-assembly-v1" or
            assembly.get("class_ids") != class_ids or
            assembly.get("tissue_support_sha256") != chain["tissue_support_sha256"] or
            assembly.get("semantic_mask_sha256") != chain["output_sha256"]["semantic_mask"] or
            assembly.get("core_partition_complete") is not True or
            assembly.get("window_count") != chain["shard_count"] or
            assembly.get("independently_fitted_windows") is not False or
            assembly.get("maximum_halo_total_variation") != 0.0 or
            assembly.get("semantic_mask_dtype") != "uint8" or
            vectors.get("format") != "bora-wsi-global-vector-v1" or
            vectors.get("source_semantic_mask_sha256") !=
                chain["output_sha256"]["semantic_mask"] or
            vectors.get("exact_mask_sha256") != chain["output_sha256"]["exact_mask"] or
            vectors.get("tissue_identity_status") != "exploratory_transfer" or
            vectors.get("minimum_area_um2") != 100.0 or
            vectors.get("minimum_area_strictly_less_removed") is not True or
            type(vectors.get("review_component_count")) is not int or
            vectors["review_component_count"] < 1 or
            vectors.get("component_ids_preserved_between_exact_and_review") is not True or
            vectors.get("exact_vector_matches_cleaned_mask") is not True or
            vectors.get("global_contours") is not True or
            vectors.get("independently_fitted_vector_windows") is not False or
            vectors.get("no_macroregion_cuts") is not True or
            vectors.get("biological_accuracy_qualified") is not False or
            vectors.get("output_sha256") != {
                "exact/mask.uint8.dat": chain["output_sha256"]["exact_mask"],
                "exact/mask.geojson": chain["output_sha256"]["exact_geojson"],
                "review/mask.geojson": chain["output_sha256"]["review_geojson"]}):
        raise ValueError("Semantic global vector/assembly parity or area gate failed")
    for role, key, expected_role in (
            ("exact", "exact_geojson", "native_exact_after_area_cleanup"),
            ("review", "review_geojson", "smoothed_review")):
        header = _geojson_header(bora_output / OUTPUT_PATHS[key])
        _same_native(header, native, f"{role} GeoJSON")
        meta = header.get("vector_export")
        if (header.get("type") != "FeatureCollection" or
                header.get("coordinate_space") != "native_image_pixel_edges" or
                header.get("origin") != "top_left" or
                header.get("predicted_not_expert_annotation") is not True or
                header.get("tissue_identity_status") != "exploratory_transfer" or
                header.get("source_semantic_mask_sha256") !=
                    chain["output_sha256"]["semantic_mask"] or
                header.get("global_model_binding_sha256") != chain["global_model_binding_sha256"] or
                not isinstance(meta, dict) or
                meta.get("format") != "bora-wsi-global-vector-v1" or
                meta.get("geometry_role") != expected_role or
                meta.get("global_contours") is not True or
                meta.get("independently_fitted_windows") is not False or
                meta.get("minimum_part_and_hole_area_um2") != 100.0 or
                meta.get("smoothing_tolerance_um") !=
                    (0.0 if role == "exact" else vectors.get("selected_smoothing_tolerance_um"))):
            raise ValueError(f"Semantic {role} GeoJSON lacks registered prediction provenance")
    if (evaluation.get("source_frame_id") != native["frame_id"] or
            evaluation.get("working_frame_id") != working["frame_id"] or
            evaluation.get("class_ids") != class_ids or
            evaluation.get("class_names") != class_names):
        raise ValueError("Expert evaluation frame/classes differ from semantic chain")
    return native, chain, evaluation


def export(base_project: Path, roi_spec_path: Path, evaluation_folder: Path,
           registration_path: Path, support_path: Path,
           materialization_folder: Path, token_folder: Path,
           token_source: Path, he_token_source: Path,
           shard_folders: list[Path], bora_output: Path, source_root: Path,
           destination: Path, *, canonical_tissue_annotations: bool = False) -> dict:
    if canonical_tissue_annotations:
        raise ValueError("Canonical tissue_annotations require verified native H&E refinement and biological review; this chain declares neither")
    paths = (base_project, roi_spec_path, evaluation_folder, registration_path,
             support_path, materialization_folder, token_folder,
             token_source, he_token_source,
             bora_output, source_root, destination,
             *shard_folders)
    if not all(Path(path).is_absolute() for path in paths):
        raise ValueError("Every source and destination path must be absolute")
    destination = destination.resolve(strict=False)
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError("Choose a new project JSON path with an existing parent")
    native, chain, evaluation = verify_semantic_result(
        roi_spec_path, evaluation_folder, registration_path, support_path,
        materialization_folder, token_folder, token_source,
        he_token_source, shard_folders,
        bora_output, source_root)
    previous = validate_experiment(base_project)
    matches = [tissue for tissue in previous["tissues"]
               if tissue["image"].get("sha256") == native["sha256"] and
                  tissue["image"].get("frame_id") == native["frame_id"] and
                  [tissue["image"].get("height"), tissue["image"].get("width")] ==
                    chain["native_shape_yx"] and
                  [tissue["image"].get("mpp_x"), tissue["image"].get("mpp_y")] ==
                    chain["native_mpp_xy"]]
    if len(matches) != 1:
        raise ValueError("Find exactly one project tissue matching semantic native frame/image/MPP")
    tissue = matches[0]
    registered = tissue.get("registered_tissue")
    if registered and registered.get("registration", {}).get("sha256") != chain["registration_sha256"]:
        raise ValueError("Project dual-MPP registration differs from semantic chain")
    chain_path = bora_output / "SEMANTIC_CHAIN_COMPLETE.json"
    artifact_id = "bora-semantic-review-" + sha256(chain_path)[:20]
    if any(item.get("artifact_id") == artifact_id for item in tissue["artifacts"]):
        raise ValueError("This Bora semantic review is already present")
    review = bora_output / OUTPUT_PATHS["review_geojson"]
    tissue["artifacts"].append({
        "artifact_id": artifact_id, "role": ROLE,
        "label": "Predicted Bora tissue boundary · exploratory same-slide holdout",
        "path": str(review), "sha256": chain["output_sha256"]["review_geojson"],
        "format": "geojson", "review_only": True, "training_eligible": False,
        "predicted_not_expert_annotation": True,
        "source_frame_id": native["frame_id"], "image_sha256": native["sha256"],
        "provenance": {
            "producer": CHAIN_FORMAT,
            "prediction_unit": "semantic_virchow2_spatial_token_native_probability",
            "identity_status": "exploratory_same_slide_holdout",
            "evidence_class": "expert_trained_same_slide_holdout_not_independent_patient_validated",
            "predicted_not_expert_annotation": True,
            "native_he_boundary_refinement_applied": False,
            "biological_accuracy_qualified": False,
            "chain_complete_sha256": sha256(chain_path),
            "roi_spec_sha256": sha256(roi_spec_path),
            "expert_annotation_sha256": evaluation["source_sha256"]["expert_geojson"],
            "same_slide_holdout_receipt_sha256": sha256(evaluation_folder / "EVALUATION.json"),
            "model_sha256": chain["model_sha256"],
            "token_materialization_receipt_sha256": sha256(
                materialization_folder / "MATERIALIZATION_COMPLETE.json"),
            "token_inventory_sha256": evaluation["source_sha256"]["token_inventory"],
            "token_receipt_sha256": chain["token_receipt_sha256"],
            "registration_sha256": chain["registration_sha256"],
            "registered_support_source_sha256": chain["registered_support_source_sha256"],
            "tissue_support_sha256": chain["tissue_support_sha256"],
            "semantic_mask_sha256": chain["output_sha256"]["semantic_mask"],
            "exact_mask_sha256": chain["output_sha256"]["exact_mask"],
            "exact_geojson_sha256": chain["output_sha256"]["exact_geojson"],
            "review_geojson_sha256": chain["output_sha256"]["review_geojson"],
            "global_model_binding_sha256": chain["global_model_binding_sha256"],
            "native_shape_yx": chain["native_shape_yx"],
            "native_mpp_xy": chain["native_mpp_xy"],
            "minimum_area_um2": 100.0,
            "exact_vector_matches_cleaned_mask": True,
        },
    })
    result = export_experiment(previous, destination, base_dir=base_project.parent)
    matching = [row for row in result["tissues"] if row["tissue_id"] == tissue["tissue_id"]]
    if len(matching) != 1 or not any(row["artifact_id"] == artifact_id for row in matching[0]["artifacts"]):
        raise ValueError("Published project lost the Bora semantic review boundary")
    return {"schema": "cellphenotyper-bora-semantic-project-export-v1",
            "complete": True, "project": str(destination),
            "project_sha256": sha256(destination),
            "bora_chain_sha256": sha256(chain_path),
            "tissue_id": tissue["tissue_id"], "artifact_id": artifact_id,
            "artifact_role": ROLE, "review_only": True,
            "identity_status": "exploratory_same_slide_holdout"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-project", "roi-spec", "evaluation-folder", "registration",
                 "registered-support-tiff", "materialization-folder",
                 "token-folder", "token-source",
                 "he-token-source", "bora-output",
                 "bora-source-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shard-folder", type=Path, action="append", required=True)
    parser.add_argument("--canonical-tissue-annotations", action="store_true",
                        help="Always refused until a qualified H&E-refined chain exists")
    args = parser.parse_args()
    print(json.dumps(export(args.base_project, args.roi_spec, args.evaluation_folder,
        args.registration, args.registered_support_tiff,
        args.materialization_folder, args.token_folder,
        args.token_source, args.he_token_source, args.shard_folder,
        args.bora_output, args.bora_source_root, args.output,
        canonical_tissue_annotations=args.canonical_tissue_annotations), sort_keys=True))


if __name__ == "__main__":
    main()
