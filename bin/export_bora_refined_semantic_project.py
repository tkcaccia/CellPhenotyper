"""Publish a verified H&E-refined Bora semantic boundary as locked project review.

The exporter does no inference. It requires the independently evaluated expert
model, complete materialized token inventory, original semantic chain and the
native H&E-refined chain. Same-slide validation remains exploratory; neither
the predicted polygons nor their class names become expert annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .export_bora_semantic_project import (
        OUTPUT_PATHS, ROLE, _binding, _native_mask_coverage, _ordinary,
        verify_semantic_result,
    )
    from .export_bora_continuous_project import _frame, _geojson_header, _hash, _json, _same_native
    from .export_experiment import export_experiment, sha256, validate_experiment
else:
    from export_bora_semantic_project import (
        OUTPUT_PATHS, ROLE, _binding, _native_mask_coverage, _ordinary,
        verify_semantic_result,
    )
    from export_bora_continuous_project import _frame, _geojson_header, _hash, _json, _same_native
    from export_experiment import export_experiment, sha256, validate_experiment


CHAIN_FORMAT = "bora-wsi-native-he-refined-semantic-chain-v1"
SHARD_FORMAT = "bora-wsi-native-he-refined-semantic-shard-v1"
PROJECT_FORMAT = "cellphenotyper-bora-refined-semantic-project-export-v1"
REFINED_BINDING_FIELDS = (
    "format", "token_model_binding_sha256", "model_sha256",
    "registration_sha256", "native_he_source_sha256",
    "tissue_support_sha256", "class_ids", "class_names",
    "source_frame_id", "source_image_sha256", "native_shape_yx",
    "native_mpp_xy", "refinement_spec", "source_module_sha256",
    "source_semantic_module_sha256", "output_halo_pixels",
    "categorical_scores",
)


def verify_refined_result(roi_spec_path: Path, evaluation_folder: Path,
                          registration_path: Path, support_path: Path,
                          materialization_folder: Path, token_folder: Path,
                          token_source: Path, he_token_source: Path,
                          source_shard_folders: list[Path],
                          refined_shard_folders: list[Path], source_output: Path,
                          refined_output: Path, source_root: Path) -> tuple[dict, dict, dict]:
    """Verify both completed chains and every source/refined shard binding."""
    native, source, evaluation = verify_semantic_result(
        roi_spec_path, evaluation_folder, registration_path, support_path,
        materialization_folder, token_folder, token_source, he_token_source,
        source_shard_folders, source_output, source_root)
    registration = _json(_ordinary(registration_path, "dual-MPP registration"))
    refined_path = _ordinary(refined_output / "REFINED_SEMANTIC_CHAIN_COMPLETE.json",
                             "refined semantic completion")
    chain = _json(refined_path)
    _same_native(chain, native, "refined semantic completion")
    support_sha = source["tissue_support_sha256"]
    native_he_path = Path(_json(roi_spec_path)["inputs"]["native_image"]["path"])
    source_sha = sha256(_ordinary(source_root / "bora" / "wsi_semantic_probability.py",
                                  "semantic scorer source"))
    refinement_sha = sha256(_ordinary(source_root / "bora" / "wsi_semantic_refinement.py",
                                      "H&E refinement source"))
    _ordinary(source_root / "bora" / "wsi_semantic_refined_chain.py", "refined chain source")
    if (chain.get("format") != CHAIN_FORMAT or chain.get("review_only") is not True or
            chain.get("identity_status") != "exploratory_semantic_transfer" or
            chain.get("tissue_identity_status") != "exploratory_transfer" or
            chain.get("biological_accuracy_qualified") is not False or
            chain.get("native_he_boundary_refinement_applied") is not True or
            chain.get("expert_annotation_used_at_inference") is not False or
            chain.get("native_he_source_sha256") != native["sha256"] or
            sha256(_ordinary(native_he_path, "registered native H&E")) != native["sha256"] or
            chain.get("registration_sha256") != sha256(registration_path) or
            chain.get("registered_support_source_sha256") !=
                registration["full_support_mask"]["sha256"] or
            chain.get("tissue_support_sha256") != support_sha or
            chain.get("model_sha256") != source["model_sha256"] or
            chain.get("token_source_sha256") != source["token_source_sha256"] or
            chain.get("he_token_source_sha256") != source["he_token_source_sha256"] or
            chain.get("token_probability_sha256") != source["token_probability_sha256"] or
            chain.get("token_receipt_sha256") != source["token_receipt_sha256"] or
            chain.get("class_ids") != source["class_ids"] or
            chain.get("class_names") != evaluation["class_names"] or
            chain.get("minimum_area_um2") != 100.0 or
            chain.get("exact_vector_matches_cleaned_mask") is not True or
            chain.get("no_macroregion_cuts") is not True or
            chain.get("output_paths") != OUTPUT_PATHS or
            not isinstance(chain.get("output_sha256"), dict) or
            set(chain["output_sha256"]) != set(OUTPUT_PATHS) or
            type(chain.get("shard_count")) is not int or
            chain["shard_count"] != source["shard_count"] or
            not isinstance(chain.get("source_shard_receipt_sha256"), list) or
            not isinstance(chain.get("refined_shard_receipt_sha256"), list) or
            len(chain["source_shard_receipt_sha256"]) != chain["shard_count"] or
            len(chain["refined_shard_receipt_sha256"]) != chain["shard_count"] or
            not isinstance(chain.get("global_model_binding_sha256"), str) or
            len(chain["global_model_binding_sha256"]) != 64 or
            not isinstance(chain.get("refinement_spec"), dict)):
        raise ValueError("Refined semantic chain is incomplete or source-mismatched")
    if (len(source_shard_folders) != chain["shard_count"] or
            len(refined_shard_folders) != chain["shard_count"] or
            len({Path(p).resolve() for p in refined_shard_folders}) != chain["shard_count"]):
        raise ValueError("Complete distinct refined/source shard inventories are required")
    support = np.load(refined_output / OUTPUT_PATHS["support"],
                      mmap_mode="r", allow_pickle=False)
    source_by_core = {}
    for folder in source_shard_folders:
        receipt_path = _ordinary(Path(folder) / "SHARD_COMPLETE.json", "source shard")
        receipt = _json(receipt_path)
        core = tuple(receipt.get("core_yxhw", ()))
        if core in source_by_core:
            raise ValueError("Duplicate source semantic core")
        source_by_core[core] = (Path(folder).resolve(), receipt, sha256(receipt_path))
    pairs = []
    for folder in refined_shard_folders:
        folder = Path(folder).resolve(strict=True)
        receipt_path = _ordinary(folder / "SHARD_COMPLETE.json", "refined shard")
        receipt = _json(receipt_path)
        core = tuple(receipt.get("core_yxhw", ()))
        if core not in source_by_core:
            raise ValueError("Refined semantic core has no source shard")
        source_folder, source_receipt, source_receipt_sha = source_by_core[core]
        scores_path = _ordinary(folder / "probability.npy", "refined probabilities")
        window = receipt.get("window")
        scores = np.load(scores_path, mmap_mode="r", allow_pickle=False)
        y, x, height, width = core if len(core) == 4 else (None,) * 4
        halo = receipt.get("output_halo_pixels")
        if (any(type(v) is not int for v in (y, x, height, width, halo)) or
                min(y, x) < 0 or min(height, width) < 1 or not 1 <= halo <= 256 or
                y + height > native["height"] or x + width > native["width"]):
            raise ValueError("Refined semantic core/halo is outside the native frame")
        oy, ox = max(0, y - halo), max(0, x - halo)
        ey, ex = min(native["height"], y + height + halo), \
                 min(native["width"], x + width + halo)
        context = receipt.get("required_context_yx")
        source_y, source_x = source_receipt["origin_yx"]
        source_h, source_w = source_receipt["probability_shape_yxc"][:2]
        if (not isinstance(context, list) or len(context) != 2 or
                any(type(v) is not int or v < 0 for v in context) or
                source_y > max(0, oy - context[0]) or
                source_x > max(0, ox - context[1]) or
                source_y + source_h < min(native["height"], ey + context[0]) or
                source_x + source_w < min(native["width"], ex + context[1])):
            raise ValueError("Refined shard lacks source H&E context halo")
        fields = {field: receipt.get(field) for field in REFINED_BINDING_FIELDS}
        boundary = receipt.get("boundary_refinement")
        if (receipt.get("format") != SHARD_FORMAT or
                any(field not in receipt for field in REFINED_BINDING_FIELDS) or
                receipt.get("global_model_binding_sha256") != _binding(fields) or
                receipt.get("source_shard_receipt_sha256") != source_receipt_sha or
                receipt.get("source_shard_probability_sha256") !=
                    source_receipt["probability_sha256"] or
                Path(receipt.get("source_shard_path", "")).resolve() != source_folder or
                receipt.get("source_input_origin_yx") != source_receipt["origin_yx"] or
                receipt.get("source_input_shape_yxc") !=
                    source_receipt["probability_shape_yxc"] or
                receipt.get("token_model_binding_sha256") !=
                    source_receipt["token_model_binding_sha256"] or
                receipt.get("source_semantic_module_sha256") != source_sha or
                receipt.get("source_module_sha256") != refinement_sha or
                receipt.get("global_model_binding_sha256") !=
                    chain["global_model_binding_sha256"] or
                receipt.get("registration_sha256") != chain["registration_sha256"] or
                receipt.get("native_he_source_sha256") != native["sha256"] or
                receipt.get("tissue_support_sha256") != support_sha or
                receipt.get("model_sha256") != chain["model_sha256"] or
                receipt.get("class_ids") != chain["class_ids"] or
                receipt.get("class_names") != chain["class_names"] or
                receipt.get("refinement_spec") != chain["refinement_spec"] or
                receipt.get("source_frame_id") != native["frame_id"] or
                receipt.get("source_image_sha256") != native["sha256"] or
                receipt.get("native_shape_yx") != chain["native_shape_yx"] or
                receipt.get("native_mpp_xy") != chain["native_mpp_xy"] or
                receipt.get("native_he_boundary_refinement_applied") is not True or
                receipt.get("expert_annotation_used_at_inference") is not False or
                receipt.get("biological_accuracy_qualified") is not False or
                receipt.get("partition_invariant_by_finite_halo") is not True or
                receipt.get("categorical_scores") !=
                    "one-hot-inside-support/uniform-outside" or
                receipt.get("origin_yx") != [oy, ox] or
                not isinstance(window, dict) or
                Path(window.get("probability_path", "")).resolve() != scores_path or
                window.get("probability_sha256") != receipt.get("probability_sha256") or
                window.get("global_model_binding_sha256") !=
                    chain["global_model_binding_sha256"] or
                window.get("core_yxhw") != list(core) or
                window.get("origin_yx") != [oy, ox] or
                window.get("independently_fitted") is not False or
                receipt.get("probability_sha256") != sha256(scores_path) or
                scores.dtype != np.float32 or
                list(scores.shape) != [ey - oy, ex - ox, len(chain["class_ids"])] or
                list(scores.shape) != receipt.get("probability_shape_yxc") or
                not isinstance(boundary, dict) or
                boundary.get("method") != "bora-local-he-boundary-v1" or
                boundary.get("class_ids") != chain["class_ids"] or
                boundary.get("mpp_xy") != chain["native_mpp_xy"] or
                boundary.get("parameters") != chain["refinement_spec"] or
                boundary.get("processing_core_size") != 512 or
                boundary.get("changed_outside_movable_pixels") != 0 or
                boundary.get("partition_invariant_by_finite_halo") is not True or
                boundary.get("expert_annotation_used") is not False or
                boundary.get("biological_accuracy_validated") is not False):
            raise ValueError("Refined shard probability or source/H&E binding differs")
        for offset in range(0, scores.shape[0], 128):
            block = scores[offset:offset + 128]
            tissue = np.asarray(support[oy + offset:oy + offset + len(block),
                                        ox:ex]).astype(bool)
            if (not np.isfinite(block).all() or
                    not np.allclose(block.sum(-1), 1, rtol=0, atol=2e-6) or
                    np.any((block[tissue] != 0) & (block[tissue] != 1)) or
                    not np.allclose(block[~tissue], 1 / block.shape[-1],
                                    rtol=0, atol=2e-6)):
                raise ValueError("Refined shard is not categorical on tissue support")
        pairs.append((tuple(core[:2]), source_receipt_sha, sha256(receipt_path)))
    pairs.sort()
    if ([item[1] for item in pairs] != chain["source_shard_receipt_sha256"] or
            [item[2] for item in pairs] != chain["refined_shard_receipt_sha256"] or
            sorted(source["shard_receipt_sha256"]) !=
                sorted(chain["source_shard_receipt_sha256"])):
        raise ValueError("Refined chain shard inventories differ from semantic source")
    for key, relative in OUTPUT_PATHS.items():
        path = _ordinary(refined_output / relative, f"refined {key}")
        if sha256(path) != _hash(chain["output_sha256"].get(key), key):
            raise ValueError(f"Refined semantic output hash differs: {key}")
    if chain["output_sha256"]["support"] != support_sha:
        raise ValueError("Refined support differs from registered GrandQC support")
    _native_mask_coverage(refined_output, native, chain["class_ids"])
    assembly = _json(refined_output / OUTPUT_PATHS["assembly_receipt"])
    vectors = _json(refined_output / OUTPUT_PATHS["vector_receipt"])
    for label, record in (("assembly", assembly), ("vectors", vectors)):
        _same_native(record, native, f"refined {label}")
        if record.get("global_model_binding_sha256") != chain["global_model_binding_sha256"]:
            raise ValueError(f"Refined {label} model binding differs")
    if (assembly.get("format") != "bora-wsi-probability-assembly-v1" or
            assembly.get("class_ids") != chain["class_ids"] or
            assembly.get("tissue_support_sha256") != support_sha or
            assembly.get("semantic_mask_sha256") !=
                chain["output_sha256"]["semantic_mask"] or
            assembly.get("core_partition_complete") is not True or
            assembly.get("window_count") != chain["shard_count"] or
            assembly.get("maximum_halo_total_variation") != 0.0 or
            vectors.get("format") != "bora-wsi-global-vector-v1" or
            vectors.get("source_semantic_mask_sha256") !=
                chain["output_sha256"]["semantic_mask"] or
            vectors.get("exact_mask_sha256") != chain["output_sha256"]["exact_mask"] or
            vectors.get("exact_vector_matches_cleaned_mask") is not True or
            vectors.get("minimum_area_um2") != 100.0 or
            vectors.get("minimum_area_strictly_less_removed") is not True or
            vectors.get("global_contours") is not True or
            vectors.get("no_macroregion_cuts") is not True or
            vectors.get("biological_accuracy_qualified") is not False or
            vectors.get("output_sha256") != {
                "exact/mask.uint8.dat": chain["output_sha256"]["exact_mask"],
                "exact/mask.geojson": chain["output_sha256"]["exact_geojson"],
                "review/mask.geojson": chain["output_sha256"]["review_geojson"]}):
        raise ValueError("Refined global vector or assembly parity failed")
    for role, key, expected in (("exact", "exact_geojson", "native_exact_after_area_cleanup"),
                                 ("review", "review_geojson", "smoothed_review")):
        header = _geojson_header(refined_output / OUTPUT_PATHS[key])
        _same_native(header, native, f"refined {role} GeoJSON")
        meta = header.get("vector_export")
        if (header.get("type") != "FeatureCollection" or
                header.get("coordinate_space") != "native_image_pixel_edges" or
                header.get("predicted_not_expert_annotation") is not True or
                header.get("source_semantic_mask_sha256") !=
                    chain["output_sha256"]["semantic_mask"] or
                header.get("global_model_binding_sha256") !=
                    chain["global_model_binding_sha256"] or
                not isinstance(meta, dict) or
                meta.get("format") != "bora-wsi-global-vector-v1" or
                meta.get("geometry_role") != expected or
                meta.get("global_contours") is not True or
                meta.get("minimum_part_and_hole_area_um2") != 100.0):
            raise ValueError(f"Refined {role} GeoJSON lacks prediction provenance")
    return native, chain, evaluation


def export(base_project: Path, roi_spec_path: Path, evaluation_folder: Path,
           registration_path: Path, support_path: Path,
           materialization_folder: Path, token_folder: Path,
           token_source: Path, he_token_source: Path,
           source_shard_folders: list[Path], refined_shard_folders: list[Path],
           source_output: Path, refined_output: Path, source_root: Path,
           destination: Path, *, canonical_tissue_annotations: bool = False) -> dict:
    if canonical_tissue_annotations:
        raise ValueError("Independent-patient biological validation and expert review are required for canonical tissue_annotations")
    paths = (base_project, roi_spec_path, evaluation_folder, registration_path,
             support_path, materialization_folder, token_folder, token_source,
             he_token_source, source_output, refined_output, source_root,
             destination, *source_shard_folders, *refined_shard_folders)
    if not all(Path(path).is_absolute() for path in paths):
        raise ValueError("Every source and destination path must be absolute")
    destination = destination.resolve(strict=False)
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError("Choose a new project JSON path with an existing parent")
    native, chain, evaluation = verify_refined_result(
        roi_spec_path, evaluation_folder, registration_path, support_path,
        materialization_folder, token_folder, token_source, he_token_source,
        source_shard_folders, refined_shard_folders, source_output,
        refined_output, source_root)
    previous = validate_experiment(base_project)
    matches = [tissue for tissue in previous["tissues"]
               if tissue["image"].get("sha256") == native["sha256"] and
                  tissue["image"].get("frame_id") == native["frame_id"] and
                  [tissue["image"].get("height"), tissue["image"].get("width")] ==
                    chain["native_shape_yx"] and
                  [tissue["image"].get("mpp_x"), tissue["image"].get("mpp_y")] ==
                    chain["native_mpp_xy"]]
    if len(matches) != 1:
        raise ValueError("Find exactly one project tissue matching refined native frame/image/MPP")
    tissue = matches[0]
    registered = tissue.get("registered_tissue")
    if registered and registered.get("registration", {}).get("sha256") != chain["registration_sha256"]:
        raise ValueError("Project registration differs from refined chain")
    chain_path = refined_output / "REFINED_SEMANTIC_CHAIN_COMPLETE.json"
    artifact_id = "bora-refined-semantic-review-" + sha256(chain_path)[:20]
    if any(item.get("artifact_id") == artifact_id for item in tissue["artifacts"]):
        raise ValueError("This refined Bora review is already in the project")
    review_path = refined_output / OUTPUT_PATHS["review_geojson"]
    tissue["artifacts"].append({
        "artifact_id": artifact_id, "role": ROLE,
        "label": "Predicted by Bora · H&E-refined exploratory tissue boundary",
        "path": str(review_path), "sha256": chain["output_sha256"]["review_geojson"],
        "format": "geojson", "review_only": True, "training_eligible": False,
        "predicted_not_expert_annotation": True,
        "source_frame_id": native["frame_id"], "image_sha256": native["sha256"],
        "provenance": {
            "producer": CHAIN_FORMAT,
            "prediction_unit": "semantic_virchow2_spatial_token_native_he_refined_probability",
            "identity_status": "exploratory_same_slide_holdout",
            "evidence_class": "expert_trained_same_slide_holdout_not_independent_patient_validated",
            "predicted_not_expert_annotation": True,
            "native_he_boundary_refinement_applied": True,
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
            "native_he_source_sha256": native["sha256"],
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
    status = ("H&E-refined Bora tissue boundaries available as locked exploratory "
              "predictions; same-slide holdout passed, independent-patient validation pending.")
    previous.setdefault("settings", {})["bora_status"] = status
    previous.setdefault("provenance", {})["bora_status"] = status
    interpretation = previous["provenance"].get("interpretation")
    if isinstance(interpretation, str) and "no bora" in interpretation.lower():
        previous["provenance"]["interpretation"] = status
    result = export_experiment(previous, destination, base_dir=base_project.parent)
    if not any(row["tissue_id"] == tissue["tissue_id"] and
               any(item["artifact_id"] == artifact_id for item in row["artifacts"])
               for row in result["tissues"]):
        raise ValueError("Exported project lost refined Bora review")
    return {"schema": PROJECT_FORMAT, "complete": True,
            "project_path": str(destination), "project_sha256": sha256(destination),
            "artifact_id": artifact_id, "review_only": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-project", "roi-spec", "evaluation-folder", "registration",
                 "support", "materialization-folder", "token-folder", "token-source",
                 "he-token-source", "source-output", "refined-output",
                 "source-root", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-shard", type=Path, action="append", required=True)
    parser.add_argument("--refined-shard", type=Path, action="append", required=True)
    args = parser.parse_args()
    report = export(args.base_project, args.roi_spec, args.evaluation_folder,
                    args.registration, args.support, args.materialization_folder,
                    args.token_folder, args.token_source, args.he_token_source,
                    args.source_shard, args.refined_shard, args.source_output,
                    args.refined_output, args.source_root, args.destination)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
