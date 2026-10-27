"""Attach a source-bound continuous Bora boundary as exploratory review only.

This route is deliberately separate from the hard-group Bora exporter.  A
smooth anonymous boundary is neither a semantic tissue annotation nor a
training label.  It requires a completed, native-frame global chain and never
starts inference or edits the source experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

if __package__:
    from .bora_fullslide_action import sha256
    from .export_experiment import export_experiment, validate_experiment
else:
    from bora_fullslide_action import sha256
    from export_experiment import export_experiment, validate_experiment


CHAIN_FORMAT = "bora-wsi-continuous-chain-v1"
PREDICTION_UNIT = "continuous_rgb9_rgb19_native_probability"
EVIDENCE_CLASS = "verified_source_bound_continuous_chain"
OUTPUT_PATHS = {
    "native_support_npy": "support.npy",
    "assembly_receipt": "assembly/ASSEMBLY_COMPLETE.json",
    "semantic_mask": "assembly/semantic-mask.uint8.dat",
    "vector_receipt": "vectors/GLOBAL_VECTOR_COMPLETE.json",
    "exact_geojson": "vectors/exact/mask.geojson",
    "review_geojson": "vectors/review/mask.geojson",
    "exact_mask": "vectors/exact/mask.uint8.dat",
}
HASH = re.compile(r"[0-9a-f]{64}\Z")


def _json(path: Path, *, maximum: int = 2 * 1024 * 1024) -> dict:
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError(f"Missing or oversized Bora metadata: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Bora metadata must be an object: {path}")
    return value


def _hash(value, name: str) -> str:
    if not isinstance(value, str) or HASH.fullmatch(value) is None:
        raise ValueError(f"Invalid {name} SHA-256")
    return value


def _geojson_header(path: Path) -> dict:
    """Read only bounded metadata, not a potentially slide-sized feature list."""
    with path.open("rb") as stream:
        prefix = stream.read(64 * 1024)
    marker = b',"features":['
    end = prefix.find(marker)
    if end < 0:
        raise ValueError("Bora GeoJSON lacks its bounded native metadata header")
    try:
        value = json.loads(prefix[:end] + b"}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Bora GeoJSON native metadata is malformed") from exc
    if not isinstance(value, dict):
        raise ValueError("Bora GeoJSON header is not an object")
    return value


def _frame(record: dict, name: str) -> dict:
    if not isinstance(record, dict):
        raise ValueError(f"Missing registered {name} frame")
    for key in ("height", "width"):
        if type(record.get(key)) is not int or record[key] <= 0:
            raise ValueError(f"Registered {name} frame has invalid {key}")
    for key in ("mpp_x", "mpp_y"):
        number = record.get(key)
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"Registered {name} frame has invalid {key}")
    if not isinstance(record.get("frame_id"), str) or not record["frame_id"]:
        raise ValueError(f"Registered {name} frame lacks an ID")
    _hash(record.get("sha256"), name + " image")
    return record


def _same_native(record: dict, native: dict, name: str) -> None:
    expected = dict(native_shape_yx=[native["height"], native["width"]],
                    native_mpp_xy=[native["mpp_x"], native["mpp_y"]],
                    source_frame_id=native["frame_id"],
                    source_image_sha256=native["sha256"])
    if not isinstance(record, dict) or any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Bora {name} native frame, image, shape or MPP differs from registration")


def verify_continuous_result(registration_path: Path, support_path: Path,
                             output: Path) -> tuple[dict, dict]:
    """Verify immutable chain bytes and native provenance before project export."""
    registration = _json(registration_path)
    if (registration.get("format") != "cellphenotyper-dual-mpp-registration-v1" or
            registration.get("verified") is not True or
            registration.get("support_sampling") !=
                "absolute pixel centre, nearest, full-frame GrandQC mask"):
        raise ValueError("Verified dual-MPP registration is required")
    working = _frame(registration.get("image"), "working")
    native = _frame(registration.get("cell_full_image"), "native")
    for axis in ("x", "y"):
        a = working["width" if axis == "x" else "height"] * working[f"mpp_{axis}"]
        b = native["width" if axis == "x" else "height"] * native[f"mpp_{axis}"]
        if abs(a - b) > max(working[f"mpp_{axis}"], native[f"mpp_{axis}"]):
            raise ValueError("Registered working and native frames differ in physical extent")
    support = registration.get("full_support_mask")
    if (not isinstance(support, dict) or
            [support.get("height"), support.get("width")] !=
                [native["height"], native["width"]] or
            _hash(support.get("sha256"), "registered support") != sha256(support_path)):
        raise ValueError("Registered GrandQC support bytes or native shape changed")
    chain_path = output / "CONTINUOUS_CHAIN_COMPLETE.json"
    chain = _json(chain_path)
    _same_native(chain, native, "completion")
    if (chain.get("format") != CHAIN_FORMAT or
            chain.get("review_only") is not True or
            chain.get("identity_status") != "anonymous_exploratory_review" or
            chain.get("tissue_identity_status") != "exploratory_transfer" or
            chain.get("biological_accuracy_qualified") is not False or
            chain.get("registration_sha256") != sha256(registration_path) or
            chain.get("registered_support_source_sha256") != support["sha256"] or
            chain.get("minimum_area_um2") != 100.0 or
            chain.get("exact_vector_matches_cleaned_mask") is not True or
            chain.get("global_contours") is not True or
            chain.get("no_macroregion_cuts") is not True or
            chain.get("independently_fitted_windows") is not False or
            chain.get("support_conversion") !=
                "registered-native-GrandQC-TIFF/integer-binary-to-uint8-NPY/v1"):
        raise ValueError("Bora continuous completion lacks source or review-only geometry gates")
    if (chain.get("output_paths") != OUTPUT_PATHS or
            not isinstance(chain.get("output_sha256"), dict) or
            set(chain["output_sha256"]) != set(OUTPUT_PATHS)):
        raise ValueError("Bora continuous completion lacks exact output inventory")
    for key, relative in OUTPUT_PATHS.items():
        path = output / relative
        expected = _hash(chain["output_sha256"][key], key)
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"Bora continuous output hash changed: {key}")
    for key, output_key in (("tissue_support_sha256", "native_support_npy"),
                            ("assembly_receipt_sha256", "assembly_receipt"),
                            ("semantic_mask_sha256", "semantic_mask"),
                            ("vector_receipt_sha256", "vector_receipt"),
                            ("exact_geojson_sha256", "exact_geojson"),
                            ("review_geojson_sha256", "review_geojson"),
                            ("exact_mask_sha256", "exact_mask")):
        if chain.get(key) != chain["output_sha256"][output_key]:
            raise ValueError(f"Bora continuous completion {key} differs from output")
    class_ids = chain.get("class_ids")
    names = chain.get("class_names")
    if (not isinstance(class_ids, list) or not 2 <= len(class_ids) <= 32 or
            any(type(value) is not int or not 1 <= value <= 255 for value in class_ids) or
            len(set(class_ids)) != len(class_ids) or
            not isinstance(names, dict) or
            names != {str(value): f"anonymous group {index}"
                      for index, value in enumerate(class_ids, 1)} or
            type(chain.get("shard_count")) is not int or chain["shard_count"] < 1 or
            not isinstance(chain.get("shard_receipt_sha256"), list) or
            len(chain["shard_receipt_sha256"]) != chain["shard_count"]):
        raise ValueError("Bora continuous class or shard inventory is invalid")
    for digest in chain["shard_receipt_sha256"]:
        _hash(digest, "shard receipt")
    model_sha = _hash(chain.get("global_model_binding_sha256"), "global model binding")
    assembly = _json(output / OUTPUT_PATHS["assembly_receipt"])
    vectors = _json(output / OUTPUT_PATHS["vector_receipt"])
    for label, record in (("assembly", assembly), ("vector", vectors)):
        _same_native(record, native, label)
        if record.get("global_model_binding_sha256") != model_sha:
            raise ValueError(f"Bora {label} model binding differs from completion")
    if (assembly.get("format") != "bora-wsi-probability-assembly-v1" or
            assembly.get("class_ids") != class_ids or
            assembly.get("tissue_support_sha256") != chain["tissue_support_sha256"] or
            assembly.get("semantic_mask_sha256") != chain["semantic_mask_sha256"] or
            assembly.get("core_partition_complete") is not True or
            assembly.get("window_count") != chain["shard_count"] or
            assembly.get("independently_fitted_windows") is not False or
            assembly.get("maximum_halo_total_variation") != 0.0 or
            assembly.get("semantic_mask_dtype") != "uint8" or
            assembly.get("biological_accuracy_qualified") is not False):
        raise ValueError("Bora probability assembly is incomplete or source-mismatched")
    if (vectors.get("format") != "bora-wsi-global-vector-v1" or
            vectors.get("source_semantic_mask_sha256") != chain["semantic_mask_sha256"] or
            vectors.get("exact_mask_sha256") != chain["exact_mask_sha256"] or
            vectors.get("tissue_identity_status") != "exploratory_transfer" or
            vectors.get("minimum_area_um2") != 100.0 or
            vectors.get("minimum_area_strictly_less_removed") is not True or
            vectors.get("component_ids_preserved_between_exact_and_review") is not True or
            vectors.get("exact_vector_matches_cleaned_mask") is not True or
            vectors.get("global_contours") is not True or
            vectors.get("independently_fitted_vector_windows") is not False or
            vectors.get("no_macroregion_cuts") is not True or
            vectors.get("biological_accuracy_qualified") is not False or
            vectors.get("output_sha256") != {
                "exact/mask.uint8.dat": chain["exact_mask_sha256"],
                "exact/mask.geojson": chain["exact_geojson_sha256"],
                "review/mask.geojson": chain["review_geojson_sha256"]}):
        raise ValueError("Bora global vectors lack exact cleanup or review provenance")
    for role, path_key, geometry_role in (
            ("exact", "exact_geojson", "native_exact_after_area_cleanup"),
            ("review", "review_geojson", "smoothed_review")):
        header = _geojson_header(output / OUTPUT_PATHS[path_key])
        _same_native(header, native, f"{role} GeoJSON")
        export = header.get("vector_export")
        if (header.get("type") != "FeatureCollection" or
                header.get("coordinate_space") != "native_image_pixel_edges" or
                header.get("origin") != "top_left" or
                header.get("predicted_not_expert_annotation") is not True or
                header.get("tissue_identity_status") != "exploratory_transfer" or
                header.get("source_semantic_mask_sha256") != chain["semantic_mask_sha256"] or
                header.get("global_model_binding_sha256") != model_sha or
                not isinstance(export, dict) or
                export.get("format") != "bora-wsi-global-vector-v1" or
                export.get("geometry_role") != geometry_role or
                export.get("global_contours") is not True or
                export.get("independently_fitted_windows") is not False or
                export.get("minimum_part_and_hole_area_um2") != 100.0):
            raise ValueError(f"Bora {role} GeoJSON lacks native review provenance")
        if role == "exact" and export.get("smoothing_tolerance_um") != 0.0:
            raise ValueError("Bora exact GeoJSON has smoothing applied")
        if role == "review" and export.get("smoothing_tolerance_um") != vectors.get("selected_smoothing_tolerance_um"):
            raise ValueError("Bora review geometry differs from selected smoothing")
    return native, chain


def export(base_project: Path, registration_path: Path, support_path: Path,
           bora_output: Path, destination: Path) -> dict:
    paths = (base_project, registration_path, support_path, bora_output, destination)
    if not all(path.is_absolute() for path in paths):
        raise ValueError("All project, registration, support and Bora paths must be absolute")
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError("Choose a new project JSON path with an existing parent")
    base_project, registration_path, support_path, bora_output = (
        path.resolve(strict=True) for path in paths[:4])
    destination = destination.resolve(strict=False)
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError("Choose a new project JSON path with an existing parent")
    native, chain = verify_continuous_result(registration_path, support_path, bora_output)
    previous = validate_experiment(base_project)
    matches = [tissue for tissue in previous["tissues"]
               if tissue["image"].get("sha256") == native["sha256"] and
                  tissue["image"].get("frame_id") == native["frame_id"] and
                  [tissue["image"].get("height"), tissue["image"].get("width")] ==
                    chain["native_shape_yx"] and
                  [tissue["image"].get("mpp_x"), tissue["image"].get("mpp_y")] ==
                    chain["native_mpp_xy"]]
    if len(matches) != 1:
        raise ValueError("Find exactly one project tissue in the Bora native image/frame/shape/MPP")
    tissue = matches[0]
    registered = tissue.get("registered_tissue")
    if registered is not None:
        reference = registered.get("registration", {})
        if reference.get("sha256") != chain["registration_sha256"]:
            raise ValueError("Project dual-MPP registration differs from the Bora chain")
    chain_path = bora_output / "CONTINUOUS_CHAIN_COMPLETE.json"
    artifact_id = "bora-continuous-review-" + sha256(chain_path)[:20]
    if any(item.get("artifact_id") == artifact_id for item in tissue["artifacts"]):
        raise ValueError("This Bora continuous review boundary is already in the project")
    review = bora_output / OUTPUT_PATHS["review_geojson"]
    tissue["artifacts"].append({
        "artifact_id": artifact_id,
        "role": "bora_exploratory_boundary_review",
        "label": "Bora continuous boundary (anonymous exploratory review)",
        "path": str(review), "sha256": chain["review_geojson_sha256"],
        "format": "geojson", "review_only": True,
        "source_frame_id": native["frame_id"],
        "image_sha256": native["sha256"],
        "provenance": {
            "producer": CHAIN_FORMAT,
            "prediction_unit": PREDICTION_UNIT,
            "identity_status": "anonymous_exploratory_review",
            "evidence_class": EVIDENCE_CLASS,
            "geometry": "smoothed global review of exact native mask; parts/holes <100 um2 removed",
            "chain_complete_sha256": sha256(chain_path),
            "registration_sha256": chain["registration_sha256"],
            "registered_support_source_sha256": chain["registered_support_source_sha256"],
            "tissue_support_sha256": chain["tissue_support_sha256"],
            "semantic_mask_sha256": chain["semantic_mask_sha256"],
            "exact_mask_sha256": chain["exact_mask_sha256"],
            "exact_geojson_sha256": chain["exact_geojson_sha256"],
            "review_geojson_sha256": chain["review_geojson_sha256"],
            "global_model_binding_sha256": chain["global_model_binding_sha256"],
            "native_shape_yx": chain["native_shape_yx"],
            "native_mpp_xy": chain["native_mpp_xy"],
            "minimum_area_um2": 100.0,
            "biological_accuracy_qualified": False,
            "exact_vector_matches_cleaned_mask": True,
        },
    })
    result = export_experiment(previous, destination, base_dir=base_project.parent)
    matching = [row for row in result["tissues"] if row["tissue_id"] == tissue["tissue_id"]]
    if len(matching) != 1 or not any(row["artifact_id"] == artifact_id for row in matching[0]["artifacts"]):
        raise ValueError("Published project lost the Bora continuous review boundary")
    return {"schema": "cellphenotyper-bora-continuous-project-export-v1",
            "complete": True, "project": str(destination),
            "project_sha256": sha256(destination),
            "bora_chain_sha256": sha256(chain_path),
            "tissue_id": tissue["tissue_id"], "artifact_id": artifact_id,
            "artifact_role": "bora_exploratory_boundary_review",
            "review_only": True, "prediction_unit": PREDICTION_UNIT}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-project", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registered-support-tiff", type=Path, required=True)
    parser.add_argument("--bora-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.base_project, args.registration,
        args.registered_support_tiff, args.bora_output, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
