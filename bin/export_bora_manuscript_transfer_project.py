"""Attach a verified manuscript-transfer contour as locked exploratory review.

The existing evaluated Bora route is deliberately unaffected.  This exporter
never promotes a prediction into expert annotations or training ROIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .bora_fullslide_action import _geojson_header, sha256
    from .export_experiment import export_experiment, validate_experiment
else:
    from bora_fullslide_action import _geojson_header, sha256
    from export_experiment import export_experiment, validate_experiment


TOKEN_FORMAT = "bora-exploratory-manuscript-transfer-tokens-v1"
ASSEMBLY_FORMAT = "bora-exploratory-transfer-native-assembly-v1"
VECTOR_FORMAT = "bora-exploratory-transfer-unknown-aware-global-vector-v1"
ROLE = "bora_exploratory_boundary_review"
IDENTITY = "exploratory_manuscript_transfer_unvalidated"
SUPPORT_SAMPLING = "absolute pixel centre, nearest, full-frame GrandQC mask"
SUPPORT_CONVERSION = "registered-GrandQC-TIFF/absolute-centre-nearest-binary-to-native-uint8-NPY/v2"


def _verify_registered_support(registration: dict, source_tiff: Path,
                               converted_npy: Path) -> str:
    """Recheck absolute-centre nearest sampling at every native pixel."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    native = registration["cell_full_image"]
    shape = (native["height"], native["width"])
    source = registration.get("full_support_mask", {})
    source_shape = (source.get("height"), source.get("width"))
    if (registration.get("support_sampling") != SUPPORT_SAMPLING or
            any(type(value) is not int or value < 1 for value in source_shape) or
            any(source_size > native_size for source_size, native_size
                in zip(source_shape, shape)) or
            not converted_npy.is_file() or converted_npy.is_symlink()):
        raise ValueError("Registered native support frame or conversion is missing")
    support = np.load(converted_npy, mmap_mode="r", allow_pickle=False)
    if support.shape != shape or support.dtype != np.uint8:
        raise ValueError("Converted native support must be binary uint8 NPY")
    source_h, source_w = source_shape
    native_h, native_w = shape
    with rasterio.open(source_tiff) as dataset:
        if (dataset.count != 1 or dataset.dtypes[0] not in ("uint8", "uint16", "bool") or
                (dataset.height, dataset.width) != source_shape):
            raise ValueError("Registered GrandQC support TIFF frame differs")
        allowed = ((0, 1, 255, 65535) if dataset.dtypes[0] == "uint16" else (0, 1, 255))
        for y in range(0, native_h, 128):
            y1 = min(y + 128, native_h)
            sy = ((2*np.arange(y, y1, dtype=np.int64)+1)*source_h)//(2*native_h)
            for x in range(0, native_w, 4096):
                x1 = min(x + 4096, native_w)
                sx = ((2*np.arange(x, x1, dtype=np.int64)+1)*source_w)//(2*native_w)
                sy0, sy1 = int(sy[0]), int(sy[-1]) + 1
                sx0, sx1 = int(sx[0]), int(sx[-1]) + 1
                raw = dataset.read(1, window=Window(sx0, sy0, sx1-sx0, sy1-sy0))
                binary = np.asarray(support[y:y1, x:x1])
                expected = raw[np.ix_(sy-sy0, sx-sx0)] > 0
                if (not np.isin(raw, allowed).all() or np.any(binary > 1) or
                        not np.array_equal(expected, binary.astype(bool))):
                    raise ValueError("Native support NPY differs from registered absolute-centre sampling")
    return sha256(converted_npy)


def _receipt(path: Path, expected: str, *, review_only: bool = True) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024**2:
        raise ValueError(f"Missing ordinary Bora completion receipt: {path}")
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or value.get("format") != expected or
            value.get("complete") is not True or
            review_only and value.get("biological_accuracy_qualified") is not False):
        raise ValueError(f"Incomplete or misidentified Bora result: {path}")
    return value


def verify_result(registration_path: Path, materialization_folder: Path,
                  support_path: Path, result_folder: Path) -> tuple[dict, dict]:
    """Recheck byte, frame, unknown and global-vector provenance for export."""
    paths = (registration_path, materialization_folder, support_path, result_folder)
    if not all(path.is_absolute() for path in paths):
        raise ValueError("Use absolute registered Bora input and result paths")
    registration_path, materialization_folder, support_path, result_folder = (
        path.resolve(strict=True) for path in paths)
    registration = json.loads(registration_path.read_text())
    if (registration.get("format") != "cellphenotyper-dual-mpp-registration-v1" or
            registration.get("verified") is not True):
        raise ValueError("Verified dual-MPP registration required")
    native = registration.get("cell_full_image")
    if not isinstance(native, dict) or not isinstance(native.get("sha256"), str):
        raise ValueError("Registered native image identity missing")
    if sha256(support_path) != registration.get("full_support_mask", {}).get("sha256"):
        raise ValueError("Native support TIFF differs from registration")
    converted_support_sha = _verify_registered_support(
        registration, support_path, result_folder / "support.npy")
    materialized = _receipt(materialization_folder / "MATERIALIZATION_COMPLETE.json",
                            "cellphenotyper-bora-semantic-token-materialization-v1",
                            review_only=False)
    if (materialized.get("registration_sha256") != sha256(registration_path) or
            materialized.get("native_image_sha256") != native["sha256"]):
        raise ValueError("Transfer materialization registration changed")
    token_path = result_folder / "tokens" / "TRANSFER_TOKEN_COMPLETE.json"
    assembly_path = result_folder / "assembly" / "ASSEMBLY_COMPLETE.json"
    vector_path = result_folder / "vectors" / "GLOBAL_VECTOR_COMPLETE.json"
    token = _receipt(token_path, TOKEN_FORMAT)
    assembly = _receipt(assembly_path, ASSEMBLY_FORMAT)
    vector = _receipt(vector_path, VECTOR_FORMAT)
    if (token.get("identity_status") !=
            "exploratory_manuscript_transfer_token_review_only" or
            token.get("native_geometry_available") is not False or
            token.get("registration_sha256") != sha256(registration_path) or
            token.get("materialization_receipt_sha256") !=
                sha256(materialization_folder / "MATERIALIZATION_COMPLETE.json") or
            token.get("independent_pm016_validation") is not False or
            token.get("model_sha256") !=
                "21a17b968c85db0d592cf1c014f9ef00d1019105732e54e3784afa00b067d847" or
            token.get("policy_sha256") !=
                "6cf74e443e84f3832fca53863a50c84d728065337f6ff876cbb435209f305fbf"):
        raise ValueError("Exact frozen V5 exploratory transfer receipt required")
    if (assembly.get("identity_status") != IDENTITY or
            assembly.get("core_partition_complete") is not True or
            assembly.get("exact_categorical_halo_parity") is not True or
            assembly.get("independently_fitted_windows") is not False or
            assembly.get("geojson_published") is not False or
            assembly.get("registration_sha256") != sha256(registration_path) or
            assembly.get("tissue_support_sha256") != converted_support_sha or
            assembly.get("registered_support_source_sha256") != sha256(support_path) or
            assembly.get("support_sampling") != SUPPORT_SAMPLING or
            assembly.get("support_conversion") != SUPPORT_CONVERSION or
            assembly.get("support_binding_format") !=
                "bora-verified-registered-native-support-v2" or
            assembly.get("support_verified_native_pixels") !=
                native["height"] * native["width"] or
            assembly.get("registered_support_shape_yx") !=
                [registration["full_support_mask"]["height"],
                 registration["full_support_mask"]["width"]] or
            assembly.get("native_shape_yx") != [native["height"], native["width"]] or
            assembly.get("native_mpp_xy") != [native["mpp_x"], native["mpp_y"]] or
            assembly.get("source_frame_id") != native["frame_id"] or
            assembly.get("source_image_sha256") != native["sha256"] or
            assembly.get("class_ids") != token.get("class_ids") or
            assembly.get("class_names") != token.get("class_names")):
        raise ValueError("Native assembly is incomplete or mismatched")
    if (vector.get("identity_status") != IDENTITY or
            vector.get("global_contours") is not True or
            vector.get("independently_fitted_windows") is not False or
            vector.get("no_macroregion_cuts") is not True or
            vector.get("no_stair_step_fallback") is not True or
            vector.get("source_assembly_receipt_sha256") != sha256(assembly_path) or
            vector.get("source_label_sha256") != assembly.get("label_sha256") or
            vector.get("source_unknown_sha256") != assembly.get("unknown_sha256") or
            vector.get("registered_support_source_sha256") != sha256(support_path) or
            vector.get("support_sampling") != SUPPORT_SAMPLING or
            vector.get("support_conversion") != SUPPORT_CONVERSION or
            vector.get("registered_support_shape_yx") !=
                assembly.get("registered_support_shape_yx") or
            vector.get("native_shape_yx") != assembly.get("native_shape_yx") or
            vector.get("native_mpp_xy") != assembly.get("native_mpp_xy") or
            vector.get("minimum_area_um2") != 100.0 or
            not isinstance(vector.get("selected_smoothing_tolerance_um"), (int, float)) or
            vector["selected_smoothing_tolerance_um"] <= 0 or
            vector.get("review_coordinates", 10**20) >=
                vector.get("exact_coordinates", -1)):
        raise ValueError("One smooth unknown-preserving global vector is required")
    for relative, digest in vector.get("output_sha256", {}).items():
        if relative not in {"exact/mask.uint8.dat", "exact/mask.geojson",
                            "review/mask.geojson"} or \
                sha256(result_folder / "vectors" / relative) != digest:
            raise ValueError("Global vector output inventory or bytes changed")
    if set(vector.get("output_sha256", {})) != {
            "exact/mask.uint8.dat", "exact/mask.geojson", "review/mask.geojson"}:
        raise ValueError("Complete exact/review vector inventory required")
    for role in ("exact", "review"):
        header = _geojson_header(result_folder / "vectors" / role / "mask.geojson")
        details = header.get("vector_export", {})
        if (header.get("format") != VECTOR_FORMAT or
                header.get("type") != "FeatureCollection" or
                header.get("coordinate_space") != "native_image_pixel_edges" or
                header.get("origin") != "top_left" or
                header.get("source_frame_id") != native["frame_id"] or
                header.get("source_image_sha256") != native["sha256"] or
                header.get("native_shape_yx") != assembly["native_shape_yx"] or
                header.get("native_mpp_xy") != assembly["native_mpp_xy"] or
                header.get("source_label_sha256") != assembly["label_sha256"] or
                header.get("source_unknown_sha256") != assembly["unknown_sha256"] or
                header.get("registered_support_source_sha256") != sha256(support_path) or
                header.get("support_sampling") != SUPPORT_SAMPLING or
                header.get("registered_support_shape_yx") !=
                    assembly.get("registered_support_shape_yx") or
                header.get("global_model_binding_sha256") !=
                    assembly["global_binding_sha256"] or
                header.get("identity_status") != IDENTITY or
                header.get("biological_accuracy_qualified") is not False or
                details.get("global_contours") is not True or
                details.get("independently_fitted_windows") is not False or
                details.get("unknown_holes_exempt_from_minimum") is not True or
                details.get("minimum_part_and_unprotected_hole_area_um2") != 100.0 or
                details.get("geometry_role") !=
                    ("native_exact_unknown_preserved" if role == "exact" else
                     "smoothed_review_unknown_preserved") or
                details.get("smoothing_tolerance_um") !=
                    (0.0 if role == "exact" else vector["selected_smoothing_tolerance_um"])):
            raise ValueError("GeoJSON is stale, hard-group, stair-step or wrong-frame")
    for name, digest in (("label.uint8.dat", assembly.get("label_sha256")),
                         ("unknown.uint8.dat", assembly.get("unknown_sha256"))):
        if sha256(result_folder / "assembly" / name) != digest:
            raise ValueError("Native label/unknown bytes changed")
    return native, dict(token=token, assembly=assembly, vector=vector,
                        support_npy_sha256=converted_support_sha,
                        vector_receipt_sha256=sha256(vector_path))


def export(base_project: Path, registration_path: Path,
           materialization_folder: Path, support_path: Path,
           result_folder: Path, destination: Path) -> dict:
    paths = (base_project, registration_path, materialization_folder,
             support_path, result_folder, destination)
    if not all(path.is_absolute() for path in paths):
        raise ValueError("All manuscript-transfer paths must be absolute")
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError("Choose a new project path with an existing parent")
    native, chain = verify_result(registration_path, materialization_folder,
                                  support_path, result_folder)
    previous = validate_experiment(base_project)
    matches = [tissue for tissue in previous["tissues"] if
        tissue["image"].get("sha256") == native["sha256"] and
        tissue["image"].get("frame_id") == native["frame_id"] and
        [tissue["image"].get("height"), tissue["image"].get("width")] ==
            chain["assembly"]["native_shape_yx"] and
        [tissue["image"].get("mpp_x"), tissue["image"].get("mpp_y")] ==
            chain["assembly"]["native_mpp_xy"]]
    if len(matches) != 1:
        raise ValueError("Find exactly one project tissue in the registered native frame")
    tissue = matches[0]
    if tissue.get("registered_tissue") is not None and \
            tissue["registered_tissue"].get("registration", {}).get("sha256") != \
            sha256(registration_path):
        raise ValueError("Project tissue registration differs from transfer")
    vector_path = result_folder / "vectors" / "review" / "mask.geojson"
    artifact_id = "bora-manuscript-transfer-" + chain["vector_receipt_sha256"][:20]
    if any(item.get("artifact_id") == artifact_id for item in tissue["artifacts"]):
        raise ValueError("This manuscript-transfer review is already present")
    tissue["artifacts"].append({
        "artifact_id": artifact_id, "role": ROLE,
        "label": "Bora tumor/stroma manuscript transfer (unvalidated review)",
        "path": str(vector_path),
        "sha256": chain["vector"]["output_sha256"]["review/mask.geojson"],
        "format": "geojson", "review_only": True,
        "source_frame_id": native["frame_id"], "image_sha256": native["sha256"],
        "provenance": {
            "producer": VECTOR_FORMAT,
            "prediction_unit": "Virchow2 spatial token; native H&E boundary refinement",
            "identity_status": IDENTITY,
            "evidence_class": "frozen_v5_manuscript_source_transfer_unvalidated_on_pm016",
            "biological_accuracy_qualified": False,
            "independent_pm016_validation": False,
            "unknown_semantics": "abstention_not_background_or_tissue_class",
            "model_sha256": chain["token"]["model_sha256"],
            "policy_sha256": chain["token"]["policy_sha256"],
            "materialization_receipt_sha256":
                chain["token"]["materialization_receipt_sha256"],
            "registration_sha256": sha256(registration_path),
            "registered_support_source_sha256": sha256(support_path),
            "tissue_support_sha256": chain["support_npy_sha256"],
            "support_sampling": SUPPORT_SAMPLING,
            "support_conversion": SUPPORT_CONVERSION,
            "support_verified_native_pixels": native["height"] * native["width"],
            "registered_support_shape_yx":
                chain["assembly"]["registered_support_shape_yx"],
            "native_label_sha256": chain["assembly"]["label_sha256"],
            "native_unknown_sha256": chain["assembly"]["unknown_sha256"],
            "global_vector_receipt_sha256": chain["vector_receipt_sha256"],
            "global_contours": True, "no_stair_step_fallback": True,
            "minimum_area_um2": 100.0,
            "smoothing_tolerance_um":
                chain["vector"]["selected_smoothing_tolerance_um"],
        },
    })
    result = export_experiment(previous, destination, base_dir=base_project.parent)
    matching = [row for row in result["tissues"] if row["tissue_id"] == tissue["tissue_id"]]
    if len(matching) != 1 or not any(item["artifact_id"] == artifact_id
                                     for item in matching[0]["artifacts"]):
        raise ValueError("Exported project lost the locked transfer review")
    return {"format": "cellphenotyper-bora-manuscript-transfer-export-v1",
            "complete": True, "review_only": True,
            "biological_accuracy_qualified": False,
            "project_path": str(destination), "project_sha256": sha256(destination),
            "artifact_id": artifact_id,
            "vector_receipt_sha256": chain["vector_receipt_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-project", "registration", "materialization-folder",
                 "registered-support-tiff", "bora-result", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.base_project, args.registration,
        args.materialization_folder, args.registered_support_tiff,
        args.bora_result, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
