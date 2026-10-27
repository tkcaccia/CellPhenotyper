"""Export a locked, seam-failed KODAMA/Bora diagnostic review project.

This does not convert KODAMA groups into tumor/stroma labels or expert ROIs.
The r7 corner result is accepted only if its own completion receipt exists;
otherwise use the completed r6 smoothed review explicitly.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile


PRODUCER = {
    "r6": "bora-recovered-kodama-he-smoothed-diagnostic-review-v1",
    "r7": "bora-recovered-kodama-he-diagnostic-review-v1",
}
ROLE = {"r6": "smoothed_review_only", "r7": "corner_rounded_review_only"}
PREDICTION_UNIT = "KODAMA_two_group_token_field_native_HE_refinement"
MASK_ALGORITHM = ("Frozen r2 complete token field -> unchanged "
                  "COLON_BOUNDARY_SPEC native H&E refinement -> optional global vectorizer")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, *, maximum: int = 8 * 1024**2) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError(f"Expected bounded ordinary JSON file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _bound(path: Path, destination: Path) -> dict:
    return {"path": os.path.relpath(path, destination.parent), "sha256": _sha(path)}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def export(base_project: Path, geojson: Path, mask_receipt: Path,
           diagnostic_receipt: Path, vector_receipt: Path,
           smoothing_receipt: Path, output: Path, *, stage: str = "r6",
           corner_receipt: Path | None = None) -> dict:
    if stage not in PRODUCER:
        raise ValueError("Choose r6 or r7 explicitly")
    inputs = [base_project, geojson, mask_receipt, diagnostic_receipt,
              vector_receipt, smoothing_receipt]
    if stage == "r7":
        _require(corner_receipt is not None, "r7 requires corner completion receipt")
        inputs.append(corner_receipt)
    else:
        _require(corner_receipt is None, "r6 must not cite a failed r7 receipt")
    if not all(p.is_absolute() for p in [*inputs, output]):
        raise ValueError("Use absolute input and output paths")
    inputs = [p.resolve(strict=True) for p in inputs]
    base_project, geojson, mask_receipt, diagnostic_receipt, vector_receipt, smoothing_receipt = inputs[:6]
    if stage == "r7":
        corner_receipt = inputs[6]
    output = output.resolve(strict=False)
    if (output.exists() or output.is_symlink() or
            output.parent != base_project.parent or not output.parent.is_dir()):
        raise ValueError("Use a new versioned project filename beside the base project")
    base = _read(base_project)
    geo = _read(geojson, maximum=32 * 1024**2)
    mask, diagnostic, vector, smoothing = map(_read,
        (mask_receipt, diagnostic_receipt, vector_receipt, smoothing_receipt))
    corner = _read(corner_receipt) if stage == "r7" else None
    _require(base.get("schema") == "cellphenotyper-experiment" and
             base.get("schema_version") == 1 and isinstance(base.get("tissues"), list),
             "Base project is not a CellPhenotyper experiment")
    _require(geo.get("type") == "FeatureCollection" and
             geo.get("coordinate_space") == "native_image_pixel_edges" and
             geo.get("vector_export", {}).get("geometry_role") == ROLE[stage] and
             geo.get("vector_export", {}).get("format") == "bora-wsi-global-vector-v1" and
             geo.get("vector_export", {}).get("global_contours") is True and
             geo.get("vector_export", {}).get("independently_fitted_windows") is False and
             geo.get("vector_export", {}).get("minimum_part_and_hole_area_um2") == 100 and
             geo.get("predicted_not_expert_annotation") is True and
             geo.get("tissue_identity_status") == "exploratory_transfer",
             "Review GeoJSON is not the signed whole-slide diagnostic geometry")
    matches = [(t, t.get("image", {})) for t in base["tissues"] if
               t.get("image", {}).get("sha256") == geo.get("source_image_sha256") and
               t.get("image", {}).get("frame_id") == geo.get("source_frame_id") and
               [t.get("image", {}).get("height"), t.get("image", {}).get("width")] ==
                   geo.get("native_shape_yx") and
               [t.get("image", {}).get("mpp_x"), t.get("image", {}).get("mpp_y")] ==
                   geo.get("native_mpp_xy")]
    _require(len(matches) == 1, "Review geometry must match exactly one image/frame/MPP")
    tissue, image = matches[0]
    _require(not any(a.get("role") == "bora_exploratory_boundary_review"
                     for a in tissue.get("artifacts", [])),
             "Base project already contains a Bora boundary review")
    _require(mask.get("format") == "pm016-recovered-bora-full-slide-diagnostic-v1" and
             mask.get("algorithm") == MASK_ALGORITHM and
             mask.get("accepted_annotation") is False and
             mask.get("technical_seam_gate_pass") is False and
             mask.get("source_files_sha256", {}).get("native_he") == image["sha256"] and
             isinstance(mask.get("r2_token_field_sha256"), str) and
             isinstance(mask.get("r2_model_receipt_sha256"), str),
             "Frozen recovered KODAMA/Bora diagnostic lineage is absent")
    _require(diagnostic.get("format") == mask["format"] and
             diagnostic.get("status") == "DIAGNOSTIC_ONLY_NOT_ACCEPTED" and
             diagnostic.get("accepted_annotation") is False and
             diagnostic.get("technical_seam_gate_pass") is False and
             diagnostic.get("biological_accuracy_qualified") is False and
             diagnostic.get("mask_receipt_sha256") == _sha(mask_receipt) and
             diagnostic.get("vector", {}).get("receipt_sha256") == _sha(vector_receipt),
             "Recovered diagnostic did not bind its source mask and global vector")
    exact_geo_sha = vector.get("output_sha256", {}).get("exact/mask.geojson")
    _require(vector.get("format") == "bora-wsi-global-vector-v1" and
             vector.get("source_image_sha256") == image["sha256"] and
             vector.get("source_frame_id") == image["frame_id"] and
             vector.get("native_shape_yx") == geo["native_shape_yx"] and
             vector.get("native_mpp_xy") == geo["native_mpp_xy"] and
             vector.get("source_semantic_mask_sha256") ==
                 mask.get("native_refinement", {}).get("mask_sha256") ==
                 geo.get("source_semantic_mask_sha256") and
             vector.get("global_model_binding_sha256") ==
                 geo.get("global_model_binding_sha256") and
             vector.get("minimum_area_um2") == 100 and
             vector.get("minimum_area_strictly_less_removed") is True and
             vector.get("exact_vector_matches_cleaned_mask") is True and
             vector.get("global_contours") is True and
             vector.get("no_macroregion_cuts") is True and
             vector.get("independently_fitted_vector_windows") is False and
             isinstance(exact_geo_sha, str) and len(exact_geo_sha) == 64,
             "Global vector or 100 µm² cleanup does not match recovered mask")
    smoothing_hashes = set(smoothing.get("source_sha256", {}).values())
    _require(smoothing.get("format") == "bora-wsi-review-smoothing-v1" and
             smoothing.get("status") == "REVIEW_ONLY_NOT_ACCEPTED" and
             smoothing.get("selected", {}).get("status") == "PASSED_DIAGNOSTIC_REVIEW" and
             smoothing.get("technical_r4_seam_failure_remains") is True and
             smoothing.get("new_tissue_classifier_used") is False and
             smoothing.get("biological_accuracy_qualified") is False and
             {_sha(vector_receipt), exact_geo_sha, image["sha256"]} <= smoothing_hashes,
             "r6 smoothing is not bound to the seam-failed recovered mask")
    if stage == "r7":
        corner_hashes = set(corner.get("source_sha256", {}).values())
        _require(corner.get("format") == "bora-wsi-corner-rounding-v1" and
                 corner.get("status") == "REVIEW_ONLY_NOT_ACCEPTED" and
                 corner.get("technical_r4_seam_failure_remains") is True and
                 corner.get("long_straight_model_seam_cannot_be_removed_by_local_rounding") is True and
                 corner.get("exact_r4_and_r6_unchanged") is True and
                 corner.get("biological_accuracy_qualified") is False and
                 corner.get("review_geojson_sha256") == _sha(geojson) and
                 {smoothing["review_geojson_sha256"], _sha(vector_receipt),
                  exact_geo_sha, image["sha256"]} <= corner_hashes and
                 any(a.get("status") == "PASSED_REVIEW_ONLY" for a in corner.get("attempts", [])),
                 "r7 corner rounding did not complete the signed diagnostic chain")
    else:
        _require(smoothing.get("review_geojson_sha256") == _sha(geojson),
                 "r6 GeoJSON differs from its completion receipt")
    features = geo.get("features")
    _require(isinstance(features, list) and 0 < len(features) <= 5000,
             "Expected bounded recovered review features")
    for feature in features:
        props = feature.get("properties", {})
        group = props.get("value")
        _require(group in (1, 2) and
                 props.get("name") == props.get("predicted_tissue") ==
                    f"KODAMA Group {group}" and
                 props.get("boundary_role") == ROLE[stage] and
                 props.get("predicted_not_expert_annotation") is True and
                 props.get("tissue_identity_status") == "exploratory_transfer" and
                 isinstance(props.get("area_um2"), (int, float)) and
                 props["area_um2"] >= 100 and
                 feature.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon"),
                 "Review contains a small or misidentified diagnostic geometry")
    result = copy.deepcopy(base)
    selected = next(t for t in result["tissues"] if t["tissue_id"] == tissue["tissue_id"])
    receipt_paths = {"mask": mask_receipt, "diagnostic": diagnostic_receipt,
                     "vector": vector_receipt, "smoothing": smoothing_receipt}
    if stage == "r7":
        receipt_paths["corner"] = corner_receipt
    provenance = {
        "producer": PRODUCER[stage], "prediction_unit": PREDICTION_UNIT,
        "identity_status": "exploratory_transfer",
        "biological_accuracy_qualified": False, "accepted_annotation": False,
        "technical_seam_gate_pass": False, "minimum_area_um2": 100,
        "r2_token_field_sha256": mask["r2_token_field_sha256"],
        "r2_model_receipt_sha256": mask["r2_model_receipt_sha256"],
        "source_semantic_mask_sha256": geo["source_semantic_mask_sha256"],
        "global_model_binding_sha256": geo["global_model_binding_sha256"],
        "source_exact_geojson_sha256": exact_geo_sha,
        "r6_review_geojson_sha256": smoothing["review_geojson_sha256"],
        "review_geojson_sha256": _sha(geojson),
        **{f"{name}_receipt": _bound(path, output)
           for name, path in receipt_paths.items()},
    }
    artifact = {
        "artifact_id": f"bora-recovered-{stage}-diagnostic-{_sha(geojson)[:20]}",
        "role": "bora_exploratory_boundary_review",
        "label": f"KODAMA→Bora {stage} diagnostic; seam gate failed; review only",
        "path": os.path.relpath(geojson, output.parent),
        "sha256": _sha(geojson), "format": "geojson", "review_only": True,
        "training_eligible": False, "predicted_not_expert_annotation": True,
        "source_frame_id": image["frame_id"], "image_sha256": image["sha256"],
        "provenance": provenance,
    }
    selected.setdefault("artifacts", []).append(artifact)
    data = (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".recovered-bora-review-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)  # refuses an existing versioned project
    finally:
        Path(temporary).unlink(missing_ok=True)
    checked = _read(output)
    _require(checked["tissues"][base["tissues"].index(tissue)]["artifacts"][-1] == artifact,
             "Saved project differs from validated diagnostic artifact")
    return {"format": "cellphenotyper-recovered-bora-diagnostic-project-v1",
            "project": str(output), "project_sha256": _sha(output),
            "stage": stage, "artifact_sha256": artifact["sha256"],
            "diagnostic_only": True, "technical_seam_gate_pass": False,
            "training_eligible": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-project", "geojson", "mask-receipt", "diagnostic-receipt",
                 "vector-receipt", "smoothing-receipt", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--stage", choices=("r6", "r7"), default="r6")
    parser.add_argument("--corner-receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.base_project, args.geojson, args.mask_receipt,
                            args.diagnostic_receipt, args.vector_receipt,
                            args.smoothing_receipt, args.output,
                            stage=args.stage, corner_receipt=args.corner_receipt),
                     sort_keys=True))


if __name__ == "__main__":
    main()
