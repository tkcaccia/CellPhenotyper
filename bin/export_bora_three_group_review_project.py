"""Create a versioned PM016 three-group KODAMA/Bora CellPhenotyper project.

Bora polygons open as ordinary editable tissue annotations in wsiTools. Their
model provenance and diagnostic seam status remain in the project metadata.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


KODAMA_FILES = {
    "kodama_three_group_embedding_plot": "embedding_plot",
    "kodama_three_group_spatial_map": "spatial_map",
    "kodama_three_group_clusters": "clusters",
    "kodama_three_group_tiles": "tiles",
    "kodama_three_group_binding_receipt": "kodama_binding",
}
OLD_KODAMA_ROLES = {
    "kodama_two_group_map", "kodama_map_binding_receipt",
    "kodama_two_group_clusters", "kodama_two_group_receipt",
}
PRODUCER = "bora-kodama-three-group-he-diagnostic-review-v1"
PREDICTION_UNIT = "KODAMA_three_group_token_field_native_HE_refinement"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path, limit: int = 32 * 1024**2) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
        raise ValueError(f"Expected bounded ordinary JSON file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def bound(path: Path, output: Path) -> dict:
    return {"path": os.path.relpath(path, output.parent), "sha256": digest(path)}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def export(*, base_project: Path, geojson: Path, run_started_receipt: Path,
           mask_receipt: Path,
           diagnostic_receipt: Path, vector_receipt: Path,
           kodama_binding: Path, embedding_plot: Path, spatial_map: Path,
           clusters: Path, tiles: Path, output: Path,
           base_review_geojson: Path | None = None,
           smoothing_receipt: Path | None = None,
           smoothing_binding: Path | None = None) -> dict:
    smoothed = all(p is not None for p in
                   (base_review_geojson, smoothing_receipt, smoothing_binding))
    require(smoothed or all(p is None for p in
                            (base_review_geojson, smoothing_receipt, smoothing_binding)),
            "Supply all three topology-guarded smoothing inputs or none")
    named = {key: value for key, value in locals().items()
             if isinstance(value, Path) and key != "output"}
    require(all(p.is_absolute() for p in [*named.values(), output]),
            "Use absolute input and output paths")
    named = {key: p.resolve(strict=True) for key, p in named.items()}
    base_project = named["base_project"]
    output = output.resolve(strict=False)
    require(output.parent == base_project.parent and output.parent.is_dir() and
            not output.exists() and not output.is_symlink(),
            "Use a new versioned project filename beside the base project")
    base = read_json(base_project)
    require(base.get("schema") == "cellphenotyper-experiment" and
            base.get("schema_version") == 1 and
            isinstance(base.get("tissues"), list),
            "Base project is not a CellPhenotyper experiment")
    geo = read_json(named["geojson"])
    started = read_json(named["run_started_receipt"], 8 * 1024**2)
    mask = read_json(named["mask_receipt"], 8 * 1024**2)
    diagnostic = read_json(named["diagnostic_receipt"], 8 * 1024**2)
    vector = read_json(named["vector_receipt"], 8 * 1024**2)
    kodama = read_json(named["kodama_binding"], 8 * 1024**2)
    require(kodama.get("schema") == "cellphenotyper-pm016-kodama-three-group-binding" and
            kodama.get("status") == "exploratory_unlabeled_three_group_review" and
            kodama.get("observation_count") == 1103 and
            kodama.get("label_semantics") == "anonymous_KODAMA_domains_not_tissue_classes" and
            list(kodama.get("cluster_sizes", {}).keys()) == ["1", "2", "3"],
            "KODAMA binding is not the reviewed anonymous three-group run")
    paths = {"kodama_three_group_embedding_plot": named["embedding_plot"],
             "kodama_three_group_spatial_map": named["spatial_map"],
             "kodama_three_group_clusters": named["clusters"],
             "kodama_three_group_tiles": named["tiles"]}
    require(all(kodama["files"].get(role, {}).get("sha256") == digest(path)
                for role, path in paths.items()),
            "KODAMA binding file hashes differ from staged artifacts")
    with named["tiles"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 1103 and
            {int(row["cluster"]) for row in rows} == {1, 2, 3} and
            len({row["label"] for row in rows}) == 1103,
            "KODAMA tiles do not contain 1,103 unique three-group observations")
    require(geo.get("type") == "FeatureCollection" and
            geo.get("coordinate_space") == "native_image_pixel_edges" and
            geo.get("predicted_not_expert_annotation") is True and
            geo.get("vector_export", {}).get("format") == "bora-wsi-global-vector-v1" and
            geo.get("vector_export", {}).get("global_contours") is True and
            geo.get("vector_export", {}).get("independently_fitted_windows") is False and
            geo.get("vector_export", {}).get("minimum_part_and_hole_area_um2") == 100,
            "Bora review geometry lacks the native global-vector contract")
    features = geo.get("features")
    require(isinstance(features, list) and 0 < len(features) <= 5000,
            "Bora review needs bounded features")
    observed = set()
    for feature in features:
        props = feature.get("properties", {})
        group = props.get("value")
        require(group in (1, 2, 3) and
                props.get("name") == props.get("predicted_tissue") == f"KODAMA Group {group}" and
                props.get("predicted_not_expert_annotation") is True and
                props.get("area_um2", 0) >= 100 and
                feature.get("geometry", {}).get("type") in ("Polygon", "MultiPolygon"),
                "Bora review has a small or incorrectly labelled component")
        observed.add(group)
    require(observed == {1, 2, 3}, "Bora result does not contain all three groups")
    matches = [t for t in base["tissues"] if
               t.get("image", {}).get("sha256") == geo.get("source_image_sha256") and
               t.get("image", {}).get("frame_id") == geo.get("source_frame_id") and
               [t.get("image", {}).get("height"), t.get("image", {}).get("width")] ==
                   geo.get("native_shape_yx") and
               [t.get("image", {}).get("mpp_x"), t.get("image", {}).get("mpp_y")] ==
                   geo.get("native_mpp_xy")]
    require(len(matches) == 1, "Bora geometry does not match one source image/frame")
    image = matches[0]["image"]
    require(started.get("format") ==
                "pm016-kodama-three-group-recovered-bora-diagnostic-v2" and
            started.get("status") == "RUNNING_DIAGNOSTIC" and
            started.get("input_files", {}).get("native_he", {}).get("sha256") ==
                image["sha256"] and
            started.get("input_files", {}).get("three_csv", {}).get("sha256") ==
                digest(named["clusters"]) and
            started.get("three_group_join", {}).get("new_csv_sha256") ==
                digest(named["clusters"]) and
            started.get("expert_annotation_used") is False and
            mask.get("format") == started["format"] and
            mask.get("status") ==
                "complete_mask_diagnostic_not_expert_annotation" and
            mask.get("three_group_join") == started.get("three_group_join") and
            mask.get("refinement", {}).get("mask_sha256") ==
                geo.get("source_semantic_mask_sha256") and
            mask.get("refinement", {}).get("global_all_three_classes_preserved") is True and
            mask.get("token", {}).get("seam_gate_pass") ==
                mask.get("technical_seam_gate_pass") and
            mask.get("biological_accuracy_qualified") is False and
            diagnostic.get("format") == started["format"] and
            diagnostic.get("status") ==
                "complete_diagnostic_not_biological_annotation" and
            diagnostic.get("three_group_join") == mask.get("three_group_join") and
            diagnostic.get("model") == mask.get("model") and
            diagnostic.get("token") == mask.get("token") and
            diagnostic.get("refinement") == mask.get("refinement") and
            diagnostic.get("vectorization", {}).get("receipt_sha256") ==
                digest(named["vector_receipt"]) and
            diagnostic.get("biological_accuracy_qualified") is False and
            vector.get("source_image_sha256") == image["sha256"] and
            vector.get("source_frame_id") == image["frame_id"] and
            vector.get("source_semantic_mask_sha256") ==
                geo.get("source_semantic_mask_sha256") and
            vector.get("global_model_binding_sha256") ==
                geo.get("global_model_binding_sha256") and
            vector.get("output_sha256", {}).get("review/mask.geojson") ==
                digest(named["base_review_geojson"] if smoothed else named["geojson"]) and
            vector.get("global_contours") is True and
            vector.get("no_macroregion_cuts") is True,
            "Bora mask, vector, diagnostic and image are not a bound chain")
    if smoothed:
        smoothing = read_json(named["smoothing_receipt"], 8 * 1024**2)
        binding = read_json(named["smoothing_binding"], 8 * 1024**2)
        require(
            binding.get("format") == "pm016-three-class-bora-review-smoothing-v3" and
            binding.get("status") == "diagnostic_review_only_seam_gate_failed" and
            binding.get("review_geojson_sha256") == digest(named["geojson"]) and
            binding.get("smoothing_receipt_sha256") == digest(named["smoothing_receipt"]) and
            binding.get("v2_diagnostic_receipt_sha256") == digest(named["diagnostic_receipt"]) and
            binding.get("v2_mask_receipt_sha256") == digest(named["mask_receipt"]) and
            binding.get("v2_vector_receipt_sha256") == digest(named["vector_receipt"]) and
            binding.get("v2_exact_geojson_sha256") ==
                vector.get("output_sha256", {}).get("exact/mask.geojson") and
            binding.get("component_count") == len(features) and
            binding.get("classes") == [1, 2, 3] and
            binding.get("technical_seam_gate_pass") is False and
            binding.get("biological_accuracy_qualified") is False and
            smoothing.get("format") == "bora-wsi-review-smoothing-v1" and
            smoothing.get("status") == "REVIEW_ONLY_NOT_ACCEPTED" and
            smoothing.get("review_geojson_sha256") == digest(named["geojson"]) and
            smoothing.get("selected", {}).get("status") == "PASSED_DIAGNOSTIC_REVIEW" and
            smoothing.get("selected", {}).get("tolerance_um") ==
                binding.get("selected_tolerance_um") and
            smoothing.get("selected", {}).get("full_native_raster_agreement", {}).get(
                "new_background_in_exact_tissue") == 0 and
            smoothing.get("source_sha256", {}).get(str(named["vector_receipt"])) ==
                digest(named["vector_receipt"]) and
            smoothing.get("technical_r4_seam_failure_remains") is True and
            smoothing.get("biological_accuracy_qualified") is False and
            smoothing.get("exact_cleaned_mask_and_geojson_unchanged") is True,
            "Topology-guarded Bora review smoothing is not bound to the v2 exact result")
    result = copy.deepcopy(base)
    tissue = next(t for t in result["tissues"] if t["tissue_id"] == matches[0]["tissue_id"])
    artifacts = tissue.get("artifacts", [])
    require(not any(a.get("role") == "bora_exploratory_boundary_review"
                    for a in artifacts),
            "Base project already contains a Bora review; start from original project")
    tissue["artifacts"] = [a for a in artifacts if a.get("role") not in OLD_KODAMA_ROLES]
    for role, key in KODAMA_FILES.items():
        path = named[key]
        tissue["artifacts"].append({"role": role, **bound(path, output)})
    provenance = {
        "producer": ("bora-kodama-three-group-topology-guarded-review-v3"
                     if smoothed else PRODUCER),
        "prediction_unit": PREDICTION_UNIT,
        "group_count": 3, "accepted_annotation": False,
        "biological_accuracy_qualified": False, "minimum_area_um2": 100,
        "kodama_cluster_sha256": digest(named["clusters"]),
        "kodama_binding_sha256": digest(named["kodama_binding"]),
        "technical_seam_gate_pass": diagnostic.get("technical_seam_gate_pass") is True,
    }
    # Input argument names end with `_receipt`; keep manifest keys stable.
    provenance.update({
        "run_started_receipt": bound(named["run_started_receipt"], output),
        "mask_receipt": bound(named["mask_receipt"], output),
        "diagnostic_receipt": bound(named["diagnostic_receipt"], output),
        "vector_receipt": bound(named["vector_receipt"], output),
        "kodama_binding_receipt": bound(named["kodama_binding"], output),
    })
    if smoothed:
        provenance.update({
            "base_review_geojson": bound(named["base_review_geojson"], output),
            "smoothing_receipt": bound(named["smoothing_receipt"], output),
            "smoothing_binding_receipt": bound(named["smoothing_binding"], output),
            "review_smoothing_tolerance_um": binding["selected_tolerance_um"],
            "topology_guarded_review": True,
        })
    tissue["artifacts"].append({
        "artifact_id": "bora-kodama-three-review-" + digest(named["geojson"])[:20],
        "role": "bora_exploratory_boundary_review",
        "label": ("Bora KODAMA 3 groups · latest topology-guarded prediction · click to edit"
                  if smoothed else "Bora KODAMA 3 anonymous groups · diagnostic locked reference"),
        **bound(named["geojson"], output), "format": "geojson",
        "review_only": True, "training_eligible": False,
        "predicted_not_expert_annotation": True,
        "source_frame_id": image["frame_id"], "image_sha256": image["sha256"],
        "provenance": provenance,
    })
    result["name"] = f"{tissue['name']} · KODAMA 3 groups → Bora · editable review"
    result.setdefault("settings", {})["bora_annotations_mode"] = "ordinary_editable"
    result.setdefault("settings", {})["bora_status"] = (
        "Latest topology-guarded Bora review; technical seam check still fails"
        if smoothed else "Three anonymous KODAMA groups refined by Bora; diagnostic review only")
    result.setdefault("provenance", {})["interpretation"] = (
        "Bora boundaries are anonymous predictions. Click a boundary to edit "
        "its separate review draft; export corrected ROIs separately. "
        "No tissue identity or biological accuracy is established.")
    data = (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".kodama-three-bora-review-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"project": str(output), "project_sha256": digest(output),
            "bora_geojson_sha256": digest(named["geojson"]),
            "kodama_binding_sha256": digest(named["kodama_binding"]),
            "boundary_count": len(features), "group_count": 3,
            "diagnostic_only": True, "training_eligible": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base_project", "geojson", "run_started_receipt", "mask_receipt", "diagnostic_receipt",
                "vector_receipt", "kodama_binding", "embedding_plot", "spatial_map",
                "clusters", "tiles", "output"):
        parser.add_argument("--" + key.replace("_", "-"), required=True, type=Path)
    for key in ("base_review_geojson", "smoothing_receipt", "smoothing_binding"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path)
    args = vars(parser.parse_args())
    print(json.dumps(export(**args), sort_keys=True))


if __name__ == "__main__":
    main()
