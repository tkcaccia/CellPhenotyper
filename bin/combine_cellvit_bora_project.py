#!/usr/bin/env python3
"""Attach a checked Bora full-slide annotation to a complete indexed CellViT project.

This CPU-only export consumes previously accepted files; it does not rerun Bora
or CellViT and never promotes an incomplete CellViT population to a final WSI.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

from bora_contextual_fullslide import _rebase_project, _validate_project, sha256
from export_experiment import atomic_json


HEADS = frozenset(("pannuke", "consep", "lizard", "nucls_main",
                   "nucls_super", "ocelot", "midog", "panoptils"))


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _complete_cell_project(project_path: Path, receipt_path: Path,
                           expected_shards: int | None) -> tuple[dict, dict, dict]:
    project = _validate_project(project_path)
    receipt = _read(receipt_path)
    if (receipt.get("schema") != "cellphenotyper-cellvit-fullslide-complete-v2" or
            receipt.get("every_core_verified") is not True or
            receipt.get("feature_vectors_retained") is not True or
            set(receipt.get("requested_heads", [])) != HEADS or
            len(receipt["requested_heads"]) != len(HEADS) or
            receipt.get("project_sha256") != sha256(project_path)):
        raise ValueError("CellViT project lacks its complete, eight-head checked receipt")
    counts = [receipt.get(key) for key in
              ("planned_shards", "complete_shards", "shard_count")]
    if (any(type(value) is not int or value <= 0 for value in counts) or
            len(set(counts)) != 1 or
            (expected_shards is not None and counts[0] != expected_shards)):
        raise ValueError("CellViT shard coverage is incomplete or differs from the expected slide")
    if len(project["tissues"]) != 1:
        raise ValueError("Combined export requires one exact full-slide tissue")
    tissue = project["tissues"][0]
    cells = tissue.get("cell_classification")
    if not isinstance(cells, dict) or not isinstance(cells.get("index"), dict):
        raise ValueError("CellViT project lacks its full-slide indexed cell source")
    image = tissue["image"]
    if (receipt.get("source_image_sha256") != image["sha256"] or
            receipt.get("partition_manifest_sha256") != cells["sha256"] or
            receipt.get("index_sha256") != cells["index"]["sha256"]):
        raise ValueError("CellViT receipt does not bind the project image, cells and index")
    source_path = (project_path.parent / cells["path"]).resolve(strict=True)
    source = _read(source_path)
    if (source.get("schema") != "cellphenotyper-cellvit-shards-v1" or
            source.get("partition_count") != counts[0] or
            len(source.get("partitions", [])) != counts[0] or
            source.get("plan_sha256") != receipt.get("plan_sha256") or
            source.get("cell_count") != receipt.get("cell_count") or
            source.get("image", {}).get("sha256") != image["sha256"]):
        raise ValueError("CellViT partition manifest is not complete for this slide")
    index_path = (project_path.parent / cells["index"]["path"]).resolve(strict=True)
    with closing(sqlite3.connect(index_path.as_uri() + "?mode=ro", uri=True)) as connection:
        metadata = {key: json.loads(value) for key, value in connection.execute(
            "SELECT key,value FROM metadata WHERE key IN "
            "('schema_version','cell_count','feature_set','models')")}
    feature = metadata.get("feature_set", {})
    if (metadata.get("schema_version") != 3 or
            metadata.get("cell_count") != receipt.get("cell_count") or
            feature.get("kind") != "cellvit-cell" or
            feature.get("descriptor", {}).get("feature_count") != 1280 or
            len(feature.get("feature_columns", [])) != 1280):
        raise ValueError("CellViT index lacks the complete 1,280-D cell population")
    models = metadata.get("models", [])
    if (not isinstance(models, list) or len(models) != len(HEADS) or
            {model.get("taxonomy", "").lower() for model in models} != HEADS or
            any(model.get("status") != "available" for model in models)):
        raise ValueError("CellViT index does not contain all eight available classifiers")
    return project, tissue, receipt


def combine(project_path: Path, cell_receipt_path: Path, bora_geojson_path: Path,
            bora_receipt_path: Path, output: Path, *, expected_bora_sha256: str,
            expected_shards: int | None = None) -> dict:
    paths = [Path(p).resolve(strict=True) for p in
             (project_path, cell_receipt_path, bora_geojson_path, bora_receipt_path)]
    project_path, cell_receipt_path, bora_geojson_path, bora_receipt_path = paths
    requested_output = Path(output)
    if (requested_output.name != "experiment.cellphenotyper.json" or
            not requested_output.parent.parent.is_dir()):
        raise ValueError("Use experiment.cellphenotyper.json in a new output folder")
    output_root = requested_output.parent.parent.resolve(strict=True) / requested_output.parent.name
    output = output_root / requested_output.name
    if output_root.exists():
        raise FileExistsError("Combined output folder already exists; use a new versioned folder")
    project, tissue, cell_receipt = _complete_cell_project(
        project_path, cell_receipt_path, expected_shards)
    geo_sha = sha256(bora_geojson_path)
    receipt = _read(bora_receipt_path)
    geo = _read(bora_geojson_path)
    image = tissue["image"]
    if (geo_sha != expected_bora_sha256 or
            receipt.get("format") != "bora-wsi-review-smoothing-v1" or
            receipt.get("status") != "PREDICTED_TISSUE_ANNOTATION" or
            receipt.get("review_geojson_sha256") != geo_sha or
            receipt.get("simplify_outer_boundary") is not True or
            receipt.get("biological_accuracy_qualified") is not False):
        raise ValueError("Bora annotation does not match the accepted guarded result")
    if (geo.get("type") != "FeatureCollection" or
            geo.get("source_image_sha256") != image["sha256"] or
            geo.get("source_frame_id") != image["frame_id"] or
            geo.get("native_shape_yx") != [image["height"], image["width"]] or
            geo.get("native_mpp_xy") != [image["mpp_x"], image["mpp_y"]] or
            geo.get("predicted_not_expert_annotation") is not True or
            geo.get("vector_export", {}).get("global_contours") is not True or
            geo.get("vector_export", {}).get("minimum_part_and_hole_area_um2", 0) < 1000 or
            not isinstance(geo.get("features"), list) or not geo["features"]):
        raise ValueError("Bora geometry does not bind this complete native H&E frame")
    if any(feature.get("geometry", {}).get("type") not in ("Polygon", "MultiPolygon") or
           feature.get("properties", {}).get("predicted_not_expert_annotation") is not True or
           feature.get("properties", {}).get("area_um2", 0) < 1000
           for feature in geo["features"]):
        raise ValueError("Bora annotation contains unsupported or subthreshold geometry")
    if any(artifact.get("role") == "tissue_annotations" for artifact in tissue["artifacts"]):
        raise ValueError("CellViT base project already contains tissue annotations")
    # Build and verify a sibling directory first. Relative links to retained
    # cell assets have the same spelling before and after the atomic rename.
    stage_root = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    try:
        staged_output = stage_root / output.name
        combined = _rebase_project(project_path, staged_output)
        targets = [(bora_geojson_path, stage_root / "BORA_TISSUE_ANNOTATIONS.geojson"),
                   (bora_receipt_path, stage_root / "BORA_ACCEPTED_COMPLETE.json"),
                   (cell_receipt_path, stage_root / "CELLVIT_PROJECT_COMPLETE.json")]
        for source, target in targets:
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if sha256(source) != sha256(target):
                raise ValueError("Copied project attachment differs from its checked source")
        combined["provenance"]["cellvit_fullslide_complete"] = {
            "path": "CELLVIT_PROJECT_COMPLETE.json",
            "sha256": sha256(stage_root / "CELLVIT_PROJECT_COMPLETE.json")}
        combined_tissue = combined["tissues"][0]
        combined_tissue["artifacts"].append({
            "artifact_id": "bora-accepted-" + geo_sha[:20],
            "role": "tissue_annotations", "label": "Bora · KODAMA 3 groups",
            "path": "BORA_TISSUE_ANNOTATIONS.geojson", "sha256": geo_sha,
            "format": "geojson", "source_frame_id": image["frame_id"],
            "image_sha256": image["sha256"],
            "predicted_not_expert_annotation": True, "training_eligible": False,
            "review_only": False,
            "provenance": {"algorithm": "bora-wsi-review-smoothing-v1",
                "receipt": {"path": "BORA_ACCEPTED_COMPLETE.json",
                            "sha256": sha256(stage_root / "BORA_ACCEPTED_COMPLETE.json")},
                "class_identity": "anonymous_KODAMA_groups"}})
        combined["settings"]["bora_annotations_mode"] = "ordinary_editable"
        combined["settings"]["bora_status"] = (
            "Predicted KODAMA groups; editable tissue annotation; biological identity unvalidated")
        atomic_json(staged_output, combined, overwrite=False)
        _validate_project(staged_output)
        if output_root.exists():
            raise FileExistsError("Combined output folder appeared before publication")
        os.rename(stage_root, output_root)
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)
    return {"project": str(output), "cell_count": cell_receipt["cell_count"],
            "bora_geojson_sha256": geo_sha, "bora_components": len(geo["features"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-project", type=Path, required=True)
    parser.add_argument("--cell-receipt", type=Path, required=True)
    parser.add_argument("--bora-geojson", type=Path, required=True)
    parser.add_argument("--bora-receipt", type=Path, required=True)
    parser.add_argument("--expected-bora-sha256", required=True)
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = combine(args.cell_project, args.cell_receipt, args.bora_geojson,
                     args.bora_receipt, args.output,
                     expected_bora_sha256=args.expected_bora_sha256,
                     expected_shards=args.expected_shards)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
