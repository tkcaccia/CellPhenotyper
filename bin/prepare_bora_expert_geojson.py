"""Bind explicitly reviewed wsiTools ROIs to Bora's native-pixel expert schema.

This CPU-only tool never infers tissue classes, annotators or train/holdout splits.
It reads a copied CellPhenotyper experiment and a separate wsiTools GeoJSON export;
neither input is modified.  Imported/model-derived annotations are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


MAX_JSON_BYTES = 32 * 1024 * 1024
MANUAL_SOURCES = {"drawn": "drawn_roi_", "brush": "brush_roi_"}
FORBIDDEN_KEYS = {"source_role", "source_file", "model", "model_id", "prediction", "predicted"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"Missing or oversized JSON: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def bound_image(project: dict, project_path: Path, registration: dict, image_path: Path) -> dict:
    tissues = project.get("tissues")
    if not isinstance(tissues, list) or len(tissues) != 1:
        raise ValueError("A copied experiment with exactly one tissue is required")
    image = tissues[0].get("image")
    native = registration.get("cell_full_image")
    if not isinstance(image, dict) or not isinstance(native, dict):
        raise ValueError("Project or registration has no native image frame")
    if Path(image_path).resolve() != (project_path.parent / image.get("path", "")).resolve():
        raise ValueError("Image path differs from copied project tissue image")
    keys = ("frame_id", "sha256", "width", "height", "mpp_x", "mpp_y",
            "origin_x", "origin_y", "coordinate_space", "coordinate_convention")
    if any(image.get(key) != native.get(key) for key in keys):
        raise ValueError("Copied project image frame differs from registration")
    if digest(image_path) != native["sha256"]:
        raise ValueError("Native image bytes differ from registered SHA-256")
    if (native.get("coordinate_space") != "crop-native-pixels" or
            native.get("coordinate_convention") != "pixel-edge-origin_centres-in-native-pixels" or
            native.get("origin_x") != 0 or native.get("origin_y") != 0):
        raise ValueError("Only registered zero-origin native pixel-edge images are accepted")
    for key in ("width", "height", "mpp_x", "mpp_y"):
        value = native.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid native image {key}")
    return native


def prepare(project_path: Path, registration_path: Path, image_path: Path,
            inventory_path: Path, export_path: Path, review_path: Path) -> dict:
    project = read_json(project_path)
    registration = read_json(registration_path)
    export = read_json(export_path)
    review = read_json(review_path)
    native = bound_image(project, project_path, registration, image_path)
    if export.get("type") != "FeatureCollection" or not isinstance(export.get("features"), list):
        raise ValueError("wsiTools GeoJSON FeatureCollection required")
    export_sha = digest(export_path)
    if review.get("format") != "cellphenotyper-wsitools-expert-review-v1":
        raise ValueError("Review manifest format missing or unsupported")
    if review.get("source_geojson_sha256") != export_sha:
        raise ValueError("Review manifest is not bound to exact exported GeoJSON")
    if review.get("project_sha256") != digest(project_path):
        raise ValueError("Review manifest is not bound to copied experiment project")
    class_names = review.get("class_names")
    if (not isinstance(class_names, dict) or len(class_names) < 2 or
            any(not isinstance(k, str) or not k.isdecimal() or not 1 <= int(k) <= 253 or
                not isinstance(v, str) or not v.strip() or v.strip().lower() == "annotation"
                for k, v in class_names.items()) or
            len(set(class_names.values())) != len(class_names)):
        raise ValueError("At least two explicitly named tissue classes are required")
    assignments = review.get("reviewed_features")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("No reviewed feature assignments")
    rows = []
    ids = set()
    for feature in export["features"]:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError("Every exported entry must be a GeoJSON Feature")
        fid = feature.get("id")
        if not isinstance(fid, str) or not fid or fid in ids:
            raise ValueError("Every exported ROI needs a distinct stable string ID")
        ids.add(fid)
        props = feature.get("properties")
        if not isinstance(props, dict):
            raise ValueError(f"ROI {fid} has no properties")
        src = props.get("source")
        if src not in MANUAL_SOURCES or not fid.startswith(MANUAL_SOURCES[src]):
            raise ValueError(f"ROI {fid} is not an explicitly drawn/brush ROI; imported/predicted layers are refused")
        if any(key in props and props[key] not in (None, "") for key in FORBIDDEN_KEYS):
            raise ValueError(f"ROI {fid} carries model/import provenance")
        meta = props.get("wsiTools")
        if (not isinstance(meta, dict) or meta.get("coordinate_space") != "level0_slide_pixels" or
                meta.get("display_transform_applied") is not False or
                meta.get("slide_width") != native["width"] or
                meta.get("slide_height") != native["height"]):
            raise ValueError(f"ROI {fid} is not in the registered level-0 image frame")
        assignment = assignments.get(fid)
        if not isinstance(assignment, dict):
            raise ValueError(f"ROI {fid} has no reviewer assignment")
        cid = assignment.get("class_id")
        split = assignment.get("split")
        annotator = assignment.get("annotator_id")
        roi_id = assignment.get("roi_id")
        viewer_class = props.get("class")
        classified = props.get("classification")
        if (type(cid) is not int or str(cid) not in class_names or
                viewer_class != class_names[str(cid)] or
                not isinstance(classified, dict) or classified.get("name") != viewer_class):
            raise ValueError(f"ROI {fid} class does not match reviewer taxonomy and viewer class")
        if (split not in ("train", "holdout") or not isinstance(annotator, str) or not annotator.strip() or
                not isinstance(roi_id, str) or not roi_id.strip() or
                assignment.get("annotation_source") != "independent_expert"):
            raise ValueError(f"ROI {fid} needs explicit split, annotator, ROI ID and independent expert provenance")
        geom = feature.get("geometry")
        # Reuse the exact downstream polygon/bounds validator rather than
        # silently repairing a viewer export or altering expert geometry.
        from bora_semantic_roi_intake import _geometry
        _geometry(geom, native["width"], native["height"])
        rows.append({"type": "Feature", "geometry": geom,
                     "properties": {"annotation_id": fid, "class_id": cid,
                                    "split": split, "roi_id": roi_id,
                                    "annotator_id": annotator,
                                    "annotation_source": "independent_expert"}})
    if set(assignments) != ids:
        raise ValueError("Review manifest has extra or missing feature IDs")
    expected_classes = {int(k) for k in class_names}
    by_split = {split: [f for f in rows if f["properties"]["split"] == split]
                for split in ("train", "holdout")}
    for split, selected in by_split.items():
        if {f["properties"]["class_id"] for f in selected} != expected_classes:
            raise ValueError(f"Every declared class needs a {split} ROI")
    if ({f["properties"]["roi_id"] for f in by_split["train"]} &
            {f["properties"]["roi_id"] for f in by_split["holdout"]}):
        raise ValueError("Train and holdout ROI IDs must be disjoint")
    from bora_semantic_roi_intake import _geometry
    geometries = [(f["properties"], f["geometry"],
                   _geometry(f["geometry"], native["width"], native["height"]))
                  for f in rows]
    for i, (a, geom_a, bounds_a) in enumerate(geometries):
        for b, geom_b, bounds_b in geometries[i + 1:]:
            ax0, ay0, ax1, ay1 = bounds_a
            bx0, by0, bx1, by1 = bounds_b
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                try:
                    from shapely.geometry import shape
                except ImportError as exc:
                    raise ValueError("Potentially overlapping ROI bounds require Shapely for exact topology") from exc
                if shape(geom_a).intersection(shape(geom_b)).area > 0:
                    raise ValueError("Expert ROIs overlap")
            if a["split"] != b["split"]:
                dx = max(0, ax0 - bx1, bx0 - ax1) * native["mpp_x"]
                dy = max(0, ay0 - by1, by0 - ay1) * native["mpp_y"]
                if math.hypot(dx, dy) < 224:
                    raise ValueError("Train and holdout ROI bounds need a 224 µm spatial gap")
    inventory_sha = digest(inventory_path)
    return {"type": "FeatureCollection", "schema": "cellphenotyper-expert-tissue-v1",
            "source": {"source_frame_id": native["frame_id"],
                       "image_sha256": native["sha256"],
                       "native_shape_yx": [native["height"], native["width"]],
                       "native_mpp_xy": [native["mpp_x"], native["mpp_y"]],
                       "token_inventory_sha256": inventory_sha,
                       "coordinate_space": "native_image_pixel_edges",
                       "copied_project_sha256": digest(project_path),
                       "export_geojson_sha256": export_sha,
                       "review_manifest_sha256": digest(review_path)},
            "class_names": class_names, "features": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "registration", "image", "inventory", "geojson", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh path")
    result = prepare(args.project, args.registration, args.image, args.inventory,
                     args.geojson, args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(result['features'])} independently reviewed ROIs to {args.output}")


if __name__ == "__main__":
    main()
