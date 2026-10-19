#!/usr/bin/env python3
"""Validate an ROI GeoJSON against its exact level-0 analysis image."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.validation import explain_validity


ASSOCIATION_KEYS = ("source_image", "image", "image_name", "slide", "slide_name")
COORDINATE_SPACE_KEYS = ("coordinate_space", "coordinate_system")
PIXEL_COORDINATE_SPACES = {
    "level0_pixels",
    "image_level0_pixels",
    "full_image_level0_pixels",
    "pixel",
    "pixels",
}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_image_id(value: Any) -> str:
    name = Path(str(value)).name.strip().lower()
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif", ".btf", ".svs", ".ndpi", ".czi"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def nested_dicts(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield payload
    for key in ("properties", "metadata", "pipeline_metadata"):
        value = payload.get(key)
        if isinstance(value, dict):
            yield value
    for feature in payload.get("features") or []:
        if not isinstance(feature, dict):
            continue
        yield feature
        properties = feature.get("properties")
        if isinstance(properties, dict):
            yield properties


def collect_declared_values(payload: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for item in nested_dicts(payload):
        for key in keys:
            value = item.get(key)
            if value is None or isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text and text not in values:
                values.append(text)
    return values


def image_identity_matches(value: str, image_path: Path, sample_id: str) -> bool:
    observed = canonical_image_id(value)
    sample_identity = canonical_image_id(sample_id)
    expected = {canonical_image_id(image_path.name), sample_identity}
    if "__" in sample_identity:
        expected.add(sample_identity.split("__", 1)[0])
    return bool(observed and observed in expected)


def parse_crs(payload: dict[str, Any]) -> str | None:
    crs = payload.get("crs")
    if crs is None:
        return None
    if isinstance(crs, str):
        return crs
    if isinstance(crs, dict):
        properties = crs.get("properties")
        if isinstance(properties, dict):
            for key in ("name", "href", "code"):
                if properties.get(key) is not None:
                    return str(properties[key])
        if crs.get("name") is not None:
            return str(crs["name"])
    return json.dumps(crs, sort_keys=True)


def polygon_parts(geometry: Polygon | MultiPolygon) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    return list(geometry.geoms)


def infer_image_identity(image_path: Path, image_report_path: Path) -> tuple[int, int, str | None, dict]:
    report = json.loads(image_report_path.read_text(encoding="utf-8"))
    width = int(report.get("width_px") or 0)
    height = int(report.get("height_px") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Converted-image QC report has invalid dimensions: width={width}, height={height}"
        )
    reported_image = Path(str(report.get("image") or "")).name
    if reported_image and canonical_image_id(reported_image) != canonical_image_id(image_path.name):
        raise RuntimeError(
            "Converted-image QC report does not belong to the staged analysis image: "
            f"report={reported_image}, image={image_path.name}"
        )
    return width, height, report.get("file_sha256"), report


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    roi_path = Path(args.geojson).resolve()
    image_path = Path(args.image).resolve()
    image_report_path = Path(args.image_report).resolve()
    if not roi_path.is_file() or roi_path.stat().st_size <= 0:
        raise RuntimeError(f"ROI GeoJSON is missing or empty: {roi_path}")
    if not image_path.is_file() or image_path.stat().st_size <= 0:
        raise RuntimeError(f"Analysis image is missing or empty: {image_path}")

    payload = json.loads(roi_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise RuntimeError("ROI must be a GeoJSON FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise RuntimeError("ROI FeatureCollection must contain at least one feature")

    width, height, image_sha256, image_report = infer_image_identity(
        image_path, image_report_path
    )
    frame = box(0.0, 0.0, float(width), float(height))
    warnings: list[str] = []
    failures: list[str] = []
    feature_records: list[dict[str, Any]] = []
    polygon_count = 0
    hole_count = 0
    vertex_count = 0
    total_area_px2 = 0.0
    outside_area_px2 = 0.0

    declared_images = collect_declared_values(payload, ASSOCIATION_KEYS)
    mismatched_images = [
        value
        for value in declared_images
        if not image_identity_matches(value, image_path, args.sample_id)
    ]
    if mismatched_images:
        failures.append(
            "ROI image association does not match the analysis image/sample: "
            + ", ".join(mismatched_images)
        )
    elif not declared_images and args.source_kind == "provided":
        warnings.append(
            "Provided ROI has no source-image identifier; association relies on the explicit pipeline input pairing."
        )

    coordinate_spaces = collect_declared_values(payload, COORDINATE_SPACE_KEYS)
    unsupported_spaces = [
        value for value in coordinate_spaces if value.strip().lower() not in PIXEL_COORDINATE_SPACES
    ]
    if unsupported_spaces:
        failures.append(
            "ROI coordinate space must be level-0 image pixels; unsupported declaration(s): "
            + ", ".join(unsupported_spaces)
        )
    elif not coordinate_spaces and args.source_kind == "provided":
        warnings.append(
            "Provided ROI has no coordinate-space declaration; coordinates are interpreted as level-0 image pixels."
        )

    declared_crs = parse_crs(payload)
    if declared_crs:
        normalized_crs = declared_crs.strip().lower()
        if normalized_crs not in PIXEL_COORDINATE_SPACES:
            failures.append(
                "Geographic/world GeoJSON CRS is unsupported for image ROIs; expected level-0 pixel coordinates "
                f"but observed crs={declared_crs}."
            )

    for feature_index, feature in enumerate(features):
        record: dict[str, Any] = {"feature_index": feature_index, "status": "invalid"}
        if not isinstance(feature, dict) or feature.get("type") not in (None, "Feature"):
            record["issues"] = ["Entry is not a GeoJSON Feature"]
            failures.append(f"ROI feature {feature_index} is not a GeoJSON Feature")
            feature_records.append(record)
            continue
        geometry_payload = feature.get("geometry")
        if not isinstance(geometry_payload, dict):
            record["issues"] = ["Missing geometry"]
            failures.append(f"ROI feature {feature_index} has no geometry")
            feature_records.append(record)
            continue
        if geometry_payload.get("type") not in {"Polygon", "MultiPolygon"}:
            record["geometry_type"] = geometry_payload.get("type")
            record["issues"] = ["Geometry must be Polygon or MultiPolygon"]
            failures.append(
                f"ROI feature {feature_index} uses unsupported geometry type {geometry_payload.get('type')}"
            )
            feature_records.append(record)
            continue
        try:
            geometry = shape(geometry_payload)
        except Exception as exc:
            record["issues"] = [f"Geometry parse error: {exc}"]
            failures.append(f"ROI feature {feature_index} geometry cannot be parsed: {exc}")
            feature_records.append(record)
            continue
        if geometry.is_empty or geometry.area <= 0:
            record["issues"] = ["Geometry is empty or has zero area"]
            failures.append(f"ROI feature {feature_index} is empty or has zero area")
            feature_records.append(record)
            continue
        if not geometry.is_valid:
            reason = explain_validity(geometry)
            record["issues"] = [f"Invalid geometry: {reason}"]
            failures.append(
                f"ROI feature {feature_index} is invalid ({reason}); geometry was not repaired automatically"
            )
            feature_records.append(record)
            continue
        bounds = [float(value) for value in geometry.bounds]
        if not all(math.isfinite(value) for value in bounds):
            record["issues"] = ["Geometry contains non-finite coordinates"]
            failures.append(f"ROI feature {feature_index} contains non-finite coordinates")
            feature_records.append(record)
            continue

        outside_area = float(geometry.difference(frame).area)
        outside_fraction = outside_area / max(float(geometry.area), 1.0e-12)
        parts = polygon_parts(geometry)
        feature_holes = sum(len(part.interiors) for part in parts)
        feature_vertices = sum(
            len(part.exterior.coords)
            + sum(len(interior.coords) for interior in part.interiors)
            for part in parts
        )
        record.update(
            {
                "status": "valid" if outside_area <= args.bounds_tolerance_px2 else "out_of_bounds",
                "geometry_type": geometry.geom_type,
                "bounds_xyxy": bounds,
                "area_px2": float(geometry.area),
                "outside_image_area_px2": outside_area,
                "outside_image_fraction": outside_fraction,
                "polygon_count": len(parts),
                "hole_count": feature_holes,
                "vertex_count": feature_vertices,
            }
        )
        if outside_area > args.bounds_tolerance_px2:
            failures.append(
                f"ROI feature {feature_index} extends outside the level-0 image by "
                f"{outside_area:.6g} px^2 ({outside_fraction:.3%}); geometry was not clipped"
            )
        polygon_count += len(parts)
        hole_count += feature_holes
        vertex_count += feature_vertices
        total_area_px2 += float(geometry.area)
        outside_area_px2 += outside_area
        feature_records.append(record)

    roi_sha256 = sha256_file(roi_path)
    accepted_geometry_count = sum(
        1 for record in feature_records if record.get("status") == "valid"
    )
    if accepted_geometry_count == 0:
        failures.append("ROI contains no valid in-bounds polygon geometry")

    passed = not failures
    mode = str(args.mode).lower()
    accepted = passed or mode == "warn"
    status = "pass" if passed else ("warning" if accepted else "fail")
    report = {
        "schema_version": 1,
        "status": status,
        "mode": mode,
        "accepted": accepted,
        "geometry_modified": False,
        "repair_policy": "never_modify_input_geometry",
        "roi_path": str(roi_path),
        "roi_source_kind": args.source_kind,
        "roi_source_name": args.source_name,
        "roi_sha256": roi_sha256,
        "roi_size_bytes": roi_path.stat().st_size,
        "sample_id": args.sample_id,
        "image_path": str(image_path),
        "image_width_px": width,
        "image_height_px": height,
        "image_sha256": image_sha256,
        "image_qc_report": str(image_report_path),
        "image_qc_status": image_report.get("status"),
        "declared_image_associations": declared_images,
        "mismatched_image_associations": mismatched_images,
        "declared_coordinate_spaces": coordinate_spaces,
        "declared_crs": declared_crs,
        "coordinate_interpretation": "level0_pixels",
        "feature_count": len(features),
        "valid_in_bounds_feature_count": accepted_geometry_count,
        "polygon_count": polygon_count,
        "hole_count": hole_count,
        "vertex_count": vertex_count,
        "total_area_px2_without_overlap_correction": total_area_px2,
        "outside_image_area_px2": outside_area_px2,
        "bounds_tolerance_px2": args.bounds_tolerance_px2,
        "warnings": warnings,
        "failures": failures,
        "features": feature_records,
    }
    return report, accepted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geojson", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-report", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--source-kind", choices=("provided", "generated"), required=True)
    parser.add_argument("--source-name", default="")
    parser.add_argument("--mode", choices=("fail", "warn"), default="fail")
    parser.add_argument("--bounds-tolerance-px2", type=float, default=1.0e-6)
    args = parser.parse_args()
    if args.bounds_tolerance_px2 < 0:
        parser.error("--bounds-tolerance-px2 must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    try:
        report, accepted = validate(args)
    except Exception as exc:
        accepted = args.mode == "warn"
        report = {
            "schema_version": 1,
            "status": "warning" if accepted else "fail",
            "mode": args.mode,
            "accepted": accepted,
            "geometry_modified": False,
            "roi_path": str(Path(args.geojson).resolve()),
            "image_path": str(Path(args.image).resolve()),
            "sample_id": args.sample_id,
            "warnings": [],
            "failures": [str(exc)],
        }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    stream = sys.stdout if accepted else sys.stderr
    print(
        f"[{'INFO' if accepted else 'ERROR'}] ROI validation status={report['status']} "
        f"features={report.get('valid_in_bounds_feature_count', 0)}/"
        f"{report.get('feature_count', 0)} report={report_path}",
        file=stream,
        flush=True,
    )
    for message in report.get("warnings", []):
        print(f"[WARN] {message}", file=stream, flush=True)
    for message in report.get("failures", []):
        print(f"[ERROR] {message}", file=stream, flush=True)
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
