#!/usr/bin/env python3
"""Coordinate-frame helpers for GeoJSON produced from an analysis crop.

Published tissue annotations use level-zero pixel-edge coordinates of the
original converted OME-TIFF.  Working rasters remain in crop coordinates.
The exact integer translation is read from PREPARE_ANALYSIS_CROP's shift.json;
no scale, rotation or geometry simplification is applied here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


COORDINATE_SCHEMA = "cellphenotyper.geojson_coordinates.v1"
ORIGINAL_FRAME = "original_image_level0_pixel_edges"
CROP_FRAME = "analysis_crop_level0_pixel_edges"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_integer(value: Any, name: str) -> int:
    result = _finite_number(value, name)
    if result <= 0 or int(result) != result:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def read_crop_transform(path: str | Path) -> dict:
    source = Path(path)
    payload = json.loads(source.read_text())
    offset = payload.get("offset_crop_to_original")
    crop_size = payload.get("crop_size")
    if not isinstance(offset, dict) or not isinstance(crop_size, dict):
        raise ValueError(
            "shift.json must contain offset_crop_to_original and crop_size"
        )
    dx = _finite_number(offset.get("dx"), "offset_crop_to_original.dx")
    dy = _finite_number(offset.get("dy"), "offset_crop_to_original.dy")
    if int(dx) != dx or int(dy) != dy or dx < 0 or dy < 0:
        raise ValueError("The crop-to-original offset must use non-negative integers")
    width = _positive_integer(crop_size.get("width"), "crop_size.width")
    height = _positive_integer(crop_size.get("height"), "crop_size.height")
    full = payload.get("full_size")
    full_size = None
    if full is not None:
        if not isinstance(full, dict):
            raise ValueError("full_size must be an object when provided")
        full_size = {
            "width": _positive_integer(full.get("width"), "full_size.width"),
            "height": _positive_integer(full.get("height"), "full_size.height"),
        }
        if int(dx) + width > full_size["width"] or int(dy) + height > full_size["height"]:
            raise ValueError("Crop extent exceeds the declared original image dimensions")
    bbox = payload.get("crop_bbox_xyxy")
    if bbox is not None:
        if not isinstance(bbox, dict) or not {"x0", "y0", "x1", "y1"} <= set(bbox):
            raise ValueError("crop_bbox_xyxy is incomplete")
        observed = tuple(
            _finite_number(bbox[key], f"crop_bbox_xyxy.{key}")
            for key in ("x0", "y0", "x1", "y1")
        )
        if observed != (dx, dy, dx + width, dy + height):
            raise ValueError("Crop offset, size and bounding box disagree")
    return {
        "schema_version": COORDINATE_SCHEMA,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "dx": int(dx),
        "dy": int(dy),
        "crop_size": {"width": width, "height": height},
        "full_size": full_size,
    }


def _translate_position(position: Any, dx: float, dy: float) -> list:
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        raise ValueError("GeoJSON coordinate position must have at least x and y")
    x = _finite_number(position[0], "GeoJSON x")
    y = _finite_number(position[1], "GeoJSON y")
    return [x + dx, y + dy, *list(position[2:])]


def _translate_coordinates(value: Any, depth: int, dx: float, dy: float) -> Any:
    if depth == 1:
        return _translate_position(value, dx, dy)
    if not isinstance(value, (list, tuple)):
        raise ValueError("Malformed nested GeoJSON coordinates")
    return [_translate_coordinates(item, depth - 1, dx, dy) for item in value]


_COORDINATE_DEPTH = {
    "Point": 1,
    "MultiPoint": 2,
    "LineString": 2,
    "MultiLineString": 3,
    "Polygon": 3,
    "MultiPolygon": 4,
}


def translate_geometry(geometry: dict | None, dx: float, dy: float) -> dict | None:
    if geometry is None:
        return None
    if not isinstance(geometry, dict):
        raise ValueError("GeoJSON geometry must be an object or null")
    kind = geometry.get("type")
    translated = dict(geometry)
    if kind == "GeometryCollection":
        translated["geometries"] = [
            translate_geometry(child, dx, dy)
            for child in geometry.get("geometries", [])
        ]
    elif kind in _COORDINATE_DEPTH:
        translated["coordinates"] = _translate_coordinates(
            geometry.get("coordinates"), _COORDINATE_DEPTH[kind], dx, dy
        )
    else:
        raise ValueError(f"Unsupported GeoJSON geometry type: {kind!r}")
    translated.pop("bbox", None)
    return translated


def _geometry_bounds(geometry: dict | None) -> tuple[float, float, float, float] | None:
    if geometry is None:
        return None
    values: list[tuple[float, float]] = []

    def collect(value: Any, depth: int) -> None:
        if depth == 1:
            point = _translate_position(value, 0.0, 0.0)
            values.append((float(point[0]), float(point[1])))
        else:
            for item in value:
                collect(item, depth - 1)

    kind = geometry.get("type")
    if kind == "GeometryCollection":
        child_bounds = [
            bound for child in geometry.get("geometries", [])
            if (bound := _geometry_bounds(child)) is not None
        ]
        if not child_bounds:
            return None
        return (
            min(bound[0] for bound in child_bounds),
            min(bound[1] for bound in child_bounds),
            max(bound[2] for bound in child_bounds),
            max(bound[3] for bound in child_bounds),
        )
    if kind not in _COORDINATE_DEPTH:
        raise ValueError(f"Unsupported GeoJSON geometry type: {kind!r}")
    collect(geometry.get("coordinates"), _COORDINATE_DEPTH[kind])
    if not values:
        return None
    return (
        min(point[0] for point in values),
        min(point[1] for point in values),
        max(point[0] for point in values),
        max(point[1] for point in values),
    )


def rebase_feature_collection(
    payload: dict,
    transform: dict,
    *,
    source_frame: str = CROP_FRAME,
    target_frame: str = ORIGINAL_FRAME,
) -> dict:
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        raise ValueError("GeoJSON must be a FeatureCollection")
    dx, dy = int(transform["dx"]), int(transform["dy"])
    crop = transform["crop_size"]
    tolerance = 1e-6
    translated_features = []
    for index, feature in enumerate(payload["features"], 1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError(f"GeoJSON member {index} is not a Feature")
        bounds = _geometry_bounds(feature.get("geometry"))
        if bounds is not None and (
            bounds[0] < -tolerance or bounds[1] < -tolerance
            or bounds[2] > crop["width"] + tolerance
            or bounds[3] > crop["height"] + tolerance
        ):
            raise ValueError(
                f"Feature {index} lies outside the analysis crop: bounds={bounds}, "
                f"crop={crop}"
            )
        item = dict(feature)
        item["geometry"] = translate_geometry(feature.get("geometry"), dx, dy)
        item.pop("bbox", None)
        properties = dict(feature.get("properties") or {})
        properties["coordinate_space"] = target_frame
        item["properties"] = properties
        translated_features.append(item)
    result = dict(payload)
    result["features"] = translated_features
    result.pop("bbox", None)
    result["cellphenotyper_coordinate_system"] = {
        "schema_version": COORDINATE_SCHEMA,
        "coordinate_space": target_frame,
        "units": "level_zero_pixels",
        "pixel_reference": "pixel_edges",
        "source_coordinate_space": source_frame,
        "transform": {
            "type": "translation",
            "crop_to_original_offset_xy": [dx, dy],
            "scale_xy": [1.0, 1.0],
        },
        "crop_size_xy": [crop["width"], crop["height"]],
        "original_size_xy": (
            [transform["full_size"]["width"], transform["full_size"]["height"]]
            if transform.get("full_size") else None
        ),
        "shift_json_sha256": transform["source_sha256"],
    }
    return result


def rebase_geojson_file(path: str | Path, shift_json: str | Path) -> dict:
    target = Path(path)
    transform = read_crop_transform(shift_json)
    payload = rebase_feature_collection(json.loads(target.read_text()), transform)
    target.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    return transform


def original_to_crop_feature_collection(payload: dict, shift_json: str | Path) -> dict:
    transform = read_crop_transform(shift_json)
    metadata = payload.get("cellphenotyper_coordinate_system")
    if not isinstance(metadata, dict) or metadata.get("coordinate_space") != ORIGINAL_FRAME:
        raise ValueError("Published tissue GeoJSON does not declare original-image coordinates")
    declared = metadata.get("transform", {}).get("crop_to_original_offset_xy")
    if declared != [transform["dx"], transform["dy"]]:
        raise ValueError("GeoJSON crop offset does not match shift.json")
    declared_hash = metadata.get("shift_json_sha256")
    if declared_hash is not None and declared_hash != transform["source_sha256"]:
        raise ValueError("GeoJSON shift-file hash does not match shift.json")
    result = dict(payload)
    result["features"] = []
    crop = transform["crop_size"]
    tolerance = 1e-6
    for index, feature in enumerate(payload.get("features", []), 1):
        item = dict(feature)
        item["geometry"] = translate_geometry(
            feature.get("geometry"), -transform["dx"], -transform["dy"]
        )
        bounds = _geometry_bounds(item["geometry"])
        if bounds is not None and (
            bounds[0] < -tolerance or bounds[1] < -tolerance
            or bounds[2] > crop["width"] + tolerance
            or bounds[3] > crop["height"] + tolerance
        ):
            raise ValueError(
                f"Feature {index} does not map back inside the analysis crop: "
                f"bounds={bounds}, crop={crop}"
            )
        properties = dict(feature.get("properties") or {})
        properties["coordinate_space"] = CROP_FRAME
        item["properties"] = properties
        result["features"].append(item)
    result["cellphenotyper_coordinate_system"] = {
        **metadata,
        "coordinate_space": CROP_FRAME,
        "source_coordinate_space": ORIGINAL_FRAME,
        "internal_inverse_transform_applied": True,
    }
    return result
