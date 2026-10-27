#!/usr/bin/env python3
"""Export or validate portable CellPhenotyper experiments from existing files.

No encoder, workflow engine, network client or GPU library is imported here.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote


SCHEMA = "cellphenotyper-experiment"
VERSION = 1
FILENAME = "experiment.cellphenotyper.json"
FEATURE_WIDTHS = {"virchow2-pooled": ("tile", 2560), "virchow2-spatial": ("token", 1280)}
COORDINATE_CONVENTION = "pixel-edge-origin_centres-in-native-pixels"
NON_PREDICTORS = {"label", "class", "class_id", "cluster", "sample_id", "tissue_id", "specimen_id",
                  "patient_id", "split", "observation_type", "area_px", "source_id", "cell_id",
                  "probability", "type", "type_prob"}
SPATIAL_ROLES = {"spatial_transcriptomics_object": "rds", "spatial_review_bundle": "json"}
SPATIAL_BUNDLE_SCHEMA = "cellphenotyper-spatial-review-bundle"
SPATIAL_MAX_BYTES = 16 * 1024 * 1024
SPATIAL_MAX_OBSERVATIONS = 10000
GIGATIME_SCORE_SEMANTICS = "uncalibrated_virtual_marker_score"
GIGATIME_ROLES = {"gigatime_marker_image", "cell_segmentation_mask", "gigatime_cell_scores"}
LOCAL_BORA_ROLES = {"bora_local_tumor_probability", "bora_local_uncertainty",
    "bora_local_exact_semantic_mask", "bora_local_review_geometry",
    "bora_local_review_regions", "bora_local_prediction_provenance"}
LOCAL_BORA_OPTIONAL_ROLES = {"bora_local_exact_semantic_vector"}
LOCAL_BORA_RASTER_ROLES = {"bora_local_tumor_probability", "bora_local_uncertainty",
    "bora_local_exact_semantic_mask"}
LOCAL_BORA_REVIEW_REASONS = {"high_uncertainty", "cell_disagreement",
    "boundary_disagreement", "uncertain_boundary"}
LOCAL_BORA_PROVENANCE_FORMAT = "cellphenotyper-bora-local-prediction-v1"
LOCAL_BORA_MAX_REVIEW_BYTES = 2 * 1024 * 1024
LOCAL_BORA_MAX_REVIEW_REGIONS = 1000
CELL_CLASSIFICATION_SCHEMA = "cellphenotyper-cell-classification"
CELL_FEATURE_KINDS = {"cellvit-cell", "cell-morphology"}
IDENTITY_AFFINE = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
CELL_PRODUCER_FOLDERS = ("03c_cellvitpp", "03_cellvitpp")
SEGMENTATION_REVIEW_FOLDERS = ("03e_cell_segmentation_review", "03_cell_segmentation_review")


class ExperimentError(ValueError):
    """Invalid or incomplete cached experiment assets."""


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ExperimentError(f"{label} must be an object")
    return value


def _array(value, label):
    if not isinstance(value, list):
        raise ExperimentError(f"{label} must be an array")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ExperimentError(f"{label} must be a non-empty string")
    return value


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ExperimentError(f"{label} must be a finite number")
    if positive and value <= 0:
        raise ExperimentError(f"{label} must be positive")
    return value


def _identifier(value, label):
    value = _text(value, label)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value):
        raise ExperimentError(f"{label} may contain only letters, digits, dots, colons, underscores and hyphens")
    return value


def _digest(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", value):
        raise ExperimentError(f"{label} must be a SHA-256 digest")
    return value.lower()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(value, root, label):
    value = _text(value, label)
    if "://" in value:
        raise ExperimentError(f"{label} must reference an existing local file; remote downloads are never automatic")
    # Preserve published symlinks instead of replacing them with ephemeral cache targets.
    path = Path(os.path.abspath(Path(root) / Path(value).expanduser()))
    if not path.is_file():
        raise ExperimentError(f"{label} is missing: {path}")
    return path


def _asset(record, root, destination, label):
    record = copy.deepcopy(_mapping(record, label))
    path = _path(record.get("path"), root, f"{label}.path")
    actual = sha256(path)
    if record.get("sha256") is not None and _digest(record["sha256"], f"{label}.sha256") != actual:
        raise ExperimentError(f"{label} SHA-256 mismatch: {path}")
    if path == destination or (destination.exists() and os.path.samefile(path, destination)):
        raise ExperimentError(f"Output would overwrite a source asset: {path}")
    record.update(path=Path(os.path.relpath(path, destination.parent)).as_posix(), sha256=actual)
    return record, path


def image_shape(path):
    """Read image headers only, never the complete image array."""
    try:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".ppm"}:
            from PIL import Image
            with Image.open(path) as image:
                return image.width, image.height
        import tifffile
        with tifffile.TiffFile(path) as image:
            return int(image.pages[0].imagewidth), int(image.pages[0].imagelength)
    except (ImportError, OSError, ValueError) as exc:
        raise ExperimentError(f"Cannot inspect image header {path}: {exc}") from exc


def _csv(path):
    return (gzip.open if path.name.lower().endswith(".gz") else open)(path, "rt", newline="", encoding="utf-8-sig")


def _validate_features(feature, path, image, label):
    kind = feature.get("kind")
    if kind == "cellvit-cell" and feature.get("format") == "partitioned-sqlite":
        from index_cellvit_partitions import POINTER_SCHEMA
        pointer = _read_json(path)
        pointer_image = pointer.get("image")
        identity_keys = ("sha256", "width", "height", "mpp_x", "mpp_y",
                         "frame_id", "coordinate_convention", "coordinate_space",
                         "origin_x", "origin_y")
        descriptor = _mapping(feature.get("descriptor"), f"{label}.descriptor")
        columns = feature.get("feature_columns")
        if (feature.get("observation_unit") != "cell" or feature.get("coordinate_frame") != "image" or
                feature.get("frame_id") != image.get("frame_id") or
                type(feature.get("row_count")) is not int or feature["row_count"] < 0 or
                descriptor.get("backbone") not in {"SAM", "HIPT", "Virchow2"} or
                type(descriptor.get("feature_count")) is not int or
                not 0 < descriptor["feature_count"] <= 4096 or
                not isinstance(columns, list) or
                len(columns) != descriptor["feature_count"] or
                len(set(columns)) != len(columns) or NON_PREDICTORS.intersection(columns) or
                pointer.get("schema") != POINTER_SCHEMA or
                pointer.get("feature_storage") != "partition-row-reference-v1" or
                not isinstance(pointer_image, dict) or
                any(pointer_image.get(key) != image.get(key) for key in identity_keys) or
                pointer.get("feature_set_id") != feature.get("feature_set_id") or
                pointer.get("feature_columns") != columns or
                pointer.get("descriptor") != descriptor or
                pointer.get("row_count") != feature["row_count"]):
            raise ExperimentError(f"{label}: invalid indexed CellViT feature pointer or source frame")
        return feature["row_count"]
    if kind in CELL_FEATURE_KINDS:
        return _validate_cell_features(feature, path, image, label)
    if kind not in FEATURE_WIDTHS:
        raise ExperimentError(f"{label}.kind must be virchow2-pooled or virchow2-spatial")
    unit, width = FEATURE_WIDTHS[kind]
    if feature.get("observation_unit") != unit:
        raise ExperimentError(f"{label}: {kind} requires observation_unit={unit}")
    expected_format = "csv.gz" if path.name.lower().endswith(".csv.gz") else "csv" if path.suffix.lower() == ".csv" else None
    if expected_format is None or feature.get("format") != expected_format:
        raise ExperimentError(f"{label}.format must match an existing .csv or .csv.gz feature table")
    columns = feature.get("feature_columns")
    if (not isinstance(columns, list) or len(columns) != width or
            any(not isinstance(x, str) or not x.strip() for x in columns) or len(set(columns)) != width):
        raise ExperimentError(f"{label}.feature_columns must contain exactly {width} unique feature names")
    keys = [_text(feature.get(key), f"{label}.{key}") for key in ("id_column", "x_column", "y_column")]
    if len(set(keys)) != 3 or set(keys) & set(columns):
        raise ExperimentError(f"{label}: identity/coordinate columns must be distinct from predictor columns")
    if NON_PREDICTORS.intersection(columns):
        raise ExperimentError(f"{label}: labels, splits, identities and measurement metadata are not predictor columns")
    clusters = feature.get("cluster_columns", [])
    if (not isinstance(clusters, list) or any(not isinstance(x, str) or not x for x in clusters) or
            len(set(clusters)) != len(clusters) or set(clusters) & set(columns + keys)):
        raise ExperimentError(f"{label}: cluster_columns must be unique metadata columns excluded from predictors and coordinates")
    if feature.get("coordinate_frame") != "image":
        raise ExperimentError(f"{label}.coordinate_frame must be image; transform cached coordinates explicitly first")
    encoder = _mapping(feature.get("encoder"), f"{label}.encoder")
    if encoder.get("name") != "Virchow2":
        raise ExperimentError(f"{label}.encoder.name must be Virchow2")
    if not isinstance(encoder.get("revision"), str) or not re.fullmatch(r"[a-fA-F0-9]{40}", encoder["revision"]):
        raise ExperimentError(f"{label}.encoder.revision must be an immutable 40-character commit")
    _digest(encoder.get("checkpoint_sha256"), f"{label}.encoder.checkpoint_sha256")
    preprocessing = _mapping(feature.get("preprocessing"), f"{label}.preprocessing")
    if not preprocessing:
        raise ExperimentError(f"{label}.preprocessing must describe the cached feature definition")
    grid = _mapping(feature.get("grid"), f"{label}.grid")
    for key in ("tile_width_px", "tile_height_px", "spacing_x_px", "spacing_y_px"):
        _number(grid.get(key), f"{label}.grid.{key}", positive=True)
    for key in ("origin_x_px", "origin_y_px"):
        _number(grid.get(key), f"{label}.grid.{key}")
    declared_edge_policy = grid.get("edge_policy")
    if declared_edge_policy is not None or grid.get("representative_policy") is not None:
        if (declared_edge_policy != "white-padded" or
                grid.get("representative_policy") != "logical-centre-clipped-to-image" or
                unit != "tile" or any(grid[key] != int(grid[key]) for key in
                    ("tile_width_px", "tile_height_px", "spacing_x_px", "spacing_y_px", "origin_x_px", "origin_y_px")) or
                grid["tile_width_px"] != grid["spacing_x_px"] or grid["tile_height_px"] != grid["spacing_y_px"]):
            raise ExperimentError(f"{label}: unsupported partial-edge grid policies/geometry")
    seen = set()
    seen_grid_positions = set()
    try:
        with _csv(path) as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            if len(set(header)) != len(header) or any(key not in header for key in [*keys, *columns, *clusters]):
                raise ExperimentError(f"{label}: CSV has missing or duplicate declared columns")
            for line, row in enumerate(reader, 2):
                identifier = row[keys[0]]
                if not identifier or not identifier.strip() or identifier in seen:
                    raise ExperimentError(f"{label}: missing or duplicate observation ID at CSV line {line}")
                seen.add(identifier)
                if None in row or any(row[key] is None for key in [*keys, *columns, *clusters]):
                    raise ExperimentError(f"{label}: malformed CSV row at line {line}")
                values = [float(row[key]) for key in [keys[1], keys[2], *columns]]
                if not all(math.isfinite(value) for value in values):
                    raise ExperimentError(f"{label}: nonfinite coordinates/features at CSV line {line}")
                if not (0 <= values[0] < image["width"] and 0 <= values[1] < image["height"]):
                    raise ExperimentError(f"{label}: observation centre outside image at CSV line {line}")
                if declared_edge_policy is not None:
                    position = []
                    for value, axis, length in ((values[0], "x", image["width"]), (values[1], "y", image["height"])):
                        origin, spacing = grid[f"origin_{axis}_px"], grid[f"spacing_{axis}_px"]
                        size = grid["tile_width_px" if axis == "x" else "tile_height_px"]
                        index = int((length - 1 - origin) // spacing) if value == length - 1 else round((value - origin - size // 2) / spacing)
                        start = origin + index * spacing
                        expected = min(start + size // 2, length - 1)
                        if index < 0 or start < 0 or start >= length or abs(value - expected) > 1e-6:
                            raise ExperimentError(f"{label}: observation contradicts logical clipped grid representative at CSV line {line}")
                        position.append(index)
                    if tuple(position) in seen_grid_positions:
                        raise ExperimentError(f"{label}: duplicate logical grid position at CSV line {line}")
                    seen_grid_positions.add(tuple(position))
                observed_unit = row.get("observation_type")
                if observed_unit and observed_unit not in ({"grid", "tile", "spatial_grid"} if unit == "tile" else {"token"}):
                    raise ExperimentError(f"{label}: observation_type contradicts {unit} at CSV line {line}")
    except (OSError, EOFError, csv.Error, ValueError) as exc:
        if isinstance(exc, ExperimentError):
            raise
        raise ExperimentError(f"{label}: cannot read cached features: {exc}") from exc
    if not seen:
        raise ExperimentError(f"{label}: feature table has no observations")
    if feature.get("row_count") is not None and (type(feature["row_count"]) is not int or feature["row_count"] != len(seen)):
        raise ExperimentError(f"{label}.row_count disagrees with the feature table")
    return len(seen)


def _validate_cell_features(feature, path, image, label, *, cells=None):
    """Validate typed cell observations, with an explicit dynamic predictor allowlist."""
    if feature.get("observation_unit") != "cell" or feature.get("coordinate_frame") != "image":
        raise ExperimentError(f"{label}: cellvit-cell requires cell observations in the image frame")
    descriptor = _mapping(feature.get("descriptor"), f"{label}.descriptor")
    width = descriptor.get("feature_count")
    if type(width) is not int or not 0 < width <= 4096:
        raise ExperimentError(f"{label}.descriptor.feature_count must be an explicit positive bounded integer")
    for key in ("backbone", "runtime_version", "feature_definition"):
        _text(descriptor.get(key), f"{label}.descriptor.{key}")
    morphology = feature.get("kind") == "cell-morphology"
    if morphology:
        if (descriptor.get("detector") != "HoVer-NeXt" or
                descriptor["backbone"] not in {"HoVer-NeXt-ConvNeXtV2-Tiny", "HoVer-NeXt-ConvNeXtV2-Base", "HoVer-NeXt-ConvNeXtV2-Large"} or
                descriptor["feature_definition"] != "hovernext-native-instance-morphology-rgb-v1"):
            raise ExperimentError(f"{label}: unsupported morphology detector/feature definition")
        revision = descriptor.get("upstream_source_commit")
        if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision) or descriptor["runtime_version"] != revision:
            raise ExperimentError(f"{label}: morphology runtime must bind its immutable upstream source")
        if not re.fullmatch(r"[a-f0-9]{32}", str(descriptor.get("checkpoint_archive_md5", ""))):
            raise ExperimentError(f"{label}: morphology checkpoint archive identity is missing")
    elif descriptor["backbone"] not in {"HIPT", "SAM", "Virchow2"}:
        raise ExperimentError(f"{label}: unsupported cell feature backbone; declare HIPT, SAM or Virchow2 explicitly")
    _digest(descriptor.get("checkpoint_sha256"), f"{label}.descriptor.checkpoint_sha256")
    if feature.get("encoder") is not None:
        encoder = _mapping(feature["encoder"], f"{label}.encoder")
        if (encoder.get("name") != ("HoVer-NeXt" if morphology else "CellViT++") or encoder.get("revision") != descriptor["runtime_version"] or
                _digest(encoder.get("checkpoint_sha256"), f"{label}.encoder.checkpoint_sha256") != descriptor["checkpoint_sha256"].lower() or
                ("backbone" in encoder and encoder["backbone"] != descriptor["backbone"])):
            raise ExperimentError(f"{label}: cell encoder and descriptor disagree")
    if descriptor.get("coordinate_frame") != "image" or descriptor.get("observation_unit") != "cell":
        raise ExperimentError(f"{label}: descriptor must explicitly describe cell features in image coordinates")
    for key in ("sampling", "preprocessing"):
        if not _mapping(descriptor.get(key), f"{label}.descriptor.{key}"):
            raise ExperimentError(f"{label}.descriptor.{key} must describe actual feature production")
    preprocessing = descriptor["preprocessing"]
    if not morphology:
        _digest(preprocessing.get("runtime_wheel_sha256"), f"{label}.descriptor.preprocessing.runtime_wheel_sha256")
    for key in (("mask_geometry", "colour") if morphology else ("normalization", "resampling", "token_provenance")):
        _text(preprocessing.get(key), f"{label}.descriptor.preprocessing.{key}")
    if feature.get("preprocessing") is not None:
        declared = _mapping(feature["preprocessing"], f"{label}.preprocessing").get("feature_definition")
        if declared is not None and declared != descriptor["feature_definition"]:
            raise ExperimentError(f"{label}: preprocessing and descriptor feature_definition disagree")
    sampling = descriptor["sampling"]
    _number(sampling.get("model_input_mpp"), f"{label}.model_input_mpp", positive=True)
    for key in (() if morphology else ("patch_size_px", "token_size_px")):
        if type(sampling.get(key)) is not int or sampling[key] <= 0:
            raise ExperimentError(f"{label}.sampling.{key} must be a positive integer")
    if morphology and sampling.get("resampling") != "none":
        raise ExperimentError(f"{label}: morphology import requires exact native sampling without resampling")
    if not morphology and (type(sampling.get("patch_overlap_px")) is not int or not 0 <= sampling["patch_overlap_px"] < sampling["patch_size_px"]):
        raise ExperimentError(f"{label}: invalid model patch overlap")
    try:
        json.dumps(descriptor, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExperimentError(f"{label}: descriptor is not finite JSON metadata") from exc
    columns = feature.get("feature_columns")
    if morphology and columns != ["area_um2", "perimeter_um", "bbox_width_um", "bbox_height_um", "major_axis_um", "minor_axis_um", "eccentricity", "extent", "mean_red", "mean_green", "mean_blue"]:
        raise ExperimentError(f"{label}: morphology predictor order differs from its versioned definition")
    if (not isinstance(columns, list) or len(columns) != width or
            any(not isinstance(column, str) or not column.strip() for column in columns) or
            len(set(columns)) != width or NON_PREDICTORS.intersection(columns)):
        raise ExperimentError(f"{label}: explicit predictor columns must match descriptor.feature_count and exclude metadata")
    keys = [_text(feature.get(key), f"{label}.{key}") for key in ("id_column", "x_column", "y_column")]
    if len(set(keys)) != 3 or set(keys).intersection(columns) or "source_id" in columns:
        raise ExperimentError(f"{label}: identity/coordinate columns cannot be predictors")
    if feature.get("format") == "partitioned-sqlite":
        if cells is not None or feature.get("kind") != "cellvit-cell":
            raise ExperimentError(f"{label}: partitioned cell features require an indexed CellViT source")
        pointer = _read_json(path)
        from index_cellvit_partitions import POINTER_SCHEMA, LEGACY_POINTER_SCHEMA
        storage = pointer.get("feature_storage")
        valid_pointer = ((pointer.get("schema") == POINTER_SCHEMA and
                          storage == "partition-row-reference-v1") or
                         (pointer.get("schema") == LEGACY_POINTER_SCHEMA and
                          storage is None))
        if (not valid_pointer or pointer.get("image") != {k: image[k] for k in pointer.get("image", {})} or
                pointer.get("feature_set_id") != feature.get("feature_set_id") or
                pointer.get("feature_columns") != columns or pointer.get("descriptor") != descriptor or
                pointer.get("row_count") != feature.get("row_count")):
            raise ExperimentError(f"{label}: partition pointer differs from indexed cell feature/frame contract")
        return feature["row_count"]
    expected_format = "csv.gz" if path.name.lower().endswith(".csv.gz") else "csv" if path.suffix.lower() == ".csv" else None
    if expected_format is None or feature.get("format") != expected_format:
        raise ExperimentError(f"{label}.format must match an existing .csv or .csv.gz feature table")
    if feature.get("frame_id") != image["frame_id"]:
        raise ExperimentError(f"{label}: feature frame_id differs from the image frame")
    expected_cells = {cell["cell_id"]: cell for cell in cells} if cells is not None else None
    seen = set()
    try:
        with _csv(path) as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            if len(header) != len(set(header)) or any(key not in header for key in [*keys, "source_id", *columns]):
                raise ExperimentError(f"{label}: CSV has missing or duplicate declared columns")
            for line, row in enumerate(reader, 2):
                identifier = row[keys[0]]
                if not identifier or not identifier.strip() or identifier in seen:
                    raise ExperimentError(f"{label}: missing or duplicate cell ID at CSV line {line}")
                if None in row or any(row[key] is None for key in [*keys, "source_id", *columns]):
                    raise ExperimentError(f"{label}: malformed cell feature row at CSV line {line}")
                seen.add(identifier)
                x, y, *values = [float(row[key]) for key in [keys[1], keys[2], *columns]]
                if not all(math.isfinite(value) for value in [x, y, *values]) or not (0 <= x < image["width"] and 0 <= y < image["height"]):
                    raise ExperimentError(f"{label}: invalid cell coordinates/features at CSV line {line}")
                if not row["source_id"]:
                    raise ExperimentError(f"{label}: missing immutable source_id at CSV line {line}")
                if expected_cells is not None:
                    cell = expected_cells.get(identifier)
                    if cell is None or cell["source_id"] != row["source_id"] or (cell["x"], cell["y"]) != (x, y):
                        raise ExperimentError(f"{label}: classification/feature cell ID or coordinate join mismatch")
    except (OSError, EOFError, csv.Error, ValueError) as exc:
        if isinstance(exc, ExperimentError):
            raise
        raise ExperimentError(f"{label}: cannot read cell features: {exc}") from exc
    if type(feature.get("row_count")) is not int or feature["row_count"] != len(seen):
        raise ExperimentError(f"{label}.row_count disagrees with the cell table")
    if expected_cells is not None and seen != set(expected_cells):
        raise ExperimentError(f"{label}: classification and feature cells are not an exact ID join")
    return len(seen)


def _cell_frame_identity(image, label):
    identity = {key: image.get(key) for key in ("sha256", "width", "height", "mpp_x", "mpp_y")}
    _digest(identity["sha256"], f"{label}.sha256")
    for key in ("width", "height"):
        if type(identity[key]) is not int or identity[key] <= 0:
            raise ExperimentError(f"{label}.{key} must be a positive integer")
    for key in ("mpp_x", "mpp_y"):
        _number(identity[key], f"{label}.{key}", positive=True)
    if image.get("coordinate_convention") != COORDINATE_CONVENTION:
        raise ExperimentError(f"{label}: explicit native pixel coordinate convention is required")
    if any(key not in image for key in ("coordinate_space", "origin_x", "origin_y")):
        raise ExperimentError(f"{label}: explicit coordinate_space and native origin are required")
    identity.update(coordinate_convention=COORDINATE_CONVENTION,
                    coordinate_space=image["coordinate_space"], origin_x=image["origin_x"], origin_y=image["origin_y"])
    if identity["coordinate_space"] != "crop-native-pixels" or identity["origin_x"] != 0 or identity["origin_y"] != 0:
        raise ExperimentError(f"{label}: only the explicit source-image native origin is supported")
    for key in ("origin_x", "origin_y"):
        _number(identity[key], f"{label}.{key}")
    return identity


def _classification_registration(source_image, target_image):
    source = _cell_frame_identity(source_image, "classification.image")
    target = _cell_frame_identity(target_image, "tissue.image")
    if source != target:
        raise ExperimentError("Cell classification image hash, dimensions, MPP or native coordinate geometry differs from the tissue")
    for image in (source_image, target_image):
        _identifier(image.get("frame_id"), "cell classification frame_id")
    return dict(source_frame_id=source_image["frame_id"], target_frame_id=target_image["frame_id"],
                source_to_image_affine=copy.deepcopy(IDENTITY_AFFINE), identity_verified=source)


def _validate_classification_payload(payload, classification_path, tissue):
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExperimentError("Cell classification contains nonfinite or unsupported JSON metadata") from exc
    if payload.get("schema") != CELL_CLASSIFICATION_SCHEMA or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ExperimentError("Unsupported cell classification schema/version")
    registration = _classification_registration(_mapping(payload.get("image"), "classification.image"), tissue["image"])
    feature = _mapping(payload.get("feature_set"), "classification.feature_set")
    if feature.get("kind") not in CELL_FEATURE_KINDS or feature.get("frame_id") != payload["image"]["frame_id"]:
        raise ExperimentError("Classification nested feature kind/frame differs from its source image")
    feature_path = _verify_asset(feature, classification_path.parent, "classification.feature_set")
    models, model_ids = {}, set()
    for model in _array(payload.get("models"), "classification.models"):
        _mapping(model, "classification.model")
        mid = _identifier(model.get("model_id"), "classification.model_id")
        if mid in model_ids:
            raise ExperimentError("Duplicate classification model_id")
        model_ids.add(mid)
        _text(model.get("taxonomy"), "classification.model.taxonomy")
        if model.get("backbone") != feature.get("descriptor", {}).get("backbone"):
            raise ExperimentError("Classification model and cached feature backbones differ")
        if model.get("status") not in {"available", "unavailable"}:
            raise ExperimentError("Classification model status must be available or unavailable")
        classes = _array(model.get("class_names"), "classification.model.class_names")
        if any(not isinstance(name, str) or not name for name in classes) or len(set(classes)) != len(classes):
            raise ExperimentError("Classification class_names must be unique ordered strings")
        if model["status"] == "available":
            if not classes:
                raise ExperimentError("Available classification model requires an ordered ontology")
            _digest(model.get("checkpoint_sha256"), "classification.model.checkpoint_sha256")
            semantics = {"mlp": "uncalibrated_model_probability", "native_segmentation_head": "uncalibrated_pixel_type_vote_fraction", "native_instance_head": "uncalibrated_softmax_cell_class_vote_counts"}
            if model.get("kind") not in semantics or model.get("score_semantics") != semantics[model["kind"]]:
                raise ExperimentError("Classification model kind and score semantics are incompatible")
            if model["kind"] == "native_segmentation_head" and (model["taxonomy"] != "pannuke" or
                    model["checkpoint_sha256"] != feature["descriptor"].get("checkpoint_sha256")):
                raise ExperimentError("Native PanNuke head must match the segmentation checkpoint and taxonomy")
            if feature["kind"] == "cell-morphology":
                if model["kind"] != "native_instance_head" or model["taxonomy"] not in {"pannuke", "lizard"} or model["checkpoint_sha256"] != feature["descriptor"]["checkpoint_sha256"]:
                    raise ExperimentError("Morphology observations require their own native detector evidence; CellViT heads cannot consume them")
            elif model["kind"] == "native_instance_head":
                raise ExperimentError("Native HoVer-NeXt evidence requires its separate morphology feature contract")
        else:
            _text(model.get("reason"), "classification.model.reason")
        models[mid] = model
    if not models:
        raise ExperimentError("Cell classification requires at least one explicit model availability record")
    cells = _array(payload.get("cells"), "classification.cells")
    seen, source_ids = set(), set()
    for cell in cells:
        _mapping(cell, "classification.cell")
        cid = _text(cell.get("cell_id"), "classification.cell_id")
        sid = _text(cell.get("source_id"), "classification.source_id")
        if cid in seen or sid in source_ids:
            raise ExperimentError("Duplicate canonical or source cell ID")
        seen.add(cid)
        source_ids.add(sid)
        for axis, length in (("x", tissue["image"]["width"]), ("y", tissue["image"]["height"])):
            if not 0 <= _number(cell.get(axis), f"classification.cell.{axis}") < length:
                raise ExperimentError("Classification cell lies outside the source image")
        predictions = _mapping(cell.get("predictions"), "classification.cell.predictions")
        if set(predictions) != model_ids:
            raise ExperimentError("Classification cell predictions must explicitly cover every declared model")
        for mid, prediction in predictions.items():
            _mapping(prediction, "classification.prediction")
            model, status = models[mid], prediction.get("status")
            if status not in {"predicted", "unavailable", "abstained"}:
                raise ExperimentError("Unknown cell prediction status")
            if status != "predicted":
                if any(prediction.get(key) is not None for key in ("class_id", "label", "probability", "probabilities")):
                    raise ExperimentError("Unavailable/abstained prediction cannot contain a biological assignment")
                _text(prediction.get("reason"), "classification.prediction.reason")
                continue
            class_id = prediction.get("class_id")
            names = model["class_names"]
            if model["status"] != "available" or type(class_id) is not int or not 0 <= class_id < len(names) or prediction.get("label") != names[class_id]:
                raise ExperimentError("Predicted class_id/label does not match its available model ontology")
            if (model.get("kind") == "native_instance_head" and prediction.get("probabilities") is None and
                    prediction.get("probability") is None and prediction.get("probability_vector_status") == "not_exported_by_upstream"):
                continue
            score = _number(prediction.get("probability"), "classification.probability")
            if not 0 <= score <= 1:
                raise ExperimentError("Classification probability must be in [0,1]")
            vector = prediction.get("probabilities")
            if vector is None:
                if (model.get("kind") != "native_segmentation_head" or model.get("score_semantics") != "uncalibrated_pixel_type_vote_fraction" or
                        prediction.get("probability_vector_status") != "not_exported_by_upstream"):
                    raise ExperimentError("Missing probability vector requires the explicit native PanNuke vote-score contract")
            else:
                if (not isinstance(vector, list) or len(vector) != len(names) or
                        any(not 0 <= _number(value, "classification.probabilities") <= 1 for value in vector) or
                        abs(sum(vector) - 1) > 1e-5 or abs(vector[class_id] - score) > 1e-7):
                    raise ExperimentError("Classification probability vector disagrees with its ontology or selected score")
    _validate_cell_features(feature, feature_path, payload["image"], "classification.feature_set", cells=cells)
    return registration


def _validate_cell_feature_manifest(path, classification):
    if classification.get("provenance", {}).get("feature_manifest_sha256") != sha256(path):
        raise ExperimentError("Classification provenance does not bind the exact feature manifest SHA-256")
    manifest = _read_json(path)
    if manifest.get("schema") != "cellphenotyper-cell-features" or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ExperimentError("Unsupported cell feature manifest schema/version")
    if manifest.get("image") != classification["image"]:
        raise ExperimentError("Cell feature manifest image/frame differs from classification")
    feature = _mapping(manifest.get("feature_set"), "cell feature manifest.feature_set")
    _verify_asset(feature, path.parent, "cell feature manifest.feature_set")
    without_path = lambda value: {key: child for key, child in value.items() if key != "path"}
    if without_path(feature) != without_path(classification["feature_set"]):
        raise ExperimentError("Cell feature manifest signature/metadata differs from classification")
    metadata = _mapping(manifest.get("cell_metadata"), "cell feature manifest.cell_metadata")
    cell_metadata_path = _verify_asset(metadata, path.parent, "cell feature manifest.cell_metadata")
    cells = _read_json(cell_metadata_path)
    if cells.get("image") != classification["image"]:
        raise ExperimentError("Cell feature metadata image/frame differs from classification")
    original = _array(cells.get("cells"), "cell feature metadata.cells")
    predicted = {cell["cell_id"]: cell for cell in classification["cells"]}
    seen = set()
    for cell in original:
        _mapping(cell, "cell feature metadata.cell")
        cid = _text(cell.get("cell_id"), "cell feature metadata.cell_id")
        if cid in seen or cid not in predicted or any(predicted[cid].get(key) != value for key, value in cell.items()):
            raise ExperimentError("Cell feature metadata identity/geometry differs from classification")
        seen.add(cid)
    if seen != set(predicted):
        raise ExperimentError("Cell feature metadata and classification are not an exact cell ID join")
    normalized = manifest.get("source", {}).get("normalized_segmentation")
    if normalized is not None:
        normalized_path = _verify_asset(_mapping(normalized, "normalized segmentation reference"), path.parent, "normalized segmentation")
        if classification.get("provenance", {}).get("normalized_segmentation_sha256") != normalized["sha256"]:
            raise ExperimentError("Classification provenance differs from the normalized segmentation hash")
        normalized_payload = _read_json(normalized_path)
        metadata = _mapping(normalized_payload.get("metadata"), "normalized segmentation.metadata")
        image = classification["image"]
        if metadata.get("source_image_sha256") != image["sha256"] or metadata.get("sha256") != image["sha256"] or any(metadata.get(key) != image[key] for key in
                ("width", "height", "mpp_x", "mpp_y", "coordinate_space", "coordinate_convention", "origin_x", "origin_y", "frame_id")):
            raise ExperimentError("Normalized segmentation native image/frame differs from classification")
        normalized_cells = _array(normalized_payload.get("cells"), "normalized segmentation.cells")
        normalized_ids = set()
        for cell in normalized_cells:
            _mapping(cell, "normalized segmentation.cell")
            cid = _text(cell.get("cell_id"), "normalized segmentation.cell_id")
            expected = predicted.get(cid)
            if (cid in normalized_ids or expected is None or cell.get("source_id") != expected["source_id"] or
                    cell.get("centroid") != [expected["x"], expected["y"]] or
                    any(cell.get(key) != expected[key] for key in ("contour", "bbox") if key in expected)):
                raise ExperimentError("Normalized segmentation is not an exact immutable cell identity/geometry join")
            normalized_ids.add(cid)
        if normalized_ids != set(predicted):
            raise ExperimentError("Normalized segmentation has missing or extra cells")
    return manifest


def _validate_segmentation_review(reference, path, classification, manifest):
    """Share the exact row contract with the streaming indexed-QC derivative."""
    from segmentation_review_validation import validate_payload
    try:
        validate_payload(reference, _read_json(path), classification, manifest)
    except (ValueError, TypeError) as exc:
        raise ExperimentError(str(exc)) from exc


def _prepare_indexed_cell_classification(tissue, root, destination=None):
    """Validate the immutable streaming-produced index without loading its population."""
    reference = tissue["cell_classification"]
    if destination is None:
        path = _verify_asset(reference, root, "tissue.cell_classification")
        item = copy.deepcopy(reference)
    else:
        item, path = _asset(reference, root, destination, "tissue.cell_classification")
    for key in ("index", "feature_manifest"):
        if key not in item:
            raise ExperimentError("Indexed cell review requires an explicit index and feature manifest")
        if destination is None:
            resolved = _verify_asset(item[key], root, f"cell_classification.{key}")
        else:
            item[key], resolved = _asset(item[key], root, destination, f"cell_classification.{key}")
        if key == "index":
            index_path = resolved
        else:
            manifest_path = resolved
    if item["index"].get("format") != "sqlite":
        raise ExperimentError("Cell index must declare format=sqlite")
    if "tissue_context" in item:
        raise ExperimentError("Indexed cell review requires indexed tissue-context evidence; raw whole-population context cannot be silently loaded or ignored")
    review_path = None
    if "segmentation_review" in item:
        if destination is None:
            review_path = _verify_asset(item["segmentation_review"], root, "cell_classification.segmentation_review")
        else:
            item["segmentation_review"], review_path = _asset(item["segmentation_review"], root, destination, "cell_classification.segmentation_review")
    try:
        with sqlite3.connect(f"file:{index_path}?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        indexed_version = json.loads(row[0]) if row else None
        if indexed_version in (2, 3):
            if review_path is not None or path != manifest_path:
                raise ValueError("Partitioned CellViT indexes require one manifest and no legacy review sidecar")
            from index_cellvit_partitions import validate_partition_index
            pointer_path = index_path.parent / index_path.with_suffix(".features.json").name
            image_root = Path(destination).parent if destination is not None else root
            image_path = _path(tissue["image"]["path"], image_root, "partition source image")
            metadata = validate_partition_index(index_path, path, pointer_path, image_path)
        else:
            from index_cell_classification import validate_index
            metadata = validate_index(index_path, path, manifest_path, segmentation_review_path=review_path)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        raise ExperimentError(f"Cannot validate cell index: {exc}") from exc
    indexed_reviews = [sidecar for sidecar in metadata.get("sidecars", []) if sidecar.get("role") == "segmentation_review"]
    if bool(indexed_reviews) != (review_path is not None) or len(indexed_reviews) > 1:
        raise ExperimentError("Indexed segmentation review must be registered exactly once with its checked source")
    if indexed_reviews:
        sidecar = indexed_reviews[0]
        reference = item["segmentation_review"]
        expected = dict(source_frame_id=metadata["image"]["frame_id"], coordinate_convention=COORDINATE_CONVENTION)
        if destination is not None:
            for key, value in expected.items():
                reference.setdefault(key, value)
        if reference.get("sha256") != sidecar.get("sha256") or any(reference.get(key) != value or sidecar.get(key) != value for key, value in expected.items()):
            raise ExperimentError("Indexed segmentation review hash, frame or convention differs from its original source")
    registration = _classification_registration(metadata["image"], tissue["image"])
    if "registration" in item and item["registration"] != registration:
        raise ExperimentError("Indexed cell source registration differs from the checked native image")
    if destination is None and "registration" not in item:
        raise ExperimentError("Indexed classification requires an explicit verified frame registration")
    item["registration"] = registration
    count = metadata["cell_count"]
    for key, value in (("cell_count", count), ("review_status", "available" if count else "empty")):
        if key in item and item[key] != value:
            raise ExperimentError(f"Indexed classification {key} differs from its source population")
        item[key] = value
    feature = copy.deepcopy(metadata["feature_set"])
    feature_source = metadata["feature_store"] if indexed_version in (2, 3) else metadata["feature_csv"]
    feature["path"] = str(_path(feature_source["path"], index_path.parent, "indexed cell feature store"))
    feature["source_frame_id"] = metadata["image"]["frame_id"]
    feature["frame_id"] = tissue["image"]["frame_id"]
    feature["registration"] = registration
    return item, feature


def _prepare_cell_classification(tissue, root, destination=None):
    """Nested paths are relative to the classification JSON, never the project index."""
    reference = tissue.get("cell_classification")
    if reference is None:
        return None
    if isinstance(reference, dict) and "index" in reference:
        return _prepare_indexed_cell_classification(tissue, root, destination)
    if destination is None:
        path = _verify_asset(reference, root, "tissue.cell_classification")
        item = copy.deepcopy(reference)
    else:
        item, path = _asset(reference, root, destination, "tissue.cell_classification")
    payload = _read_json(path)
    registration = _validate_classification_payload(payload, path, tissue)
    manifest = None
    if "feature_manifest" in item:
        if destination is None:
            manifest_path = _verify_asset(item["feature_manifest"], root, "cell_classification.feature_manifest")
        else:
            item["feature_manifest"], manifest_path = _asset(item["feature_manifest"], root, destination, "cell_classification.feature_manifest")
        manifest = _validate_cell_feature_manifest(manifest_path, payload)
    if "segmentation_review" in item:
        if manifest is None:
            raise ExperimentError("Segmentation review requires a checked feature manifest and source hash lineage")
        if destination is None:
            review_path = _verify_asset(item["segmentation_review"], root, "cell_classification.segmentation_review")
        else:
            item["segmentation_review"], review_path = _asset(item["segmentation_review"], root, destination, "cell_classification.segmentation_review")
            item["segmentation_review"].setdefault("source_frame_id", payload["image"]["frame_id"])
            item["segmentation_review"].setdefault("coordinate_convention", COORDINATE_CONVENTION)
        _validate_segmentation_review(item["segmentation_review"], review_path, payload, manifest)
    if "tissue_context" in item:
        if manifest is None:
            raise ExperimentError("Tissue context requires a checked feature manifest and immutable cell lineage")
        if destination is None:
            context_path = _verify_asset(item["tissue_context"], root, "cell_classification.tissue_context")
        else:
            item["tissue_context"], context_path = _asset(item["tissue_context"], root, destination, "cell_classification.tissue_context")
        try:
            from cell_tissue_context import validate_context_sidecar
            validate_context_sidecar(context_path, path, manifest_path)
        except (ImportError, ValueError, OSError) as exc:
            raise ExperimentError(f"Cannot verify cell tissue context: {exc}") from exc
    if "registration" in item and item["registration"] != registration:
        raise ExperimentError("Cell classification registration mapping does not match independently verified native image identities")
    if destination is None and "registration" not in item:
        raise ExperimentError("Cell classification requires an explicit verified frame registration")
    item["registration"] = registration
    for key, value in (("cell_count", len(payload["cells"])), ("review_status", "empty" if not payload["cells"] else "available")):
        if key in item and item[key] != value:
            raise ExperimentError(f"Cell classification {key} disagrees with the actual cell review")
        item[key] = value
    feature = copy.deepcopy(payload["feature_set"])
    feature["path"] = str(_path(feature["path"], path.parent, "classification.feature_set.path"))
    feature["source_frame_id"] = payload["image"]["frame_id"]
    feature["frame_id"] = tissue["image"]["frame_id"]
    feature["registration"] = registration
    return item, feature


def _register_definition(definitions, feature, image):
    """A shared feature ID denotes the same representation across tissues."""
    if feature["kind"] in CELL_FEATURE_KINDS:
        signature = json.dumps(dict(kind=feature["kind"], observation_unit=feature["observation_unit"],
                                    feature_columns=feature["feature_columns"], descriptor=feature["descriptor"]),
                               sort_keys=True, allow_nan=False)
        fid = feature["feature_set_id"]
        if fid in definitions and definitions[fid] != signature:
            raise ExperimentError(f"Feature set {fid} has incompatible definitions across tissues")
        definitions[fid] = signature
        return
    grid = feature["grid"]
    physical_grid = [round(grid[key] * image[mpp], 6) for key, mpp in
                     (("tile_width_px", "mpp_x"), ("tile_height_px", "mpp_y"),
                      ("spacing_x_px", "mpp_x"), ("spacing_y_px", "mpp_y"))]
    signature = json.dumps(dict(kind=feature["kind"], observation_unit=feature["observation_unit"],
                                feature_columns=feature["feature_columns"], encoder=feature["encoder"],
                                preprocessing=feature["preprocessing"], physical_grid=physical_grid,
                                representative_policy=grid.get("representative_policy"), edge_policy=grid.get("edge_policy")),
                           sort_keys=True, allow_nan=False)
    fid = feature["feature_set_id"]
    if fid in definitions and definitions[fid] != signature:
        raise ExperimentError(f"Feature set {fid} has incompatible definitions across tissues; assign a distinct feature_set_id to another feature space")
    definitions[fid] = signature


def _namespace_id(experiment_id, tissue_id, dataset_id, original_id):
    """Match wsiTools' existing R URLencode(reserved=TRUE) namespace exactly."""
    # R's default repeated=FALSE leaves a component containing any %HH intact.
    def encode(value):
        return value if re.search(r"%[a-fA-F0-9]{2}", value) else quote(value, safe="~._-")
    return "::".join(encode(value) for value in (experiment_id, tissue_id, dataset_id, original_id))


def _spatial_fields(record, required, optional, label):
    record = _mapping(record, label)
    unknown = set(record) - set(required) - set(optional)
    missing = set(required) - set(record)
    if unknown or missing:
        raise ExperimentError(f"{label} has unknown or missing contract fields: {', '.join(sorted(unknown | missing))}")
    return record


def _no_bundle_paths(value, label="spatial review bundle"):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in {"path", "paths"} or key.lower().endswith(("_path", "_paths")):
                raise ExperimentError(f"{label} must reference assets by artifact ID, not internal paths")
            _no_bundle_paths(child, label)
    elif isinstance(value, list):
        for child in value:
            _no_bundle_paths(child, label)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ExperimentError(f"{label} contains a nonfinite JSON value")


def _read_spatial_bundle(path, expected_sha256):
    """Read bounded metadata only; the original RDS is never deserialized."""
    def constant(value):
        raise ExperimentError(f"Spatial review bundle contains nonfinite JSON value: {value}")

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ExperimentError(f"Spatial review bundle contains duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        if path.stat().st_size > SPATIAL_MAX_BYTES:
            raise ExperimentError("Spatial review bundle exceeds the 16 MiB metadata limit")
        with path.open("rb") as stream:
            content = stream.read(SPATIAL_MAX_BYTES + 1)
        if len(content) > SPATIAL_MAX_BYTES:
            raise ExperimentError("Spatial review bundle exceeds the 16 MiB metadata limit")
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ExperimentError("Spatial review bundle SHA-256 changed while reading")
        result = _mapping(json.loads(content.decode("utf-8"), parse_constant=constant,
                                     object_pairs_hook=object_pairs), "spatial review bundle")
        _no_bundle_paths(result)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, ExperimentError):
            raise
        raise ExperimentError(f"Cannot read spatial review bundle {path}: {exc}") from exc
    return result


def _spatial_mapping(mapping, image):
    mapping = _spatial_fields(mapping, ("confirmed", "same_section", "image_sha256", "frame_id",
                                       "coordinate_space", "affine"), (), "spatial coordinate_mapping")
    if mapping.get("confirmed") is not True or mapping.get("same_section") is not True:
        raise ExperimentError("Spatial coordinate_mapping requires confirmed=true and same_section=true")
    if (mapping.get("frame_id") != image["frame_id"] or
            _digest(mapping.get("image_sha256"), "mapping.image_sha256") != _digest(image["sha256"], "image.sha256")):
        raise ExperimentError("Spatial coordinate_mapping image hash/frame differs from its tissue")
    space = mapping.get("coordinate_space")
    if not isinstance(space, str) or space not in {"native_pixels", "level0_slide_pixels", "source_pixels"}:
        raise ExperimentError("Spatial coordinate_mapping has unknown coordinate_space; no mapping is inferred")
    affine = _array(mapping.get("affine"), "coordinate_mapping.affine")
    if len(affine) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in affine):
        raise ExperimentError("Spatial coordinate_mapping requires a finite invertible 3x3 affine")
    for row in affine:
        for value in row:
            _number(value, "coordinate_mapping.affine value")
    if any(abs(value - expected) > 1e-10 for value, expected in zip(affine[2], (0, 0, 1))):
        raise ExperimentError("Spatial affine final row must be [0,0,1]")
    determinant = affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0]
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ExperimentError("Spatial coordinate_mapping affine must be invertible")
    if space != "source_pixels" and any(abs(affine[i][j] - int(i == j)) > 1e-10
                                        for i in range(3) for j in range(3)):
        raise ExperimentError("A nonidentity spatial affine must declare coordinate_space=source_pixels")


def _spatial_footprint(row):
    footprint = row.get("footprint")
    width, height = row.get("footprint_width_px"), row.get("footprint_height_px")
    if footprint is not None:
        footprint = _mapping(footprint, "observation.footprint")
        x0, y0, x1, y1 = [_number(footprint.get(key), f"footprint.{key}") for key in ("x0", "y0", "x1", "y1")]
        if (not math.isfinite(x1 - x0) or not math.isfinite(y1 - y0) or x1 <= x0 or y1 <= y0
                or not (x0 <= row["x"] <= x1 and y0 <= row["y"] <= y1)):
            raise ExperimentError("Spatial footprint must have positive area and contain its observation centre")
        for value, expected in ((width, x1 - x0), (height, y1 - y0)):
            if value is not None and abs(_number(value, "footprint dimension", positive=True) - expected) > 1e-7:
                raise ExperimentError("Spatial footprint dimensions disagree with its edge rectangle")
    elif width is not None or height is not None:
        _number(width, "footprint_width_px", positive=True)
        _number(height, "footprint_height_px", positive=True)


def _validate_spatial_bundle(bundle, source, artifact, tissue, experiment_id):
    image = tissue["image"]
    dataset = artifact["dataset_id"]
    _spatial_fields(bundle, ("schema", "schema_version", "experiment_id", "tissue_id", "dataset_id",
                            "observation_unit", "frame_id", "image_sha256", "source_artifact_id",
                            "source_object_sha256", "coordinate_mapping", "observations", "reductions"), (),
                    "spatial review bundle")
    if (bundle.get("schema") != SPATIAL_BUNDLE_SCHEMA or type(bundle.get("schema_version")) is not int
            or bundle["schema_version"] != 1):
        raise ExperimentError(f"Expected {SPATIAL_BUNDLE_SCHEMA} schema_version=1")
    expected = dict(experiment_id=experiment_id, tissue_id=tissue["tissue_id"], dataset_id=dataset,
                    observation_unit=artifact["observation_unit"], frame_id=image["frame_id"],
                    source_artifact_id=source["artifact_id"])
    if any(bundle.get(key) != value for key, value in expected.items()):
        raise ExperimentError("Spatial review bundle experiment/tissue/dataset/unit/frame/source linkage differs from its artifacts")
    for key, expected_digest in (("image_sha256", image["sha256"]), ("source_object_sha256", source["sha256"])):
        if _digest(bundle.get(key), f"bundle.{key}") != _digest(expected_digest, key):
            raise ExperimentError(f"Spatial review bundle {key} differs from its linked asset")
    _spatial_mapping(bundle.get("coordinate_mapping"), image)
    observations = _array(bundle.get("observations"), "bundle.observations")
    if len(observations) != artifact["row_count"] or not 1 <= len(observations) <= SPATIAL_MAX_OBSERVATIONS:
        raise ExperimentError("Spatial review bundle row_count must match 1..10000 observations")
    ids, originals = set(), set()
    for row in observations:
        _spatial_fields(row, ("id", "original_id", "dataset_id", "tissue_id", "frame_id", "observation_unit",
                              "x", "y", "cluster_values"),
                        ("footprint", "footprint_width_px", "footprint_height_px"), "spatial observation")
        original = _text(row.get("original_id"), "observation.original_id")
        identifier = _text(row.get("id"), "observation.id")
        canonical = _namespace_id(experiment_id, tissue["tissue_id"], dataset, original)
        if identifier != canonical or identifier in ids or original in originals:
            raise ExperimentError("Spatial observation IDs must be unique canonical IDs with stable original_id values")
        ids.add(identifier)
        originals.add(original)
        for key in ("dataset_id", "tissue_id", "frame_id", "observation_unit"):
            if row.get(key) != expected[key]:
                raise ExperimentError(f"Spatial observation {key} differs from its declared dataset")
        x, y = _number(row.get("x"), "observation.x"), _number(row.get("y"), "observation.y")
        if not (0 <= x < image["width"] and 0 <= y < image["height"]):
            raise ExperimentError("Spatial observation centre lies outside its declared image frame")
        clusters = row.get("cluster_values")
        if clusters == []:  # jsonlite serializes an empty named R list as [].
            clusters = {}
        _mapping(clusters, "observation.cluster_values")
        for key, value in clusters.items():
            _text(key, "cluster metadata field")
            if value is not None and (not isinstance(value, (str, int, float, bool)) or
                                      isinstance(value, (int, float)) and not math.isfinite(value)):
                raise ExperimentError("Spatial cluster values must be scalar categorical metadata or null")
        _spatial_footprint(row)
    reductions = _array(bundle.get("reductions"), "bundle.reductions")
    reduction_ids = set()
    prefix = _namespace_id(experiment_id, tissue["tissue_id"], dataset, "")
    for reduction in reductions:
        _spatial_fields(reduction, ("id", "observation_unit", "dataset_id", "frame_id", "points"),
                        ("label",), "spatial reduction")
        identifier = _text(reduction.get("id"), "reduction.id")
        if (not identifier.startswith(prefix) or identifier == prefix or identifier in reduction_ids or
                _namespace_id(experiment_id, tissue["tissue_id"], dataset, identifier[len(prefix):]) != identifier):
            raise ExperimentError("Spatial reduction IDs must be unique canonical dataset IDs")
        reduction_ids.add(identifier)
        if reduction.get("label") is not None:
            _text(reduction["label"], "reduction.label")
        for key in ("dataset_id", "frame_id", "observation_unit"):
            if reduction.get(key) != expected[key]:
                raise ExperimentError(f"Spatial reduction {key} differs from its dataset")
        points = _array(reduction.get("points"), "reduction.points")
        point_ids = set()
        for point in points:
            _spatial_fields(point, ("id", "x", "y"), (), "spatial reduction point")
            point_id = _text(point.get("id"), "reduction point.id")
            if point_id not in ids or point_id in point_ids:
                raise ExperimentError("Spatial reduction points require unique known canonical IDs")
            point_ids.add(point_id)
            _number(point.get("x"), "reduction point.x")
            _number(point.get("y"), "reduction point.y")
        if point_ids != ids:
            raise ExperimentError("Spatial reduction IDs must exactly match dataset observations; row order is irrelevant")
    return ids


def _validate_spatial_artifacts(payload, root):
    all_ids = set()
    for tissue in payload["tissues"]:
        artifacts = tissue["artifacts"]
        spatial = [artifact for artifact in artifacts if artifact.get("role") in SPATIAL_ROLES]
        if not spatial:
            continue
        review_datasets = {artifact.get("dataset_id") for artifact in artifacts
                           if artifact.get("role") == "review_observations" and isinstance(artifact.get("dataset_id"), str)}
        datasets = {}
        for artifact in spatial:
            role = artifact["role"]
            dataset = _identifier(artifact.get("dataset_id"), "spatial artifact.dataset_id")
            kind = "object" if role == "spatial_transcriptomics_object" else "review"
            if artifact.get("artifact_id") != f"spatial:{dataset}:{kind}":
                raise ExperimentError("Spatial artifact_id must be spatial:<dataset>:object or spatial:<dataset>:review")
            if artifact.get("format") != SPATIAL_ROLES[role]:
                raise ExperimentError(f"Spatial {role} requires format={SPATIAL_ROLES[role]}")
            if Path(artifact["path"]).suffix.lower() != "." + SPATIAL_ROLES[role]:
                raise ExperimentError("Spatial artifact format must match its RDS/JSON filename")
            if not isinstance(artifact.get("observation_unit"), str) or artifact["observation_unit"] not in {"spot", "cell"}:
                raise ExperimentError("Spatial object observation_unit must be spot or cell, never tile/token")
            count = artifact.get("row_count")
            if type(count) is not int or not 1 <= count <= SPATIAL_MAX_OBSERVATIONS:
                raise ExperimentError("Spatial artifact row_count must be an integer in 1..10000")
            if (artifact.get("frame_id") != tissue["image"]["frame_id"] or
                    _digest(artifact.get("image_sha256"), "spatial image_sha256") != _digest(tissue["image"]["sha256"], "image.sha256")):
                raise ExperimentError("Spatial artifact image hash/frame differs from its tissue")
            pair = datasets.setdefault(dataset, {})
            if kind in pair or dataset in review_datasets:
                raise ExperimentError("Spatial dataset_id collides with another spatial object or review_observations dataset")
            pair[kind] = artifact
        for dataset, pair in datasets.items():
            if set(pair) != {"object", "review"}:
                raise ExperimentError(f"Spatial dataset {dataset} needs both object and review bundle artifacts")
            source, artifact = pair["object"], pair["review"]
            if (source.get("review_bundle_artifact_id") != artifact["artifact_id"] or
                    artifact.get("source_artifact_id") != source["artifact_id"]):
                raise ExperimentError("Spatial object/review artifacts require bidirectional artifact linkage")
            if (source["row_count"] != artifact["row_count"] or source["observation_unit"] != artifact["observation_unit"] or
                    _digest(artifact.get("source_object_sha256"), "source_object_sha256") != _digest(source["sha256"], "source.sha256")):
                raise ExperimentError("Spatial object/review row_count, unit or source hash differ")
            _verify_asset(source, root, "spatial source object")
            path = _verify_asset(artifact, root, "spatial review bundle")
            bundle = _read_spatial_bundle(path, _digest(artifact["sha256"], "bundle.sha256"))
            ids = _validate_spatial_bundle(bundle, source, artifact, tissue, payload["experiment_id"])
            if all_ids.intersection(ids):
                raise ExperimentError("Canonical spatial observation IDs collide across datasets")
            all_ids.update(ids)


def _gigatime_tiff(path, artifact, image):
    """Inspect TIFF headers only; no full-slide arrays or inference imports."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            if artifact["format"] == "ome-tiff" and not tf.is_ome:
                raise ExperimentError("A declared OME-TIFF artifact must contain OME metadata")
            if artifact["role"] == "cell_segmentation_mask":
                if len(tf.pages) != 1:
                    raise ExperimentError("A cell segmentation mask must be a single-plane integer TIFF")
                page = tf.pages[0]
                if len(page.shape) != 2 or page.dtype.kind not in {"u", "i"}:
                    raise ExperimentError("A cell segmentation mask must be a single-plane integer TIFF")
                height, width = page.shape
                if (width, height) != (image["width"], image["height"]):
                    raise ExperimentError("Cell segmentation mask dimensions must match the full native tissue image; no mask resize is inferred")
                bits = 8 * page.dtype.itemsize - int(page.dtype.kind == "i")
                return dict(width=int(width), height=int(height), max_label=(1 << bits) - 1)
            series = tf.series[0]
            if len(tf.series) != 1 or set(series.axes) - set("CYX"):
                raise ExperimentError("GigaTIME marker images require one OME series with only C/Y/X axes")
            root = ET.fromstring(tf.ome_metadata)
            pixels = next((element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "Pixels"), None)
            names = [element.attrib.get("Name") for element in pixels
                     if element.tag.rsplit("}", 1)[-1] == "Channel"] if pixels is not None else []
            channels = _array(artifact.get("channels"), "marker image.channels")
            if not channels or len(names) != len(tf.pages):
                raise ExperimentError("Marker image requires explicit named OME channels and zero-based TIFF pages")
            markers, indices, dimensions = set(), set(), None
            for channel in channels:
                _spatial_fields(channel, ("marker", "page"), ("display_max",), "marker image channel")
                marker = _text(channel["marker"], "channel.marker")
                index = channel["page"]
                if type(index) is not int or not 0 <= index < len(tf.pages):
                    raise ExperimentError("Marker channel.page must identify an existing zero-based TIFF page")
                if marker in markers or index in indices or names[index] != marker:
                    raise ExperimentError("Marker/page declarations must be unique and match OME channel names exactly")
                markers.add(marker); indices.add(index)
                page = tf.pages[index]
                if len(page.shape) != 2 or page.dtype.name not in {"uint8", "uint16", "float32", "float64"}:
                    raise ExperimentError("GigaTIME channel pages must be scalar uint8/uint16/float TIFF planes")
                if "display_max" in channel:
                    _number(channel["display_max"], "channel.display_max", positive=True)
                elif page.dtype.kind == "f":
                    raise ExperimentError("Floating-point marker pages require an explicit positive channel.display_max")
                if dimensions is not None and tuple(page.shape) != dimensions:
                    raise ExperimentError("Marker channels must share the same raster dimensions")
                dimensions = tuple(page.shape)
            return dict(width=int(dimensions[1]), height=int(dimensions[0]), markers=markers)
    except (ImportError, OSError, ValueError, ET.ParseError) as exc:
        if isinstance(exc, ExperimentError):
            raise
        raise ExperimentError(f"Cannot inspect GigaTIME TIFF header {path}: {exc}") from exc


def _gigatime_extent(artifact, image):
    extent = _spatial_fields(artifact.get("extent"), ("x", "y", "width", "height"), (), "marker image.extent")
    for key in ("x", "y", "width", "height"):
        _number(extent[key], f"extent.{key}", positive=key in {"width", "height"})
    if extent["x"] < 0 or extent["y"] < 0 or extent["x"] + extent["width"] > image["width"] + 1e-7 or extent["y"] + extent["height"] > image["height"] + 1e-7:
        raise ExperimentError("Marker image extent must lie inside its declared native H&E frame")
    return extent


def _gigatime_affine(artifact, source, header):
    convention = artifact.get("source_coordinate_convention")
    if convention not in ("zero-based-pixel-index", "native-image-pixel-centres"):
        raise ExperimentError("Declare the GigaTIME cell source_coordinate_convention explicitly")
    affine = _array(artifact.get("source_to_image_affine"), "source_to_image_affine")
    if len(affine) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in affine):
        raise ExperimentError("Cell scores require a finite invertible 3x3 source_to_image_affine")
    for row in affine:
        for value in row:
            _number(value, "source_to_image_affine value")
    determinant = affine[0][0] * affine[1][1] - affine[0][1] * affine[1][0]
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ExperimentError("Cell score source_to_image_affine must be invertible")
    extent = source["extent"]
    sx, sy = extent["width"] / header["width"], extent["height"] / header["height"]
    expected = ([[sx, 0, extent["x"] + sx / 2], [0, sy, extent["y"] + sy / 2], [0, 0, 1]]
                if convention == "zero-based-pixel-index" else [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if any(not math.isclose(affine[i][j], expected[i][j], rel_tol=1e-10, abs_tol=1e-8)
           for i in range(3) for j in range(3)):
        raise ExperimentError("Cell score affine must match the marker image extent; raw indices require the explicit half-pixel centre shift, native H&E centres require identity")
    return affine


def _gigatime_scores(artifact, path, source, source_header, mask, mask_header, image, fill):
    if artifact.get("score_semantics") != GIGATIME_SCORE_SEMANTICS:
        raise ExperimentError("GigaTIME scores must declare uncalibrated_virtual_marker_score semantics")
    if artifact.get("value_encoding", "native_export_values") != "native_export_values":
        raise ExperimentError("GigaTIME score values must retain native_export_values; no normalization is automatic")
    if artifact.get("coordinate_convention", COORDINATE_CONVENTION) != COORDINATE_CONVENTION:
        raise ExperimentError("GigaTIME output coordinates require the native H&E pixel-centre convention")
    if artifact.get("mask_name") != mask["mask_name"]:
        raise ExperimentError("Cell score mask_name differs from its linked segmentation mask")
    affine = _gigatime_affine(artifact, source, source_header)
    keys = [_text(artifact.get(key), f"cell scores.{key}") for key in ("id_column", "x_column", "y_column")]
    if len(set(keys)) != 3:
        raise ExperimentError("Cell score identity and coordinate columns must be distinct")
    marker_columns = _array(artifact.get("marker_columns"), "cell scores.marker_columns")
    markers, columns = set(), []
    for item in marker_columns:
        _spatial_fields(item, ("marker", "column"), (), "cell score marker column")
        marker, column = _text(item["marker"], "score marker"), _text(item["column"], "score column")
        if marker not in source_header["markers"] or marker in markers or column in columns or column in keys:
            raise ExperimentError("Cell marker columns must be unique, explicit and match declared source image markers")
        markers.add(marker); columns.append(column)
    if not columns:
        raise ExperimentError("Declare at least one explicit GigaTIME marker score column")
    expected_format = "csv.gz" if path.name.lower().endswith(".csv.gz") else "csv" if path.suffix.lower() == ".csv" else None
    if artifact.get("format") != expected_format or expected_format is None:
        raise ExperimentError("Cell score format must match its CSV/CSV.GZ filename")
    seen = set()
    try:
        with _csv(path) as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            if len(set(header)) != len(header) or any(key not in header for key in [*keys, *columns]):
                missing = [column for column in [*keys, *columns] if column not in header]
                raise ExperimentError(f"Cell score table has missing or duplicate declared columns; missing: {', '.join(missing) or 'none'}")
            for line, row in enumerate(reader, 2):
                if None in row or any(row[key] is None for key in [*keys, *columns]):
                    raise ExperimentError(f"Malformed cell score CSV row at line {line}")
                identifier = row[keys[0]]
                if not re.fullmatch(r"[1-9][0-9]*", identifier) or identifier in seen or int(identifier) > mask_header["max_label"]:
                    raise ExperimentError(f"Cell label IDs must be unique positive decimal integers representable by their mask dtype (line {line})")
                seen.add(identifier)
                try:
                    x, y = float(row[keys[1]]), float(row[keys[2]])
                except ValueError as exc:
                    raise ExperimentError(f"CSV line {line}, coordinate columns {keys[1]}/{keys[2]} must contain finite numbers") from exc
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ExperimentError(f"CSV line {line}, coordinate columns {keys[1]}/{keys[2]} must be finite")
                if artifact["source_coordinate_convention"] == "zero-based-pixel-index" and not (0 <= x < source_header["width"] and 0 <= y < source_header["height"]):
                    raise ExperimentError(f"CSV line {line}: raw cell centroid lies outside its linked marker raster")
                tx = affine[0][0] * x + affine[0][1] * y + affine[0][2]
                ty = affine[1][0] * x + affine[1][1] * y + affine[1][2]
                if not (0 <= tx < image["width"] and 0 <= ty < image["height"]):
                    raise ExperimentError(f"CSV line {line}: transformed cell centroid lies outside its native tissue image")
                extent = source["extent"]
                if not (extent["x"] <= tx < extent["x"] + extent["width"] and extent["y"] <= ty < extent["y"] + extent["height"]):
                    raise ExperimentError(f"CSV line {line}: cell centroid lies outside its linked marker image extent")
                for column in columns:
                    try:
                        value = float(row[column])
                    except ValueError as exc:
                        raise ExperimentError(f"CSV line {line}, column '{column}': expected a finite nonnegative exported score") from exc
                    if not math.isfinite(value) or value < 0:
                        raise ExperimentError(f"CSV line {line}, column '{column}': GigaTIME exported cell scores must be finite and nonnegative; values are not rescaled")
                if "mask_name" in header and row.get("mask_name") != mask["mask_name"]:
                    raise ExperimentError(f"CSV line {line}, column 'mask_name': cell score mask_name contradicts its linked mask")
    except (OSError, EOFError, csv.Error, ValueError) as exc:
        if isinstance(exc, ExperimentError):
            raise
        raise ExperimentError(f"Cannot read GigaTIME cell scores: {exc}") from exc
    if fill and "row_count" not in artifact:
        artifact["row_count"] = len(seen)
    if type(artifact.get("row_count")) is not int or artifact["row_count"] != len(seen):
        raise ExperimentError("GigaTIME cell score row_count disagrees with its table")


@contextmanager
def _gigatime_error_context(tissue, artifact):
    try:
        yield
    except (ExperimentError, OSError, ValueError) as exc:
        raise ExperimentError(f"Tissue '{tissue['tissue_id']}', artifact '{artifact['artifact_id']}', file '{artifact.get('path', '<missing>')}': {exc}") from exc


def _validate_gigatime_artifacts(payload, root, *, fill=False):
    for tissue in payload["tissues"]:
        typed = [artifact for artifact in tissue["artifacts"] if artifact.get("role") in GIGATIME_ROLES]
        if not typed:
            continue
        artifacts = {artifact["artifact_id"]: artifact for artifact in tissue["artifacts"]}
        image = tissue["image"]
        headers, paths = {}, {}
        for artifact in typed:
            with _gigatime_error_context(tissue, artifact):
                if (artifact.get("frame_id") != image["frame_id"] or
                        _digest(artifact.get("image_sha256"), "GigaTIME image_sha256") != _digest(image["sha256"], "image.sha256")):
                    raise ExperimentError("GigaTIME artifact source image SHA-256/frame differs from its tissue")
                path = _verify_asset(artifact, root, "GigaTIME artifact")
                paths[artifact["artifact_id"]] = path
                role = artifact["role"]
                if role in {"gigatime_marker_image", "cell_segmentation_mask"}:
                    formats = {"ome-tiff"} if role == "gigatime_marker_image" else {"tiff", "ome-tiff"}
                    if not isinstance(artifact.get("format"), str) or artifact["format"] not in formats or path.suffix.lower() not in {".tif", ".tiff"}:
                        raise ExperimentError("GigaTIME marker images require OME-TIFF; cell masks require TIFF/OME-TIFF")
                    if role == "gigatime_marker_image":
                        if artifact.get("score_semantics") != GIGATIME_SCORE_SEMANTICS:
                            raise ExperimentError("Marker image must declare uncalibrated_virtual_marker_score semantics")
                        _gigatime_extent(artifact, image)
                    else:
                        _text(artifact.get("mask_name"), "segmentation mask.mask_name")
                        if artifact.get("segmentation_unit") not in ("cell", "nucleus"):
                            raise ExperimentError("Segmentation unit must be cell or nucleus; a grid/tissue mask is not cell segmentation")
                    header = _gigatime_tiff(path, artifact, image)
                    headers[artifact["artifact_id"]] = header
                    if role == "cell_segmentation_mask":
                        for key in ("width", "height"):
                            if fill and key not in artifact:
                                artifact[key] = header[key]
                            if type(artifact.get(key)) is not int or artifact[key] != header[key]:
                                raise ExperimentError("Declared cell mask dimensions disagree with its TIFF header")
        for artifact in typed:
            if artifact["role"] != "gigatime_cell_scores":
                continue
            with _gigatime_error_context(tissue, artifact):
                source_id = _text(artifact.get("source_artifact_id"), "cell scores.source_artifact_id")
                mask_id = _text(artifact.get("mask_artifact_id"), "cell scores.mask_artifact_id")
                if (source_id not in artifacts or artifacts[source_id]["role"] != "gigatime_marker_image" or
                        mask_id not in artifacts or artifacts[mask_id]["role"] != "cell_segmentation_mask"):
                    raise ExperimentError("Cell scores require explicit marker-image and cell-segmentation-mask artifact links in the same tissue")
                _gigatime_scores(artifact, paths[artifact["artifact_id"]], artifacts[source_id], headers[source_id],
                                 artifacts[mask_id], headers[mask_id], image, fill)


def validate_experiment(path: Path) -> dict:
    """Verify schema, local assets, hashes, identity and every cached CSV row."""
    path = Path(path).absolute()
    payload = _read_json(path)
    if payload.get("schema") != SCHEMA or type(payload.get("schema_version")) is not int or payload["schema_version"] != VERSION:
        raise ExperimentError(f"Expected {SCHEMA} schema_version={VERSION}")
    _identifier(payload.get("experiment_id"), "experiment_id")
    _text(payload.get("name"), "name")
    _mapping(payload.get("settings"), "settings")
    _mapping(payload.get("provenance"), "provenance")
    tissues = payload.get("tissues")
    if not isinstance(tissues, list) or not tissues:
        raise ExperimentError("Experiment must contain at least one tissue")
    seen, definitions = set(), {}
    for index, tissue in enumerate(tissues):
        label = f"tissues[{index}]"
        _mapping(tissue, label)
        tissue_id = _identifier(tissue.get("tissue_id"), f"{label}.tissue_id")
        if tissue_id in seen:
            raise ExperimentError(f"Duplicate tissue_id: {tissue_id}")
        seen.add(tissue_id)
        feature_ids = set()
        _text(tissue.get("name"), f"{label}.name")
        if "specimen_id" in tissue:
            _text(tissue["specimen_id"], f"{label}.specimen_id")
        if "provenance" in tissue:
            _mapping(tissue["provenance"], f"{label}.provenance")
        image = _mapping(tissue.get("image"), f"{label}.image")
        if image.get("coordinate_convention", COORDINATE_CONVENTION) != COORDINATE_CONVENTION:
            raise ExperimentError(f"{label}: unsupported image coordinate convention")
        _identifier(image.get("frame_id"), f"{label}.image.frame_id")
        for key in ("width", "height"):
            if type(image.get(key)) is not int or image[key] <= 0:
                raise ExperimentError(f"{label}.image.{key} must be a positive integer")
        for key in ("mpp_x", "mpp_y"):
            _number(image.get(key), f"{label}.image.{key}", positive=True)
        image_path = _verify_asset(image, path.parent, f"{label}.image")
        if image_shape(image_path) != (image["width"], image["height"]):
            raise ExperimentError(f"{label}: image header dimensions disagree with manifest")
        if "registered_tissue" in tissue:
            try:
                from registered_tissue_assets import validate_registered_tissue
                validate_registered_tissue(tissue["registered_tissue"], path.parent, image)
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
                raise ExperimentError(f"{label}: dual-MPP tissue registration is invalid: {exc}") from exc
        artifacts = tissue.get("artifacts")
        features = tissue.get("feature_sets")
        if not isinstance(artifacts, list) or not isinstance(features, list):
            raise ExperimentError(f"{label}.artifacts and feature_sets must be arrays")
        artifact_ids = set()
        for artifact in artifacts:
            _mapping(artifact, f"{label}.artifact")
            aid = _identifier(artifact.get("artifact_id"), f"{label}.artifact_id")
            if aid in artifact_ids:
                raise ExperimentError(f"Duplicate artifact_id: {aid}")
            artifact_ids.add(aid)
            _text(artifact.get("role"), f"{label}.artifact.role")
            if artifact["role"] in GIGATIME_ROLES:
                with _gigatime_error_context(tissue, artifact):
                    _verify_asset(artifact, path.parent, f"{label}.artifact {aid}")
            else:
                _verify_asset(artifact, path.parent, f"{label}.artifact {aid}")
        for feature in features:
            _mapping(feature, f"{label}.feature")
            fid = _identifier(feature.get("feature_set_id"), f"{label}.feature_set_id")
            if fid in feature_ids:
                raise ExperimentError(f"Duplicate feature_set_id: {fid}")
            feature_ids.add(fid)
            feature_path = _verify_asset(feature, path.parent, f"{label}.feature {fid}")
            _validate_features(feature, feature_path, image, f"{label}.feature {fid}")
            _register_definition(definitions, feature, image)
        prepared = _prepare_cell_classification(tissue, path.parent)
        if prepared is not None:
            reference, expected = prepared
            if reference != tissue["cell_classification"]:
                raise ExperimentError("Cell classification reference is missing verified review metadata")
            matching = [feature for feature in features if feature["feature_set_id"] == expected["feature_set_id"]]
            expected["path"] = Path(os.path.relpath(expected["path"], path.parent)).as_posix()
            if len(matching) != 1 or matching[0] != expected:
                raise ExperimentError("Cell classification nested feature must be registered exactly once with identical signature/frame/path")
    for tissue in payload["tissues"]:
        for artifact in tissue["artifacts"]:
            if artifact.get("role") == "virchow2_rgb_image" and artifact.get("method") == "continuous-deployment-head-v1":
                _validate_continuous_rgb(tissue, artifact, path.parent)
            if artifact.get("role") == "virchow2_rgb_image" and artifact.get("method") == "dense-he-guided-readout-v1":
                _validate_dense_rgb(tissue, artifact, path.parent)
        _validate_bora_group_artifacts(tissue, path.parent)
        _validate_bora_local_artifacts(tissue, path.parent)
    _validate_spatial_artifacts(payload, path.parent)
    _validate_gigatime_artifacts(payload, path.parent)
    _validate_observation_reductions(payload, path.parent)
    return payload


def _verify_asset(record, root, label):
    path = _path(record.get("path"), root, f"{label}.path")
    if Path(record["path"]).is_absolute():
        raise ExperimentError(f"{label}.path must be relative to the experiment file")
    if sha256(path) != _digest(record.get("sha256"), f"{label}.sha256"):
        raise ExperimentError(f"{label} SHA-256 mismatch: {path}")
    return path


def _read_json(path):
    try:
        return _mapping(json.loads(Path(path).read_text(encoding="utf-8")), str(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Cannot read JSON {path}: {exc}") from exc


def atomic_json(path: Path, payload: dict, *, overwrite=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # A same-directory hard link publishes a complete new index without
            # replacing a file created concurrently after the initial check.
            os.link(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def export_experiment(spec: dict, output: Path, *, base_dir=Path("."), overwrite=False) -> dict:
    """Write a checked experiment from an explicit spec; never copy/re-encode assets."""
    output = Path(os.path.abspath(output))
    if output.exists() and not overwrite:
        raise ExperimentError(f"Output already exists: {output}; use --overwrite to replace only this index")
    spec = copy.deepcopy(_mapping(spec, "spec"))
    if "schema" in spec and (spec["schema"] != SCHEMA or type(spec.get("schema_version")) is not int or spec["schema_version"] != VERSION):
        raise ExperimentError(f"Cannot re-export an unknown experiment schema; expected {SCHEMA} version {VERSION}")
    tissues = spec.get("tissues")
    if not isinstance(tissues, list) or not tissues:
        raise ExperimentError("Spec must contain at least one tissue")
    normalized = []
    tissue_ids, definitions = set(), {}
    for index, tissue in enumerate(tissues):
        label = f"tissues[{index}]"
        _mapping(tissue, label)
        supplied_image = _mapping(tissue.get("image"), f"{label}.image")
        if supplied_image.get("coordinate_convention", COORDINATE_CONVENTION) != COORDINATE_CONVENTION:
            raise ExperimentError(f"{label}.image.coordinate_convention conflicts with the native image pixel-centre contract; transform coordinates explicitly")
        image, image_path = _asset(tissue.get("image"), base_dir, output, f"{label}.image")
        actual_shape = image_shape(image_path)
        for key, observed in zip(("width", "height"), actual_shape):
            if key in image and image[key] != observed:
                raise ExperimentError(f"{label}.image.{key} disagrees with image header")
            image[key] = observed
        for key in ("mpp_x", "mpp_y"):
            _number(image.get(key), f"{label}.image.{key}", positive=True)
        name = _text(tissue.get("name"), f"{label}.name")
        default_id = "tissue-" + hashlib.sha256(f"{name}\n{image['sha256']}".encode()).hexdigest()[:20]
        tid = _identifier(tissue.get("tissue_id", default_id), f"{label}.tissue_id")
        if tid in tissue_ids:
            raise ExperimentError(f"Duplicate tissue_id: {tid}; provide distinct explicit IDs for distinct tissues")
        tissue_ids.add(tid)
        feature_ids = set()
        if "frame_id" not in image and tissue.get("cell_classification") is not None:
            reference = _mapping(tissue["cell_classification"], f"{label}.cell_classification")
            if "index" in reference:
                index_path = _path(reference["index"].get("path"), base_dir, "cell index")
                try:
                    with sqlite3.connect(f"file:{index_path}?mode=ro", uri=True) as connection:
                        row = connection.execute("SELECT value FROM metadata WHERE key='image'").fetchone()
                    image["frame_id"] = json.loads(row[0]).get("frame_id")
                except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
                    raise ExperimentError(f"Cannot read indexed cell image identity: {exc}") from exc
            else:
                classification_path = _path(reference.get("path"), base_dir, f"{label}.cell_classification.path")
                image["frame_id"] = _read_json(classification_path).get("image", {}).get("frame_id")
        image.setdefault("frame_id", f"{tid}:image")
        image["coordinate_convention"] = COORDINATE_CONVENTION
        image.setdefault("coordinate_space", "crop-native-pixels")
        image.setdefault("origin_x", 0)
        image.setdefault("origin_y", 0)
        result = copy.deepcopy(tissue)
        result.update(tissue_id=tid, name=name, image=image, artifacts=[], feature_sets=[])
        if "registered_tissue" in tissue:
            declared = _mapping(tissue["registered_tissue"], f"{label}.registered_tissue")
            registered = {}
            for field in ("registration", "image"):
                registered[field], _ = _asset(declared.get(field), base_dir, output, f"{label}.registered_tissue.{field}")
            registered["assets"] = []
            for ai, asset in enumerate(_array(declared.get("assets", []), f"{label}.registered_tissue.assets")):
                normalized_asset, _ = _asset(asset, base_dir, output, f"{label}.registered_tissue.assets[{ai}]")
                registered["assets"].append(normalized_asset)
            try:
                from registered_tissue_assets import validate_registered_tissue
                validate_registered_tissue(registered, output.parent, image)
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
                raise ExperimentError(f"{label}: dual-MPP tissue registration is invalid: {exc}") from exc
            result["registered_tissue"] = registered
        for key in ("specimen_id", "provenance"):
            if key in tissue:
                result[key] = tissue[key]
        if "specimen_id" in result:
            _text(result["specimen_id"], f"{label}.specimen_id")
        if "provenance" in result:
            _mapping(result["provenance"], f"{label}.provenance")
        for ai, artifact in enumerate(_array(tissue.get("artifacts", []), f"{label}.artifacts")):
            if isinstance(artifact, dict) and artifact.get("role") in GIGATIME_ROLES:
                context = dict(artifact, artifact_id=artifact.get("artifact_id", f"{tid}:artifact-{ai + 1}"))
                with _gigatime_error_context(result, context):
                    item, _ = _asset(artifact, base_dir, output, f"{label}.artifacts[{ai}]")
            else:
                item, _ = _asset(artifact, base_dir, output, f"{label}.artifacts[{ai}]")
            if (item.get("role") == "virchow2_rgb_image" and item.get("method") in {"continuous-deployment-head-v1", "dense-he-guided-readout-v1"}) or item.get("role") == "bora_refined_groups":
                item["provenance"], _ = _asset(artifact["provenance"], base_dir, output, f"{label}.artifact provenance")
            item.setdefault("artifact_id", f"{tid}:artifact-{ai + 1}")
            item.setdefault("frame_id", image["frame_id"])
            result["artifacts"].append(item)
        supplied_features = copy.deepcopy(_array(tissue.get("feature_sets", []), f"{label}.feature_sets"))
        prepared = _prepare_cell_classification(result, base_dir, output)
        if prepared is not None:
            reference, cell_feature = prepared
            result["cell_classification"] = reference
            existing = [feature for feature in supplied_features if feature.get("feature_set_id") == cell_feature["feature_set_id"]]
            if existing:
                normalized_expected, _ = _asset(cell_feature, base_dir, output, f"{label}.cell feature")
                normalized_existing, _ = _asset(existing[0], base_dir, output, f"{label}.existing cell feature")
                if len(existing) != 1 or normalized_existing != normalized_expected:
                    raise ExperimentError("Existing cell feature conflicts with classification nested signature/frame/path")
            else:
                supplied_features.append(cell_feature)
        for fi, feature in enumerate(supplied_features):
            item, feature_path = _asset(feature, base_dir, output, f"{label}.feature_sets[{fi}]")
            prefix = item.pop("feature_columns_prefix", None)
            if "feature_columns" not in item and prefix is not None:
                prefix = _text(prefix, f"{label}.feature_columns_prefix")
                with _csv(feature_path) as stream:
                    item["feature_columns"] = [column for column in next(csv.reader(stream), []) if column.startswith(prefix)]
            item.setdefault("feature_set_id", item.get("kind", "features"))
            fid = _identifier(item["feature_set_id"], f"{label}.feature_set_id")
            if fid in feature_ids:
                raise ExperimentError(f"Duplicate feature_set_id: {fid}")
            feature_ids.add(fid)
            item["row_count"] = _validate_features(item, feature_path, image, f"{label}.feature {fid}")
            _register_definition(definitions, item, image)
            result["feature_sets"].append(item)
        normalized.append(result)
    identity = "\n".join(sorted(f"{t['tissue_id']}:{t['image']['sha256']}" for t in normalized))
    payload = copy.deepcopy(spec)
    payload.update(schema=SCHEMA, schema_version=VERSION,
                   experiment_id=spec.get("experiment_id", "experiment-" + hashlib.sha256(identity.encode()).hexdigest()[:24]),
                   name=spec.get("name", "CellPhenotyper experiment"),
                   settings=spec.get("settings", {}), provenance=spec.get("provenance", {}), tissues=normalized)
    _identifier(payload["experiment_id"], "experiment_id")
    _text(payload["name"], "name")
    _mapping(payload["settings"], "settings")
    _mapping(payload["provenance"], "provenance")
    # Validate every field before publishing. No temporary JSON can claim completion.
    for tissue in normalized:
        _identifier(tissue["image"]["frame_id"], "image.frame_id")
        aids = set()
        for artifact in tissue["artifacts"]:
            aid = _identifier(artifact["artifact_id"], "artifact_id")
            if aid in aids:
                raise ExperimentError(f"Duplicate artifact_id: {aid}")
            aids.add(aid)
            _text(artifact.get("role"), "artifact.role")
    for tissue in payload["tissues"]:
        for artifact in tissue["artifacts"]:
            if artifact.get("role") == "virchow2_rgb_image" and artifact.get("method") == "continuous-deployment-head-v1":
                _validate_continuous_rgb(tissue, artifact, output.parent)
            if artifact.get("role") == "virchow2_rgb_image" and artifact.get("method") == "dense-he-guided-readout-v1":
                _validate_dense_rgb(tissue, artifact, output.parent)
        _validate_bora_group_artifacts(tissue, output.parent)
        _validate_bora_local_artifacts(tissue, output.parent)
    _validate_spatial_artifacts(payload, output.parent)
    _validate_gigatime_artifacts(payload, output.parent, fill=True)
    _validate_observation_reductions(payload, output.parent)
    atomic_json(output, payload, overwrite=overwrite)
    return payload


def _single(paths, label):
    paths = sorted(set(path for path in paths if path.is_file()))
    if len(paths) != 1:
        raise ExperimentError(f"Expected one {label}, found {len(paths)}; use --spec to disambiguate")
    return paths[0]


def _crop_calibration(image, sample, *, required):
    """Read the crop's independent sidecar; never substitute an expected MPP."""
    shift_path = image.parent / "shift.json"
    if not shift_path.is_file():
        if required:
            raise ExperimentError(f"{sample}: matching durable crop has no shift.json calibration: {image}")
        return None, None
    shift = _read_json(shift_path)
    calibration_keys = {"source_mpp", "microns_per_pixel", "mpp_x", "mpp_y",
        "source_mpp_x", "source_mpp_y", "microns_per_pixel_x", "microns_per_pixel_y"}
    if not calibration_keys.intersection(shift) and not required:
        return None, shift
    calibration = {}
    for axis in ("x", "y"):
        keys = (f"mpp_{axis}", f"source_mpp_{axis}", f"microns_per_pixel_{axis}",
                "source_mpp", "microns_per_pixel")
        values = [_number(shift[key], f"{sample}.durable mpp_{axis}", positive=True)
                  for key in keys if key in shift]
        if not values:
            raise ExperimentError(f"{sample}: durable crop shift.json has no independent MPP calibration for {axis}")
        if any(value != values[0] for value in values):
            raise ExperimentError(f"{sample}: durable crop calibration sources conflict: {image}")
        calibration[f"mpp_{axis}"] = values[0]
    if "crop_size" in shift:
        size = _mapping(shift["crop_size"], f"{sample}.crop_size")
        if (size.get("width"), size.get("height")) != image_shape(image):
            raise ExperimentError(f"{sample}: durable crop dimensions disagree with shift.json: {image}")
    return calibration, shift


def _select_source_image(candidates, sample, *, source_sha256=None, shape=None,
                         mpp=None, crop=False, require_calibration=False):
    """Resolve producer identity before preferring any publication directory."""
    candidates = sorted(set(path for path in candidates if path.is_file()))
    if source_sha256 is None:
        matching = [_single(candidates, f"calibrated image for {sample}")]
    else:
        source_sha256 = _digest(source_sha256, f"{sample}.source image SHA-256")
        matching = [path for path in candidates if sha256(path) == source_sha256]
        if not matching:
            raise ExperimentError(f"{sample}: source image SHA-256 has no matching durable image; cached source is missing or changed")
    if shape is not None and any(image_shape(path) != tuple(shape) for path in matching):
        raise ExperimentError(f"{sample}: producer source image dimensions disagree with the durable crop")
    shifts = []
    if crop:
        for path in matching:
            calibration, shift = _crop_calibration(path, sample,
                required=require_calibration or len(candidates) > 1)
            if calibration is not None and mpp is not None and any(
                    calibration[f"mpp_{axis}"] != mpp[index] for index, axis in enumerate(("x", "y"))):
                raise ExperimentError(f"{sample}: durable crop MPP calibration differs from the producer: {path}")
            shifts.append((calibration, shift))
    if len(matching) > 1:
        # Equal bytes do not establish the same original location/calibration.
        identities = [(calibration, {key: shift.get(key) for key in
            ("offset_crop_to_original", "full_size")} if shift is not None else None)
            for calibration, shift in shifts]
        if not crop or any(identity != identities[0] for identity in identities[1:]):
            raise ExperimentError(f"{sample}: matching duplicate crop calibration/origins conflict; use --spec to disambiguate")
        prepared = [path for path in matching if path.parent.name == "prepared_crop"]
        if len(prepared) != 1:
            raise ExperimentError(f"{sample}: identical matching source images lack one verified prepared_crop; use --spec to disambiguate")
        return prepared[0]
    return matching[0]


def _id_rows(path, id_column, label):
    """Read small coordinate/metadata records with strict string identity joins."""
    with _csv(path) as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if len(set(header)) != len(header) or id_column not in header:
            raise ExperimentError(f"{label}: missing or duplicate declared columns")
        rows = {}
        for row in reader:
            key = row.get(id_column)
            if not key or not key.strip() or key in rows or None in row or any(x is None for x in row.values()):
                raise ExperimentError(f"{label}: malformed, missing or duplicate observation ID")
            rows[key] = row
    if not rows:
        raise ExperimentError(f"{label}: no observations")
    return header, rows


def _validate_observation_reductions(payload, root):
    """Validate the existing ID-keyed viewer reduction contract before publication."""
    for tissue in payload["tissues"]:
        image = tissue["image"]
        features = {item["feature_set_id"]: item for item in tissue["feature_sets"]}
        for artifact in tissue["artifacts"]:
            if artifact["role"] != "observation_reduction" or "source_feature_set_id" not in artifact:
                continue
            source = features.get(artifact["source_feature_set_id"])
            if source is None:
                raise ExperimentError("KODAMA reduction refers to an unknown feature set")
            if (artifact.get("frame_id") != image["frame_id"] or
                    artifact.get("observation_unit") != source["observation_unit"] or
                    artifact.get("image_sha256", image["sha256"]) != image["sha256"]):
                raise ExperimentError("KODAMA reduction native frame/image/unit differs from its source")
            if artifact.get("dataset_id", "experiment:" + source["feature_set_id"]) != "experiment:" + source["feature_set_id"]:
                raise ExperimentError("KODAMA reduction dataset and source_feature_set_id disagree")
            dims = artifact.get("dimension_columns")
            id_column = artifact.get("id_column")
            if (not isinstance(dims, list) or len(dims) != 2 or any(not isinstance(x, str) or not x for x in dims) or
                    not isinstance(id_column, str) or len(set([id_column, *dims])) != 3):
                raise ExperimentError("KODAMA reduction needs two distinct dimension columns and an ID column")
            path = _path(artifact["path"], root, "KODAMA reduction")
            expected_format = "csv.gz" if path.name.endswith(".csv.gz") else "csv"
            if artifact.get("format") != expected_format:
                raise ExperimentError("KODAMA reduction format must match its coordinate CSV")
            header, incoming = _id_rows(path, id_column, "KODAMA reduction")
            _, local = _id_rows(_path(source["path"], root, "KODAMA features"), source["id_column"], "KODAMA source features")
            if set(incoming) != set(local) or artifact.get("row_count") != len(local):
                raise ExperimentError("KODAMA reduction IDs/count must match every source observation exactly once")
            if not set(dims).issubset(header):
                raise ExperimentError("KODAMA reduction has missing dimension columns")
            try:
                if any(not math.isfinite(float(row[column])) for row in incoming.values() for column in dims):
                    raise ValueError("nonfinite coordinate")
                native = artifact.get("native_coordinate_columns")
                if native is not None:
                    if not isinstance(native, list) or len(native) != 2 or not set(native).issubset(header):
                        raise ValueError("native_coordinate_columns must declare two existing columns")
                    for key, row in incoming.items():
                        if tuple(float(row[x]) for x in native) != tuple(float(local[key][x]) for x in (source["x_column"], source["y_column"])):
                            raise ValueError("native source coordinates differ")
            except (ValueError, KeyError, TypeError) as exc:
                raise ExperimentError(f"KODAMA reduction contains invalid or mismatched coordinates: {exc}") from exc


def _publish_cluster_features(run, sample, feature, clusters):
    """Append ID-joined categorical metadata without changing any predictor text."""
    source = Path(feature["path"])
    with _csv(source) as stream:
        header = next(csv.reader(stream))
    additions = [key for key in clusters[0] if key != "__observation_id"] if clusters else []
    if not additions:
        return
    if set(header) & set(additions):
        raise ExperimentError(f"{sample}: cluster metadata would overwrite an existing feature column")
    identity = hashlib.sha256(json.dumps(dict(source=sha256(source), metadata=clusters), sort_keys=True).encode()).hexdigest()
    destination = run / "00_execution" / "project_assets" / f"{sample}-virchow2-clusters-{identity[:24]}.csv.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".cluster-features-", delete=False) as handle:
            temporary = Path(handle.name)
        with _csv(source) as stream, temporary.open("wb") as binary:
            # Stable gzip bytes across repeated exports and relocations.
            import io
            with gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as target:
                    reader, writer = csv.reader(stream), csv.writer(target)
                    original_header = next(reader)
                    writer.writerow([*original_header, *additions])
                    id_index = original_header.index(feature["id_column"])
                    by_id = {item.pop("__observation_id"): item for item in copy.deepcopy(clusters)}
                    # The internal join ID is never a predictor or output column.
                    if "__observation_id" in original_header:
                        raise ExperimentError("Reserved cluster join key occurs in source table")
                    for row in reader:
                        values = by_id.pop(row[id_index], None)
                        if values is None:
                            raise ExperimentError(f"{sample}: clustering IDs do not exactly match source features")
                        writer.writerow([*row, *[values[key] for key in additions]])
                    if by_id:
                        raise ExperimentError(f"{sample}: clustering has extra observation IDs")
        if destination.exists():
            if sha256(destination) != sha256(temporary):
                raise ExperimentError(f"Derived feature asset was changed: {destination}")
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    feature.update(path=str(destination), format="csv.gz", cluster_columns=additions,
                   metadata_derivation=dict(original_feature_sha256=sha256(source), join="exact-observation-id",
                                            predictor_values="original CSV field text preserved"))


def _discover_kodama(run, sample, tissue, marker):
    folder = run / "10_kodama" / sample
    csvs = sorted(folder.glob("**/kodama_full_*.csv"))
    rdata = sorted(folder.glob("**/kodama_full_*.RData"))
    for declared in folder.glob("**/kodama_full_*.csv.json"):
        if not Path(str(declared).removesuffix(".json")).is_file():
            raise ExperimentError(f"{sample}: declared KODAMA coordinate CSV is missing: {declared}")
    if not csvs:
        if rdata:
            raise ExperimentError(f"{sample}: KODAMA RData exists but portable coordinate CSV/identity metadata is missing; rerun KODAMA with the current exporter")
        tissue["provenance"]["kodama_review"] = dict(status="not_produced", reason="No KODAMA stage output is present")
        return
    feature = tissue["feature_sets"][0]
    original_features = Path(feature["path"])
    _, expected = _id_rows(original_features, "cell_id", f"{sample} source tile observations")
    for path in csvs:
        metadata_path = Path(str(path) + ".json")
        if not metadata_path.is_file():
            raise ExperimentError(f"{sample}: KODAMA coordinate identity metadata is missing: {metadata_path}")
        metadata = _read_json(metadata_path)
        if (not isinstance(marker.get("frame_contract"), dict) or
                metadata.get("schema") != "cellphenotyper-kodama-reduction" or metadata.get("schema_version") != 1 or
                metadata.get("path") != path.name or metadata.get("source_image_sha256") != sha256(Path(tissue["image"]["path"])) or
                metadata.get("source_embeddings_sha256") != sha256(original_features) or
                metadata.get("frame_contract") != marker.get("frame_contract") or
                metadata.get("encoder_revision") != marker.get("encoder_revision") or
                metadata.get("checkpoint_sha256") != marker.get("checkpoint_sha256") or
                metadata.get("observation_unit") != "tile" or
                metadata.get("id_column") != "cell_id" or
                metadata.get("native_coordinate_columns") != ["native_x", "native_y"] or
                metadata.get("dimension_columns") != ["kodama_1", "kodama_2"] or
                metadata.get("selected_modes") not in ("tile", ["tile"])):
            raise ExperimentError(f"{sample}: KODAMA identity metadata differs from the current Virchow2 source/frame")
        artifact = dict(role="observation_reduction", path=str(path), format="csv", source_feature_set_id="virchow2-pooled",
                        observation_unit="tile", row_count=metadata.get("row_count"), id_column=metadata.get("id_column"),
                        dimension_columns=metadata.get("dimension_columns"), native_coordinate_columns=metadata.get("native_coordinate_columns"),
                        reduction_id=path.stem.replace("_", "-"), method="KODAMA.visualization",
                        label="KODAMA tissue map", image_sha256=metadata["source_image_sha256"])
        tissue["artifacts"].extend([artifact, dict(role="kodama_identity", path=str(metadata_path))])
    tissue["artifacts"].append(dict(role="original_feature_table", path=str(original_features)))
    merged = {key: {"__observation_id": key} for key in expected}
    cluster_paths = sorted((run / "11_clustering" / sample).glob("**/*_cluster.csv"))
    for path in cluster_paths:
        _, rows = _id_rows(path, "label", f"{sample} cluster metadata")
        if set(rows) != set(expected):
            raise ExperimentError(f"{sample}: clustering IDs must match every tile exactly once")
        variant = path.name.removeprefix(sample + "_").removesuffix("_cluster.csv")
        summary_path = path.with_name(path.name.removesuffix(".csv") + "_summary.csv")
        with _csv(summary_path) as stream:
            summaries = list(csv.DictReader(stream))
        if len(summaries) != 1:
            raise ExperimentError(f"{sample}: clustering summary must contain one settings record")
        algorithm = _identifier(summaries[0].get("cluster_algorithm"), "clustering algorithm")
        prefix = algorithm if variant == "standard" else f"{algorithm}_{_identifier(variant, 'cluster variant')}"
        selected = [key for key in ("cluster", "interpretable_cluster", "interpretation_status", "uncertainty_reason") if key in next(iter(rows.values()))]
        if "cluster" not in selected:
            raise ExperimentError(f"{sample}: clustering has no membership column")
        for key, row in rows.items():
            for column in selected:
                name = f"{prefix}_{column}"
                if name in merged[key]:
                    raise ExperimentError(f"{sample}: ambiguous duplicate cluster metadata field {name}")
                merged[key][name] = row[column]
        tissue["artifacts"].extend([dict(role="clustering_source", path=str(path)), dict(role="clustering_settings", path=str(summary_path))])
    if cluster_paths:
        _publish_cluster_features(run, sample, feature, list(merged.values()))
    tissue["provenance"]["kodama_review"] = dict(status="available", reductions=len(csvs), cluster_variants=len(cluster_paths), join="exact-observation-id")


def _discover_virchow_tissues(run: Path) -> list[dict]:
    """Discover complete current Virchow2 pooled exports in a durable run tree."""
    run = Path(os.path.abspath(run))
    markers = sorted((run / "09_embeddings").glob("**/VIRCHOW2_COMPLETE.json"))
    tissues = []
    for marker_path in markers:
        sample = marker_path.relative_to(run / "09_embeddings").parts[0]
        marker = _read_json(marker_path)
        metadata_path = _single((run / "09_grid_tiles" / sample).glob("*grid_metadata.json"), f"grid metadata for {sample}")
        objects_path = _single((run / "09_grid_tiles" / sample).glob("*grid_objects.csv"), f"grid observations for {sample}")
        metadata = _read_json(metadata_path)
        features = _single(marker_path.parent.glob("virchow2_embeddings_shard*.csv*"), f"embedding table for {sample}")
        for key, asset in (("grid_sha256", objects_path), ("grid_metadata_sha256", metadata_path), ("embeddings_sha256", features)):
            if _digest(marker.get(key), f"{sample}.{key}") != sha256(asset):
                raise ExperimentError(f"{sample}: completion record {key} does not match {asset}")
        if (marker.get("encoder") != "Virchow2" or marker.get("features_per_tile") != 2560 or
                marker.get("inner_square_used") is not False or marker.get("tile_overlap_source_px") != 0 or
                metadata.get("tile_layout") != "nonoverlapping" or metadata.get("tile_origin") != "top_left"):
            raise ExperimentError(f"{sample}: expected current complete non-overlapping Virchow2 pooled tiles")
        footprint = metadata.get("extraction_tile_size_source_px")
        if (not isinstance(footprint, int) or footprint <= 0 or
                footprint != metadata.get("grid_stride_source_px") or metadata.get("context_overlap_source_px") != 0):
            raise ExperimentError(f"{sample}: inconsistent non-overlapping grid geometry")
        if metadata.get("model_tile_size_px") != 224 or metadata.get("target_mpp") != .5:
            raise ExperimentError(f"{sample}: Virchow2 model sampling must be 224 pixels at 0.5 um/pixel")
        for axis in ("x", "y"):
            mpp = _number(metadata.get(f"source_mpp_{axis}"), f"{sample}.source_mpp_{axis}", positive=True)
            if abs(footprint * mpp / 224 - .5) > .01:
                raise ExperimentError(f"{sample}: source tile footprint does not match the physical Virchow2 field")
        frame = marker.get("frame_contract")
        recorded_hash = marker.get("image_sha256")
        if frame is not None:
            _mapping(frame, f"{sample}.frame_contract")
            frame_hash = _digest(frame.get("source_sha256"), f"{sample}.frame_contract.source_sha256")
            if recorded_hash is not None and _digest(recorded_hash, f"{sample}.image_sha256") != frame_hash:
                raise ExperimentError(f"{sample}: encoder image and frame digests disagree")
            recorded_hash = frame_hash
            if (frame.get("source_shape_yx") != [metadata["image_height_px"], metadata["image_width_px"]] or
                    frame.get("native_mpp_xy") != [metadata["source_mpp_x"], metadata["source_mpp_y"]] or
                    frame.get("coordinate_space") != "native_image_pixel_edges" or
                    frame.get("native_frame_origin_xy") != [0, 0] or
                    frame.get("checkpoint_sha256") != marker.get("checkpoint_sha256") or
                    frame.get("encoder_revision") != marker.get("encoder_revision")):
                raise ExperimentError(f"{sample}: encoder frame geometry or checkpoint differs from the grid")
        crops = list((run / "03_stardist" / sample).glob("**/crop_roi.tif"))
        candidates = crops or [Path(metadata.get("image", ""))]
        if not crops and not candidates[0].is_file():
            candidates = list((run / "01_input" / sample).glob("*.ome.tif*"))
        image = _select_source_image(candidates, sample, source_sha256=recorded_hash,
            shape=(metadata.get("image_width_px"), metadata.get("image_height_px")),
            mpp=(metadata["source_mpp_x"], metadata["source_mpp_y"]), crop=bool(crops))
        identity_status = ("Encoder completion source SHA-256 matches current image bytes; dimensions and sampling verified"
            if recorded_hash is not None else
            "Current image bytes and header verified; the legacy encoder completion record contains no source-image digest")
        # Match producer IDs/coordinates, rather than trusting arbitrary CSV ordering.
        with _csv(features) as emb_stream, objects_path.open(newline="", encoding="utf-8") as grid_stream:
            count = 0
            for embedding, observation in itertools.zip_longest(csv.DictReader(emb_stream), csv.DictReader(grid_stream)):
                if (embedding is None or observation is None or embedding.get("cell_id") != observation.get("label") or
                        float(embedding.get("cx", "nan")) != float(observation.get("x", "nan")) or
                        float(embedding.get("cy", "nan")) != float(observation.get("y", "nan"))):
                    raise ExperimentError(f"{sample}: embedding IDs/coordinates disagree with grid observations")
                count += 1
        if count != marker.get("observations") or count != metadata.get("retained_units"):
            raise ExperimentError(f"{sample}: completion/grid observation counts disagree")
        tissue = dict(name=sample, image=dict(path=str(image), mpp_x=metadata.get("source_mpp_x"),
                                            mpp_y=metadata.get("source_mpp_y")), artifacts=[],
                      provenance=dict(feature_export_complete_sha256=sha256(marker_path),
                                      image_identity_verification=identity_status),
                      feature_sets=[dict(kind="virchow2-pooled", observation_unit="tile", path=str(features),
                                         format="csv.gz" if features.name.endswith(".gz") else "csv",
                                         feature_columns=[f"feat_{i}" for i in range(1, 2561)],
                                         id_column="cell_id", x_column="cx", y_column="cy", coordinate_frame="image",
                                         encoder=dict(name="Virchow2", revision=marker.get("encoder_revision"),
                                                      checkpoint_sha256=marker.get("checkpoint_sha256")),
                                         preprocessing=dict(feature_definition="CLS+mean_patch", model_tile_px=224,
                                                            model_mpp=.5, input_normalization="ImageNet RGB", edge_padding="white"),
                                         grid=dict(tile_width_px=footprint, tile_height_px=footprint,
                                                   spacing_x_px=footprint, spacing_y_px=footprint,
                                                   origin_x_px=0, origin_y_px=0,
                                                   representative_policy="logical-centre-clipped-to-image", edge_policy="white-padded"))])
        shift_path = image.parent / "shift.json"
        if shift_path.is_file():
            shift = _read_json(shift_path)
            offset = shift.get("offset_crop_to_original", {})
            if "dx" in offset and "dy" in offset:
                dx, dy = (_number(offset[key], f"{sample}.crop_offset.{key}") for key in ("dx", "dy"))
                tissue["provenance"]["crop_to_original"] = dict(affine=[[1, 0, dx], [0, 1, dy], [0, 0, 1]],
                                                                full_size=shift.get("full_size"), shift_sha256=sha256(shift_path))
        for role, folder, pattern in (("tissue_annotations", "15_cluster_geojson", "**/mask.geojson"),
                                       ("refined_mask", "15_cluster_geojson", "**/mask.ome.tif")):
            for asset in sorted((run / folder / sample).glob(pattern)):
                tissue["artifacts"].append(dict(role=role, path=str(asset)))
        _discover_kodama(run, sample, tissue, marker)
        tissues.append(tissue)
    return tissues


def _cell_producer_outputs(run):
    return [(path.relative_to(run / folder).parts[0], path)
            for folder in CELL_PRODUCER_FOLDERS
            for path in sorted((run / folder).glob("*/**/cellvit_cell_classification.json"))]


def _validate_continuous_rgb(tissue, artifact, root):
    """Validate immutable global RGB/recipe/support before registering a layer."""
    from PIL import Image, ImageChops
    image = tissue["image"]
    proof_path = _verify_asset(artifact["provenance"], root, "continuous RGB provenance")
    proof = _read_json(proof_path)
    expected = dict(format="bora-continuous-rgb-v1", method="continuous-deployment-head-v1",
        source_frame_id=image["frame_id"], image_sha256=image["sha256"], continuous_fit=True,
        window_local_fits=False, full_RGB9_plus_19_applied=False,
        fallback_status="not-applicable-continuous-head", coordinate_space="native_image_pixel_edges",
        origin="top_left", label_kind="anonymous-KODAMA-coarse-groups", human_review_labels_used=False,
        semantic_tissue_labels_assigned=False, support_convention="binary-255-supported")
    if any(proof.get(k) != v for k, v in expected.items()):
        raise ExperimentError("Continuous RGB provenance differs from its registered image/frame/method")
    rgb_path = _verify_asset(artifact, root, "continuous RGB image")
    width, height = image_shape(rgb_path)
    scale = [image["width"] / width, image["height"] / height]
    extent = dict(x=0, y=0, width=image["width"], height=image["height"])
    affine = [[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]]
    if (proof.get("rgb_file_sha256") != sha256(rgb_path) or
            proof.get("native_shape_yx") != [image["height"], image["width"]] or
            proof.get("native_mpp_xy") != [image["mpp_x"], image["mpp_y"]] or
            proof.get("rgb_shape_yx") != [height, width] or proof.get("extent") != extent or
            proof.get("image_to_source_affine") != affine or
            proof.get("rgb_mpp_xy") != [image["mpp_x"] * image["width"] / width, image["mpp_y"] * image["height"] / height] or
            any(artifact.get(k) != v for k, v in dict(source_frame_id=image["frame_id"],
                image_sha256=image["sha256"], method=expected["method"], extent=extent,
                image_to_source_affine=affine, width=width, height=height).items()) or
            proof.get("conversion") != dict(conversion="continuous RGB9+19 deployment", components=[9, 19], weights=[.5, .5])):
        raise ExperimentError("Continuous RGB image/physical extent/affine disagrees with the native image")
    def companion(name):
        record = _mapping(proof.get(name), "continuous RGB " + name)
        relative = Path(_text(record.get("path"), name + ".path"))
        path = (proof_path.parent / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(proof_path.parent.resolve()):
            raise ExperimentError("Continuous RGB companion must be contained beside its provenance")
        return _verify_asset(record, proof_path.parent, name)
    recipe_path, support_path = companion("recipe"), companion("support_png")
    if proof.get("recipe_sha256") != sha256(recipe_path):
        raise ExperimentError("Continuous RGB recipe digest differs")
    envelope = _read_json(recipe_path)
    state = _mapping(envelope.get("payload"), "continuous recipe payload")
    canonical_state = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if envelope.get("sha256") != hashlib.sha256(canonical_state).hexdigest():
        raise ExperimentError("Continuous RGB recipe payload checksum mismatch")
    from virchow2_sharedcache import PREPROCESSING, canonical
    from virchow2_runtime import CHECKPOINT_SHA256
    encoder = dict(model_name="Virchow2", checkpoint_sha256=CHECKPOINT_SHA256,
        preprocessing_sha256=hashlib.sha256(canonical(PREPROCESSING)).hexdigest(), spatial_feature_dim=1280, tile_size_px=224)
    bindings = dict(source_frame_id=image["frame_id"], image_sha256=image["sha256"],
        method=expected["method"], encoder_identity=encoder, continuous_fit=True, window_local_fits=False, global_fit=True)
    if (proof.get("encoder_identity") != encoder or state.get("format") != expected["format"] or
            state.get("method") != expected["method"] or state.get("continuous_fit") is not True or
            state.get("window_local_fits") is not False or state.get("provenance", {}).get("bindings") != bindings):
        raise ExperimentError("Continuous RGB fitted recipe is not bound to this exact source/encoder/method")
    with Image.open(rgb_path) as rgb, Image.open(support_path) as support:
        if rgb.mode != "RGB" or support.mode != "RGB" or rgb.size != support.size:
            raise ExperimentError("Continuous RGB and binary support must be aligned 8-bit RGB PNGs")
        channels = support.split()
        if (ImageChops.difference(channels[0], channels[1]).getbbox() or
                ImageChops.difference(channels[0], channels[2]).getbbox() or
                any(value not in (0, 255) for count, value in channels[0].getcolors(maxcolors=3) or [(-1, -1)])):
            raise ExperimentError("Continuous RGB support must have identical binary 0/255 channels")
    return proof, recipe_path, support_path


def _discover_continuous_rgb(run, tissue):
    sample = tissue["name"]
    for proof_path in sorted((run / "15_cluster_geojson" / sample).glob("**/rgb_provenance.json")):
        proof = _read_json(proof_path)
        if proof.get("method") == "dense-he-guided-readout-v1":
            _discover_dense_rgb(run, tissue, proof_path, proof)
            continue
        image = tissue["image"]
        path = Path(image["path"])
        width, height = image_shape(path)
        image.update(sha256=sha256(path), width=width, height=height)
        # This is the very same native-image identity the CellViT producer uses.
        from cellvit_models import image_frame
        frame = image_frame(path, path.parent / "shift.json")
        if "frame_id" in image and image["frame_id"] != frame["frame_id"]:
            raise ExperimentError("Continuous RGB image identity conflicts with the cell producer")
        image.setdefault("frame_id", frame["frame_id"])
        rgb_path = proof_path.parent / "rgb_continuous.png"
        rgb_width, rgb_height = image_shape(rgb_path)
        artifact = dict(role="virchow2_rgb_image", path=str(rgb_path), sha256=sha256(rgb_path), format="png",
            source_frame_id=image["frame_id"], image_sha256=image["sha256"], coordinate_space="native_image_pixel_edges",
            origin="top_left", width=rgb_width, height=rgb_height, extent=proof.get("extent"),
            image_to_source_affine=proof.get("image_to_source_affine"), method="continuous-deployment-head-v1",
            full_RGB9_plus_19_applied=False, fallback_status="not-applicable-continuous-head",
            label="Virchow2 RGB (continuous whole-field fit)",
            provenance=dict(path=str(proof_path), sha256=sha256(proof_path)))
        checked = copy.deepcopy(artifact)
        checked["path"] = os.path.relpath(rgb_path, run)
        checked["provenance"]["path"] = os.path.relpath(proof_path, run)
        _, recipe, support = _validate_continuous_rgb(tissue, checked, run)
        tissue["artifacts"].extend([artifact, dict(role="virchow2_rgb_recipe", path=str(recipe)),
            dict(role="virchow2_rgb_support", path=str(support)), dict(role="virchow2_rgb_provenance", path=str(proof_path))])
        diagnostic = proof.get("diagnostics")
        if diagnostic is not None:
            diagnostic_path = _verify_asset(diagnostic, proof_path.parent, "continuous RGB diagnostics")
            data = _read_json(diagnostic_path)
            tissue["artifacts"].append(dict(role="virchow2_rgb_diagnostics", path=str(diagnostic_path)))
            for name, record in _mapping(data.get("raw_components"), "raw RGB components").items():
                if name not in {"rgb9", "completion19", "combined"}:
                    raise ExperimentError("Unexpected raw RGB component")
                relative = Path(_text(record.get("path"), "raw RGB component.path"))
                if relative.is_absolute() or not (diagnostic_path.parent / relative).resolve().is_relative_to(diagnostic_path.parent.resolve()):
                    raise ExperimentError("Raw RGB component must remain beside its diagnostic record")
                component_path = _verify_asset(record, diagnostic_path.parent, "raw RGB component")
                if image_shape(component_path) != (rgb_width, rgb_height):
                    raise ExperimentError("Raw RGB component image differs from the continuous field")
                tissue["artifacts"].append(dict(role="virchow2_rgb_raw_component", component=name, path=str(component_path)))


DENSE_RGB_BINDINGS = ("method", "source_frame_id", "image_sha256", "encoder_identity", "continuous_fit",
    "dense_readout", "window_local_fits", "readout", "array_bindings", "conversion", "support_png",
    "support_convention", "image_to_source_affine", "native_shape_yx", "native_mpp_xy", "rgb_shape_yx", "rgb_mpp_xy", "rgb_file")
DENSE_ARRAY_BINDINGS = ("component9_probability_sha256", "component19_probability_sha256", "input_probability_sha256",
    "output_probability_sha256", "guide_rgb_sha256", "support_bool_sha256")


def _dense_companion(record, proof_path, label):
    record = _mapping(record, label)
    raw = _text(record.get("path"), label + ".path")
    relative = Path(raw)
    if relative.is_absolute() or raw.startswith("~") or re.match(r"^[A-Za-z]:", raw) or any(part in {"", ".", ".."} for part in raw.split("/")) or "\\" in raw:
        raise ExperimentError("Dense RGB companion must use a contained relative path")
    path = (proof_path.parent / relative).resolve()
    if not path.is_relative_to(proof_path.parent.resolve()):
        raise ExperimentError("Dense RGB companion escapes its provenance directory")
    return _verify_asset(record, proof_path.parent, label)


def _recipe_payload(path, label):
    envelope = _read_json(path)
    state = _mapping(envelope.get("payload"), label + " payload")
    checksum = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if envelope.get("sha256") != checksum:
        raise ExperimentError(label + " payload checksum mismatch")
    return state


def _physical_equal(actual, expected):
    """Permit arithmetic-order roundoff only in physical geometry, not identity."""
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected) and
                all(_physical_equal(a, b) for a, b in zip(actual, expected)))
    return (type(actual) in (int, float) and type(expected) in (int, float) and
            math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))


def _validate_dense_rgb(tissue, artifact, root):
    """A new CPU readout with its own recipe; never relabel a frozen fit."""
    from PIL import Image, ImageChops
    image = tissue["image"]
    proof_path = _verify_asset(artifact["provenance"], root, "dense RGB provenance")
    proof = _read_json(proof_path)
    expected = dict(format="bora-dense-he-guided-rgb-v1", method="dense-he-guided-readout-v1", complete=True,
        source_frame_id=image["frame_id"], image_sha256=image["sha256"], continuous_fit=False, dense_readout=True,
        window_local_fits=False, full_RGB9_plus_19_applied=False, fallback_status="not-applicable-dense-readout",
        coordinate_space="native_image_pixel_edges", origin="top_left", label_kind="anonymous-KODAMA-coarse-groups",
        human_review_labels_used=False, expert_annotation_labels_used=False, semantic_tissue_labels_assigned=False,
        gpu_used=False, encoder_calls=0, semantic_classifier_calls=0, original_recipe_fit_repeated=False,
        support_convention="binary-255-supported")
    if any(proof.get(k) != value or (isinstance(value, bool) and type(proof.get(k)) is not bool) for k, value in expected.items()):
        raise ExperimentError("Dense RGB provenance differs from its registered source/frame/method")
    rgb_path = _verify_asset(artifact, root, "dense RGB image")
    width, height = image_shape(rgb_path)
    extent = _mapping(artifact.get("extent"), "dense RGB native extent")
    if set(extent) != {"x", "y", "width", "height"}:
        raise ExperimentError("Dense RGB extent requires x/y/width/height")
    for key in extent:
        _number(extent[key], "dense RGB extent." + key, positive=key in {"width", "height"})
    if extent["x"] < 0 or extent["y"] < 0 or extent["x"] + extent["width"] > image["width"] or extent["y"] + extent["height"] > image["height"]:
        raise ExperimentError("Dense RGB extent must remain inside the native image")
    scale = [extent["width"] / width, extent["height"] / height]
    affine = [[scale[0], 0, extent["x"]], [0, scale[1], extent["y"]], [0, 0, 1]]
    geometry = dict(native_shape_yx=[image["height"], image["width"]], native_mpp_xy=[image["mpp_x"], image["mpp_y"]],
        rgb_shape_yx=[height, width], rgb_mpp_xy=[image["mpp_x"] * scale[0], image["mpp_y"] * scale[1]],
        extent=extent, image_to_source_affine=affine)
    physical_keys = {"native_mpp_xy", "rgb_mpp_xy", "image_to_source_affine"}
    matches = lambda key, actual, expected: _physical_equal(actual, expected) if key in physical_keys else actual == expected
    if (proof.get("rgb_file_sha256") != sha256(rgb_path) or any(not matches(k, proof.get(k), value) for k, value in geometry.items()) or
            any(not matches(k, artifact.get(k), value) for k, value in dict(source_frame_id=image["frame_id"], image_sha256=image["sha256"],
                method=expected["method"], extent=extent, image_to_source_affine=affine, width=width, height=height).items())):
        raise ExperimentError("Dense RGB physical extent/affine differs from the native image")
    recorded_rgb = _dense_companion(proof.get("rgb_file"), proof_path, "dense RGB file binding")
    if recorded_rgb.resolve() != rgb_path.resolve() or proof["rgb_file"].get("sha256") != proof["rgb_file_sha256"]:
        raise ExperimentError("Dense RGB file binding differs from its registered artifact")
    readout = _mapping(proof.get("readout"), "dense RGB readout")
    weight = readout.get("component9_weight")
    if (type(weight) not in (int, float) or weight not in (1, .75, .5) or readout != dict(radius_um=24, epsilon_od2=.001,
            guide_space="optical-density-rgb", component9_weight=weight) or
            proof.get("conversion") != dict(conversion="dense H&E-guided probability readout", components=[9, 19], weights=[weight, 1 - weight])):
        raise ExperimentError("Dense RGB weights or H&E readout parameters differ from the versioned method")
    arrays = _mapping(proof.get("array_bindings"), "dense RGB array identities")
    if set(arrays) != set(DENSE_ARRAY_BINDINGS) or any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value) for value in arrays.values()):
        raise ExperimentError("Dense RGB requires the complete six lowercase SHA-256 numeric-array bindings")
    recipe = _dense_companion(proof.get("recipe"), proof_path, "dense readout recipe")
    source_recipe = _dense_companion(proof.get("source_recipe"), proof_path, "frozen source RGB recipe")
    support = _dense_companion(proof.get("support_png"), proof_path, "dense support PNG")
    if proof.get("recipe_sha256") != sha256(recipe):
        raise ExperimentError("Dense RGB readout recipe alias SHA-256 differs")
    state = _recipe_payload(recipe, "Dense RGB readout recipe")
    if (state.get("format") != "bora-dense-he-guided-readout-recipe-v1" or state.get("source_recipe") != proof["source_recipe"] or
            state.get("bindings") != {key: proof[key] for key in DENSE_RGB_BINDINGS}):
        raise ExperimentError("Dense RGB readout recipe is not bound to its exact provenance/arrays/source recipe")
    original = _recipe_payload(source_recipe, "Frozen source RGB recipe")
    source_bindings = original.get("provenance", {}).get("bindings", {})
    from virchow2_sharedcache import PREPROCESSING, canonical
    from virchow2_runtime import CHECKPOINT_SHA256
    encoder = dict(model_name="Virchow2", checkpoint_sha256=CHECKPOINT_SHA256,
        preprocessing_sha256=hashlib.sha256(canonical(PREPROCESSING)).hexdigest(), spatial_feature_dim=1280, tile_size_px=224)
    expected_source = dict(source_frame_id=image["frame_id"], image_sha256=image["sha256"], encoder_identity=encoder,
        method="continuous-deployment-head-v1", continuous_fit=True, global_fit=True, window_local_fits=False)
    if (proof.get("encoder_identity") != encoder or original.get("format") != "bora-continuous-rgb-v1" or
            original.get("method") != "continuous-deployment-head-v1" or
            original.get("continuous_fit") is not True or original.get("window_local_fits") is not False or
            any(source_bindings.get(k) != value for k, value in expected_source.items())):
        raise ExperimentError("Dense RGB frozen source recipe must retain its unchanged continuous fit and encoder/frame identity")
    with Image.open(rgb_path) as rgb, Image.open(support) as mask:
        if rgb.mode != "RGB" or mask.mode != "RGB" or rgb.size != mask.size:
            raise ExperimentError("Dense RGB and binary support must be aligned RGB8 images")
        channels = mask.split()
        if (ImageChops.difference(channels[0], channels[1]).getbbox() or ImageChops.difference(channels[0], channels[2]).getbbox() or
                any(value not in (0, 255) for _, value in channels[0].getcolors(maxcolors=3) or [(-1, -1)])):
            raise ExperimentError("Dense RGB support must have identical binary 0/255 RGB channels")
    return proof, recipe, source_recipe, support


def _discover_dense_rgb(run, tissue, proof_path, proof):
    from cellvit_models import image_frame
    image = tissue["image"]
    path = Path(image["path"])
    width, height = image_shape(path)
    frame = image_frame(path, path.parent / "shift.json")
    if "frame_id" in image and image["frame_id"] != frame["frame_id"]:
        raise ExperimentError("Dense RGB image identity conflicts with the registered source")
    image.update(frame_id=frame["frame_id"], sha256=sha256(path), width=width, height=height)
    rgb = _dense_companion(proof.get("rgb_file"), proof_path, "dense RGB image")
    rgb_width, rgb_height = image_shape(rgb)
    artifact = dict(role="virchow2_rgb_image", path=str(rgb), sha256=sha256(rgb), format="png",
        source_frame_id=image["frame_id"], image_sha256=image["sha256"], coordinate_space="native_image_pixel_edges", origin="top_left",
        width=rgb_width, height=rgb_height, extent=proof.get("extent"), image_to_source_affine=proof.get("image_to_source_affine"),
        method="dense-he-guided-readout-v1", full_RGB9_plus_19_applied=False, fallback_status="not-applicable-dense-readout",
        label="Virchow2 RGB (dense H&E-guided readout)", provenance=dict(path=str(proof_path), sha256=sha256(proof_path)))
    checked = copy.deepcopy(artifact)
    checked["path"] = os.path.relpath(rgb, run)
    checked["provenance"]["path"] = os.path.relpath(proof_path, run)
    _, recipe, source_recipe, support = _validate_dense_rgb(tissue, checked, run)
    tissue["artifacts"].extend([artifact, dict(role="virchow2_rgb_recipe", path=str(recipe)),
        dict(role="virchow2_rgb_source_recipe", path=str(source_recipe)), dict(role="virchow2_rgb_support", path=str(support)),
        dict(role="virchow2_rgb_provenance", path=str(proof_path))])


def _bora_group_contract(tissue, vector_path, mask_path, proof_path):
    """Check group identity and source linkage without importing an encoder."""
    image = tissue["image"]
    proof = _read_json(proof_path)
    expected = dict(format="cellphenotyper-anonymous-group-vector-v1", layer_kind="anonymous_refined_groups",
        label_kind="anonymous", source_frame_id=image["frame_id"], image_sha256=image["sha256"],
        native_shape_yx=[image["height"], image["width"]], native_mpp_xy=[image["mpp_x"], image["mpp_y"]],
        exact_input_raster_preserved=True, exact_geojson_raster_agreement=True,
        semantic_identity_is_not_cluster_identity=True)
    if any(proof.get(k) != value for k, value in expected.items()):
        raise ExperimentError("Bora refined-group provenance differs from anonymous/source-frame contract")
    for path, producer_name in ((vector_path, "groups.geojson"), (mask_path, "groups.ome.tif")):
        if _digest(proof.get("sha256", {}).get(producer_name), "Bora group asset digest") != sha256(path):
            raise ExperimentError("Bora refined-group source asset SHA-256 mismatch")
    ids = _array(proof.get("anonymous_group_ids"), "Bora anonymous group IDs")
    mapping = _mapping(proof.get("group_to_tissue_class"), "Bora group-to-tissue mapping")
    if (not ids or any(type(g) is not int or not 1 <= g <= 255 for g in ids) or len(set(ids)) != len(ids) or
            proof.get("anonymous_group_count") != len(ids) or set(mapping) != {str(g) for g in ids} or
            any(type(v) is not int or not 1 <= v <= 255 for v in mapping.values())):
        raise ExperimentError("Bora refined-group IDs/count/mapping are invalid")
    semantic_ids = _array(proof.get("semantic_class_ids"), "Bora semantic class IDs")
    if (any(type(g) is not int for g in semantic_ids) or len(set(semantic_ids)) != len(semantic_ids) or
            not set(semantic_ids).issubset(set(mapping.values())) or proof.get("semantic_class_count") != len(semantic_ids)):
        raise ExperimentError("Bora semantic class count must remain separate from anonymous group count")
    if image_shape(mask_path) != (image["width"], image["height"]):
        raise ExperimentError("Bora refined-group native raster dimensions differ")
    vector = _read_json(vector_path)
    if (vector.get("type") != "FeatureCollection" or vector.get("layer_kind") != "anonymous_refined_groups" or
            vector.get("label_kind") != "anonymous" or vector.get("source_frame_id") != image["frame_id"] or
            vector.get("image_sha256") != image["sha256"] or
            vector.get("native_shape_yx") != [image["height"], image["width"]] or
            vector.get("native_mpp_xy") != [image["mpp_x"], image["mpp_y"]] or
            vector.get("coordinate_space") != "native_image_pixel_edges" or vector.get("origin") != "top_left"):
        raise ExperimentError("Bora group vector has wrong layer/source/coordinate semantics")
    observed = []
    for feature in _array(vector.get("features"), "Bora group vector features"):
        prop = _mapping(feature.get("properties"), "Bora group feature properties")
        group = prop.get("anonymous_group_id")
        if (type(group) is not int or group not in ids or prop.get("value") != group or
                prop.get("label_kind") != "anonymous" or prop.get("semantic_tissue_class_id") != mapping[str(group)] or
                "predicted_tissue_id" in prop or "predicted_tissue" in prop):
            raise ExperimentError("Bora anonymous IDs must remain distinct from semantic tissue identities")
        observed.append(group)
    if len(observed) != len(ids) or set(observed) != set(ids):
        raise ExperimentError("Bora anonymous groups were dropped, duplicated or merged by tissue class")
    return proof


def _validate_bora_group_artifacts(tissue, root):
    artifacts = {a["artifact_id"]: a for a in tissue["artifacts"]}
    for artifact in tissue["artifacts"]:
        if artifact.get("role") != "bora_refined_groups":
            continue
        mask = artifacts.get(artifact.get("mask_artifact_id"))
        if mask is None or mask.get("role") != "bora_refined_groups_mask":
            raise ExperimentError("Bora refined groups require their separately registered raster artifact")
        proof_path = _verify_asset(artifact.get("provenance"), root, "Bora group provenance")
        proof = _bora_group_contract(tissue, _verify_asset(artifact, root, "Bora group vector"),
                                     _verify_asset(mask, root, "Bora group raster"), proof_path)
        for item in (artifact, mask):
            if (item.get("frame_id") != tissue["image"]["frame_id"] or item.get("label_kind") != "anonymous" or
                    item.get("source_frame_id") != tissue["image"]["frame_id"] or item.get("image_sha256") != tissue["image"]["sha256"] or
                    item.get("anonymous_group_ids") != proof["anonymous_group_ids"] or
                    item.get("anonymous_group_count") != proof["anonymous_group_count"] or
                    item.get("semantic_class_count") != proof["semantic_class_count"] or
                    item.get("group_to_tissue_class") != proof["group_to_tissue_class"]):
                raise ExperimentError("Bora group artifact metadata differs from its verified producer")


def _bora_local_polygon(geometry, width, height, label):
    """Check bounded two-dimensional GeoJSON polygons without loading image pixels."""
    geometry = _mapping(geometry, label)
    kind = geometry.get("type")
    if kind not in {"Polygon", "MultiPolygon"}:
        raise ExperimentError(f"{label} must be a Polygon or MultiPolygon")
    polygons = [geometry.get("coordinates")] if kind == "Polygon" else geometry.get("coordinates")
    if not isinstance(polygons, list) or not polygons:
        raise ExperimentError(f"{label} has no polygons")
    for polygon in polygons:
        if not isinstance(polygon, list) or not polygon:
            raise ExperimentError(f"{label} has no rings")
        for ring in polygon:
            if not isinstance(ring, list) or len(ring) < 4 or ring[0] != ring[-1]:
                raise ExperimentError(f"{label} has an open or incomplete ring")
            for point in ring:
                if (not isinstance(point, list) or len(point) != 2 or
                        any(type(value) not in (int, float) or not math.isfinite(value)
                            for value in point) or
                        not 0 <= point[0] <= width or not 0 <= point[1] <= height):
                    raise ExperimentError(f"{label} has a nonfinite or out-of-frame vertex")


def _validate_bora_local_geojson(path, image, role, model_sha256):
    if role == "bora_local_review_regions" and path.stat().st_size > LOCAL_BORA_MAX_REVIEW_BYTES:
        raise ExperimentError("Bora review-region GeoJSON exceeds the 2 MiB bounded queue limit")
    collection = _read_json(path)
    expected = dict(type="FeatureCollection", coordinate_space="native_image_pixel_edges", origin="top_left",
        source_frame_id=image["frame_id"], image_sha256=image["sha256"],
        native_shape_yx=[image["height"], image["width"]])
    if any(collection.get(key) != value for key, value in expected.items()):
        raise ExperimentError(f"{role} GeoJSON differs from the native image frame")
    if role == "bora_local_review_regions" and (
            collection.get("format") != "bora-local-review-regions-v1" or
            collection.get("model_sha256") != model_sha256 or
            collection.get("native_mpp_xy") != [image["mpp_x"], image["mpp_y"]] or
            collection.get("priority_score_is_calibrated_probability") is not False or
            collection.get("expert_annotations_used") is not False):
        raise ExperimentError("Bora review regions differ from the bounded, uncalibrated model contract")
    features = _array(collection.get("features"), f"{role}.features")
    if role == "bora_local_review_regions" and len(features) > LOCAL_BORA_MAX_REVIEW_REGIONS:
        raise ExperimentError("Bora review-region GeoJSON exceeds 1000 regions")
    seen = set()
    for index, feature in enumerate(features):
        label = f"{role}.features[{index}]"
        feature = _mapping(feature, label)
        if feature.get("type") != "Feature":
            raise ExperimentError(f"{label} must be a GeoJSON Feature")
        _bora_local_polygon(feature.get("geometry"), image["width"], image["height"], label)
        if role != "bora_local_review_regions":
            continue
        properties = _mapping(feature.get("properties"), f"{label}.properties")
        region_id = _identifier(properties.get("review_region_id"), f"{label}.review_region_id")
        if region_id in seen:
            raise ExperimentError("Bora review-region IDs must be unique")
        seen.add(region_id)
        score = _number(properties.get("priority_score"), f"{label}.priority_score")
        if not 0 <= score <= 1 or properties.get("reason") not in LOCAL_BORA_REVIEW_REASONS:
            raise ExperimentError("Bora review-region score/reason is invalid")
        rank = properties.get("priority_rank")
        if type(rank) is not int or rank < 1:
            raise ExperimentError("Bora review-region priority_rank must be a positive integer")
        summary = _mapping(properties.get("summary"), f"{label}.summary")
        required_summary = {"mean_tumor_probability", "mean_uncertainty", "tissue_support_fraction"}
        allowed_summary = required_summary | {"mean_cell_disagreement", "mean_boundary_disagreement"}
        if not required_summary.issubset(summary) or not set(summary).issubset(allowed_summary):
            raise ExperimentError("Bora review-region summary is missing or has unknown measures")
        for name, value in summary.items():
            value = _number(value, f"{label}.summary.{name}")
            if not 0 <= value <= 1:
                raise ExperimentError("Bora review-region summary values must be in [0,1]")


def _validate_bora_exact_vector(vector, mask_artifact, root, image, classes):
    """Require every native pixel centre in the exact vector to equal its source mask.

    This opt-in geometry is separate from the fitted review contour. A file
    digest or producer assertion alone cannot prove that the polygons describe
    the categorical image-channel raster, so compare all pixels in row blocks.
    """
    if (vector.get("format") != "geojson" or vector.get("source_mask_artifact_id") != mask_artifact["artifact_id"] or
            vector.get("source_mask_sha256") != mask_artifact["sha256"] or
            vector.get("derived_from_exact_mask") is not True or
            vector.get("geometry_simplified") is not False):
        raise ExperimentError("Bora exact vector must bind the unsimplified source mask artifact and SHA-256")
    path = _verify_asset(vector, root, "Bora exact semantic vector")
    collection = _read_json(path)
    expected = dict(type="FeatureCollection", coordinate_space="native_image_pixel_edges",
        origin="top_left", native_shape_yx=[image["height"], image["width"]],
        native_mpp_xy=[image["mpp_x"], image["mpp_y"]])
    if any(collection.get(key) != value for key, value in expected.items()):
        raise ExperimentError("Bora exact vector differs from the native image frame")
    # Older exact Bora exports did not repeat the source frame/hash in the
    # FeatureCollection. When present, both must agree; the artifact binding
    # above remains mandatory for every version.
    for key, value in (("source_frame_id", image["frame_id"]), ("image_sha256", image["sha256"])):
        if key in collection and collection[key] != value:
            raise ExperimentError(f"Bora exact vector {key} disagrees with the image")
    export = _mapping(collection.get("vector_export"), "Bora exact vector.vector_export")
    if (export.get("geometry_simplified") is not False or
            export.get("window_divisions_retained") is not False):
        raise ExperimentError("Bora exact vector must be unsimplified and globally dissolved")
    semantic = collection.get("semantic_boundary")
    if semantic is not None:
        semantic = _mapping(semantic, "Bora exact vector.semantic_boundary")
        if (semantic.get("role") != "native_exact" or semantic.get("geometry_simplified") is not False or
                semantic.get("predicted_not_expert_annotation") is not True):
            raise ExperimentError("Bora exact vector semantic-boundary metadata is incompatible")
    features = _array(collection.get("features"), "Bora exact vector.features")
    names = {entry["id"]: entry["name"] for entry in classes}
    if len(features) != len(names):
        raise ExperimentError("Bora exact vector must contain one native feature per semantic class")
    try:
        import numpy as np
        import shapely
        from shapely.geometry import shape
        import tifffile
        with tifffile.TiffFile(_verify_asset(mask_artifact, root, "Bora exact semantic mask")) as raster:
            labels = raster.pages[0].asarray()
        if labels.shape != (image["height"], image["width"]):
            raise ExperimentError("Bora exact vector source mask has changed native shape")
        observed = set(int(value) for value in np.unique(labels) if value)
        if observed != set(names):
            raise ExperimentError("Bora exact vector class IDs differ from its native mask")
        seen = set()
        for feature in features:
            if feature.get("type") != "Feature":
                raise ExperimentError("Bora exact vector requires GeoJSON Features")
            properties = _mapping(feature.get("properties"), "Bora exact vector.properties")
            value = properties.get("value")
            if (type(value) is not int or value not in names or value in seen or
                    properties.get("predicted_tissue_id") != value or
                    properties.get("name") != names[value] or
                    properties.get("predicted_tissue") != names[value] or
                    _mapping(properties.get("classification"), "Bora exact vector.classification").get("name") != names[value]):
                raise ExperimentError("Bora exact vector class IDs/names differ from prediction provenance")
            seen.add(value)
            geometry = feature.get("geometry")
            _bora_local_polygon(geometry, image["width"], image["height"], "Bora exact vector geometry")
            polygon = shape(geometry)
            if polygon.is_empty or not polygon.is_valid:
                raise ExperimentError("Bora exact vector contains invalid geometry")
            shapely.prepare(polygon)
            for y0 in range(0, image["height"], 256):
                y1 = min(y0 + 256, image["height"])
                yy, xx = np.mgrid[y0:y1, 0:image["width"]]
                inside = shapely.contains_xy(polygon, xx + .5, yy + .5)
                if not np.array_equal(inside, labels[y0:y1] == value):
                    raise ExperimentError("Bora exact vector does not rasterize to the native semantic mask")
        if seen != observed:
            raise ExperimentError("Bora exact vector omits a semantic class")
        if semantic is not None:
            raster_digest = hashlib.sha256(np.ascontiguousarray(labels).view(np.uint8)).hexdigest()
            if _digest(semantic.get("source_mask_sha256"), "Bora exact vector pixel digest") != raster_digest:
                raise ExperimentError("Bora exact vector pixel digest disagrees with its source mask")
    except (ImportError, OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ExperimentError):
            raise
        raise ExperimentError(f"Cannot validate Bora exact vector pixels: {exc}") from exc


def _validate_bora_local_artifacts(tissue, root):
    """Register a complete, opt-in native prediction without promoting review geometry."""
    image = tissue["image"]
    predictions = {}
    for artifact in tissue["artifacts"]:
        role = artifact.get("role")
        if role not in LOCAL_BORA_ROLES | LOCAL_BORA_OPTIONAL_ROLES:
            continue
        prediction_id = _identifier(artifact.get("prediction_id"), "Bora local prediction_id")
        bundle = predictions.setdefault(prediction_id, {})
        if role in bundle:
            raise ExperimentError(f"Duplicate {role} for Bora prediction {prediction_id}")
        bundle[role] = artifact
        expected = dict(source_frame_id=image["frame_id"], image_sha256=image["sha256"],
            native_shape_yx=[image["height"], image["width"]],
            coordinate_space="native_image_pixel_edges", origin="top_left")
        if any(artifact.get(key) != value for key, value in expected.items()):
            raise ExperimentError(f"{role} differs from the native source frame/image")
    if len(predictions) > 1:
        raise ExperimentError("A tissue may register only one Bora local prediction candidate at a time")
    for prediction_id, bundle in predictions.items():
        if not LOCAL_BORA_ROLES.issubset(bundle):
            missing = ", ".join(sorted(LOCAL_BORA_ROLES - set(bundle)))
            raise ExperimentError(f"Bora local prediction {prediction_id} is incomplete: {missing}")
        provenance_artifact = bundle["bora_local_prediction_provenance"]
        provenance_id = provenance_artifact["artifact_id"]
        for role, artifact in bundle.items():
            if artifact.get("provenance_artifact_id") != provenance_id:
                raise ExperimentError(f"{role} is not linked to its Bora provenance artifact")
            expected_format = "json" if role == "bora_local_prediction_provenance" else "geojson" if role in {
                "bora_local_review_geometry", "bora_local_review_regions", "bora_local_exact_semantic_vector"} else None
            if expected_format and artifact.get("format") != expected_format:
                raise ExperimentError(f"{role} has an unsupported format")
        if bundle["bora_local_exact_semantic_mask"].get("measurement_authority") is not True:
            raise ExperimentError("Bora exact semantic mask must be the measurement authority")
        for role in ("bora_local_review_geometry", "bora_local_review_regions"):
            if bundle[role].get("review_only") is not True:
                raise ExperimentError(f"{role} must be marked review_only")
        proof = _read_json(_verify_asset(provenance_artifact, root, "Bora local provenance"))
        expected = dict(format=LOCAL_BORA_PROVENANCE_FORMAT, prediction_id=prediction_id,
            source_frame_id=image["frame_id"], image_sha256=image["sha256"],
            native_shape_yx=[image["height"], image["width"]],
            native_mpp_xy=[image["mpp_x"], image["mpp_y"]],
            exact_semantic_mask_authoritative=True, review_geometry_non_authoritative=True,
            reference_used_for_inference=False)
        if any(proof.get(key) != value for key, value in expected.items()):
            raise ExperimentError("Bora local provenance differs from native frame/authority contract")
        model_sha = _digest(proof.get("local_model_sha256"), "Bora local model digest")
        _digest(proof.get("source_encoder_sha256"), "Bora source encoder digest")
        for key in ("rasterization_method", "boundary_refinement_method"):
            _text(proof.get(key), f"Bora local provenance.{key}")
        classes = _array(proof.get("class_order"), "Bora local class_order")
        class_ids = []
        for entry in classes:
            entry = _mapping(entry, "Bora local class")
            class_id = entry.get("id")
            if type(class_id) is not int or not 1 <= class_id <= 65535:
                raise ExperimentError("Bora local class IDs must be positive uint16 labels")
            class_ids.append(class_id)
            _text(entry.get("name"), "Bora local class name")
        if len(class_ids) < 2 or len(class_ids) != len(set(class_ids)) or proof.get("probability_target_class_id") not in class_ids:
            raise ExperimentError("Bora local class order/probability target is invalid")
        declared_hashes = _mapping(proof.get("artifact_sha256"), "Bora local asset digests")
        required_hashes = LOCAL_BORA_ROLES - {"bora_local_prediction_provenance"}
        expected_hashes = required_hashes | (LOCAL_BORA_OPTIONAL_ROLES if
            "bora_local_exact_semantic_vector" in bundle else set())
        if set(declared_hashes) != expected_hashes:
            raise ExperimentError("Bora local provenance must bind every registered prediction asset")
        for role, artifact in bundle.items():
            if role == "bora_local_prediction_provenance":
                continue
            path = _verify_asset(artifact, root, role)
            if _digest(declared_hashes.get(role), f"{role} proof digest") != artifact["sha256"]:
                raise ExperimentError(f"{role} differs from the prediction provenance")
            if role == "bora_local_exact_semantic_vector":
                continue
            if role in LOCAL_BORA_RASTER_ROLES:
                if artifact.get("format") not in {"tiff", "ome-tiff"} or path.suffix.lower() not in {".tif", ".tiff"}:
                    raise ExperimentError(f"{role} must be a TIFF/OME-TIFF raster")
                try:
                    import tifffile
                    with tifffile.TiffFile(path) as raster:
                        page = raster.pages[0]
                        dtype = page.dtype
                        if (page.imagewidth != image["width"] or page.imagelength != image["height"] or
                                page.samplesperpixel != 1 or
                                (role == "bora_local_exact_semantic_mask" and (dtype.kind != "u" or dtype.itemsize > 2)) or
                                (role != "bora_local_exact_semantic_mask" and dtype.kind != "f")):
                            raise ExperimentError(f"{role} has wrong native shape/channels/dtype")
                except (OSError, ValueError, IndexError) as exc:
                    raise ExperimentError(f"Cannot inspect {role} TIFF header: {exc}") from exc
            else:
                _validate_bora_local_geojson(path, image, role, model_sha)
        if "bora_local_exact_semantic_vector" in bundle:
            _validate_bora_exact_vector(bundle["bora_local_exact_semantic_vector"],
                bundle["bora_local_exact_semantic_mask"], root, image, classes)


def _discover_bora_groups(run, tissue):
    for proof_path in sorted((run / "15_cluster_geojson" / tissue["name"]).glob("**/GROUPS_VECTOR_COMPLETE.json")):
        from cellvit_models import image_frame
        image = tissue["image"]
        image_path = Path(image["path"])
        width, height = image_shape(image_path)
        frame = image_frame(image_path, image_path.parent / "shift.json")
        if "frame_id" in image and image["frame_id"] != frame["frame_id"]:
            raise ExperimentError("Bora groups disagree with the registered image frame")
        image.update(frame_id=frame["frame_id"], sha256=sha256(image_path), width=width, height=height)
        vector, mask = proof_path.parent / "groups.geojson", proof_path.parent / "groups.ome.tif"
        proof = _bora_group_contract(tissue, vector, mask, proof_path)
        aid = "bora-groups-" + sha256(proof_path)[:20]
        metadata = dict(frame_id=image["frame_id"], source_frame_id=image["frame_id"],
            image_sha256=image["sha256"], coordinate_space="native_image_pixel_edges", origin="top_left",
            label_kind="anonymous", anonymous_group_ids=proof["anonymous_group_ids"],
            anonymous_group_count=proof["anonymous_group_count"], semantic_class_count=proof["semantic_class_count"],
            group_to_tissue_class=proof["group_to_tissue_class"])
        tissue["artifacts"].extend([
            dict(role="bora_refined_groups", artifact_id=aid, path=str(vector), format="geojson",
                 label="Bora refined groups (anonymous)", mask_artifact_id=aid + ":mask",
                 provenance=dict(path=str(proof_path), sha256=sha256(proof_path)), **metadata),
            dict(role="bora_refined_groups_mask", artifact_id=aid + ":mask", path=str(mask), format="ome-tiff",
                 label="Bora refined groups (anonymous raster)", **metadata),
            dict(role="bora_refined_groups_provenance", artifact_id=aid + ":provenance", path=str(proof_path))])


def _durable_cell_image(run, sample, source_image):
    image = _select_source_image((run / "03_stardist" / sample).glob("**/crop_roi.tif"), sample,
        source_sha256=_digest(source_image.get("sha256"), f"{sample}.cell source image SHA-256"),
        shape=(source_image.get("width"), source_image.get("height")),
        mpp=(source_image.get("mpp_x"), source_image.get("mpp_y")), crop=True, require_calibration=True)
    shift_path = _path("shift.json", image.parent, f"{sample}.durable image calibration")
    calibration, _ = _crop_calibration(image, sample, required=True)
    width, height = image_shape(image)
    return dict(path=str(image), sha256=sha256(image), width=width, height=height, **calibration,
                coordinate_convention=COORDINATE_CONVENTION, coordinate_space="crop-native-pixels", origin_x=0, origin_y=0), shift_path


def discover_run(run: Path) -> list[dict]:
    """Union checked tissue embeddings and cell-identification exports by sample."""
    run = Path(os.path.abspath(run))
    contract_path = run / "00_execution" / "analysis_contract.json"
    selected_detector = None
    if contract_path.is_file():
        contract = _read_json(contract_path)
        selected_detector = contract.get("scientific_route", {}).get("cell_detection_mode")
    tissues = _discover_virchow_tissues(run)
    by_sample = {}
    for tissue in tissues:
        if tissue["name"] in by_sample:
            raise ExperimentError(f"Duplicate Virchow2 exports for sample {tissue['name']}; use --spec to disambiguate")
        by_sample[tissue["name"]] = tissue
    cell_samples = set()
    # A StarDist run must not pick up stale CellViT files left by an older run
    # in the same output tree and present them as the selected detector's cells.
    cell_outputs = [] if selected_detector == "stardist" else _cell_producer_outputs(run)
    for sample, classification_path in cell_outputs:
        if sample in cell_samples:
            raise ExperimentError(f"Duplicate CellViT classification exports for sample {sample}; use --spec to disambiguate")
        cell_samples.add(sample)
        classification = _read_json(classification_path)
        source_image = _mapping(classification.get("image"), f"{sample}.classification.image")
        image, shift_path = _durable_cell_image(run, sample, source_image)
        image["frame_id"] = source_image.get("frame_id")
        _classification_registration(source_image, image)
        manifest_path = _path("cellvit_cell_features.manifest.json", classification_path.parent, f"{sample}.cell feature manifest")
        tissue = by_sample.get(sample)
        if tissue is None:
            tissue = dict(name=sample, image=image, artifacts=[], feature_sets=[], provenance={})
            tissues.append(tissue)
            by_sample[sample] = tissue
        else:
            existing = tissue["image"]
            if (sha256(Path(existing["path"])) != image["sha256"] or image_shape(Path(existing["path"])) != (image["width"], image["height"]) or
                    existing["mpp_x"] != image["mpp_x"] or existing["mpp_y"] != image["mpp_y"]):
                raise ExperimentError(f"{sample}: Virchow2 and cell exports do not share the same durable calibrated image")
            existing.update(image)
        _validate_classification_payload(classification, classification_path, tissue)
        _validate_cell_feature_manifest(manifest_path, classification)
        tissue["cell_classification"] = dict(path=str(classification_path), sha256=sha256(classification_path),
            feature_manifest=dict(path=str(manifest_path), sha256=sha256(manifest_path)))
        reviews = [path for folder in SEGMENTATION_REVIEW_FOLDERS
                   for path in (run / folder / sample).glob("**/segmentation_qc.json") if path.is_file()]
        if reviews:
            review = _single(reviews, f"segmentation review for {sample}")
            tissue["cell_classification"]["segmentation_review"] = dict(path=str(review), sha256=sha256(review))
        tissue["provenance"].update(cell_classification_sha256=sha256(classification_path),
            cell_image_identity_verification="Exact durable crop bytes, dimensions, independent shift.json calibration and native coordinates match the producer",
            cell_image_calibration_sha256=sha256(shift_path))
    for tissue in tissues:
        if selected_detector is not None:
            tissue.setdefault("provenance", {}).update(
                cell_detection_mode=selected_detector,
                cell_detection_contract_sha256=sha256(contract_path))
        _discover_continuous_rgb(run, tissue)
        _discover_bora_groups(run, tissue)
    return tissues


def export_runs(runs, output, *, name="CellPhenotyper experiment", overwrite=False):
    tissues = list(itertools.chain.from_iterable(discover_run(Path(run)) for run in runs))
    if not tissues:
        raise ExperimentError("No complete Virchow2 or CellViT cell-identification exports found; use --spec to register existing images/caches explicitly")
    return export_experiment(dict(name=name, settings={}, provenance={"producer": "CellPhenotyper", "inference_started": False},
                                  tissues=tissues), Path(output), overwrite=overwrite)


def try_export_run(run: Path, *, name="CellPhenotyper experiment") -> dict:
    """Keep legacy reports available and publish explicit export availability."""
    destination = Path(run) / "00_execution" / FILENAME
    try:
        payload = export_runs([run], destination, name=name, overwrite=True)
        status = dict(status="available", path=FILENAME, experiment_id=payload["experiment_id"],
                      tissue_count=len(payload["tissues"]), inference_started=False)
    except Exception as exc:
        has_producer = bool(list((Path(run) / "09_embeddings").glob("**/VIRCHOW2_COMPLETE.json")) or _cell_producer_outputs(Path(run)))
        status = dict(status="error" if has_producer else "unavailable",
                      path=None, error=str(exc), inference_started=False,
                      previous_index_retained=destination.exists())
    atomic_json(destination.parent / "experiment_export_status.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create or validate a wsiTools experiment from existing images, marker outputs and feature caches. Never starts inference.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", action="append", type=Path, help="Existing complete Virchow2 or CellViT output root; repeat to combine runs")
    source.add_argument("--spec", type=Path, help="Explicit JSON asset specification; paths are relative to this file")
    source.add_argument("--validate", type=Path, help="Verify an existing experiment and every asset/hash/feature row")
    parser.add_argument("--output", type=Path, help="Destination experiment.cellphenotyper.json")
    parser.add_argument("--name", default="CellPhenotyper experiment", help="Friendly experiment name for --run")
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace an existing experiment index only")
    args = parser.parse_args(argv)
    try:
        if args.validate:
            payload = validate_experiment(args.validate)
            print(f"Valid experiment: {payload['name']} ({len(payload['tissues'])} tissues)")
            return 0
        if not args.output:
            parser.error("--output is required when exporting")
        if args.spec:
            if args.spec.absolute() == args.output.absolute():
                raise ExperimentError("Output must differ from the source specification")
            payload = export_experiment(_read_json(args.spec), args.output, base_dir=args.spec.parent, overwrite=args.overwrite)
        else:
            payload = export_runs(args.run, args.output, name=args.name, overwrite=args.overwrite)
        print(f"Ready to open in wsiTools: {args.output}\n{payload['name']} — {len(payload['tissues'])} tissues. Images and feature caches remain external; no inference was started.")
        return 0
    except (ExperimentError, OSError, ValueError) as exc:
        action = "validate" if args.validate else "export"
        print(f"Cannot {action} experiment: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
