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
                  "patient_id", "split", "observation_type", "area_px"}
SPATIAL_ROLES = {"spatial_transcriptomics_object": "rds", "spatial_review_bundle": "json"}
SPATIAL_BUNDLE_SCHEMA = "cellphenotyper-spatial-review-bundle"
SPATIAL_MAX_BYTES = 16 * 1024 * 1024
SPATIAL_MAX_OBSERVATIONS = 10000
GIGATIME_SCORE_SEMANTICS = "uncalibrated_virtual_marker_score"
GIGATIME_ROLES = {"gigatime_marker_image", "cell_segmentation_mask", "gigatime_cell_scores"}


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
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
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
    seen = set()
    try:
        with _csv(path) as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            if len(set(header)) != len(header) or any(key not in header for key in [*keys, *columns]):
                raise ExperimentError(f"{label}: CSV has missing or duplicate declared columns")
            for line, row in enumerate(reader, 2):
                identifier = row[keys[0]]
                if not identifier or not identifier.strip() or identifier in seen:
                    raise ExperimentError(f"{label}: missing or duplicate observation ID at CSV line {line}")
                seen.add(identifier)
                if None in row or any(row[key] is None for key in [*keys, *columns]):
                    raise ExperimentError(f"{label}: malformed CSV row at line {line}")
                values = [float(row[key]) for key in [keys[1], keys[2], *columns]]
                if not all(math.isfinite(value) for value in values):
                    raise ExperimentError(f"{label}: nonfinite coordinates/features at CSV line {line}")
                if not (0 <= values[0] < image["width"] and 0 <= values[1] < image["height"]):
                    raise ExperimentError(f"{label}: observation centre outside image at CSV line {line}")
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


def _register_definition(definitions, feature, image):
    """A shared feature ID denotes the same representation across tissues."""
    grid = feature["grid"]
    physical_grid = [round(grid[key] * image[mpp], 6) for key, mpp in
                     (("tile_width_px", "mpp_x"), ("tile_height_px", "mpp_y"),
                      ("spacing_x_px", "mpp_x"), ("spacing_y_px", "mpp_y"))]
    signature = json.dumps(dict(kind=feature["kind"], observation_unit=feature["observation_unit"],
                                feature_columns=feature["feature_columns"], encoder=feature["encoder"],
                                preprocessing=feature["preprocessing"], physical_grid=physical_grid),
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
    _validate_spatial_artifacts(payload, path.parent)
    _validate_gigatime_artifacts(payload, path.parent)
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
        image.setdefault("frame_id", f"{tid}:image")
        image["coordinate_convention"] = COORDINATE_CONVENTION
        result = copy.deepcopy(tissue)
        result.update(tissue_id=tid, name=name, image=image, artifacts=[], feature_sets=[])
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
            item.setdefault("artifact_id", f"{tid}:artifact-{ai + 1}")
            item.setdefault("frame_id", image["frame_id"])
            result["artifacts"].append(item)
        for fi, feature in enumerate(_array(tissue.get("feature_sets", []), f"{label}.feature_sets")):
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
    _validate_spatial_artifacts(payload, output.parent)
    _validate_gigatime_artifacts(payload, output.parent, fill=True)
    atomic_json(output, payload, overwrite=overwrite)
    return payload


def _single(paths, label):
    paths = sorted(set(path for path in paths if path.is_file()))
    if len(paths) != 1:
        raise ExperimentError(f"Expected one {label}, found {len(paths)}; use --spec to disambiguate")
    return paths[0]


def discover_run(run: Path) -> list[dict]:
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
        crops = list((run / "03_stardist" / sample).glob("**/crop_roi.tif"))
        candidates = crops or [Path(metadata.get("image", ""))]
        if not crops and not candidates[0].is_file():
            candidates = list((run / "01_input" / sample).glob("*.ome.tif*"))
        image = _single(candidates, f"calibrated image for {sample}")
        if image_shape(image) != (metadata.get("image_width_px"), metadata.get("image_height_px")):
            raise ExperimentError(f"{sample}: grid and image dimensions disagree")
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
                                      image_identity_verification="Current image bytes and header verified; the legacy encoder completion record contains no source-image digest"),
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
                                                   origin_x_px=0, origin_y_px=0))])
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
        tissues.append(tissue)
    return tissues


def export_runs(runs, output, *, name="CellPhenotyper experiment", overwrite=False):
    tissues = list(itertools.chain.from_iterable(discover_run(Path(run)) for run in runs))
    if not tissues:
        raise ExperimentError("No complete Virchow2 feature exports found; use --spec to register existing images/caches explicitly")
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
        status = dict(status="unavailable" if not list((Path(run) / "09_embeddings").glob("**/VIRCHOW2_COMPLETE.json")) else "error",
                      path=None, error=str(exc), inference_started=False,
                      previous_index_retained=destination.exists())
    atomic_json(destination.parent / "experiment_export_status.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create or validate a wsiTools experiment from existing images, marker outputs and feature caches. Never starts inference.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", action="append", type=Path, help="Existing complete Virchow2 output root; repeat to combine runs")
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
