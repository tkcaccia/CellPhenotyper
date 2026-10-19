#!/usr/bin/env python3
"""Export or validate portable CellPhenotyper experiments from existing files.

No encoder, workflow engine, network client or GPU library is imported here.
"""
from __future__ import annotations

import argparse
import copy
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
from pathlib import Path


SCHEMA = "cellphenotyper-experiment"
VERSION = 1
FILENAME = "experiment.cellphenotyper.json"
FEATURE_WIDTHS = {"virchow2-pooled": ("tile", 2560), "virchow2-spatial": ("token", 1280)}
COORDINATE_CONVENTION = "pixel-edge-origin_centres-in-native-pixels"
NON_PREDICTORS = {"label", "class", "class_id", "cluster", "sample_id", "tissue_id", "specimen_id",
                  "patient_id", "split", "observation_type", "area_px"}


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
    if path == destination:
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
    return (gzip.open if path.name.endswith(".gz") else open)(path, "rt", newline="", encoding="utf-8-sig")


def _validate_features(feature, path, image, label):
    kind = feature.get("kind")
    if kind not in FEATURE_WIDTHS:
        raise ExperimentError(f"{label}.kind must be virchow2-pooled or virchow2-spatial")
    unit, width = FEATURE_WIDTHS[kind]
    if feature.get("observation_unit") != unit:
        raise ExperimentError(f"{label}: {kind} requires observation_unit={unit}")
    expected_format = "csv.gz" if path.name.endswith(".csv.gz") else "csv" if path.suffix == ".csv" else None
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


def atomic_json(path: Path, payload: dict):
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
        os.replace(temporary, path)
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
        result = dict(tissue_id=tid, name=name, image=image, artifacts=[], feature_sets=[])
        for key in ("specimen_id", "provenance"):
            if key in tissue:
                result[key] = tissue[key]
        if "specimen_id" in result:
            _text(result["specimen_id"], f"{label}.specimen_id")
        if "provenance" in result:
            _mapping(result["provenance"], f"{label}.provenance")
        for ai, artifact in enumerate(_array(tissue.get("artifacts", []), f"{label}.artifacts")):
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
    payload = dict(schema=SCHEMA, schema_version=VERSION,
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
    atomic_json(output, payload)
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
    parser = argparse.ArgumentParser(description="Create a wsiTools-ready experiment from existing CellPhenotyper images and cached features. Never starts inference.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", action="append", type=Path, help="Existing output root; repeat to combine runs")
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
        print(f"Cannot export experiment: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
