#!/usr/bin/env python3
"""Portable, completion-last reference mappings; no atlas/runtime ML imports.

Receipts bind bytes, not a biological truth claim. The copied atlas manifest is
metadata only: consumers never follow its upstream paths or load atlas arrays.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


FORMAT = "cellphenotyper_reference_mapping"
SCHEMA_VERSION = "1.0.0"
IDENTITY = ["observation_uid", "sample_id", "observation_id"]
REFERENCE_COLUMNS = ["reference_assignment", "reference_status", "reference_distance",
                     "reference_radius", "reference_margin", "reference_nearest_group",
                     "reference_atlas_id"]
NUMERIC_COLUMNS = ["reference_distance", "reference_radius", "reference_margin"]
FORMATS = [("cell_profiles", ["cell_uid", "sample_id", "cell_id"], "cell"),
           ("region_profiles", ["region_uid", "sample_id", "region_id"], "tissue_region"),
           ("observation_profiles", IDENTITY, None)]
ATLAS_IDENTITY_KEYS = ["schema_version", "version", "feature_groups", "observation_unit",
    "feature_schemas", "label_column", "label_scope", "source_profiles", "descriptions",
    "threshold_quantile", "minimum_group_size", "representatives_per_group"]


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _json(data):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    return json.loads(data, parse_constant=reject, object_pairs_hook=unique)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _inside(root, relative, *, basename=False):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Artifact path must be a safe relative path")
    value = Path(relative)
    if value.is_absolute() or any(part in (".", "..") for part in relative.split("/")):
        raise ValueError("Artifact path escapes its bundle")
    if basename and (value.name != relative or relative in (".", "..")):
        raise ValueError("Receipt artifact paths must be safe sibling basenames")
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError(f"Missing or out-of-bundle artifact: {relative}")
    return candidate


def _file(root, relative):
    path = _inside(root, relative)
    return {"path": relative, "sha256": _sha256(path)}


def _csv_shape(path):
    # pandas may silently fill short rows or rename repeated headers.
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            columns = next(reader)
        except StopIteration:
            raise ValueError("Empty CSV artifact") from None
        if not columns or len(set(columns)) != len(columns) or any(not c for c in columns):
            raise ValueError("CSV has empty or duplicate columns")
        count = 0
        for row in reader:
            if len(row) != len(columns):
                raise ValueError("CSV row width differs from its schema")
            count += 1
    return columns, count


def _profile_snapshot(directory, atlas):
    root = Path(directory).resolve()
    matches = [entry for entry in FORMATS if (root / (entry[0] + "_manifest.json")).exists()]
    if len(matches) != 1:
        raise ValueError("Expected exactly one source profile manifest")
    stem, source_identity, expected_unit = matches[0]
    manifest_file = _file(root, stem + "_manifest.json")
    manifest = _json(_inside(root, manifest_file["path"]).read_bytes())
    unit = manifest.get("observation_unit", "cell" if stem == "cell_profiles" else None)
    if unit not in ("cell", "tissue_region") or (expected_unit and unit != expected_unit) or unit != atlas["observation_unit"]:
        raise ValueError("Reference mapping observation_unit differs from source profile/atlas")
    table_name = stem + (".csv" if (root / (stem + ".csv")).is_file() else ".parquet")
    table_file = _file(root, table_name)
    path = _inside(root, table_name)
    if path.suffix == ".csv":
        columns, count = _csv_shape(path)
        frame = pd.read_csv(path, dtype={key: str for key in source_identity},
                            keep_default_na=False, float_precision="round_trip")
    else:
        frame = pd.read_parquet(path)
        columns, count = list(frame.columns), len(frame)
    if len(set(columns)) != len(columns) or not set(source_identity) <= set(columns):
        raise ValueError("Source profile has invalid identity columns")
    if set(REFERENCE_COLUMNS) & set(columns):
        raise ValueError("Source profile already contains reference-assignment columns")
    for key in source_identity:
        values = frame[key]
        if values.isna().any() or values.astype(str).str.strip().eq("").any() or values.astype(str).str.contains(r"[\x00-\x1f\x7f]", regex=True).any():
            raise ValueError(f"Invalid source profile identity: {key}")
        frame[key] = values.astype(str)
    for source_key, normalized in zip(source_identity, IDENTITY):
        if normalized in frame and not frame[normalized].equals(frame[source_key]):
            raise ValueError("Source profile contains conflicting normalized identity aliases")
        frame[normalized] = frame[source_key]
    if "observation_unit" in frame and not frame.observation_unit.eq(unit).all():
        raise ValueError("Source profile contains contradictory observation_unit")
    frame["observation_unit"] = unit
    if frame.observation_uid.duplicated().any() or frame.duplicated(["sample_id", "observation_id"]).any():
        raise ValueError("Source profile contains duplicate observation identities")
    if count != len(frame) or count != manifest.get("observation_count", manifest.get("cell_count", manifest.get("region_count", count))):
        raise ValueError("Source profile row count differs from manifest")
    declared = manifest.get("files", {}).get(table_name)
    if declared is not None and declared != table_file["sha256"]:
        raise ValueError("Source profile table hash differs from manifest")
    rows_file = None
    if (root / "feature_rows.csv").exists():
        rows_file = _file(root, "feature_rows.csv")
        _csv_shape(_inside(root, "feature_rows.csv"))
        rows = pd.read_csv(_inside(root, "feature_rows.csv"), dtype=str, keep_default_na=False)
        if not set(source_identity) <= set(rows) or not rows[source_identity].equals(frame[source_identity]):
            raise ValueError("Source feature_rows identities/order differ from profile")
        declared = manifest.get("files", {}).get("feature_rows.csv")
        if declared is not None and declared != rows_file["sha256"]:
            raise ValueError("Source feature_rows hash differs from manifest")
    blocks, available = {}, np.ones(count, dtype=bool)
    for group in atlas["feature_groups"]:
        definition = manifest.get("feature_blocks", {}).get(group)
        if not isinstance(definition, dict) or definition.get("reference_compatible") is False:
            raise ValueError(f"Missing or incompatible source feature group: {group}")
        block_file = _file(root, definition.get("path"))
        if definition.get("sha256") is not None and definition["sha256"] != block_file["sha256"]:
            raise ValueError(f"Source feature block hash differs from manifest: {group}")
        values = np.load(_inside(root, block_file["path"]), mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[0] != count or values.shape[1] < 1 or values.dtype.kind not in "fiu":
            raise ValueError(f"Invalid source feature block geometry/dtype: {group}")
        if definition.get("shape", list(values.shape)) != list(values.shape):
            raise ValueError(f"Source feature block shape differs from manifest: {group}")
        schema = {"observation_unit": unit, "dimension": int(values.shape[1]),
                  "feature_names": definition.get("feature_names"), "feature_definition": definition.get("feature_definition")}
        if not schema["feature_definition"] or _canonical(schema) != _canonical(atlas["feature_schemas"][group]):
            raise ValueError(f"Source feature schema differs from frozen atlas: {group}")
        for start in range(0, count, 8192):
            available[start:start + 8192] &= np.isfinite(values[start:start + 8192]).all(axis=1)
        blocks[group] = block_file
    identity_columns = list(dict.fromkeys(source_identity + IDENTITY + ["observation_unit"]))
    record = {"format": stem, "observation_unit": unit, "observation_count": count,
              "identity_columns": identity_columns, "source_columns": columns,
              "normalized_columns": list(frame.columns), "manifest": manifest_file,
              "table": table_file, "feature_rows": rows_file, "feature_blocks": blocks}
    _check_profile_hashes(root, record)
    return frame, record, available


def _check_profile_hashes(root, record):
    files = [record["manifest"], record["table"], *record["feature_blocks"].values()]
    if record["feature_rows"] is not None:
        files.append(record["feature_rows"])
    elif (Path(root) / "feature_rows.csv").exists():
        raise ValueError("Source profile changed: feature_rows appeared")
    for item in files:
        if _sha256(_inside(Path(root), item["path"])) != item["sha256"]:
            raise ValueError(f"Source profile changed: {item['path']}")


def _validate_atlas(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "1.0.0":
        raise ValueError("Unsupported frozen reference atlas schema")
    if manifest.get("observation_unit") not in ("cell", "tissue_region"):
        raise ValueError("Invalid frozen reference atlas observation_unit")
    groups = manifest.get("feature_groups")
    if not isinstance(groups, list) or not groups or not all(isinstance(g, str) and g for g in groups) or len(set(groups)) != len(groups):
        raise ValueError("Invalid frozen atlas feature groups")
    if set(manifest.get("feature_schemas", {})) != set(groups):
        raise ValueError("Frozen atlas feature schemas differ from selected groups")
    try:
        identity = {key: manifest[key] for key in ATLAS_IDENTITY_KEYS}
        if "marker_filter_contracts" in manifest:
            identity["marker_filter_contracts"] = manifest["marker_filter_contracts"]
        expected_id = hashlib.sha256(_canonical(identity).encode()).hexdigest()[:24]
    except KeyError as exc:
        raise ValueError("Incomplete frozen atlas identity") from exc
    if manifest.get("atlas_id") != expected_id:
        raise ValueError("Frozen reference atlas ID does not match its identity")
    records = manifest.get("groups")
    if not isinstance(records, list) or not records:
        raise ValueError("Frozen reference atlas has no groups")
    names = []
    for record in records:
        name, radius = record.get("reference_group"), record.get("acceptance_radius")
        if not isinstance(name, str) or not name or name == "unknown":
            raise ValueError("Invalid frozen atlas group")
        if radius is not None and (isinstance(radius, bool) or not isinstance(radius, (float, int)) or not np.isfinite(radius) or radius < 0):
            raise ValueError("Invalid frozen atlas acceptance radius")
        if record.get("assignment_supported") != (radius is not None):
            raise ValueError("Contradictory frozen atlas assignment support")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate frozen atlas group")


def _atlas_snapshot(directory):
    root = Path(directory).resolve()
    manifest_path = _inside(root, "atlas_manifest.json")
    data = manifest_path.read_bytes()
    hash_value = hashlib.sha256(data).hexdigest()
    if _inside(root, "atlas_manifest.sha256").read_text().strip() != hash_value:
        raise ValueError("Immutable atlas manifest changed")
    manifest = _json(data)
    _validate_atlas(manifest)
    for name, expected in manifest["files"].items():
        if _sha256(_inside(root, name)) != expected:
            raise ValueError(f"Immutable atlas artifact changed: {name}")
    if _sha256(manifest_path) != hash_value:
        raise ValueError("Immutable atlas manifest changed during capture")
    return manifest, data


def _atlas_record(manifest, data, name):
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest(),
            **{key: manifest[key] for key in ("atlas_id", "schema_version", "version", "observation_unit", "feature_groups")}}


def _column_schema(columns, identity):
    return [{"name": name, "role": "reference" if name in REFERENCE_COLUMNS else "identity" if name in identity else "source",
             "type": "float64_or_nan" if name in NUMERIC_COLUMNS else "string" if name in REFERENCE_COLUMNS or name in identity else "source_profile"}
            for name in columns]


def _validate_interpretations(frame, atlas, available):
    records = {record["reference_group"]: record for record in atlas["groups"]}
    statuses = {"assigned", "missing_features", "insufficient_reference", "outside_reference", "ambiguous_reference"}
    if not set(frame.reference_status) <= statuses or not frame.reference_atlas_id.eq(atlas["atlas_id"]).all():
        raise ValueError("Invalid reference status or foreign reference atlas ID")
    for key in NUMERIC_COLUMNS:
        try:
            # Python's decimal conversion preserves CSV float64 round trips;
            # pandas.to_numeric can move a valid radius by one ULP.
            frame[key] = np.asarray([float(value) if value != "" else np.nan for value in frame[key]], dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid reference numeric field: {key}") from exc
        if np.isinf(frame[key].to_numpy()).any():
            raise ValueError(f"Nonfinite reference numeric field: {key}")
    for row, complete in zip(frame.itertuples(index=False), available):
        assignment, status, nearest = row.reference_assignment, row.reference_status, row.reference_nearest_group
        distance, radius, margin = row.reference_distance, row.reference_radius, row.reference_margin
        if status == "missing_features":
            if complete or assignment != "unknown" or nearest != "" or not np.isnan([distance, radius, margin]).all():
                raise ValueError("Contradictory missing_features reference assignment")
            continue
        if not complete or nearest not in records or not np.isfinite(distance) or distance < 0:
            raise ValueError("Invalid reference nearest group/distance or feature availability")
        expected_radius = records[nearest]["acceptance_radius"]
        if (len(records) == 1 and not np.isnan(margin)) or (len(records) > 1 and (not np.isfinite(margin) or margin < 0)):
            raise ValueError("Invalid reference margin")
        if expected_radius is None:
            expected_status = "insufficient_reference"
            if not np.isnan(radius):
                raise ValueError("Unsupported reference group must have unknown radius")
        else:
            if not np.isfinite(radius) or radius != expected_radius:
                raise ValueError("Reference radius differs from frozen atlas")
            expected_status = ("outside_reference" if distance > radius + 1e-7 else
                               "ambiguous_reference" if len(records) > 1 and margin <= 1e-7 else "assigned")
        if status != expected_status or assignment != (nearest if status == "assigned" else "unknown"):
            raise ValueError("Reference assignment/status is inconsistent with frozen atlas thresholds")


def _read_assignments(path, record, source, source_record, atlas, available):
    columns, count = _csv_shape(path)
    expected_columns = list(source.columns) + REFERENCE_COLUMNS
    identity = source_record["identity_columns"]
    if columns != expected_columns or record.get("columns") != columns or record.get("row_count") != count or count != len(source):
        raise ValueError("Reference assignment schema/order/count differs from source profile")
    if record.get("column_schema") != _column_schema(columns, identity):
        raise ValueError("Reference assignment column schema differs from contract")
    pieces, offset = [], 0
    with pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, chunksize=8192) as reader:
        for chunk in reader:
            expected = source.iloc[offset:offset + len(chunk)]
            # Compare the producer's CSV representation of every original field,
            # including missing measurements. Chunking avoids retaining a second
            # complete copy of all base measurements in the export environment.
            encoded = pd.read_csv(io.StringIO(expected.to_csv(index=False)), dtype=str, keep_default_na=False, na_filter=False)
            if not chunk[list(source.columns)].reset_index(drop=True).equals(encoded):
                raise ValueError("Reference assignment identities/order or original source columns changed")
            selected = chunk[identity + REFERENCE_COLUMNS].copy()
            _validate_interpretations(selected, atlas, available[offset:offset + len(chunk)])
            pieces.append(selected)
            offset += len(chunk)
    if not pieces:
        return pd.DataFrame(columns=identity + REFERENCE_COLUMNS)
    return pd.concat(pieces, ignore_index=True)


def load_reference_mapping(receipt_path, profile_dir, *, expected_unit=None):
    """Validate a portable bundle against the caller-selected exact profile.

    Return identity + seven interpretation fields, the unmodified receipt, and
    local artifact Paths. Unreceipted legacy CSVs cannot claim profile binding.
    """
    receipt_path = Path(receipt_path).resolve()
    data = receipt_path.read_bytes()
    record = _json(data)
    if record.get("format") != FORMAT or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported or missing reference mapping completion receipt")
    root = receipt_path.parent
    assignments = _inside(root, record["assignments"]["path"], basename=True)
    atlas_path = _inside(root, record["atlas"]["path"], basename=True)
    if len({receipt_path, assignments, atlas_path}) != 3:
        raise ValueError("Reference mapping artifacts must be distinct")
    atlas_data = atlas_path.read_bytes()
    atlas = _json(atlas_data)
    _validate_atlas(atlas)
    if record["atlas"] != _atlas_record(atlas, atlas_data, record["atlas"]["path"]):
        raise ValueError("Frozen atlas identity/hash differs from reference mapping receipt")
    if record.get("observation_unit") != atlas["observation_unit"] or (expected_unit is not None and expected_unit != atlas["observation_unit"]):
        raise ValueError("Reference mapping observation_unit differs from requested export unit")
    source, source_record, available = _profile_snapshot(profile_dir, atlas)
    if source_record != record.get("source_profile"):
        raise ValueError("Source profile hashes/identity differ from reference mapping receipt")
    if record.get("reference_columns") != REFERENCE_COLUMNS:
        raise ValueError("Reference mapping interpretation columns differ from contract")
    assignments_hash = _sha256(assignments)
    if assignments_hash != record["assignments"].get("sha256"):
        raise ValueError("Reference assignments hash differs from completion receipt")
    frame = _read_assignments(assignments, record["assignments"], source, source_record, atlas, available)
    _check_profile_hashes(Path(profile_dir).resolve(), source_record)
    if _sha256(assignments) != assignments_hash or atlas_path.read_bytes() != atlas_data or receipt_path.read_bytes() != data:
        raise ValueError("Reference mapping bundle changed during validation")
    return frame, record, {"receipt": receipt_path, "assignments": assignments, "atlas_manifest": atlas_path}


def write_reference_mapping(query_dir, atlas_dir, output, mapping_function):
    """Run a mapper with pre/post source identities and publish completion last.

    The callback keeps the in-memory map_profile API unchanged. Existing output
    paths (including partial bundles) are never overwritten. A failed write or
    source race may leave incomplete data files, but never a valid completion.
    """
    output = Path(output).absolute()
    if output.suffix != ".csv":
        raise ValueError("Reference mapping bundle output must have a .csv suffix")
    atlas_output, receipt = output.with_suffix(".atlas.json"), output.with_suffix(".mapping.json")
    if any(os.path.lexists(path) for path in (output, atlas_output, receipt)):
        raise FileExistsError("Reference mapping outputs are immutable; choose a new output path")
    atlas, atlas_data = _atlas_snapshot(atlas_dir)
    source, before, _ = _profile_snapshot(query_dir, atlas)
    del source
    result = mapping_function(query_dir, atlas_dir)
    current_atlas, current_atlas_data = _atlas_snapshot(atlas_dir)
    source, after, available = _profile_snapshot(query_dir, current_atlas)
    if before != after or atlas_data != current_atlas_data:
        raise ValueError("Reference mapping source changed during mapping; completion refused")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as handle:
        result.to_csv(handle, index=False)
    with atlas_output.open("xb") as handle:
        handle.write(atlas_data)
    columns = list(result.columns)
    record = {"format": FORMAT, "schema_version": SCHEMA_VERSION,
              "observation_unit": atlas["observation_unit"], "reference_columns": REFERENCE_COLUMNS,
              "source_profile": before, "atlas": _atlas_record(atlas, atlas_data, atlas_output.name),
              "assignments": {"path": output.name, "sha256": _sha256(output), "row_count": len(result),
                              "columns": columns, "column_schema": _column_schema(columns, before["identity_columns"])}}
    _read_assignments(output, record["assignments"], source, after, atlas, available)
    _check_profile_hashes(Path(query_dir).resolve(), before)
    if _atlas_snapshot(atlas_dir)[1] != atlas_data or atlas_output.read_bytes() != atlas_data or _sha256(output) != record["assignments"]["sha256"]:
        raise ValueError("Reference mapping inputs/output changed before completion")
    with receipt.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return result
