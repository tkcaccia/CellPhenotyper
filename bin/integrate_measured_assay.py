#!/usr/bin/env python3
"""Import explicitly matched measured assays as an immutable, separate modality.

No registration, cell matching, normalization, correlation-based assignment, or
biological-accuracy assessment is performed here. See --help and the versioned
input contract enforced below. Serial sections and Visium remain region-level.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from cell_profile_io import sha256_file


INPUT_SCHEMA = "cellphenotyper.measured_assay_input.v1"
OUTPUT_SCHEMA = "cellphenotyper.measured_assay.v1"
REGION_SCHEMA = "cellphenotyper.region_registry.v1"
TARGET_FRAME = "original_slide_micrometres"
IDENTIFIERS = {"sample_id", "cell_id", "cell_uid", "region_id", "region_uid", "assay_observation_id"}


def required_text(record, name):
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or empty {name}")
    return value.strip()


def number(record, name, *, minimum=0., strictly_positive=False):
    value = record.get(name)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Missing or invalid {name}") from None
    if not math.isfinite(value) or value < minimum or (strictly_positive and value <= 0):
        raise ValueError(f"Invalid {name}: must be finite and {'positive' if strictly_positive else f'>= {minimum}'}")
    return value


def verified_artifact(record, base):
    if not isinstance(record, dict):
        raise ValueError("Provenance artifact must have path and sha256")
    raw = Path(required_text(record, "path"))
    path = raw if raw.is_absolute() else base / raw
    expected = required_text(record, "sha256").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or not path.is_file():
        raise ValueError(f"Invalid or unavailable provenance artifact: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"Provenance artifact hash mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": expected, "role": record.get("role", "unspecified")}


def read_table(path, max_observations):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        if pq.ParquetFile(path).metadata.num_rows > max_observations:
            raise ValueError("Observation table exceeds --max-observations")
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
        if len(header) != len(set(header)):
            raise ValueError("Duplicate table columns")
        # Read IDs as strings, including IDs such as NA or 001. Numeric missing
        # values are parsed explicitly below instead of changing identity.
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=max_observations + 1)
    else:
        raise ValueError("Observation tables must be CSV or Parquet")
    if len(frame) > max_observations:
        raise ValueError("Observation table exceeds --max-observations")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate table columns")
    return frame


def validate_ids(frame, key):
    if key not in frame:
        raise ValueError(f"Missing identity column {key}")
    if frame[key].isna().any():
        raise ValueError(f"Missing {key}")
    if not frame[key].map(lambda v: isinstance(v, str)).all():
        raise ValueError(f"{key} must be an explicit string identifier, not a numeric surrogate")
    if frame[key].str.strip().eq("").any() or not frame[key].eq(frame[key].str.strip()).all():
        raise ValueError(f"Empty or whitespace-ambiguous {key}")
    if frame[key].duplicated().any():
        raise ValueError(f"Duplicate {key}")


def numeric_values(frame, columns, *, allow_missing):
    if not set(columns) <= set(frame):
        raise ValueError(f"Missing numeric columns: {sorted(set(columns) - set(frame))}")
    parsed = frame[columns].replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan})
    parsed = parsed.apply(pd.to_numeric, errors="raise").astype(np.float64)
    values = parsed.to_numpy()
    if np.isinf(values).any() or (not allow_missing and np.isnan(values).any()):
        raise ValueError("Nonfinite values or missing coordinates in observation table")
    return parsed


def load_registry(*, cell_profiles, regions, regions_manifest, max_observations):
    if bool(cell_profiles) == bool(regions):
        raise ValueError("Choose exactly one of --cell-profiles and --regions")
    if cell_profiles:
        root = Path(cell_profiles)
        manifest_path = root / "cell_profiles_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        candidates = [root / name for name in ("cell_profiles.parquet", "cell_profiles.csv")
                      if (root / name).is_file() and name in manifest.get("files", {})]
        if not candidates:
            raise ValueError("Canonical profile manifest has no hashed cell table")
        table_path = candidates[0]
        key, unit, count_key = "cell_uid", "cell", "cell_count"
    else:
        if not regions_manifest:
            raise ValueError("--regions requires --regions-manifest with region/coordinate lineage")
        manifest_path = Path(regions_manifest)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != REGION_SCHEMA:
            raise ValueError(f"Region registry requires schema_version={REGION_SCHEMA}")
        table_path = Path(regions)
        key, unit, count_key = "region_uid", "spatial_bin", "region_count"
        definition = manifest.get("region_definition", {})
        required_text(definition, "kind")
        required_text(definition, "description")
        definition["artifact"] = verified_artifact(definition.get("artifact"), manifest_path.parent)
    expected = manifest.get("files", {}).get(table_path.name)
    if not isinstance(expected, str) or sha256_file(table_path) != expected:
        raise ValueError("Registry table hash does not match its immutable manifest")
    if manifest.get("observation_unit") != unit or manifest.get("coordinate_system") != TARGET_FRAME:
        raise ValueError("Registry observation unit or coordinate system is incompatible")
    sample = required_text(manifest, "sample_id")
    frame = read_table(table_path, max_observations)
    validate_ids(frame, key)
    if frame.empty:
        raise ValueError("Measured-assay import requires a nonempty eligible registry")
    if manifest.get(count_key) != len(frame):
        raise ValueError("Registry observation count does not match manifest")
    if "sample_id" not in frame or set(frame.sample_id) != {sample}:
        raise ValueError("Registry sample identity mismatch")
    frame[["x_um", "y_um"]] = numeric_values(frame, ["x_um", "y_um"], allow_missing=False)
    if unit == "spatial_bin":
        frame["area_um2"] = numeric_values(frame, ["area_um2"], allow_missing=False).area_um2
        if (frame.area_um2 <= 0).any():
            raise ValueError("Region areas must be positive")
        if "cell_uid" in frame:
            raise ValueError("Region registry must not present regions as single-cell identities")
    return frame, manifest, key, {"manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path), "table_path": str(table_path.resolve()),
        "table_sha256": expected, "observation_unit": unit, "sample_id": sample}


def validate_contract(contract, base, registry, registry_record, measured_path):
    if contract.get("schema_version") != INPUT_SCHEMA:
        raise ValueError(f"Assay manifest requires schema_version={INPUT_SCHEMA}")
    assay_id = required_text(contract, "assay_id")
    for key in ("assay_type", "assay_platform", "assay_protocol", "panel_version"):
        required_text(contract, key)
    unit = registry_record["observation_unit"]
    if contract.get("observation_unit") != unit or contract.get("sample_id") != registry_record["sample_id"]:
        raise ValueError("Assay and registry sample/observation units do not match")
    design = contract.get("reference_design")
    if design not in {"same_section_registered", "serial_section_region_level"}:
        raise ValueError("Unsupported reference_design")
    if unit == "cell" and design != "same_section_registered":
        raise ValueError("Serial sections cannot be imported as matched individual cells")
    if unit == "cell" and "visium" in contract["assay_platform"].lower():
        raise ValueError("Visium observations must use a separate spatial_bin registry, not individual cells")
    coordinates = contract.get("coordinates", {})
    required_text(coordinates, "source_frame")
    if coordinates.get("source_units") not in {"pixel", "um"}:
        raise ValueError("Source coordinate units must be pixel or um")
    if coordinates.get("target_frame") != TARGET_FRAME or coordinates.get("target_units") != "um":
        raise ValueError("Target coordinates must explicitly use original-slide micrometres")
    registration = contract.get("registration", {})
    if registration.get("status") != "passed":
        raise ValueError("Registration must have status=passed")
    if registration.get("locked_before_prediction_review") is not True:
        raise ValueError("Registration must be locked independently before prediction review")
    required_text(registration, "method")
    landmarks = number(registration, "independent_landmarks", minimum=3.)
    if landmarks != int(landmarks):
        raise ValueError("Registration landmark count must be an integer")
    median = number(registration, "median_error_um")
    p95 = number(registration, "p95_error_um")
    acceptance = number(registration, "acceptance_p95_um", strictly_positive=True)
    if p95 < median or p95 > acceptance:
        raise ValueError("Registration p95 error is inconsistent or exceeds the locked acceptance gate")
    if registration.get("target_observations_sha256") != registry_record["table_sha256"]:
        raise ValueError("Registration target registry hash mismatch")
    if registration.get("source_observations_sha256") != sha256_file(measured_path):
        raise ValueError("Registration source measured table hash mismatch")
    transform = verified_artifact(registration.get("transform_artifact"), base)
    matching = contract.get("matching", {})
    expected_method = "provided_one_to_one_ids" if unit == "cell" else "provided_region_ids"
    if matching.get("method") != expected_method or matching.get("independent_of_predicted_markers") is not True:
        raise ValueError("Matching must be explicitly provided and independent of predicted markers; no inferred correlation matching")
    required_text(matching, "protocol")
    if matching.get("eligible_observation_count") != len(registry):
        raise ValueError("Eligible count must include the complete supplied registry; do not hide unmatched observations")
    minimum = number(matching, "minimum_matched_fraction", strictly_positive=True)
    declared = number(matching, "matched_fraction")
    if minimum > 1 or declared > 1:
        raise ValueError("Matching fractions cannot exceed one")
    max_distance = number(matching, "maximum_match_distance_um", strictly_positive=True)
    artifacts = contract.get("source_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("At least one hashed independent assay source artifact is required")
    artifacts = [verified_artifact(record, base) for record in artifacts]
    markers = contract.get("markers")
    if not isinstance(markers, list) or not markers:
        raise ValueError("Declare at least one measured marker with source column and units")
    names, columns, outputs = [], [], []
    for marker in markers:
        name = required_text(marker, "name")
        column = required_text(marker, "column")
        for field in ("units", "measurement_type", "normalization"):
            required_text(marker, field)
        if column in IDENTIFIERS or column in {"assay_x", "assay_y", "registered_x_um", "registered_y_um"}:
            raise ValueError("Marker source column collides with identity or coordinate metadata")
        names.append(name)
        columns.append(column)
        outputs.append(f"measured__{quote(assay_id, safe='')}__{quote(name, safe='')}")
    if len(set(names)) != len(names) or len(set(columns)) != len(columns):
        raise ValueError("Duplicate marker name or source column")
    return columns, outputs, {"transform_artifact": transform, "source_artifacts": artifacts,
                             "minimum_matched_fraction": minimum, "declared_matched_fraction": declared,
                             "maximum_match_distance_um": max_distance}


def integrate(*, measured, assay_manifest, outdir, cell_profiles=None, regions=None,
              regions_manifest=None, max_observations=5_000_000, max_matrix_values=50_000_000):
    output = Path(outdir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Measured modalities require a new output directory; existing profiles are immutable")
    if max_observations < 1 or max_matrix_values < 1:
        raise ValueError("Observation/matrix guards must be positive")
    registry, registry_manifest, key, record = load_registry(cell_profiles=cell_profiles, regions=regions,
        regions_manifest=regions_manifest, max_observations=max_observations)
    # Do not add even a child directory to the canonical immutable product.
    if cell_profiles and Path(cell_profiles).resolve() in output.resolve().parents:
        raise ValueError("Output must be separate from the canonical cell-profile directory")
    manifest_path = Path(assay_manifest)
    contract = json.loads(manifest_path.read_text())
    columns, output_columns, evidence = validate_contract(contract, manifest_path.parent, registry, record, measured)
    if len(registry) * len(columns) > max_matrix_values:
        raise ValueError("Measured matrix exceeds --max-matrix-values")
    assay = read_table(measured, max_observations)
    validate_ids(assay, key)
    validate_ids(assay, "assay_observation_id")
    if "sample_id" not in assay or (not assay.empty and set(assay.sample_id) != {record["sample_id"]}):
        raise ValueError("Measured table sample identity mismatch")
    if key == "region_uid" and "cell_uid" in assay:
        raise ValueError("Region-level assay table must not assert per-cell identities")
    foreign = set(assay[key]) - set(registry[key])
    if foreign:
        raise ValueError(f"Measured table contains {len(foreign)} foreign {key} values")
    values = numeric_values(assay, columns, allow_missing=True)
    coordinates = numeric_values(assay, ["assay_x", "assay_y", "registered_x_um", "registered_y_um"], allow_missing=False)
    fraction = len(assay) / len(registry)
    if not math.isclose(fraction, evidence["declared_matched_fraction"], abs_tol=1e-8):
        raise ValueError("Declared matched_fraction disagrees with actual unique matches")
    if fraction < evidence["minimum_matched_fraction"]:
        raise ValueError("Matched fraction is below the locked minimum; unmatched observations remain in the denominator")
    target = numeric_values(registry.set_index(key).loc[assay[key]], ["x_um", "y_um"], allow_missing=False)
    distances = np.linalg.norm(target.to_numpy() - coordinates[["registered_x_um", "registered_y_um"]].to_numpy(), axis=1)
    if (distances > evidence["maximum_match_distance_um"]).any():
        raise ValueError("A provided match exceeds maximum_match_distance_um; no nearest/correlation reassignment is performed")
    identity = [name for name in ("sample_id", key, "cell_id" if key == "cell_uid" else "region_id",
                                  "x_um", "y_um", "area_um2" if key == "region_uid" else "segmentation_id") if name in registry]
    result = registry[identity].copy()
    payload = assay[[key, "assay_observation_id"]].copy()
    for name in coordinates:
        payload[name] = coordinates[name].to_numpy()
    payload["match_distance_um"] = distances
    payload[output_columns] = values.to_numpy()
    result = result.merge(payload, on=key, how="left", validate="one_to_one", sort=False)
    if result[key].tolist() != registry[key].tolist():
        raise RuntimeError("Measured join changed canonical observation order")
    result["assay_id"] = contract["assay_id"]
    result["measured_matched"] = result[key].isin(set(assay[key]))
    result["measured_marker_count"] = result[output_columns].notna().sum(axis=1)
    result["measured_status"] = np.where(~result.measured_matched, "unmatched",
        np.where(result.measured_marker_count.eq(0), "matched_all_markers_missing", "matched"))
    matrix = result[output_columns].to_numpy(dtype=np.float64)
    # All validation precedes the first output write; inputs are never modified.
    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / "measured_observations.csv", index=False)
    result.to_parquet(output / "measured_observations.parquet", index=False)
    np.save(output / "measured_values.npy", matrix, allow_pickle=False)
    result[["sample_id", key]].to_csv(output / "measured_rows.csv", index=False)
    files = {name: sha256_file(output / name) for name in (
        "measured_observations.csv", "measured_observations.parquet", "measured_values.npy", "measured_rows.csv")}
    manifest = {"schema_version": OUTPUT_SCHEMA, "modality": "measured_assay",
        "observation_unit": record["observation_unit"], "identity_key": key,
        "sample_id": record["sample_id"], "assay_id": contract["assay_id"],
        "assay_type": contract["assay_type"], "assay_platform": contract["assay_platform"],
        "reference_design": contract["reference_design"], "coordinate_system": TARGET_FRAME,
        "observation_count": len(registry), "matched_observation_count": len(assay),
        "unmatched_observation_count": len(registry) - len(assay), "matched_fraction": fraction,
        "all_markers_missing_count": int(result.measured_marker_count.eq(0).sum()),
        "join_policy": "immutable_registry_left_join_no_imputation_no_reassignment",
        "predicted_features_modified": False, "biological_accuracy_validated": False,
        "value_semantics": "independently_measured_assay_values_in_declared_units_without_import_normalization",
        "matrix": {"path": "measured_values.npy", "dtype": "float64", "shape": list(matrix.shape),
                   "row_index": "measured_rows.csv", "feature_names": output_columns,
                   "marker_names": [marker["name"] for marker in contract["markers"]],
                   "missing_value": "NaN", "sha256": files["measured_values.npy"]},
        "markers": [dict(marker, output_column=column, missing_observations=int(result[column].isna().sum()))
                    for marker, column in zip(contract["markers"], output_columns)],
        "canonical_registry": record, "registration": contract["registration"],
        "matching": dict(contract["matching"], observed_matched_fraction=fraction,
                         observed_max_match_distance_um=float(distances.max()) if len(distances) else None),
        "coordinates": contract["coordinates"], "assay_protocol": contract["assay_protocol"],
        "panel_version": contract["panel_version"], "verified_evidence": evidence,
        "inputs": {"assay_manifest_path": str(manifest_path.resolve()), "assay_manifest_sha256": sha256_file(manifest_path),
                   "measured_table_path": str(Path(measured).resolve()), "measured_table_sha256": sha256_file(measured)},
        "region_definition": registry_manifest.get("region_definition"), "files": files,
        "limitations": ["Importer verifies declared identity/provenance and registration gates; it does not perform registration or independently adjudicate its accuracy.",
            "Measured values remain assay-specific; no assay interchangeability, diagnostic validity or virtual-marker biological accuracy is established.",
            "Serial sections and Visium are region-level modalities; region signal must not be represented as a measured value of an individual cell.",
            "Missing marker values and unmatched eligible observations are preserved; no correlation-based matching, normalization or imputation is performed."]}
    (output / "measured_assay_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return result, manifest


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--cell-profiles", help="Immutable canonical profile directory")
    source.add_argument("--regions", help="Explicit region_uid registry CSV/Parquet")
    ap.add_argument("--regions-manifest", help=f"Region lineage manifest ({REGION_SCHEMA})")
    ap.add_argument("--measured", required=True, help="Explicitly keyed measured CSV/Parquet")
    ap.add_argument("--assay-manifest", required=True, help=f"Assay/registration contract ({INPUT_SCHEMA})")
    ap.add_argument("--outdir", required=True, help="New standalone measured modality directory")
    ap.add_argument("--max-observations", type=int, default=5_000_000)
    ap.add_argument("--max-matrix-values", type=int, default=50_000_000)
    return ap


def main():
    _, manifest = integrate(**vars(parser().parse_args()))
    print(json.dumps({"observation_unit": manifest["observation_unit"],
                      "observations": manifest["observation_count"],
                      "matched": manifest["matched_observation_count"],
                      "biological_accuracy_validated": False}))


if __name__ == "__main__":
    main()
