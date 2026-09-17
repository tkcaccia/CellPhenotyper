#!/usr/bin/env python3
"""Build immutable cell or tissue-region references and retrieve real neighbours.

Feature blocks use reference-only z-scores, with each block divided by the
square root of its width. Distances and leave-one-out acceptance radii are
descriptive, not calibrated cell-type probabilities. Discovery labels in input
profiles are never modified. Unsupervised tissue labels are scoped to slides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


SCHEMA_VERSION = "1.0.0"
IDENTITY = ["observation_uid", "sample_id", "observation_id"]
PROFILE_FORMATS = {
    "cell": ("cell_profiles", ["cell_uid", "sample_id", "cell_id"], "cell"),
    "region": ("region_profiles", ["region_uid", "sample_id", "region_id"], "tissue_region"),
    "observation": ("observation_profiles", IDENTITY, None),
}
SEARCH_COLUMNS = [
    "query_observation_uid", "query_sample_id", "observation_unit", "reference_atlas_id",
    "search_status", "rank", "reference_observation_uid", "reference_sample_id",
    "reference_observation_id", "reference_group", "similarity_distance", "feature_groups",
    "marker_column", "query_marker_score", "reference_marker_score", "absolute_marker_difference",
]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def contained_path(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Profile/atlas artifact escapes its directory: {relative}")
    return candidate


@dataclass
class Profile:
    root: Path
    cells: pd.DataFrame
    manifest: dict
    blocks: dict
    schemas: dict
    hashes: dict
    unit: str
    identity_columns: list


def load_profile(directory, groups, expected=None):
    root = Path(directory).resolve()
    formats = [entry for entry in PROFILE_FORMATS.values() if (root / (entry[0] + "_manifest.json")).exists()]
    if len(formats) != 1:
        raise ValueError("Profile directory must contain exactly one cell, region or observation profile manifest")
    stem, source_identity, expected_unit = formats[0]
    manifest_path = root / (stem + "_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    unit = manifest.get("observation_unit", "cell" if stem == "cell_profiles" else None)
    if unit not in ("cell", "tissue_region") or (expected_unit is not None and unit != expected_unit):
        raise ValueError("Profile needs a compatible explicit observation_unit (cell or tissue_region)")
    table = root / (stem + ".csv")
    if table.is_file():
        cells = pd.read_csv(table, dtype={key: str for key in source_identity}, keep_default_na=False, float_precision="round_trip")
    else:
        table = root / (stem + ".parquet")
        cells = pd.read_parquet(table)
    if not set(source_identity) <= set(cells):
        raise ValueError(f"Profile is missing observation identity columns: {source_identity}")
    for key in source_identity:
        if cells[key].isna().any() or cells[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"Profile has missing {key}")
        cells[key] = cells[key].astype(str)
    for source_key, normalized in zip(source_identity, IDENTITY):
        if normalized in cells and not cells[normalized].equals(cells[source_key]):
            raise ValueError("Profile contains conflicting observation identity aliases")
        cells[normalized] = cells[source_key]
    if "observation_unit" in cells and not cells.observation_unit.eq(unit).all():
        raise ValueError("Profile has mixed or contradictory observation units")
    cells["observation_unit"] = unit
    if cells.observation_uid.duplicated().any() or cells.duplicated(["sample_id", "observation_id"]).any():
        raise ValueError("Profile contains duplicate observation identities")
    count = manifest.get("observation_count", manifest.get("cell_count", manifest.get("region_count", len(cells))))
    if count != len(cells):
        raise ValueError("Profile row count differs from manifest")
    declared_hash = manifest.get("files", {}).get(table.name)
    table_hash = digest(table)
    if declared_hash and table_hash != declared_hash:
        raise ValueError("Profile table hash differs from manifest")
    rows_path = root / "feature_rows.csv"
    if rows_path.exists():
        rows = pd.read_csv(rows_path, dtype=str, keep_default_na=False)
        if not set(source_identity) <= set(rows) or not rows[source_identity].equals(cells[source_identity]):
            raise ValueError("Feature-row identities do not align with the observation table")
    blocks, schemas = {}, {}
    hashes = {"manifest": digest(manifest_path), "table": table_hash, "blocks": {}}
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("Select at least one distinct feature group explicitly")
    for group in groups:
        record = manifest.get("feature_blocks", {}).get(group)
        if record is None:
            raise ValueError(f"Profile {root.name} is missing feature block {group}")
        if record.get("reference_compatible") is False:
            raise ValueError(f"Feature block {group} is not reference-compatible: immutable model/preprocessing provenance is missing")
        path = contained_path(root, record["path"])
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != len(cells) or matrix.shape[1] < 1:
            raise ValueError(f"Feature block {group} has invalid row/dimension alignment")
        if matrix.dtype.kind not in "fiu":
            raise ValueError(f"Feature block {group} is not numeric")
        if record.get("shape", list(matrix.shape)) != list(matrix.shape):
            raise ValueError(f"Feature block {group} shape differs from manifest")
        definition = record.get("feature_definition")
        if not definition:
            raise ValueError(f"Feature block {group} lacks a feature_definition for reference compatibility")
        names = record.get("feature_names")
        if names is not None and len(names) != matrix.shape[1]:
            raise ValueError(f"Feature block {group} feature names/dimensions differ")
        schema = {"observation_unit": unit, "dimension": int(matrix.shape[1]), "feature_names": names, "feature_definition": definition}
        if expected is not None and canonical_json(schema) != canonical_json(expected[group]):
            raise ValueError(f"Incompatible feature schema/definition for {group}")
        blocks[group], schemas[group] = matrix, schema
        hashes["blocks"][group] = digest(path)
    return Profile(root, cells, manifest, blocks, schemas, hashes, unit, source_identity)


def complete_rows(profile, groups):
    present = np.ones(len(profile.cells), dtype=bool)
    for group in groups:
        for start in range(0, len(present), 8192):
            present[start:start + 8192] &= np.isfinite(profile.blocks[group][start:start + 8192]).all(axis=1)
    return present


def marker_filter_definition(profile, marker, block_cache=None):
    """Verify a requested score against its actual canonical marker block.

    Reuse the producer's portable feature_definition/reference_compatible
    contract, not specimen-specific quantification sidecars or column names
    alone. Only this marker's values must be present/consistent; other markers
    may be missing. Raw table scores retain float64 precision, while their
    canonical block projection is checked using the producer's float32 cast.
    """
    if marker not in profile.manifest.get("biological_marker_features", []) or marker not in profile.cells:
        raise ValueError("marker_not_declared")
    owners = [(name, record) for name, record in profile.manifest.get("feature_blocks", {}).items()
              if isinstance(record, dict) and isinstance(record.get("feature_names"), list)
              and marker in record["feature_names"]]
    if len(owners) != 1:
        raise ValueError("marker_block_missing_or_ambiguous")
    name, record = owners[0]
    if record.get("reference_compatible") is not True:
        raise ValueError("marker_block_not_reference_compatible")
    definition = record.get("feature_definition")
    parts = marker.split("__")
    if not isinstance(definition, dict) or len(parts) != 4 or parts[0] != "predicted" or parts[-1] != "mean":
        raise ValueError("marker_definition_unavailable")
    if (definition.get("schema_version") != "cellphenotyper.gigatime.v1"
            or definition.get("compartment") != parts[1]
            or not definition.get("compartment_semantics")
            or not isinstance(definition.get("marker_names"), list)
            or parts[2] not in definition.get("marker_names", [])
            or not isinstance(definition.get("default_phenotype_excluded_channels"), list)
            or parts[2] in definition["default_phenotype_excluded_channels"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(definition.get("checkpoint_sha256", "")))
            or definition.get("prediction_precision") != "float32"
            or definition.get("reduction_precision") != "float64"
            or definition.get("model_arithmetic") in (None, "unspecified")
            or not definition.get("prediction_settings")
            or (parts[1] != "nucleus" and definition.get("compartment_construction") is None)):
        raise ValueError("marker_definition_unverified")
    try:
        mpp = float(definition["effective_mpp"])
        if not np.isfinite(mpp) or mpp <= 0:
            raise ValueError()
        canonical_json(definition)
    except (KeyError, TypeError, ValueError):
        raise ValueError("marker_definition_unverified") from None
    cache = {} if block_cache is None else block_cache
    if name not in cache:
        try:
            # The existing profile loader verifies the real array/row/schema
            # contract. Load once per compartment, even for a 21-marker panel.
            loaded = load_profile(profile.root, [name])
            values = loaded.blocks[name]
            actual_hash = loaded.hashes["blocks"][name]
            if record.get("sha256") != actual_hash or values.dtype != np.dtype("float32"):
                raise ValueError()
            if loaded.hashes["manifest"] != profile.hashes["manifest"] or loaded.hashes["table"] != profile.hashes["table"]:
                raise ValueError()
            names = record["feature_names"]
            if len(set(names)) != len(names):
                raise ValueError()
            cache[name] = (values, actual_hash)
        except (OSError, TypeError, KeyError, ValueError):
            raise ValueError("marker_block_payload_invalid") from None
    values, actual_hash = cache[name]
    try:
        raw = pd.to_numeric(profile.cells[marker].replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan}), errors="raise").to_numpy(float)
        encoded = values[:, record["feature_names"].index(marker)]
        if np.isinf(raw).any() or np.isinf(encoded).any() or not np.array_equal(raw.astype(np.float32), encoded, equal_nan=True):
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError("marker_table_block_values_disagree") from None
    profile.hashes.setdefault("marker_filter_blocks", {})[name] = actual_hash
    return {"observation_unit": profile.unit, "marker_column": marker,
            "feature_definition": definition, "block_precision": str(values.dtype),
            "value_projection": "raw_profile_scores_cast_float32_equal_canonical_block"}


def reference_marker_filters(profiles, markers):
    """An unavailable ancillary marker never prevents morphology-only use."""
    caches = [{} for _ in profiles]
    records = {}
    for marker in markers:
        definitions, unavailable = [], []
        for profile, cache in zip(profiles, caches):
            try:
                definitions.append(marker_filter_definition(profile, marker, cache))
            except ValueError as exc:
                unavailable.append({"sample_ids": sorted(profile.cells.sample_id.unique().tolist()), "reason": str(exc)})
        if unavailable:
            records[marker] = {"status": "unavailable", "reasons": sorted(unavailable, key=canonical_json)}
        elif len({canonical_json(value) for value in definitions}) != 1:
            records[marker] = {"status": "incompatible", "reason": "reference_marker_definitions_differ"}
        else:
            records[marker] = {"status": "compatible", "definition": definitions[0]}
    return records


def group_key(sample_id, label, column, scope):
    parts = [str(sample_id), column, str(label)] if scope == "sample" else [column, str(label)]
    return ":".join(quote(part, safe="") for part in parts)


def build_atlas(profile_dirs, outdir, version, groups, label_column="tissue_domain",
                label_scope="auto", descriptions=None, quantile=0.95, min_group_size=3,
                representatives=5):
    outdir = Path(outdir)
    if outdir.exists():
        raise FileExistsError("Reference atlas paths are immutable; choose a new version directory")
    if not str(version).strip() or not 0 < quantile <= 1 or min_group_size < 3 or representatives < 1:
        raise ValueError("A version, 0<quantile<=1, min_group_size>=3, and representatives>=1 are required")
    if not profile_dirs:
        raise ValueError("At least one reference profile is required")
    shared_labels = {"phenotype", "reviewed_label"}
    scope = ("shared" if label_column in shared_labels else "sample") if label_scope == "auto" else label_scope
    if scope not in ("sample", "shared"):
        raise ValueError("label_scope must be auto, sample or shared")
    if label_column == "tissue_domain" and scope != "sample":
        raise ValueError("Unsupervised tissue_domain labels must remain sample-scoped; provide reviewed_label to align domains")
    profiles, schema = [], None
    for directory in profile_dirs:
        profile = load_profile(directory, groups, schema)
        if label_column not in profile.cells:
            raise ValueError(f"Reference profile is missing label column {label_column}")
        schema = profile.schemas
        profiles.append(profile)
    cells = pd.concat([profile.cells for profile in profiles], ignore_index=True)
    if cells.observation_uid.duplicated().any() or cells.duplicated(["sample_id", "observation_id"]).any():
        raise ValueError("Duplicate observation identities across reference profiles")
    labels = cells[label_column].fillna("").astype(str)
    valid_labels = ~labels.str.strip().str.lower().isin(["", "nan", "unknown", "unmatched", "unavailable"])
    if label_column == "tissue_domain":
        valid_labels &= ~labels.isin(["0", "0.0"])
    available = np.concatenate([complete_rows(profile, groups) for profile in profiles])
    keep = available & valid_labels.to_numpy()
    if not keep.any():
        raise ValueError("No reference observations have all selected feature blocks and a usable label")
    cells["reference_label"] = labels
    cells["reference_group"] = [group_key(s, lab, label_column, scope) for s, lab in zip(cells.sample_id, labels)]
    marker_columns = sorted(set.intersection(*(set(p.manifest.get("biological_marker_features", [])) for p in profiles)))
    marker_filters = reference_marker_filters(profiles, marker_columns)
    columns = list(dict.fromkeys(IDENTITY + profiles[0].identity_columns + ["observation_unit", label_column, "reference_label", "reference_group"] + marker_columns))
    refs = cells.loc[keep, columns].copy()
    order = np.argsort(refs.observation_uid.to_numpy(), kind="stable")
    refs = refs.iloc[order].reset_index(drop=True)
    group_order = sorted(refs.reference_group.unique())
    descriptions = descriptions or {}
    if not set(descriptions) <= {"reviewed", "predicted"}:
        raise ValueError("Descriptions JSON must separate 'reviewed' and 'predicted' dictionaries")
    for kind, values in descriptions.items():
        if not isinstance(values, dict) or not set(values) <= set(group_order) or not all(isinstance(v, str) for v in values.values()):
            raise ValueError(f"Invalid or unknown {kind} description group IDs")
    features, preprocessing, slices, distributions = [], {}, {}, {}
    offset = 0
    for index, group in enumerate(groups):
        raw = np.concatenate([p.blocks[group] for p in profiles], axis=0)[keep][order].astype(np.float32)
        mean = raw.mean(axis=0, dtype=np.float64)
        std = raw.std(axis=0, dtype=np.float64)
        scale = np.where(std > 1e-8, std, 1.0)
        normalized = ((raw - mean) / scale / np.sqrt(raw.shape[1])).astype(np.float32)
        features.append(normalized)
        preprocessing[f"mean_{index}"] = mean
        preprocessing[f"scale_{index}"] = scale
        distributions[f"p05_p50_p95_{index}"] = np.quantile(raw, [0.05, 0.5, 0.95], axis=0)
        slices[group] = [offset, offset + raw.shape[1]]
        offset += raw.shape[1]
    matrix = np.concatenate(features, axis=1)
    prototypes, prototype_scales, prototype_quantiles, records = [], [], [], []
    for group in group_order:
        indices = np.flatnonzero(refs.reference_group.eq(group).to_numpy())
        observations = matrix[indices]
        center = observations.mean(axis=0, dtype=np.float64)
        distance = np.linalg.norm(observations - center, axis=1)
        count = len(indices)
        threshold = float(np.quantile(distance * count / (count - 1), quantile)) if count >= min_group_size else None
        nearest = sorted(range(count), key=lambda i: (distance[i], refs.observation_uid.iloc[indices[i]]))[:representatives]
        records.append({
            "reference_group": group, "label": str(refs.reference_label.iloc[indices[0]]),
            "source_samples": sorted(refs.sample_id.iloc[indices].unique().tolist()), "observation_count": count,
            "acceptance_radius": threshold, "assignment_supported": threshold is not None,
            "representative_observation_uids": refs.observation_uid.iloc[indices[nearest]].tolist(),
            "reviewed_description": descriptions.get("reviewed", {}).get(group),
            "predicted_description": descriptions.get("predicted", {}).get(group),
            "distance_distribution": dict(zip(["p05", "p50", "p95"], map(float, np.quantile(distance, [0.05, 0.5, 0.95])))),
        })
        prototypes.append(center)
        prototype_scales.append(observations.std(axis=0, dtype=np.float64))
        prototype_quantiles.append(np.quantile(observations, [0.05, 0.5, 0.95], axis=0))
    source_records = sorted([
        {"sample_ids": sorted(p.cells.sample_id.unique().tolist()), "hashes": p.hashes}
        for p in profiles
    ], key=canonical_json)
    identity = {
        "schema_version": SCHEMA_VERSION, "version": str(version), "feature_groups": list(groups), "observation_unit": profiles[0].unit,
        "feature_schemas": schema, "label_column": label_column, "label_scope": scope,
        "source_profiles": source_records, "descriptions": descriptions,
        "threshold_quantile": quantile, "minimum_group_size": min_group_size,
        "representatives_per_group": representatives,
        "marker_filter_contracts": marker_filters,
    }
    manifest = {**identity, "atlas_id": hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:24],
        "reference_observation_count": len(refs), "excluded_missing_features": int((~available).sum()),
        "excluded_unlabelled": int((available & ~valid_labels.to_numpy()).sum()),
        "feature_slices": slices, "marker_columns": marker_columns, "groups": records,
        "normalization": "reference_mean_and_std_per_feature_then_inverse_sqrt_block_dimension",
        "threshold_method": "locked_reference_leave_one_out_distance_to_own_group_mean",
        "prototype_distribution_scale": "reference-standardized, dimension-balanced features",
        "interpretation": "Reference resemblance only; distances/radii are not calibrated probabilities or biological validation.",
    }
    outdir.mkdir(parents=True, exist_ok=False)
    refs.to_csv(outdir / "reference_observations.csv", index=False)
    np.save(outdir / "reference_features.npy", matrix, allow_pickle=False)
    np.save(outdir / "prototypes.npy", np.asarray(prototypes, dtype=np.float32), allow_pickle=False)
    np.save(outdir / "prototype_scales.npy", np.asarray(prototype_scales, dtype=np.float32), allow_pickle=False)
    np.save(outdir / "prototype_quantiles.npy", np.asarray(prototype_quantiles, dtype=np.float32), allow_pickle=False)
    np.savez(outdir / "preprocessing.npz", **preprocessing)
    np.savez(outdir / "feature_distributions.npz", **distributions)
    manifest["files"] = {p.name: digest(p) for p in sorted(outdir.iterdir()) if p.is_file()}
    write_json(outdir / "atlas_manifest.json", manifest)
    (outdir / "atlas_manifest.sha256").write_text(digest(outdir / "atlas_manifest.json") + "\n")
    return manifest


def load_atlas(directory):
    root = Path(directory).resolve()
    manifest_path = root / "atlas_manifest.json"
    if digest(manifest_path) != (root / "atlas_manifest.sha256").read_text().strip():
        raise ValueError("Immutable atlas manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported reference atlas schema")
    for name, expected in manifest["files"].items():
        if digest(contained_path(root, name)) != expected:
            raise ValueError(f"Immutable atlas artifact changed: {name}")
    refs = pd.read_csv(root / "reference_observations.csv", dtype={key: str for key in IDENTITY}, keep_default_na=False, float_precision="round_trip")
    if refs.observation_uid.duplicated().any():
        raise ValueError("Duplicate reference observation identities")
    features = np.load(root / "reference_features.npy", mmap_mode="r", allow_pickle=False)
    prototypes = np.load(root / "prototypes.npy", allow_pickle=False)
    width = sum(manifest["feature_schemas"][group]["dimension"] for group in manifest["feature_groups"])
    if features.shape != (len(refs), width) or prototypes.shape != (len(manifest["groups"]), width):
        raise ValueError("Atlas arrays/identities do not align")
    return root, manifest, refs, features, prototypes


def transform_profile(profile, atlas_root, manifest, groups, indices=None):
    indices = slice(None) if indices is None else indices
    transformed = []
    with np.load(atlas_root / "preprocessing.npz", allow_pickle=False) as stats:
        for group in groups:
            index = manifest["feature_groups"].index(group)
            raw = profile.blocks[group][indices]
            transformed.append(((raw - stats[f"mean_{index}"]) / stats[f"scale_{index}"] / np.sqrt(raw.shape[1])).astype(np.float32))
    values = np.concatenate(transformed, axis=1)
    return values, np.isfinite(values).all(axis=1)


def map_profile(query_dir, atlas_dir):
    root, manifest, _, _, prototypes = load_atlas(atlas_dir)
    groups = manifest["feature_groups"]
    profile = load_profile(query_dir, groups, manifest["feature_schemas"])
    result = profile.cells.copy()
    # Keep all discovery labels, then append a separate reference interpretation.
    reserved = ["reference_assignment", "reference_status", "reference_distance", "reference_radius", "reference_margin", "reference_nearest_group", "reference_atlas_id"]
    if set(reserved) & set(result):
        raise ValueError("Query profile already contains reference-assignment columns")
    result["reference_assignment"] = "unknown"
    result["reference_status"] = "missing_features"
    result["reference_distance"] = np.nan
    result["reference_radius"] = np.nan
    result["reference_margin"] = np.nan
    result["reference_nearest_group"] = ""
    result["reference_atlas_id"] = manifest["atlas_id"]
    for start in range(0, len(result), 1024):
        query, available = transform_profile(profile, root, manifest, groups, slice(start, start + 1024))
        ids = np.flatnonzero(available) + start
        if not len(ids):
            continue
        distances = cdist(query[available], prototypes)
        for row_id, distance in zip(ids, distances):
            order = np.argsort(distance, kind="stable")
            record = manifest["groups"][int(order[0])]
            nearest = float(distance[order[0]])
            radius = record["acceptance_radius"]
            margin = float(distance[order[1]] - nearest) if len(order) > 1 else np.nan
            result.at[row_id, "reference_distance"] = nearest
            result.at[row_id, "reference_margin"] = margin
            result.at[row_id, "reference_nearest_group"] = record["reference_group"]
            if radius is None:
                status = "insufficient_reference"
            elif nearest > radius + 1e-7:
                status = "outside_reference"
            elif len(order) > 1 and margin <= 1e-7:
                status = "ambiguous_reference"
            else:
                status = "assigned"
                result.at[row_id, "reference_assignment"] = record["reference_group"]
            result.at[row_id, "reference_radius"] = radius if radius is not None else np.nan
            result.at[row_id, "reference_status"] = status
    return result


def search_similar(query_dir, atlas_dir, observation_uids, groups, k=10,
                   marker_column=None, min_marker_difference=None, exclude_same_sample=False):
    root, manifest, refs, reference, _ = load_atlas(atlas_dir)
    if not groups or not set(groups) <= set(manifest["feature_groups"]):
        raise ValueError("Explicit search feature groups must be present in the atlas")
    if k < 1 or len(set(observation_uids)) != len(observation_uids) or not observation_uids:
        raise ValueError("Search requires unique observation IDs and k>=1")
    if bool(marker_column) != (min_marker_difference is not None):
        raise ValueError("Marker discordance requires both marker_column and min_marker_difference")
    if min_marker_difference is not None and (not np.isfinite(min_marker_difference) or min_marker_difference < 0):
        raise ValueError("Marker difference must be finite and nonnegative")
    profile = load_profile(query_dir, groups, manifest["feature_schemas"])
    lookup = dict(zip(profile.cells.observation_uid, range(len(profile.cells))))
    if not set(observation_uids) <= set(lookup):
        raise ValueError("Requested query observation_uid is absent from the profile")
    if marker_column and (marker_column not in manifest["marker_columns"] or marker_column not in profile.manifest.get("biological_marker_features", [])):
        raise ValueError("Discordance filter requires a shared declared biological-marker score column")
    if marker_column:
        comparison = manifest.get("marker_filter_contracts", {}).get(marker_column)
        if not comparison or comparison.get("status") != "compatible":
            reason = comparison.get("reason", comparison.get("status")) if comparison else "legacy_atlas_without_marker_definition"
            raise ValueError(f"Marker discordance unavailable for {marker_column}: {reason}; use compatible verified marker profiles and rebuild the atlas")
        try:
            query_definition = marker_filter_definition(profile, marker_column)
        except ValueError as exc:
            raise ValueError(f"Marker discordance unavailable for query {marker_column}: {exc}") from exc
        if canonical_json(query_definition) != canonical_json(comparison["definition"]):
            raise ValueError(f"Marker discordance requires a compatible marker definition for {marker_column}; model, compartment, precision or preprocessing differs")
    query, available = transform_profile(profile, root, manifest, groups, [lookup[uid] for uid in observation_uids])
    selected_columns = np.concatenate([np.arange(*manifest["feature_slices"][group]) for group in groups])
    output = []
    for query_index, uid in enumerate(observation_uids):
        row_id = lookup[uid]
        base = {"query_observation_uid": uid, "query_sample_id": profile.cells.sample_id.iloc[row_id], "reference_atlas_id": manifest["atlas_id"], "observation_unit": profile.unit, "feature_groups": ";".join(groups)}
        if not available[query_index]:
            output.append({**base, "search_status": "missing_features"})
            continue
        eligible = refs.observation_uid.ne(uid).to_numpy(copy=True)
        if exclude_same_sample:
            eligible &= refs.sample_id.ne(profile.cells.sample_id.iloc[row_id]).to_numpy()
        if marker_column:
            query_marker = pd.to_numeric(pd.Series([profile.cells[marker_column].iloc[row_id]]), errors="coerce").iloc[0]
            marker_values = pd.to_numeric(refs[marker_column], errors="coerce").to_numpy(float)
            difference = np.abs(marker_values - query_marker)
            if not np.isfinite(query_marker):
                output.append({**base, "search_status": "missing_marker"})
                continue
            eligible &= np.isfinite(difference) & (difference >= min_marker_difference)
        indices = np.flatnonzero(eligible)
        if not len(indices):
            output.append({**base, "search_status": "no_matching_reference"})
            continue
        distances = np.empty(len(indices), dtype=np.float64)
        for start in range(0, len(indices), 4096):
            selected = reference[np.ix_(indices[start:start + 4096], selected_columns)]
            distances[start:start + 4096] = cdist(query[query_index:query_index + 1], selected)[0]
        ordered = np.lexsort((refs.observation_uid.iloc[indices].to_numpy(), distances))[:k]
        for rank, local_index in enumerate(ordered, 1):
            index = indices[local_index]
            record = {**base, "search_status": "matched", "rank": rank,
                "reference_observation_uid": refs.observation_uid.iloc[index], "reference_sample_id": refs.sample_id.iloc[index],
                "reference_observation_id": refs.observation_id.iloc[index], "reference_group": refs.reference_group.iloc[index],
                "similarity_distance": float(distances[local_index]), "feature_groups": ";".join(groups),
            }
            if marker_column:
                record.update({"marker_column": marker_column, "query_marker_score": float(query_marker),
                    "reference_marker_score": float(marker_values[index]), "absolute_marker_difference": float(difference[index])})
            output.append(record)
    return pd.DataFrame(output).reindex(columns=SEARCH_COLUMNS)


def save_result(frame, path):
    # Opening exclusively prevents a query export from overwriting an older result.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        frame.to_csv(handle, index=False)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build a new immutable reference version")
    build.add_argument("--profiles", nargs="+", required=True)
    build.add_argument("--outdir", required=True)
    build.add_argument("--version", required=True)
    build.add_argument("--feature-groups", nargs="+", required=True)
    build.add_argument("--label-column", default="tissue_domain")
    build.add_argument("--label-scope", choices=["auto", "sample", "shared"], default="auto")
    build.add_argument("--descriptions", help="JSON with separate reviewed/predicted dictionaries keyed by reference group")
    build.add_argument("--quantile", type=float, default=0.95)
    build.add_argument("--min-group-size", type=int, default=3)
    build.add_argument("--representatives", type=int, default=5)
    for command in ("map", "search"):
        child = sub.add_parser(command)
        child.add_argument("--query", required=True)
        child.add_argument("--atlas", required=True)
        child.add_argument("--output", required=True)
        if command == "search":
            child.add_argument("--observation-uid", "--cell-uid", dest="observation_uid", nargs="+", required=True)
            child.add_argument("--feature-groups", nargs="+", required=True)
            child.add_argument("--k", type=int, default=10)
            child.add_argument("--marker-column")
            child.add_argument("--min-marker-difference", type=float)
            child.add_argument("--exclude-same-sample", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.command == "build":
        descriptions = json.loads(Path(args.descriptions).read_text()) if args.descriptions else None
        result = build_atlas(args.profiles, args.outdir, args.version, args.feature_groups,
            args.label_column, args.label_scope, descriptions, args.quantile, args.min_group_size, args.representatives)
        print(f"Reference atlas {result['atlas_id']}: {result['reference_observation_count']} {result['observation_unit']} observations, {len(result['groups'])} groups")
    elif args.command == "map":
        from reference_mapping_io import write_reference_mapping
        result = write_reference_mapping(args.query, args.atlas, args.output, map_profile)
        print(f"Reference mapping: {len(result)} observations, {int(result.reference_status.eq('assigned').sum())} assigned")
    else:
        result = search_similar(args.query, args.atlas, args.observation_uid, args.feature_groups, args.k,
            args.marker_column, args.min_marker_difference, args.exclude_same_sample)
        save_result(result, args.output)
        print(f"Similarity retrieval: {len(result)} output rows")


if __name__ == "__main__":
    main()
