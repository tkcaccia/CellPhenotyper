#!/usr/bin/env python3
"""Read source-bound shared niches without importing fitting/model runtimes.

Completion receipts bind bytes and engineering lineage, not biological truth.
Only the explicitly supplied profile directory is read: paths for other cohort
specimens are never inferred or followed. Global assignments are scanned in
batches; identity sets still require memory proportional to the cohort size.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FORMAT = "cellphenotyper_cohort_niches"
VERSION = "1.0.0"
KEYS = ["sample_id", "cell_id", "cell_uid"]
COHORT_COLUMNS = ["cohort_niche_id", "cohort_niche_status", "cohort_niche_centroid_margin",
                  "cohort_niche_stability", "cohort_niche_model_id"]
MODEL = "cohort_niche_model.json"
SUMMARY = "cohort_niche_summary.json"
ASSIGNMENTS = "cohort_niche_assignments.parquet"
COMPLETION = "cohort_niches_completion.json"
SOURCE_FILES = {"cell_profiles.parquet", "feature_rows.csv", "neighborhood_summary.json"}
GEOMETRY = ["x_um", "y_um", "in_tissue_support"]
ASSIGNED = {"assigned_fixed_k", "assigned_exploratory", "single_niche_no_supported_subdivision"}
UNASSIGNED = {"isolated_no_neighborhood", "outside_tissue_support"}
ASSIGNMENT_BATCH_ROWS = 65536


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    data = json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=reject)
    _canonical(data)  # Also rejects overflowed numeric literals such as 1e999.
    if not isinstance(data, dict):
        raise ValueError("Cohort metadata must be JSON objects")
    return data


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or Path(relative).is_absolute() or any(p in ("", ".", "..") for p in relative.split("/"))):
        raise ValueError("Cohort artifact path must be a contained relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping cohort artifact: {relative}")
    return path


def _hashes(root, files):
    if not isinstance(files, dict) or not files:
        raise ValueError("Cohort artifact hash inventory is missing")
    for name, expected in files.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid cohort artifact SHA256")
        if _sha(_inside(root, name)) != expected:
            raise ValueError(f"Cohort source SHA256 mismatch: {name}")


def _literal(value):
    return (isinstance(value, str) and bool(value.strip())
            and not re.search(r"[\x00-\x1f\x7f]", value))


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _identity(frame):
    if not set(KEYS) <= set(frame) or any(not frame[key].map(_literal).all() for key in KEYS):
        raise ValueError("Cohort identifiers must be nonempty literal strings")
    if frame.cell_uid.duplicated().any() or frame.duplicated(KEYS[:2]).any():
        raise ValueError("Duplicate cohort cell identities")


def _metadata(bundle):
    completion_path = _inside(bundle, COMPLETION)
    completion_hash = _sha(completion_path)
    completion = _json(completion_path)
    if set(completion.get("files", {})) != {MODEL, SUMMARY, ASSIGNMENTS}:
        raise ValueError("Cohort completion requires the exact three output artifacts")
    files = {**completion["files"], COMPLETION: completion_hash}
    _hashes(bundle, files)
    model, summary = _json(_inside(bundle, MODEL)), _json(_inside(bundle, SUMMARY))
    for item in (completion, model, summary):
        if item.get("format") != FORMAT or item.get("schema_version") != VERSION:
            raise ValueError("Unsupported cohort niche format/version")
    identity = dict(model)
    model_id = identity.pop("cohort_niche_model_id", None)
    if model_id != hashlib.sha256(_canonical(identity).encode()).hexdigest():
        raise ValueError("Cohort model ID does not match its definition")
    if any(item.get("cohort_niche_model_id") != model_id for item in (summary, completion)):
        raise ValueError("Cohort model/summary/completion identities disagree")
    if (model.get("analysis") != "pooled_exploratory_discovery"
            or model.get("reference_assignment") is not False
            or model.get("calibrated_biological_confidence") is not False):
        raise ValueError("Cohort discovery must remain exploratory, not reference assignment/confidence")
    sources = model.get("sources")
    if not isinstance(sources, list) or len(sources) < 2 or any(not isinstance(s, dict) for s in sources):
        raise ValueError("Cohort model requires at least two specimen identities")
    samples = [source.get("sample_id") for source in sources]
    if not all(map(_literal, samples)) or len(set(samples)) != len(samples) or samples != sorted(samples):
        raise ValueError("Cohort specimen identities must be unique and canonically ordered")
    for source in sources:
        if (not _integer(source.get("cell_count"), 1) or set(source.get("files", {})) != SOURCE_FILES
                or not isinstance(source.get("graphs"), dict) or not source["graphs"]
                or ("store_files" in source and (not isinstance(source["store_files"], dict) or not source["store_files"]))):
            raise ValueError("Invalid cohort source identity inventory")
        combined = {"cell_profiles_manifest.json": source.get("manifest_sha256"), **source["files"], **source["graphs"]}
        if any(name in combined and combined[name] != value for name, value in source.get("store_files", {}).items()):
            raise ValueError("Conflicting cohort store source artifact hashes")
        # Validate path/hash syntax without touching other specimens' paths.
        for name, value in {**combined, **source.get("store_files", {})}.items():
            if (not isinstance(name, str) or not name or "\\" in name or Path(name).is_absolute()
                    or any(p in ("", ".", "..") for p in name.split("/"))
                    or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
                raise ValueError("Invalid portable cohort source path/hash")
    discovery = model.get("discovery", {})
    k = discovery.get("selected_k")
    if (not _integer(k) or summary.get("selected_k") != k
            or summary.get("cell_count") != sum(s["cell_count"] for s in sources)
            or summary.get("specimen_count") != len(sources)
            or summary.get("source_profiles_unchanged") is not True
            or summary.get("tissue_k_is_independent") is not True
            or model.get("contracts", {}).get("feature_representation_version") != "2.0.0"
            or discovery.get("confidence_is_calibrated_probability") is not False):
        raise ValueError("Inconsistent cohort discovery/summary contracts")
    fixed = discovery.get("fixed_k")
    if fixed is not None and (not _integer(fixed, 1) or fixed != k):
        raise ValueError("Cohort fixed K disagrees with selected K")
    if (not _integer(discovery.get("eligible_cells"))
            or discovery["eligible_cells"] > summary["cell_count"]
            or (k == 0) != (discovery["eligible_cells"] == 0)):
        raise ValueError("Invalid cohort eligible population/selected K")
    scaling = discovery.get("scaling")
    groups = model.get("feature_groups")
    if (not isinstance(groups, dict) or not isinstance(scaling, dict)
            or not set(scaling) <= set(groups)):
        raise ValueError("Invalid cohort fitted feature-group definitions")
    dimension = 0
    for name, definition in scaling.items():
        axes = definition.get("matrix_columns")
        columns, missing = definition.get("columns"), definition.get("missing_indicator_columns")
        if (not isinstance(columns, list) or not isinstance(missing, list)
                or not isinstance(groups[name], list)
                or not set(columns + missing) <= set(groups[name])
                or len(columns) != len(set(columns)) or len(missing) != len(set(missing))
                or axes != ([{"kind": "standardized_value", "source_column": c} for c in columns]
                            + [{"kind": "missing_indicator", "source_column": c} for c in missing])
                or not _integer(definition.get("dimensions"), 1) or definition["dimensions"] != len(axes)):
            raise ValueError("Cohort fitted feature axes are inconsistent")
        for key in ("means", "std"):
            value = np.asarray(definition.get(key, []))
            if value.shape != (len(columns),) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError("Invalid cohort fitted feature scaling")
            if key == "std" and (value <= 0).any():
                raise ValueError("Invalid cohort fitted feature scale")
        dimension += len(axes)
    if summary.get("feature_dimensions") != dimension or discovery.get("feature_dimensions", 0) != dimension:
        raise ValueError("Cohort model feature dimension differs from fitted axes")
    if k:
        centers = np.asarray(discovery.get("centroids_scaled"))
        if centers.shape != (k, dimension) or centers.dtype.kind not in "fiu" or not np.isfinite(centers).all():
            raise ValueError("Cohort centroid shape differs from fitted feature axes")
    elif "centroids_scaled" in discovery:
        raise ValueError("Empty cohort discovery must not claim fitted centroids")
    if summary.get("phenotype_composition_policy") != model.get("phenotype_composition_policy"):
        raise ValueError("Cohort phenotype interpretation differs between model and summary")
    return model, summary, completion, files


def _source(root, model, sample_id):
    manifest = _json(_inside(root, "cell_profiles_manifest.json"))
    sample = manifest.get("sample_id")
    if not _literal(sample) or (sample_id is not None and sample_id != sample):
        raise ValueError("Requested cohort specimen differs from source profile")
    matches = [source for source in model["sources"] if source["sample_id"] == sample]
    if len(matches) != 1:
        raise ValueError("Source specimen is absent from cohort model")
    source = matches[0]
    files = {"cell_profiles_manifest.json": source["manifest_sha256"], **source["files"],
             **source["graphs"], **source.get("store_files", {})}
    _hashes(root, files)
    if manifest.get("cell_count") != source["cell_count"] or manifest.get("observation_unit") != "cell":
        raise ValueError("Cohort source profile count/unit mismatch")
    if any(manifest.get("files", {}).get(name) != value for name, value in source["files"].items()):
        raise ValueError("Cohort source table identities differ from manifest")
    graphs = {value["path"]: value.get("sha256") for value in manifest.get("spatial_graphs", {}).values()}
    if graphs != source["graphs"]:
        raise ValueError("Cohort source graph identities differ from manifest")
    neighborhood = _json(_inside(root, "neighborhood_summary.json"))
    for key, value in model["contracts"].items():
        if neighborhood.get(key) != value:
            raise ValueError(f"Cohort source neighbourhood contract differs: {key}")
    if model.get("feature_group_definitions_by_specimen", {}).get(sample) != neighborhood.get("feature_group_definitions"):
        raise ValueError("Cohort source feature aggregation definitions differ")
    if manifest.get("neighborhood_feature_store") is not None:
        from neighborhood_feature_io import FeatureColumns
        columns = FeatureColumns(root, manifest=manifest)
        expected = {name: value for name, value in columns.source_files.items() if name not in
                    {"cell_profiles_manifest.json", "cell_profiles.parquet", "feature_rows.csv"}}
        if source.get("store_files") != expected:
            raise ValueError("Cohort source array-backed feature inventory differs from model")
        declared = neighborhood.get("feature_groups", {})
        if any(group not in declared or definition["columns"] != declared[group]
               for group, definition in columns.groups.items()):
            raise ValueError("Cohort source array-backed feature groups differ")
    elif "store_files" in source:
        raise ValueError("Cohort model requires an absent array-backed feature store")
    for name in neighborhood.get("selected_cell_feature_groups", []):
        block = manifest.get("feature_blocks", {}).get(name, {})
        shape = block.get("shape")
        if not isinstance(shape, list) or len(shape) != 2 or shape[0] != source["cell_count"]:
            raise ValueError("Cohort source feature shape differs")
        definition = {"feature_definition": block.get("feature_definition"),
                      "feature_names": block.get("feature_names"), "dimensions": shape[1]}
        if (block.get("reference_compatible") is not True
                or model.get("source_feature_definitions", {}).get(name) != definition):
            raise ValueError("Cohort source portable feature definition differs")
        path, checksum = block.get("path"), block.get("sha256")
        if path in files and files[path] != checksum:
            raise ValueError("Conflicting cohort source artifact hashes")
        files[path] = checksum
    _hashes(root, files)
    parquet = pq.ParquetFile(_inside(root, "cell_profiles.parquet"))
    names = parquet.schema_arrow.names
    needed = KEYS + GEOMETRY + [c for c in names if c.endswith("_neighbor_count") or c in ("niche_id", "niche_status")]
    if len(set(names)) != len(names) or not set(needed) <= set(names) or set(COHORT_COLUMNS) & set(names):
        raise ValueError("Invalid or already cohort-attached source profile schema")
    frame = parquet.read(columns=needed).to_pandas()
    _identity(frame)
    if len(frame) != source["cell_count"] or not frame.sample_id.eq(sample).all():
        raise ValueError("Source profile cells differ from cohort specimen")
    # Literal CSV reading also rejects short/long rows and duplicate headers.
    with _inside(root, "feature_rows.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != KEYS:
            raise ValueError("Invalid cohort feature_rows header")
        for expected in frame[KEYS].itertuples(index=False, name=None):
            if next(reader, None) != list(expected):
                raise ValueError("Cohort feature_rows identity/order mismatch")
        if next(reader, None) is not None:
            raise ValueError("Cohort feature_rows identity/order mismatch")
    return frame, source, files


def _assignments(bundle, model, summary, source_frame):
    parquet = pq.ParquetFile(_inside(bundle, ASSIGNMENTS))
    required = KEYS + ["source_row_index"] + GEOMETRY + COHORT_COLUMNS
    optional = {"source_niche_id", "source_niche_status"}
    columns = parquet.schema_arrow.names
    if (len(columns) != len(set(columns)) or not set(required) <= set(columns)
            or set(columns) - set(required) - optional):
        raise ValueError("Invalid cohort assignment schema")
    sample = source_frame.sample_id.iloc[0]
    sources = {source["sample_id"]: source for source in model["sources"]}
    offsets, offset = {}, 0
    for name, source in sources.items():
        offsets[name] = offset
        offset += source["cell_count"]
    seen, uids, cell_ids, counts, selected, eligible = 0, set(), set(), {}, [], 0
    k = model["discovery"]["selected_k"]
    fixed = model["discovery"].get("fixed_k")
    for batch in parquet.iter_batches(batch_size=ASSIGNMENT_BATCH_ROWS):
        frame = batch.to_pandas()
        _identity(frame)
        if not frame.sample_id.isin(sources).all():
            raise ValueError("Foreign specimen in cohort assignments")
        if (frame.source_row_index.dtype.kind not in "iu" or frame.source_row_index.isna().any()
                or (frame.source_row_index < 0).any()
                or (frame.source_row_index >= frame.sample_id.map(lambda s: sources[s]["cell_count"])).any()):
            raise ValueError("Invalid cohort source row indices")
        ordinal = frame.source_row_index.to_numpy() + frame.sample_id.map(offsets).to_numpy()
        if not np.array_equal(ordinal, np.arange(seen, seen + len(frame))):
            raise ValueError("Cohort assignment source row order/count differs")
        pairs = list(zip(frame.sample_id, frame.cell_id))
        if uids.intersection(frame.cell_uid) or cell_ids.intersection(pairs):
            raise ValueError("Duplicate cohort identities across batches/specimens")
        uids.update(frame.cell_uid); cell_ids.update(pairs)
        seen += len(frame)
        if (frame.in_tissue_support.dtype.kind != "b" or frame.in_tissue_support.isna().any()
                or any(frame[c].dtype.kind not in "fiu" for c in ("x_um", "y_um"))
                or not np.isfinite(frame[["x_um", "y_um"]].to_numpy(float)).all()):
            raise ValueError("Invalid cohort support/coordinate values")
        if not frame.cohort_niche_model_id.eq(model["cohort_niche_model_id"]).all():
            raise ValueError("Assignment cohort model ID differs")
        status = frame.cohort_niche_status
        if not status.isin(ASSIGNED | UNASSIGNED).all():
            raise ValueError("Invalid cohort niche status")
        assigned = status.isin(ASSIGNED)
        ids = frame.cohort_niche_id
        if ids.dtype.kind not in "iu" or not ids.notna().equals(assigned):
            raise ValueError("Cohort niche ID/status mismatch")
        if ((ids[assigned] < 1).any() or (ids[assigned] > k).any()
                or not status.eq("outside_tissue_support").equals(~frame.in_tissue_support)):
            raise ValueError("Cohort niche ID/support mismatch")
        expected_status = ("assigned_fixed_k" if fixed is not None else
                           "single_niche_no_supported_subdivision" if k == 1 else "assigned_exploratory")
        if not status[assigned].eq(expected_status).all():
            raise ValueError("Cohort assignment status differs from discovery policy")
        for column in ("cohort_niche_centroid_margin", "cohort_niche_stability"):
            if frame[column].dtype.kind not in "fiu":
                raise ValueError("Cohort niche scores must be numerical")
            scores = frame[column].to_numpy(float)
            quantified = assigned.to_numpy() & (k > 1)
            if (not np.isnan(scores[~quantified]).all() or not np.isfinite(scores[quantified]).all()
                    or (scores[quantified] < -1e-12).any() or (scores[quantified] > 1 + 1e-12).any()):
                raise ValueError("Invalid cohort niche score/missingness semantics")
        eligible += int(assigned.sum())
        for name, count in status.value_counts().items():
            counts[name] = counts.get(name, 0) + int(count)
        selected.append(frame.loc[frame.sample_id.eq(sample)].copy())
    if (seen != summary["cell_count"] or seen != parquet.metadata.num_rows
            or counts != summary.get("status_counts") or eligible != model["discovery"].get("eligible_cells")):
        raise ValueError("Cohort assignment counts differ from model/summary")
    frame = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=columns)
    if not frame[KEYS].equals(source_frame[KEYS]):
        raise ValueError("Cohort assignment identities/order differ from current profile")
    for column in GEOMETRY:
        if not np.array_equal(frame[column].to_numpy(), source_frame[column].to_numpy()):
            raise ValueError("Cohort assignment geometry/support differs from current profile")
    count_columns = [c for c in source_frame if c.endswith("_neighbor_count")]
    if not count_columns:
        raise ValueError("Source profile lacks neighbourhood counts")
    values = source_frame[count_columns].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Source profile has invalid neighbourhood counts")
    expected = source_frame.in_tissue_support & (values.max(axis=1) > 0)
    if not frame.cohort_niche_status.isin(ASSIGNED).equals(expected):
        raise ValueError("Cohort assignment eligibility differs from current profile")
    for column in ("niche_id", "niche_status"):
        original = "source_" + column
        if column in source_frame:
            if original not in frame or not ((frame[original] == source_frame[column]) |
                    (frame[original].isna() & source_frame[column].isna())).fillna(False).all():
                raise ValueError("Cohort source niche labels differ from current profile")
        elif original in frame and frame[original].notna().any():
            raise ValueError("Cohort claims source niche labels absent from current profile")
    return frame[KEYS + COHORT_COLUMNS].copy()


def verify_cohort_sources(bundle, profile_dir, record):
    """Recheck receipt-anchored bytes; never refit or follow upstream paths."""
    _hashes(Path(bundle).resolve(), record["bundle_files"])
    _hashes(Path(profile_dir).resolve(), record["source_files"])


def load_cohort_bundle(bundle, *, profile_dir, sample_id=None):
    """Return exact-source-ordered assignments and portable interpretation metadata."""
    bundle, root = Path(bundle).resolve(), Path(profile_dir).resolve()
    model, summary, completion, bundle_files = _metadata(bundle)
    source_frame, source, source_files = _source(root, model, sample_id)
    frame = _assignments(bundle, model, summary, source_frame)
    record = {"format": FORMAT, "schema_version": VERSION,
              "cohort_niche_model_id": model["cohort_niche_model_id"],
              "sample_id": source["sample_id"], "cell_count": source["cell_count"],
              "source_identity": source, "source_files": source_files, "bundle_files": bundle_files,
              "model": model, "summary": summary, "completion": completion,
              "verification": "exact bundle/source bytes, canonical identities/order, geometry, support, eligibility and status semantics; other source directories not read; no refit",
              "interpretation": "Shared exploratory niches, separate from specimen discovery and frozen reference assignments; margins/stability are not calibrated biological probabilities."}
    verify_cohort_sources(bundle, root, record)
    return frame, record
