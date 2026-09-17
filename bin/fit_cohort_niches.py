#!/usr/bin/env python3
"""Discover shared niches from immutable, specimen-specific spatial profiles.

This is pooled exploratory discovery, not frozen-reference assignment. Physical
graphs are verified and referenced, never concatenated into a cross-slide graph.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import tempfile
from urllib.parse import unquote

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from analyze_cell_neighborhoods import discover_niches
from cell_profile_io import sha256_file


FORMAT = "cellphenotyper_cohort_niches"
VERSION = "1.0.0"
KEYS = ["sample_id", "cell_id", "cell_uid"]
CANONICAL_STORE_FILES = {"cell_profiles_manifest.json", "cell_profiles.parquet", "feature_rows.csv"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def contained(root, name):
    relative = Path(name)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping cohort input artifact: {name}")
    return path


def checked_file(root, name, expected):
    path = contained(root, name)
    if not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f"Cohort input SHA256 mismatch: {name}")
    return path


def source_record(root):
    root = Path(root).resolve()
    manifest_path = contained(root, "cell_profiles_manifest.json")
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", {})
    paths = {name: checked_file(root, name, files.get(name)) for name in
             ("cell_profiles.parquet", "feature_rows.csv", "neighborhood_summary.json")}
    summary = json.loads(paths["neighborhood_summary.json"].read_text())
    if summary.get("feature_representation_version") != "2.0.0":
        raise ValueError("Cohort discovery requires spatial representation 2.0.0; reassemble legacy profiles separately")
    sample = manifest.get("sample_id")
    if not isinstance(sample, str) or not sample.strip() or summary.get("specimens") != [sample]:
        raise ValueError("Cohort input must identify exactly one nonempty literal specimen")
    count = manifest.get("cell_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1 or pq.ParquetFile(paths["cell_profiles.parquet"]).metadata.num_rows != count:
        raise ValueError("Cohort source cell count mismatch")
    groups = summary.get("feature_groups")
    if not isinstance(groups, dict) or not groups or any(not isinstance(v, list) or len(v) != len(set(v)) or any(not isinstance(c, str) or not c for c in v) for v in groups.values()):
        raise ValueError("Missing or invalid neighbourhood feature group definitions")
    definitions = summary.get("feature_group_definitions", {})
    if set(definitions) != set(groups) or any(definitions[name].get("columns") != columns for name, columns in groups.items()):
        raise ValueError("Cohort feature groups disagree with exact producer definitions")
    selected = summary.get("selected_cell_feature_groups", [])
    derived = {definition.get("source_feature_block") for definition in definitions.values()} - {None}
    if set(selected) != derived:
        raise ValueError("Selected cohort feature blocks disagree with producer group definitions")
    raw_radii = summary.get("radii_um")
    try:
        radii = [float(value) for value in raw_radii]
    except (TypeError, ValueError):
        raise ValueError("Cohort graph radii must be finite, positive and distinct") from None
    if (not isinstance(raw_radii, list) or not radii or any(isinstance(value, bool) for value in raw_radii)
            or any(not math.isfinite(value) or value <= 0 for value in radii) or len(set(radii)) != len(radii)):
        raise ValueError("Cohort graph radii must be finite, positive and distinct")
    graph_files, graph_paths = {}, {}
    if not manifest.get("spatial_graphs"):
        raise ValueError("Cohort source lacks specimen-specific spatial graphs")
    for radius, record in manifest["spatial_graphs"].items():
        try:
            radius_value = float(radius)
        except (TypeError, ValueError):
            raise ValueError("Cohort graph radius differs from declared neighbourhood radii") from None
        if not math.isfinite(radius_value) or radius_value <= 0 or radius_value in graph_paths:
            raise ValueError("Cohort graph radius differs from declared neighbourhood radii")
        path = checked_file(root, record["path"], record.get("sha256"))
        graph = sparse.load_npz(path)
        if (graph.format != "csr" or graph.shape != (count, count) or graph.dtype.kind not in "fiu"
                or not np.isfinite(graph.data).all() or (graph.data < 0).any()):
            raise ValueError("Invalid specimen-specific cohort source graph")
        graph.check_format(full_check=True)
        canonical_graph = graph.copy()
        canonical_graph.sum_duplicates()
        if canonical_graph.nnz != graph.nnz:
            raise ValueError("Cohort graph contains duplicate stored edges")
        # Zero-distance edges are real neighbours. Numeric subtraction alone
        # cannot detect a zero edge whose reverse entry is absent.
        adjacency = graph.copy()
        adjacency.data = np.ones(graph.nnz, dtype=np.int8)
        coo = graph.tocoo()
        if ((coo.row == coo.col).any() or (adjacency - adjacency.T).nnz or (graph - graph.T).nnz
                or (graph.data > radius_value + 1e-9).any()):
            raise ValueError("Cohort graph violates radius/symmetry/self-edge contract")
        graph_files[record["path"]] = record["sha256"]
        graph_paths[radius_value] = path
    if set(graph_paths) != set(radii):
        raise ValueError("Cohort graph radius set differs from declared neighbourhood radii")
    identity = {"sample_id": sample, "cell_count": count, "manifest_sha256": manifest_hash,
                "files": {name: files[name] for name in paths}, "graphs": graph_files}
    result = {"root": root, "manifest": manifest, "summary": summary, "groups": groups,
              "paths": paths, "identity": identity, "graph_paths": graph_paths}
    if manifest.get("neighborhood_feature_store") is not None:
        from neighborhood_feature_io import FeatureColumns
        columns = FeatureColumns(root, manifest=manifest)
        if any(group not in groups or definition["columns"] != groups[group]
               for group, definition in columns.groups.items()):
            raise ValueError("Cohort array-backed feature groups differ from neighbourhood definitions")
        result["columns"] = columns
        identity["store_files"] = {name: value for name, value in columns.source_files.items()
                                   if name not in CANONICAL_STORE_FILES}
    return result


def compatibility(records):
    """Align declared feature axes, never guess from arbitrary numeric columns."""
    contracts = ("feature_representation_version", "radii_um", "graph_contract", "density_contract",
                 "orientation_contract", "phenotype_unknown_category", "empty_neighborhood_fractions")
    first = records[0]["summary"]
    for record in records[1:]:
        for key in contracts:
            if key not in first or canonical(record["summary"].get(key)) != canonical(first[key]):
                raise ValueError(f"Incompatible neighbourhood contract: {key}")
    groups, feature_definitions, group_definitions = {}, {}, {}
    for record in records:
        for group, columns in record["groups"].items():
            groups.setdefault(group, set()).update(columns)
            definition = dict(record['summary']['feature_group_definitions'][group])
            # Different observed phenotype vocabularies align to a union. No
            # other source axis or aggregation is silently changed.
            if group == 'composition':
                definition.pop('columns', None)
                definition.pop('phenotypes', None)
            if group in group_definitions and canonical(definition) != canonical(group_definitions[group]):
                raise ValueError(f"Incompatible feature aggregation definition: {group}")
            group_definitions[group] = definition
        for name in record["summary"].get("selected_cell_feature_groups", []):
            definition = record["manifest"].get("feature_blocks", {}).get(name)
            if not definition or definition.get("reference_compatible") is not True or not definition.get("feature_definition"):
                raise ValueError(f"Unverified source feature definition cannot be pooled across slides: {name}")
            value = {"feature_definition": definition["feature_definition"],
                     "feature_names": definition.get("feature_names"), "dimensions": definition["shape"][1]}
            if name in feature_definitions and canonical(value) != canonical(feature_definitions[name]):
                raise ValueError(f"Incompatible cross-slide feature definition: {name}")
            feature_definitions[name] = value
            checked_file(record["root"], definition["path"], definition.get("sha256"))
    return {key: sorted(value) for key, value in sorted(groups.items())}, feature_definitions, {key: first[key] for key in contracts}


def composition_policy(records):
    """Do not equate detector labels solely because their strings match."""
    definitions, unverified, declared_unknown = {}, [], set()
    for record in records:
        sample = record['identity']['sample_id']
        summary = record['summary']
        labels = set(summary['feature_group_definitions'].get('composition', {}).get('phenotypes', []))
        informative = labels - {summary['phenotype_unknown_category'], 'unknown'}
        phenotype = record['manifest'].get('phenotype_definition', {})
        if (phenotype.get('verified') is True and phenotype.get('definition', {}).get('unknown_label_semantics') ==
                'unknown, __unknown__, and producer unknown_<unmapped_type_id> are missing categorical assignments'):
            declared_unknown.update(label for label in informative if label.startswith('unknown_'))
            informative = {label for label in informative if not label.startswith('unknown_')}
        if not informative:
            continue
        if phenotype.get('verified') is not True or not phenotype.get('definition'):
            unverified.append(sample)
        else:
            definitions[sample] = phenotype['definition']
    if unverified:
        return {'included_in_shared_fit': False, 'reason': 'unverified_informative_phenotype_taxonomy',
                'unverified_specimens': unverified, 'source_profiles_preserved': True}
    if len({canonical(value) for value in definitions.values()}) > 1:
        return {'included_in_shared_fit': False, 'reason': 'incompatible_verified_phenotype_taxonomies',
                'source_profiles_preserved': True}
    return {'included_in_shared_fit': True,
            'reason': 'matched_verified_definitions' if definitions else 'all_phenotypes_unknown_no_biological_information',
            'verified_definition': next(iter(definitions.values()), None),
            'source_declared_unknown_labels': sorted(declared_unknown), 'source_profiles_preserved': True}


def normalize_missing_composition(pooled, groups, policy):
    """Coalesce explicit unknown aliases only in the temporary shared features."""
    result = {name: list(columns) for name, columns in groups.items()}
    aliases = {}
    allow_fallback = (policy.get('verified_definition') or {}).get('unknown_label_semantics') == (
        'unknown, __unknown__, and producer unknown_<unmapped_type_id> are missing categorical assignments')
    for column in result.get('composition', []):
        prefix, label = column.split('_phenotype_fraction:', 1)
        label = unquote(label)
        if label in {'unknown', '__unknown__'} or label in policy.get('source_declared_unknown_labels', []) or (allow_fallback and label.startswith('unknown_')):
            aliases.setdefault(prefix, []).append(column)
    mapping = {}
    for prefix, columns in aliases.items():
        target = prefix + '_phenotype_fraction:__unknown__'
        pooled[target] = pooled[columns].sum(axis=1, min_count=1)
        result['composition'] = [column for column in result['composition'] if column not in columns]
        result['composition'].append(target)
        mapping[target] = columns
    result['composition'] = sorted(result.get('composition', []))
    return result, mapping


class CohortValues:
    """Virtual feature union over ordered specimens, never a wide pooled frame."""
    def __init__(self, records, groups, aliases=None):
        self.records, self.groups = records, groups
        self.aliases = aliases or {}
        self.columns = {c for columns in groups.values() for c in columns}
        self.offsets = np.r_[0, np.cumsum([r["identity"]["cell_count"] for r in records])]

    def read(self, rows, columns):
        if isinstance(rows, slice):
            rows = np.arange(*rows.indices(int(self.offsets[-1])), dtype=np.int64)
        else:
            rows = np.asarray(rows)
            if rows.ndim != 1 or (rows.size and rows.dtype.kind not in "iu"):
                raise ValueError("Cohort feature rows must be integer positions")
            rows = rows.astype(np.int64, copy=False)
        if ((rows < 0) | (rows >= self.offsets[-1])).any():
            raise ValueError("Cohort feature row position out of bounds")
        requested = list(columns)
        if any(c not in self.columns and c not in self.aliases for c in requested):
            raise ValueError("Requested feature is outside the declared cohort union")
        needed = list(dict.fromkeys(c for name in requested for c in self.aliases.get(name, [name])))
        values = self._raw(rows, needed)
        index = {name: i for i, name in enumerate(needed)}
        result = np.empty((len(rows), len(requested)), dtype=np.float64)
        for target, name in enumerate(requested):
            if name not in self.aliases:
                result[:, target] = values[:, index[name]]
            else:
                block = values[:, [index[c] for c in self.aliases[name]]]
                valid = np.isfinite(block)
                total = np.where(valid, block, 0).sum(axis=1)
                total[~valid.any(axis=1)] = np.nan
                result[:, target] = total
        return result

    def _raw(self, rows, columns):
        result = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
        for i, record in enumerate(self.records):
            selected = np.flatnonzero((rows >= self.offsets[i]) & (rows < self.offsets[i + 1]))
            if not len(selected):
                continue
            local_rows = rows[selected] - self.offsets[i]
            present = {c for group in record["groups"].values() for c in group}
            available = [c for c in columns if c in present]
            if available:
                positions = [j for j, c in enumerate(columns) if c in present]
                block = record["columns"].read(local_rows, available)
                if block.shape != (len(local_rows), len(available)) or block.dtype != np.float64 or np.isinf(block).any():
                    raise ValueError("Invalid cohort source feature block")
                result[np.ix_(selected, positions)] = block
            for j, column in enumerate(columns):
                if column not in present and column in self.groups.get("composition", []):
                    count_column = column.split("_phenotype_fraction:")[0] + "_neighbor_count"
                    if count_column not in record["frame"]:
                        raise ValueError("Cannot align phenotype vocabulary without exact neighbour count")
                    count = record["frame"][count_column].to_numpy()[local_rows]
                    result[selected, j] = np.where(count > 0, 0., np.nan)
        return result


def verify_streamed_projection(record, frame, row_batch_size):
    """Recompute source projections with bounded rows, axes and edge batches."""
    from urllib.parse import quote

    reader = record["columns"]
    for name in record["summary"].get("selected_cell_feature_groups", []):
        definition = record["manifest"]["feature_blocks"][name]
        path = checked_file(record["root"], definition["path"], definition.get("sha256"))
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if (values.ndim != 2 or values.shape[0] != len(frame) or values.shape[1] < 1
                or list(values.shape) != definition.get("shape") or values.dtype.kind not in "fiu"):
            raise ValueError("Cohort source feature array geometry/dtype is invalid")
        labels = definition.get("feature_names") or [f"feature_{i}" for i in range(values.shape[1])]
        if len(labels) != values.shape[1] or len(set(map(str, labels))) != len(labels):
            raise ValueError("Cohort source feature names do not match its exact array")
        own_names = [f"own_{name}:{quote(str(label), safe='')}" for label in labels]
        if record["groups"].get(f"own:{name}") != own_names:
            raise ValueError("Cohort own feature columns differ from source array axes")
        for lo in range(0, len(frame), row_batch_size):
            hi = min(len(frame), lo + row_batch_size)
            own_count = np.zeros(hi - lo, dtype=np.int64)
            for start in range(0, values.shape[1], 64):
                end = min(start + 64, values.shape[1])
                block = np.asarray(values[lo:hi, start:end], dtype=np.float64)
                if np.isinf(block).any():
                    raise ValueError("Infinite cohort source features are invalid")
                if not np.array_equal(reader.read(slice(lo, hi), own_names[start:end]), block, equal_nan=True):
                    raise ValueError("Cohort own feature values differ from exact source array projection")
                own_count += np.isfinite(block).sum(axis=1)
            if not np.array_equal(frame[f"own_{name}_observed_fraction"].to_numpy(float)[lo:hi], own_count / values.shape[1]):
                raise ValueError("Cohort own feature coverage differs from its source array")
        expected_mean_names = []
        for radius, graph_path in sorted(record["graph_paths"].items()):
            graph = sparse.load_npz(graph_path)
            mean_names = [f"r{radius:g}um_{name}_mean:{quote(str(label), safe='')}" for label in labels]
            expected_mean_names.extend(mean_names)
            for lo in range(0, len(frame), row_batch_size):
                hi = min(len(frame), lo + row_batch_size)
                degree = np.maximum(np.diff(graph.indptr[lo:hi + 1]), 1)
                coverage_sum = np.zeros(hi - lo, dtype=np.float64)
                for start in range(0, values.shape[1], 64):
                    end = min(start + 64, values.shape[1])
                    sums = np.zeros((hi - lo, end - start), dtype=np.float64)
                    observed = np.zeros_like(sums)
                    first, last = graph.indptr[lo], graph.indptr[hi]
                    for edge in range(first, last, row_batch_size):
                        stop = min(last, edge + row_batch_size)
                        members = graph.indices[edge:stop]
                        local_rows = np.searchsorted(graph.indptr[lo:hi + 1], np.arange(edge, stop), side="right") - 1
                        block = np.asarray(values[members, start:end], dtype=np.float64)
                        finite = np.isfinite(block)
                        np.add.at(sums, local_rows, np.where(finite, block, 0.))
                        np.add.at(observed, local_rows, finite)
                    means = np.divide(sums, observed, out=np.full_like(sums, np.nan), where=observed > 0)
                    if not np.allclose(reader.read(slice(lo, hi), mean_names[start:end]), means,
                                       rtol=1e-12, atol=1e-12, equal_nan=True):
                        raise ValueError("Cohort neighbour means differ from source graph and finite source features")
                    coverage_sum += (observed / degree[:, None]).sum(axis=1)
                if not np.allclose(frame[f"r{radius:g}um_{name}_observed_fraction"].to_numpy(float)[lo:hi],
                                   coverage_sum / values.shape[1], rtol=0, atol=1e-12):
                    raise ValueError("Cohort neighbour feature coverage differs from source graph and features")
        if record["groups"].get(name) != expected_mean_names:
            raise ValueError("Cohort neighbour mean columns differ from source array axes/radii")
        checked_file(record["root"], definition["path"], definition.get("sha256"))


def pool_profiles(records, groups, *, streaming=False, row_batch_size=4096):
    from urllib.parse import quote

    frames, missing = [], {}
    all_columns = {col for columns in groups.values() for col in columns}
    for record in records:
        scalar_columns = set(pq.ParquetFile(record["paths"]["cell_profiles.parquet"]).schema_arrow.names)
        if streaming and "columns" not in record:
            from neighborhood_feature_io import FeatureColumns
            record["columns"] = FeatureColumns(record["root"], manifest=record["manifest"])
        available = set(record["columns"].available_columns) if streaming else scalar_columns
        own_columns = {col for columns in record["groups"].values() for col in columns}
        required = set(KEYS + ["in_tissue_support", "x_um", "y_um"])
        counts = {f"r{float(radius):g}um_neighbor_count" for radius in record["summary"]["radii_um"]}
        selected = record["summary"].get("selected_cell_feature_groups", [])
        coverage = {f"own_{name}_observed_fraction" for name in selected}
        coverage.update(f"r{float(radius):g}um_{name}_observed_fraction"
                        for radius in record["summary"]["radii_um"] for name in selected)
        if not (required | own_columns | counts | coverage) <= available:
            raise ValueError("Declared cohort features or canonical fields are absent from source table")
        retain = required | counts | coverage | (scalar_columns & {"niche_id", "niche_status", "phenotype_source"})
        if not retain <= scalar_columns:
            raise ValueError("Cohort canonical scalar fields must remain in the source Parquet table")
        if not streaming:
            retain |= own_columns
        frame = pd.read_parquet(record["paths"]["cell_profiles.parquet"], columns=sorted(retain))
        ids = pd.read_csv(record["paths"]["feature_rows.csv"], dtype=str, keep_default_na=False)
        if list(ids.columns) != KEYS or len(frame) != record["identity"]["cell_count"]:
            raise ValueError("Cohort canonical row inventory mismatch")
        if any(frame[key].isna().any() or not frame[key].map(lambda x: isinstance(x, str) and bool(x.strip())).all() for key in KEYS):
            raise ValueError("Cohort identifiers must be nonempty literal strings")
        if not ids.equals(frame[KEYS]) or frame.cell_uid.duplicated().any() or frame.duplicated(KEYS[:2]).any():
            raise ValueError("Cohort feature rows do not match canonical identities and order")
        if set(frame.sample_id) != {record["identity"]["sample_id"]}:
            raise ValueError("Foreign specimen in cohort profile")
        if frame.in_tissue_support.dtype.kind != "b" or frame.in_tissue_support.isna().any():
            raise ValueError("Cohort support membership must be explicit booleans")
        if not np.isfinite(frame[["x_um", "y_um"]].to_numpy(float)).all():
            raise ValueError("Invalid cohort physical coordinates")
        for column in counts:
            values = frame[column].to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
                raise ValueError("Invalid cohort neighbour counts")
        xy = frame[["x_um", "y_um"]].to_numpy(float)
        supported = frame.in_tissue_support.to_numpy(bool)
        # Source graph rows use the exact original feature_rows order verified
        # above. Check them before eligibility or any feature union is formed.
        for radius, path in record["graph_paths"].items():
            graph = sparse.load_npz(path)
            degree = np.diff(graph.indptr)
            if not np.array_equal(degree, frame[f"r{radius:g}um_neighbor_count"].to_numpy(float)):
                raise ValueError("Cohort neighbour counts differ from exact source graph degrees")
            if ((degree > 0) & ~supported).any():
                raise ValueError("Cohort graph has edges incident to unsupported cells")
            # Fixed-size edge blocks avoid an additional dense edge-by-XY
            # allocation for a large radius graph.
            for start in range(0, graph.nnz, 65536):
                end = min(start + 65536, graph.nnz)
                rows = np.searchsorted(graph.indptr, np.arange(start, end), side="right") - 1
                cols = graph.indices[start:end]
                distances = np.linalg.norm(xy[rows] - xy[cols], axis=1)
                if not np.allclose(graph.data[start:end], distances, rtol=1e-9, atol=1e-9):
                    raise ValueError("Cohort graph distances differ from canonical physical coordinates")

        # Exact source-array projection binds compatibility definitions to the
        # actual values being pooled, not just to a same-named array checksum.
        # The declared dense preflight has already passed; conversions operate
        # on <=64 feature columns at once, and one radius graph at a time.
        if streaming:
            verify_streamed_projection(record, frame, row_batch_size)
        for name in ([] if streaming else selected):
            definition = record["manifest"]["feature_blocks"][name]
            path = checked_file(record["root"], definition["path"], definition.get("sha256"))
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if (values.ndim != 2 or values.shape[0] != len(frame) or values.shape[1] < 1
                    or list(values.shape) != definition.get("shape") or values.dtype.kind not in "fiu"):
                raise ValueError("Cohort source feature array geometry/dtype is invalid")
            labels = definition.get("feature_names") or [f"feature_{i}" for i in range(values.shape[1])]
            if len(labels) != values.shape[1] or len(set(map(str, labels))) != len(labels):
                raise ValueError("Cohort source feature names do not match its exact array")
            own_names = [f"own_{name}:{quote(str(label), safe='')}" for label in labels]
            if record["groups"].get(f"own:{name}") != own_names:
                raise ValueError("Cohort own feature columns differ from source array axes")
            own_count = np.zeros(len(frame), dtype=np.int64)
            for start in range(0, values.shape[1], 64):
                end = min(start + 64, values.shape[1])
                block = np.asarray(values[:, start:end], dtype=float)
                if np.isinf(block).any():
                    raise ValueError("Infinite cohort source features are invalid")
                if not np.array_equal(frame[own_names[start:end]].to_numpy(float), block, equal_nan=True):
                    raise ValueError("Cohort own feature values differ from exact source array projection")
                own_count += np.isfinite(block).sum(axis=1)
            if not np.array_equal(frame[f"own_{name}_observed_fraction"].to_numpy(float), own_count / values.shape[1]):
                raise ValueError("Cohort own feature coverage differs from its source array")
            expected_mean_names = []
            for radius, graph_path in sorted(record["graph_paths"].items()):
                adjacency = sparse.load_npz(graph_path)
                adjacency.data = np.ones(adjacency.nnz, dtype=float)
                # Keep the stored reduction order: graph remapping by assembly
                # can legitimately leave unsorted column indices. Sparse
                # kernels/vector blocking may still differ by a few float64
                # ULPs, so means permit only 1e-12 absolute/relative roundoff.
                denominator = np.maximum(np.diff(adjacency.indptr), 1)
                mean_names = [f"r{radius:g}um_{name}_mean:{quote(str(label), safe='')}" for label in labels]
                expected_mean_names.extend(mean_names)
                coverage_sum = np.zeros(len(frame), dtype=float)
                for start in range(0, values.shape[1], 64):
                    end = min(start + 64, values.shape[1])
                    block = np.asarray(values[:, start:end], dtype=float)
                    finite = np.isfinite(block)
                    observed = adjacency @ finite.astype(float)
                    sums = adjacency @ np.where(finite, block, 0)
                    means = np.divide(sums, observed, out=np.full_like(sums, np.nan), where=observed > 0)
                    if not np.allclose(frame[mean_names[start:end]].to_numpy(float), means,
                                       rtol=1e-12, atol=1e-12, equal_nan=True):
                        raise ValueError("Cohort neighbour means differ from source graph and finite source features")
                    coverage_sum += (observed / denominator[:, None]).sum(axis=1)
                if not np.allclose(frame[f"r{radius:g}um_{name}_observed_fraction"].to_numpy(float),
                                   coverage_sum / values.shape[1], rtol=0, atol=1e-12):
                    raise ValueError("Cohort neighbour feature coverage differs from source graph and features")
            if record["groups"].get(name) != expected_mean_names:
                raise ValueError("Cohort neighbour mean columns differ from source array axes/radii")
            checked_file(record["root"], definition["path"], definition.get("sha256"))
        added = {}
        for column in ([] if streaming else sorted(all_columns - own_columns)):
            if column in groups.get("composition", []) and "_phenotype_fraction:" in column:
                count_column = column.split("_phenotype_fraction:")[0] + "_neighbor_count"
                if count_column not in frame:
                    raise ValueError("Cannot align phenotype vocabulary without exact neighbour count")
                added[column] = np.where(frame[count_column] > 0, 0., np.nan)
            else:
                added[column] = np.full(len(frame), np.nan)
        if added:
            frame = pd.concat([frame, pd.DataFrame(added, index=frame.index)], axis=1)
        if streaming:
            record["frame"] = frame
            virtual = CohortValues([record], groups)
            missing_count = 0
            columns = sorted(all_columns)
            for lo in range(0, len(frame), row_batch_size):
                for start in range(0, len(columns), 64):
                    values = virtual.read(slice(lo, min(lo + row_batch_size, len(frame))), columns[start:start + 64])
                    missing_count += int(np.isnan(values).sum())
        else:
            values = frame[sorted(all_columns)].to_numpy(float)
            if np.isinf(values).any():
                raise ValueError("Infinite cohort features are invalid; missing features must be NaN")
            missing_count = int(np.isnan(values).sum())
        frame["source_row_index"] = np.arange(len(frame))
        missing[record["identity"]["sample_id"]] = {
            "absent_feature_groups": sorted(set(groups) - set(record["groups"])),
            "missing_feature_values": missing_count,
            "phenotype_sources": sorted(frame.phenotype_source.unique().tolist()) if "phenotype_source" in frame else ["unverified"],
        }
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    if pooled.cell_uid.duplicated().any() or pooled.duplicated(KEYS[:2]).any():
        raise ValueError("Repeated canonical identities across cohort specimens")
    return pooled, missing


def verify_sources(records):
    """Recheck only captured source identities, before and after output writes."""
    for record in records:
        identity = record["identity"]
        checked_file(record["root"], "cell_profiles_manifest.json", identity["manifest_sha256"])
        for name, expected in {**identity["files"], **identity["graphs"], **identity.get("store_files", {})}.items():
            checked_file(record["root"], name, expected)
        if "columns" in record:
            record["columns"].recheck()
        for name in record['summary'].get('selected_cell_feature_groups', []):
            feature = record['manifest']['feature_blocks'][name]
            checked_file(record['root'], feature['path'], feature['sha256'])


def fit_cohort(profiles, outdir, *, max_k=8, fixed_k=None, seed=17, repeats=5,
               fit_limit=20000, feature_weights=None, max_working_mb=1024, row_batch_size=4096):
    outdir = Path(outdir)
    if outdir.exists():
        raise FileExistsError("Cohort discovery requires a new output directory")
    if not math.isfinite(max_working_mb) or max_working_mb <= 0:
        raise ValueError("max_working_mb must be positive and finite")
    if not isinstance(row_batch_size, int) or isinstance(row_batch_size, bool) or row_batch_size < 1:
        raise ValueError("row_batch_size must be a positive integer")
    records = sorted([source_record(path) for path in profiles], key=lambda r: r["identity"]["sample_id"])
    samples = [record["identity"]["sample_id"] for record in records]
    if len(samples) < 2 or len(samples) != len(set(samples)):
        raise ValueError("Cohort discovery requires at least two distinct specimens, each supplied exactly once")
    groups, feature_definitions, contracts = compatibility(records)
    streaming = any("store_files" in record["identity"] for record in records)
    # Account for tables, scaling, missingness augmentation, training and model
    # copies. This bounds our dense pooled representation, not interpreter RSS.
    n = sum(record["identity"]["cell_count"] for record in records)
    d = len({c for columns in groups.values() for c in columns})
    estimated = (n * max(1, d) * 96 + n * max(max_k, fixed_k or 1) * 32) / 1024**2
    if not streaming and estimated > max_working_mb:
        raise ValueError(f"Cohort dense working estimate {estimated:.1f} MiB exceeds {max_working_mb:g} MiB; use a larger declared budget or fewer explicitly selected feature groups")
    if streaming:
        projection_mb = min(row_batch_size, n) * min(64, max(1, d)) * 96 / 1024**2
        if projection_mb > max_working_mb:
            raise ValueError(f"Cohort projection working estimate {projection_mb:.1f} MiB exceeds {max_working_mb:g} MiB; lower row_batch_size or increase the declared budget")
    pooled, missing = pool_profiles(records, groups, streaming=streaming, row_batch_size=row_batch_size)
    phenotype_policy = composition_policy(records)
    fitting_groups = dict(groups)
    if not phenotype_policy['included_in_shared_fit']:
        fitting_groups.pop('composition', None)
        if 'composition' in (feature_weights or {}):
            raise ValueError('Requested composition weight requires matched verified phenotype definitions')
    else:
        # An empty frame derives the exact same alias plan without adding a
        # wide temporary representation to the streaming canonical registry.
        alias_frame = pd.DataFrame(columns=groups.get("composition", []), dtype=float) if streaming else pooled
        fitting_groups, aliases = normalize_missing_composition(alias_frame, fitting_groups, phenotype_policy)
        phenotype_policy['temporary_unknown_alias_columns'] = aliases
    fit_options = dict(max_k=max_k, fixed_k=fixed_k, seed=seed, repeats=repeats,
                       fit_limit=fit_limit, feature_weights=feature_weights)
    if streaming:
        from niche_clustering import discover_niches_streaming
        virtual = CohortValues(records, groups, phenotype_policy.get('temporary_unknown_alias_columns'))
        # The child must not exist on entry. Only this caller-owned scratch
        # tree is removed; canonical sources and completed outputs are intact.
        with tempfile.TemporaryDirectory(prefix="cellphenotyper-cohort-stream-") as scratch:
            niches, discovery = discover_niches_streaming(pooled, fitting_groups, virtual.read,
                Path(scratch) / "fit", max_working_mb=max_working_mb, row_batch_size=row_batch_size, **fit_options)
    else:
        niches, discovery = discover_niches(pooled, fitting_groups, **fit_options)
    if not niches[KEYS[:2]].equals(pooled[KEYS[:2]]):
        raise ValueError("Cohort discovery changed canonical row order")
    runtime = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow")}
    producer = {name: sha256_file(Path(__file__).with_name(name)) for name in
                ("fit_cohort_niches.py", "analyze_cell_neighborhoods.py", "cell_profile_io.py")}
    if streaming:
        producer.update({name: sha256_file(Path(__file__).with_name(name)) for name in
                         ("neighborhood_feature_io.py", "niche_clustering.py")})
    model = {"format": FORMAT, "schema_version": VERSION, "analysis": "pooled_exploratory_discovery",
             "sources": [r["identity"] for r in records], "contracts": contracts,
             "feature_groups": fitting_groups, "available_feature_groups": groups, "source_feature_definitions": feature_definitions,
             "feature_group_definitions_by_specimen": {r['identity']['sample_id']: r['summary']['feature_group_definitions'] for r in records},
             "discovery": discovery, "runtime": runtime, "producer": producer,
             "feature_weights": feature_weights or {}, "max_working_mb": max_working_mb,
             "sample_weighting": "per-cell; unequal specimen counts can influence discovery",
             "phenotype_composition_policy": phenotype_policy,
             "reference_assignment": False, "calibrated_biological_confidence": False}
    model["cohort_niche_model_id"] = digest(model)
    assignments = pooled[KEYS + ["source_row_index", "x_um", "y_um", "in_tissue_support"]].copy()
    for column in ("niche_id", "niche_status"):
        if column in pooled:
            assignments["source_" + column] = pooled[column]
    for column in niches:
        if column not in KEYS:
            assignments["cohort_" + column] = niches[column]
    assignments["cohort_niche_model_id"] = model["cohort_niche_model_id"]
    summary = {"format": FORMAT, "schema_version": VERSION, "cohort_niche_model_id": model["cohort_niche_model_id"],
               "cell_count": n, "specimen_count": len(samples), "selected_k": discovery["selected_k"],
               "feature_dimensions": discovery.get("feature_dimensions", 0), "estimated_dense_working_mb": estimated,
               "missingness": missing, "phenotype_composition_policy": phenotype_policy,
               "status_counts": assignments.cohort_niche_status.value_counts().to_dict(),
               "graph_policy": "No graph is constructed or changed. Each verified source graph contains only its own specimen.",
               "source_profiles_unchanged": True, "tissue_k_is_independent": True,
               "verification_tolerances": {"own_features": "exact including NaNs",
                   "neighbor_means_rtol_atol": [1e-12, 1e-12], "coverage_atol": 1e-12,
                   "graph_distance_um_rtol_atol": [1e-9, 1e-9]},
               "limitations": ["Exploratory shared niches, not independently validated biological identities.",
                   "Missingness can influence clustering; inspect coverage before interpreting niches.",
                   "No stain/site batch correction is inferred or applied.",
                   "Phenotype labels from different detector taxonomies must not be treated as interchangeable."]}
    if streaming:
        summary["streaming_execution"] = discovery["streaming_execution"]
        summary["estimated_dense_working_mb_scope"] = "Informational legacy comparison; no dense cohort feature matrix was allocated"
    verify_sources(records)
    outdir.mkdir(parents=True)
    assignments.to_parquet(outdir / "cohort_niche_assignments.parquet", index=False)
    for filename, data in (("cohort_niche_model.json", model), ("cohort_niche_summary.json", summary)):
        (outdir / filename).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    completion = {"format": FORMAT, "schema_version": VERSION, "cohort_niche_model_id": model["cohort_niche_model_id"],
                  "files": {name: sha256_file(outdir / name) for name in
                      ("cohort_niche_assignments.parquet", "cohort_niche_model.json", "cohort_niche_summary.json")}}
    verify_sources(records)
    (outdir / "cohort_niches_completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    return assignments, model, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--fixed-k", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fit-limit", type=int, default=20000)
    parser.add_argument("--max-working-mb", type=float, default=1024)
    parser.add_argument("--row-batch-size", type=int, default=4096)
    parser.add_argument("--feature-weight", action="append", default=[], metavar="GROUP=WEIGHT")
    parser.add_argument("--feature-weights", default="{}", help="JSON object of separate group weights")
    args = vars(parser.parse_args())
    weights = json.loads(args.pop('feature_weights'))
    if not isinstance(weights, dict):
        parser.error('--feature-weights must be a JSON object')
    for item in args.pop("feature_weight"):
        name, value = item.rsplit("=", 1)
        if name in weights:
            parser.error(f"Repeated feature weight: {name}")
        weights[name] = float(value)
    args["feature_weights"] = weights
    _, _, summary = fit_cohort(**args)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
