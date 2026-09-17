"""Bounded own-cell/finite-neighbour arrays in canonical source row order.

Graph traversal retains the analyzer's sorted-cell order, including stored
zero-distance edges. Payloads are memory-mapped NPY files, not compressed or
chunked Zarr. No feature axis or canonical cell is sampled or discarded.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from analyze_cell_neighborhoods import KEYS, feature_group_definitions
from cell_profile_io import sha256_file
from neighborhood_feature_io import _numeric_float64


def build_arrays(root, profiles, graphs, canonical_to_sorted, blocks, summary, *,
                 row_batch_size=4096, column_batch_size=64, max_working_mb=1024):
    """Return narrow profiles, store records and a sorted-order fitting reader.

``blocks`` maps each source block to its verified manifest record and mmap.
The scalar profiles arrive in canonical order; graph rows are analyzer-sorted.
Read callbacks accept analyzer-sorted row indices, just like legacy fitting.
    """
    root = Path(root)
    n = len(profiles)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (row_batch_size, column_batch_size)):
        raise ValueError("Neighbourhood row/column batch sizes must be positive integers")
    row_batch_size = min(row_batch_size, max(n, 1))
    dimensions = max((array.shape[1] for _, array in blocks.values()), default=1)
    columns_per_batch = min(column_batch_size, dimensions)
    # Row sums/counts/means, edge values/masks/zero-filled values and coverage
    # vectors. Sparse graphs and the narrow scalar table are separate O(E)/O(N)
    # structures, not an all-features dense representation.
    estimate = (row_batch_size * columns_per_batch * 96 + n * 40) / 1024**2
    if estimate > max_working_mb:
        raise ValueError(f"Array aggregation working estimate {estimate:.1f} MiB exceeds {max_working_mb:g} MiB; reduce explicit batch sizes or increase budget")
    sorted_to_canonical = np.argsort(canonical_to_sorted)
    if not np.array_equal(np.sort(canonical_to_sorted), np.arange(n)):
        raise ValueError("Array aggregation requires an exact canonical/sorted permutation")
    groups = summary["feature_groups"]
    store, locations, arrays, fields, definitions = {}, {}, {}, {}, {}
    for name, (record, values) in sorted(blocks.items()):
        labels = record.get("feature_names") or [f"feature_{i}" for i in range(values.shape[1])]
        if (not name or name in {"density", "composition", "orientation"}
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)):
            raise ValueError("Neighbourhood block name is invalid or reserved")
        if (len(labels) != values.shape[1] or not labels or len(set(map(str, labels))) != len(labels)
                or any(label in KEYS for label in labels)):
            raise ValueError("Neighbourhood source feature names must be unique non-key axes")
        labels = list(map(str, labels))
        definitions[name] = KEYS + labels
        d = len(labels)
        own = [f"own_{name}:{quote(label, safe='')}" for label in labels]
        own_group = f"own:{name}"
        groups[own_group], groups[name] = own, []
        arrays[record["path"]] = values
        for axis, column in enumerate(own):
            locations[column] = (record["path"], axis)
        store[own_group] = {"columns": own, "segments": [{
            "path": record["path"], "sha256": record["sha256"],
            "shape": list(values.shape), "dtype": str(values.dtype), "columns": own}]}
        store[name] = {"columns": groups[name], "segments": []}
        own_count = np.zeros(n, dtype=np.int64)
        for lo in range(0, n, row_batch_size):
            hi = min(n, lo + row_batch_size)
            for start in range(0, d, column_batch_size):
                block = _numeric_float64(values[lo:hi, start:start + column_batch_size])
                own_count[lo:hi] += np.isfinite(block).sum(axis=1)
        fields[f"own_{name}_observed_fraction"] = own_count / d
        for radius, graph in sorted(graphs.items()):
            logical = [f"r{radius:g}um_{name}_mean:{quote(label, safe='')}" for label in labels]
            path = f"neighborhood_features/{name}_{radius:g}um.npy"
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError("Neighbourhood payload already exists")
            means = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float64, shape=(n, d))
            coverage = np.zeros(n, dtype=np.float64)
            counts = np.diff(graph.indptr)
            for lo in range(0, n, row_batch_size):
                hi = min(n, lo + row_batch_size)
                output_rows = sorted_to_canonical[lo:hi]
                first, last = int(graph.indptr[lo]), int(graph.indptr[hi])
                for start in range(0, d, column_batch_size):
                    stop = min(d, start + column_batch_size)
                    sums = np.zeros((hi - lo, stop - start), dtype=np.float64)
                    observed = np.zeros_like(sums)
                    # Cap even a high-degree row's gather: never load the full
                    # neighbours-by-features or all-cells-by-features matrix.
                    for edge in range(first, last, row_batch_size):
                        end = min(last, edge + row_batch_size)
                        target_rows = np.searchsorted(graph.indptr[lo:hi + 1], np.arange(edge, end), side="right") - 1
                        source_rows = sorted_to_canonical[graph.indices[edge:end]]
                        block = _numeric_float64(values[source_rows, start:stop])
                        finite = np.isfinite(block)
                        np.add.at(sums, target_rows, np.where(finite, block, 0))
                        np.add.at(observed, target_rows, finite)
                    result = np.divide(sums, observed, out=np.full_like(sums, np.nan), where=observed > 0)
                    means[output_rows, start:stop] = result
                    coverage[output_rows] += (observed / np.maximum(counts[lo:hi], 1)[:, None]).sum(axis=1)
            means.flush()
            arrays[path] = means
            for axis, column in enumerate(logical):
                locations[column] = (path, axis)
            groups[name].extend(logical)
            store[name]["segments"].append({"path": path, "sha256": sha256_file(destination),
                "shape": [n, d], "dtype": str(means.dtype), "columns": logical})
            fields[f"r{radius:g}um_{name}_observed_fraction"] = coverage / d
    if (set(fields) | set(locations)) & set(profiles):
        raise ValueError("Array-backed neighbourhood fields collide with canonical columns")
    profiles = pd.concat([profiles, pd.DataFrame(fields, index=profiles.index)], axis=1)
    summary["feature_group_definitions"] = feature_group_definitions(groups, definitions, graphs, profiles.phenotype)
    summary["array_aggregation"] = {"row_batch_size": row_batch_size, "column_batch_size": column_batch_size,
        "estimated_working_mb": estimate, "budget_mb": max_working_mb,
        "storage": "memory_mapped_npy; no compression; not chunked_zarr",
        "order": "canonical payload rows; sorted analyzer CSR edge accumulation",
        "scope": "Bounded aggregation buffers only; excludes sparse graphs, scalar table, runtime and file-backed pages"}

    def read_values(rows, columns):
        selected = np.arange(*rows.indices(n)) if isinstance(rows, slice) else np.asarray(rows)
        canonical = sorted_to_canonical[selected]
        result = np.empty((len(canonical), len(columns)), dtype=np.float64)
        requests = defaultdict(list)
        for output, column in enumerate(columns):
            if column in locations:
                path, axis = locations[column]
                requests[path].append((output, axis))
            elif column in profiles:
                result[:, output] = _numeric_float64(profiles[column].iloc[canonical].to_numpy())
            else:
                raise ValueError(f"Unknown neighbourhood logical column {column}")
        for path, request in requests.items():
            output, axes = zip(*request)
            result[:, output] = _numeric_float64(arrays[path][np.ix_(canonical, axes)])
        return result

    return profiles, store, read_values
