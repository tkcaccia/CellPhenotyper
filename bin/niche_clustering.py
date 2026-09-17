#!/usr/bin/env python3
"""Disk-backed equivalent of exploratory self-plus-neighbour niche discovery.

The caller owns scratch-directory cleanup and the immutable raw-value reader.
Only the bounded training subset is materialized as a dense feature matrix;
full-population scaling, prediction and distances use row batches. This is CPU
KMeans, not learned image-model inference or a biological validation method.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


KEYS = ["sample_id", "cell_id"]
FEATURE_REPRESENTATION_VERSION = "2.0.0"


def _batches(count, size):
    for start in range(0, count, size):
        yield slice(start, min(count, start + size))


def _read(read_values, rows, columns):
    values = np.asarray(read_values(rows, list(columns)))
    if values.dtype != np.dtype("float64") or values.shape != (len(rows), len(columns)):
        raise ValueError("Niche value reader must return float64 rows-by-columns in the exact requested order")
    # NaN and infinity follow the existing isfinite-based availability policy.
    # Never modify this array: it can be a view of the caller's source data.
    return values


def _preflight(estimated_bytes, budget_bytes, phase):
    if estimated_bytes > budget_bytes:
        raise ValueError(f"Streaming niche {phase} working estimate {estimated_bytes / 1024**2:.3f} MiB "
                         f"exceeds max_working_mb={budget_bytes / 1024**2:g}; increase the declared budget "
                         "or explicitly reduce row_batch_size/fit_limit. No features or cells were dropped.")


def _predict_batched(model, data, batch_size):
    predictions = np.empty(len(data), dtype=np.int64)
    for rows in _batches(len(data), batch_size):
        predictions[rows] = model.predict(data[rows])
    return predictions


def discover_niches_streaming(
    profiles, groups, read_values, work_dir, *, fixed_k=None, max_k=8, seed=17,
    repeats=5, feature_weights=None, fit_limit=20000, max_working_mb=1024,
    row_batch_size=4096,
):
    """Fit the legacy niche representation without a global dense feature copy.

    ``read_values(rows, columns)`` must return float64 values in positional row
    and exact logical-column order; rows are bounded integer index arrays.
    Raw missingness is retained by the source reader. ``work_dir`` must not
    exist. Its optional scaled_features.npy is scratch, not a portable model.

    Memory estimates cover numerical working buffers, not total process RSS,
    the caller's already-loaded profiles, Python metadata, or OS file-page
    cache. Reduction order differs from the dense implementation; float64
    agreement is tolerance-tested, not guaranteed bitwise or at exact ties.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from scipy.spatial.distance import cdist

    for name, value, minimum in (("max_k", max_k, 2), ("repeats", repeats, 2),
                                 ("fit_limit", fit_limit, 10), ("row_batch_size", row_batch_size, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if fixed_k is not None and (isinstance(fixed_k, bool) or not isinstance(fixed_k, (int, np.integer)) or fixed_k < 1):
        raise ValueError("fixed_k must be positive")
    if not math.isfinite(max_working_mb) or max_working_mb <= 0:
        raise ValueError("max_working_mb must be positive and finite")
    if not isinstance(groups, dict) or any(not isinstance(name, str) or not isinstance(columns, list)
            or any(not isinstance(column, str) or not column for column in columns)
            or len(set(columns)) != len(columns) for name, columns in groups.items()):
        raise ValueError("Niche groups require exact ordered, unique logical column lists")
    weights = feature_weights or {}
    if set(weights) - set(groups):
        raise ValueError(f"Unknown feature weight groups: {sorted(set(weights) - set(groups))}")
    for name, columns in groups.items():
        if columns and name in weights and (not math.isfinite(weights[name]) or weights[name] <= 0):
            raise ValueError("Feature group weights must be positive finite values")
    if not set(KEYS + ["in_tissue_support"]) <= set(profiles):
        raise ValueError("Niche profiles require canonical identities and in_tissue_support")
    if profiles.duplicated(KEYS).any() or profiles[KEYS].isna().any().any():
        raise ValueError("Niche profiles require unique, nonmissing canonical identities")
    counts = [column for column in profiles if column.endswith("_neighbor_count")]
    if (not counts or profiles.in_tissue_support.isna().any()
            or profiles.in_tissue_support.dtype.kind != "b"):
        raise ValueError("Niche profiles require explicit support and neighbourhood counts")
    count = len(profiles)
    max_group = max((len(columns) for columns in groups.values()), default=0)
    logical_dimensions = sum(map(len, groups.values()))
    budget = max_working_mb * 1024**2
    # Includes full-length numeric outputs/eligibility/indices and concurrent
    # candidate vectors; text identity objects already belong to the caller.
    base_bytes = count * 144 + logical_dimensions * 96
    reduction_bytes = base_bytes + min(count, row_batch_size) * max_group * 64
    _preflight(reduction_bytes, budget, "reduction")
    work_dir = Path(work_dir)
    if work_dir.exists():
        raise FileExistsError("Streaming niche scratch requires a new work_dir; existing files are never overwritten")
    supported = profiles.in_tissue_support.to_numpy(bool)
    connected = np.zeros(count, dtype=bool)
    for column in counts:
        values = profiles[column].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Neighbour counts must be finite nonnegative integers")
        connected |= values > 0
    eligible = supported & connected
    eligible_rows = np.flatnonzero(eligible)
    population = len(eligible_rows)
    work_dir.mkdir(parents=True)
    output = profiles[KEYS].copy()
    output["niche_id"] = pd.array([pd.NA] * count, dtype="Int64")
    output["niche_status"] = np.where(supported, "isolated_no_neighborhood", "outside_tissue_support")
    output["niche_centroid_margin"] = np.nan
    output["niche_stability"] = np.nan
    scaling, states = {}, []
    dimension = 0
    for name, columns in sorted(groups.items()):
        if not columns:
            continue
        observations = np.zeros(len(columns), dtype=np.int64)
        total = np.zeros(len(columns), dtype=np.float64)
        for batch in _batches(population, row_batch_size):
            raw = _read(read_values, eligible_rows[batch], columns)
            finite = np.isfinite(raw)
            observations += finite.sum(axis=0)
            total += np.where(finite, raw, 0).sum(axis=0)
        means = np.divide(total, observations, out=np.zeros(len(columns)), where=observations > 0)
        square_sum = np.zeros(len(columns), dtype=np.float64)
        for batch in _batches(population, row_batch_size):
            raw = _read(read_values, eligible_rows[batch], columns)
            centered = np.where(np.isfinite(raw), raw - means, 0)
            square_sum += (centered ** 2).sum(axis=0)
        std = np.sqrt(np.divide(square_sum, observations, out=np.zeros(len(columns)), where=observations > 0))
        if not np.isfinite(means).all() or not np.isfinite(std).all():
            raise ValueError("Niche float64 scaling overflowed; no feature values were silently discarded")
        keep = (std > 1e-12) & (observations > 1)
        missing = (observations > 0) & (observations < population)
        if not keep.any() and not missing.any():
            continue
        numeric = [column for column, retain in zip(columns, keep) if retain]
        missing_columns = [column for column, retain in zip(columns, missing) if retain]
        width = len(numeric) + len(missing_columns)
        weight = weights.get(name, 1.0)
        scaling[name] = {"columns": numeric, "means": means[keep].tolist(), "std": std[keep].tolist(),
            "missing_indicator_columns": missing_columns, "weight": weight, "dimensions": width,
            "matrix_columns": [{"kind": "standardized_value", "source_column": c} for c in numeric]
                + [{"kind": "missing_indicator", "source_column": c} for c in missing_columns]}
        states.append((columns, means, std, keep, missing, dimension, width, weight))
        dimension += width
    training_count = min(population, fit_limit)
    maximum_k = fixed_k if fixed_k is not None else min(max_k, max(1, training_count // 2))
    # Train + sklearn input/centering/initialization/subsample/unique-row copies
    # are bounded by fit_limit, while prediction/cdist never exceed row_batch_size.
    training_bytes = (base_bytes + training_count * dimension * 8 * 6
                      + training_count * maximum_k * 8 * 4
                      + min(population, row_batch_size) * (dimension * 32 + maximum_k * 32)
                      + maximum_k * maximum_k * 16)
    working_bytes = max(reduction_bytes, training_bytes)
    _preflight(working_bytes, budget, "training/prediction")
    execution = {"implementation": "streamed_float64_v1", "row_batch_size": int(row_batch_size),
        "estimated_numerical_working_mb": working_bytes / 1024**2,
        "max_working_mb": max_working_mb, "scaled_disk_bytes": population * dimension * 8,
        "scaled_rows": population, "scaled_dimensions": dimension,
        "memory_scope": "Numerical buffers only; excludes input profile table, Python metadata, interpreter/libraries and OS file-page cache.",
        "reduction_policy": "Two-pass float64 sums in ascending eligible-row batches; no bitwise guarantee. Reduction roundoff and exact decision-boundary ties may differ from the dense implementation.",
        "regression_float64_rtol_atol": [1e-10, 1e-10],
        "regression_tolerance_scope": "Well-conditioned synthetic equivalence-test tolerance, not a universal numerical-error bound or biological confidence. Large offsets relative to within-feature variation can exceed this tolerance and affect near-tie decisions.",
        "raw_missingness_preserved": True, "features_or_cells_dropped_for_budget": False}
    meta = {"feature_representation_version": FEATURE_REPRESENTATION_VERSION,
        "missing_feature_policy": "raw_NaNs_preserved; centered_missing_values_zero_in_fit_only; independent_variable_missingness_indicators",
        "missingness_interpretation": "Availability can separate exploratory niches; missingness-driven clusters are not evidence of biological differences.",
        "seed": seed, "repeats": repeats, "fit_limit": fit_limit, "eligible_cells": population,
        "fixed_k": fixed_k, "scaling": scaling, "candidate_diagnostics": [],
        "confidence_is_calibrated_probability": False,
        "selection_rule": "ARI>=0.75 and centroid_silhouette>=0.45; maximize ARI*centroid_silhouette-0.01*K; otherwise K=1",
        "streaming_execution": execution}
    if not population:
        if fixed_k is not None:
            raise ValueError("Cannot force niches when no cell has a supported neighborhood")
        meta["selected_k"] = 0
        return output, meta
    data = (np.lib.format.open_memmap(work_dir / "scaled_features.npy", mode="w+", dtype=np.float64,
                                     shape=(population, dimension)) if dimension else np.empty((population, 0)))
    for columns, means, std, keep, missing, offset, width, weight in states:
        factor = math.sqrt(weight / width)
        numeric_count = int(keep.sum())
        for batch in _batches(population, row_batch_size):
            raw = _read(read_values, eligible_rows[batch], columns)
            finite = np.isfinite(raw)
            if numeric_count:
                centered = np.where(finite, raw - means, 0)
                data[batch, offset:offset + numeric_count] = centered[:, keep] / std[keep] * factor
            if missing.any():
                data[batch, offset + numeric_count:offset + width] = (~finite[:, missing]).astype(float) * factor
    if dimension:
        data.flush()
    rng = np.random.default_rng(seed)
    train_indices = np.sort(rng.choice(population, training_count, replace=False))
    train = np.asarray(data[train_indices], dtype=np.float64)
    distinct = len(np.unique(train, axis=0)) if dimension else 1
    if fixed_k and fixed_k > distinct:
        raise ValueError(f"Requested {fixed_k} niches but only {distinct} distinct feature profiles exist")
    candidates = [fixed_k] if fixed_k is not None else list(range(2, min(max_k, distinct, training_count // 2) + 1))
    selected, best_score = None, -np.inf
    for k in candidates:
        if k == 1:
            break
        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(train)
        order = sorted(range(len(model.cluster_centers_)), key=lambda i: tuple(model.cluster_centers_[i]))
        inverse = np.argsort(order)
        centers = model.cluster_centers_[order]
        labels = inverse[_predict_batched(model, data, row_batch_size)]
        baseline_train = labels[train_indices]
        margin = np.empty(population, dtype=np.float64)
        for batch in _batches(population, row_batch_size):
            distances = cdist(data[batch], centers)
            own = distances[np.arange(len(distances)), labels[batch]]
            other = np.partition(distances, 1, axis=1)[:, 1]
            margin[batch] = (other - own) / np.maximum(other, 1e-12)
        stability_count = np.zeros(population, dtype=np.float64)
        ari_values = []
        for repeat in range(repeats):
            subset = np.sort(rng.choice(training_count, max(k, int(training_count * .8)), replace=False))
            bootstrap = KMeans(n_clusters=k, random_state=seed + repeat + 1, n_init=5).fit(train[subset])
            pred = _predict_batched(bootstrap, data, row_batch_size)
            contingency = np.zeros((k, k), dtype=int)
            np.add.at(contingency, (pred[train_indices], baseline_train), 1)
            row, col = linear_sum_assignment(-contingency)
            mapping = np.empty(k, dtype=int)
            mapping[row] = col
            for batch in _batches(population, row_batch_size):
                stability_count[batch] += mapping[pred[batch]] == labels[batch]
            ari_values.append(float(adjusted_rand_score(baseline_train, mapping[pred[train_indices]])))
        stability = float(np.mean(ari_values))
        separation = float(margin[train_indices].mean())
        min_count = int(np.bincount(baseline_train, minlength=k).min())
        accepted = fixed_k is not None or (stability >= .75 and separation >= .45 and min_count >= max(2, int(training_count * .01)))
        score = stability * separation - .01 * k
        meta["candidate_diagnostics"].append({"k": int(k), "subsample_ari": stability,
            "centroid_silhouette": separation, "smallest_training_cluster": min_count,
            "score": score, "passes_selection": bool(accepted)})
        if accepted and score > best_score:
            selected = (labels, centers, margin, stability_count / repeats)
            best_score = score
    if selected is None:
        labels = np.zeros(population, dtype=np.int64)
        total = np.zeros(dimension, dtype=np.float64)
        for batch in _batches(population, row_batch_size):
            total += data[batch].sum(axis=0)
        centers = (total / population)[None, :]
        margin, stability = np.full(population, np.nan), np.full(population, np.nan)
        status = "assigned_fixed_k" if fixed_k == 1 else "single_niche_no_supported_subdivision"
    else:
        labels, centers, margin, stability = selected
        status = "assigned_fixed_k" if fixed_k is not None else "assigned_exploratory"
    output.loc[eligible, "niche_id"] = labels + 1
    output.loc[eligible, "niche_status"] = status
    output.loc[eligible, "niche_centroid_margin"] = margin
    output.loc[eligible, "niche_stability"] = stability
    meta.update({"selected_k": int(len(centers)), "training_cells": training_count,
        "feature_dimensions": dimension, "centroids_scaled": centers.tolist(),
        "training_scope": "pooled specimens; graphs remain specimen-specific; sample identity excluded from features"})
    if dimension:
        data.flush()
    return output, meta
