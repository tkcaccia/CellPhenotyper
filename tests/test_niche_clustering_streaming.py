"""Small independent dense-reference tests; no image models or remote inputs."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parents[1] / "bin"))
import niche_clustering as module


def legacy_matrix(profiles, groups, weights, eligible):
    """Frozen 2.0.0 dense scaling, before streaming integration."""
    arrays, metadata = [], {}
    for name, columns in sorted(groups.items()):
        if not columns:
            continue
        raw = profiles[columns].to_numpy(float)
        available = np.isfinite(raw)
        count = available[eligible].sum(axis=0)
        means = np.divide(np.where(available[eligible], raw[eligible], 0).sum(axis=0), count,
                          out=np.zeros(len(columns)), where=count > 0)
        centered = np.where(available, raw - means, 0)
        std = np.sqrt(np.divide((centered[eligible] ** 2).sum(axis=0), count,
                               out=np.zeros(len(columns)), where=count > 0))
        keep = (std > 1e-12) & (count > 1)
        missing = ~available
        variable_missing = missing[eligible].any(axis=0) & ~missing[eligible].all(axis=0)
        if not keep.any() and not variable_missing.any():
            continue
        z = centered[:, keep] / std[keep]
        if variable_missing.any():
            z = np.column_stack([z, missing[:, variable_missing].astype(float)])
        weight = weights.get(name, 1.0)
        arrays.append(z * math.sqrt(weight / z.shape[1]))
        retained = [c for c, ok in zip(columns, keep) if ok]
        missing_columns = [c for c, ok in zip(columns, variable_missing) if ok]
        metadata[name] = {"columns": retained, "means": means[keep].tolist(), "std": std[keep].tolist(),
            "missing_indicator_columns": missing_columns, "weight": weight, "dimensions": z.shape[1],
            "matrix_columns": [{"kind": "standardized_value", "source_column": c} for c in retained]
                + [{"kind": "missing_indicator", "source_column": c} for c in missing_columns]}
    return (np.column_stack(arrays) if arrays else np.zeros((len(profiles), 0))), metadata


def legacy_discovery(profiles, groups, *, fixed_k=None, max_k=8, seed=17, repeats=5,
                     feature_weights=None, fit_limit=20000):
    """Frozen original fitting/selection operations, independent of production."""
    count_columns = [c for c in profiles if c.endswith("_neighbor_count")]
    eligible = profiles.in_tissue_support.to_numpy(bool) & (profiles[count_columns].max(axis=1).to_numpy() > 0)
    output = profiles[module.KEYS].copy()
    output["niche_id"] = pd.Series([pd.NA] * len(profiles), dtype="Int64")
    output["niche_status"] = np.where(profiles.in_tissue_support, "isolated_no_neighborhood", "outside_tissue_support")
    output["niche_centroid_margin"] = np.nan
    output["niche_stability"] = np.nan
    matrix, scaling = legacy_matrix(profiles, groups, feature_weights or {}, eligible)
    meta = {"scaling": scaling, "eligible_cells": int(eligible.sum()), "fixed_k": fixed_k,
            "candidate_diagnostics": [], "seed": seed, "repeats": repeats, "fit_limit": fit_limit}
    if not eligible.any():
        if fixed_k is not None:
            raise ValueError("Cannot force niches when no cell has a supported neighborhood")
        meta["selected_k"] = 0
        return output, meta, matrix
    data = matrix[eligible]
    rng = np.random.default_rng(seed)
    train_indices = np.sort(rng.choice(len(data), min(len(data), fit_limit), replace=False))
    train = data[train_indices]
    distinct = len(np.unique(train, axis=0)) if train.shape[1] else 1
    if fixed_k and fixed_k > distinct:
        raise ValueError(f"Requested {fixed_k} niches but only {distinct} distinct feature profiles exist")
    candidates = [fixed_k] if fixed_k is not None else list(range(2, min(max_k, distinct, len(train) // 2) + 1))
    selected, best_score = None, -np.inf
    for k in candidates:
        if k == 1:
            break
        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(train)
        order = sorted(range(len(model.cluster_centers_)), key=lambda i: tuple(model.cluster_centers_[i]))
        labels, centers = np.argsort(order)[model.predict(data)], model.cluster_centers_[order]
        baseline_train = labels[train_indices]
        distances = cdist(data, centers)
        own = distances[np.arange(len(data)), labels]
        other = np.partition(distances, 1, axis=1)[:, 1]
        margin = (other - own) / np.maximum(other, 1e-12)
        stability_count, ari_values = np.zeros(len(data)), []
        for repeat in range(repeats):
            subset = np.sort(rng.choice(len(train), max(k, int(len(train) * .8)), replace=False))
            bootstrap = KMeans(n_clusters=k, random_state=seed + repeat + 1, n_init=5).fit(train[subset])
            pred = bootstrap.predict(data)
            contingency = np.zeros((k, k), dtype=int)
            np.add.at(contingency, (pred[train_indices], baseline_train), 1)
            row, col = linear_sum_assignment(-contingency)
            mapping = np.empty(k, dtype=int)
            mapping[row] = col
            aligned = mapping[pred]
            stability_count += aligned == labels
            ari_values.append(float(adjusted_rand_score(baseline_train, aligned[train_indices])))
        stability, separation = float(np.mean(ari_values)), float(margin[train_indices].mean())
        min_count = int(np.bincount(baseline_train, minlength=k).min())
        accepted = fixed_k is not None or (stability >= .75 and separation >= .45 and min_count >= max(2, int(len(train) * .01)))
        score = stability * separation - .01 * k
        meta["candidate_diagnostics"].append({"k": int(k), "subsample_ari": stability,
            "centroid_silhouette": separation, "smallest_training_cluster": min_count,
            "score": score, "passes_selection": bool(accepted)})
        if accepted and score > best_score:
            selected = (labels, centers, margin, stability_count / repeats)
            best_score = score
    if selected is None:
        labels = np.zeros(len(data), dtype=int)
        centers = np.mean(data, axis=0, keepdims=True)
        margin, stability = np.full(len(data), np.nan), np.full(len(data), np.nan)
        status = "assigned_fixed_k" if fixed_k == 1 else "single_niche_no_supported_subdivision"
    else:
        labels, centers, margin, stability = selected
        status = "assigned_fixed_k" if fixed_k is not None else "assigned_exploratory"
    output.loc[eligible, "niche_id"] = labels + 1
    output.loc[eligible, "niche_status"] = status
    output.loc[eligible, "niche_centroid_margin"] = margin
    output.loc[eligible, "niche_stability"] = stability
    meta.update(selected_k=int(len(centers)), training_cells=len(train),
                feature_dimensions=matrix.shape[1], centroids_scaled=centers.tolist())
    return output, meta, matrix


def fixture(seed=29, n=127, *, offset=0., spread=1.):
    rng = np.random.default_rng(seed)
    # Asymmetric, deliberately unequal groups and nontrivial source ordering.
    cluster = (np.arange(n) >= n // 3).astype(float)
    frame = pd.DataFrame({"sample_id": np.where(np.arange(n) % 3, "B", "A"),
        "cell_id": ["000" + str(i) for i in range(n)], "in_tissue_support": np.ones(n, bool),
        "r25um_neighbor_count": rng.integers(0, 3, n), "r50um_neighbor_count": rng.integers(1, 8, n),
        "r100um_neighbor_count": rng.integers(2, 20, n),
        "own:first": offset + spread * (cluster * 13 + rng.normal(size=n)),
        "own:second": -offset + spread * (cluster * -8 + rng.normal(size=n)),
        "neighbor:first": rng.normal(size=n) + cluster * 9,
        "neighbor:constant_missing": np.where(np.arange(n) % 4, 7., np.nan),
        "neighbor:always_missing": np.nan, "neighbor:constant": 4.,
        "density25": rng.normal(size=n) + cluster * 2,
        "density100": rng.normal(size=n) + cluster * 3})
    frame.loc[[1, 8, 21, 88], "own:first"] = np.nan
    frame.loc[[9, 12, 28], "own:second"] = np.inf
    frame.loc[[0, n - 1], "in_tissue_support"] = False
    frame.loc[2, ["r25um_neighbor_count", "r50um_neighbor_count", "r100um_neighbor_count"]] = 0
    frame.loc[3, "r25um_neighbor_count"] = 0  # Eligible through wider radii.
    groups = {"own:morphology": ["own:first", "own:second"],
        "morphology": ["neighbor:first", "neighbor:constant_missing", "neighbor:always_missing", "neighbor:constant"],
        "density": ["density25", "density100"], "empty": []}
    return frame, groups


def narrow(frame):
    return frame[module.KEYS + ["in_tissue_support"] + [c for c in frame if c.endswith("_neighbor_count")]].copy()


class Reader:
    def __init__(self, frame, batch_size):
        self.frame, self.batch_size, self.calls = frame, batch_size, []
    def __call__(self, rows, columns):
        positions = np.arange(len(self.frame))[rows] if isinstance(rows, slice) else np.asarray(rows)
        assert 0 < len(positions) <= self.batch_size
        assert np.all(np.diff(positions) > 0), "Source reads must preserve ascending eligible row order"
        self.calls.append((positions.copy(), list(columns)))
        return self.frame.iloc[positions][columns].to_numpy(dtype=np.float64)


def compare(frame, groups, tmp_path, *, batch_size=11, atol=1e-10, **kwargs):
    before = frame.copy(deep=True)
    expected, old_meta, old_matrix = legacy_discovery(frame, groups, **kwargs)
    reader = Reader(frame, batch_size)
    result, meta = module.discover_niches_streaming(narrow(frame), groups, reader, tmp_path / "scratch",
        row_batch_size=batch_size, **kwargs)
    pd.testing.assert_frame_equal(result.drop(columns="niche_centroid_margin"), expected.drop(columns="niche_centroid_margin"))
    np.testing.assert_allclose(result.niche_centroid_margin, expected.niche_centroid_margin, rtol=atol, atol=atol, equal_nan=True)
    for name in ("selected_k", "eligible_cells", "fixed_k", "seed", "repeats", "fit_limit", "training_cells", "feature_dimensions"):
        assert meta.get(name) == old_meta.get(name)
    assert meta["scaling"].keys() == old_meta["scaling"].keys()
    for group, definition in old_meta["scaling"].items():
        for key, value in definition.items():
            if key in ("means", "std"):
                np.testing.assert_allclose(meta["scaling"][group][key], value, rtol=atol, atol=atol)
            else:
                assert meta["scaling"][group][key] == value
    assert len(meta["candidate_diagnostics"]) == len(old_meta["candidate_diagnostics"])
    for actual, old in zip(meta["candidate_diagnostics"], old_meta["candidate_diagnostics"]):
        for name in old:
            if name in ("centroid_silhouette", "score"):
                assert actual[name] == pytest.approx(old[name], rel=atol, abs=atol)
            else:
                assert actual[name] == old[name]
    if "centroids_scaled" in old_meta:
        np.testing.assert_allclose(meta["centroids_scaled"], old_meta["centroids_scaled"], rtol=atol, atol=atol)
    eligible = frame.in_tissue_support.to_numpy(bool) & (frame.filter(like="_neighbor_count").max(axis=1).to_numpy() > 0)
    if old_matrix.shape[1] and eligible.any():
        stored = np.load(tmp_path / "scratch/scaled_features.npy", mmap_mode="r")
        assert isinstance(stored, np.memmap) and stored.shape == (eligible.sum(), old_matrix.shape[1])
        np.testing.assert_allclose(stored, old_matrix[eligible], rtol=atol, atol=atol)
    else:
        assert not (tmp_path / "scratch/scaled_features.npy").exists()
    assert all(eligible[rows].all() for rows, _ in reader.calls)
    pd.testing.assert_frame_equal(frame, before)
    assert str(tmp_path) not in json.dumps(meta, allow_nan=False)
    return result, meta, reader


@pytest.mark.parametrize("seed,batch,fixed", [(17, 1, 2), (29, 7, 2), (39, 31, None), (51, 19, 1)])
def test_dense_equivalence_asymmetric_multiple_radii_and_missingness(tmp_path, seed, batch, fixed):
    frame, groups = fixture(seed)
    result, meta, reader = compare(frame, groups, tmp_path, batch_size=batch, seed=seed, fixed_k=fixed,
        max_k=4, repeats=3, fit_limit=67, feature_weights={"own:morphology": 4., "morphology": .3})
    assert len(result) == len(frame) and result.cell_id.iloc[0] == "0000"
    assert result.niche_id.iloc[[0, 2, len(frame) - 1]].isna().all()
    assert pd.notna(result.niche_id.iloc[3])
    missing = meta["scaling"]["morphology"]
    assert "neighbor:constant_missing" in missing["missing_indicator_columns"]
    assert "neighbor:constant_missing" not in missing["columns"]
    assert "neighbor:always_missing" not in missing["missing_indicator_columns"]
    assert "neighbor:constant" not in missing["columns"]
    assert reader.calls


def test_large_numeric_offsets_preserve_decisions_and_disclose_roundoff(tmp_path):
    frame, groups = fixture(offset=2.**32, spread=2.**16)
    _, meta, _ = compare(frame, groups, tmp_path, batch_size=13, fixed_k=2, repeats=3, fit_limit=83)
    assert "no bitwise guarantee" in meta["streaming_execution"]["reduction_policy"]


def test_extreme_offset_small_spread_is_not_claimed_bitwise_or_universally_tight(tmp_path):
    # This deliberately ill-conditioned case exceeds the ordinary 1e-10 check.
    # Freeze a separate, declared numeric envelope while still requiring exact
    # labels/status/stability. It is not evidence of an all-input error bound.
    frame, groups = fixture(offset=1e12, spread=1.)
    old, old_meta, _ = legacy_discovery(frame, groups, fixed_k=2, repeats=2, fit_limit=83)
    new, meta = module.discover_niches_streaming(narrow(frame), groups, Reader(frame, 13),
        tmp_path / "scratch", row_batch_size=13, fixed_k=2, repeats=2, fit_limit=83)
    pd.testing.assert_frame_equal(old.drop(columns="niche_centroid_margin"), new.drop(columns="niche_centroid_margin"))
    np.testing.assert_allclose(new.niche_centroid_margin, old.niche_centroid_margin,
                               rtol=1e-4, atol=1e-4, equal_nan=True)
    np.testing.assert_allclose(meta["centroids_scaled"], old_meta["centroids_scaled"], rtol=1e-4, atol=1e-4)
    assert "Large offsets" in meta["streaming_execution"]["regression_tolerance_scope"]


@pytest.mark.parametrize("fixed", [None, 1])
def test_zero_dimensional_k1_has_missing_scores_and_no_memmap(tmp_path, fixed):
    frame, _ = fixture()
    frame["constant"] = 4.
    result, meta, _ = compare(frame, {"constant": ["constant"], "empty": []}, tmp_path,
                             fixed_k=fixed, repeats=2)
    assert meta["selected_k"] == 1 and meta["feature_dimensions"] == 0
    assert result.niche_centroid_margin.isna().all() and result.niche_stability.isna().all()


def test_no_eligible_cells_never_reads_features_or_creates_zero_width_matrix(tmp_path):
    frame, groups = fixture()
    frame.loc[:, "in_tissue_support"] = False
    result, meta, reader = compare(frame, groups, tmp_path, repeats=2)
    assert not reader.calls and meta["selected_k"] == 0 and result.niche_id.isna().all()
    with pytest.raises(ValueError, match="Cannot force"):
        module.discover_niches_streaming(narrow(frame), groups, reader, tmp_path / "forced", fixed_k=1)


def test_missingness_only_dimensions_survive_without_invented_numeric_values(tmp_path):
    frame, _ = fixture()
    frame["missing"] = np.where(np.arange(len(frame)) % 2, 3., np.nan)
    _, meta, _ = compare(frame, {"own:modality": ["missing"]}, tmp_path, fixed_k=2, repeats=2)
    assert meta["scaling"]["own:modality"]["columns"] == []
    assert meta["scaling"]["own:modality"]["matrix_columns"] == [
        {"kind": "missing_indicator", "source_column": "missing"}]


def test_source_prediction_distance_and_training_allocations_are_bounded(tmp_path, monkeypatch):
    frame, groups = fixture(n=997)
    row_batch, train_limit = 17, 61
    reader = Reader(frame, row_batch)
    fits, predictions, distances = [], [], []
    original_fit, original_predict = KMeans.fit, KMeans.predict
    import scipy.spatial.distance as distance_module
    original_cdist = distance_module.cdist
    def fit(self, values, *args, **kwargs):
        assert values.shape[0] <= train_limit
        fits.append(values.shape)
        return original_fit(self, values, *args, **kwargs)
    def predict(self, values, *args, **kwargs):
        assert values.shape[0] <= row_batch
        predictions.append(values.shape)
        return original_predict(self, values, *args, **kwargs)
    def bounded_cdist(left, right, *args, **kwargs):
        assert left.shape[0] <= row_batch and right.shape[0] <= 3
        distances.append((left.shape, right.shape))
        return original_cdist(left, right, *args, **kwargs)
    monkeypatch.setattr(KMeans, "fit", fit)
    monkeypatch.setattr(KMeans, "predict", predict)
    monkeypatch.setattr(distance_module, "cdist", bounded_cdist)
    result, meta = module.discover_niches_streaming(narrow(frame), groups, reader, tmp_path / "scratch",
        row_batch_size=row_batch, fit_limit=train_limit, fixed_k=3, repeats=2, max_working_mb=4)
    assert len(result) == len(frame) and fits and predictions and distances
    assert meta["streaming_execution"]["estimated_numerical_working_mb"] < 4
    assert meta["streaming_execution"]["scaled_rows"] > train_limit


def test_all_dimensions_survive_readonly_source_and_disk_backed_scaling(tmp_path, monkeypatch):
    n, width, batch, fit_limit = 513, 83, 13, 43
    rng = np.random.default_rng(991)
    source = rng.normal(size=(n, width))
    source[np.arange(n) % 7 == 0, 2:6] = np.nan
    path = tmp_path / "raw.npy"
    np.save(path, source)
    raw = np.load(path, mmap_mode="r")
    frame = pd.DataFrame({"sample_id": ["A"] * n, "cell_id": [f"{i:06d}" for i in range(n)],
                          "in_tissue_support": True, "r50um_neighbor_count": 2})
    names = [f"feature_{i}" for i in range(width)]
    calls = []
    def read(rows, columns):
        assert columns == names and len(rows) <= batch
        # Return a read-only source view, not a defensive copy that could mask
        # accidental in-place mutation of raw values during normalization.
        values = raw[rows[0]:rows[-1] + 1]
        assert not values.flags.writeable and len(values) == len(rows)
        calls.append(len(values))
        return values
    original_asarray = np.asarray
    def guarded_asarray(value, *args, **kwargs):
        if isinstance(value, np.memmap) and value.ndim == 2:
            assert value.shape[0] <= max(batch, fit_limit), "Do not materialize a global feature memmap as ndarray"
        return original_asarray(value, *args, **kwargs)
    monkeypatch.setattr(np, "asarray", guarded_asarray)
    result, meta = module.discover_niches_streaming(frame, {"all": names}, read, tmp_path / "scratch",
        row_batch_size=batch, fit_limit=fit_limit, fixed_k=2, repeats=2, max_working_mb=2)
    monkeypatch.setattr(np, "asarray", original_asarray)
    assert len(result) == n and meta["scaling"]["all"]["columns"] == names
    assert meta["feature_dimensions"] == width + 4
    assert max(calls) == batch
    np.testing.assert_array_equal(np.load(path), source)
    stored = np.load(tmp_path / "scratch/scaled_features.npy", mmap_mode="r")
    assert stored.shape == (n, width + 4) and isinstance(stored, np.memmap)
    dense = frame.copy()
    for index, name in enumerate(names):
        dense[name] = source[:, index]
    expected, _ = legacy_matrix(dense, {"all": names}, {}, np.ones(n, bool))
    np.testing.assert_allclose(stored, expected, rtol=1e-10, atol=1e-10)


def test_reduction_preflight_happens_before_any_value_read_or_scratch(tmp_path):
    frame, groups = fixture()
    reader = Reader(frame, 4096)
    with pytest.raises(ValueError, match="reduction working estimate"):
        module.discover_niches_streaming(narrow(frame), groups, reader, tmp_path / "scratch", max_working_mb=.001)
    assert not reader.calls and not (tmp_path / "scratch").exists()


def test_training_preflight_never_creates_scaled_matrix_or_fits(tmp_path, monkeypatch):
    frame, groups = fixture(n=997)
    reader = Reader(frame, 1)
    monkeypatch.setattr(KMeans, "fit", lambda *a, **k: pytest.fail("Preflight must happen before fitting"))
    with pytest.raises(ValueError, match="training/prediction working estimate"):
        module.discover_niches_streaming(narrow(frame), groups, reader, tmp_path / "scratch",
            row_batch_size=1, fit_limit=997, max_working_mb=.2)
    assert reader.calls and not (tmp_path / "scratch/scaled_features.npy").exists()


def test_existing_scratch_is_not_overwritten(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    marker = scratch / "preserve.txt"
    marker.write_text("existing user scratch")
    frame, groups = fixture()
    reader = Reader(frame, 1)
    with pytest.raises(FileExistsError):
        module.discover_niches_streaming(narrow(frame), groups, reader, scratch)
    assert marker.read_text() == "existing user scratch" and not reader.calls


@pytest.mark.parametrize("fault", ["fractional_count", "missing_count", "missing_support", "string_support", "duplicate_id"])
def test_invalid_scalar_population_never_reaches_feature_reader(tmp_path, fault):
    frame, groups = fixture()
    population = narrow(frame)
    if fault == "fractional_count":
        population["r25um_neighbor_count"] = population.r25um_neighbor_count.astype(float)
        population.loc[0, "r25um_neighbor_count"] = .5
    elif fault == "missing_count":
        population["r25um_neighbor_count"] = population.r25um_neighbor_count.astype(float)
        population.loc[0, "r25um_neighbor_count"] = np.nan
    elif fault == "missing_support":
        population["in_tissue_support"] = population.in_tissue_support.astype("boolean")
        population.loc[0, "in_tissue_support"] = pd.NA
    elif fault == "string_support":
        population["in_tissue_support"] = population.in_tissue_support.astype(str)
    else:
        population.loc[1, module.KEYS] = population.loc[0, module.KEYS]
    reader = Reader(frame, 4096)
    with pytest.raises(ValueError):
        module.discover_niches_streaming(population, groups, reader, tmp_path / "scratch")
    assert not reader.calls


def test_empty_population_and_empty_feature_groups_are_supported(tmp_path):
    frame, _ = fixture()
    empty, meta = module.discover_niches_streaming(narrow(frame.iloc[:0]), {},
        lambda *_: pytest.fail("No values needed"), tmp_path / "empty")
    assert len(empty) == 0 and meta["selected_k"] == 0
    result, meta = module.discover_niches_streaming(narrow(frame), {},
        lambda *_: pytest.fail("No values needed"), tmp_path / "constant", fixed_k=1)
    assert len(result) == len(frame) and meta["selected_k"] == 1 and meta["feature_dimensions"] == 0


@pytest.mark.parametrize("fault", ["dtype", "shape"])
def test_reader_contract_cannot_silently_drop_dimensions(tmp_path, fault):
    frame, groups = fixture()
    def read(rows, columns):
        values = frame.iloc[rows][columns].to_numpy(float)
        return values.astype(np.float32) if fault == "dtype" else values[:, :-1]
    with pytest.raises(ValueError, match="float64 rows-by-columns"):
        module.discover_niches_streaming(narrow(frame), groups, read, tmp_path / "scratch", row_batch_size=7)


@pytest.mark.parametrize("kwargs,match", [({"row_batch_size": 0}, "row_batch_size"),
    ({"max_working_mb": float('inf')}, "max_working_mb"), ({"fixed_k": 0}, "fixed_k"),
    ({"fit_limit": 9}, "fit_limit"), ({"feature_weights": {"foreign": 1}}, "Unknown"),
    ({"feature_weights": {"density": 0}}, "positive finite")])
def test_invalid_parameters_fail_without_reads(tmp_path, kwargs, match):
    frame, groups = fixture()
    reader = Reader(frame, 4096)
    with pytest.raises(ValueError, match=match):
        module.discover_niches_streaming(narrow(frame), groups, reader, tmp_path / "scratch", **kwargs)
    assert not reader.calls
