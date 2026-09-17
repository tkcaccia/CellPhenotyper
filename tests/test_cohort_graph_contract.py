"""Cohort graph/source projection validation; tiny synthetic profiles only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from test_cohort_niches import make_spatial, rewrite_manifest
from fit_cohort_niches import source_record, compatibility, pool_profiles, fit_cohort
from cell_profile_io import sha256_file


def pair(tmp_path):
    return [make_spatial(tmp_path / sample, sample) for sample in ("A", "B")]


def pool(paths):
    records = [source_record(path) for path in paths]
    groups, _, _ = compatibility(records)
    return pool_profiles(records, groups)


def rewrite_table(root, update):
    path = root / "cell_profiles.parquet"
    frame = pd.read_parquet(path)
    update(frame)
    frame.to_parquet(path, index=False)
    rewrite_manifest(root, lambda manifest: manifest["files"].update({path.name: sha256_file(path)}))


def rewrite_graph(root, graph):
    path = root / "neighborhood_graph_3um.npz"
    sparse.save_npz(path, graph)
    rewrite_manifest(root, lambda manifest: manifest["spatial_graphs"]["3"].update(sha256=sha256_file(path)))


@pytest.mark.parametrize("fault", ["asymmetric_zero", "duplicate_stored_edge"])
def test_graph_structure_is_checked_even_when_numeric_subtraction_cancels(tmp_path, fault):
    _, b = pair(tmp_path)
    if fault == "asymmetric_zero":
        graph = sparse.csr_matrix(([0.], ([0], [1])), shape=(7, 7))
        assert (graph - graph.T).nnz == 0, "Numeric symmetry alone must miss this defect"
        error = "symmetry"
    else:
        graph = sparse.csr_matrix((np.array([.5, .5, 1.]), np.array([1, 1, 0]),
                                   np.array([0, 2, 3, 3, 3, 3, 3, 3])), shape=(7, 7))
        assert (graph - graph.T).nnz == 0
        error = "duplicate stored edges"
    rewrite_graph(b, graph)
    with pytest.raises(ValueError, match=error):
        source_record(b)


@pytest.mark.parametrize("radius", ["nan", "inf", "-1", "6", "duplicate_alias"])
def test_graph_radii_are_exact_finite_positive_and_nonduplicated(tmp_path, radius):
    _, b = pair(tmp_path)
    def change(manifest):
        record = manifest["spatial_graphs"].pop("3")
        manifest["spatial_graphs"][radius if radius != "duplicate_alias" else "3"] = record
        if radius == "duplicate_alias":
            manifest["spatial_graphs"]["3.0"] = dict(record)
    rewrite_manifest(b, change)
    with pytest.raises(ValueError, match="radius"):
        source_record(b)


@pytest.mark.parametrize("fault,error", [
    ("stale_count", "graph degrees"), ("unsupported", "unsupported cells"),
    ("coordinate", "physical coordinates"), ("distance", "physical coordinates")])
def test_graph_counts_support_and_canonical_coordinates_cannot_silently_diverge(tmp_path, fault, error):
    a, b = pair(tmp_path)
    if fault == "stale_count":
        def change(frame):
            frame.loc[6, "in_tissue_support"] = True
            frame.loc[6, "r3um_neighbor_count"] = 1
        rewrite_table(b, change)
    elif fault == "unsupported":
        rewrite_table(b, lambda frame: frame.__setitem__("in_tissue_support", False))
    elif fault == "coordinate":
        rewrite_table(b, lambda frame: frame.__setitem__("x_um", frame.x_um + np.array([.1, 0, 0, 0, 0, 0, 0])))
    else:
        graph = sparse.load_npz(b / "neighborhood_graph_3um.npz")
        graph.data *= .5
        rewrite_graph(b, graph)
    with pytest.raises(ValueError, match=error):
        fit_cohort([a, b], tmp_path / "must_not_complete", fixed_k=2, repeats=2)
    assert not (tmp_path / "must_not_complete").exists()


@pytest.mark.parametrize("column,change,error", [
    ("own_synthetic:test_feature", 999., "own feature values"),
    ("r3um_synthetic_mean:test_feature", 999., "neighbour means"),
    ("own_synthetic_observed_fraction", .25, "own feature coverage"),
    ("r3um_synthetic_observed_fraction", .25, "neighbour feature coverage"),
])
def test_rehashed_tables_cannot_claim_different_source_values_or_coverage(tmp_path, column, change, error):
    a, b = pair(tmp_path)
    rewrite_table(b, lambda frame: frame.__setitem__(column, frame[column].mask(frame.index == 0, change)))
    with pytest.raises(ValueError, match=error):
        pool([a, b])


def test_own_projection_is_exact_but_neighbor_mean_allows_only_float64_roundoff(tmp_path):
    a, b = pair(tmp_path)
    column = "r3um_synthetic_mean:test_feature"
    rewrite_table(b, lambda frame: frame.__setitem__(column, np.nextafter(frame[column], np.inf)))
    pooled, _ = pool([a, b])
    assert len(pooled) == 14, "Few-ULP sparse reduction-order variation is not a changed feature"
    own = "own_synthetic:test_feature"
    rewrite_table(b, lambda frame: frame.__setitem__(own, np.nextafter(frame[own], np.inf)))
    with pytest.raises(ValueError, match="own feature values"):
        pool([a, b])


def make_wide_float_spatial(directory, sample):
    """Actual producer assembly with reordered IDs, zero edges and 67 features."""
    from build_cell_profiles import build_profiles, parser
    from assemble_spatial_cell_profiles import assemble
    make_spatial(directory, sample)
    objects = directory / "objects.csv"
    frame = pd.read_csv(objects, dtype={"label": str}, keep_default_na=False)
    frame.loc[1, ["x", "y", "xmin", "ymin", "xmax", "ymax"]] = frame.loc[0, ["x", "y", "xmin", "ymin", "xmax", "ymax"]].to_numpy()
    frame.to_csv(objects, index=False)
    base = directory / "float_base"
    _, manifest = build_profiles(parser().parse_args(["--objects", str(objects), "--sample-id", sample,
        "--shift", str(directory / "shift.json"), "--resolution-json", str(directory / "resolution.json"), "--outdir", str(base)]))
    values = np.random.default_rng(17).normal(size=(7, 67))
    values[1, 2], values[2, 5], values[6] = np.nan, np.nan, np.nan
    path = base / "feature_blocks/synthetic.npy"
    np.save(path, values, allow_pickle=False)
    manifest["feature_blocks"]["synthetic"] = {"path": "feature_blocks/synthetic.npy", "sha256": sha256_file(path),
        "shape": list(values.shape), "dtype": "float64", "feature_names": [f"f{i}" for i in range(67)],
        "reference_compatible": True, "feature_definition": {"method": "synthetic_random_vectors_not_learned"}}
    (base / "cell_profiles_manifest.json").write_text(json.dumps(manifest))
    output = directory / "float_spatial"
    assemble(base, directory / "support.tif", directory / "shift.json", directory / "resolution.json", output,
             radii_um=(3.,), feature_groups=("synthetic",), fixed_k=1, repeats=2)
    return output


def test_actual_float_projection_zero_edges_and_bounded_64_column_verification(tmp_path, monkeypatch):
    a, b = [make_wide_float_spatial(tmp_path / sample, sample) for sample in ("A", "B")]
    graph = sparse.load_npz(a / "neighborhood_graph_3um.npz")
    assert graph.nnz > 0 and (graph.data == 0).sum() == 2
    before = {path: sha256_file(path) for root in (a, b) for path in root.rglob("*") if path.is_file()}
    original_matmul = sparse.csr_matrix.__matmul__
    widths = []
    def bounded(self, values):
        if isinstance(values, np.ndarray) and values.ndim == 2:
            widths.append(values.shape[1])
            assert values.shape[1] <= 64
        return original_matmul(self, values)
    monkeypatch.setattr(sparse.csr_matrix, "__matmul__", bounded)
    assignments, _, summary = fit_cohort([a, b], tmp_path / "cohort", fixed_k=1, repeats=2)
    assert len(assignments) == summary["cell_count"] == 14
    assert 64 in widths and 3 in widths
    assert assignments.cohort_niche_status.tolist().count("outside_tissue_support") == 2
    assert all(sha256_file(path) == expected for path, expected in before.items())
    assert sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()) < 10_000_000
