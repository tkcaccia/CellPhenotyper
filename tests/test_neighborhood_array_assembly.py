"""Real assembler equivalence across table and bounded array representations."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from assemble_spatial_cell_profiles import assemble
from cell_profile_io import sha256_file
from neighborhood_feature_io import FeatureColumns
from test_spatial_profile_assembly import fixture


def base(tmp_path, dimensions=67):
    source, mask, shift, resolution, domain = fixture(tmp_path)
    # Deliberately unsorted canonical IDs and coincident cells exercise the
    # analyzer-to-payload row permutation and structural zero-distance edges.
    cells = pd.read_parquet(source / "cell_profiles.parquet")
    cells.loc[:, "x_um"] = [51., 51., 52.]
    cells.to_parquet(source / "cell_profiles.parquet", index=False)
    tifffile.imwrite(mask, np.ones((8, 10), np.uint8))
    manifest = json.loads((source / "cell_profiles_manifest.json").read_text())
    rng = np.random.default_rng(3)
    values = rng.normal(size=(3, dimensions)).astype(np.float64)
    values[0, ::3] = np.nan
    values[:, 1] = 7
    values[2, 1] = np.nan
    values[:, -1] = np.nan
    path = source / "feature_blocks/synthetic.npy"
    np.save(path, values)
    manifest["feature_blocks"]["synthetic"] = {"path": "feature_blocks/synthetic.npy",
        "sha256": sha256_file(path), "shape": list(values.shape),
        "feature_names": [f"axis {i}/%" for i in range(dimensions)]}
    manifest["files"]["cell_profiles.parquet"] = sha256_file(source / "cell_profiles.parquet")
    (source / "cell_profiles_manifest.json").write_text(json.dumps(manifest))
    return source, mask, shift, resolution, domain


@pytest.mark.parametrize("rows,columns", [(1, 1), (2, 7), (4096, 64)])
def test_full_features_scalars_graphs_and_fitting_equivalent(tmp_path, rows, columns):
    source, mask, shift, resolution, domain = base(tmp_path)
    before = {str(p.relative_to(source)): sha256_file(p) for p in source.rglob("*") if p.is_file()}
    common = dict(domain_mask=domain, radii_um=(.75, 2.), feature_groups=("synthetic",),
                  repeats=2, fixed_k=2, max_k=2)
    wide, old = assemble(source, mask, shift, resolution, tmp_path / "table", **common)
    narrow, new = assemble(source, mask, shift, resolution, tmp_path / "arrays",
        feature_storage="arrays", row_batch_size=rows, column_batch_size=columns, **common)
    reader = FeatureColumns(tmp_path / "arrays")
    assert len(narrow.columns) < len(wide.columns) - 150
    assert new["feature_blocks"] == old["feature_blocks"]
    assert new["cell_count"] == old["cell_count"] == 3
    assert narrow.cell_id.tolist() == ["2", "1", "3"]
    assert set(reader.available_columns) == set(wide.columns)
    pd.testing.assert_frame_equal(narrow, wide[list(narrow)], check_exact=False, rtol=1e-12, atol=1e-12)
    for group, record in reader.groups.items():
        actual = reader.read(slice(None), record["columns"])
        expected = wide[record["columns"]].to_numpy(float)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    for radius in ("0.75", "2"):
        a = sparse.load_npz(tmp_path / f"table/neighborhood_graph_{radius}um.npz")
        b = sparse.load_npz(tmp_path / f"arrays/neighborhood_graph_{radius}um.npz")
        for attr in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
    a = json.loads((tmp_path / "table/neighborhood_summary.json").read_text())
    b = json.loads((tmp_path / "arrays/neighborhood_summary.json").read_text())
    assert a["feature_groups"] == b["feature_groups"]
    assert a["feature_group_definitions"] == b["feature_group_definitions"]
    assert a["niche_discovery"]["selected_k"] == b["niche_discovery"]["selected_k"]
    reader.recheck()
    assert not list((tmp_path / "arrays").glob(".niche_fit_*"))
    assert before == {str(p.relative_to(source)): sha256_file(p) for p in source.rglob("*") if p.is_file()}


def test_arrays_statistics_only_and_outside_cells(tmp_path):
    source, mask, shift, resolution, domain = fixture(tmp_path)
    wide, _ = assemble(source, mask, shift, resolution, tmp_path / "table", feature_groups=())
    narrow, result = assemble(source, mask, shift, resolution, tmp_path / "arrays", feature_groups=(), feature_storage="arrays")
    pd.testing.assert_frame_equal(wide, narrow)
    assert "neighborhood_feature_store" not in result
    assert narrow.niche_id.isna().all()


@pytest.mark.parametrize("option,value", [("feature_storage", "invalid"), ("row_batch_size", 0), ("column_batch_size", -1)])
def test_invalid_options_fail_before_output(tmp_path, option, value):
    source, mask, shift, resolution, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        assemble(source, mask, shift, resolution, tmp_path / "bad", **{option: value})
    assert not (tmp_path / "bad").exists()


def test_memory_budget_not_silently_dropping_features(tmp_path):
    source, mask, shift, resolution, _ = base(tmp_path)
    with pytest.raises(ValueError, match="working estimate"):
        assemble(source, mask, shift, resolution, tmp_path / "bad", feature_storage="arrays", max_working_mb=.0001)
    assert not (tmp_path / "bad/cell_profiles_manifest.json").exists()


def test_bounded_aggregation_retains_all_axes_on_larger_sparse_fixture(tmp_path, record_property):
    from assemble_neighborhood_arrays import build_arrays
    from time import perf_counter
    n, d, batch = 3000, 129, 31
    rng = np.random.default_rng(43)
    values = rng.normal(size=(n, d))
    values[rng.random((n, d)) < .15] = np.nan
    values[:, -1] = np.nan
    root = tmp_path / "profile"
    (root / "feature_blocks").mkdir(parents=True)
    path = root / "feature_blocks/synthetic.npy"
    np.save(path, values)
    reads = []

    class BoundedSource:
        shape, dtype = values.shape, values.dtype

        def __getitem__(self, key):
            rows = key[0]
            count = len(range(*rows.indices(n))) if isinstance(rows, slice) else len(rows)
            assert count <= batch  # Includes dense-graph edge gathers.
            result = values[key]
            reads.append(result.shape)
            return result

    record = {"path": "feature_blocks/synthetic.npy", "sha256": sha256_file(path),
              "shape": [n, d], "feature_names": [f"axis{i}" for i in range(d)]}
    sorted_to_canonical = rng.permutation(n)
    canonical_to_sorted = np.argsort(sorted_to_canonical)
    rows = np.repeat(np.arange(n), 4)
    columns = ((np.arange(n)[:, None] + np.array([-2, -1, 1, 2])) % n).ravel()
    graph = sparse.csr_matrix((np.zeros(n * 4), (rows, columns)), shape=(n, n))
    profiles = pd.DataFrame({"sample_id": ["s"] * n, "cell_id": list(map(str, range(n))),
        "cell_uid": [f"s:{i}" for i in range(n)], "phenotype": ["unknown"] * n})
    summary = {"feature_groups": {"density": [], "composition": [], "orientation": []}}
    start = perf_counter()
    narrow, store, read = build_arrays(root, profiles, {1.: graph, 2.: graph}, canonical_to_sorted,
        {"synthetic": (record, BoundedSource())}, summary,
        row_batch_size=batch, column_batch_size=17, max_working_mb=1)
    elapsed = perf_counter() - start
    assert len(narrow) == n and len(narrow.columns) == len(profiles.columns) + 3
    assert len(store["synthetic"]["columns"]) == d * 2
    assert reads and max(shape[0] for shape in reads) <= batch
    assert max(shape[1] for shape in reads) <= 17
    # Independent full dense reference is test-only, never the implementation.
    ordered = values[sorted_to_canonical]
    finite = np.isfinite(ordered)
    adjacency = graph.copy(); adjacency.data[:] = 1
    counts = adjacency @ finite.astype(float)
    sums = adjacency @ np.where(finite, ordered, 0)
    expected = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)[canonical_to_sorted]
    for segment in store["synthetic"]["segments"]:
        actual = np.load(root / segment["path"], mmap_mode="r")
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(read(np.arange(9), store["own:synthetic"]["columns"]), values[sorted_to_canonical[:9]])
    record_property("canonical_cells", n)
    record_property("source_dimensions", d)
    record_property("retained_own_plus_neighbour_dimensions", d * 3)
    record_property("aggregation_seconds", elapsed)
    record_property("new_neighbour_payload_bytes", sum((root / record["path"]).stat().st_size for record in store["synthetic"]["segments"]))
