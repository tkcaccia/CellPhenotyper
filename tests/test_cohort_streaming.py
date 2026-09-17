"""Array-backed/mixed cohort parity and portable source binding; no inference."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from test_cohort_niches import make_spatial, rewrite_manifest
from test_cohort_graph_contract import make_wide_float_spatial
from test_cohort_niche_io import reseal
from cell_profile_io import sha256_file
from neighborhood_feature_io import FeatureColumns, finalize_store
import fit_cohort_niches as cohort
from cohort_niche_io import load_cohort_bundle, verify_cohort_sources, COHORT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def as_store(source, destination):
    """Repackage actual producer values with the real completion writer."""
    shutil.copytree(source, destination)
    path = destination / "cell_profiles_manifest.json"
    manifest = json.loads(path.read_text())
    summary = json.loads((destination / "neighborhood_summary.json").read_text())
    frame = pd.read_parquet(destination / "cell_profiles.parquet")
    groups, moved = {}, []
    for index, (name, definition) in enumerate(summary["feature_group_definitions"].items()):
        block_name = definition.get("source_feature_block")
        if not block_name:
            continue
        columns = summary["feature_groups"][name]
        segments = []
        if name == "own:" + block_name:
            block = manifest["feature_blocks"][block_name]
            segments.append({key: copy.deepcopy(block[key]) for key in ("path", "sha256", "shape", "dtype")})
            segments[-1]["columns"] = columns
        else:
            for begin in range(0, len(columns), 64):
                selected = columns[begin:begin + 64]
                values = frame[selected].to_numpy(dtype=np.float64)
                output = destination / "neighborhood_features" / f"group_{index}_{begin}.npy"
                output.parent.mkdir(exist_ok=True)
                np.save(output, values, allow_pickle=False)
                segments.append({"path": str(output.relative_to(destination)), "sha256": sha256_file(output),
                                 "shape": list(values.shape), "dtype": "float64", "columns": selected})
        groups[name] = {"columns": columns, "segments": segments}
        moved.extend(columns)
    assert groups, "A store fixture must have actual derived feature groups"
    narrow = frame.drop(columns=moved)
    narrow.to_parquet(destination / "cell_profiles.parquet", index=False)
    narrow.to_csv(destination / "cell_profiles.csv", index=False)
    for name in ("cell_profiles.parquet", "cell_profiles.csv"):
        manifest["files"][name] = sha256_file(destination / name)
    manifest["neighborhood_feature_store"] = finalize_store(destination, groups)
    path.write_text(json.dumps(manifest))
    return destination


def tree_hashes(directory):
    return {str(p.relative_to(directory)): sha256_file(p) for p in directory.rglob("*") if p.is_file()}


def reseal_store(root):
    """Rehash deliberately altered fixture payloads to test semantic checks."""
    manifest_path = root / "cell_profiles_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = root / manifest["neighborhood_feature_store"]["path"]
    store = json.loads(path.read_text())
    for group in store["groups"].values():
        for segment in group["segments"]:
            segment["sha256"] = sha256_file(root / segment["path"])
    path.write_text(json.dumps(store))
    manifest["neighborhood_feature_store"]["sha256"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("fixed_k", [None, 2])
def test_actual_dense_and_array_backed_or_mixed_cohorts_are_equivalent(tmp_path, mixed, fixed_k):
    legacy = [make_spatial(tmp_path / sample, sample) for sample in ("A", "B")]
    streamed = [as_store(legacy[0], tmp_path / "array_A"),
                legacy[1] if mixed else as_store(legacy[1], tmp_path / "array_B")]
    frozen = {str(root): tree_hashes(root) for root in set(legacy + streamed)}
    expected, dense_model, dense_summary = cohort.fit_cohort(legacy, tmp_path / "dense", fixed_k=fixed_k, repeats=2)
    actual, model, summary = cohort.fit_cohort(streamed[::-1], tmp_path / "streamed",
        fixed_k=fixed_k, repeats=2, row_batch_size=2)
    pd.testing.assert_frame_equal(actual.drop(columns="cohort_niche_model_id"),
        expected.drop(columns="cohort_niche_model_id"), check_exact=False, rtol=1e-10, atol=1e-10)
    assert model["feature_groups"] == dense_model["feature_groups"]
    assert model["source_feature_definitions"] == dense_model["source_feature_definitions"]
    assert model["contracts"] == dense_model["contracts"]
    assert model["sample_weighting"] == dense_model["sample_weighting"]
    assert summary["missingness"] == dense_summary["missingness"]
    assert summary["selected_k"] == dense_summary["selected_k"]
    for name, record in model["discovery"]["scaling"].items():
        other = dense_model["discovery"]["scaling"][name]
        for field in ("columns", "missing_indicator_columns", "matrix_columns", "weight", "dimensions"):
            assert record[field] == other[field]
        for field in ("means", "std"):
            np.testing.assert_allclose(record[field], other[field], rtol=1e-10, atol=1e-10)
    assert "streaming_execution" not in dense_model["discovery"]
    assert summary["streaming_execution"]["row_batch_size"] == 2
    assert summary["streaming_execution"]["features_or_cells_dropped_for_budget"] is False
    assert {str(root): tree_hashes(root) for root in set(legacy + streamed)} == frozen
    for root in streamed:
        loaded, record = load_cohort_bundle(tmp_path / "streamed", profile_dir=root)
        selected = actual[actual.sample_id == record["sample_id"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(loaded, selected[cohort.KEYS + COHORT_COLUMNS])
        verify_cohort_sources(tmp_path / "streamed", root, record)
        assert ("store_files" in record["source_identity"]) == (root.name.startswith("array_"))


def test_virtual_union_missing_blocks_unknown_aliases_and_requested_row_order(tmp_path):
    legacy = [make_spatial(tmp_path / "A", "A", phenotype="unknown"),
              make_spatial(tmp_path / "B", "B", phenotype="unknown_99", with_features=False)]
    definition = {"method": "synthetic_test_labels", "unknown_label_semantics":
        "unknown, __unknown__, and producer unknown_<unmapped_type_id> are missing categorical assignments"}
    for source in legacy:
        rewrite_manifest(source, lambda manifest: manifest["phenotype_definition"].update(verified=True, definition=definition))
    dense_records = [cohort.source_record(root) for root in legacy]
    groups, _, _ = cohort.compatibility(dense_records)
    dense, missing = cohort.pool_profiles(dense_records, groups)
    policy = cohort.composition_policy(dense_records)
    fitting_groups, aliases = cohort.normalize_missing_composition(dense, groups, policy)
    sources = [as_store(legacy[0], tmp_path / "array_A"), legacy[1]]
    records = [cohort.source_record(root) for root in sources]
    narrow, actual_missing = cohort.pool_profiles(records, groups, streaming=True, row_batch_size=2)
    virtual = cohort.CohortValues(records, groups, aliases)
    columns = sorted({column for group in fitting_groups.values() for column in group})
    for rows in (np.array([13, 0, 8, 2, 8]), slice(None, None, -1), np.array([], dtype=int)):
        np.testing.assert_array_equal(virtual.read(rows, columns), dense.iloc[rows][columns].to_numpy(float))
    assert actual_missing == missing
    assert not any(name.startswith("own_synthetic:") for name in narrow)
    assert "r3um_synthetic_mean:test_feature" not in narrow
    np.testing.assert_array_equal(virtual.read(np.array([7, 13]), ["own_synthetic:test_feature"]), [[np.nan], [np.nan]])
    result, model, _ = cohort.fit_cohort(sources, tmp_path / "cohort", repeats=2, row_batch_size=2)
    assert len(result) == 14 and model["phenotype_composition_policy"]["temporary_unknown_alias_columns"] == aliases


def test_streamed_source_verification_bounds_rows_and_preserves_67_axes_zero_edges(tmp_path, monkeypatch):
    sources = [as_store(make_wide_float_spatial(tmp_path / name, name), tmp_path / ("array_" + name)) for name in ("A", "B")]
    original = FeatureColumns.read
    calls = []
    def bounded(self, rows, columns):
        count = len(range(*rows.indices(self.cell_count))) if isinstance(rows, slice) else len(rows)
        assert count <= 2
        calls.append((count, len(columns)))
        return original(self, rows, columns)
    monkeypatch.setattr(FeatureColumns, "read", bounded)
    parquet_read = pd.read_parquet
    def narrow_only(*args, **kwargs):
        selected = kwargs.get("columns")
        assert selected is not None
        assert not any(name.startswith("own_synthetic:") or "_synthetic_mean:" in name for name in selected)
        return parquet_read(*args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", narrow_only)
    result, model, summary = cohort.fit_cohort(sources, tmp_path / "out", fixed_k=1, repeats=2, row_batch_size=2)
    assert len(result) == 14 and result.cohort_niche_status.eq("outside_tissue_support").sum() == 2
    assert (2, 64) in calls and any(width == 3 for _, width in calls)
    assert len(model["feature_groups"]["own:synthetic"]) == 67
    assert summary["streaming_execution"]["scaled_dimensions"] == model["discovery"]["feature_dimensions"]


@pytest.mark.parametrize("fault", ["mean", "coverage", "store_hash"])
def test_array_store_corruption_or_rehashed_false_projection_is_rejected(tmp_path, fault):
    sources = [as_store(make_spatial(tmp_path / name, name), tmp_path / ("array_" + name)) for name in ("A", "B")]
    root = sources[1]
    if fault in {"mean", "store_hash"}:
        reader = FeatureColumns(root)
        segment = reader.groups["synthetic"]["segments"][0]
        path = root / segment["path"]
        values = np.load(path, allow_pickle=False)
        values[0, 0] += 1
        np.save(path, values, allow_pickle=False)
        if fault == "mean":
            reseal_store(root)
        match = "neighbour means" if fault == "mean" else "SHA256"
    else:
        path = root / "cell_profiles.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "own_synthetic_observed_fraction"] = .25
        frame.to_parquet(path, index=False)
        rewrite_manifest(root, lambda manifest: manifest["files"].update({path.name: sha256_file(path)}))
        match = "own feature coverage"
    with pytest.raises(ValueError, match=match):
        cohort.fit_cohort(sources, tmp_path / "not_completed", fixed_k=2, repeats=2, row_batch_size=2)
    assert not (tmp_path / "not_completed").exists()


def test_store_receipt_round_trip_copy_model_free_reader_and_missing_inventory(tmp_path):
    sources = [as_store(make_spatial(tmp_path / name, name), tmp_path / ("array_" + name)) for name in ("A", "B")]
    output = tmp_path / "cohort"
    assignments, model, _ = cohort.fit_cohort(sources, output, fixed_k=2, repeats=2, row_batch_size=2)
    relocated = tmp_path / "relocated"
    shutil.copytree(output, relocated / "bundle")
    shutil.copytree(sources[0], relocated / "profile")
    expected, record = load_cohort_bundle(relocated / "bundle", profile_dir=relocated / "profile")
    assert len(expected) == 7 and "neighborhood_features/feature_store.json" in record["source_identity"]["store_files"]
    python = ROOT / ".venv-spatialdata/bin/python"
    if python.is_file():
        script = "import sys; sys.path.insert(0,sys.argv[1]); from cohort_niche_io import load_cohort_bundle; f,r=load_cohort_bundle(sys.argv[2],profile_dir=sys.argv[3]); assert len(f)==7; assert not ({'sklearn','torch'} & set(sys.modules)); print('MODEL_FREE_STORE_COHORT_OK')"
        result = subprocess.run([str(python), "-c", script, str(ROOT / "bin"), str(relocated / "bundle"), str(relocated / "profile")],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "MODEL_FREE_STORE_COHORT_OK" in result.stdout
    model["sources"][0].pop("store_files")
    reseal(output, model=model, assignments=assignments)
    with pytest.raises(ValueError, match="array-backed feature inventory"):
        load_cohort_bundle(output, profile_dir=sources[0])


def test_source_mutation_after_streaming_fit_prevents_completion_and_cleans_scratch(tmp_path, monkeypatch):
    import niche_clustering
    sources = [as_store(make_spatial(tmp_path / name, name), tmp_path / ("array_" + name)) for name in ("A", "B")]
    original = niche_clustering.discover_niches_streaming
    scratch = []
    def changed(*args, **kwargs):
        scratch.append(Path(args[3]))
        result = original(*args, **kwargs)
        path = sources[0] / "neighborhood_features/feature_store.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result
    monkeypatch.setattr(niche_clustering, "discover_niches_streaming", changed)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        cohort.fit_cohort(sources, tmp_path / "not_completed", fixed_k=2, repeats=2, row_batch_size=2)
    assert scratch and not scratch[0].parent.exists()
    assert not (tmp_path / "not_completed").exists()


def test_source_mutation_during_output_write_has_no_completion_receipt(tmp_path, monkeypatch):
    sources = [as_store(make_spatial(tmp_path / name, name), tmp_path / ("array_" + name)) for name in ("A", "B")]
    original = pd.DataFrame.to_parquet
    def changed(frame, path, *args, **kwargs):
        result = original(frame, path, *args, **kwargs)
        if Path(path).name == "cohort_niche_assignments.parquet":
            store = sources[0] / "neighborhood_features/feature_store.json"
            store.write_bytes(store.read_bytes() + b" ")
        return result
    monkeypatch.setattr(pd.DataFrame, "to_parquet", changed)
    output = tmp_path / "partial_output"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        cohort.fit_cohort(sources, output, fixed_k=2, repeats=2, row_batch_size=2)
    assert (output / "cohort_niche_assignments.parquet").exists()
    assert not (output / "cohort_niches_completion.json").exists()


def test_streaming_does_not_apply_dense_budget_or_silently_drop_feature_axes(tmp_path):
    legacy = [make_wide_float_spatial(tmp_path / name, name) for name in ("A", "B")]
    stores = [as_store(root, tmp_path / ("array_" + name)) for root, name in zip(legacy, ("A", "B"))]
    _, baseline, dense_summary = cohort.fit_cohort(legacy, tmp_path / "dense", fixed_k=1, repeats=2)
    _, streamed, summary = cohort.fit_cohort(stores, tmp_path / "streamed", fixed_k=1, repeats=2,
                                            row_batch_size=2, max_working_mb=.15)
    assert dense_summary["estimated_dense_working_mb"] > .15
    assert summary["streaming_execution"]["estimated_numerical_working_mb"] < .15
    assert streamed["feature_groups"] == baseline["feature_groups"]
    assert streamed["discovery"]["feature_dimensions"] == baseline["discovery"]["feature_dimensions"]
    with pytest.raises(ValueError, match="dense working estimate"):
        cohort.fit_cohort(legacy, tmp_path / "dense_refused", fixed_k=1, repeats=2, max_working_mb=.15)
    with pytest.raises(ValueError, match="working estimate"):
        cohort.fit_cohort(stores, tmp_path / "stream_refused", fixed_k=1, repeats=2, max_working_mb=.000001)
    assert not (tmp_path / "stream_refused").exists()


@pytest.mark.parametrize("row_batch_size", [0, True, 1.5])
def test_invalid_row_batch_fails_before_reading_sources(tmp_path, row_batch_size):
    with pytest.raises(ValueError, match="row_batch_size"):
        cohort.fit_cohort([], tmp_path / "not_completed", row_batch_size=row_batch_size)
    assert not (tmp_path / "not_completed").exists()
