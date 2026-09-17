"""Strict default CSV -> R KODAMA contracts, with no learned-image inference."""
import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from test_uni2_binary_r import (
    ANALYSIS, fixture as binary_fixture, load, native_library, readback, rrun, rscript,
)


def csv_fixture(tmp_path, *, modes=("tile",), ids=None, values=None, compressed=False,
                extraction_centers=False):
    ids = list(ids) if ids is not None else ["010", "001", "a", "7"]
    if values is None:
        values = np.array([[1, 1, 1, 1], [2, 30, 4, 8], [3, 50, 6, 14], [4, 80, 9, 22]], dtype=np.float64)
    values = np.asarray(values)
    directories = {mode: tmp_path / mode for mode in ("tile", "cyto", "inner_square", "nuclei")}
    for directory in directories.values():
        directory.mkdir()
    coordinates = {label: (index + .25, index + 4.25) for index, label in enumerate(ids)}
    paths = {}
    feature_names = [f"feat_{index+1}" for index in range(values.shape[1])]
    for mode_index, mode in enumerate(modes):
        order = np.arange(len(ids)) if mode_index == 0 else np.arange(len(ids))[::-1]
        names = [ids[index] for index in order]
        rows = pd.DataFrame({"cell_id": names,
            "x": [coordinates[name][0] for name in names], "y": [coordinates[name][1] for name in names],
            "observation_type": "grid"})
        if extraction_centers:
            rows = rows.rename(columns={"x": "cx", "y": "cy"})
            rows[["cx", "cy"]] = np.floor(rows[["cx", "cy"]]).astype(int)
        rows = pd.concat([rows, pd.DataFrame(values[order] + mode_index, columns=feature_names)], axis=1)
        paths[mode] = []
        for shard, positions in enumerate(np.array_split(np.arange(len(ids)), 2)):
            path = directories[mode] / (f"uni2_embeddings_shard{shard:04}.csv" + (".gz" if compressed else ""))
            rows.iloc[positions].to_csv(path, index=False, float_format="%.17g")
            paths[mode].append(path)
    annotation = tmp_path / "objects.csv"
    pd.DataFrame({"label": ids[::-1], "x": [coordinates[name][0] for name in ids[::-1]],
        "y": [coordinates[name][1] for name in ids[::-1]], "polygon_label": "grid"}).to_csv(annotation, index=False)
    return directories, annotation, paths, ids, values


def read_strings(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def reject_load(rscript, source, output, modes="tile", top=100):
    result = load(rscript, source, output, modes, top, check=False)
    assert result.returncode != 0, "Malformed CSV unexpectedly entered the canonical KODAMA input"
    assert not (output / "rawdata.RData").exists()
    return result


@pytest.mark.parametrize("compressed", [False, True])
def test_csv_literal_ids_all_selected_modes_source_hashes_and_explicit_analysis_mapping(tmp_path, rscript, compressed):
    ids = ["010", "001", "NA", "NaN", "9007199254740993", "7"]
    values = np.arange(24, dtype=np.float64).reshape(6, 4)
    source = csv_fixture(tmp_path, modes=("tile", "inner_square", "nuclei", "cyto"), ids=ids,
                         values=values, compressed=compressed)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [source[1], *[path for paths in source[2].values() for path in paths]]}
    output = tmp_path / "loaded"
    load(rscript, source, output, "full")
    metadata, arrays = readback(rscript, output)
    order = np.argsort(ids)
    assert metadata["common_ids"] == metadata["ann_ids"] == sorted(ids)
    provenance = json.loads((output / "embedding_input_provenance.json").read_text())
    assert provenance["format"] == "cellphenotyper_uni2_csv_input"
    assert provenance["schema_version"] == "1.0.0"
    assert provenance["analysis_observation_ids"] == sorted(ids)
    assert provenance["annotation_sha256"] == before[source[1]]
    for index, mode in enumerate(source[2]):
        assert metadata["modes"][mode]["ids"] == sorted(ids)
        np.testing.assert_array_equal(arrays[mode], (values + index)[order])
        source_ids = ids if index == 0 else ids[::-1]
        mapping = provenance["source_to_analysis"][mode]
        assert mapping["source_observation_ids"] == source_ids
        assert mapping["analysis_row_from_source"] == [source_ids.index(value) + 1 for value in sorted(ids)]
        assert provenance["modes"][mode]["selected_feature_names"] == ["feat_1", "feat_2", "feat_3", "feat_4"]
        mode_json = json.dumps(provenance["modes"][mode], sort_keys=True)
        for path in source[2][mode]:
            assert before[path] in mode_json
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}


@pytest.mark.parametrize("target", ["shard", "annotation"])
@pytest.mark.parametrize("bad_id", ["", " ", " 001", "001 ", "a\tb", "a\nb", "a\rb", "a\x01b"])
def test_csv_empty_whitespace_and_control_ids_are_rejected_without_trimming_or_dropping(tmp_path, rscript, target, bad_id):
    source = csv_fixture(tmp_path)
    path = source[2]["tile"][0] if target == "shard" else source[1]
    frame = read_strings(path)
    frame.loc[0, "cell_id" if target == "shard" else "label"] = bad_id
    frame.to_csv(path, index=False)
    reject_load(rscript, source, tmp_path / "bad")


@pytest.mark.parametrize("target", ["within_shard", "across_shards", "annotation"])
def test_csv_duplicate_ids_are_rejected_instead_of_deduplicated(tmp_path, rscript, target):
    source = csv_fixture(tmp_path)
    if target == "annotation":
        path, key = source[1], "label"
        frame = read_strings(path)
        frame.loc[1, key] = frame.loc[0, key]
    else:
        path, key = source[2]["tile"][1 if target == "across_shards" else 0], "cell_id"
        frame = read_strings(path)
        frame.loc[1, key] = read_strings(source[2]["tile"][0]).loc[0, key]
    frame.to_csv(path, index=False)
    reject_load(rscript, source, tmp_path / "bad")


@pytest.mark.parametrize("target", ["tile_missing", "inner_missing", "inner_foreign", "annotation_missing", "annotation_foreign"])
def test_csv_selected_mode_and_annotation_id_sets_must_match_exactly(tmp_path, rscript, target):
    source = csv_fixture(tmp_path, modes=("tile", "inner_square"))
    if target.startswith("annotation"):
        path, key = source[1], "label"
    else:
        path, key = source[2]["tile" if target.startswith("tile") else "inner_square"][0], "cell_id"
    frame = read_strings(path)
    if target.endswith("missing"):
        frame = frame.iloc[1:]
    else:
        frame.loc[0, key] = "foreign_cell"
    frame.to_csv(path, index=False)
    reject_load(rscript, source, tmp_path / "bad", "tile,inner_square")


@pytest.mark.parametrize("kind", ["feature_order", "feature_name", "feature_missing", "feature_extra", "duplicate_feature_name"])
def test_csv_every_shard_requires_the_exact_ordered_feature_schema(tmp_path, rscript, kind):
    source = csv_fixture(tmp_path)
    path = source[2]["tile"][1]
    frame = read_strings(path)
    if kind == "feature_order":
        frame = frame[["cell_id", "x", "y", "observation_type", "feat_2", "feat_1", "feat_3", "feat_4"]]
    elif kind == "feature_name":
        frame = frame.rename(columns={"feat_4": "feat_99"})
    elif kind == "feature_missing":
        frame = frame.drop(columns="feat_4")
    elif kind == "feature_extra":
        frame["feat_99"] = "0"
    else:
        frame.columns = ["feat_3" if column == "feat_4" else column for column in frame]
    frame.to_csv(path, index=False)
    reject_load(rscript, source, tmp_path / "bad", top=1)


@pytest.mark.parametrize("bad_value", ["NaN", "NA", "Inf", "-Inf", "not_numeric", ""])
@pytest.mark.parametrize("top", [1, 100])
def test_csv_all_features_must_be_numeric_and_finite_even_when_not_selected(tmp_path, rscript, bad_value, top):
    values = np.array([[0, 0, 0, 0], [100, 10, 1, 0], [200, 20, 2, 0], [300, 30, 3, 0]], dtype=float)
    source = csv_fixture(tmp_path, values=values)
    path = source[2]["tile"][1]
    frame = read_strings(path)
    frame.loc[1, "feat_4"] = bad_value  # Zero-variance column would not be selected at top=1.
    frame.to_csv(path, index=False)
    reject_load(rscript, source, tmp_path / "bad", top=top)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("location", ["shard", "annotation"])
def test_csv_conflicting_exact_centroid_coordinates_are_rejected(tmp_path, rscript, axis, location):
    source = csv_fixture(tmp_path)
    path = source[2]["tile"][0] if location == "shard" else source[1]
    frame = read_strings(path)
    frame.loc[0, axis] = str(float(frame.loc[0, axis]) + 1)
    frame.to_csv(path, index=False)
    reject_load(rscript, source, tmp_path / "bad")


def test_csv_extraction_cx_cy_are_not_equated_to_floating_point_centroids(tmp_path, rscript):
    source = csv_fixture(tmp_path, extraction_centers=True)
    output = tmp_path / "loaded"
    load(rscript, source, output)
    metadata, arrays = readback(rscript, output)
    assert metadata["common_ids"] == sorted(source[3])
    np.testing.assert_array_equal(arrays["tile"], source[4][np.argsort(source[3])])


@pytest.mark.parametrize("kind", ["extra_field", "missing_field", "unterminated_quote", "truncated_gzip"])
def test_csv_and_gzip_parse_damage_cannot_yield_a_partial_successful_load(tmp_path, rscript, kind):
    source = csv_fixture(tmp_path, compressed=kind == "truncated_gzip")
    path = source[2]["tile"][1]
    if kind == "truncated_gzip":
        # The rows decompress fully, but the gzip trailer is absent. A command
        # that ignores gzip's exit status would otherwise accept this input.
        path.write_bytes(path.read_bytes()[:-8])
    else:
        lines = path.read_text().splitlines()
        if kind == "extra_field":
            lines[-1] += ",extra"
        elif kind == "missing_field":
            lines[-1] = lines[-1].rsplit(",", 1)[0]
        else:
            lines[-1] = '"' + lines[-1]
        path.write_text("\n".join(lines) + "\n")
    reject_load(rscript, source, tmp_path / "bad")


def test_csv_variance_selection_matches_binary_values_feature_names_and_analysis_order(tmp_path, rscript):
    ids = ["2", "1", "20", "10"]
    values = np.array([[1, 1, 1, 1], [2, 30, 4, 8], [3, 50, 6, 14], [4, 80, 9, 22]], dtype=np.float64)
    csv_root, binary_root = tmp_path / "csv", tmp_path / "binary"
    csv_root.mkdir()
    binary_root.mkdir()
    csv = csv_fixture(csv_root, ids=ids, values=values, compressed=True)
    binary = binary_fixture(binary_root, ids=ids, values=values)
    load(rscript, csv, tmp_path / "csv_loaded", top=2)
    load(rscript, binary, tmp_path / "binary_loaded", top=2)
    one, csv_values = readback(rscript, tmp_path / "csv_loaded")
    two, binary_values = readback(rscript, tmp_path / "binary_loaded")
    assert one["common_ids"] == two["common_ids"] == sorted(ids)
    assert one["modes"]["tile"]["ids"] == two["modes"]["tile"]["ids"] == sorted(ids)
    assert one["modes"]["tile"]["features"] == two["modes"]["tile"]["features"] == ["feat_2", "feat_4"]
    np.testing.assert_array_equal(csv_values["tile"], binary_values["tile"])


def analysis_command(rscript, raw, output):
    return [rscript, str(ANALYSIS), str(raw), str(output), "--embedding-mode", "tile",
        "--dims-to-run", "3", "--spark-top", "6", "--landmarks", "20", "--kodama-ncomp", "50",
        "--n-cores", "1", "--backend", "cpu", "--export-native-graph", "false"]


def test_actual_csv_rawdata_to_cpu_kodama_preserves_receipts_and_ncomp_request_distinct_from_pca(tmp_path, rscript, native_library):
    ids = [f"{index:03}" for index in range(40, 0, -1)]
    source = csv_fixture(tmp_path, ids=ids, values=np.random.default_rng(2003).normal(size=(40, 6)), compressed=True)
    raw, output = tmp_path / "raw", tmp_path / "native"
    load(rscript, source, raw)
    result = subprocess.run(analysis_command(rscript, raw / "rawdata.RData", output),
        env={**os.environ, "R_LIBS": native_library}, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((output / "clustering_representations.json").read_text())
    receipt = metadata["embedding_input_provenance"]
    assert receipt["format"] == "cellphenotyper_uni2_csv_input"
    assert receipt["analysis_observation_ids"] == sorted(ids)
    assert receipt == json.loads((raw / "embedding_input_provenance.json").read_text())
    assert metadata["rawdata_input_sha256"] == hashlib.sha256((raw / "rawdata.RData").read_bytes()).hexdigest()
    # A 3-D PCA fixture necessarily clamps the effective value to 3. This proves
    # the independent requested argument, not fitting 50 components in this test.
    assert metadata["requested_kodama_ncomp"] == 50
    assert metadata["effective_kodama_ncomp"] == 3
    rrun(rscript, r'''
      a<-commandArgs(TRUE);p<-new.env();k<-new.env()
      load(file.path(a[1],"pca_full_3.RData"),envir=p)
      load(file.path(a[1],"kodama_full_3.RData"),envir=k)
      stopifnot(ncol(p$pca)==3L,
        identical(rownames(p$pca),p$embedding_input_provenance$analysis_observation_ids),
        identical(rownames(k$vis),p$embedding_input_provenance$analysis_observation_ids),
        identical(p$pca_model_metadata$embedding_input_provenance,k$representation_metadata$embedding_input_provenance))
    ''', output)


def test_valid_legacy_rawdata_without_receipt_remains_usable_without_inventing_provenance(tmp_path, rscript, native_library):
    ids = [f"{index:03}" for index in range(40, 0, -1)]
    source = csv_fixture(tmp_path, ids=ids, values=np.random.default_rng(2003).normal(size=(40, 6)))
    raw, output = tmp_path / "raw", tmp_path / "legacy"
    load(rscript, source, raw)
    rrun(rscript, 'a<-commandArgs(TRUE);load(a[1]);save(ann,xy,embeddings_raw,file=a[1])', raw / "rawdata.RData")
    result = subprocess.run(analysis_command(rscript, raw / "rawdata.RData", output),
        env={**os.environ, "R_LIBS": native_library}, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((output / "clustering_representations.json").read_text())
    assert metadata["embedding_input_provenance"] is None
    rrun(rscript, 'a<-commandArgs(TRUE);load(file.path(a[1],"pca_full_3.RData"));'
         'stopifnot(identical(rownames(pca),sort(ann$label)),nrow(pca)==40L)', output)


@pytest.mark.parametrize("mutation", [
    'rownames(embeddings_raw$tile)[2]<-rownames(embeddings_raw$tile)[1]',
    'rownames(embeddings_raw$tile)[1]<-" "',
    'embeddings_raw$tile<-embeddings_raw$tile[-1,,drop=FALSE]',
    'rownames(embeddings_raw$tile)[1]<-"foreign_cell"',
    'embeddings_raw$tile[1,1]<-NA_real_',
    'embeddings_raw$tile[1,1]<-Inf',
    'colnames(embeddings_raw$tile)[2]<-colnames(embeddings_raw$tile)[1]',
    'ann$label[2]<-ann$label[1]',
    'ann$x[1]<-NA_real_',
])
def test_bad_legacy_rawdata_cannot_bypass_validation_by_omitting_provenance(tmp_path, rscript, native_library, mutation):
    ids = [f"{index:03}" for index in range(40, 0, -1)]
    source = csv_fixture(tmp_path, ids=ids, values=np.random.default_rng(2003).normal(size=(40, 6)))
    raw = tmp_path / "raw"
    load(rscript, source, raw)
    rrun(rscript, 'a<-commandArgs(TRUE);load(a[1]);' + mutation +
         ';save(ann,xy,embeddings_raw,file=a[1])', raw / "rawdata.RData")
    output = tmp_path / "rejected"
    result = subprocess.run(analysis_command(rscript, raw / "rawdata.RData", output),
        env={**os.environ, "R_LIBS": native_library}, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (output / "pca_full_3.RData").exists()
    assert not (output / "kodama_full_3.RData").exists()
