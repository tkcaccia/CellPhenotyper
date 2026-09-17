import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path):
    rscript = shutil.which("Rscript")
    if not rscript:
        pytest.skip("Rscript unavailable")
    check = subprocess.run([rscript, "-e", "if(!requireNamespace('data.table',quietly=TRUE))quit(status=77)"], capture_output=True)
    if check.returncode == 77:
        pytest.skip("R data.table unavailable")
    directory = tmp_path / "quant"
    directory.mkdir()
    for compartment in ("nuclei", "cyto"):
        pd.DataFrame({"label_id": ["001", "002", "003"], "centroid_x_px": [1, 2, 3], "centroid_y_px": [2, 3, 4],
                      "DAPI__mean": [.1, .2, .3], "CD8__mean": [.3, .4, .5], "TRITC__mean": [.8]*3,
                      "Cy5__mean": [.7]*3}).to_csv(directory / f"s_{compartment}_gigatime_quantification.csv", index=False)
    return rscript, directory


def run_loader(source, output):
    return subprocess.run([source[0], str(ROOT / "bin/load_gigatime_kodama_rawdata.R"), str(source[1]), str(output)], capture_output=True, text=True)


def test_marker_kodama_excludes_background_and_preserves_string_ids(source, tmp_path):
    output = tmp_path / "output"
    result = run_loader(source, output)
    assert result.returncode == 0, result.stdout + result.stderr
    names = (output / "gigatime_marker_features.txt").read_text().splitlines()
    assert len(names) == 4
    assert not any("TRITC" in name or "Cy5" in name for name in names)
    code = "load(commandArgs(TRUE)[1]);stopifnot(identical(common_ids,c('001','002','003')),all(is.finite(embeddings_raw$tile)))"
    check = subprocess.run([source[0], "-e", code, str(output / "rawdata.RData")], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


@pytest.mark.parametrize("fault", ["duplicate", "population", "missing"])
def test_marker_kodama_never_silently_drops_or_imputes(source, tmp_path, fault):
    path = source[1] / "s_cyto_gigatime_quantification.csv"
    frame = pd.read_csv(path, dtype={"label_id": str})
    if fault == "duplicate":
        frame.loc[1, "label_id"] = "001"
    elif fault == "population":
        frame = frame.iloc[:2]
    else:
        frame.loc[1, "CD8__mean"] = float("nan")
    frame.to_csv(path, index=False)
    result = run_loader(source, tmp_path / "bad")
    assert result.returncode != 0
    assert "no rows" in result.stderr or "no intersection" in result.stderr or "no silent imputation" in result.stderr


def test_literal_na_leading_zero_and_large_ids_remain_distinct_strings(source, tmp_path):
    for compartment in ("nuclei", "cyto"):
        path = source[1] / f"s_{compartment}_gigatime_quantification.csv"
        frame = pd.read_csv(path, dtype={"label_id": str})
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        frame["label_id"] = ["001", "1", "NA", "9007199254740993"]
        frame.to_csv(path, index=False)
    output = tmp_path / "literal_ids"
    result = run_loader(source, output)
    assert result.returncode == 0, result.stdout + result.stderr
    code = "load(commandArgs(TRUE)[1]);stopifnot(identical(common_ids,c('001','1','9007199254740993','NA')),nrow(embeddings_raw$tile)==4L)"
    check = subprocess.run([source[0], "-e", code, str(output / "rawdata.RData")], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


@pytest.mark.parametrize("invalid_id", ["", " 001", "001 ", "00 1", "00\t1", "00\n1", "00\x011", "00\u00a01"])
def test_whitespace_control_and_empty_ids_fail_without_normalizing(source, tmp_path, invalid_id):
    # The same malformed ID on both sides cannot pass via a successful merge.
    for compartment in ("nuclei", "cyto"):
        path = source[1] / f"s_{compartment}_gigatime_quantification.csv"
        frame = pd.read_csv(path, dtype={"label_id": str})
        frame.loc[0, "label_id"] = invalid_id
        frame.to_csv(path, index=False)
    result = run_loader(source, tmp_path / "invalid_id")
    assert result.returncode != 0
    assert "cell IDs" in result.stderr or "parsing warning" in result.stderr
    assert not (tmp_path / "invalid_id/rawdata.RData").exists()


@pytest.mark.parametrize("compartment", ["nuclei", "cyto"])
@pytest.mark.parametrize("column", ["centroid_x_px", "centroid_y_px"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), "not-a-number", True])
def test_each_compartment_requires_finite_numeric_coordinates(source, tmp_path, compartment, column, bad_value):
    path = source[1] / f"s_{compartment}_gigatime_quantification.csv"
    frame = pd.read_csv(path, dtype={"label_id": str})
    frame[column] = frame[column].astype(object)
    if bad_value is True:
        frame[column] = [True] * len(frame)
    else:
        frame.loc[0, column] = bad_value
    frame.to_csv(path, index=False)
    result = run_loader(source, tmp_path / "bad_coordinates")
    assert result.returncode != 0
    assert compartment in result.stderr and column in result.stderr and "finite numeric" in result.stderr
    assert not (tmp_path / "bad_coordinates/rawdata.RData").exists()


def test_legitimate_different_centroids_and_marker_column_order_are_preserved(source, tmp_path):
    path = source[1] / "s_cyto_gigatime_quantification.csv"
    frame = pd.read_csv(path, dtype={"label_id": str})
    frame["centroid_x_px"] += .75
    frame["centroid_y_px"] -= .5
    frame["DAPI__mean"] += .1
    # Whole-cell centroid differences reflect its different mask, not ID errors.
    frame.iloc[::-1, ::-1].to_csv(path, index=False)
    output = tmp_path / "valid_centroid_shift"
    result = run_loader(source, output)
    assert result.returncode == 0, result.stdout + result.stderr
    code = """load(commandArgs(TRUE)[1]);stopifnot(
      identical(common_ids,c('001','002','003')),
      identical(as.numeric(xy[,1]),c(1,2,3)),identical(as.numeric(xy[,2]),c(2,3,4)),
      identical(colnames(embeddings_raw$tile),c('nuclei__DAPI__mean','nuclei__CD8__mean','cyto__DAPI__mean','cyto__CD8__mean')),
      isTRUE(all.equal(as.numeric(embeddings_raw$tile[,'cyto__DAPI__mean']),c(.2,.3,.4))))"""
    check = subprocess.run([source[0], "-e", code, str(output / "rawdata.RData")], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


@pytest.mark.parametrize("column", ["label_id", "centroid_x_px", "DAPI__mean"])
def test_duplicate_header_names_are_rejected(source, tmp_path, column):
    path = source[1] / "s_cyto_gigatime_quantification.csv"
    lines = path.read_text().splitlines()
    original_names = lines[0].split(",")
    index = original_names.index(column)
    path.write_text("\n".join([line + "," + line.split(",")[index] for line in lines]) + "\n")
    result = run_loader(source, tmp_path / "duplicate_header")
    assert result.returncode != 0
    assert "duplicate CSV header" in result.stderr


def test_csv_parser_warning_is_an_error_not_partial_input(source, tmp_path):
    path = source[1] / "s_cyto_gigatime_quantification.csv"
    path.write_text(path.read_text() + "broken_row,1\n")
    result = run_loader(source, tmp_path / "malformed_csv")
    assert result.returncode != 0
    assert "Marker CSV parsing warning" in result.stderr


@pytest.mark.parametrize("change", ["different_marker", "extra_marker"])
def test_biological_marker_schema_must_match_between_compartments(source, tmp_path, change):
    path = source[1] / "s_cyto_gigatime_quantification.csv"
    frame = pd.read_csv(path, dtype={"label_id": str})
    if change == "different_marker":
        frame = frame.rename(columns={"CD8__mean": "CD4__mean"})
    else:
        frame["PanCK__mean"] = .25
    frame.to_csv(path, index=False)
    result = run_loader(source, tmp_path / "schema_mismatch")
    assert result.returncode != 0
    assert "biological mean-marker schemas differ" in result.stderr


def test_background_qc_columns_do_not_define_biological_schema(source, tmp_path):
    path = source[1] / "s_cyto_gigatime_quantification.csv"
    frame = pd.read_csv(path, dtype={"label_id": str}).drop(columns=["TRITC__mean", "Cy5__mean"])
    frame.to_csv(path, index=False)
    result = run_loader(source, tmp_path / "biological_only")
    assert result.returncode == 0, result.stdout + result.stderr
