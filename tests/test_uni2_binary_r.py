"""Actual Python binary writer -> R loader/KODAMA handoff, no image inference."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from uni2_embedding_io import write_binary_shard, binary_shard_paths, discover_embedding_shards, sha256_file

LOADER = ROOT / "bin/load_kodama_rawdata.R"
ANALYSIS = ROOT / "bin/run_kodama_analysis.R"


@pytest.fixture(scope="module")
def rscript():
    executable = shutil.which("Rscript")
    if not executable:
        pytest.skip("Rscript unavailable")
    check = subprocess.run([executable, "-e", 'p<-c("data.table","jsonlite","digest");if(!all(vapply(p,requireNamespace,logical(1),quietly=TRUE)))quit(status=77)'], capture_output=True, text=True)
    if check.returncode == 77:
        pytest.skip("R binary I/O packages unavailable")
    assert check.returncode == 0, check.stderr
    return executable


def rrun(rscript, code, *args, check=True, env=None):
    result = subprocess.run([rscript, "-e", code, *map(str, args)], capture_output=True, text=True, timeout=90, env=env)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def fixture(tmp_path, dtype=np.float64, modes=("tile",), ids=None, values=None, declare_mode=True):
    ids = ids or ["010", "001", "a", "7"]
    if values is None:
        values = np.array([[1, -0.0, 2, .1], [np.nextafter(1., 2.), 9, 3, .2],
                           [3, 2, 7, .3], [4, 5, 11, .4]], dtype=dtype)
    directories = {mode: tmp_path / mode for mode in ("tile", "cyto", "inner_square", "nuclei")}
    for directory in directories.values():
        directory.mkdir()
    coords = {cell_id: (float(i) + .25, float(i) + 4.25) for i, cell_id in enumerate(ids)}
    manifests = {}
    for index, mode in enumerate(modes):
        order = np.arange(len(ids)) if index == 0 else np.arange(len(ids))[::-1]
        names = [ids[i] for i in order]
        row_values = values[order] + index if index else values[order]
        records = pd.DataFrame({"cell_id": names, "x": [coords[name][0] for name in names],
                                "y": [coords[name][1] for name in names], "observation_type": "grid"})
        manifests[mode] = []
        for shard, positions in enumerate(np.array_split(np.arange(len(ids)), 2)):
            manifests[mode].append(write_binary_shard(directories[mode] / f"uni2_embeddings_shard{shard:04}",
                records.iloc[positions], row_values[positions], embedding_mode=mode if declare_mode else None))
    annotation = tmp_path / "objects.csv"
    pd.DataFrame({"label": ids[::-1], "x": [coords[name][0] for name in ids[::-1]],
                  "y": [coords[name][1] for name in ids[::-1]], "polygon_label": "grid"}).to_csv(annotation, index=False)
    return directories, annotation, manifests, ids, values


def load(rscript, source, destination, modes="tile", top=100, check=True):
    directories, annotation, *_ = source
    result = subprocess.run([rscript, str(LOADER), *[str(directories[mode]) for mode in ("tile", "cyto", "inner_square", "nuclei")],
        str(annotation), str(destination), modes, str(top)], capture_output=True, text=True, timeout=90)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def readback(rscript, destination):
    rrun(rscript, r'''
      a<-commandArgs(TRUE);load(file.path(a[1],"rawdata.RData"))
      result<-list(ann_ids=ann$label,common_ids=common_ids,xy=unname(xy),provenance=embedding_input_provenance,modes=list())
      for(mode in names(embeddings_raw)){
        value<-embeddings_raw[[mode]];if(is.null(value))next
        writeBin(as.double(t(value)),file.path(a[1],paste0(mode,".readback.bin")),size=8,endian="little")
        result$modes[[mode]]<-list(ids=rownames(value),features=colnames(value),shape=dim(value))
      }
      jsonlite::write_json(result,file.path(a[1],"readback.json"),auto_unbox=FALSE,digits=NA,null="null")
    ''', destination)
    metadata = json.loads((destination / "readback.json").read_text())
    arrays = {mode: np.fromfile(destination / f"{mode}.readback.bin", dtype="<f8").reshape(value["shape"])
              for mode, value in metadata["modes"].items()}
    return metadata, arrays


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_actual_python_written_shards_preserve_precision_ids_modes_and_explicit_order(tmp_path, rscript, dtype):
    source = fixture(tmp_path, dtype, ("tile", "inner_square"))
    output = tmp_path / "loaded"
    load(rscript, source, output, "tile,inner_square")
    metadata, arrays = readback(rscript, output)
    ids, values = source[-2:]
    order = np.argsort(ids)
    assert metadata["common_ids"] == sorted(ids) == metadata["ann_ids"]
    assert metadata["modes"]["tile"]["ids"] == sorted(ids)
    np.testing.assert_array_equal(arrays["tile"], values[order].astype(np.float64))
    np.testing.assert_array_equal(np.signbit(arrays["tile"]), np.signbit(values[order]))
    np.testing.assert_array_equal(arrays["inner_square"], (values + 1)[order].astype(np.float64))
    assert metadata["provenance"]["source_to_analysis"]["tile"]["source_observation_ids"] == ids
    assert metadata["provenance"]["source_to_analysis"]["tile"]["analysis_row_from_source"] == (order + 1).tolist()
    provenance = json.loads((output / "embedding_input_provenance.json").read_text())
    assert provenance["modes"]["tile"]["encoder_binding_status"] == "legacy_unverified_no_encoder_receipt"
    assert provenance["modes"]["tile"]["shards"][0]["manifest"]["dtype"] == np.dtype(dtype).str
    assert provenance["annotation_sha256"] == hashlib.sha256(source[1].read_bytes()).hexdigest()


def test_missing_encoder_and_mode_receipts_do_not_invent_provenance(tmp_path, rscript):
    source = fixture(tmp_path, declare_mode=False)
    load(rscript, source, tmp_path / "output")
    provenance = json.loads((tmp_path / "output/embedding_input_provenance.json").read_text())
    assert provenance["modes"]["tile"]["shards"][0]["mode_binding"] == "caller_mode_unverified_in_legacy_conversion"
    assert provenance["modes"]["tile"]["encoder_binding_status"] == "legacy_unverified_no_encoder_receipt"


def production_receipt_fixture(tmp_path, monkeypatch, *, paired=False, mode="tile"):
    from test_uni2_binary_storage import main_environment
    functions, _ = main_environment(tmp_path, monkeypatch, "binary", "token_subset" if paired else None, mode)
    functions["main"]()
    return tmp_path / "primary"


def refresh_root_inventory(root):
    complete = root / ".uni2_embedding_complete.json"
    record = json.loads(complete.read_text())
    for payload in record["payload_inventory"]:
        path = root / payload["path"]
        payload.update(sha256=sha256_file(path), size_bytes=path.stat().st_size)
    complete.write_text(json.dumps(record))


def receipt_read(rscript, source, output, mode="tile", check=True):
    return rrun(rscript, '''a<-commandArgs(TRUE);source(a[1]);
      result<-load_uni2_binary_mode(a[2],a[3],100)
      jsonlite::write_json(result$provenance,a[4],auto_unbox=TRUE,null="null")''',
      ROOT / "bin/uni2_embedding_io.R", source, mode, output, check=check)


@pytest.mark.parametrize("mode", ["tile", "nuclei", "cyto", "inner_square"])
def test_actual_no_model_producer_receipts_are_verified_by_r(tmp_path, monkeypatch, rscript, mode):
    root = production_receipt_fixture(tmp_path, monkeypatch, mode=mode)
    output = tmp_path / "receipt_read.json"
    before = {path: sha256_file(path) for path in root.rglob("*") if path.is_file()}
    receipt_read(rscript, root, output, mode)
    record = json.loads(output.read_text())
    assert record["encoder_binding_status"] == "receipt_consistency_verified_encoder_inputs_not_independently_verified"
    assert record["extraction_receipt_validation"]["storage_receipts"] == "verified_exact_root_grid_payload_inventory_hashes_sizes_modes_and_row_coverage"
    assert record["extraction_receipt_validation"]["cache_contract_digest"] == "declared_sha256_agrees_across_receipts_not_recomputed_from_python_canonical_json"
    assert len(record["encoder_receipts"]) == 3  # one root plus two actual extraction grids
    assert before == {path: sha256_file(path) for path in before}


def test_actual_paired_receipts_survive_full_r_loader_handoff(tmp_path, monkeypatch, rscript):
    root = production_receipt_fixture(tmp_path, monkeypatch, paired=True)
    directories = {"tile": root, "inner_square": tmp_path / "secondary",
                   "cyto": tmp_path / "cyto", "nuclei": tmp_path / "nuclei"}
    for mode in ("cyto", "nuclei"):
        directories[mode].mkdir()
    load(rscript, (directories, tmp_path / "objects.csv"), tmp_path / "loaded", "tile,inner_square")
    record = json.loads((tmp_path / "loaded/embedding_input_provenance.json").read_text())
    for mode in ("tile", "inner_square"):
        assert record["modes"][mode]["encoder_binding_status"].startswith("receipt_consistency_verified")
        assert len(record["modes"][mode]["encoder_receipts"]) == 3
    # A legitimate Nextflow staged directory symlink is not a foreign payload.
    staged = tmp_path / "staged"
    staged.symlink_to(root, target_is_directory=True)
    receipt_read(rscript, staged, tmp_path / "staged_read.json")


def test_receipt_backed_rows_require_observation_type_even_when_all_hashes_are_refreshed(tmp_path, monkeypatch, rscript):
    from test_uni2_binary_storage import producer_functions
    root = production_receipt_fixture(tmp_path, monkeypatch)
    shard = discover_embedding_shards(root)[0]
    manifest = json.loads(shard.read_text())
    rows_path = binary_shard_paths(shard)[1]
    rows = pd.read_csv(rows_path, dtype={"cell_id": str}).drop(columns="observation_type")
    rows.to_csv(rows_path, index=False)
    manifest.update(rows_sha256=sha256_file(rows_path), rows_size_bytes=rows_path.stat().st_size,
                    row_columns=list(rows))
    shard.write_text(json.dumps(manifest))
    grid = shard.parent / ".uni2_grid_complete.json"
    marker = json.loads(grid.read_text())
    marker["shard_files"] = producer_functions()["embedding_shard_inventory"](shard.parent, "binary")
    grid.write_text(json.dumps(marker)); refresh_root_inventory(root)
    result = receipt_read(rscript, root, tmp_path / "bad.json", check=False)
    assert result.returncode != 0 and "observation type differs" in result.stderr


@pytest.mark.parametrize("corruption", ["rehashed_features", "rehashed_rows", "root_omission", "root_size",
    "missing_grid", "missing_root", "missing_mode", "grid_mode", "root_mode", "contract_mode",
    "grid_contract", "grid_payload", "grid_payload_size", "grid_rows", "overlapping_intervals",
    "root_rows", "root_grids", "extra_grid", "root_path_escape", "duplicate_payload"])
def test_r_rejects_contradictory_existing_extraction_receipts(tmp_path, monkeypatch, rscript, corruption):
    root = production_receipt_fixture(tmp_path, monkeypatch)
    complete = root / ".uni2_embedding_complete.json"
    grids = sorted(root.rglob(".uni2_grid_complete.json"))
    grid = grids[0]
    shard = discover_embedding_shards(root)[0]
    if corruption in {"rehashed_features", "rehashed_rows"}:
        manifest = json.loads(shard.read_text())
        kind = "features" if corruption == "rehashed_features" else "rows"
        path = shard.parent / manifest[f"{kind}_file"]
        if kind == "features":
            value = np.fromfile(path, dtype=manifest["dtype"]); value[0] += 1; value.tofile(path)
        else:
            rows = pd.read_csv(path, dtype={"cell_id": str}); rows.cell_id = "changed"; rows.to_csv(path, index=False)
        manifest[f"{kind}_sha256"] = sha256_file(path)
        manifest[f"{kind}_size_bytes"] = path.stat().st_size
        shard.write_text(json.dumps(manifest))  # self-consistent shard, original root/grid remain binding
    elif corruption == "missing_grid":
        grid.unlink()
    elif corruption == "missing_root":
        complete.unlink()
    elif corruption == "missing_mode":
        modify_manifest(shard, lambda value: value.pop("embedding_mode"))
        marker = json.loads(grid.read_text())
        marker["shard_files"][0].update(sha256=sha256_file(shard), size_bytes=shard.stat().st_size)
        grid.write_text(json.dumps(marker)); refresh_root_inventory(root)
    elif corruption in {"grid_mode", "grid_contract", "grid_payload", "grid_payload_size", "grid_rows", "overlapping_intervals"}:
        marker = json.loads(grid.read_text())
        if corruption == "grid_mode": marker["embedding_mode"] = "nuclei"
        elif corruption == "grid_contract": marker["cache_contract_sha256"] = "f" * 64
        elif corruption == "grid_payload": marker["shard_files"][0]["payload_files"].pop()
        elif corruption == "grid_payload_size": marker["shard_files"][0]["payload_files"][0]["size_bytes"] += 1
        elif corruption == "grid_rows": marker["rows_written"] += 1
        else:
            marker["index_start"] += 1; marker["index_end"] += 1
        grid.write_text(json.dumps(marker)); refresh_root_inventory(root)
    elif corruption == "extra_grid":
        (root / "unlisted").mkdir()
        (root / "unlisted/.uni2_grid_complete.json").write_bytes(grid.read_bytes())
    else:
        record = json.loads(complete.read_text())
        if corruption == "root_omission": record["payload_inventory"].pop()
        elif corruption == "root_size": record["payload_inventory"][0]["size_bytes"] += 1
        elif corruption == "root_mode": record["embedding_mode"] = "nuclei"
        elif corruption == "contract_mode": record["cache_contract"]["parameters"]["embedding_mode"] = "nuclei"
        elif corruption == "root_rows": record["expected_observations"] += 1
        elif corruption == "root_grids": record["completed_grids"] += 1
        elif corruption == "root_path_escape": record["payload_inventory"][0]["path"] = "../outside"
        elif corruption == "duplicate_payload": record["payload_inventory"].append(record["payload_inventory"][0])
        complete.write_text(json.dumps(record))
    output = tmp_path / "bad.json"
    result = receipt_read(rscript, root, output, check=False)
    assert result.returncode != 0 and not output.exists(), result.stdout + result.stderr
    assert "Binary UNI2" in result.stderr or "binary UNI2" in result.stderr


def test_all_four_selected_modes_retain_their_exact_rows_and_features(tmp_path, rscript):
    modes = ("tile", "nuclei", "cyto", "inner_square")
    source = fixture(tmp_path, modes=modes)
    output = tmp_path / "full_stack"
    load(rscript, source, output, "full")
    metadata, arrays = readback(rscript, output)
    order = np.argsort(source[-2])
    assert set(arrays) == set(modes)
    for index, mode in enumerate(modes):
        np.testing.assert_array_equal(arrays[mode], (source[-1] + index)[order])
        assert metadata["modes"][mode]["ids"] == sorted(source[-2])


def test_binary_and_legacy_selected_modes_cannot_be_mixed(tmp_path, rscript):
    source = fixture(tmp_path)
    pd.DataFrame({"cell_id": source[-2], "feat_1": [1, 2, 3, 4]}).to_csv(
        source[0]["inner_square"] / "old_embeddings_shard0000.csv", index=False)
    result = load(rscript, source, tmp_path / "mixed_modes", "tile,inner_square", check=False)
    assert result.returncode != 0 and "across selected modes" in result.stderr


def test_binary_variance_preselection_and_legacy_analysis_order_match(tmp_path, rscript):
    values = np.array([[1, 1, 1, 1], [2, 30, 4, 8], [3, 50, 6, 14], [4, 80, 9, 22]], dtype=np.float64)
    ids = ["2", "1", "20", "10"]
    source = fixture(tmp_path, ids=ids, values=values)
    load(rscript, source, tmp_path / "binary", top=2)
    binary, bvalues = readback(rscript, tmp_path / "binary")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    legacy_source = fixture(legacy, ids=ids, values=values)
    for manifest in legacy_source[2]["tile"]:
        record = json.loads(manifest.read_text())
        rows = pd.read_csv(manifest.parent / record["rows_file"], dtype={"cell_id": str})
        matrix = np.fromfile(manifest.parent / record["features_file"], dtype="<f8").reshape(record["shape"])
        table = pd.concat([rows, pd.DataFrame(matrix, columns=record["feature_names"])], axis=1)
        table.to_csv(manifest.with_name(manifest.name.replace(".embedding.json", ".csv")), index=False)
        for path in (manifest, manifest.parent / record["rows_file"], manifest.parent / record["features_file"]):
            path.unlink()
    load(rscript, legacy_source, tmp_path / "csv", top=2)
    csv, cvalues = readback(rscript, tmp_path / "csv")
    assert binary["common_ids"] == csv["common_ids"] == sorted(ids)
    assert binary["modes"]["tile"]["features"] == csv["modes"]["tile"]["features"] == ["feat_2", "feat_4"]
    csv_analysis_order = [csv["modes"]["tile"]["ids"].index(value) for value in csv["common_ids"]]
    np.testing.assert_array_equal(bvalues["tile"], cvalues["tile"][csv_analysis_order])


def modify_manifest(path, mutation):
    record = json.loads(path.read_text())
    mutation(record)
    path.write_text(json.dumps(record))


@pytest.mark.parametrize("mutation", [lambda m: m.update(dtype=">f4"), lambda m: m.update(order="F"),
    lambda m: m.update(shape=[1, 4]), lambda m: m.update(features_sha256="0" * 64),
    lambda m: m.update(features_file="../outside.features.bin"), lambda m: m.update(schema_version="2.0.0"),
    lambda m: m.update(feature_names=["duplicate"] * 4), lambda m: m.update(embedding_mode="nuclei")])
def test_malformed_storage_receipts_fail_closed(tmp_path, rscript, mutation):
    source = fixture(tmp_path)
    modify_manifest(source[2]["tile"][0], mutation)
    result = load(rscript, source, tmp_path / "bad", check=False)
    assert result.returncode != 0 and not (tmp_path / "bad/rawdata.RData").exists()


@pytest.mark.parametrize("corruption", ["truncated", "trailing", "nan", "duplicate_rows", "foreign_id", "duplicate_annotation", "foreign_annotation", "coordinate"])
def test_payload_and_annotation_corruption_is_rejected_without_dropping_rows(tmp_path, rscript, corruption):
    source = fixture(tmp_path)
    manifest = source[2]["tile"][0]
    record = json.loads(manifest.read_text())
    features = manifest.parent / record["features_file"]
    rows_file = manifest.parent / record["rows_file"]
    if corruption in {"truncated", "trailing", "nan"}:
        content = features.read_bytes()
        if corruption == "truncated":
            content = content[:-1]
        elif corruption == "trailing":
            content += b"X"
        else:
            content = np.array([np.nan], dtype="<f8").tobytes() + content[8:]
        features.write_bytes(content)
        record["features_sha256"] = hashlib.sha256(content).hexdigest()
        record["features_size_bytes"] = len(content)
    elif corruption in {"duplicate_rows", "foreign_id", "coordinate"}:
        rows = pd.read_csv(rows_file, dtype={"cell_id": str})
        if corruption == "duplicate_rows":
            rows.loc[1, "cell_id"] = rows.loc[0, "cell_id"]
        elif corruption == "foreign_id":
            rows.loc[0, "cell_id"] = "unrelated"
        else:
            rows.loc[0, "x"] += 10
        rows.to_csv(rows_file, index=False)
        record["rows_sha256"] = hashlib.sha256(rows_file.read_bytes()).hexdigest()
        record["rows_size_bytes"] = rows_file.stat().st_size
    else:
        rows = pd.read_csv(source[1], dtype={"label": str})
        rows.loc[0, "label"] = rows.loc[1, "label"] if corruption == "duplicate_annotation" else "foreign"
        rows.to_csv(source[1], index=False)
    manifest.write_text(json.dumps(record))
    result = load(rscript, source, tmp_path / "bad", check=False)
    assert result.returncode != 0 and not (tmp_path / "bad/rawdata.RData").exists()


def test_mixed_formats_and_orphan_payloads_are_rejected(tmp_path, rscript):
    source = fixture(tmp_path)
    extra = source[0]["tile"] / "old_embeddings_shard0999.csv"
    extra.write_text("cell_id,feat_1\n1,1\n")
    assert load(rscript, source, tmp_path / "mixed", check=False).returncode != 0
    extra.unlink()
    extra = source[0]["tile"] / "orphan.features.bin"
    extra.write_bytes(b"1234")
    assert load(rscript, source, tmp_path / "orphan", check=False).returncode != 0


def test_cross_shard_duplicates_are_not_silently_dropped(tmp_path, rscript):
    source = fixture(tmp_path)
    first, second = source[2]["tile"]
    record = json.loads(second.read_text())
    path = second.parent / record["rows_file"]
    rows = pd.read_csv(path, dtype={"cell_id": str})
    rows.loc[0, "cell_id"] = source[-2][0]
    rows.to_csv(path, index=False)
    record.update(rows_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows_size_bytes=path.stat().st_size)
    second.write_text(json.dumps(record))
    result = load(rscript, source, tmp_path / "duplicates", check=False)
    assert result.returncode != 0 and "cross-shard" in result.stderr


def test_duplicate_json_keys_rejected(tmp_path, rscript):
    source = fixture(tmp_path)
    path = source[2]["tile"][0]
    content = path.read_text().strip()
    path.write_text(content[:-1] + ',"dtype":"<f4"}')
    result = load(rscript, source, tmp_path / "duplicate_json", check=False)
    assert result.returncode != 0 and "Duplicate binary UNI2 manifest key" in result.stderr


@pytest.fixture(scope="module")
def native_library(rscript):
    candidates = [os.environ.get("KODAMA_NATIVE_R_LIBRARY", ""),
                  str(Path.home() / "Documents/KODAMA-cpp 2/tmp/Rlib-kodama-latest")]
    result = rrun(rscript, r'''
      for(lib in unique(c(commandArgs(TRUE),.libPaths()))){
        ns<-file.path(lib,"KODAMA","NAMESPACE")
        if(nzchar(lib)&&file.exists(ns)&&any(grepl("export(KODAMA.graph.materialize)",readLines(ns),fixed=TRUE))){cat(lib);quit(status=0)}
      };quit(status=77)
    ''', *candidates, check=False)
    if result.returncode == 77:
        pytest.skip("Existing native KODAMA runtime unavailable")
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_actual_binary_rawdata_to_native_cpu_kodama_retains_storage_receipts(tmp_path, rscript, native_library):
    ids = [f"{i:03}" for i in range(40, 0, -1)]
    values = np.random.default_rng(2003).normal(size=(40, 6)).astype(np.float32)
    source = fixture(tmp_path, np.float32, ids=ids, values=values)
    raw = tmp_path / "raw"
    load(rscript, source, raw)
    output = tmp_path / "native"
    command = [rscript, str(ANALYSIS), str(raw / "rawdata.RData"), str(output),
        "--embedding-mode", "tile", "--dims-to-run", "3", "--spark-top", "6", "--landmarks", "20",
        "--kodama-ncomp", "50", "--n-cores", "1", "--backend", "cpu", "--export-native-graph", "false"]
    result = subprocess.run(command, env={**os.environ, "R_LIBS": native_library}, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((output / "clustering_representations.json").read_text())
    provenance = metadata["embedding_input_provenance"]
    assert provenance["analysis_observation_ids"] == sorted(ids)
    assert metadata["rawdata_input_sha256"] == hashlib.sha256((raw / "rawdata.RData").read_bytes()).hexdigest()
    assert metadata["requested_kodama_ncomp"] == 50 and metadata["effective_kodama_ncomp"] == 3
    assert provenance["modes"]["tile"]["encoder_binding_status"] == "legacy_unverified_no_encoder_receipt"
    rrun(rscript, r'''
      a<-commandArgs(TRUE);p<-new.env();k<-new.env()
      load(file.path(a[1],"pca_full_3.RData"),envir=p);load(file.path(a[1],"kodama_full_3.RData"),envir=k)
      stopifnot(identical(rownames(p$pca),p$embedding_input_provenance$analysis_observation_ids),
        identical(rownames(k$vis),p$embedding_input_provenance$analysis_observation_ids),
        identical(p$pca_model_metadata$embedding_input_provenance,k$representation_metadata$embedding_input_provenance))
    ''', output)
    # Tampered duplicate/order changes cannot enter the legacy sanitization path.
    rrun(rscript, 'a<-commandArgs(TRUE);load(a[1]);embeddings_raw$tile<-embeddings_raw$tile[40:1,,drop=FALSE];save(ann,xy,embeddings_raw,embedding_input_provenance,file=a[1])', raw / "rawdata.RData")
    command[3] = str(tmp_path / "reordered")
    result = subprocess.run(command, env={**os.environ, "R_LIBS": native_library}, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "matrix IDs/order" in result.stderr
