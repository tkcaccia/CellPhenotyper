"""Tiny real native-R hierarchy runs; synthetic features, no learned models."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from test_kodama_graph_export import native_library, rscript  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "bin/run_hierarchy_kodama.R"
HELPERS = [ROOT / "bin/kodama_graph_export.R", ROOT / "bin/kodama_graph_clustering.R"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(root, manifest):
    path = root / "input_manifest.json"
    path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    return path


def make_input(root, *, constant=False):
    root.mkdir()
    n = 30
    ids = ["001", "NA", "cell C", "a,b", 'cell"quoted'] + [f"cell_{i:02d}" for i in range(5, n)]
    xy = np.column_stack((np.arange(n) * 0.37 + 13.25, np.arange(n)[::-1] * 0.91 - 2.75))
    rows = root / "rows.csv"
    with rows.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "x", "y"])
        writer.writerows((label, format(x, ".17g"), format(y, ".17g")) for label, (x, y) in zip(ids, xy))
    rng = np.random.default_rng(58)
    local = rng.normal(size=(n, 4)) + np.repeat([-4.0, 4.0], n // 2)[:, None]
    context = rng.normal(size=(n, 4)) + np.repeat([-2.0, 2.0], n // 2)[:, None]
    if constant:
        local[:] = [0.0, 2.0, -3.5, 8.0]
        context[:] = [0.5, 4.0, 6.0, 1.0]
    matrices = {"local": local, "context": context, "combined": np.column_stack((local, context))}
    representations = {}
    for name, value in matrices.items():
        path = root / f"{name}.f64"
        path.write_bytes(value.astype("<f8").tobytes(order="C"))
        representations[name] = {"path": path.name, "sha256": sha(path), "shape": list(value.shape), "constant": constant}
    manifest = {
        "format": "cellphenotyper_hierarchy_kodama_input", "schema_version": "1.0.0",
        "rows": {"path": "rows.csv", "sha256": sha(rows)}, "representations": representations,
        "parameters": {"ncomp": 50, "M": 3, "Tcycle": 2, "landmarks": 20, "cores": 1,
                       "neighbors": 8, "seed": 17, "cluster_seeds": [17, 29], "resolutions": [0.1, 0.5, 1.5]},
        "scope": {"parent": "synthetic_parent", "observations": "all fixed-parent rows"},
        "provenance": {"features": "synthetic Gaussian axes; learned-model-free", "nested": {"null": None, "flag": True}},
    }
    return write_manifest(root, manifest), manifest, ids, xy, matrices


def run_cli(rscript, native_library, path, output, *, expected=None, runner=RUNNER, env=None):
    return subprocess.run([rscript, str(runner), str(path), str(output), expected or sha(path), str(native_library)],
                          text=True, capture_output=True, timeout=90,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})})


def require_pass(result):
    assert result.returncode == 0, result.stdout + result.stderr


def require_fail(result, output, message=None):
    assert result.returncode != 0, result.stdout + result.stderr
    if message:
        assert message in result.stderr, result.stdout + result.stderr
    assert not (output / "runner_summary.json").exists()


def test_real_native_all_rows_three_representations_and_pinned_receipts(tmp_path, rscript, native_library):
    path, manifest, ids, xy, matrices = make_input(tmp_path / "inputs")
    before = {str(p): sha(p) for p in [*path.parent.iterdir(), RUNNER, *HELPERS]}
    output = tmp_path / "output"
    require_pass(run_cli(rscript, native_library, path, output))
    summary = json.loads((output / "runner_summary.json").read_text())
    frame = pd.read_csv(output / "partitions.csv", dtype={"label": str}, keep_default_na=False)
    assert list(frame) == ["representation", "resolution", "seed", "label", "cluster", "degree", "affinity_margin", "own_affinity_fraction"]
    assert len(frame) == 30 * 3 * 3 * 2
    assert summary["status"] == "pass"
    assert summary["partition_run_count"] == 18 and summary["partition_rows"] == len(frame)
    assert summary["feature_protocol"] == "raw_data_native_retained_graph"
    assert "parallel" in summary["repeatability_warning"]
    assert summary["scope"] == manifest["scope"] and summary["provenance"] == manifest["provenance"]
    assert summary["sources_before"] == summary["sources_after"]
    assert summary["producer_before"] == summary["producer_after"]
    assert summary["runtime_before"] == summary["runtime_after"]
    assert summary["native_package_before_load"] == summary["native_package_after"]
    assert summary["runtime"]["packages"]["KODAMA"]["version"] == "0.99.7"
    assert any(key.endswith("KODAMA.so") for key in summary["runtime_before"])
    for _, group in frame.groupby(["representation", "resolution", "seed"], sort=False):
        assert group.label.tolist() == ids
        assert np.array_equal(group.cluster.to_numpy() == 0, group.degree.to_numpy() == 0)
        supported = group.degree > 0
        margin = pd.to_numeric(group.loc[supported, "affinity_margin"])
        fraction = pd.to_numeric(group.loc[supported, "own_affinity_fraction"])
        assert np.isfinite(margin).all() and margin.between(-1.0, 1.0).all()
        assert np.isfinite(fraction).all() and fraction.between(0.0, 1.0).all()
        assert (group.loc[~supported, ["affinity_margin", "own_affinity_fraction"]] == "NA").all().all()
    for name, values in matrices.items():
        record = summary["representations"][name]
        assert record["status"] == "fit" and record["dimensions"] == values.shape[1]
        assert record["effective_native_parameters"]["ncomp"] == 50
        assert record["effective_native_parameters"]["classifier"] == "knn"
        assert record["effective_native_parameters"]["backend"] == "cpu"
        assert record["effective_native_parameters"]["visual.init"] is False
        assert record["effective_native_parameters"]["return.graph"] == "handle"
        assert record["spatial_graph_builds"] == 0 and record["graph_builds"] == 1
        assert record["actual_cores"] == 1
        assert record["ncomp_applicability"].startswith("not_PLS_rank")
        assert record["graph_sha256"] == sha(output / name / "kodama_graph.rds")
    for relative, digest in summary["output_sha256"].items():
        assert sha(output / relative) == digest
    assert "runner_summary.json" not in summary["output_sha256"]
    assert {str(p): sha(p) for p in [*path.parent.iterdir(), RUNNER, *HELPERS]} == before
    # Independent actual R readback proves little-endian row-major values,
    # exact physical coordinates, literal IDs/order, and portable graph binding.
    code = r'''
      a<-commandArgs(TRUE);source(a[1]);m<-jsonlite::read_json(a[2]);
      rows<-read.csv(file.path(dirname(a[2]),"rows.csv"),colClasses="character",na.strings=character());
      for(name in c("local","context","combined")) {
        e<-new.env();load(file.path(a[3],name,"input_features.RData"),envir=e)
        con<-file(file.path(dirname(a[2]),paste0(name,".f64")),"rb")
        v<-readBin(con,"double",n=prod(unlist(m$representations[[name]]$shape)),size=8,endian="little");close(con)
        expected<-matrix(v,nrow=30,byrow=TRUE)
        stopifnot(identical(unname(e$pca),expected),identical(rownames(e$pca),rows$label),
          identical(rownames(e$xy),rows$label),identical(as.numeric(e$xy[,1]),as.numeric(rows$x)),
          identical(as.numeric(e$xy[,2]),as.numeric(rows$y)))
        g<-load_portable_kodama_graph(file.path(a[3],name),rows$label)
        stopifnot(g$metadata$all_observations,g$metadata$native_corrected,!g$metadata$projected)
      }
    '''
    result = subprocess.run([rscript, "-e", code, str(HELPERS[0]), str(path), str(output)],
                            text=True, capture_output=True, timeout=30)
    require_pass(result)


def test_constant_representations_skip_without_fabricated_assignments(tmp_path, rscript, native_library):
    path, _, _, _, _ = make_input(tmp_path / "inputs", constant=True)
    output = tmp_path / "output"
    require_pass(run_cli(rscript, native_library, path, output))
    summary = json.loads((output / "runner_summary.json").read_text())
    assert summary["partition_run_count"] == 0 and summary["partition_rows"] == 0
    assert pd.read_csv(output / "partitions.csv").empty
    assert all(r["status"] == "skipped_constant" for r in summary["representations"].values())
    assert not any((output / name).exists() for name in ("local", "context", "combined"))


def test_positive_nondefault_ncomp_is_passed_to_native_without_pca_conflation(tmp_path, rscript, native_library):
    path, manifest, _, _, _ = make_input(tmp_path / "inputs")
    manifest["parameters"]["ncomp"] = 7
    write_manifest(path.parent, manifest)
    output = tmp_path / "output"
    require_pass(run_cli(rscript, native_library, path, output))
    summary = json.loads((output / "runner_summary.json").read_text())
    assert summary["representations"]["local"]["effective_native_parameters"]["ncomp"] == 7
    assert summary["representations"]["local"]["dimensions"] == 4


@pytest.mark.parametrize("bad", [0, -1, 1.25, "50", None, True])
def test_invalid_ncomp_fails_closed(tmp_path, rscript, native_library, bad):
    path, manifest, _, _, _ = make_input(tmp_path / "inputs")
    manifest["parameters"]["ncomp"] = bad
    write_manifest(path.parent, manifest)
    output = tmp_path / "output"
    require_fail(run_cli(rscript, native_library, path, output), output, "ncomp")
    assert not output.exists()


@pytest.mark.parametrize("mutation,message", [
    ("manifest_hash", "manifest SHA256"), ("schema", "schema"), ("missing_rep", "representations"),
    ("duplicate_key", "unique nonempty keys"), ("payload_hash", "SHA256 differs"),
    ("short_bytes", "byte size"), ("extra_bytes", "byte size"), ("nonfinite", "finite float64"),
    ("shape", "row count"), ("shape_fraction", "shape D"), ("constant_false", "constant flag"),
    ("constant_true", "constant flag"), ("path_escape", "contained basename"),
    ("symlink_escape", "escapes manifest"), ("duplicate_path", "distinct filenames"),
    ("bad_resolutions", "resolutions"), ("duplicate_seeds", "cluster_seeds"),
])
def test_malformed_manifest_or_payload_fails_before_output(tmp_path, rscript, native_library, mutation, message):
    path, manifest, _, _, _ = make_input(tmp_path / "inputs", constant=mutation == "constant_false")
    rep = manifest["representations"]["local"]
    binary = path.parent / rep["path"]
    expected = None
    if mutation == "manifest_hash":
        expected = "0" * 64
    elif mutation == "schema":
        manifest["schema_version"] = "999"
    elif mutation == "missing_rep":
        del manifest["representations"]["combined"]
    elif mutation == "payload_hash":
        rep["sha256"] = "0" * 64
    elif mutation in {"short_bytes", "extra_bytes", "nonfinite"}:
        data = binary.read_bytes()
        binary.write_bytes(data[:-1] if mutation == "short_bytes" else data + b"x" if mutation == "extra_bytes"
                           else np.array([np.inf], dtype="<f8").tobytes() + data[8:])
        rep["sha256"] = sha(binary)
    elif mutation == "shape":
        rep["shape"][0] = 29
    elif mutation == "shape_fraction":
        rep["shape"][1] = 4.5
    elif mutation == "constant_false":
        rep["constant"] = False
    elif mutation == "constant_true":
        rep["constant"] = True
    elif mutation == "path_escape":
        rep["path"] = "../local.f64"
    elif mutation == "symlink_escape":
        external = tmp_path / "external.f64"
        external.write_bytes(binary.read_bytes())
        link = path.parent / "link.f64"
        link.symlink_to(external)
        rep["path"] = link.name
    elif mutation == "duplicate_path":
        manifest["representations"]["context"] = dict(rep)
    elif mutation == "bad_resolutions":
        manifest["parameters"]["resolutions"] = [0.0]
    elif mutation == "duplicate_seeds":
        manifest["parameters"]["cluster_seeds"] = [17, 17]
    write_manifest(path.parent, manifest)
    if mutation == "duplicate_key":
        path.write_text(path.read_text().replace('"schema_version": "1.0.0"', '"schema_version": "1.0.0", "schema_version": "1.0.0"'))
    output = tmp_path / "output"
    require_fail(run_cli(rscript, native_library, path, output, expected=expected), output, message)
    assert not output.exists()


@pytest.mark.parametrize("bad", ["", " ", " padded", "padded ", "a\tb", "a\nb", "duplicate"])
def test_invalid_literal_ids_rejected_with_exact_source_hash(tmp_path, rscript, native_library, bad):
    path, manifest, _, _, _ = make_input(tmp_path / "inputs")
    rows = path.parent / "rows.csv"
    with rows.open(newline="") as stream:
        values = list(csv.reader(stream))
    values[1][0] = values[2][0] if bad == "duplicate" else bad
    with rows.open("w", newline="") as stream:
        csv.writer(stream).writerows(values)
    manifest["rows"]["sha256"] = sha(rows)
    write_manifest(path.parent, manifest)
    output = tmp_path / "output"
    require_fail(run_cli(rscript, native_library, path, output), output, "Observation IDs")


@pytest.mark.parametrize("bad", ["NaN", "Inf", "not-a-number", ""])
def test_nonfinite_or_missing_coordinate_rejected(tmp_path, rscript, native_library, bad):
    path, manifest, _, _, _ = make_input(tmp_path / "inputs")
    rows = path.parent / "rows.csv"
    with rows.open(newline="") as stream:
        values = list(csv.reader(stream))
    values[1][1] = bad
    with rows.open("w", newline="") as stream:
        csv.writer(stream).writerows(values)
    manifest["rows"]["sha256"] = sha(rows)
    write_manifest(path.parent, manifest)
    output = tmp_path / "output"
    require_fail(run_cli(rscript, native_library, path, output), output, "finite numeric")


def test_preexisting_output_and_explicit_missing_native_library_rejected(tmp_path, rscript, native_library):
    path, _, _, _, _ = make_input(tmp_path / "inputs")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("do not overwrite")
    require_fail(run_cli(rscript, native_library, path, output), output, "already exists")
    assert marker.read_text() == "do not overwrite"
    empty_library = tmp_path / "empty_library"
    empty_library.mkdir()
    new_output = tmp_path / "new_output"
    require_fail(run_cli(rscript, empty_library, path, new_output), new_output, "Explicit native library lacks")


def test_isolates_are_zero_with_missing_evidence_not_nearest_assignments(tmp_path, rscript):
    code = r'''
      a<-commandArgs(TRUE);source(a[1]);source(a[2]);source(a[3]);
      ids<-c("001","NA","isolated");d<-Matrix::sparseMatrix(i=c(1L,2L),j=c(2L,1L),x=c(0,0),dims=c(3L,3L),dimnames=list(ids,ids));
      g<-prepare_kodama_affinity_graph(list(observation_ids=ids,distance_graph=d),ids)
      result<-hierarchy_partition(g,"local",.5,17L)
      stopifnot(identical(result$label,ids),identical(result$cluster,c(1L,1L,0L)),
        identical(result$degree,c(1L,1L,0L)),is.na(result$affinity_margin[3]),
        is.na(result$own_affinity_fraction[3]),all(result$own_affinity_fraction[1:2]==1))
    '''
    result = subprocess.run([rscript, "-e", code, str(RUNNER), *map(str, HELPERS)],
                            text=True, capture_output=True, timeout=30)
    require_pass(result)


@pytest.mark.parametrize("changed", ["input", "producer"])
def test_consumption_time_mutation_prevents_completion(tmp_path, rscript, native_library, changed):
    path, _, _, _, _ = make_input(tmp_path / "inputs")
    copied_bin = tmp_path / "copied_bin"
    copied_bin.mkdir()
    for source in [RUNNER, *HELPERS]:
        shutil.copyfile(source, copied_bin / source.name)
    copied_helper = copied_bin / "kodama_graph_clustering.R"
    # Controlled synthetic fault injection in an isolated producer copy. The
    # hook changes one already-bound input/code file AFTER native consumption.
    copied_helper.write_text(copied_helper.read_text() + r'''
      original_prepare <- prepare_kodama_affinity_graph
      prepare_kodama_affinity_graph <- function(...) {
        result <- original_prepare(...)
        target <- Sys.getenv("HIERARCHY_MUTATE_TARGET")
        cat("\n", file=target, append=TRUE)
        result
      }
    ''')
    target = path.parent / "local.f64" if changed == "input" else copied_helper
    output = tmp_path / "output"
    result = run_cli(rscript, native_library, path, output, runner=copied_bin / RUNNER.name,
                     env={"HIERARCHY_MUTATE_TARGET": str(target)})
    require_fail(result, output, "changed during consumption")
    assert (output / "local" / "kodama_graph.rds").exists()
