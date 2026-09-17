"""Executable portable native-graph contracts; synthetic data, no image/model."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "bin/kodama_graph_export.R"
ANALYSIS = ROOT / "bin/run_kodama_analysis.R"
RSCRIPT = shutil.which("Rscript")


@pytest.fixture(scope="module")
def rscript():
    if not RSCRIPT:
        pytest.skip("Rscript unavailable")
    check = subprocess.run([RSCRIPT, "-e",
        'stopifnot(all(vapply(c("Matrix","digest","jsonlite"), requireNamespace, logical(1), quietly=TRUE)))'],
        text=True, capture_output=True)
    assert check.returncode == 0, check.stderr
    return RSCRIPT


def run_r(rscript, code, *args, check=True):
    result = subprocess.run([rscript, "-e", f"source({json.dumps(str(HELPER))});\n" + code, *map(str, args)],
        text=True, capture_output=True, timeout=60)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("value", [[], ["maybe"]])
def test_export_flag_fails_closed_on_missing_or_invalid_value(rscript, value):
    result = subprocess.run([rscript, str(ANALYSIS), "unused_input", "unused_output",
        "--export-native-graph", *value], text=True, capture_output=True)
    assert result.returncode != 0 and "--export-native-graph" in result.stderr


FAKE = r'''
a <- commandArgs(TRUE)
ids <- c("003", "cell_B", "cell_A", "isolate")
pca <- matrix(seq_len(12),nrow=4,dimnames=list(ids,NULL))
pca_file <- file.path(a[1],"pca_full_3.RData")
save(pca,file=pca_file)
fit <- list(knn_is_kodama_corrected=TRUE, res=matrix(1L,2,4),
  parameters=list(classifier="knn",ncomp=3L,landmarks=2L),
  knn=list(indices=matrix(c(2L,3L,1L,4L,4L,4L,2L,4L),4,2),
    distances=matrix(c(0,2,3,Inf,Inf,Inf,4,Inf),4,2)))
materialize <- function(fit) fit$knn
export_graph <- function(...) export_portable_kodama_graph(fit,ids,a[1],pca_file,
  package_version="synthetic",package_revision="synthetic",materialize=materialize,...)
'''


def test_portable_graph_preserves_directed_zero_edges_ids_and_isolates(tmp_path, rscript):
    run_r(rscript, FAKE + r'''
      receipt<-export_graph()
      g<-load_portable_kodama_graph(a[1],ids,receipt$file_sha256)
      stopifnot(identical(g$observation_ids,ids), identical(dim(g$distance_graph),c(4L,4L)),
        length(g$distance_graph@x)==4L, sum(g$distance_graph@x==0)==1L,
        g$distance_graph[1,2]==0, g$distance_graph[2,3]==2,
        !any(g$distance_graph@i==3L), !diff(g$distance_graph@p)[4])
      stopifnot(receipt$omitted_infinite_edges==4L,
        receipt$stored_zero_distance_edges==1L, receipt$native_neighbors==2L)
      tryCatch({export_graph();stop("overwrite passed")},error=function(e)
        stopifnot(grepl("overwrite",conditionMessage(e))))
    ''', tmp_path)
    metadata = json.loads((tmp_path / "kodama_graph.json").read_text())
    assert metadata["all_observations"] and not metadata["projected"]
    assert metadata["native_corrected"] and metadata["directed"]
    assert metadata["stored_edges"] == 4
    assert len(metadata["file_sha256"]) == 64
    assert metadata["ncomp_applicability"].startswith("not_PLS_rank")


@pytest.mark.parametrize("mutation,expected", [
    ("fit$knn_is_kodama_corrected<-FALSE", "not confirmed corrected"),
    ("fit$knn$indices[1,1]<-0L", "one-based"),
    ("fit$knn$indices[1,1]<-1.5", "one-based"),
    ("fit$knn$distances[1,1]<-NaN", "nonnegative"),
    ("fit$knn$distances[1,1]<- -1", "nonnegative"),
    ("fit$knn$indices[3,2]<-1L", "duplicate directed"),
    ("fit$res<-matrix(1L,2,3)", "observation count"),
    ("ids[4]<-ids[1]", "unique nonempty"),
])
def test_bad_native_payload_fails_before_output(tmp_path, rscript, mutation, expected):
    result = run_r(rscript, FAKE + "\n" + mutation + "\nexport_graph()", tmp_path, check=False)
    assert result.returncode != 0 and expected in result.stderr
    assert not (tmp_path / "kodama_graph.rds").exists()


@pytest.mark.parametrize("options", [
    "projected=TRUE", "expected_observation_ids=rev(ids)",
    'expected_observation_ids=c(ids,"missing_observation")',
])
def test_subset_projection_or_reordered_identity_cannot_be_claimed_complete(tmp_path, rscript, options):
    result = run_r(rscript, FAKE + f"\nexport_graph({options})", tmp_path, check=False)
    assert result.returncode != 0 and "projected or subset" in result.stderr
    assert not (tmp_path / "kodama_graph.rds").exists()


@pytest.mark.parametrize("mutation,expected", [
    ('ids<-rev(ids)', "IDs/order"),
    ('writeLines("changed",pca_file)', "PCA checksum"),
    ('r<-readRDS(file.path(a[1],"kodama_graph.rds"));r$distance_graph@x[1]<-99;saveRDS(r,file.path(a[1],"kodama_graph.rds"))', "SHA256"),
    ('m<-jsonlite::read_json(file.path(a[1],"kodama_graph.json"));m$native_neighbors<-999;jsonlite::write_json(m,file.path(a[1],"kodama_graph.json"),auto_unbox=TRUE)', "metadata receipt"),
])
def test_portable_receipt_rejects_changed_graph_source_or_ids(tmp_path, rscript, mutation, expected):
    result = run_r(rscript, FAKE + "\nreceipt<-export_graph()\n" + mutation +
        "\nload_portable_kodama_graph(a[1],ids,receipt$file_sha256)", tmp_path, check=False)
    assert result.returncode != 0 and expected in result.stderr


@pytest.fixture(scope="module")
def native_library(rscript):
    candidates = [os.environ.get("KODAMA_NATIVE_R_LIBRARY", ""),
        str(Path.home() / "Documents/KODAMA-cpp 2/tmp/Rlib-kodama-latest")]
    result = subprocess.run([rscript, "-e", r'''
      candidates<-c(commandArgs(TRUE),.libPaths())
      for(lib in unique(candidates[nzchar(candidates)])) {
        file<-file.path(lib,"KODAMA","NAMESPACE")
        if(file.exists(file)&&any(grepl("export(KODAMA.graph.materialize)",readLines(file),fixed=TRUE))) {
          cat(lib);quit(status=0)
        }
      }
      quit(status=77)
    ''', *candidates], text=True, capture_output=True)
    if result.returncode == 77:
        pytest.skip("Installed native KODAMA graph API unavailable")
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_actual_native_cpu_handle_export_exact_roundtrip(tmp_path, rscript, native_library):
    run_r(rscript, r'''
      a<-commandArgs(TRUE);.libPaths(c(a[2],.libPaths()));library(KODAMA)
      set.seed(17);pca<-matrix(rnorm(40*6),40,6);ids<-sprintf("synthetic_%02d",1:40)
      rownames(pca)<-ids;pca_file<-file.path(a[1],"pca_full_6.RData");save(pca,file=pca_file)
      fit<-KODAMA.matrix(pca,M=2,Tcycle=2,ncomp=2,landmarks=20,splitting=4,
        graph.neighbors=8,knn.k=3,n.cores=1,backend="cpu",seed=17,
        visual.init=FALSE,progress=FALSE,return.graph="handle")
      native<-KODAMA.graph.materialize(fit)
      receipt<-export_portable_kodama_graph(fit,ids,a[1],pca_file)
      restored<-load_portable_kodama_graph(a[1],ids,receipt$file_sha256)
      for(i in seq_along(ids)) for(k in seq_len(ncol(native$indices))) {
        j<-native$indices[i,k];d<-native$distances[i,k]
        if(is.finite(d)&&i!=j) stopifnot(identical(as.numeric(restored$distance_graph[i,j]),d))
      }
      stopifnot(length(restored$distance_graph@x)==sum(is.finite(native$distances)),
        fit$graph_builds==1L,receipt$native_corrected,
        ncol(fit$res)==length(ids),length(ids)>fit$parameters$landmarks)
    ''', tmp_path, native_library)


def test_actual_native_producer_cli_opt_in_and_default(tmp_path, rscript, native_library):
    run_r(rscript, r'''
      a<-commandArgs(TRUE);set.seed(18);ids<-sprintf("cell_%03d",1:40)
      ann<-data.frame(label=ids,x=rep(1:8,5)*10,y=rep(1:5,each=8)*10,
        polygon_label="unknown",row.names=ids);xy<-as.matrix(ann[,c("x","y")])
      values<-matrix(rnorm(40*6),40,6,dimnames=list(ids,paste0("f",1:6)))
      embeddings_raw<-list(tile=values)
      save(ann,xy,embeddings_raw,file=file.path(a[1],"rawdata.RData"))
    ''', tmp_path)
    for enabled in (False, True):
        output = tmp_path / ("graph" if enabled else "default")
        result = subprocess.run([rscript, str(ANALYSIS), str(tmp_path / "rawdata.RData"), str(output),
            "--embedding-mode", "tile", "--dims-to-run", "3", "--spark-top", "6",
            "--landmarks", "20", "--kodama-ncomp", "50", "--n-cores", "1", "--backend", "cpu",
            "--export-native-graph", str(enabled).lower()],
            env={**os.environ, "R_LIBS": native_library},
            text=True, capture_output=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        meta = json.loads((output / "clustering_representations.json").read_text())
        assert meta["kodama_graph_available"] is enabled
        assert (output / "kodama_graph.rds").exists() is enabled
        assert meta["requested_kodama_ncomp"] == 50 and meta["effective_kodama_ncomp"] == 3
        assert meta["actual_kodama_classifier"] == "knn"
        if enabled:
            run_r(rscript, 'a<-commandArgs(TRUE);e<-new.env();load(file.path(a[1],"pca_full_3.RData"),envir=e);g<-load_portable_kodama_graph(a[1],e$common_ids);stopifnot(nrow(g$distance_graph)==40L)', output)
