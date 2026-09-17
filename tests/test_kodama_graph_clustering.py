"""Actual sparse-R/CLI fixtures; no native KODAMA or biological inference."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "bin/kodama_graph_clustering.R"
EXPORT = ROOT / "bin/kodama_graph_export.R"
CLUSTER = ROOT / "bin/Rcode_Clustering.R"


@pytest.fixture(scope="module")
def rscript():
    executable = shutil.which("Rscript")
    if not executable:
        pytest.skip("Rscript unavailable")
    check = subprocess.run([executable, "-e", 'p<-c("Matrix","igraph","digest","jsonlite");if(!all(vapply(p,requireNamespace,logical(1),quietly=TRUE)))quit(status=77)'], capture_output=True, text=True)
    if check.returncode == 77:
        pytest.skip("Optional R graph packages unavailable")
    assert check.returncode == 0, check.stderr
    return executable


def run_r(rscript, code, *args, check=True):
    result = subprocess.run([rscript, "-e", f"source({json.dumps(str(HELPER))});" + code, *map(str, args)], capture_output=True, text=True, timeout=120)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def make_graph_fixture(rscript, directory, *, isolated=False):
    directory.mkdir()
    run_r(rscript, r'''
      a<-commandArgs(TRUE);source(a[2]);n<-as.integer(a[3]);ids<-sprintf("%03d",seq_len(n))
      vis<-cbind(seq_len(n),rev(seq_len(n)));rownames(vis)<-ids
      pca<-matrix(seq_len(n*6),nrow=n,dimnames=list(ids,paste0("PC",1:6)))
      xy<-vis;common_ids<-ids
      save(pca,xy,common_ids,file=file.path(a[1],"pca_full_6.RData"))
      index<-matrix(NA_integer_,n,5);distance<-matrix(Inf,n,5)
      for(i in seq_len(n)){
        peers<-if(i<=6) setdiff(1:6,i) else if(i<=12)setdiff(7:12,i) else rep(i,5)
        index[i,]<-peers
        if(i<=12)distance[i,]<-.2
      }
      distance[1,1]<-0
      fit<-list(knn_is_kodama_corrected=TRUE,res=matrix(0,1,n),parameters=list(classifier="knn",ncomp=50))
      receipt<-export_portable_kodama_graph(fit,ids,a[1],file.path(a[1],"pca_full_6.RData"),
        package_version="test-fixture",package_revision="not-native-inference",
        materialize=function(fit)list(indices=index,distances=distance))
      representation_metadata<-list(pca_file="pca_full_6.RData",pca_file_md5=receipt$pca_file_md5,
        kodama_graph_available=TRUE,kodama_graph_file="kodama_graph.rds",
        kodama_graph_manifest_file="kodama_graph.json",kodama_graph_sha256=receipt$file_sha256)
      save(vis,xy,common_ids,representation_metadata,file=file.path(a[1],"kodama_full_6.RData"))
    ''', directory, EXPORT, 13 if isolated else 12)
    return directory


@pytest.fixture
def graph_fixture(tmp_path, rscript):
    return make_graph_fixture(rscript, tmp_path / "graph")


def cluster(rscript, source, destination, *extra, check=True):
    destination.mkdir()
    output = destination / "fixture_cluster.csv"
    result = subprocess.run([rscript, str(CLUSTER), str(source), str(output), "--dim", "6",
        "--cluster-representation", "kodama_graph", "--algorithm", "leiden", "--resolution", ".3",
        "--stability-runs", "3", "--abstain-uncertain", "true", *extra], capture_output=True, text=True, timeout=120)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result, output


def test_sparse_zero_distance_max_union_and_isolated_vertex(rscript):
    run_r(rscript, r'''
      ids<-c("001","b","isolated")
      d<-Matrix::sparseMatrix(i=c(1,2),j=c(2,1),x=c(0,3),dims=c(3,3))
      g<-prepare_kodama_affinity_graph(list(observation_ids=ids,distance_graph=d))
      stopifnot(length(d@x)==2,d@x[2]==0,igraph::vcount(g$graph)==3,
        igraph::ecount(g$graph)==1,identical(g$observation_ids,ids),
        g$affinity[1,2]==1,g$affinity[2,1]==1,g$degree[3]==0,
        g$affinity[1,3]==0,g$connected_components==2)
      fit<-run_native_kodama_leiden(g,.3,seed=1)
      e<-kodama_graph_assignment_evidence(g,fit$membership)
      stopifnot(length(unique(fit$membership))==2,is.na(e$affinity_margin[3]),
        e$is_isolated[3],all(e$affinity_margin[1:2]==1),is.na(fit$silhouette))
    ''')


def test_all_isolates_remain_explicit_without_invented_communities(rscript):
    run_r(rscript, r'''
      n<-5000L;ids<-as.character(seq_len(n));d<-Matrix::sparseMatrix(i=integer(),j=integer(),x=numeric(),dims=c(n,n))
      g<-prepare_kodama_affinity_graph(list(observation_ids=ids,distance_graph=d))
      fit<-run_native_kodama_leiden(g,.3)
      e<-kodama_graph_assignment_evidence(g,fit$membership)
      stopifnot(length(fit$membership)==n,length(unique(fit$membership))==n,
        all(e$is_isolated),all(is.na(e$affinity_margin)),length(g$affinity@x)==0)
    ''')


def test_sparse_symmetrization_matches_exact_small_dense_reference(rscript):
    run_r(rscript, r'''
      set.seed(201);n<-31L
      edges<-expand.grid(i=seq_len(n),j=seq_len(n));edges<-edges[edges$i!=edges$j,]
      edges<-edges[sample.int(nrow(edges),147),];edges$d<-runif(nrow(edges))*100
      edges$d[c(1,8)]<-0
      d<-Matrix::sparseMatrix(i=edges$i,j=edges$j,x=edges$d,dims=c(n,n))
      expected<-matrix(0,n,n)
      for(k in seq_len(nrow(edges))){
        i<-edges$i[k];j<-edges$j[k];w<-1/(1+edges$d[k])
        expected[i,j]<-expected[j,i]<-max(expected[i,j],expected[j,i],w)
      }
      g<-prepare_kodama_affinity_graph(list(observation_ids=as.character(seq_len(n)),distance_graph=d))
      stopifnot(identical(unname(as.matrix(g$affinity)),expected))
    ''')


def test_hundred_thousand_vertices_never_coerced_to_dense(rscript):
    run_r(rscript, r'''
      library(Matrix);library(igraph)
      trace(".sparse2dense",where=asNamespace("Matrix"),print=FALSE,
        tracer=quote(stop("FORBIDDEN_SPARSE_TO_DENSE")))
      trace("as.matrix",signature="Matrix",where=asNamespace("Matrix"),print=FALSE,
        tracer=quote(stop("FORBIDDEN_SPARSE_TO_DENSE")))
      tiny<-Matrix::sparseMatrix(i=1,j=2,x=1,dims=c(3,3))
      trap<-try(as.matrix(tiny),silent=TRUE)
      stopifnot(inherits(trap,"try-error"),grepl("FORBIDDEN_SPARSE_TO_DENSE",as.character(trap)))
      old_path<-try(pmax(tiny,Matrix::t(tiny)),silent=TRUE)
      stopifnot(inherits(old_path,"try-error"),grepl("FORBIDDEN_SPARSE_TO_DENSE",as.character(old_path)))
      n<-100000L
      d<-Matrix::sparseMatrix(i=c(1,2,4),j=c(2,1,n),x=c(0,3,7),dims=c(n,n))
      g<-prepare_kodama_affinity_graph(list(observation_ids=as.character(seq_len(n)),distance_graph=d))
      stopifnot(length(g$observation_ids)==n,igraph::vcount(g$graph)==n,
        igraph::ecount(g$graph)==2,length(g$affinity@x)==4,
        g$affinity[1,2]==1,g$affinity[4,n]==1/8,sum(g$degree==0)==n-4L,
        as.numeric(object.size(g$affinity))<30000000)
      fit<-run_native_kodama_leiden(g,.3)
      e<-kodama_graph_assignment_evidence(g,fit$membership)
      merged<-collapse_kodama_graph_to_target(g,seq_len(n),n-1L)
      stopifnot(sum(e$is_isolated)==n-4L,all(is.na(e$affinity_margin[e$is_isolated])),
        length(merged$membership)==n,length(unique(merged$membership))==n-1L)
    ''')


def test_forced_merge_uses_supported_affinity_and_recomputes_final_evidence(rscript):
    run_r(rscript, r'''
      d<-Matrix::sparseMatrix(i=c(1,2),j=c(2,3),x=c(0,9),dims=c(4,4))
      g<-prepare_kodama_affinity_graph(list(observation_ids=as.character(1:4),distance_graph=d))
      three<-collapse_kodama_graph_to_target(g,1:4,3)
      stopifnot(identical(three$membership,c(1L,1L,2L,3L)),grepl("affinity_sum=1",three$merge_history))
      two<-collapse_kodama_graph_to_target(g,1:4,2)
      e<-kodama_graph_assignment_evidence(g,two$membership)
      stopifnot(all(e$affinity_margin[1:3]==1),is.na(e$affinity_margin[4]))
      failed<-try(collapse_kodama_graph_to_target(g,c(1,1,1,2),3),silent=TRUE)
      stopifnot(inherits(failed,"try-error"))
      separate<-Matrix::sparseMatrix(i=1,j=2,x=0,dims=c(4,4))
      g<-prepare_kodama_affinity_graph(list(observation_ids=as.character(1:4),distance_graph=separate))
      failed<-try(collapse_kodama_graph_to_target(g,1:4,2),silent=TRUE)
      stopifnot(inherits(failed,"try-error"),grepl("no supported affinity merge",as.character(failed)))
    ''')


@pytest.mark.parametrize("mutation", ["d@x[1]<-NaN", "d@x[1]<- -1", "d@x[1]<-Inf", "ids[2]<-ids[1]", "d<-as.matrix(d)"])
def test_bad_sparse_geometry_and_ids_rejected(rscript, mutation):
    result = run_r(rscript, r'''
      ids<-c("a","b","c");d<-Matrix::sparseMatrix(i=1,j=2,x=1,dims=c(3,3));
    ''' + mutation + ";prepare_kodama_affinity_graph(list(observation_ids=ids,distance_graph=d))", check=False)
    assert result.returncode != 0


def test_cli_native_graph_preserves_ids_and_ignores_visualization_geometry(graph_fixture, tmp_path, rscript):
    _, output = cluster(rscript, graph_fixture, tmp_path / "first", "--target-clusters", "2")
    first = pd.read_csv(output, dtype={"label": str})
    assert first.label.tolist() == [f"{i:03}" for i in range(1, 13)]
    assert first.cluster.nunique() == 2 and first.clustering_dimensions.eq(0).all()
    assert first.graph_degree.gt(0).all() and first.assignment_vote_margin.eq(1).all()
    assert first.graph_affinity_rule.eq("stored_distance_to_1_over_1_plus_d_symmetric_max_union").all()
    summary = pd.read_csv(output.parent / "fixture_cluster_summary.csv").iloc[0]
    assert summary.landmark_assignment_mode == "native_graph"
    assert pd.isna(summary.actual_k)
    assert summary.target_strategy == "maximum_total_cross_community_graph_affinity_supported_merges_only"
    assert summary.graph_connected_components == 2 and summary.graph_isolated_observations == 0
    assert summary.stability_mean_adjusted_rand_index == 1
    run_r(rscript, 'a<-commandArgs(TRUE);load(a[1]);vis<-vis* -1000+717;save(vis,xy,common_ids,representation_metadata,file=a[1])', graph_fixture / "kodama_full_6.RData")
    _, other = cluster(rscript, graph_fixture, tmp_path / "other", "--target-clusters", "2")
    second = pd.read_csv(other, dtype={"label": str})
    assert first.cluster.tolist() == second.cluster.tolist()
    assert first.assignment_vote_margin.tolist() == second.assignment_vote_margin.tolist()


def test_cli_isolated_vertex_abstains_and_unsupported_target_fails(tmp_path, rscript):
    source = make_graph_fixture(rscript, tmp_path / "isolated", isolated=True)
    _, output = cluster(rscript, source, tmp_path / "preserved")
    table = pd.read_csv(output, dtype={"label": str})
    isolated = table[table.label.eq("013")].iloc[0]
    assert len(table) == 13 and isolated.graph_degree == 0
    assert isolated.is_abstained and pd.isna(isolated.interpretable_cluster)
    assert pd.isna(isolated.assignment_vote_margin)
    assert isolated.uncertainty_reason == "isolated_native_graph_vertex"
    _, off_output = cluster(rscript, source, tmp_path / "policy_off", "--abstain-uncertain", "false")
    off = pd.read_csv(off_output, dtype={"label": str}).set_index("label")
    assert pd.isna(off.loc["013", "interpretable_cluster"]) and off.loc["013", "is_abstained"]
    result, _ = cluster(rscript, source, tmp_path / "unsupported", "--target-clusters", "2", check=False)
    assert result.returncode != 0 and "no supported affinity merge" in result.stderr


@pytest.mark.parametrize("extra", [("--algorithm", "walktrap"), ("--resolution", "auto"), ("--profile", "fine"),
    ("--cluster-dimensions", "3"), ("--landmark-cells", "10"), ("--k", "50"), ("--landmark-sample-strategy", "random")])
def test_cli_rejects_options_it_cannot_honor(graph_fixture, tmp_path, rscript, extra):
    result, _ = cluster(rscript, graph_fixture, tmp_path / "invalid", *extra, check=False)
    assert result.returncode != 0 and "kodama_graph" in result.stderr


def test_cli_graph_checksum_and_order_binding(graph_fixture, tmp_path, rscript):
    run_r(rscript, 'a<-commandArgs(TRUE);load(a[1]);vis<-vis[nrow(vis):1,,drop=FALSE];save(vis,xy,common_ids,representation_metadata,file=a[1])', graph_fixture / "kodama_full_6.RData")
    result, _ = cluster(rscript, graph_fixture, tmp_path / "foreign_order", check=False)
    assert result.returncode != 0 and "IDs/order must match exactly" in result.stderr


@pytest.mark.parametrize("mutation", ["representation_metadata<-NULL",
    "representation_metadata$kodama_graph_available<-FALSE",
    "representation_metadata$kodama_graph_sha256<-NULL",
    "representation_metadata$kodama_graph_sha256<-paste(rep(\"0\",64),collapse=\"\")",
    "representation_metadata$kodama_graph_file<-\"foreign.rds\""])
def test_cli_requires_explicit_fresh_producer_graph_binding(graph_fixture, tmp_path, rscript, mutation):
    run_r(rscript, 'a<-commandArgs(TRUE);load(a[1]);' + mutation + ';save(vis,xy,common_ids,representation_metadata,file=a[1])', graph_fixture / "kodama_full_6.RData")
    result, output = cluster(rscript, graph_fixture, tmp_path / "unbound", check=False)
    assert result.returncode != 0 and not output.exists()
    assert "representation_metadata" in result.stderr or "SHA256 checksum differs" in result.stderr


def test_actual_nextflow_module_executes_graph_native_command(graph_fixture, tmp_path, rscript):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    project = tmp_path / "nextflow_project"
    (project / "bin").mkdir(parents=True)
    (project / "modules").mkdir()
    (project / "lib").mkdir()
    for path in (CLUSTER, HELPER, EXPORT):
        shutil.copy2(path, project / "bin" / path.name)
    shutil.copy2(ROOT / "modules/run_rcode_clustering.nf", project / "modules")
    shutil.copy2(ROOT / "lib/PipelineHelpers.groovy", project / "lib")
    shutil.copy2(ROOT / "lib/ProcessCode.groovy", project / "lib")
    (project / "objects.csv").write_text("label\n001\n")
    (project / "main.nf").write_text('''
      nextflow.enable.dsl=2
      include { RUN_RCODE_CLUSTERING } from './modules/run_rcode_clustering'
      workflow { RUN_RCODE_CLUSTERING(Channel.of(tuple('fixture_standard','fixture','standard','standard','0.3',file(params.source),file('objects.csv')))) }
    ''')
    library = subprocess.run([rscript, "-e", "cat(.libPaths()[1])"], capture_output=True, text=True, check=True).stdout
    params = dict(source=str(graph_fixture), outdir_base=str(project / "results"),
        publish_dir_mode="copy", _executor_max_cpus=2, _executor_max_memory_gb=4,
        cluster_cpus=1, cluster_memory_gb=2, cluster_time="5m", cluster_target_clusters=2,
        cluster_forced_count_sensitivity_acknowledged=True, cluster_r_script="bin/Rcode_Clustering.R",
        cluster_r_library_dir=library, cluster_representation="kodama_graph", cluster_algorithm="leiden",
        cluster_landmark_cells=0, cluster_representation_dimensions=0, cluster_kodama_dim=6,
        cluster_seed=1, cluster_stability_runs=3, cluster_assignment_min_vote_margin=.1,
        cluster_stability_min_fraction=.67, cluster_abstain_uncertain=True)
    (project / "params.json").write_text(json.dumps(params))
    (project / "nextflow.config").write_text("process.executor='local'\ntrace { enabled=true; file='trace.tsv'; overwrite=true }\n")
    result = subprocess.run([nextflow, "run", "main.nf", "-params-file", "params.json", "-ansi-log", "false"],
        cwd=project, env={**os.environ, "NXF_OFFLINE": "true", "NXF_ANSI_LOG": "false"},
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    trace = pd.read_csv(project / "trace.tsv", sep="\t")
    assert len(trace) == 1 and trace.status.eq("COMPLETED").all()
    table = pd.read_csv(project / "results/11_clustering/fixture/fixture_standard_cluster.csv", dtype={"label": str})
    assert len(table) == 12 and table.cluster.nunique() == 2
    assert table.cluster_representation.eq("kodama_graph").all()
    command = next((project / "work").rglob(".command.sh")).read_text()
    assert "--cluster-representation kodama_graph" in command
    assert "--k " not in command and "--landmark-assign-k" not in command and "--fine-multiplier" not in command


def test_graph_parameter_schema_requires_compatible_settings():
    import jsonschema
    schema = json.loads((ROOT / "nextflow_schema.json").read_text())
    validator = jsonschema.Draft7Validator(schema)
    assert not list(validator.iter_errors({"cluster_representation": "kodama_graph", "cluster_landmark_cells": 0}))
    for extra in ({}, {"cluster_landmark_cells": 100}, {"cluster_landmark_cells": 0, "cluster_algorithm": "walktrap"},
                  {"cluster_landmark_cells": 0, "cluster_resolution": "auto"}, {"cluster_landmark_cells": 0, "cluster_secondary_variant": "fine"}):
        assert list(validator.iter_errors({"cluster_representation": "kodama_graph", **extra}))


def test_clustering_nextflow_fingerprint_includes_both_graph_helpers():
    module = (ROOT / "modules/run_rcode_clustering.nf").read_text()
    assert "bin/kodama_graph_clustering.R" in module and "bin/kodama_graph_export.R" in module
    assert "nativeGraphMode" in module and "coordinateOptions" in module
