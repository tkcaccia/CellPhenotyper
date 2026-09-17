"""Actual R sparse-kernel tests; synthetic PCA, no learned-image inference."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "bin/kodama_graph_clustering.R"
HELPER = ROOT / "bin/kodama_spatial_regularization.R"


@pytest.fixture(scope="module")
def rscript():
    executable = shutil.which("Rscript")
    if not executable:
        pytest.skip("Rscript is unavailable")
    result = subprocess.run([executable, "-e",
        'if(!all(vapply(c("Matrix","igraph"),requireNamespace,logical(1),quietly=TRUE)))quit(status=77)'],
        text=True, capture_output=True, timeout=60)
    if result.returncode == 77:
        pytest.skip("Optional R sparse graph packages unavailable")
    assert result.returncode == 0, result.stderr
    return executable


def run_r(rscript, body):
    script = f"source({json.dumps(str(BASE))});source({json.dumps(str(HELPER))});" + body
    result = subprocess.run([rscript, "-e", script], text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


FIXTURE = r'''
  ids <- c("001", "b", "03", "four", "isolate")
  original <- Matrix::sparseMatrix(i=c(1,2,3), j=c(2,3,4), x=c(0,3,1), dims=c(5,5))
  g <- prepare_kodama_affinity_graph(list(observation_ids=ids,distance_graph=original))
  pca <- matrix(c(0,0, 1,0, 3,0, 3,4, 9,7), nrow=5, byrow=TRUE, dimnames=list(ids,c("PC1","PC2")))
  edges <- data.frame(source_index=c(1L,2L,3L),target_index=c(2L,4L,4L),
    distance_um=c(10,20,10),boundary_weight=c(1,.2,0))
'''


def test_independent_dense_oracle_and_no_input_or_rng_mutation(rscript):
    run_r(rscript, FIXTURE + r'''
      set.seed(817);before<-serialize(list(g,edges,pca,.Random.seed),NULL)
      result<-regularize_kodama_graph(g,edges,pca,.35)
      stopifnot(identical(before,serialize(list(g,edges,pca,.Random.seed),NULL)))
      expected<-matrix(0,5,5)
      expected[1,2]<-expected[2,1]<-1
      expected[2,3]<-expected[3,2]<-1/4
      expected[3,4]<-expected[4,3]<-1/2
      distances<-vapply(seq_len(nrow(edges)),function(k)
        sqrt(sum((pca[edges$source_index[k],]-pca[edges$target_index[k],])^2)),numeric(1))
      sigma<-median(distances[distances>0])
      kernel<-exp(-distances^2/(2*sigma^2))
      raw<-kernel*edges$boundary_weight
      increments<-raw/sum(kernel)*(.35*sum(expected)/2)
      for(k in seq_len(nrow(edges))){i<-edges$source_index[k];j<-edges$target_index[k]
        expected[i,j]<-expected[j,i]<-expected[i,j]+increments[k]}
      stopifnot(isTRUE(all.equal(unname(as.matrix(result$affinity)),expected,tolerance=1e-14)),
        isTRUE(all.equal(result$spatial_local_edges$pca_distance,distances,tolerance=1e-14)),
        isTRUE(all.equal(result$spatial_local_edges$added_affinity_weight,increments,tolerance=1e-14)),
        identical(result$observation_ids,ids), result$degree[5]==0L,
        result$spatial_regularization$applied,
        isTRUE(all.equal(result$spatial_regularization$added_affinity_mass,
          .35*sum(g$affinity)*sum(raw)/sum(kernel),tolerance=1e-14)),
        result$spatial_regularization$actual_added_to_original_mass_fraction<=.35,
        identical(result$degree,as.integer(igraph::degree(result$graph))),
        identical(result$strength,as.numeric(Matrix::rowSums(result$affinity))),
        result$connected_components==2L)
    ''')


def test_zero_weight_returns_exact_original_graph_and_sparse_slots(rscript):
    run_r(rscript, FIXTURE + r'''
      result<-regularize_kodama_graph(g,edges,pca,0)
      stopifnot(identical(result[names(g)],g), identical(result$affinity,g$affinity),
        identical(result$graph,g$graph), !result$spatial_regularization$applied,
        result$spatial_regularization$reason=="spatial_weight_zero",
        all(result$spatial_local_edges$added_affinity_weight==0),
        identical(result$spatial_local_edges$source_index,edges$source_index))
    ''')


def test_random_sparse_oracle_and_candidate_order_independence(rscript):
    run_r(rscript, r'''
      for(seed in c(1,37,817,2003)){
        set.seed(seed);n<-37L;ids<-sprintf("%04d",seq_len(n))
        candidates<-which(upper.tri(matrix(FALSE,n,n)),arr.ind=TRUE)
        selected<-sample.int(nrow(candidates),71)
        original<-Matrix::sparseMatrix(i=candidates[selected,1],j=candidates[selected,2],
          x=runif(length(selected)),dims=c(n,n))
        g<-prepare_kodama_affinity_graph(list(observation_ids=ids,distance_graph=original))
        selected<-sample.int(nrow(candidates),83)
        edges<-data.frame(source_index=candidates[selected,1],target_index=candidates[selected,2],
          distance_um=runif(83)+1,boundary_weight=runif(83))
        edges$boundary_weight[seq(1,83,9)]<-0
        x<-matrix(rnorm(n*7),nrow=n,dimnames=list(ids,paste0("PC",1:7)))
        for(boundary in c(TRUE,FALSE)){
          actual<-regularize_kodama_graph(g,edges,x,.7,boundary)
          squared<-vapply(seq_len(nrow(edges)),function(k)
            sum((x[edges$source_index[k],]-x[edges$target_index[k],])^2),numeric(1))
          kernel<-exp(-squared/(2*median(sqrt(squared[squared>0]))^2))
          raw<-kernel
          if(boundary)raw<-raw*edges$boundary_weight
          weights<-(.7*sum(g$affinity)/(2*sum(kernel)))*raw
          expected<-g$affinity+Matrix::sparseMatrix(
            i=c(edges$source_index,edges$target_index),j=c(edges$target_index,edges$source_index),
            x=rep(weights,2),dims=c(n,n),dimnames=list(ids,ids))
          stopifnot(max(abs((expected-actual$affinity)@x),0)<1e-13,
            isTRUE(all.equal(actual$spatial_regularization$added_affinity_mass,
              .7*sum(g$affinity)*sum(raw)/sum(kernel),tolerance=1e-14)),
            actual$spatial_regularization$actual_added_to_original_mass_fraction<=.7+1e-14)
          shuffled<-regularize_kodama_graph(g,edges[sample.int(nrow(edges)),],x,.7,boundary)
          stopifnot(max(abs((shuffled$affinity-actual$affinity)@x),0)<1e-13)
        }
      }
    ''')


def test_boundary_control_suppression_and_original_nonlocal_edges_retained(rscript):
    run_r(rscript, FIXTURE + r'''
      aware<-regularize_kodama_graph(g,edges,pca,.1,TRUE)
      control<-regularize_kodama_graph(g,edges,pca,.1,FALSE)
      stopifnot(aware$spatial_local_edges$added_affinity_weight[3]==0,
        control$spatial_local_edges$added_affinity_weight[3]>0,
        aware$affinity[3,4]==g$affinity[3,4],
        control$affinity[3,4]>g$affinity[3,4],
        aware$affinity[2,3]==g$affinity[2,3],
        aware$spatial_regularization$original_nonlocal_edges_retained,
        isTRUE(all.equal(aware$spatial_local_edges$added_affinity_weight,
          control$spatial_local_edges$added_affinity_weight*edges$boundary_weight,tolerance=1e-14)),
        identical(aware$spatial_regularization$feature_only_local_affinity_mass,
          control$spatial_regularization$feature_only_local_affinity_mass),
        all(aware$affinity@x>0), all(control$affinity@x>0))
      # Input physical distances are audit/provenance fields, not a hidden
      # second smoothing kernel. The edge producer has already bounded them.
      edges$distance_um<-edges$distance_um*3
      other<-regularize_kodama_graph(g,edges,pca,.1,TRUE)
      stopifnot(identical(other$affinity,aware$affinity))
      # A uniformly strong boundary must suppress total mass rather than get
      # normalized back to the control's requested ten-percent addition.
      edges$boundary_weight<-.001
      weak<-regularize_kodama_graph(g,edges,pca,.1,TRUE)
      stopifnot(isTRUE(all.equal(weak$spatial_regularization$added_affinity_mass,
        control$spatial_regularization$added_affinity_mass*.001,tolerance=1e-14)))
    ''')


@pytest.mark.parametrize("setup,reason", [
    ('edges<-edges[FALSE,]', "no_physical_edges"),
    ('edges$boundary_weight<-0', "local_zero_mass"),
    ('original<-Matrix::sparseMatrix(i=integer(),j=integer(),x=numeric(),dims=c(5,5));'
     'g<-prepare_kodama_affinity_graph(list(observation_ids=ids,distance_graph=original))', "original_zero_mass"),
])
def test_explicit_noop_empty_or_zero_mass_without_invented_edges(rscript, setup, reason):
    run_r(rscript, FIXTURE + setup + f'''
      result<-regularize_kodama_graph(g,edges,pca,.1)
      stopifnot(identical(result[names(g)],g), result$spatial_regularization$reason=={json.dumps(reason)},
        !result$spatial_regularization$applied, all(result$spatial_local_edges$added_affinity_weight==0))
    ''')


def test_zero_and_mixed_zero_pca_distances_keep_correct_kernel(rscript):
    run_r(rscript, FIXTURE + r'''
      pca[,]<-0;edges$boundary_weight<-c(1,.5,1)
      result<-regularize_kodama_graph(g,edges,pca,.1)
      stopifnot(result$spatial_regularization$sigma_pca==0,
        all(result$spatial_local_edges$feature_kernel==1),
        isTRUE(all.equal(result$spatial_local_edges$added_affinity_weight,
          c(1,.5,1)/3*(.1*sum(g$affinity)/2),tolerance=1e-14)))
      pca[4,1]<-2
      result<-regularize_kodama_graph(g,edges,pca,.1)
      stopifnot(result$spatial_regularization$sigma_pca==2,
        result$spatial_local_edges$feature_kernel[1]==1,
        all(abs(result$spatial_local_edges$feature_kernel[2:3]-exp(-.5))<1e-15))
    ''')


def test_single_vertex_no_candidates_and_candidate_can_connect_an_isolate(rscript):
    run_r(rscript, FIXTURE + r'''
      edges<-rbind(edges,data.frame(source_index=4L,target_index=5L,distance_um=5,boundary_weight=1))
      result<-regularize_kodama_graph(g,edges,pca,.1)
      stopifnot(igraph::vcount(result$graph)==5L,result$degree[5]==1L,result$connected_components==1L,
        result$spatial_regularization$added_native_strength_ratio$native_zero_node_count==1L,
        result$spatial_regularization$added_native_strength_ratio$native_zero_with_added_strength_node_count==1L)
      ids<-"only";d<-Matrix::sparseMatrix(i=integer(),j=integer(),x=numeric(),dims=c(1,1))
      one<-prepare_kodama_affinity_graph(list(observation_ids=ids,distance_graph=d))
      x<-matrix(0,1,1,dimnames=list(ids,"PC1"))
      result<-regularize_kodama_graph(one,edges[FALSE,],x,.1)
      stopifnot(identical(result[names(one)],one),igraph::vcount(result$graph)==1L)
    ''')


def test_per_node_mass_and_edge_overlap_audit_matches_independent_sums(rscript):
    run_r(rscript, FIXTURE + r'''
      result<-regularize_kodama_graph(g,edges,pca,.2)
      increment<-numeric(5)
      for(k in seq_len(nrow(edges))){
        increment[edges$source_index[k]]<-increment[edges$source_index[k]]+result$spatial_local_edges$added_affinity_weight[k]
        increment[edges$target_index[k]]<-increment[edges$target_index[k]]+result$spatial_local_edges$added_affinity_weight[k]
      }
      ratios<-increment[g$strength>0]/g$strength[g$strength>0]
      record<-result$spatial_regularization
      expected<-unname(quantile(ratios,c(0,.25,.5,.75,.95,.99,1)))
      stopifnot(isTRUE(all.equal(unname(unlist(record$added_native_strength_ratio$finite_ratio_quantiles)),expected,tolerance=1e-14)),
        record$added_native_strength_ratio$max==max(ratios),
        record$added_native_strength_ratio$ratio_overflow_node_count==0,
        record$candidate_existing_edge_count==2L,record$candidate_new_edge_count==1L,
        record$positive_added_existing_edge_count==1L,record$positive_added_new_edge_count==1L)
    ''')


def test_large_feature_values_use_stable_norm_without_squared_overflow(rscript):
    run_r(rscript, FIXTURE + r'''
      small<-regularize_kodama_graph(g,edges,pca,.1)
      large<-regularize_kodama_graph(g,edges,pca*1e200,.1)
      stopifnot(all(is.finite(large$spatial_local_edges$pca_distance)),
        isTRUE(all.equal(small$affinity,large$affinity,tolerance=1e-14)))
    ''')


def test_hundred_thousand_vertices_never_dense(rscript):
    run_r(rscript, r'''
      library(Matrix);library(igraph)
      trace(".sparse2dense",where=asNamespace("Matrix"),print=FALSE,
        tracer=quote(stop("FORBIDDEN_SPARSE_TO_DENSE")))
      trace("as.matrix",signature="Matrix",where=asNamespace("Matrix"),print=FALSE,
        tracer=quote(stop("FORBIDDEN_SPARSE_TO_DENSE")))
      trap<-try(as.matrix(Matrix::sparseMatrix(i=1,j=2,x=1,dims=c(3,3))),silent=TRUE)
      stopifnot(inherits(trap,"try-error"),grepl("FORBIDDEN_SPARSE_TO_DENSE",as.character(trap)))
      n<-100000L;ids<-as.character(seq_len(n))
      original<-Matrix::sparseMatrix(i=c(1,4),j=c(2,n),x=c(0,7),dims=c(n,n))
      g<-prepare_kodama_affinity_graph(list(observation_ids=ids,distance_graph=original))
      x<-cbind(PC1=seq_len(n),PC2=rep(0,n));rownames(x)<-ids
      edges<-data.frame(source_index=c(1L,2L,3L),target_index=c(3L,3L,n),
        distance_um=c(1,1,2),boundary_weight=c(1,0,.5))
      result<-regularize_kodama_graph(g,edges,x,.1)
      stopifnot(inherits(result$affinity,"dgCMatrix"),length(result$affinity@x)<=10,
        igraph::vcount(result$graph)==n,identical(result$observation_ids,ids),
        as.numeric(object.size(result$affinity))<30000000,
        result$affinity[2,3]==0,result$affinity[4,n]==g$affinity[4,n],
        sum(result$degree==0)>=n-5L)
    ''')


@pytest.mark.parametrize("mutation", [
    'edges$source_index[1]<-0', 'edges$source_index[1]<-1.5',
    'edges$target_index[1]<-6', 'edges$target_index[1]<-1',
    'edges<-rbind(edges,edges[1,])', 'edges$source_index<-as.character(edges$source_index)',
    'edges$distance_um[1]<-0', 'edges$distance_um[1]<-NA_real_',
    'edges$boundary_weight[1]<-1.1', 'edges$boundary_weight[1]<- -1',
    'edges$boundary_weight[1]<-Inf', 'edges$boundary_weight<-NULL',
    'edges$feature_kernel<-0', 'pca<-pca[5:1,]', 'rownames(pca)<-NULL',
    'pca[1,1]<-NaN', 'pca<-pca[,FALSE,drop=FALSE]', 'pca<-as.data.frame(pca)',
    'g$observation_ids[2]<-g$observation_ids[1]', 'g$affinity<-as.matrix(g$affinity)',
    'g$affinity[1,2]<-2', 'g$affinity[1,1]<-1', 'g$affinity@x[1]<-Inf',
    'g$degree[1]<-99L', 'g$strength[1]<-99', 'g$connected_components<-99L',
    'igraph::E(g$graph)$weight[1]<-99',
])
def test_malformed_source_contract_rejected_even_at_zero_weight(rscript, mutation):
    run_r(rscript, FIXTURE + mutation + r'''
      error<-try(regularize_kodama_graph(g,edges,pca,0),silent=TRUE)
      stopifnot(inherits(error,"try-error"))
    ''')


@pytest.mark.parametrize("arguments", [
    'spatial_weight=-1', 'spatial_weight=Inf', 'spatial_weight=NA_real_',
    'spatial_weight=c(.1,.2)', 'spatial_weight=TRUE', 'spatial_weight=".1"',
    'boundary_aware=1', 'boundary_aware=NA', 'boundary_aware=c(TRUE,FALSE)',
])
def test_invalid_options_rejected(rscript, arguments):
    run_r(rscript, FIXTURE + f'''
      error<-try(regularize_kodama_graph(g,edges,pca,{arguments}),silent=TRUE)
      stopifnot(inherits(error,"try-error"))
    ''')


def test_repeated_application_and_numeric_overflow_fail_explicitly(rscript):
    run_r(rscript, FIXTURE + r'''
      once<-regularize_kodama_graph(g,edges,pca,.1)
      failure<-try(regularize_kodama_graph(once,edges,pca,.1),silent=TRUE)
      stopifnot(inherits(failure,"try-error"),grepl("unregularized",as.character(failure)))
      failure<-try(regularize_kodama_graph(g,edges,pca,.Machine$double.xmax),silent=TRUE)
      stopifnot(inherits(failure,"try-error"),grepl("mass overflowed",as.character(failure)))
      pca[1,1]<-.Machine$double.xmax;pca[2,1]<- -.Machine$double.xmax
      failure<-try(regularize_kodama_graph(g,edges,pca,.1),silent=TRUE)
      stopifnot(inherits(failure,"try-error"),grepl("differences overflow",as.character(failure)))
    ''')
