"""Executable R fixtures verify routing/identity, not pathology accuracy."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "bin" / "Rcode_Clustering.R"
KODAMA = ROOT / "bin" / "run_kodama_analysis.R"
RSCRIPT = shutil.which("Rscript")


@pytest.fixture(scope="module")
def rscript():
    if not RSCRIPT:
        pytest.skip("Rscript unavailable")
    result = subprocess.run([RSCRIPT, "-e", 'p<-c("bluster","igraph","cluster","BiocNeighbors"); if(!all(vapply(p,requireNamespace,logical(1),quietly=TRUE))) quit(status=77)'], capture_output=True, text=True)
    if result.returncode == 77:
        pytest.skip("R clustering packages unavailable")
    assert result.returncode == 0, result.stderr
    return RSCRIPT


@pytest.fixture
def data_directory(tmp_path, rscript):
    source = tmp_path / "kodama"
    source.mkdir()
    # UMAP coordinates deliberately carry no group information; PCA dimensions
    # 3:6 carry it. The PCA sidecar row order is deliberately reversed.
    code = r'''
      args <- commandArgs(trailingOnly=TRUE)
      set.seed(717)
      n <- 120L
      common_ids <- sprintf("cell_%03d",seq_len(n))
      truth <- rep(c(0,1),each=n/2)
      vis <- matrix(rnorm(n*2),ncol=2)
      rownames(vis) <- common_ids
      pca <- cbind(vis,matrix(rnorm(n*4,sd=.05),ncol=4)+truth*8)
      rownames(pca) <- common_ids
      colnames(pca) <- paste0("PC",seq_len(ncol(pca)))
      pca <- pca[n:1,,drop=FALSE]
      xy <- vis
      save(pca,xy,common_ids,file=file.path(args[1],"pca_full_6.RData"))
      representation_metadata <- list(
        schema_version="1.0.0",pca_file="pca_full_6.RData",
        pca_file_md5=unname(tools::md5sum(file.path(args[1],"pca_full_6.RData"))),
        pca_dimensions=6L,visualization_projected=TRUE,
        requested_kodama_ncomp=50L,effective_kodama_ncomp=6L)
      save(vis,xy,common_ids,representation_metadata,file=file.path(args[1],"kodama_full_6.RData"))
      write.csv(data.frame(label=common_ids,truth=truth),file.path(args[1],"truth.csv"),row.names=FALSE)
    '''
    result = subprocess.run([rscript, "-e", code, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return source


def execute(rscript, directory, output, *extra, check=True):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [rscript, str(CLUSTER), str(directory), str(output), "--dim", "6", "--algorithm", "walktrap", "--walktrap-clusters", "2", "--target-clusters", "2", "--k", "5", "--landmark-cells", "0", "--landmark-sample-strategy", "random", "--stability-runs", "2", *extra]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if check:
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
    return result


def test_actual_r_pca_uses_all_dimensions_and_keeps_visualization_separate(data_directory, tmp_path, rscript):
    output = tmp_path / "highdim" / "sample_cluster.csv"
    execute(rscript, data_directory, output, "--cluster-representation", "pca")
    observed = pd.read_csv(output)
    truth = pd.read_csv(data_directory / "truth.csv")
    assert observed.label.tolist() == truth.label.tolist()
    assert adjusted_rand_score(truth.truth, observed.cluster) == 1
    assert observed.cluster_representation.eq("pca").all()
    assert observed.clustering_dimensions.eq(6).all()
    assert observed.cluster.nunique() == 2
    summary = pd.read_csv(output.parent / "sample_cluster_summary.csv").iloc[0]
    assert summary.vis_dims == 2 and summary.clustering_dimensions == 6
    assert summary.visualization_projected
    assert summary.target_strategy == "nearest_centroid_merge_in_pca_space"
    assert summary.claim_status == "sensitivity_only"
    assert (output.parent / "sample_cluster_kodama_membership.png").is_file()
    assert observed.stability_fraction.eq(1).all()


def test_default_remains_umap_baseline_and_matched_comparison_is_descriptive(data_directory, tmp_path, rscript):
    baseline = tmp_path / "baseline" / "reference_cluster.csv"
    execute(rscript, data_directory, baseline)
    reference = pd.read_csv(baseline)
    assert reference.cluster_representation.eq("umap2d").all()
    assert reference.clustering_dimensions.eq(2).all()
    output = tmp_path / "pca" / "sample_cluster.csv"
    execute(rscript, data_directory, output, "--cluster-representation", "pca", "--compare-with-clusters", str(baseline))
    comparison = pd.read_csv(output.parent / "sample_representation_comparison.csv").iloc[0]
    assert comparison.current_representation == "pca"
    assert comparison.reference_representation == "umap2d"
    assert comparison.matched_observations == 120
    assert comparison.input_bundle_identity_verified
    assert comparison.adjusted_rand_index_raw < .2
    assert comparison.comparison_claim == "descriptive_partition_agreement_only_not_accuracy_or_biological_validation"


def test_pca_large_n_landmark_assignment_uses_highdimensional_space(data_directory, tmp_path, rscript):
    output = tmp_path / "landmarks" / "sample_cluster.csv"
    execute(rscript, data_directory, output, "--cluster-representation", "pca", "--cluster-dimensions", "5", "--landmark-cells", "40", "--landmark-assign-k", "3")
    observed = pd.read_csv(output)
    summary = pd.read_csv(output.parent / "sample_cluster_summary.csv").iloc[0]
    assert summary.landmark_cells_used == 40
    assert summary.clustering_dimensions == 5
    assert observed.clustering_dimensions.eq(5).all()
    truth = pd.read_csv(data_directory / "truth.csv")
    assert adjusted_rand_score(truth.truth, observed.cluster) == 1
    assert len(observed) == 120


def test_auto_resolution_selects_minimum_abstention_candidate(data_directory, tmp_path, rscript):
    output = tmp_path / "auto_abstention" / "sample_cluster.csv"
    output.parent.mkdir(parents=True)
    command = [
        rscript,
        str(CLUSTER),
        str(data_directory),
        str(output),
        "--dim", "6",
        "--cluster-representation", "pca",
        "--algorithm", "leiden",
        "--resolution", "auto",
        "--target-clusters", "0",
        "--k", "5",
        "--landmark-cells", "40",
        "--landmark-assign-k", "5",
        "--landmark-sample-strategy", "random",
        "--stability-runs", "2",
        "--auto-selection", "minimum_abstention",
        "--abstain-uncertain", "true",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    candidates = pd.read_csv(output.parent / "sample_cluster_resolution_candidates.csv")
    eligible = candidates.loc[candidates.selection_eligible]
    selected = candidates.loc[candidates.selected]
    assert len(candidates) == 6
    assert len(selected) == 1
    assert selected.iloc[0].estimated_abstained_count == eligible.estimated_abstained_count.min()
    summary = pd.read_csv(output.parent / "sample_cluster_summary.csv").iloc[0]
    assert summary.auto_resolution_selection == "minimum_abstention"
    assert summary.selected_resolution == selected.iloc[0].resolution
    assert summary.selection_estimated_abstained_count == selected.iloc[0].estimated_abstained_count


def test_pca_refuses_missing_scores_and_unverified_graph(data_directory, tmp_path, rscript):
    (data_directory / "pca_full_6.RData").unlink()
    output = tmp_path / "missing" / "sample_cluster.csv"
    result = execute(rscript, data_directory, output, "--cluster-representation", "pca", check=False)
    assert result.returncode != 0 and "Saved PCA scores missing" in result.stderr
    result = subprocess.run([rscript, str(CLUSTER), str(data_directory), str(output),
        "--dim", "6", "--cluster-representation", "kodama_graph"], capture_output=True, text=True)
    assert result.returncode != 0 and "explicitly bound producer representation_metadata" in result.stderr


def test_pca_rejects_changed_sidecar_checksum(data_directory, tmp_path, rscript):
    result = subprocess.run([rscript, "-e", 'a<-commandArgs(TRUE);load(a[1]);pca[1,1]<-pca[1,1]+1;save(pca,file=a[1])', str(data_directory / "pca_full_6.RData")], capture_output=True, text=True)
    assert result.returncode == 0
    result = execute(rscript, data_directory, tmp_path / "changed" / "sample_cluster.csv", "--cluster-representation", "pca", check=False)
    assert result.returncode != 0 and "checksum differs" in result.stderr


def test_grid_landmarks_not_silently_reduced_to_two_dimensions(data_directory, tmp_path, rscript):
    result = execute(rscript, data_directory, tmp_path / "badgrid" / "sample_cluster.csv", "--cluster-representation", "pca", "--landmark-cells", "40", "--landmark-sample-strategy", "grid", check=False)
    assert result.returncode != 0 and "Two-dimensional grid landmark sampling is not valid" in result.stderr


def test_legacy_sidecar_ids_cannot_be_silently_intersected(data_directory, tmp_path, rscript):
    code = r'''
      a<-commandArgs(TRUE)
      load(file.path(a[1],"kodama_full_6.RData"))
      save(vis,xy,common_ids,file=file.path(a[1],"kodama_full_6.RData"))
      load(file.path(a[1],"pca_full_6.RData"))
      rownames(pca)[1]<-"foreign_observation"
      save(pca,file=file.path(a[1],"pca_full_6.RData"))
    '''
    result = subprocess.run([rscript, "-e", code, str(data_directory)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = execute(rscript, data_directory, tmp_path / "ids" / "sample_cluster.csv", "--cluster-representation", "pca", check=False)
    assert result.returncode != 0 and "observation IDs must match exactly" in result.stderr


def test_r_parsers_and_ncomp_export_contract(rscript):
    for path in (CLUSTER, KODAMA):
        result = subprocess.run([rscript, "-e", f"invisible(parse(file={json.dumps(str(path))}));cat('OK')"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    source = KODAMA.read_text()
    assert "requested_kodama_ncomp <- as.integer(kodama_ncomp)" in source
    assert "effective_kodama_ncomp <- min(requested_kodama_ncomp, dims_use)" in source
    assert source.count("\n    ncomp = effective_kodama_ncomp,") == 2
    assert "pca_scores_projected = FALSE" in source
    assert "pca_file_md5 = unname(tools::md5sum(pca_rdata))" in source
    assert "selected_features_saved = save_selected_features" in source
    # Native KODAMA inference is not claimed tested here: the local installed
    # KODAMA3.3 lacks the pipeline's KODAMA.pca/native backend API.
