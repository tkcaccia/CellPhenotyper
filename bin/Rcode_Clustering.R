args <- commandArgs(trailingOnly = TRUE)
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

if (length(args) < 2L) {
  stop(paste(
    "Usage: Rscript Rcode_Clustering.R <kodama_dir> <out_csv>",
    "[--dim N] [--k N] [--algorithm louvain|leiden|walktrap]",
    "[--cluster-representation umap2d|pca|kodama_graph] [--cluster-dimensions N] [--compare-with-clusters CSV]",
    "[--target-clusters N] [--landmark-cells N] [--landmark-assign-k N]",
    "[--landmark-sample-strategy random|grid|knn_inverse_distance]",
    "[--landmark-density-knn-k N] [--landmark-density-power X]",
    "[--walktrap-clusters N] [--resolution auto|X]",
    "[--profile standard|fine] [--seed N] [--stability-runs N]",
    "[--auto-selection minimum_abstention|quality_score]",
    "[--assignment-min-vote-margin X] [--stability-min-fraction X]",
    "[--abstain-uncertain true|false] [--observations CSV]",
    "[--grandqc-kodama-outlier-enable true|false] [--grandqc-kodama-outlier-knn N]",
    "[--grandqc-kodama-outlier-quantile X] [--grandqc-kodama-outlier-mad-multiplier X]",
    "[--grandqc-kodama-outlier-min-reference N]"
  ))
}

kodama_dir <- args[1]
out_csv <- args[2]

selected_file_dim <- 20L
cluster_representation <- "umap2d"
cluster_dimensions <- 0L
comparison_clusters <- NULL
requested_k <- 50L
cluster_algorithm <- "leiden"
target_clusters <- 0L
walktrap_clusters <- 4L
landmark_cells <- 10000L
landmark_assign_k <- 50L
landmark_sample_strategy <- "knn_inverse_distance"
landmark_density_knn_k <- 50L
landmark_density_power <- 2.0
landmark_grid_bins <- 100L
landmark_grid_max_per_bin <- 20L
walktrap_max_cells <- landmark_cells
walktrap_assign_k <- landmark_assign_k
resolution_mode <- "fixed"
fixed_resolution <- 0.3
leiden_objective <- "modularity"
resolution_grid <- c(0.005, 0.01, 0.02, 0.03, 0.04, 0.05)
score_margin <- 0.015
cluster_profile <- "standard"
clustering_seed <- 1L
stability_runs <- 3L
auto_selection <- "minimum_abstention"
assignment_min_vote_margin <- 0.10
stability_min_fraction <- 0.67
abstain_uncertain <- FALSE
observations_csv <- NULL
grandqc_kodama_outlier_enable <- TRUE
grandqc_kodama_outlier_knn <- 15L
grandqc_kodama_outlier_quantile <- 0.995
grandqc_kodama_outlier_mad_multiplier <- 6.0
grandqc_kodama_outlier_min_reference <- 50L
active_seed <- clustering_seed
fine_resolution_multiplier <- 1.35
fine_score_margin <- 0.03
fine_resolution_max <- 1.20
fine_min_cluster_increase <- 1L
silhouette_max_cells <- 4000L
merge_min_size_fraction <- 0.01
merge_min_size_floor <- 15L

if (length(args) > 2L) {
  i <- 3L
  while (i <= length(args)) {
    flag <- args[i]
    if (flag == "--cluster-representation" && i + 1L <= length(args)) {
      cluster_representation <- tolower(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--cluster-dimensions" && i + 1L <= length(args)) {
      cluster_dimensions <- suppressWarnings(as.integer(args[i + 1L]))
      i <- i + 2L
      next
    }
    if (flag == "--compare-with-clusters" && i + 1L <= length(args)) {
      comparison_clusters <- args[i + 1L]
      i <- i + 2L
      next
    }
    if (flag == "--dim" && i + 1L <= length(args)) {
      selected_file_dim <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--k" && i + 1L <= length(args)) {
      requested_k <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--algorithm" && i + 1L <= length(args)) {
      cluster_algorithm <- tolower(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--target-clusters" && i + 1L <= length(args)) {
      target_clusters <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--walktrap-clusters" && i + 1L <= length(args)) {
      walktrap_clusters <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag %in% c("--landmark-cells", "--walktrap-max-cells") && i + 1L <= length(args)) {
      landmark_cells <- as.integer(args[i + 1L])
      walktrap_max_cells <- landmark_cells
      i <- i + 2L
      next
    }
    if (flag %in% c("--landmark-assign-k", "--walktrap-assign-k") && i + 1L <= length(args)) {
      landmark_assign_k <- as.integer(args[i + 1L])
      walktrap_assign_k <- landmark_assign_k
      i <- i + 2L
      next
    }
    if (flag == "--landmark-sample-strategy" && i + 1L <= length(args)) {
      landmark_sample_strategy <- tolower(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--landmark-density-knn-k" && i + 1L <= length(args)) {
      landmark_density_knn_k <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--landmark-density-power" && i + 1L <= length(args)) {
      landmark_density_power <- as.numeric(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--landmark-grid-bins" && i + 1L <= length(args)) {
      landmark_grid_bins <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--landmark-grid-max-per-bin" && i + 1L <= length(args)) {
      landmark_grid_max_per_bin <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--resolution" && i + 1L <= length(args)) {
      value <- tolower(args[i + 1L])
      if (value == "auto") {
        resolution_mode <- "auto"
      } else {
        resolution_mode <- "fixed"
        fixed_resolution <- as.numeric(args[i + 1L])
      }
      i <- i + 2L
      next
    }
    if (flag == "--leiden-objective" && i + 1L <= length(args)) {
      leiden_objective <- tolower(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--profile" && i + 1L <= length(args)) {
      cluster_profile <- tolower(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--seed" && i + 1L <= length(args)) {
      clustering_seed <- as.integer(args[i + 1L])
      active_seed <- clustering_seed
      i <- i + 2L
      next
    }
    if (flag == "--stability-runs" && i + 1L <= length(args)) {
      stability_runs <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--auto-selection" && i + 1L <= length(args)) {
      auto_selection <- tolower(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--assignment-min-vote-margin" && i + 1L <= length(args)) {
      assignment_min_vote_margin <- as.numeric(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--stability-min-fraction" && i + 1L <= length(args)) {
      stability_min_fraction <- as.numeric(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--abstain-uncertain" && i + 1L <= length(args)) {
      abstain_uncertain <- tolower(args[i + 1L]) %in% c("true", "1", "yes", "y", "on")
      i <- i + 2L
      next
    }
    if (flag == "--observations" && i + 1L <= length(args)) {
      observations_csv <- args[i + 1L]
      i <- i + 2L
      next
    }
    if (flag == "--grandqc-kodama-outlier-enable" && i + 1L <= length(args)) {
      grandqc_kodama_outlier_enable <- tolower(args[i + 1L]) %in% c("true", "1", "yes", "y", "on")
      i <- i + 2L
      next
    }
    if (flag == "--grandqc-kodama-outlier-knn" && i + 1L <= length(args)) {
      grandqc_kodama_outlier_knn <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--grandqc-kodama-outlier-quantile" && i + 1L <= length(args)) {
      grandqc_kodama_outlier_quantile <- as.numeric(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--grandqc-kodama-outlier-mad-multiplier" && i + 1L <= length(args)) {
      grandqc_kodama_outlier_mad_multiplier <- as.numeric(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--grandqc-kodama-outlier-min-reference" && i + 1L <= length(args)) {
      grandqc_kodama_outlier_min_reference <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--fine-multiplier" && i + 1L <= length(args)) {
      fine_resolution_multiplier <- as.numeric(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--fine-score-margin" && i + 1L <= length(args)) {
      fine_score_margin <- as.numeric(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--fine-resolution-max" && i + 1L <= length(args)) {
      fine_resolution_max <- as.numeric(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--fine-min-cluster-increase" && i + 1L <= length(args)) {
      fine_min_cluster_increase <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    i <- i + 1L
  }
}

if (!cluster_representation %in% c("umap2d", "pca", "kodama_graph")) {
  stop("--cluster-representation must be umap2d, pca, or kodama_graph")
}
native_graph_mode <- identical(cluster_representation, "kodama_graph")
if (native_graph_mode) {
  if (cluster_algorithm != "leiden" || resolution_mode != "fixed" || cluster_profile != "standard")
    stop("kodama_graph currently requires Leiden, fixed resolution and profile standard")
  if (cluster_dimensions != 0L) stop("kodama_graph does not accept coordinate dimensions")
  unsupported <- c("--k", "--walktrap-clusters", "--walktrap-max-cells", "--walktrap-assign-k",
    "--landmark-assign-k", "--landmark-sample-strategy", "--landmark-density-knn-k",
    "--landmark-density-power", "--landmark-grid-bins", "--landmark-grid-max-per-bin",
    "--fine-multiplier", "--fine-score-margin", "--fine-resolution-max", "--fine-min-cluster-increase")
  if (any(args %in% unsupported)) stop("kodama_graph does not rebuild a coordinate graph or accept landmark/fine options: ", paste(intersect(args, unsupported), collapse = ", "))
  if ("--landmark-cells" %in% args && landmark_cells != 0L)
    stop("kodama_graph requires all native observations; --landmark-cells must be 0 or omitted")
  landmark_cells <- 0L
  selected_script <- sub("^--file=", "", grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1L])
  helper_directory <- dirname(normalizePath(selected_script, mustWork = TRUE))
  source(file.path(helper_directory, "kodama_graph_export.R"))
  source(file.path(helper_directory, "kodama_graph_clustering.R"))
}
if (!is.finite(cluster_dimensions) || (cluster_dimensions != 0L && cluster_dimensions < 3L)) {
  stop("--cluster-dimensions must be 0 (all saved PCA scores) or >=3")
}
if (cluster_representation == "umap2d" && cluster_dimensions != 0L) {
  stop("--cluster-dimensions applies only to pca; umap2d is the explicit two-dimensional baseline")
}
if (!is.finite(selected_file_dim) || selected_file_dim < 2L) {
  stop("--dim must be an integer >= 2.")
}
if (!is.finite(requested_k) || requested_k < 2L) {
  stop("--k must be an integer >= 2.")
}
if (!(cluster_algorithm %in% c("louvain", "leiden", "walktrap"))) {
  stop("--algorithm must be 'louvain', 'leiden', or 'walktrap'.")
}
if (!is.finite(target_clusters) || target_clusters < 0L || target_clusters == 1L) {
  stop("--target-clusters must be 0 (disabled) or an integer >= 2.")
}
if (!is.finite(walktrap_clusters) || walktrap_clusters < 2L) {
  stop("--walktrap-clusters must be an integer >= 2.")
}
if (!is.finite(walktrap_max_cells) || walktrap_max_cells < 0L) {
  stop("--walktrap-max-cells must be an integer >= 0. Use 0 for exact all-cell walktrap.")
}
if (!is.finite(walktrap_assign_k) || walktrap_assign_k < 1L) {
  stop("--walktrap-assign-k must be an integer >= 1.")
}
if (!is.finite(landmark_cells) || landmark_cells < 0L) {
  stop("--landmark-cells must be an integer >= 0. Use 0 for exact all-cell graph clustering.")
}
if (!is.finite(landmark_assign_k) || landmark_assign_k < 1L) {
  stop("--landmark-assign-k must be an integer >= 1.")
}
if (!(landmark_sample_strategy %in% c("random", "grid", "knn_inverse_distance"))) {
  stop("--landmark-sample-strategy must be 'random', 'grid', or 'knn_inverse_distance'.")
}
if (!is.finite(landmark_density_knn_k) || landmark_density_knn_k < 1L) {
  stop("--landmark-density-knn-k must be an integer >= 1.")
}
if (!is.finite(landmark_density_power) || landmark_density_power <= 0) {
  stop("--landmark-density-power must be a number > 0.")
}
if (!is.finite(landmark_grid_bins) || landmark_grid_bins < 2L) {
  stop("--landmark-grid-bins must be an integer >= 2.")
}
if (!is.finite(landmark_grid_max_per_bin) || landmark_grid_max_per_bin < 1L) {
  stop("--landmark-grid-max-per-bin must be an integer >= 1.")
}
if (resolution_mode == "fixed" && (!is.finite(fixed_resolution) || fixed_resolution <= 0)) {
  stop("--resolution must be 'auto' or a positive number.")
}
if (!(leiden_objective %in% c("modularity", "cpm"))) {
  stop("--leiden-objective must be 'modularity' or 'CPM'.")
}
if (!(cluster_profile %in% c("standard", "fine"))) {
  stop("--profile must be 'standard' or 'fine'.")
}
if (!is.finite(fine_resolution_multiplier) || fine_resolution_multiplier <= 1) {
  stop("--fine-multiplier must be a number > 1.")
}
if (!is.finite(fine_score_margin) || fine_score_margin < 0) {
  stop("--fine-score-margin must be >= 0.")
}
if (!is.finite(fine_resolution_max) || fine_resolution_max <= 0) {
  stop("--fine-resolution-max must be > 0.")
}
if (!is.finite(fine_min_cluster_increase) || fine_min_cluster_increase < 1L) {
  stop("--fine-min-cluster-increase must be an integer >= 1.")
}
if (!is.finite(clustering_seed)) {
  stop("--seed must be an integer.")
}
if (!is.finite(stability_runs) || stability_runs < 1L || stability_runs > 20L) {
  stop("--stability-runs must be between 1 and 20.")
}
if (!(auto_selection %in% c("minimum_abstention", "quality_score"))) {
  stop("--auto-selection must be 'minimum_abstention' or 'quality_score'.")
}
if (!is.finite(assignment_min_vote_margin) || assignment_min_vote_margin < 0 || assignment_min_vote_margin > 1) {
  stop("--assignment-min-vote-margin must be between 0 and 1.")
}
if (!is.finite(stability_min_fraction) || stability_min_fraction < 0 || stability_min_fraction > 1) {
  stop("--stability-min-fraction must be between 0 and 1.")
}
if (!is.finite(grandqc_kodama_outlier_knn) || grandqc_kodama_outlier_knn < 1L) {
  stop("--grandqc-kodama-outlier-knn must be an integer >= 1")
}
if (!is.finite(grandqc_kodama_outlier_quantile) || grandqc_kodama_outlier_quantile <= 0 || grandqc_kodama_outlier_quantile >= 1) {
  stop("--grandqc-kodama-outlier-quantile must be strictly between 0 and 1")
}
if (!is.finite(grandqc_kodama_outlier_mad_multiplier) || grandqc_kodama_outlier_mad_multiplier < 0) {
  stop("--grandqc-kodama-outlier-mad-multiplier must be >= 0")
}
if (!is.finite(grandqc_kodama_outlier_min_reference) || grandqc_kodama_outlier_min_reference < 3L) {
  stop("--grandqc-kodama-outlier-min-reference must be an integer >= 3")
}

require_namespace <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Required package '%s' is not installed.", pkg))
  }
}

require_namespace("igraph")
if (!native_graph_mode) {
  require_namespace("bluster")
  require_namespace("cluster")
}
if (!native_graph_mode && landmark_cells > 0L) {
  require_namespace("BiocNeighbors")
}

list_kodama_files <- function(path) {
  files <- list.files(path, pattern = "^kodama_full_[0-9]+\\.RData$", full.names = TRUE)
  if (length(files) == 0L) {
    stop(sprintf("No kodama_full_*.RData files found in: %s", path))
  }
  dims <- suppressWarnings(as.integer(sub("^.*kodama_full_([0-9]+)\\.RData$", "\\1", files)))
  if (any(is.na(dims))) {
    stop("Unable to parse dimensions from kodama_full_*.RData filenames.")
  }
  ord <- order(dims)
  list(files = files[ord], dims = dims[ord])
}

select_kodama_file <- function(target_dim, dims, files) {
  if (target_dim %in% dims) {
    idx <- which(dims == target_dim)[1]
    return(list(file = files[idx], dim = dims[idx], exact = TRUE))
  }
  lower_idx <- which(dims < target_dim)
  if (length(lower_idx) > 0L) {
    idx <- lower_idx[which.max(dims[lower_idx])]
    return(list(file = files[idx], dim = dims[idx], exact = FALSE))
  }
  idx <- which.min(dims)
  list(file = files[idx], dim = dims[idx], exact = FALSE)
}

format_cluster_sizes <- function(membership) {
  sizes <- sort(table(as.integer(membership)), decreasing = TRUE)
  paste(sprintf("%s:%d", names(sizes), as.integer(sizes)), collapse = ";")
}

format_optional_number <- function(x, digits = 6L) {
  if (is.null(x) || length(x) == 0L || !is.finite(x[1])) {
    return("NA")
  }
  sprintf(paste0("%.", as.integer(digits), "f"), as.numeric(x[1]))
}

build_resolution_grid <- function(base_grid, cluster_profile, fine_multiplier, fine_max) {
  grid <- sort(unique(as.numeric(base_grid)))
  if (cluster_profile != "fine") {
    return(grid)
  }
  extra <- c(grid * fine_multiplier, grid * fine_multiplier * fine_multiplier)
  if (fine_max > max(grid)) {
    step <- max(0.02, min(0.10, diff(range(grid)) / max(1, length(grid) - 1)))
    extra <- c(extra, seq(max(grid) * fine_multiplier, fine_max, by = step))
  }
  extra <- extra[is.finite(extra) & extra > max(grid) & extra <= fine_max]
  sort(unique(c(grid, extra)))
}

renumber_membership <- function(membership) {
  uniq <- sort(unique(as.integer(membership)))
  out <- as.integer(match(as.integer(membership), uniq))
  names(out) <- names(membership)
  out
}

adjusted_rand_index <- function(reference, candidate) {
  reference <- as.integer(reference)
  candidate <- as.integer(candidate)
  if (length(reference) != length(candidate) || length(reference) < 2L) {
    return(NA_real_)
  }
  choose_two <- function(x) x * (x - 1) / 2
  contingency <- table(reference, candidate)
  pair_total <- choose_two(length(reference))
  if (pair_total <= 0) return(NA_real_)
  observed <- sum(choose_two(contingency))
  row_pairs <- sum(choose_two(rowSums(contingency)))
  column_pairs <- sum(choose_two(colSums(contingency)))
  expected <- row_pairs * column_pairs / pair_total
  upper <- 0.5 * (row_pairs + column_pairs)
  denominator <- upper - expected
  if (abs(denominator) < .Machine$double.eps) {
    return(if (identical(reference, candidate)) 1 else 0)
  }
  as.numeric((observed - expected) / denominator)
}

align_membership_to_reference <- function(reference, candidate) {
  reference <- as.integer(reference)
  candidate <- as.integer(candidate)
  contingency <- table(candidate, reference)
  candidate_ids <- rownames(contingency)
  reference_ids <- colnames(contingency)
  pairs <- which(contingency > 0, arr.ind = TRUE)
  mapping <- setNames(rep(NA_integer_, length(candidate_ids)), candidate_ids)
  used_reference <- character(0)
  if (nrow(pairs) > 0L) {
    pair_counts <- contingency[pairs]
    pair_order <- order(-pair_counts, pairs[, 1], pairs[, 2])
    for (pair_index in pair_order) {
      candidate_id <- candidate_ids[pairs[pair_index, 1]]
      reference_id <- reference_ids[pairs[pair_index, 2]]
      if (is.na(mapping[candidate_id]) && !(reference_id %in% used_reference)) {
        mapping[candidate_id] <- as.integer(reference_id)
        used_reference <- c(used_reference, reference_id)
      }
    }
  }
  for (candidate_id in names(mapping)[is.na(mapping)]) {
    mapping[candidate_id] <- as.integer(reference_ids[which.max(contingency[candidate_id, ])])
  }
  unname(as.integer(mapping[as.character(candidate)]))
}

mean_silhouette <- function(vis, membership) {
  if (nrow(vis) < 4L || length(unique(membership)) < 2L) {
    return(NA_real_)
  }
  idx <- seq_len(nrow(vis))
  if (nrow(vis) > silhouette_max_cells) {
    set.seed(active_seed)
    idx <- sort(sample(idx, silhouette_max_cells))
  }
  sil <- tryCatch(
    cluster::silhouette(as.integer(membership[idx]), stats::dist(if (cluster_representation == "pca") vis[idx, , drop = FALSE] else scale(vis[idx, , drop = FALSE]))),
    error = function(e) NULL
  )
  if (is.null(sil)) return(NA_real_)
  summary(sil)$avg.width
}

cluster_centroids <- function(vis, membership) {
  ids <- sort(unique(as.integer(membership)))
  out <- do.call(rbind, lapply(ids, function(id) colMeans(vis[membership == id, , drop = FALSE])))
  rownames(out) <- as.character(ids)
  out
}

collapse_clusters_to_target <- function(vis, membership, target) {
  collapsed <- renumber_membership(membership)
  initial_count <- length(unique(collapsed))
  if (target == 0L || initial_count == target) {
    return(list(
      membership = collapsed,
      applied = FALSE,
      initial_count = initial_count,
      merge_history = ""
    ))
  }
  if (initial_count < target) {
    stop(sprintf(
      "Cannot force %d clusters because %s produced only %d clusters.",
      target,
      cluster_algorithm,
      initial_count
    ))
  }

  merge_history <- character(0)
  while (length(unique(collapsed)) > target) {
    centers <- cluster_centroids(vis, collapsed)
    center_distances <- as.matrix(stats::dist(centers))
    diag(center_distances) <- Inf
    nearest <- which(
      center_distances == min(center_distances),
      arr.ind = TRUE
    )
    nearest <- nearest[nearest[, 1] < nearest[, 2], , drop = FALSE]
    nearest <- nearest[order(nearest[, 1], nearest[, 2]), , drop = FALSE]
    keep_id <- as.integer(rownames(centers)[nearest[1, 1]])
    drop_id <- as.integer(rownames(centers)[nearest[1, 2]])
    merge_distance <- center_distances[nearest[1, 1], nearest[1, 2]]
    merge_history <- c(
      merge_history,
      sprintf("%d+%d@%.6f", keep_id, drop_id, merge_distance)
    )
    collapsed[collapsed == drop_id] <- keep_id
    collapsed <- renumber_membership(collapsed)
  }

  list(
    membership = collapsed,
    applied = TRUE,
    initial_count = initial_count,
    merge_history = paste(merge_history, collapse = ";")
  )
}

merge_small_clusters <- function(vis, membership, min_size) {
  merged <- as.integer(membership)
  names(merged) <- rownames(vis)
  if (length(unique(merged)) < 2L) {
    return(list(membership = renumber_membership(merged), merged_any = FALSE))
  }

  merged_any <- FALSE
  repeat {
    sizes <- table(merged)
    if (length(sizes) < 2L) break
    small_ids <- names(sizes)[sizes < min_size]
    if (length(small_ids) == 0L) break

    small_ids <- small_ids[order(as.integer(sizes[small_ids]))]
    changed_this_round <- FALSE

    for (small_id in small_ids) {
      sizes <- table(merged)
      if (!(small_id %in% names(sizes))) next
      if (sizes[[small_id]] >= min_size) next
      if (length(sizes) < 2L) break

      candidate_ids <- names(sizes)[names(sizes) != small_id & sizes >= min_size]
      if (length(candidate_ids) == 0L) {
        candidate_ids <- names(sizes)[names(sizes) != small_id]
      }
      if (length(candidate_ids) == 0L) next

      small_center <- colMeans(vis[merged == as.integer(small_id), , drop = FALSE])
      centers <- cluster_centroids(vis, merged)
      centers <- centers[rownames(centers) %in% candidate_ids, , drop = FALSE]
      if (nrow(centers) == 0L) next

      diffs <- centers - matrix(small_center, nrow = nrow(centers), ncol = ncol(centers), byrow = TRUE)
      nearest_id <- rownames(centers)[which.min(rowSums(diffs * diffs))]
      merged[merged == as.integer(small_id)] <- as.integer(nearest_id)
      merged_any <- TRUE
      changed_this_round <- TRUE
    }

    if (!changed_this_round) break
  }

  list(membership = renumber_membership(merged), merged_any = merged_any)
}

run_louvain <- function(graph_obj, vis, resolution_value, merge_min_size, algorithm = "louvain", leiden_objective = "modularity") {
  set.seed(active_seed)
  if (algorithm == "leiden") {
    objective <- if (identical(tolower(leiden_objective), "cpm")) "CPM" else "modularity"
    cl <- igraph::cluster_leiden(graph_obj, objective_function = objective, resolution = resolution_value)
  } else {
    cl <- igraph::cluster_louvain(graph_obj, resolution = resolution_value)
  }
  membership <- renumber_membership(as.integer(cl$membership))
  sizes <- table(membership)
  tiny_threshold <- max(merge_min_size_floor, ceiling(merge_min_size_fraction * sum(sizes)))
  tiny_count <- sum(as.integer(sizes) < tiny_threshold)
  tiny_fraction <- if (length(sizes) == 0L) 0 else sum(as.integer(sizes)[as.integer(sizes) < tiny_threshold]) / sum(sizes)
  merged <- merge_small_clusters(vis, membership, merge_min_size)
  final_membership <- renumber_membership(merged$membership)
  list(
    resolution = resolution_value,
    membership = membership,
    final_membership = final_membership,
    raw_cluster_count = length(sizes),
    final_cluster_count = length(unique(final_membership)),
    modularity = igraph::modularity(graph_obj, membership = membership),
    silhouette = NA_real_,
    tiny_count = tiny_count,
    tiny_fraction = tiny_fraction,
    tiny_threshold = tiny_threshold,
    score = NA_real_
  )
}

vote_membership <- function(nn_index, ref_membership) {
  votes <- matrix(ref_membership[as.vector(nn_index)], nrow = nrow(nn_index), ncol = ncol(nn_index))
  cluster_ids <- sort(unique(as.integer(ref_membership)))
  vote_counts <- vapply(cluster_ids, function(cluster_id) rowSums(votes == cluster_id), numeric(nrow(votes)))
  if (is.null(dim(vote_counts))) {
    vote_counts <- matrix(vote_counts, ncol = 1L)
  }
  winner_col <- max.col(vote_counts, ties.method = "first")
  winner_count <- vote_counts[cbind(seq_len(nrow(vote_counts)), winner_col)]
  second_count <- rep(0, nrow(vote_counts))
  if (ncol(vote_counts) > 1L) {
    for (cluster_col in seq_len(ncol(vote_counts))) {
      eligible <- winner_col != cluster_col
      second_count[eligible] <- pmax(second_count[eligible], vote_counts[eligible, cluster_col])
    }
  }
  vote_total <- pmax(1, rowSums(vote_counts))
  list(
    membership = as.integer(cluster_ids[winner_col]),
    vote_fraction = as.numeric(winner_count / vote_total),
    vote_margin = as.numeric((winner_count - second_count) / vote_total)
  )
}

knn_query_indices <- function(data, query, k) {
  data <- as.matrix(data)
  query <- as.matrix(query)
  k <- max(1L, min(as.integer(k), nrow(data)))

  if (requireNamespace("Rnanoflann", quietly = TRUE)) {
    nn <- Rnanoflann::nn(data = data, points = query, k = k, method = "euclidean", search = "standard", trans = TRUE)
    return(list(index = matrix(as.integer(nn$indices), nrow = nrow(query), ncol = k), backend = "Rnanoflann"))
  }

  nn <- BiocNeighbors::queryKNN(
    X = data,
    query = query,
    k = k,
    BNPARAM = BiocNeighbors::KmknnParam(),
    num.threads = 1L
  )
  list(index = nn$index, backend = "BiocNeighbors")
}

knn_self_mean_distance <- function(x, k) {
  x <- as.matrix(x)
  k <- max(1L, min(as.integer(k), nrow(x) - 1L))

  if (requireNamespace("Rnanoflann", quietly = TRUE)) {
    nn <- Rnanoflann::nn(data = x, points = x, k = k + 1L, method = "euclidean", search = "standard", trans = TRUE)
    distances <- matrix(as.numeric(nn$distances), nrow = nrow(x), ncol = k + 1L)
    indices <- matrix(as.integer(nn$indices), nrow = nrow(x), ncol = k + 1L)
    # Rnanoflann returns each query point itself first for self-search; drop that zero-distance column.
    if (all(indices[, 1L] == seq_len(nrow(x))) || all(distances[, 1L] == 0)) {
      distances <- distances[, -1L, drop = FALSE]
    } else {
      distances <- distances[, seq_len(k), drop = FALSE]
    }
    return(list(mean_dist = rowMeans(distances), backend = "Rnanoflann"))
  }

  nn <- BiocNeighbors::findKNN(
    X = x,
    k = k,
    BNPARAM = BiocNeighbors::KmknnParam(),
    num.threads = 1L
  )
  list(mean_dist = rowMeans(nn$distance), backend = "BiocNeighbors")
}

assign_from_landmarks <- function(vis, landmark_vis, landmark_membership, landmark_idx, assign_k, label) {
  used_assign_k <- max(1L, min(as.integer(assign_k), nrow(landmark_vis)))
  cat(sprintf("[INFO] Assigning all %d cells from %s landmarks by %d-NN vote in %s feature space (%d dimensions)\n", nrow(vis), label, used_assign_k, cluster_representation, ncol(vis)))
  flush.console()
  nn <- knn_query_indices(landmark_vis, vis, used_assign_k)
  cat(sprintf("[INFO] Landmark assignment KNN backend: %s\n", nn$backend))
  flush.console()
  voted <- vote_membership(nn$index, landmark_membership)
  membership <- voted$membership
  vote_fraction <- voted$vote_fraction
  vote_margin <- voted$vote_margin
  membership[landmark_idx] <- landmark_membership
  vote_fraction[landmark_idx] <- 1
  vote_margin[landmark_idx] <- 1
  names(membership) <- rownames(vis)
  membership <- renumber_membership(membership)
  backend <- nn$backend
  rm(nn)
  gc(FALSE)
  list(
    membership = membership,
    assign_k = used_assign_k,
    knn_backend = backend,
    vote_fraction = vote_fraction,
    vote_margin = vote_margin
  )
}

knn_inverse_distance_sample <- function(vis, max_total, density_k, density_power) {
  max_total <- min(as.integer(max_total), nrow(vis))
  if (max_total <= 0L || nrow(vis) <= max_total) {
    return(seq_len(nrow(vis)))
  }
  used_density_k <- max(1L, min(as.integer(density_k), nrow(vis) - 1L))
  density_power <- as.numeric(density_power)
  density_reference_cells <- suppressWarnings(as.integer(Sys.getenv("KODAMA_LANDMARK_DENSITY_REFERENCE_CELLS", "50000")))
  if (!is.finite(density_reference_cells) || density_reference_cells <= 0L) {
    density_reference_cells <- nrow(vis)
  }
  if (nrow(vis) > density_reference_cells) {
    set.seed(active_seed)
    reference_idx <- sort(sample.int(nrow(vis), density_reference_cells))
    density_vis <- vis[reference_idx, , drop = FALSE]
    used_density_k <- max(1L, min(as.integer(density_k), nrow(density_vis) - 1L))
    cat(sprintf(
      "[INFO] Estimating inverse-distance landmark density on %d/%d reference cells using mean %d-NN distance\n",
      nrow(density_vis),
      nrow(vis),
      used_density_k
    ))
    flush.console()
  } else {
    reference_idx <- seq_len(nrow(vis))
    density_vis <- vis
  }
  cat(sprintf(
    "[INFO] Selecting %d inverse-distance landmarks from %d candidate cells using mean %d-NN distance and p=%.3f\n",
    max_total,
    nrow(density_vis),
    used_density_k,
    density_power
  ))
  flush.console()
  nn <- knn_self_mean_distance(density_vis, used_density_k)
  mean_dist <- nn$mean_dist
  knn_backend <- nn$backend
  cat(sprintf("[INFO] Inverse-distance landmark KNN backend: %s\n", knn_backend))
  flush.console()
  finite_positive <- mean_dist[is.finite(mean_dist) & mean_dist > 0]
  distance_floor <- if (length(finite_positive)) {
    as.numeric(stats::quantile(finite_positive, probs = 0.001, names = FALSE, na.rm = TRUE))
  } else {
    .Machine$double.eps
  }
  if (!is.finite(distance_floor) || distance_floor <= 0) {
    distance_floor <- .Machine$double.eps
  }
  weights <- (1 / pmax(mean_dist, distance_floor))^density_power
  weights[!is.finite(weights)] <- 0
  max_weight <- suppressWarnings(max(weights, na.rm = TRUE))
  if (!is.finite(max_weight) || max_weight <= 0 || sum(weights) <= 0) {
    warning("Inverse-distance landmark weights were invalid; falling back to random landmark sampling.")
    idx_local <- sort(sample.int(nrow(density_vis), max_total))
  } else {
    weights <- weights / max_weight
    idx_local <- sort(sample.int(nrow(density_vis), size = max_total, replace = FALSE, prob = weights))
  }
  idx <- sort(reference_idx[idx_local])
  attr(idx, "density_k_used") <- used_density_k
  attr(idx, "density_power") <- density_power
  attr(idx, "density_knn_backend") <- knn_backend
  attr(idx, "density_reference_cells") <- length(reference_idx)
  attr(idx, "mean_density_distance_median_all") <- stats::median(mean_dist, na.rm = TRUE)
  attr(idx, "mean_density_distance_median_selected") <- stats::median(mean_dist[idx_local], na.rm = TRUE)
  rm(nn, weights, mean_dist, density_vis, reference_idx, idx_local)
  gc(FALSE)
  idx
}

select_landmarks <- function(vis, max_cells, sample_strategy, grid_bins, grid_max_per_bin, density_k, density_power) {
  max_cells <- min(as.integer(max_cells), nrow(vis))
  if (max_cells <= 0L || nrow(vis) <= max_cells) {
    return(seq_len(nrow(vis)))
  }
  if (sample_strategy == "random") {
    return(sort(sample.int(nrow(vis), max_cells)))
  }
  if (sample_strategy == "knn_inverse_distance") {
    return(knn_inverse_distance_sample(vis, max_cells, density_k, density_power))
  }
  idx <- grid_balanced_sample(
    vis,
    bins = as.integer(grid_bins),
    max_per_bin = as.integer(grid_max_per_bin),
    max_total = max_cells
  )
  if (length(idx) < min(max_cells, 1000L)) {
    warning(sprintf("Grid-balanced landmark sample produced only %d cells; falling back to random sample.", length(idx)))
    idx <- sort(sample.int(nrow(vis), max_cells))
  }
  idx
}

run_walktrap_fixed <- function(vis, actual_k, n_clusters, max_cells, assign_k, sample_strategy, grid_bins, grid_max_per_bin, density_k, density_power) {
  set.seed(active_seed)
  n_cells <- nrow(vis)
  n_clusters <- as.integer(n_clusters)
  max_cells <- as.integer(max_cells)

  if (max_cells == 0L || n_cells <= max_cells) {
    graph_k <- max(2L, min(as.integer(actual_k), n_cells - 1L))
    cat(sprintf("[INFO] Walktrap mode: exact all-cell SNN graph with %d cells and k=%d\n", n_cells, graph_k))
    flush.console()
    graph_obj <- bluster::makeSNNGraph(as.matrix(vis), k = graph_k)
    g_walk <- igraph::cluster_walktrap(graph_obj)
    membership <- renumber_membership(as.integer(igraph::cut_at(g_walk, no = n_clusters)))
    modularity <- igraph::modularity(graph_obj, membership = membership)
    graph_cells <- n_cells
    assignment_mode <- "exact"
    used_assign_k <- 0L
    assignment_knn_backend <- NA_character_
    assignment_vote_fraction <- rep(1, n_cells)
    assignment_vote_margin <- rep(1, n_cells)
  } else {
    landmark_idx <- select_landmarks(vis, max_cells, sample_strategy, grid_bins, grid_max_per_bin, density_k, density_power)
    landmark_vis <- vis[landmark_idx, , drop = FALSE]
    graph_k <- max(2L, min(as.integer(actual_k), nrow(landmark_vis) - 1L))
    cat(sprintf(
      "[INFO] Walktrap mode: %s landmark SNN graph with %d/%d cells, graph k=%d, target clusters=%d\n",
      sample_strategy,
      nrow(landmark_vis),
      n_cells,
      graph_k,
      n_clusters
    ))
    flush.console()
    graph_obj <- bluster::makeSNNGraph(as.matrix(landmark_vis), k = graph_k)
    g_walk <- igraph::cluster_walktrap(graph_obj)
    landmark_membership <- renumber_membership(as.integer(igraph::cut_at(g_walk, no = n_clusters)))
    names(landmark_membership) <- rownames(landmark_vis)
    modularity <- igraph::modularity(graph_obj, membership = landmark_membership)

    assigned <- assign_from_landmarks(vis, landmark_vis, landmark_membership, landmark_idx, assign_k, "walktrap")
    membership <- assigned$membership
    used_assign_k <- assigned$assign_k
    assignment_knn_backend <- assigned$knn_backend
    assignment_vote_fraction <- assigned$vote_fraction
    assignment_vote_margin <- assigned$vote_margin
    graph_cells <- nrow(landmark_vis)
    assignment_mode <- "landmark_knn"
    rm(assigned, landmark_vis, landmark_membership)
    gc(FALSE)
  }

  list(
    resolution = NA_real_,
    membership = membership,
    final_membership = membership,
    raw_cluster_count = length(unique(membership)),
    final_cluster_count = length(unique(membership)),
    modularity = modularity,
    silhouette = mean_silhouette(vis, membership),
    tiny_count = 0L,
    tiny_fraction = 0,
    tiny_threshold = 0L,
    score = NA_real_,
    walktrap_cells_used = as.integer(graph_cells),
    walktrap_assignment_mode = assignment_mode,
    walktrap_assign_k_used = as.integer(used_assign_k),
    landmark_algorithm = "walktrap",
    landmark_cells_used = as.integer(graph_cells),
    landmark_assignment_mode = assignment_mode,
    landmark_assign_k_used = as.integer(used_assign_k),
    landmark_assignment_knn_backend = as.character(assignment_knn_backend),
    assignment_vote_fraction = assignment_vote_fraction,
    assignment_vote_margin = assignment_vote_margin
  )
}

grid_balanced_sample <- function(vis, bins, max_per_bin, max_total) {
  xr <- range(vis[, 1], finite = TRUE)
  yr <- range(vis[, 2], finite = TRUE)
  xb <- cut(vis[, 1], breaks = seq(xr[1], xr[2], length.out = bins + 1L), include.lowest = TRUE, labels = FALSE)
  yb <- cut(vis[, 2], breaks = seq(yr[1], yr[2], length.out = bins + 1L), include.lowest = TRUE, labels = FALSE)
  bin_id <- xb + bins * (yb - 1L)
  by_bin <- split(seq_len(nrow(vis)), bin_id)
  idx <- unlist(lapply(by_bin, function(ii) {
    if (length(ii) <= max_per_bin) ii else sample(ii, max_per_bin)
  }), use.names = FALSE)
  idx <- sort(unique(idx))
  if (length(idx) > max_total) {
    idx <- sort(sample(idx, max_total))
  }
  idx
}

cluster_plot_colors <- function(vis, membership, algorithm) {
  cluster_ids <- sort(unique(as.integer(membership)))
  palette_hex <- c(
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#D55E00", "#56B4E9", "#F0E442", "#000000",
    "#332288", "#88CCEE", "#44AA99", "#117733",
    "#999933", "#DDCC77", "#CC6677", "#882255"
  )
  color_index <- ((cluster_ids - 1L) %% length(palette_hex)) + 1L
  setNames(palette_hex[color_index], as.character(cluster_ids))
}

draw_membership_plot <- function(vis, membership, cluster_colors, algorithm, is_abstained = NULL, main = NULL) {
  set.seed(clustering_seed)
  plot_max_points <- 350000L
  plot_idx <- if (nrow(vis) > plot_max_points) sort(sample.int(nrow(vis), plot_max_points)) else seq_len(nrow(vis))
  accepted <- if (is.null(is_abstained)) {
    rep(TRUE, nrow(vis))
  } else {
    !as.logical(is_abstained)
  }
  plot(vis[plot_idx, 1], vis[plot_idx, 2],
    pch = 16,
    cex = 0.18,
    col = grDevices::adjustcolor("#BDBDBD", alpha.f = 0.45),
    xlab = "KODAMA dimension 1",
    ylab = "KODAMA dimension 2",
    main = main
  )
  for (cluster_id in names(cluster_colors)) {
    idx <- plot_idx[
      membership[plot_idx] == as.integer(cluster_id) & accepted[plot_idx]
    ]
    if (!length(idx)) next
    color <- cluster_colors[cluster_id]
    points(vis[idx, 1], vis[idx, 2], pch = 16, cex = 0.16, col = grDevices::adjustcolor(color, alpha.f = 0.68))
  }
  abstained_idx <- plot_idx[!accepted[plot_idx]]
  if (length(abstained_idx)) {
    points(
      vis[abstained_idx, 1], vis[abstained_idx, 2],
      pch = 16, cex = 0.20, col = grDevices::adjustcolor("#6F6F6F", alpha.f = 0.85)
    )
  }
  accepted_counts <- vapply(
    as.integer(names(cluster_colors)),
    function(cluster_id) sum(membership == cluster_id & accepted),
    integer(1)
  )
  legend_labels <- sprintf("cluster %s accepted n=%d", names(cluster_colors), accepted_counts)
  legend_colors <- unname(cluster_colors)
  if (any(!accepted)) {
    legend_labels <- c(legend_labels, sprintf("abstained n=%d", sum(!accepted)))
    legend_colors <- c(legend_colors, "#6F6F6F")
  }
  legend("topright",
    legend = legend_labels,
    col = legend_colors,
    pch = 16,
    bty = "n",
    cex = 0.72
  )
}

draw_uncertainty_plot <- function(vis, uncertainty_reason, is_abstained, assignment_vote_margin, stability_fraction, main = NULL) {
  set.seed(clustering_seed)
  plot_max_points <- 350000L
  plot_idx <- if (nrow(vis) > plot_max_points) sort(sample.int(nrow(vis), plot_max_points)) else seq_len(nrow(vis))
  status <- as.character(uncertainty_reason)
  status_levels <- c(
    "none",
    "ambiguous_assignment",
    "seed_instability",
    "ambiguous_assignment_and_seed_instability",
    "grandqc_artifact_kodama_outlier"
  )
  status_colors <- c("#BDBDBD", "#E69F00", "#56B4E9", "#CC79A7", "#D73027")
  status_labels <- c("no threshold violation", "ambiguous assignment", "seed instability", "assignment + instability", "GrandQC candidate + KODAMA outlier")
  status[!(status %in% status_levels)] <- "other"
  status_levels <- c(status_levels, "other")
  status_colors <- c(status_colors, "#333333")
  status_labels <- c(status_labels, "other uncertainty")

  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  par(mfrow = c(1, 2), mar = c(4.4, 4.4, 3.4, 1.0), oma = c(0, 0, 1.5, 0))
  plot(
    vis[plot_idx, 1], vis[plot_idx, 2], pch = 16, cex = 0.18,
    col = grDevices::adjustcolor("#D0D0D0", alpha.f = 0.45),
    xlab = "KODAMA dimension 1", ylab = "KODAMA dimension 2",
    main = sprintf("Uncertainty reason | abstained n=%d", sum(as.logical(is_abstained)))
  )
  for (idx_status in seq_along(status_levels)) {
    selected <- plot_idx[status[plot_idx] == status_levels[idx_status]]
    if (!length(selected)) next
    point_size <- if (status_levels[idx_status] == "none") 0.13 else 0.24
    alpha <- if (status_levels[idx_status] == "none") 0.35 else 0.88
    points(
      vis[selected, 1], vis[selected, 2], pch = 16, cex = point_size,
      col = grDevices::adjustcolor(status_colors[idx_status], alpha.f = alpha)
    )
  }
  present <- vapply(status_levels, function(level) any(status == level), logical(1))
  legend(
    "topright", legend = sprintf("%s n=%d", status_labels[present], vapply(status_levels[present], function(level) sum(status == level), integer(1))),
    col = status_colors[present], pch = 16, bty = "n", cex = 0.68
  )
  combined_confidence <- pmin(
    pmax(0, pmin(1, as.numeric(assignment_vote_margin))),
    pmax(0, pmin(1, as.numeric(stability_fraction)))
  )
  combined_confidence[!is.finite(combined_confidence)] <- 0
  breaks <- seq(0, 1, length.out = 101L)
  confidence_palette <- grDevices::colorRampPalette(c("#D73027", "#FEE08B", "#1A9850"))(100L)
  color_index <- pmax(1L, pmin(100L, findInterval(combined_confidence[plot_idx], breaks, all.inside = TRUE)))
  plot(
    vis[plot_idx, 1], vis[plot_idx, 2], pch = 16, cex = 0.18,
    col = grDevices::adjustcolor(confidence_palette[color_index], alpha.f = 0.72),
    xlab = "KODAMA dimension 1", ylab = "KODAMA dimension 2",
    main = "Minimum assignment/stability score"
  )
  legend(
    "topright", legend = c("lower confidence", "higher confidence"),
    col = c(confidence_palette[1], confidence_palette[100]), pch = 16, bty = "n", cex = 0.68
  )
  if (!is.null(main)) mtext(main, side = 3, outer = TRUE, line = 0.2, cex = 0.9)
}

detect_grandqc_kodama_outliers <- function(
  kodama_plot, membership, artifact_candidate, requested_k, threshold_quantile,
  mad_multiplier, min_reference
) {
  n <- nrow(kodama_plot)
  result <- list(
    is_outlier = rep(FALSE, n),
    mean_knn_distance = rep(NA_real_, n),
    cluster_threshold = rep(NA_real_, n),
    score = rep(NA_real_, n),
    reference_count = rep(0L, n)
  )
  if (!any(artifact_candidate)) return(result)
  require_namespace("BiocNeighbors")
  for (cluster_id in sort(unique(as.integer(membership)))) {
    cluster_idx <- which(as.integer(membership) == cluster_id)
    candidate_idx <- cluster_idx[artifact_candidate[cluster_idx]]
    reference_idx <- cluster_idx[!artifact_candidate[cluster_idx]]
    if (!length(candidate_idx)) next
    result$reference_count[candidate_idx] <- length(reference_idx)
    # Fail open when GrandQC marked nearly the complete cluster: KODAMA has no
    # adequate normal-tissue reference from which to establish an outlier.
    if (length(reference_idx) < min_reference) next
    reference <- kodama_plot[reference_idx, , drop = FALSE]
    candidate <- kodama_plot[candidate_idx, , drop = FALSE]
    k_reference <- min(as.integer(requested_k), nrow(reference) - 1L)
    k_query <- min(as.integer(requested_k), nrow(reference))
    if (k_reference < 1L || k_query < 1L) next
    reference_distance <- rowMeans(BiocNeighbors::findKNN(reference, k = k_reference)$distance)
    candidate_distance <- rowMeans(BiocNeighbors::queryKNN(reference, candidate, k = k_query)$distance)
    robust_limit <- stats::median(reference_distance) + mad_multiplier * stats::mad(reference_distance)
    quantile_limit <- as.numeric(stats::quantile(reference_distance, probs = threshold_quantile, names = FALSE, type = 8))
    threshold <- max(robust_limit, quantile_limit, .Machine$double.eps, na.rm = TRUE)
    result$mean_knn_distance[candidate_idx] <- candidate_distance
    result$cluster_threshold[candidate_idx] <- threshold
    result$score[candidate_idx] <- candidate_distance / threshold
    result$is_outlier[candidate_idx] <- is.finite(candidate_distance) & candidate_distance > threshold
  }
  result
}

preferred_cluster_cap <- function(n_cells) {
  max(4L, min(25L, as.integer(round(sqrt(max(1L, n_cells)) * 0.75))))
}

prefer_minimum_abstention <- function(evals) {
  abstained <- vapply(evals, function(x) as.integer(x$estimated_abstained_count %||% .Machine$integer.max), integer(1))
  evals <- evals[abstained == min(abstained)]

  mean_ari <- vapply(evals, function(x) if (is.finite(x$candidate_mean_ari %||% NA_real_)) x$candidate_mean_ari else -1, numeric(1))
  evals <- evals[mean_ari == max(mean_ari)]

  min_ari <- vapply(evals, function(x) if (is.finite(x$candidate_min_ari %||% NA_real_)) x$candidate_min_ari else -1, numeric(1))
  evals <- evals[min_ari == max(min_ari)]

  tiny_fraction <- vapply(evals, function(x) x$tiny_fraction, numeric(1))
  evals <- evals[tiny_fraction == min(tiny_fraction)]

  tiny_count <- vapply(evals, function(x) x$tiny_count, integer(1))
  evals <- evals[tiny_count == min(tiny_count)]

  sil <- vapply(evals, function(x) if (is.finite(x$silhouette)) x$silhouette else -1, numeric(1))
  evals <- evals[sil == max(sil)]

  modularity <- vapply(evals, function(x) x$modularity, numeric(1))
  evals <- evals[modularity == max(modularity)]

  counts <- vapply(evals, function(x) x$final_cluster_count, integer(1))
  evals[[which.min(counts)]]
}

select_auto_best <- function(evals, cluster_cap, score_margin, selection_mode) {
  scores <- vapply(evals, function(x) x$score, numeric(1))
  best_score <- max(scores)
  if (identical(selection_mode, "minimum_abstention")) {
    eligible <- evals[vapply(evals, function(x) x$final_cluster_count >= 2L && x$final_cluster_count <= cluster_cap, logical(1))]
    if (length(eligible) == 0L) eligible <- evals
    return(prefer_minimum_abstention(eligible))
  }

  near_best_idx <- which(scores >= (best_score - score_margin))
  near_best <- evals[near_best_idx]

  valid_count <- vapply(near_best, function(x) x$final_cluster_count >= 2L && x$final_cluster_count <= cluster_cap, logical(1))
  if (any(valid_count)) near_best <- near_best[valid_count]

  near_counts <- vapply(near_best, function(x) x$final_cluster_count, integer(1))
  near_best <- near_best[near_counts == min(near_counts)]

  near_tiny_fraction <- vapply(near_best, function(x) x$tiny_fraction, numeric(1))
  near_best <- near_best[near_tiny_fraction == min(near_tiny_fraction)]

  near_tiny_count <- vapply(near_best, function(x) x$tiny_count, integer(1))
  near_best <- near_best[near_tiny_count == min(near_tiny_count)]

  near_sil <- vapply(near_best, function(x) if (is.finite(x$silhouette)) x$silhouette else -1, numeric(1))
  near_best <- near_best[near_sil == max(near_sil)]

  near_best[[which.max(vapply(near_best, function(x) x$modularity, numeric(1)))]]
}

select_auto_fine <- function(evals, base_best, fine_score_margin, fine_min_cluster_increase, selection_mode) {
  min_target <- base_best$final_cluster_count + fine_min_cluster_increase
  more <- evals[vapply(
    evals,
    function(x) x$final_cluster_count >= min_target && x$score >= (base_best$score - fine_score_margin),
    logical(1)
  )]
  if (length(more) == 0L) {
    more <- evals[vapply(
      evals,
      function(x) x$final_cluster_count > base_best$final_cluster_count,
      logical(1)
    )]
  }
  if (length(more) == 0L) {
    more <- evals[vapply(evals, function(x) x$resolution > base_best$resolution, logical(1))]
  }
  if (length(more) == 0L) {
    return(base_best)
  }

  if (identical(selection_mode, "minimum_abstention")) {
    return(prefer_minimum_abstention(more))
  }

  counts <- vapply(more, function(x) x$final_cluster_count, integer(1))
  eligible_counts <- counts[counts >= min_target]
  if (length(eligible_counts) > 0L) {
    target_count <- min(eligible_counts)
    more <- more[counts == target_count]
  } else {
    more <- more[counts == max(counts)]
  }

  tiny_fraction <- vapply(more, function(x) x$tiny_fraction, numeric(1))
  more <- more[tiny_fraction == min(tiny_fraction)]

  tiny_count <- vapply(more, function(x) x$tiny_count, integer(1))
  more <- more[tiny_count == min(tiny_count)]

  sil <- vapply(more, function(x) if (is.finite(x$silhouette)) x$silhouette else -1, numeric(1))
  more <- more[sil == max(sil)]

  more[[which.max(vapply(more, function(x) x$modularity, numeric(1)))]]
}

run_louvain_landmark <- function(
  vis,
  actual_k,
  cluster_algorithm,
  leiden_objective,
  resolution_mode,
  fixed_resolution,
  resolution_grid_eval,
  cluster_profile,
  fine_score_margin,
  fine_min_cluster_increase,
  max_cells,
  assign_k,
  sample_strategy,
  grid_bins,
  grid_max_per_bin,
  density_k,
  density_power
) {
  set.seed(active_seed)
  n_cells <- nrow(vis)
  if (max_cells == 0L || n_cells <= max_cells) {
    landmark_idx <- seq_len(n_cells)
    assignment_mode <- "exact"
  } else {
    landmark_idx <- select_landmarks(vis, max_cells, sample_strategy, grid_bins, grid_max_per_bin, density_k, density_power)
    assignment_mode <- "landmark_knn"
  }
  sampling_density_k_used <- attr(landmark_idx, "density_k_used") %||% NA_integer_
  sampling_density_power <- attr(landmark_idx, "density_power") %||% NA_real_
  sampling_density_knn_backend <- attr(landmark_idx, "density_knn_backend") %||% NA_character_
  sampling_density_median_all <- attr(landmark_idx, "mean_density_distance_median_all") %||% NA_real_
  sampling_density_median_selected <- attr(landmark_idx, "mean_density_distance_median_selected") %||% NA_real_
  landmark_vis <- vis[landmark_idx, , drop = FALSE]
  graph_k <- max(2L, min(as.integer(actual_k), nrow(landmark_vis) - 1L))
  landmark_merge_min_size <- max(merge_min_size_floor, ceiling(merge_min_size_fraction * nrow(landmark_vis)))

  cat(sprintf(
    "[INFO] %s mode: %s SNN graph with %d/%d cells, graph k=%d, profile=%s\n",
    tools::toTitleCase(cluster_algorithm),
    ifelse(assignment_mode == "exact", "exact all-cell", paste(sample_strategy, "landmark")),
    nrow(landmark_vis),
    n_cells,
    graph_k,
    cluster_profile
  ))
  flush.console()
  graph_obj <- bluster::makeSNNGraph(as.matrix(landmark_vis), k = graph_k)

  if (resolution_mode == "auto") {
    cluster_cap <- preferred_cluster_cap(nrow(landmark_vis))
    evals <- lapply(resolution_grid_eval, function(res) {
      out <- run_louvain(graph_obj, landmark_vis, res, landmark_merge_min_size, cluster_algorithm, leiden_objective)
      out$silhouette <- mean_silhouette(landmark_vis, out$membership)
      sil_term <- if (is.finite(out$silhouette)) out$silhouette else -1
      over_cap <- max(0L, out$final_cluster_count - cluster_cap)
      out$score <- sil_term +
        0.20 * out$modularity -
        0.06 * out$final_cluster_count -
        0.60 * out$tiny_fraction -
        0.05 * out$tiny_count -
        0.12 * over_cap
      out$cluster_cap <- cluster_cap
      out
    })

    # Estimate the exact downstream abstention rule for every resolution on a
    # common landmark graph.  This keeps resolution comparisons paired: graph,
    # landmarks and all-cell KNN neighbours are identical, while Leiden is
    # repeated across the configured seeds.  The final selected resolution is
    # subsequently re-evaluated by the broader stability pass, which also
    # resamples landmarks.
    selection_seed <- as.integer(active_seed)
    candidate_nn <- NULL
    candidate_assign_k <- 0L
    if (assignment_mode != "exact") {
      candidate_assign_k <- max(1L, min(as.integer(assign_k), nrow(landmark_vis)))
      candidate_nn <- knn_query_indices(landmark_vis, vis, candidate_assign_k)
      cat(sprintf(
        "[INFO] Auto-resolution abstention study uses one paired %d-NN query (%s) for all candidates\n",
        candidate_assign_k,
        candidate_nn$backend
      ))
    }
    materialize_candidate <- function(landmark_membership) {
      landmark_membership <- renumber_membership(landmark_membership)
      if (assignment_mode == "exact") {
        membership <- landmark_membership
        vote_margin <- rep(1, n_cells)
      } else {
        voted <- vote_membership(candidate_nn$index, landmark_membership)
        membership <- voted$membership
        vote_margin <- voted$vote_margin
        membership[landmark_idx] <- landmark_membership
        vote_margin[landmark_idx] <- 1
      }
      names(membership) <- rownames(vis)
      list(membership = renumber_membership(membership), vote_margin = as.numeric(vote_margin))
    }
    for (candidate_index in seq_along(evals)) {
      primary_candidate <- materialize_candidate(evals[[candidate_index]]$final_membership)
      agreement_count <- rep(1L, n_cells)
      candidate_ari <- numeric(0)
      if (stability_runs > 1L) {
        for (candidate_seed in selection_seed + seq.int(1L, stability_runs - 1L)) {
          active_seed <<- as.integer(candidate_seed)
          replicate_candidate <- run_louvain(
            graph_obj,
            landmark_vis,
            evals[[candidate_index]]$resolution,
            landmark_merge_min_size,
            cluster_algorithm,
            leiden_objective
          )
          replicate_full <- materialize_candidate(replicate_candidate$final_membership)
          aligned <- align_membership_to_reference(primary_candidate$membership, replicate_full$membership)
          agreement_count <- agreement_count + as.integer(aligned == primary_candidate$membership)
          candidate_ari <- c(candidate_ari, adjusted_rand_index(primary_candidate$membership, replicate_full$membership))
        }
      }
      candidate_stability <- agreement_count / stability_runs
      candidate_ambiguous <- if (assignment_mode == "exact") {
        rep(FALSE, n_cells)
      } else {
        !is.finite(primary_candidate$vote_margin) | primary_candidate$vote_margin < assignment_min_vote_margin
      }
      candidate_unstable <- !is.finite(candidate_stability) | candidate_stability < stability_min_fraction
      candidate_abstained <- candidate_ambiguous | candidate_unstable
      evals[[candidate_index]]$estimated_abstained_count <- as.integer(sum(candidate_abstained))
      evals[[candidate_index]]$estimated_abstained_fraction <- as.numeric(mean(candidate_abstained))
      evals[[candidate_index]]$estimated_ambiguous_count <- as.integer(sum(candidate_ambiguous))
      evals[[candidate_index]]$estimated_unstable_count <- as.integer(sum(candidate_unstable))
      evals[[candidate_index]]$candidate_mean_ari <- if (length(candidate_ari)) mean(candidate_ari, na.rm = TRUE) else 1
      evals[[candidate_index]]$candidate_min_ari <- if (length(candidate_ari)) min(candidate_ari, na.rm = TRUE) else 1
    }
    active_seed <<- selection_seed
    set.seed(selection_seed)
    if (!is.null(candidate_nn)) rm(candidate_nn)
    gc(FALSE)

    base_best <- select_auto_best(evals, cluster_cap, score_margin, auto_selection)
    landmark_best <- if (cluster_profile == "fine") {
      select_auto_fine(evals, base_best, fine_score_margin, fine_min_cluster_increase, auto_selection)
    } else {
      base_best
    }
    best_score <- max(vapply(evals, function(x) x$score, numeric(1)))
    candidate_table <- do.call(rbind, lapply(evals, function(item) data.frame(
      resolution = as.numeric(item$resolution),
      raw_cluster_count = as.integer(item$raw_cluster_count),
      final_cluster_count = as.integer(item$final_cluster_count),
      silhouette = as.numeric(item$silhouette),
      modularity = as.numeric(item$modularity),
      tiny_count = as.integer(item$tiny_count),
      tiny_fraction = as.numeric(item$tiny_fraction),
      quality_score = as.numeric(item$score),
      selection_eligible = as.logical(if (auto_selection == "minimum_abstention") {
        item$final_cluster_count >= 2L && item$final_cluster_count <= cluster_cap
      } else {
        item$score >= (best_score - score_margin) &&
          item$final_cluster_count >= 2L && item$final_cluster_count <= cluster_cap
      }),
      estimated_ambiguous_count = as.integer(item$estimated_ambiguous_count),
      estimated_unstable_count = as.integer(item$estimated_unstable_count),
      estimated_abstained_count = as.integer(item$estimated_abstained_count),
      estimated_abstained_fraction = as.numeric(item$estimated_abstained_fraction),
      candidate_mean_adjusted_rand_index = as.numeric(item$candidate_mean_ari),
      candidate_min_adjusted_rand_index = as.numeric(item$candidate_min_ari),
      selected = isTRUE(all.equal(as.numeric(item$resolution), as.numeric(landmark_best$resolution))),
      selection_mode = auto_selection,
      stringsAsFactors = FALSE
    )))
    landmark_best$auto_resolution_candidates <- candidate_table
    cat(sprintf("[INFO] %s landmark auto-resolution grid: %s\n", tools::toTitleCase(cluster_algorithm), paste(sprintf("%.3f", resolution_grid_eval), collapse = ", ")))
    for (item in evals) {
      cat(sprintf(
        paste0(
          "  - res=%.3f raw_clusters=%d final_clusters=%d silhouette=%s modularity=%.4f ",
          "tiny=%d tiny_fraction=%.4f score=%.4f predicted_abstained=%d (%.4f) mean_ari=%.4f min_ari=%.4f\n"
        ),
        item$resolution,
        item$raw_cluster_count,
        item$final_cluster_count,
        ifelse(is.finite(item$silhouette), sprintf("%.4f", item$silhouette), "NA"),
        item$modularity,
        item$tiny_count,
        item$tiny_fraction,
        item$score,
        item$estimated_abstained_count,
        item$estimated_abstained_fraction,
        item$candidate_mean_ari,
        item$candidate_min_ari
      ))
    }
    cat(sprintf(
      "[INFO] Selected %s landmark resolution %.3f raw_clusters=%d final_clusters=%d score=%.4f predicted_abstained=%d by %s\n",
      tools::toTitleCase(cluster_algorithm),
      landmark_best$resolution,
      landmark_best$raw_cluster_count,
      landmark_best$final_cluster_count,
      landmark_best$score,
      landmark_best$estimated_abstained_count,
      auto_selection
    ))
  } else {
    effective_resolution <- fixed_resolution
    if (cluster_profile == "fine") {
      effective_resolution <- fixed_resolution * fine_resolution_multiplier
    }
    landmark_best <- run_louvain(graph_obj, landmark_vis, effective_resolution, landmark_merge_min_size, cluster_algorithm, leiden_objective)
    landmark_best$auto_resolution_candidates <- data.frame(
      resolution = as.numeric(landmark_best$resolution),
      raw_cluster_count = as.integer(landmark_best$raw_cluster_count),
      final_cluster_count = as.integer(landmark_best$final_cluster_count),
      selected = TRUE,
      selection_mode = "fixed_resolution",
      stringsAsFactors = FALSE
    )
    cat(sprintf("[INFO] Selected %s landmark fixed resolution %.3f raw_clusters=%d final_clusters=%d\n",
      tools::toTitleCase(cluster_algorithm),
      landmark_best$resolution,
      landmark_best$raw_cluster_count,
      landmark_best$final_cluster_count
    ))
  }
  flush.console()

  landmark_membership <- renumber_membership(landmark_best$final_membership)
  names(landmark_membership) <- rownames(landmark_vis)
  if (assignment_mode == "exact") {
    membership <- landmark_membership
    used_assign_k <- 0L
    assignment_knn_backend <- NA_character_
    assignment_vote_fraction <- rep(1, n_cells)
    assignment_vote_margin <- rep(1, n_cells)
  } else {
    assigned <- assign_from_landmarks(vis, landmark_vis, landmark_membership, landmark_idx, assign_k, cluster_algorithm)
    membership <- assigned$membership
    used_assign_k <- assigned$assign_k
    assignment_knn_backend <- assigned$knn_backend
    assignment_vote_fraction <- assigned$vote_fraction
    assignment_vote_margin <- assigned$vote_margin
    rm(assigned)
  }
  membership <- renumber_membership(membership)
  rm(graph_obj, landmark_vis, landmark_membership)
  gc(FALSE)

  list(
    resolution = landmark_best$resolution,
    membership = membership,
    final_membership = membership,
    raw_cluster_count = length(unique(membership)),
    final_cluster_count = length(unique(membership)),
    modularity = landmark_best$modularity,
    silhouette = mean_silhouette(vis, membership),
    tiny_count = landmark_best$tiny_count,
    tiny_fraction = landmark_best$tiny_fraction,
    tiny_threshold = landmark_best$tiny_threshold,
    score = landmark_best$score,
    landmark_algorithm = cluster_algorithm,
    landmark_cells_used = as.integer(length(landmark_idx)),
    landmark_assignment_mode = assignment_mode,
    landmark_assign_k_used = as.integer(used_assign_k),
    landmark_density_k_used = as.integer(sampling_density_k_used),
    landmark_density_power = as.numeric(sampling_density_power),
    landmark_density_knn_backend = as.character(sampling_density_knn_backend),
    landmark_assignment_knn_backend = as.character(assignment_knn_backend),
    landmark_density_median_all = as.numeric(sampling_density_median_all),
    landmark_density_median_selected = as.numeric(sampling_density_median_selected),
    assignment_vote_fraction = assignment_vote_fraction,
    assignment_vote_margin = assignment_vote_margin,
    auto_resolution_candidates = landmark_best$auto_resolution_candidates,
    estimated_abstained_count = as.integer(landmark_best$estimated_abstained_count %||% NA_integer_),
    estimated_abstained_fraction = as.numeric(landmark_best$estimated_abstained_fraction %||% NA_real_)
  )
}

selected_resolution_for_replicates <- NA_real_

run_selected_clustering <- function(vis, seed_value) {
  active_seed <<- as.integer(seed_value)
  if (native_graph_mode) return(run_native_kodama_leiden(native_graph_info, fixed_resolution, leiden_objective, active_seed))
  if (cluster_algorithm == "walktrap") {
    return(run_walktrap_fixed(
      vis,
      actual_k,
      walktrap_clusters,
      landmark_cells,
      landmark_assign_k,
      landmark_sample_strategy,
      landmark_grid_bins,
      landmark_grid_max_per_bin,
      landmark_density_knn_k,
      landmark_density_power
    ))
  }
  replicate_resolution_mode <- resolution_mode
  replicate_fixed_resolution <- fixed_resolution
  replicate_profile <- cluster_profile
  if (resolution_mode == "auto" && is.finite(selected_resolution_for_replicates)) {
    # Resolution selection is performed once on paired candidates.  Stability
    # replicates must hold that selected resolution fixed while resampling
    # landmarks; otherwise they measure a changing model-selection decision.
    replicate_resolution_mode <- "fixed"
    replicate_fixed_resolution <- selected_resolution_for_replicates
    replicate_profile <- "standard"
  }
  run_louvain_landmark(
    vis,
    actual_k,
    cluster_algorithm,
    leiden_objective,
    replicate_resolution_mode,
    replicate_fixed_resolution,
    resolution_grid_eval,
    replicate_profile,
    fine_score_margin,
    fine_min_cluster_increase,
    landmark_cells,
    landmark_assign_k,
    landmark_sample_strategy,
    landmark_grid_bins,
    landmark_grid_max_per_bin,
    landmark_density_knn_k,
    landmark_density_power
  )
}

picked <- {
  info <- list_kodama_files(kodama_dir)
  select_kodama_file(selected_file_dim, info$dims, info$files)
}

input_bundle <- new.env(parent = baseenv())
load(picked$file, envir = input_bundle)
if (!exists("vis", envir = input_bundle, inherits = FALSE)) {
  stop(sprintf("Variable 'vis' not found in: %s", picked$file))
}

plot_vis <- as.matrix(input_bundle$vis)
if (is.null(rownames(plot_vis))) {
  if (cluster_representation != "umap2d") stop("PCA/native graph comparison requires explicit observation IDs in the KODAMA visualization")
  rownames(plot_vis) <- sprintf("cell_%05d", seq_len(nrow(plot_vis)))
}
if (nrow(plot_vis) < 3L) {
  stop(sprintf("Need at least 3 cells for clustering. Found: %d", nrow(plot_vis)))
}
if (ncol(plot_vis) < 2L || !all(is.finite(plot_vis)) || anyDuplicated(rownames(plot_vis))) {
  stop("Visualization requires >=2 finite columns and unique observation IDs")
}

grandqc_artifact_candidate <- rep(FALSE, nrow(plot_vis))
grandqc_artifact_candidate_fraction <- rep(0, nrow(plot_vis))
if (!is.null(observations_csv)) {
  observations <- read.csv(observations_csv, stringsAsFactors = FALSE, colClasses = c(label = "character"))
  if (!("label" %in% names(observations)) || anyNA(observations$label) || anyDuplicated(observations$label)) {
    stop("Observation CSV must contain unique nonmissing labels")
  }
  observations$label <- as.character(observations$label)
  if (!setequal(observations$label, rownames(plot_vis))) {
    stop("Observation CSV and KODAMA representation must contain exactly the same labels")
  }
  observations <- observations[match(rownames(plot_vis), observations$label), , drop = FALSE]
  if ("grandqc_artifact_candidate" %in% names(observations)) {
    raw_candidate <- tolower(trimws(as.character(observations$grandqc_artifact_candidate)))
    if (any(!(raw_candidate %in% c("true", "false", "1", "0", "yes", "no")))) {
      stop("grandqc_artifact_candidate must contain boolean values")
    }
    grandqc_artifact_candidate <- raw_candidate %in% c("true", "1", "yes")
  }
  if ("grandqc_artifact_candidate_fraction" %in% names(observations)) {
    grandqc_artifact_candidate_fraction <- as.numeric(observations$grandqc_artifact_candidate_fraction)
    if (any(!is.finite(grandqc_artifact_candidate_fraction)) || any(grandqc_artifact_candidate_fraction < 0 | grandqc_artifact_candidate_fraction > 1)) {
      stop("grandqc_artifact_candidate_fraction must contain finite values in [0,1]")
    }
  }
  rm(observations)
}

input_vis_dims <- ncol(plot_vis)
plot_vis <- plot_vis[, seq_len(2L), drop = FALSE]
representation_source <- picked$file
representation_source_md5 <- unname(tools::md5sum(picked$file))
visualization_source_md5 <- representation_source_md5
native_graph_info <- native_graph_payload <- NULL
if (native_graph_mode) {
  metadata <- input_bundle$representation_metadata
  if (!is.list(metadata) || !isTRUE(metadata$kodama_graph_available) ||
      !identical(metadata$kodama_graph_file, "kodama_graph.rds") ||
      !identical(metadata$kodama_graph_manifest_file, "kodama_graph.json") ||
      !is.character(metadata$kodama_graph_sha256) || length(metadata$kodama_graph_sha256) != 1L ||
      is.na(metadata$kodama_graph_sha256) || !grepl("^[0-9a-f]{64}$", metadata$kodama_graph_sha256)) {
    stop("Native graph mode requires an explicitly bound producer representation_metadata graph receipt; legacy numeric IDs cannot establish graph provenance")
  }
  native_graph_payload <- load_portable_kodama_graph(kodama_dir, rownames(plot_vis),
    expected_graph_sha256 = metadata$kodama_graph_sha256)
  if (!identical(native_graph_payload$manifest$file, metadata$kodama_graph_file))
    stop("Native graph filename differs from producer representation metadata")
  native_graph_info <- prepare_kodama_affinity_graph(native_graph_payload, rownames(plot_vis))
  representation_source <- file.path(kodama_dir, native_graph_payload$manifest$file)
  representation_source_md5 <- unname(tools::md5sum(representation_source))
  # Plot coordinates are display-only. Graph helpers never accept this matrix.
  vis <- plot_vis
} else if (cluster_representation == "pca") {
  metadata <- input_bundle$representation_metadata
  pca_name <- metadata$pca_file %||% paste0("pca_full_", picked$dim, ".RData")
  if (basename(pca_name) != pca_name) stop("PCA sidecar must be inside the KODAMA output directory")
  representation_source <- file.path(kodama_dir, pca_name)
  if (!file.exists(representation_source)) stop(sprintf("Saved PCA scores missing: %s. Re-run the KODAMA stage to export a high-dimensional representation.", representation_source))
  representation_source_md5 <- unname(tools::md5sum(representation_source))
  if (!is.null(metadata$pca_file_md5) && !identical(as.character(metadata$pca_file_md5), representation_source_md5)) stop("PCA sidecar checksum differs from KODAMA representation metadata")
  pca_bundle <- new.env(parent = baseenv())
  load(representation_source, envir = pca_bundle)
  if (!exists("pca", envir = pca_bundle, inherits = FALSE)) stop("PCA sidecar lacks variable pca")
  pca_scores <- as.matrix(pca_bundle$pca)
  if (is.null(rownames(pca_scores)) || anyDuplicated(rownames(pca_scores)) || !setequal(rownames(pca_scores), rownames(plot_vis))) stop("PCA and visualization observation IDs must match exactly; no intersection or imputation is allowed")
  if (!all(is.finite(pca_scores))) stop("Saved PCA scores contain nonfinite values")
  dimensions_to_use <- if (cluster_dimensions == 0L) ncol(pca_scores) else cluster_dimensions
  if (dimensions_to_use < 3L || dimensions_to_use > ncol(pca_scores)) stop("Requested PCA clustering dimensions unavailable; require >=3 and <=saved dimensions")
  vis <- pca_scores[rownames(plot_vis), seq_len(dimensions_to_use), drop = FALSE]
  if (landmark_sample_strategy == "grid" && landmark_cells > 0L && nrow(vis) > landmark_cells) stop("Two-dimensional grid landmark sampling is not valid for PCA feature space; choose random or knn_inverse_distance")
  rm(pca_bundle, pca_scores)
} else {
  vis <- plot_vis
}
actual_vis_dims <- if (native_graph_mode) 0L else ncol(vis)
actual_k <- if (native_graph_mode) NA_integer_ else max(2L, min(as.integer(requested_k), nrow(vis) - 1L))
merge_min_size <- if (native_graph_mode) NA_integer_ else max(merge_min_size_floor, ceiling(merge_min_size_fraction * nrow(vis)))
resolution_grid_eval <- build_resolution_grid(resolution_grid, cluster_profile, fine_resolution_multiplier, fine_resolution_max)
cat(sprintf(
  "[INFO] Loaded representation=%s: cells=%d visualization_dims=%d clustering_dims=%d requested_k=%d actual_k=%d\n",
  cluster_representation,
  nrow(vis),
  input_vis_dims,
  actual_vis_dims,
  requested_k,
  actual_k
))
flush.console()

if (native_graph_mode) {
  best <- run_native_kodama_leiden(native_graph_info, fixed_resolution, leiden_objective, clustering_seed)
  cat(sprintf("[INFO] Native KODAMA dissimilarity graph: vertices=%d undirected_edges=%d isolated=%d components=%d; affinity=%s; not a physical-neighbourhood graph\n",
    nrow(vis), igraph::ecount(native_graph_info$graph), sum(native_graph_info$degree == 0L),
    native_graph_info$connected_components, native_graph_info$affinity_transform))
} else if (cluster_algorithm == "walktrap") {
  best <- run_walktrap_fixed(
    vis,
    actual_k,
    walktrap_clusters,
    landmark_cells,
    landmark_assign_k,
    landmark_sample_strategy,
    landmark_grid_bins,
    landmark_grid_max_per_bin,
    landmark_density_knn_k,
    landmark_density_power
  )
  cat(sprintf("[INFO] Clustering representation=%s dimensions=%d; visualization stored separately. --dim selected file: kodama_full_%d.RData\n", cluster_representation, actual_vis_dims, picked$dim))
  cat(sprintf("[INFO] Cluster algorithm: walktrap\n"))
  cat(sprintf("[INFO] Walktrap requested clusters=%d graph_cells=%d assignment=%s assign_k=%d final clusters=%d silhouette=%s modularity=%.4f\n",
    as.integer(walktrap_clusters),
    best$landmark_cells_used,
    best$landmark_assignment_mode,
    best$landmark_assign_k_used,
    best$final_cluster_count,
    ifelse(is.finite(best$silhouette), sprintf("%.4f", best$silhouette), "NA"),
    best$modularity
  ))
} else {
  best <- run_louvain_landmark(
    vis,
    actual_k,
    cluster_algorithm,
    leiden_objective,
    resolution_mode,
    fixed_resolution,
    resolution_grid_eval,
    cluster_profile,
    fine_score_margin,
    fine_min_cluster_increase,
    landmark_cells,
    landmark_assign_k,
    landmark_sample_strategy,
    landmark_grid_bins,
    landmark_grid_max_per_bin,
    landmark_density_knn_k,
    landmark_density_power
  )
  cat(sprintf("[INFO] Clustering representation=%s dimensions=%d; visualization stored separately. --dim selected file: kodama_full_%d.RData\n", cluster_representation, actual_vis_dims, picked$dim))
  cat(sprintf("[INFO] Cluster algorithm: %s\n", cluster_algorithm))
  cat(sprintf(
    "[INFO] %s landmark resolution=%s graph_cells=%d assignment=%s assign_k=%d final clusters=%d silhouette=%s modularity=%.4f\n",
    tools::toTitleCase(cluster_algorithm),
    ifelse(is.finite(best$resolution), sprintf("%.3f", best$resolution), "NA"),
    best$landmark_cells_used,
    best$landmark_assignment_mode,
    best$landmark_assign_k_used,
    best$final_cluster_count,
    ifelse(is.finite(best$silhouette), sprintf("%.4f", best$silhouette), "NA"),
    best$modularity
  ))
}
if (resolution_mode == "auto" && is.finite(best$resolution)) {
  selected_resolution_for_replicates <- as.numeric(best$resolution)
}

raw_membership <- renumber_membership(best$membership)
target_result <- if (native_graph_mode) collapse_kodama_graph_to_target(native_graph_info, best$final_membership, target_clusters) else collapse_clusters_to_target(
  vis,
  best$final_membership,
  target_clusters
)
final_membership <- renumber_membership(target_result$membership)

grandqc_outlier <- if (isTRUE(grandqc_kodama_outlier_enable)) {
  detect_grandqc_kodama_outliers(
    plot_vis, final_membership, grandqc_artifact_candidate,
    grandqc_kodama_outlier_knn, grandqc_kodama_outlier_quantile,
    grandqc_kodama_outlier_mad_multiplier, grandqc_kodama_outlier_min_reference
  )
} else {
  list(
    is_outlier = rep(FALSE, nrow(plot_vis)), mean_knn_distance = rep(NA_real_, nrow(plot_vis)),
    cluster_threshold = rep(NA_real_, nrow(plot_vis)), score = rep(NA_real_, nrow(plot_vis)),
    reference_count = rep(0L, nrow(plot_vis))
  )
}

raw_cluster_count <- length(unique(raw_membership))
final_cluster_count <- length(unique(final_membership))
raw_cluster_sizes <- format_cluster_sizes(raw_membership)
final_cluster_sizes <- format_cluster_sizes(final_membership)
sample_id <- sub("_cluster$", "", tools::file_path_sans_ext(basename(out_csv)))

native_graph_evidence <- if (native_graph_mode) kodama_graph_assignment_evidence(native_graph_info, final_membership) else NULL
assignment_vote_fraction <- if (native_graph_mode) native_graph_evidence$own_community_affinity_fraction else as.numeric(best$assignment_vote_fraction %||% rep(1, nrow(vis)))
assignment_vote_margin <- if (native_graph_mode) native_graph_evidence$affinity_margin else as.numeric(best$assignment_vote_margin %||% rep(1, nrow(vis)))
assignment_is_exact <- identical(as.character(best$landmark_assignment_mode %||% ""), "exact")
assignment_ambiguous <- if (native_graph_mode) !is.finite(assignment_vote_margin) | assignment_vote_margin < assignment_min_vote_margin else !assignment_is_exact & assignment_vote_margin < assignment_min_vote_margin
assignment_status <- if (native_graph_mode) {
  ifelse(native_graph_evidence$is_isolated, "unassigned_isolated_native_graph_vertex",
    ifelse(assignment_ambiguous, "abstained_ambiguous_graph_affinity", "accepted_graph_affinity"))
} else if (assignment_is_exact) {
  rep("exact_graph_assignment", nrow(vis))
} else ifelse(
  assignment_vote_margin >= assignment_min_vote_margin,
  "accepted_landmark_vote",
  "abstained_ambiguous_landmark_vote"
)

stability_seeds <- clustering_seed + seq.int(0L, stability_runs - 1L)
stability_agreement_count <- rep(1L, nrow(vis))
stability_records <- list(data.frame(
  seed = as.integer(clustering_seed),
  cluster_count = as.integer(final_cluster_count),
  adjusted_rand_index_vs_primary = 1,
  stringsAsFactors = FALSE
))
if (stability_runs > 1L) {
  cat(sprintf("[INFO] Measuring clustering stability across %d total seeds\n", stability_runs))
  flush.console()
  for (replicate_seed in stability_seeds[-1L]) {
    cat(sprintf("[INFO] Stability replicate seed=%d\n", replicate_seed))
    flush.console()
    replicate_best <- run_selected_clustering(vis, replicate_seed)
    replicate_target <- if (native_graph_mode) collapse_kodama_graph_to_target(native_graph_info, replicate_best$final_membership, target_clusters) else collapse_clusters_to_target(
      vis,
      replicate_best$final_membership,
      target_clusters
    )
    replicate_membership <- renumber_membership(replicate_target$membership)
    aligned_membership <- align_membership_to_reference(final_membership, replicate_membership)
    stability_agreement_count <- stability_agreement_count + as.integer(aligned_membership == final_membership)
    stability_records[[length(stability_records) + 1L]] <- data.frame(
      seed = as.integer(replicate_seed),
      cluster_count = as.integer(length(unique(replicate_membership))),
      adjusted_rand_index_vs_primary = adjusted_rand_index(final_membership, replicate_membership),
      stringsAsFactors = FALSE
    )
    rm(replicate_best, replicate_target, replicate_membership, aligned_membership)
    gc(FALSE)
  }
}
active_seed <- clustering_seed
stability_df <- do.call(rbind, stability_records)
stability_fraction <- as.numeric(stability_agreement_count / stability_runs)
stability_status <- ifelse(
  stability_fraction >= stability_min_fraction,
  "stable_across_seeds",
  "abstained_unstable_across_seeds"
)
uncertainty_reason <- ifelse(
  assignment_ambiguous & stability_status == "abstained_unstable_across_seeds",
  "ambiguous_assignment_and_seed_instability",
  ifelse(
    assignment_ambiguous,
    "ambiguous_assignment",
    ifelse(stability_status == "abstained_unstable_across_seeds", "seed_instability", "none")
  )
)
if (native_graph_mode) uncertainty_reason[native_graph_evidence$is_isolated] <- "isolated_native_graph_vertex"
uncertainty_reason[grandqc_outlier$is_outlier] <- "grandqc_artifact_kodama_outlier"
interpretable_cluster <- as.integer(final_membership)
if (isTRUE(abstain_uncertain)) {
  interpretable_cluster[uncertainty_reason != "none"] <- NA_integer_
}
if (native_graph_mode) interpretable_cluster[native_graph_evidence$is_isolated] <- NA_integer_
interpretable_cluster[grandqc_outlier$is_outlier] <- NA_integer_
is_abstained <- is.na(interpretable_cluster)
interpretation_status <- ifelse(
  uncertainty_reason == "none",
  "accepted",
  paste0(ifelse(is_abstained, "abstained_", "flagged_"), uncertainty_reason)
)
forced_cluster_count_requested <- target_clusters > 0L
cluster_analysis_role <- if (forced_cluster_count_requested) {
  "sensitivity_forced_cluster_count"
} else {
  "graph_derived_primary_candidate"
}

cluster_df <- data.frame(
  label = rownames(vis),
  cluster_representation = cluster_representation,
  clustering_dimensions = actual_vis_dims,
  representation_input_md5 = visualization_source_md5,
  graph_affinity_rule = if (native_graph_mode) native_graph_info$affinity_transform else NA_character_,
  graph_source_sha256 = if (native_graph_mode) native_graph_payload$manifest$file_sha256 else NA_character_,
  graph_observation_scope = if (native_graph_mode) "feature_dissimilarity_not_physical_neighbourhood" else NA_character_,
  assignment_score_semantics = if (native_graph_mode) native_graph_evidence$semantics else "landmark_vote_or_exact_graph_assignment_not_probability",
  graph_degree = if (native_graph_mode) native_graph_evidence$degree else NA_integer_,
  graph_strength = if (native_graph_mode) native_graph_evidence$strength else NA_real_,
  grandqc_artifact_candidate = grandqc_artifact_candidate,
  grandqc_artifact_candidate_fraction = grandqc_artifact_candidate_fraction,
  grandqc_kodama_outlier = grandqc_outlier$is_outlier,
  grandqc_kodama_mean_knn_distance = grandqc_outlier$mean_knn_distance,
  grandqc_kodama_cluster_threshold = grandqc_outlier$cluster_threshold,
  grandqc_kodama_outlier_score = grandqc_outlier$score,
  grandqc_kodama_reference_count = grandqc_outlier$reference_count,
  exclude_from_downstream = grandqc_outlier$is_outlier,
  cluster = as.integer(final_membership),
  interpretable_cluster = interpretable_cluster,
  assignment_vote_fraction = assignment_vote_fraction,
  assignment_vote_margin = assignment_vote_margin,
  assignment_status = assignment_status,
  stability_fraction = stability_fraction,
  stability_status = stability_status,
  uncertainty_reason = uncertainty_reason,
  is_abstained = is_abstained,
  interpretation_status = interpretation_status,
  cluster_analysis_role = cluster_analysis_role,
  forced_cluster_count_requested = forced_cluster_count_requested,
  forced_cluster_count_target = as.integer(target_clusters),
  forced_cluster_count_applied = isTRUE(target_result$applied),
  stringsAsFactors = FALSE
)
write.csv(cluster_df, out_csv, row.names = FALSE, quote = FALSE)

stability_path <- file.path(dirname(out_csv), paste0(sample_id, "_cluster_stability.csv"))
write.csv(stability_df, stability_path, row.names = FALSE, quote = FALSE)
resolution_candidates <- best$auto_resolution_candidates
if (is.null(resolution_candidates)) {
  resolution_candidates <- data.frame(
    resolution = as.numeric(best$resolution),
    raw_cluster_count = as.integer(raw_cluster_count),
    final_cluster_count = as.integer(final_cluster_count),
    selected = TRUE,
    selection_mode = if (resolution_mode == "auto") auto_selection else "not_applicable",
    stringsAsFactors = FALSE
  )
}
resolution_candidates <- cbind(
  data.frame(sample_id = sample_id, cluster_profile = cluster_profile, stringsAsFactors = FALSE),
  resolution_candidates
)
resolution_candidates_path <- file.path(dirname(out_csv), paste0(sample_id, "_cluster_resolution_candidates.csv"))
write.csv(resolution_candidates, resolution_candidates_path, row.names = FALSE, quote = FALSE)
summary_path <- file.path(dirname(out_csv), paste0(sample_id, "_cluster_summary.csv"))
summary_df <- data.frame(
  sample_id = sample_id,
  cluster_profile = cluster_profile,
  vis_dims = ncol(plot_vis),
  cluster_representation = cluster_representation,
  clustering_dimensions = actual_vis_dims,
  requested_clustering_dimensions = cluster_dimensions,
  representation_source = basename(representation_source),
  representation_source_md5 = representation_source_md5,
  visualization_source = basename(picked$file),
  visualization_source_md5 = visualization_source_md5,
  graph_source_sha256 = if (native_graph_mode) native_graph_payload$manifest$file_sha256 else NA_character_,
  graph_affinity_rule = if (native_graph_mode) native_graph_info$affinity_transform else NA_character_,
  graph_observation_scope = if (native_graph_mode) "feature_dissimilarity_not_physical_neighbourhood" else NA_character_,
  graph_connected_components = if (native_graph_mode) native_graph_info$connected_components else NA_integer_,
  graph_isolated_observations = if (native_graph_mode) sum(native_graph_evidence$is_isolated) else NA_integer_,
  graph_native_neighbors = if (native_graph_mode) native_graph_payload$metadata$native_neighbors else NA_integer_,
  graph_stored_directed_edges = if (native_graph_mode) native_graph_payload$metadata$stored_edges else NA_integer_,
  graph_source_pca_sha256 = if (native_graph_mode) native_graph_payload$metadata$pca_file_sha256 else NA_character_,
  assignment_score_semantics = if (native_graph_mode) native_graph_evidence$semantics else "landmark_vote_or_exact_graph_assignment_not_probability",
  visualization_projected = if (is.null(input_bundle$representation_metadata$visualization_projected)) NA else isTRUE(input_bundle$representation_metadata$visualization_projected),
  requested_dim = as.integer(selected_file_dim),
  loaded_dim = as.integer(picked$dim),
  requested_k = if (native_graph_mode) NA_integer_ else as.integer(requested_k),
  actual_k = as.integer(actual_k),
  cluster_algorithm = cluster_algorithm,
  leiden_objective = leiden_objective,
  clustering_seed = as.integer(clustering_seed),
  stability_runs = as.integer(stability_runs),
  stability_mean_adjusted_rand_index = if (nrow(stability_df) > 1L) {
    mean(stability_df$adjusted_rand_index_vs_primary[-1L], na.rm = TRUE)
  } else {
    1
  },
  stability_min_adjusted_rand_index = if (nrow(stability_df) > 1L) {
    min(stability_df$adjusted_rand_index_vs_primary[-1L], na.rm = TRUE)
  } else {
    1
  },
  assignment_min_vote_margin = as.numeric(assignment_min_vote_margin),
  stability_min_fraction = as.numeric(stability_min_fraction),
  abstain_uncertain = isTRUE(abstain_uncertain),
  ambiguous_assignment_count = as.integer(sum(assignment_ambiguous)),
  unstable_observation_count = as.integer(sum(stability_status == "abstained_unstable_across_seeds")),
  abstained_observation_count = as.integer(sum(is.na(interpretable_cluster))),
  grandqc_artifact_candidate_count = as.integer(sum(grandqc_artifact_candidate)),
  grandqc_kodama_outlier_enabled = isTRUE(grandqc_kodama_outlier_enable),
  grandqc_kodama_outlier_count = as.integer(sum(grandqc_outlier$is_outlier)),
  grandqc_kodama_outlier_policy = "candidate_only_cluster_conditioned_knn_distance_in_kodama_plot_fail_open",
  grandqc_kodama_outlier_knn = as.integer(grandqc_kodama_outlier_knn),
  grandqc_kodama_outlier_quantile = as.numeric(grandqc_kodama_outlier_quantile),
  grandqc_kodama_outlier_mad_multiplier = as.numeric(grandqc_kodama_outlier_mad_multiplier),
  grandqc_kodama_outlier_min_reference = as.integer(grandqc_kodama_outlier_min_reference),
  accepted_observation_fraction = as.numeric(mean(!is.na(interpretable_cluster))),
  cluster_analysis_role = cluster_analysis_role,
  claim_status = if (forced_cluster_count_requested) "sensitivity_only" else "independent_interpretation_required",
  forced_cluster_count_requested = forced_cluster_count_requested,
  forced_cluster_count_applied = isTRUE(target_result$applied),
  target_clusters = as.integer(target_clusters),
  target_applied = isTRUE(target_result$applied),
  target_strategy = if (target_clusters > 0L) {
    if (native_graph_mode) "maximum_total_cross_community_graph_affinity_supported_merges_only" else paste0("nearest_centroid_merge_in_", cluster_representation, "_space")
  } else {
    "disabled"
  },
  target_input_cluster_count = as.integer(target_result$initial_count),
  target_merge_history = target_result$merge_history,
  landmark_cells = as.integer(landmark_cells),
  landmark_sample_strategy = if (native_graph_mode) "not_applicable_native_graph" else landmark_sample_strategy,
  landmark_density_knn_k = if (native_graph_mode) NA_integer_ else as.integer(landmark_density_knn_k),
  landmark_density_power = if (native_graph_mode) NA_real_ else as.numeric(landmark_density_power),
  landmark_grid_bins = if (native_graph_mode) NA_integer_ else as.integer(landmark_grid_bins),
  landmark_grid_max_per_bin = if (native_graph_mode) NA_integer_ else as.integer(landmark_grid_max_per_bin),
  landmark_assign_k = if (native_graph_mode) NA_integer_ else as.integer(landmark_assign_k),
  landmark_algorithm = as.character(best$landmark_algorithm %||% cluster_algorithm),
  landmark_cells_used = as.integer(best$landmark_cells_used %||% NA_integer_),
  landmark_assignment_mode = as.character(best$landmark_assignment_mode %||% NA_character_),
  landmark_assign_k_used = as.integer(best$landmark_assign_k_used %||% NA_integer_),
  landmark_density_k_used = as.integer(best$landmark_density_k_used %||% NA_integer_),
  landmark_density_knn_backend = as.character(best$landmark_density_knn_backend %||% NA_character_),
  landmark_assignment_knn_backend = as.character(best$landmark_assignment_knn_backend %||% NA_character_),
  landmark_density_median_all = as.numeric(best$landmark_density_median_all %||% NA_real_),
  landmark_density_median_selected = as.numeric(best$landmark_density_median_selected %||% NA_real_),
  walktrap_clusters = as.integer(walktrap_clusters),
  walktrap_max_cells = as.integer(walktrap_max_cells),
  walktrap_assign_k = as.integer(walktrap_assign_k),
  walktrap_cells_used = as.integer(best$walktrap_cells_used %||% NA_integer_),
  walktrap_assignment_mode = as.character(best$walktrap_assignment_mode %||% NA_character_),
  walktrap_assign_k_used = as.integer(best$walktrap_assign_k_used %||% NA_integer_),
  resolution_mode = resolution_mode,
  auto_resolution_selection = if (resolution_mode == "auto") auto_selection else "not_applicable",
  auto_resolution_quality_guard = if (resolution_mode == "auto") {
    if (auto_selection == "minimum_abstention") "at_least_two_clusters_and_within_cluster_cap" else "within_quality_score_margin_and_cluster_cap"
  } else {
    "not_applicable"
  },
  auto_resolution_candidate_count = as.integer(nrow(resolution_candidates)),
  selection_estimated_abstained_count = as.integer(best$estimated_abstained_count %||% NA_integer_),
  selection_estimated_abstained_fraction = as.numeric(best$estimated_abstained_fraction %||% NA_real_),
  selected_resolution = as.numeric(best$resolution),
  raw_cluster_count = as.integer(raw_cluster_count),
  final_cluster_count = as.integer(final_cluster_count),
  merge_min_size = as.integer(merge_min_size),
  raw_cluster_sizes = raw_cluster_sizes,
  final_cluster_sizes = final_cluster_sizes,
  stringsAsFactors = FALSE
)
write.csv(summary_df, summary_path, row.names = FALSE, quote = TRUE)

if (!is.null(comparison_clusters)) {
  reference <- read.csv(comparison_clusters, stringsAsFactors = FALSE, colClasses = c(label = "character"))
  if (!all(c("label", "cluster") %in% names(reference)) || anyNA(reference$label) || anyDuplicated(reference$label)) {
    stop("Comparison CSV needs unique label and cluster columns")
  }
  if (!setequal(reference$label, cluster_df$label)) {
    stop("Representation comparison requires exactly matched observations; silent intersections are not allowed")
  }
  input_identity_verified <- FALSE
  if ("representation_input_md5" %in% names(reference)) {
    if (anyNA(reference$representation_input_md5) || !all(reference$representation_input_md5 == visualization_source_md5)) stop("Representation comparison inputs differ; matching numeric observation IDs alone is insufficient")
    input_identity_verified <- TRUE
  }
  reference <- reference[match(cluster_df$label, reference$label), , drop = FALSE]
  if (anyNA(reference$cluster)) stop("Comparison raw cluster labels must not be missing")
  reference_interpretable <- if ("interpretable_cluster" %in% names(reference)) reference$interpretable_cluster else reference$cluster
  joint <- !is.na(reference_interpretable) & !is.na(cluster_df$interpretable_cluster)
  reference_representation <- if ("cluster_representation" %in% names(reference)) paste(sort(unique(reference$cluster_representation)), collapse = ";") else "unspecified_legacy"
  comparison <- data.frame(
    sample_id = sample_id,
    current_representation = cluster_representation,
    reference_representation = reference_representation,
    current_dimensions = actual_vis_dims,
    matched_observations = nrow(cluster_df),
    input_bundle_identity_verified = input_identity_verified,
    current_graph_affinity_rule = if (native_graph_mode) native_graph_info$affinity_transform else NA_character_,
    current_graph_source_sha256 = if (native_graph_mode) native_graph_payload$manifest$file_sha256 else NA_character_,
    reference_graph_affinity_rule = if ("graph_affinity_rule" %in% names(reference)) paste(unique(reference$graph_affinity_rule), collapse = ";") else NA_character_,
    reference_graph_source_sha256 = if ("graph_source_sha256" %in% names(reference)) paste(unique(reference$graph_source_sha256), collapse = ";") else NA_character_,
    adjusted_rand_index_raw = adjusted_rand_index(reference$cluster, cluster_df$cluster),
    jointly_interpretable_observations = sum(joint),
    jointly_interpretable_fraction = mean(joint),
    adjusted_rand_index_jointly_interpretable = if (sum(joint) >= 2L) adjusted_rand_index(reference_interpretable[joint], cluster_df$interpretable_cluster[joint]) else NA_real_,
    current_cluster_count = length(unique(cluster_df$cluster)),
    reference_cluster_count = length(unique(reference$cluster)),
    reference_file_md5 = unname(tools::md5sum(comparison_clusters)),
    comparison_claim = "descriptive_partition_agreement_only_not_accuracy_or_biological_validation",
    stringsAsFactors = FALSE
  )
  write.csv(comparison, file.path(dirname(out_csv), paste0(sample_id, "_representation_comparison.csv")), row.names = FALSE)
}

pdf_path <- file.path(dirname(out_csv), paste0(sample_id, "_cluster_kodama_membership.pdf"))
pdf(pdf_path)
cluster_colors <- cluster_plot_colors(plot_vis, final_membership, cluster_algorithm)
plot_analysis_label <- if (forced_cluster_count_requested) {
  sprintf("SENSITIVITY ONLY: requested target=%d", target_clusters)
} else {
  "graph-derived count"
}
draw_membership_plot(
  plot_vis,
  final_membership,
  cluster_colors,
  cluster_algorithm,
  is_abstained = is_abstained,
  main = sprintf("%s | %s | %s", sample_id, cluster_algorithm, plot_analysis_label)
)
dev.off()

png_path <- file.path(dirname(out_csv), paste0(sample_id, "_cluster_kodama_membership.png"))
png(filename = png_path, width = 1800, height = 1400, res = 180)
draw_membership_plot(
  plot_vis,
  final_membership,
  cluster_colors,
  cluster_algorithm,
  is_abstained = is_abstained,
  main = sprintf("%s | %s | %s", sample_id, cluster_algorithm, plot_analysis_label)
)
dev.off()

uncertainty_pdf_path <- file.path(dirname(out_csv), paste0(sample_id, "_cluster_kodama_uncertainty.pdf"))
pdf(uncertainty_pdf_path, width = 13, height = 6.5)
draw_uncertainty_plot(
  plot_vis,
  uncertainty_reason,
  is_abstained,
  assignment_vote_margin,
  stability_fraction,
  main = sprintf("%s | uncertainty and abstention", sample_id)
)
dev.off()

uncertainty_png_path <- file.path(dirname(out_csv), paste0(sample_id, "_cluster_kodama_uncertainty.png"))
png(filename = uncertainty_png_path, width = 2600, height = 1300, res = 180)
draw_uncertainty_plot(
  plot_vis,
  uncertainty_reason,
  is_abstained,
  assignment_vote_margin,
  stability_fraction,
  main = sprintf("%s | uncertainty and abstention", sample_id)
)
dev.off()

cat(sprintf("[INFO] Requested --dim=%d loaded kodama_full_%d.RData\n", selected_file_dim, picked$dim))
if (!isTRUE(picked$exact)) {
  cat("[INFO] Requested dim file was not present; nearest lower available file was used.\n")
}
if (!native_graph_mode) cat(sprintf("[INFO] Requested k=%d actual k=%d\n", requested_k, actual_k))
cat(sprintf("[INFO] Cluster algorithm=%s\n", cluster_algorithm))
cat(sprintf(
  "[INFO] Stability runs=%d mean ARI=%s minimum ARI=%s unstable observations=%d\n",
  stability_runs,
  format_optional_number(summary_df$stability_mean_adjusted_rand_index, 4L),
  format_optional_number(summary_df$stability_min_adjusted_rand_index, 4L),
  summary_df$unstable_observation_count
))
cat(sprintf(
  "[INFO] Assignment minimum vote margin=%.3f ambiguous observations=%d abstention=%s abstained observations=%d\n",
  assignment_min_vote_margin,
  summary_df$ambiguous_assignment_count,
  ifelse(abstain_uncertain, "enabled", "disabled"),
  summary_df$abstained_observation_count
))
cat(sprintf(
  "[INFO] GrandQC candidates=%d KODAMA outliers=%d policy=%s\n",
  summary_df$grandqc_artifact_candidate_count,
  summary_df$grandqc_kodama_outlier_count,
  summary_df$grandqc_kodama_outlier_policy
))
if (target_clusters > 0L) {
  cat("[WARN] Forced cluster count requested: this output is a sensitivity analysis and must not be presented as the graph-derived primary partition.\n")
  cat(sprintf(
    paste0(
      "[INFO] Target clusters=%d input clusters=%d final clusters=%d ",
      "strategy=%s applied=%s\n"
    ),
    target_clusters,
    target_result$initial_count,
    final_cluster_count,
    summary_df$target_strategy,
    ifelse(target_result$applied, "yes", "no")
  ))
  if (nzchar(target_result$merge_history)) {
    cat(sprintf("[INFO] Target merge history=%s\n", target_result$merge_history))
  }
}
if (!native_graph_mode) cat(sprintf("[INFO] Landmark cells requested=%d used=%d strategy=%s assignment=%s assign_k=%d\n",
  as.integer(landmark_cells),
  as.integer(best$landmark_cells_used %||% NA_integer_),
  landmark_sample_strategy,
  as.character(best$landmark_assignment_mode %||% NA_character_),
  as.integer(best$landmark_assign_k_used %||% NA_integer_)
))
if (!native_graph_mode && landmark_sample_strategy == "knn_inverse_distance") {
  cat(sprintf("[INFO] Inverse-distance landmarks density_k=%d p=%.3f median_mean_density_distance_all=%s selected=%s\n",
    as.integer(best$landmark_density_k_used %||% landmark_density_knn_k),
    as.numeric(best$landmark_density_power %||% landmark_density_power),
    format_optional_number(best$landmark_density_median_all, 6L),
    format_optional_number(best$landmark_density_median_selected, 6L)
  ))
  cat(sprintf("[INFO] Inverse-distance landmarks KNN backend=%s assignment KNN backend=%s\n",
    as.character(best$landmark_density_knn_backend %||% NA_character_),
    as.character(best$landmark_assignment_knn_backend %||% NA_character_)
  ))
}
if (cluster_algorithm == "walktrap") {
  cat(sprintf("[INFO] Walktrap requested clusters=%d\n", walktrap_clusters))
  cat(sprintf("[INFO] Walktrap max cells=%d cells used=%d assignment=%s assign_k=%d\n",
    walktrap_max_cells,
    as.integer(best$walktrap_cells_used),
    as.character(best$walktrap_assignment_mode),
    as.integer(best$walktrap_assign_k_used)
  ))
}
cat(sprintf("[INFO] Raw clusters=%d final clusters=%d merge_min_size=%d\n", raw_cluster_count, final_cluster_count, merge_min_size))
cat(sprintf("[INFO] Raw cluster sizes=%s\n", raw_cluster_sizes))
cat(sprintf("[INFO] Final cluster sizes=%s\n", final_cluster_sizes))
cat(sprintf("[INFO] Wrote: %s\n", out_csv))
cat(sprintf("[INFO] Wrote: %s\n", summary_path))
cat(sprintf("[INFO] Wrote: %s\n", stability_path))
cat(sprintf("[INFO] Wrote: %s\n", pdf_path))
cat(sprintf("[INFO] Wrote: %s\n", png_path))
