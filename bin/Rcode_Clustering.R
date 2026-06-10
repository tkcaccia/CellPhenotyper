args <- commandArgs(trailingOnly = TRUE)
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

if (length(args) < 2L) {
  stop("Usage: Rscript Rcode_Clustering.R <kodama_dir> <out_csv> [--dim N] [--k N] [--algorithm louvain|leiden|walktrap] [--landmark-cells N] [--landmark-assign-k N] [--landmark-sample-strategy random|grid|knn_inverse_distance] [--landmark-density-knn-k N] [--landmark-density-power X] [--walktrap-clusters N] [--resolution auto|X] [--profile standard|fine]")
}

kodama_dir <- args[1]
out_csv <- args[2]

selected_file_dim <- 20L
requested_k <- 50L
cluster_algorithm <- "leiden"
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

if (!is.finite(selected_file_dim) || selected_file_dim < 2L) {
  stop("--dim must be an integer >= 2.")
}
if (!is.finite(requested_k) || requested_k < 2L) {
  stop("--k must be an integer >= 2.")
}
if (!(cluster_algorithm %in% c("louvain", "leiden", "walktrap"))) {
  stop("--algorithm must be 'louvain', 'leiden', or 'walktrap'.")
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

require_namespace <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Required package '%s' is not installed.", pkg))
  }
}

require_namespace("bluster")
require_namespace("igraph")
require_namespace("cluster")
if (landmark_cells > 0L) {
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

mean_silhouette <- function(vis, membership) {
  if (nrow(vis) < 4L || length(unique(membership)) < 2L) {
    return(NA_real_)
  }
  idx <- seq_len(nrow(vis))
  if (nrow(vis) > silhouette_max_cells) {
    set.seed(1L)
    idx <- sort(sample(idx, silhouette_max_cells))
  }
  sil <- tryCatch(
    cluster::silhouette(as.integer(membership[idx]), stats::dist(scale(vis[idx, , drop = FALSE]))),
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
  set.seed(1L)
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
  as.integer(cluster_ids[max.col(vote_counts, ties.method = "first")])
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
  cat(sprintf("[INFO] Assigning all %d cells from %s landmarks by %d-NN vote in KODAMA space\n", nrow(vis), label, used_assign_k))
  flush.console()
  nn <- knn_query_indices(landmark_vis, vis, used_assign_k)
  cat(sprintf("[INFO] Landmark assignment KNN backend: %s\n", nn$backend))
  flush.console()
  membership <- vote_membership(nn$index, landmark_membership)
  membership[landmark_idx] <- landmark_membership
  names(membership) <- rownames(vis)
  membership <- renumber_membership(membership)
  backend <- nn$backend
  rm(nn)
  gc(FALSE)
  list(membership = membership, assign_k = used_assign_k, knn_backend = backend)
}

knn_inverse_distance_sample <- function(vis, max_total, density_k, density_power) {
  max_total <- min(as.integer(max_total), nrow(vis))
  if (max_total <= 0L || nrow(vis) <= max_total) {
    return(seq_len(nrow(vis)))
  }
  used_density_k <- max(1L, min(as.integer(density_k), nrow(vis) - 1L))
  density_power <- as.numeric(density_power)
  cat(sprintf(
    "[INFO] Selecting %d inverse-distance landmarks from %d cells using mean %d-NN distance and p=%.3f\n",
    max_total,
    nrow(vis),
    used_density_k,
    density_power
  ))
  flush.console()
  nn <- knn_self_mean_distance(vis, used_density_k)
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
    idx <- sort(sample.int(nrow(vis), max_total))
  } else {
    weights <- weights / max_weight
    idx <- sort(sample.int(nrow(vis), size = max_total, replace = FALSE, prob = weights))
  }
  attr(idx, "density_k_used") <- used_density_k
  attr(idx, "density_power") <- density_power
  attr(idx, "density_knn_backend") <- knn_backend
  attr(idx, "mean_density_distance_median_all") <- stats::median(mean_dist, na.rm = TRUE)
  attr(idx, "mean_density_distance_median_selected") <- stats::median(mean_dist[idx], na.rm = TRUE)
  rm(nn, weights, mean_dist)
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
  set.seed(1L)
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
    landmark_assignment_knn_backend = as.character(assignment_knn_backend)
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
  cols <- setNames(rep("#000000", length(cluster_ids)), as.character(cluster_ids))
  cent <- do.call(rbind, lapply(cluster_ids, function(id) colMeans(vis[membership == id, , drop = FALSE])))
  rownames(cent) <- as.character(cluster_ids)
  sizes <- table(as.integer(membership))
  if (length(cluster_ids) >= 4L) {
    background <- names(which.max(sizes))
    rest <- setdiff(rownames(cent), background)
    left <- rest[which.min(cent[rest, 1])]
    bottom <- rest[which.min(cent[rest, 2])]
    green_candidates <- setdiff(rest, c(left, bottom))
    green <- if (length(green_candidates)) green_candidates[1] else NA_character_
    cols[background] <- "#000000"
    cols[left] <- "#F2D51B"
    cols[bottom] <- "#8B0000"
    if (!is.na(green)) cols[green] <- "#007000"
    extra <- setdiff(rest, c(left, bottom, green))
    if (length(extra)) cols[extra] <- rep(c("#1F77B4", "#FF7F0E", "#9467BD"), length.out = length(extra))
  } else if (length(cluster_ids) == 3L) {
    left <- rownames(cent)[which.min(cent[, 1])]
    bottom <- rownames(cent)[which.min(cent[, 2])]
    rest <- setdiff(rownames(cent), c(left, bottom))
    cols[left] <- "#F2D51B"
    cols[bottom] <- "#8B0000"
    if (length(rest)) cols[rest[1]] <- "#007000"
  } else {
    palette_hex <- c(
      "#000000", "#007000", "#8B0000", "#F2D51B",
      "#1F77B4", "#FF7F0E", "#9467BD", "#0082C8"
    )
    cols <- setNames(rep(palette_hex, length.out = length(cluster_ids)), as.character(cluster_ids))
  }
  cols
}

draw_membership_plot <- function(vis, membership, cluster_colors, algorithm, main = NULL) {
  set.seed(1L)
  plot_max_points <- 350000L
  plot_idx <- if (nrow(vis) > plot_max_points) sort(sample.int(nrow(vis), plot_max_points)) else seq_len(nrow(vis))
  plot(vis[plot_idx, 1], vis[plot_idx, 2],
    pch = 16,
    cex = 0.18,
    col = grDevices::adjustcolor("black", alpha.f = 0.25),
    xlab = "KODAMA dimension 1",
    ylab = "KODAMA dimension 2",
    main = main
  )
  for (cluster_id in names(cluster_colors)) {
    idx <- plot_idx[membership[plot_idx] == as.integer(cluster_id)]
    if (!length(idx)) next
    color <- cluster_colors[cluster_id]
    alpha <- if (identical(unname(color), "#000000")) 0.35 else 0.65
    points(vis[idx, 1], vis[idx, 2], pch = 16, cex = 0.16, col = grDevices::adjustcolor(color, alpha.f = alpha))
  }
  legend("topright",
    legend = sprintf("cluster %s n=%d", names(cluster_colors), as.integer(table(membership)[names(cluster_colors)])),
    col = cluster_colors,
    pch = 16,
    bty = "n",
    cex = 0.8
  )
}

preferred_cluster_cap <- function(n_cells) {
  max(4L, min(25L, as.integer(round(sqrt(max(1L, n_cells)) * 0.75))))
}

select_auto_best <- function(evals, cluster_cap, score_margin) {
  scores <- vapply(evals, function(x) x$score, numeric(1))
  best_score <- max(scores)
  near_best_idx <- which(scores >= (best_score - score_margin))
  near_best <- evals[near_best_idx]

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

select_auto_fine <- function(evals, base_best, fine_score_margin, fine_min_cluster_increase) {
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
  set.seed(1L)
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
    base_best <- select_auto_best(evals, cluster_cap, score_margin)
    landmark_best <- if (cluster_profile == "fine") {
      select_auto_fine(evals, base_best, fine_score_margin, fine_min_cluster_increase)
    } else {
      base_best
    }
    cat(sprintf("[INFO] %s landmark auto-resolution grid: %s\n", tools::toTitleCase(cluster_algorithm), paste(sprintf("%.3f", resolution_grid_eval), collapse = ", ")))
    for (item in evals) {
      cat(sprintf(
        "  - res=%.3f raw_clusters=%d final_clusters=%d silhouette=%s modularity=%.4f tiny=%d tiny_fraction=%.4f score=%.4f\n",
        item$resolution,
        item$raw_cluster_count,
        item$final_cluster_count,
        ifelse(is.finite(item$silhouette), sprintf("%.4f", item$silhouette), "NA"),
        item$modularity,
        item$tiny_count,
        item$tiny_fraction,
        item$score
      ))
    }
    cat(sprintf(
      "[INFO] Selected %s landmark resolution %.3f raw_clusters=%d final_clusters=%d score=%.4f\n",
      tools::toTitleCase(cluster_algorithm),
      landmark_best$resolution,
      landmark_best$raw_cluster_count,
      landmark_best$final_cluster_count,
      landmark_best$score
    ))
  } else {
    effective_resolution <- fixed_resolution
    if (cluster_profile == "fine") {
      effective_resolution <- fixed_resolution * fine_resolution_multiplier
    }
    landmark_best <- run_louvain(graph_obj, landmark_vis, effective_resolution, landmark_merge_min_size, cluster_algorithm, leiden_objective)
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
  } else {
    assigned <- assign_from_landmarks(vis, landmark_vis, landmark_membership, landmark_idx, assign_k, cluster_algorithm)
    membership <- assigned$membership
    used_assign_k <- assigned$assign_k
    assignment_knn_backend <- assigned$knn_backend
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
    landmark_density_median_selected = as.numeric(sampling_density_median_selected)
  )
}

picked <- {
  info <- list_kodama_files(kodama_dir)
  select_kodama_file(selected_file_dim, info$dims, info$files)
}

load(picked$file)
if (!exists("vis")) {
  stop(sprintf("Variable 'vis' not found in: %s", picked$file))
}

vis <- as.matrix(vis)
if (is.null(rownames(vis))) {
  rownames(vis) <- sprintf("cell_%05d", seq_len(nrow(vis)))
}
if (nrow(vis) < 3L) {
  stop(sprintf("Need at least 3 cells for clustering. Found: %d", nrow(vis)))
}
if (ncol(vis) < 2L) {
  stop(sprintf("Expected 'vis' to have at least 2 columns. Found: %d", ncol(vis)))
}

input_vis_dims <- ncol(vis)
vis <- vis[, seq_len(min(2L, ncol(vis))), drop = FALSE]
actual_vis_dims <- ncol(vis)
actual_k <- max(2L, min(as.integer(requested_k), nrow(vis) - 1L))
merge_min_size <- max(merge_min_size_floor, ceiling(merge_min_size_fraction * nrow(vis)))
resolution_grid_eval <- build_resolution_grid(resolution_grid, cluster_profile, fine_resolution_multiplier, fine_resolution_max)
cat(sprintf(
  "[INFO] Loaded KODAMA vis: cells=%d input_dims=%d clustering_dims=%d requested_k=%d actual_k=%d\n",
  nrow(vis),
  input_vis_dims,
  actual_vis_dims,
  requested_k,
  actual_k
))
flush.console()

if (cluster_algorithm == "walktrap") {
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
  cat(sprintf("[INFO] Clustering uses vis only (actual vis dims=%d). --dim selected file: kodama_full_%d.RData\n", actual_vis_dims, picked$dim))
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
  cat(sprintf("[INFO] Clustering uses vis only (actual vis dims=%d). --dim selected file: kodama_full_%d.RData\n", actual_vis_dims, picked$dim))
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

raw_membership <- renumber_membership(best$membership)
final_membership <- renumber_membership(best$final_membership)

raw_cluster_count <- length(unique(raw_membership))
final_cluster_count <- length(unique(final_membership))
raw_cluster_sizes <- format_cluster_sizes(raw_membership)
final_cluster_sizes <- format_cluster_sizes(final_membership)

cluster_df <- data.frame(
  label = rownames(vis),
  cluster = as.integer(final_membership),
  stringsAsFactors = FALSE
)
write.csv(cluster_df, out_csv, row.names = FALSE, quote = FALSE)

sample_id <- sub("_cluster$", "", tools::file_path_sans_ext(basename(out_csv)))
summary_path <- file.path(dirname(out_csv), paste0(sample_id, "_cluster_summary.csv"))
summary_df <- data.frame(
  sample_id = sample_id,
  cluster_profile = cluster_profile,
  vis_dims = actual_vis_dims,
  requested_dim = as.integer(selected_file_dim),
  loaded_dim = as.integer(picked$dim),
  requested_k = as.integer(requested_k),
  actual_k = as.integer(actual_k),
  cluster_algorithm = cluster_algorithm,
  leiden_objective = leiden_objective,
  landmark_cells = as.integer(landmark_cells),
  landmark_sample_strategy = landmark_sample_strategy,
  landmark_density_knn_k = as.integer(landmark_density_knn_k),
  landmark_density_power = as.numeric(landmark_density_power),
  landmark_grid_bins = as.integer(landmark_grid_bins),
  landmark_grid_max_per_bin = as.integer(landmark_grid_max_per_bin),
  landmark_assign_k = as.integer(landmark_assign_k),
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
  selected_resolution = as.numeric(best$resolution),
  raw_cluster_count = as.integer(raw_cluster_count),
  final_cluster_count = as.integer(final_cluster_count),
  merge_min_size = as.integer(merge_min_size),
  raw_cluster_sizes = raw_cluster_sizes,
  final_cluster_sizes = final_cluster_sizes,
  stringsAsFactors = FALSE
)
write.csv(summary_df, summary_path, row.names = FALSE, quote = TRUE)

pdf_path <- file.path(dirname(out_csv), paste0(sample_id, "_cluster_kodama_membership.pdf"))
pdf(pdf_path)
cluster_colors <- cluster_plot_colors(vis, final_membership, cluster_algorithm)
draw_membership_plot(
  vis,
  final_membership,
  cluster_colors,
  cluster_algorithm,
  main = sprintf("%s | %s", sample_id, cluster_algorithm)
)
dev.off()

png_path <- file.path(dirname(out_csv), paste0(sample_id, "_cluster_kodama_membership.png"))
png(filename = png_path, width = 1800, height = 1400, res = 180)
draw_membership_plot(
  vis,
  final_membership,
  cluster_colors,
  cluster_algorithm,
  main = sprintf("%s | %s", sample_id, cluster_algorithm)
)
dev.off()

cat(sprintf("[INFO] Requested --dim=%d loaded kodama_full_%d.RData\n", selected_file_dim, picked$dim))
if (!isTRUE(picked$exact)) {
  cat("[INFO] Requested dim file was not present; nearest lower available file was used.\n")
}
cat(sprintf("[INFO] Requested k=%d actual k=%d\n", requested_k, actual_k))
cat(sprintf("[INFO] Cluster algorithm=%s\n", cluster_algorithm))
cat(sprintf("[INFO] Landmark cells requested=%d used=%d strategy=%s assignment=%s assign_k=%d\n",
  as.integer(landmark_cells),
  as.integer(best$landmark_cells_used %||% NA_integer_),
  landmark_sample_strategy,
  as.character(best$landmark_assignment_mode %||% NA_character_),
  as.integer(best$landmark_assign_k_used %||% NA_integer_)
))
if (landmark_sample_strategy == "knn_inverse_distance") {
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
cat(sprintf("[INFO] Wrote: %s\n", pdf_path))
cat(sprintf("[INFO] Wrote: %s\n", png_path))
