#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("Usage: Rscript run_louvain_landmark_grid.R <kodama_full_N.RData> <outdir> [--algorithm louvain|leiden] [--landmarks N] [--graph-k N] [--assign-k N] [--resolutions comma-list] [--sample-strategy grid_balanced|density_power_grid|knn_inverse_distance|geosketch_plaid|leverage_score] [--snn-type rank|number|jaccard] [--graph-prune-quantile Q] [--grid-bins N] [--grid-max-per-bin N] [--density-power X] [--density-knn-k N] [--leiden-objective modularity|CPM]")
}

kodama_rdata <- args[1]
outdir <- args[2]
algorithm <- "louvain"
landmark_cells <- 10000L
graph_k <- 100L
assign_k <- 100L
resolution_values <- c(0.04, 0.05, 0.08, 0.12, 0.16)
sample_strategy <- "grid_balanced"
snn_type <- "rank"
graph_prune_quantile <- 0
grid_bins <- 100L
grid_max_per_bin <- 20L
density_power <- 2
density_knn_k <- 50L
leiden_objective <- "modularity"

if (length(args) > 2L) {
  i <- 3L
  while (i <= length(args)) {
    flag <- args[i]
    if (flag == "--algorithm" && i + 1L <= length(args)) {
      algorithm <- args[i + 1L]
      i <- i + 2L
      next
    }
    if (flag == "--landmarks" && i + 1L <= length(args)) {
      landmark_cells <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--graph-k" && i + 1L <= length(args)) {
      graph_k <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--assign-k" && i + 1L <= length(args)) {
      assign_k <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--resolutions" && i + 1L <= length(args)) {
      resolution_values <- as.numeric(strsplit(args[i + 1L], ",", fixed = TRUE)[[1]])
      i <- i + 2L
      next
    }
    if (flag == "--sample-strategy" && i + 1L <= length(args)) {
      sample_strategy <- args[i + 1L]
      i <- i + 2L
      next
    }
    if (flag == "--snn-type" && i + 1L <= length(args)) {
      snn_type <- args[i + 1L]
      i <- i + 2L
      next
    }
    if (flag == "--graph-prune-quantile" && i + 1L <= length(args)) {
      graph_prune_quantile <- as.numeric(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--grid-bins" && i + 1L <= length(args)) {
      grid_bins <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--grid-max-per-bin" && i + 1L <= length(args)) {
      grid_max_per_bin <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--density-power" && i + 1L <= length(args)) {
      density_power <- as.numeric(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--density-knn-k" && i + 1L <= length(args)) {
      density_knn_k <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--leiden-objective" && i + 1L <= length(args)) {
      leiden_objective <- args[i + 1L]
      i <- i + 2L
      next
    }
    i <- i + 1L
  }
}

if (!algorithm %in% c("louvain", "leiden")) stop("--algorithm must be louvain or leiden")
if (!is.finite(landmark_cells) || landmark_cells < 2L) stop("--landmarks must be >= 2")
if (!is.finite(graph_k) || graph_k < 2L) stop("--graph-k must be >= 2")
if (!is.finite(assign_k) || assign_k < 1L) stop("--assign-k must be >= 1")
if (any(!is.finite(resolution_values) | resolution_values <= 0)) stop("--resolutions must be positive numbers")
if (!sample_strategy %in% c("grid_balanced", "density_power_grid", "knn_inverse_distance", "geosketch_plaid", "leverage_score")) stop("--sample-strategy must be grid_balanced, density_power_grid, knn_inverse_distance, geosketch_plaid, or leverage_score")
if (!snn_type %in% c("rank", "number", "jaccard")) stop("--snn-type must be rank, number, or jaccard")
if (!is.finite(graph_prune_quantile) || graph_prune_quantile < 0 || graph_prune_quantile >= 1) stop("--graph-prune-quantile must be >= 0 and < 1")
if (!is.finite(grid_bins) || grid_bins < 2L) stop("--grid-bins must be >= 2")
if (!is.finite(grid_max_per_bin) || grid_max_per_bin < 1L) stop("--grid-max-per-bin must be >= 1")
if (!is.finite(density_power) || density_power <= 0) stop("--density-power must be > 0")
if (!is.finite(density_knn_k) || density_knn_k < 1L) stop("--density-knn-k must be >= 1")
if (!leiden_objective %in% c("modularity", "CPM")) stop("--leiden-objective must be modularity or CPM")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
set.seed(1L)

require_namespace <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Required package '%s' is not installed.", pkg))
  }
}
for (pkg in c("bluster", "igraph", "Rnanoflann", "cluster")) require_namespace(pkg)

load(kodama_rdata)
if (!exists("vis")) stop(sprintf("Variable 'vis' not found in %s", kodama_rdata))
vis <- as.matrix(vis)
if (ncol(vis) < 2L) stop("KODAMA 'vis' must have at least two columns")
vis <- vis[, 1:2, drop = FALSE]
storage.mode(vis) <- "double"
if (is.null(rownames(vis))) rownames(vis) <- sprintf("cell_%07d", seq_len(nrow(vis)))

cat(sprintf("[INFO] Loaded cells=%d from %s\n", nrow(vis), kodama_rdata))
cat(sprintf("[INFO] Settings: algorithm=%s landmarks=%d graph_k=%d assign_k=%d resolutions=%s sample_strategy=%s snn_type=%s graph_prune_quantile=%.3f grid_bins=%d grid_max_per_bin=%d density_power=%.3f density_knn_k=%d leiden_objective=%s\n",
  algorithm, landmark_cells, graph_k, assign_k, paste(resolution_values, collapse = ","), sample_strategy, snn_type, graph_prune_quantile, grid_bins, grid_max_per_bin, density_power, density_knn_k, leiden_objective
))

grid_balanced_sample <- function(x, max_total, bins = 100L, max_per_bin = 20L) {
  max_total <- min(as.integer(max_total), nrow(x))
  if (nrow(x) <= max_total) return(seq_len(nrow(x)))
  xr <- range(x[, 1], finite = TRUE)
  yr <- range(x[, 2], finite = TRUE)
  xb <- cut(x[, 1], breaks = seq(xr[1], xr[2], length.out = bins + 1L), include.lowest = TRUE, labels = FALSE)
  yb <- cut(x[, 2], breaks = seq(yr[1], yr[2], length.out = bins + 1L), include.lowest = TRUE, labels = FALSE)
  bin_id <- xb + bins * (yb - 1L)
  by_bin <- split(seq_len(nrow(x)), bin_id)
  idx <- unlist(lapply(by_bin, function(ii) {
    if (length(ii) <= max_per_bin) ii else sample(ii, max_per_bin)
  }), use.names = FALSE)
  idx <- sort(unique(idx))
  if (length(idx) > max_total) idx <- sort(sample(idx, max_total))
  if (length(idx) < min(1000L, max_total)) idx <- sort(sample.int(nrow(x), max_total))
  idx
}

density_power_grid_sample <- function(x, max_total, bins = 16L, power = 2, diagnostics_csv = NULL) {
  max_total <- min(as.integer(max_total), nrow(x))
  if (nrow(x) <= max_total) return(seq_len(nrow(x)))
  xr <- range(x[, 1], finite = TRUE)
  yr <- range(x[, 2], finite = TRUE)
  xb <- cut(x[, 1], breaks = seq(xr[1], xr[2], length.out = bins + 1L), include.lowest = TRUE, labels = FALSE)
  yb <- cut(x[, 2], breaks = seq(yr[1], yr[2], length.out = bins + 1L), include.lowest = TRUE, labels = FALSE)
  bin_id <- xb + bins * (yb - 1L)
  by_bin <- split(seq_len(nrow(x)), bin_id)
  counts <- vapply(by_bin, length, integer(1))
  bin_names <- names(by_bin)

  allocate_by_power <- function(counts, max_total, power) {
    n_bins <- length(counts)
    weights <- as.numeric(counts)^power
    alloc <- integer(n_bins)
    if (max_total >= n_bins) {
      alloc[] <- 1L
    } else {
      chosen <- sample(seq_len(n_bins), max_total, prob = weights)
      alloc[chosen] <- 1L
      return(alloc)
    }

    remaining <- max_total - sum(alloc)
    capacity <- as.integer(counts) - alloc
    while (remaining > 0L && any(capacity > 0L)) {
      active <- which(capacity > 0L)
      active_weights <- weights[active]
      ideal <- remaining * active_weights / sum(active_weights)
      add <- floor(ideal)
      add <- pmin(add, capacity[active])
      if (sum(add) == 0L) {
        ord <- active[order(ideal - floor(ideal), active_weights, decreasing = TRUE)]
        n_add <- min(remaining, length(ord))
        add_idx <- ord[seq_len(n_add)]
        alloc[add_idx] <- alloc[add_idx] + 1L
        capacity[add_idx] <- capacity[add_idx] - 1L
        remaining <- remaining - n_add
      } else {
        alloc[active] <- alloc[active] + add
        capacity[active] <- capacity[active] - add
        remaining <- remaining - sum(add)
      }
    }
    alloc
  }

  alloc <- allocate_by_power(counts, max_total, power)
  idx <- unlist(mapply(function(ii, n) {
    n <- as.integer(n)
    if (n <= 0L) return(integer(0))
    if (n >= length(ii)) return(ii)
    sample(ii, n)
  }, by_bin, alloc, SIMPLIFY = FALSE), use.names = FALSE)
  idx <- sort(unique(idx))
  if (length(idx) > max_total) idx <- sort(sample(idx, max_total))

  if (!is.null(diagnostics_csv)) {
    parts <- strsplit(bin_names, ".", fixed = TRUE)
    # bin_id is retained because split() drops empty bins and names are not guaranteed to be contiguous.
    diag <- data.frame(
      bin_id = as.integer(bin_names),
      samples_in_bin = as.integer(counts),
      selected_landmarks = as.integer(alloc),
      weight = as.numeric(counts)^power,
      stringsAsFactors = FALSE
    )
    diag$grid_x <- ((diag$bin_id - 1L) %% bins) + 1L
    diag$grid_y <- ((diag$bin_id - 1L) %/% bins) + 1L
    diag <- diag[order(diag$grid_y, diag$grid_x), , drop = FALSE]
    write.csv(diag, diagnostics_csv, row.names = FALSE, quote = TRUE)
  }
  idx
}

allocate_even_over_bins <- function(counts, max_total) {
  counts <- as.integer(counts)
  n_bins <- length(counts)
  alloc <- integer(n_bins)
  if (n_bins == 0L || max_total <= 0L) return(alloc)
  if (max_total >= n_bins) {
    alloc[] <- 1L
  } else {
    chosen <- sample(seq_len(n_bins), max_total)
    alloc[chosen] <- 1L
    return(alloc)
  }

  remaining <- max_total - sum(alloc)
  capacity <- counts - alloc
  while (remaining > 0L && any(capacity > 0L)) {
    active <- which(capacity > 0L)
    base_add <- min(floor(remaining / length(active)), max(capacity[active]))
    if (base_add <= 0L) {
      chosen <- sample(active, min(remaining, length(active)))
      alloc[chosen] <- alloc[chosen] + 1L
      capacity[chosen] <- capacity[chosen] - 1L
      remaining <- remaining - length(chosen)
    } else {
      add <- pmin(base_add, capacity[active])
      alloc[active] <- alloc[active] + add
      capacity[active] <- capacity[active] - add
      remaining <- remaining - sum(add)
    }
  }
  alloc
}

geosketch_plaid_sample <- function(x, max_total, bins = 100L, diagnostics_csv = NULL) {
  max_total <- min(as.integer(max_total), nrow(x))
  if (nrow(x) <= max_total) return(seq_len(nrow(x)))
  xr <- range(x[, 1], finite = TRUE)
  yr <- range(x[, 2], finite = TRUE)
  xb <- cut(x[, 1], breaks = seq(xr[1], xr[2], length.out = bins + 1L), include.lowest = TRUE, labels = FALSE)
  yb <- cut(x[, 2], breaks = seq(yr[1], yr[2], length.out = bins + 1L), include.lowest = TRUE, labels = FALSE)
  bin_id <- xb + bins * (yb - 1L)
  by_bin <- split(seq_len(nrow(x)), bin_id)
  counts <- vapply(by_bin, length, integer(1))
  alloc <- allocate_even_over_bins(counts, max_total)
  idx <- unlist(mapply(function(ii, n) {
    n <- as.integer(n)
    if (n <= 0L) return(integer(0))
    if (n >= length(ii)) return(ii)
    sample(ii, n)
  }, by_bin, alloc, SIMPLIFY = FALSE), use.names = FALSE)
  idx <- sort(unique(idx))
  if (length(idx) > max_total) idx <- sort(sample(idx, max_total))

  if (!is.null(diagnostics_csv)) {
    diag <- data.frame(
      bin_id = as.integer(names(by_bin)),
      samples_in_bin = as.integer(counts),
      selected_landmarks = as.integer(alloc),
      stringsAsFactors = FALSE
    )
    diag$grid_x <- ((diag$bin_id - 1L) %% bins) + 1L
    diag$grid_y <- ((diag$bin_id - 1L) %/% bins) + 1L
    diag <- diag[order(diag$grid_y, diag$grid_x), , drop = FALSE]
    write.csv(diag, diagnostics_csv, row.names = FALSE, quote = TRUE)
  }
  idx
}

knn_inverse_distance_sample <- function(x, max_total, k = 50L, power = 1, diagnostics_csv = NULL) {
  max_total <- min(as.integer(max_total), nrow(x))
  if (nrow(x) <= max_total) return(seq_len(nrow(x)))
  used_k <- max(1L, min(as.integer(k), nrow(x) - 1L))
  cat(sprintf("[INFO] Computing mean distance to %d nearest neighbors for landmark probabilities with inverse-distance power %.3f\n", used_k, power))
  density_time <- system.time({
    nn <- Rnanoflann::nn(data = as.matrix(x), points = as.matrix(x), k = used_k + 1L, method = "euclidean", search = "standard", trans = TRUE)
    distances <- matrix(as.numeric(nn$distances), nrow = nrow(x), ncol = used_k + 1L)
    indices <- matrix(as.integer(nn$indices), nrow = nrow(x), ncol = used_k + 1L)
    if (all(indices[, 1L] == seq_len(nrow(x))) || all(distances[, 1L] == 0)) {
      distances <- distances[, -1L, drop = FALSE]
    } else {
      distances <- distances[, seq_len(used_k), drop = FALSE]
    }
    mean_dist <- rowMeans(distances)
    rm(nn)
    gc(FALSE)
  })
  positive_dist <- mean_dist[is.finite(mean_dist) & mean_dist > 0]
  eps <- suppressWarnings(as.numeric(stats::quantile(positive_dist, probs = 0.001, names = FALSE, na.rm = TRUE)))
  if (!is.finite(eps) || eps <= 0) eps <- .Machine$double.eps
  safe_dist <- pmax(mean_dist, eps)
  weights <- (1 / safe_dist)^power
  weights[!is.finite(weights)] <- 0
  if (!any(weights > 0)) stop("No positive landmark weights after inverse-distance calculation")
  idx <- sort(sample.int(nrow(x), size = max_total, replace = FALSE, prob = weights))

  if (!is.null(diagnostics_csv)) {
    selected <- rep(FALSE, nrow(x))
    selected[idx] <- TRUE
    q_probs <- c(0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1)
    all_q <- stats::quantile(mean_dist, probs = q_probs, names = FALSE, na.rm = TRUE)
    selected_q <- stats::quantile(mean_dist[selected], probs = q_probs, names = FALSE, na.rm = TRUE)
    diag <- data.frame(
      metric = c("all_mean_knn_distance", "selected_mean_knn_distance"),
      k = used_k,
      inverse_distance_power = power,
      elapsed_seconds = as.numeric(density_time[["elapsed"]]),
      min = c(all_q[1], selected_q[1]),
      q001 = c(all_q[2], selected_q[2]),
      q01 = c(all_q[3], selected_q[3]),
      q05 = c(all_q[4], selected_q[4]),
      q10 = c(all_q[5], selected_q[5]),
      q25 = c(all_q[6], selected_q[6]),
      q50 = c(all_q[7], selected_q[7]),
      q75 = c(all_q[8], selected_q[8]),
      q90 = c(all_q[9], selected_q[9]),
      q95 = c(all_q[10], selected_q[10]),
      q99 = c(all_q[11], selected_q[11]),
      q999 = c(all_q[12], selected_q[12]),
      max = c(all_q[13], selected_q[13]),
      stringsAsFactors = FALSE
    )
    write.csv(diag, diagnostics_csv, row.names = FALSE, quote = TRUE)
  }
  idx
}

leverage_score_sample <- function(x, max_total, diagnostics_csv = NULL) {
  max_total <- min(as.integer(max_total), nrow(x))
  if (nrow(x) <= max_total) return(seq_len(nrow(x)))
  z <- scale(x)
  z[!is.finite(z)] <- 0
  sv <- svd(z, nu = min(ncol(z), nrow(z)), nv = 0)
  leverage <- rowSums(sv$u^2)
  leverage[!is.finite(leverage) | leverage < 0] <- 0
  if (!any(leverage > 0)) stop("No positive leverage scores")
  idx <- sort(sample.int(nrow(x), size = max_total, replace = FALSE, prob = leverage))

  if (!is.null(diagnostics_csv)) {
    selected <- rep(FALSE, nrow(x))
    selected[idx] <- TRUE
    q_probs <- c(0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1)
    all_q <- stats::quantile(leverage, probs = q_probs, names = FALSE, na.rm = TRUE)
    selected_q <- stats::quantile(leverage[selected], probs = q_probs, names = FALSE, na.rm = TRUE)
    diag <- data.frame(
      metric = c("all_leverage_score", "selected_leverage_score"),
      min = c(all_q[1], selected_q[1]),
      q001 = c(all_q[2], selected_q[2]),
      q01 = c(all_q[3], selected_q[3]),
      q05 = c(all_q[4], selected_q[4]),
      q10 = c(all_q[5], selected_q[5]),
      q25 = c(all_q[6], selected_q[6]),
      q50 = c(all_q[7], selected_q[7]),
      q75 = c(all_q[8], selected_q[8]),
      q90 = c(all_q[9], selected_q[9]),
      q95 = c(all_q[10], selected_q[10]),
      q99 = c(all_q[11], selected_q[11]),
      q999 = c(all_q[12], selected_q[12]),
      max = c(all_q[13], selected_q[13]),
      stringsAsFactors = FALSE
    )
    write.csv(diag, diagnostics_csv, row.names = FALSE, quote = TRUE)
  }
  idx
}

renumber <- function(x) {
  original_names <- names(x)
  x <- as.integer(x)
  ids <- sort(unique(x))
  out <- as.integer(match(x, ids))
  names(out) <- original_names
  out
}

renumber_by_size <- function(x) {
  original_names <- names(x)
  x <- as.integer(x)
  sizes <- sort(table(x), decreasing = TRUE)
  out <- as.integer(match(as.character(x), names(sizes)))
  names(out) <- original_names
  out
}

majority_vote <- function(nn_index, ref_labels) {
  votes <- matrix(ref_labels[as.vector(nn_index)], nrow = nrow(nn_index), ncol = ncol(nn_index))
  ids <- sort(unique(as.integer(ref_labels)))
  counts <- vapply(ids, function(id) rowSums(votes == id), numeric(nrow(votes)))
  as.integer(ids[max.col(counts, ties.method = "first")])
}

mean_silhouette <- function(x, labels, max_cells = 4000L) {
  if (length(unique(labels)) < 2L) return(NA_real_)
  idx <- seq_len(nrow(x))
  if (length(idx) > max_cells) idx <- sort(sample(idx, max_cells))
  out <- tryCatch(
    cluster::silhouette(as.integer(labels[idx]), stats::dist(scale(x[idx, , drop = FALSE]))),
    error = function(e) NULL
  )
  if (is.null(out)) return(NA_real_)
  out_summary <- tryCatch(summary(out), error = function(e) NULL)
  if (is.null(out_summary) || is.null(out_summary$avg.width)) return(NA_real_)
  as.numeric(out_summary$avg.width)
}

palette_for <- function(labels) {
  ids <- sort(unique(as.integer(labels)))
  base <- c("#000000", "#8B0000", "#F2D51B", "#007000", "#1F77B4", "#FF7F0E", "#9467BD", "#17BECF", "#7F7F7F")
  setNames(rep(base, length.out = length(ids)), as.character(ids))
}

plot_solution <- function(name, labels, note) {
  png_path <- file.path(outdir, paste0(name, ".png"))
  plot_idx <- if (nrow(vis) > 220000L) sort(sample.int(nrow(vis), 220000L)) else seq_len(nrow(vis))
  cols <- palette_for(labels)
  png(png_path, width = 1800, height = 1400, res = 180)
  plot(vis[plot_idx, 1], vis[plot_idx, 2],
    pch = 16, cex = 0.16,
    col = grDevices::adjustcolor("black", alpha.f = 0.20),
    xlab = "KODAMA dimension 1",
    ylab = "KODAMA dimension 2",
    main = sprintf("%s | %s", name, note)
  )
  for (cluster_id in names(cols)) {
    idx <- plot_idx[labels[plot_idx] == as.integer(cluster_id)]
    if (!length(idx)) next
    alpha <- if (identical(unname(cols[cluster_id]), "#000000")) 0.35 else 0.70
    points(vis[idx, 1], vis[idx, 2], pch = 16, cex = 0.16, col = grDevices::adjustcolor(cols[cluster_id], alpha.f = alpha))
  }
  sizes <- table(labels)
  legend("topright",
    legend = sprintf("cluster %s n=%d", names(cols), as.integer(sizes[names(cols)])),
    col = cols, pch = 16, bty = "n", cex = 0.75
  )
  dev.off()
  png_path
}

tag_count <- function(n) {
  if (n %% 1000L == 0L) return(sprintf("%dk", as.integer(n / 1000L)))
  as.character(n)
}

tag_fraction <- function(x) {
  gsub("[^0-9A-Za-z]+", "p", format(x, trim = TRUE, scientific = FALSE))
}

tag_number <- function(x) {
  gsub("[^0-9A-Za-z]+", "p", format(x, trim = TRUE, scientific = FALSE))
}

sampling_tag <- if (sample_strategy == "density_power_grid") {
  sprintf("densitygrid%dp%s", grid_bins, tag_number(density_power))
} else if (sample_strategy == "knn_inverse_distance") {
  sprintf("knninvdist%dp%s", density_knn_k, tag_number(density_power))
} else if (sample_strategy == "geosketch_plaid") {
  sprintf("geosketchplaid%d", grid_bins)
} else if (sample_strategy == "leverage_score") {
  "leveragescore"
} else {
  sprintf("gridbalanced%db%d", grid_bins, grid_max_per_bin)
}

landmark_diagnostics_csv <- file.path(outdir, paste0("landmark_sampling_", sampling_tag, ".csv"))
if (sample_strategy == "density_power_grid") {
  landmark_idx <- density_power_grid_sample(vis, landmark_cells, bins = grid_bins, power = density_power, diagnostics_csv = landmark_diagnostics_csv)
} else if (sample_strategy == "knn_inverse_distance") {
  landmark_idx <- knn_inverse_distance_sample(vis, landmark_cells, k = density_knn_k, power = density_power, diagnostics_csv = landmark_diagnostics_csv)
} else if (sample_strategy == "geosketch_plaid") {
  landmark_idx <- geosketch_plaid_sample(vis, landmark_cells, bins = grid_bins, diagnostics_csv = landmark_diagnostics_csv)
} else if (sample_strategy == "leverage_score") {
  landmark_idx <- leverage_score_sample(vis, landmark_cells, diagnostics_csv = landmark_diagnostics_csv)
} else {
  landmark_idx <- grid_balanced_sample(vis, landmark_cells, bins = grid_bins, max_per_bin = grid_max_per_bin)
}
landmark_vis <- vis[landmark_idx, , drop = FALSE]
used_graph_k <- max(2L, min(as.integer(graph_k), nrow(landmark_vis) - 1L))
used_assign_k <- max(1L, min(as.integer(assign_k), nrow(landmark_vis)))

cat(sprintf("[INFO] Selected landmarks=%d strategy=%s\n", length(landmark_idx), sampling_tag))
cat(sprintf("[INFO] Building SNN graph on landmarks with k=%d type=%s\n", used_graph_k, snn_type))
graph_time <- system.time({
  graph_obj <- bluster::makeSNNGraph(landmark_vis, k = used_graph_k, type = snn_type)
  graph_edges_before <- igraph::ecount(graph_obj)
  graph_edges_after <- graph_edges_before
  if (graph_prune_quantile > 0 && igraph::ecount(graph_obj) > 0) {
    edge_weights <- igraph::E(graph_obj)$weight
    if (!is.null(edge_weights) && any(is.finite(edge_weights))) {
      cutoff <- as.numeric(stats::quantile(edge_weights, probs = graph_prune_quantile, names = FALSE, na.rm = TRUE))
      keep <- edge_weights > cutoff
      if (sum(keep) > 0L) {
        graph_obj <- igraph::subgraph.edges(graph_obj, igraph::E(graph_obj)[keep], delete.vertices = FALSE)
        graph_edges_after <- igraph::ecount(graph_obj)
      }
      rm(edge_weights, keep)
    }
  }
})
cat(sprintf("[INFO] Graph build elapsed=%.1fs edges_before=%d edges_after=%d\n", graph_time[["elapsed"]], graph_edges_before, graph_edges_after))

cat(sprintf("[INFO] Querying %d-NN from all cells to landmarks once\n", used_assign_k))
knn_time <- system.time({
  nn <- Rnanoflann::nn(data = as.matrix(landmark_vis), points = as.matrix(vis), k = used_assign_k, method = "euclidean", search = "standard", trans = TRUE)
  nn_index <- matrix(as.integer(nn$indices), nrow = nrow(vis), ncol = used_assign_k)
  rm(nn)
  gc(FALSE)
})
cat(sprintf("[INFO] KNN query elapsed=%.1fs\n", knn_time[["elapsed"]]))

summaries <- list()
graph_tag <- if (graph_prune_quantile > 0) {
  sprintf("snn%spruneq%s", snn_type, tag_fraction(graph_prune_quantile))
} else {
  sprintf("snn%s", snn_type)
}
run_prefix <- sprintf("%s_%s_%s_landmark%s_graphk%d_assignk%d", algorithm, graph_tag, sampling_tag, tag_count(length(landmark_idx)), used_graph_k, used_assign_k)
for (res in resolution_values) {
  name <- sprintf("%s_res%03d", run_prefix, as.integer(round(res * 1000)))
  if (algorithm == "leiden") {
    note <- sprintf("Leiden %s res %.3f, %d landmarks, %s, graph k=%d, assignment k=%d", leiden_objective, res, length(landmark_idx), sampling_tag, used_graph_k, used_assign_k)
  } else {
    note <- sprintf("Louvain res %.3f, %d landmarks, %s, graph k=%d, assignment k=%d", res, length(landmark_idx), sampling_tag, used_graph_k, used_assign_k)
  }
  cat(sprintf("[INFO] Running %s\n", name))
  elapsed <- system.time({
    if (algorithm == "leiden") {
      cl <- igraph::cluster_leiden(graph_obj, objective_function = leiden_objective, resolution = res)
    } else {
      cl <- igraph::cluster_louvain(graph_obj, resolution = res)
    }
    landmark_labels <- renumber(as.integer(cl$membership))
    names(landmark_labels) <- rownames(landmark_vis)
    labels <- majority_vote(nn_index, landmark_labels)
    labels[landmark_idx] <- landmark_labels
    names(labels) <- rownames(vis)
    labels <- renumber_by_size(labels)
  })
  csv_path <- file.path(outdir, paste0(name, "_clusters.csv"))
  write.csv(data.frame(label = names(labels), cluster = as.integer(labels)), csv_path, row.names = FALSE, quote = FALSE)
  png_path <- plot_solution(name, labels, note)
  sizes <- sort(table(labels), decreasing = TRUE)
  summary_df <- data.frame(
    solution = name,
    method = algorithm,
    leiden_objective = if (algorithm == "leiden") leiden_objective else NA_character_,
    resolution = res,
    cells = nrow(vis),
    landmarks = length(landmark_idx),
    landmark_strategy = sample_strategy,
    landmark_grid_bins = grid_bins,
    landmark_density_power = density_power,
    landmark_density_knn_k = density_knn_k,
    snn_type = snn_type,
    graph_prune_quantile = graph_prune_quantile,
    graph_edges_before = graph_edges_before,
    graph_edges_after = graph_edges_after,
    graph_k = used_graph_k,
    assign_k = used_assign_k,
    final_clusters = length(unique(labels)),
    silhouette_sample = mean_silhouette(vis, labels),
    elapsed_seconds = as.numeric(elapsed[["elapsed"]]),
    graph_elapsed_seconds = as.numeric(graph_time[["elapsed"]]),
    knn_elapsed_seconds = as.numeric(knn_time[["elapsed"]]),
    cluster_sizes = paste(sprintf("%s:%d", names(sizes), as.integer(sizes)), collapse = ";"),
    png = png_path,
    csv = csv_path,
    stringsAsFactors = FALSE
  )
  summary_path <- file.path(outdir, paste0(name, "_summary.csv"))
  write.csv(summary_df, summary_path, row.names = FALSE, quote = TRUE)
  summaries[[name]] <- summary_df
  rm(labels, landmark_labels, cl)
  gc(FALSE)
}

combined <- do.call(rbind, summaries)
summary_file <- file.path(outdir, paste0(run_prefix, "_summary.csv"))
write.csv(combined, summary_file, row.names = FALSE, quote = TRUE)
cat(sprintf("[INFO] Wrote: %s\n", summary_file))
cat("[INFO] Done\n")
