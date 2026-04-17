args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("Usage: Rscript Rcode_Clustering.R <kodama_dir> <out_csv> [--dim N (selects kodama_full_N.RData only)] [--k N] [--resolution auto|X] [--profile standard|fine]")
}

kodama_dir <- args[1]
out_csv <- args[2]

selected_file_dim <- 20L
requested_k <- 50L
resolution_mode <- "auto"
fixed_resolution <- 0.03
resolution_grid <- c(0.005, 0.01, 0.02, 0.03, 0.04, 0.05)
score_margin <- 0.015
cluster_profile <- "standard"
fine_resolution_multiplier <- 1.35
fine_score_margin <- 0.03
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
    i <- i + 1L
  }
}

if (!is.finite(selected_file_dim) || selected_file_dim < 2L) {
  stop("--dim must be an integer >= 2.")
}
if (!is.finite(requested_k) || requested_k < 2L) {
  stop("--k must be an integer >= 2.")
}
if (resolution_mode == "fixed" && (!is.finite(fixed_resolution) || fixed_resolution <= 0)) {
  stop("--resolution must be 'auto' or a positive number.")
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

require_namespace <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Required package '%s' is not installed.", pkg))
  }
}

require_namespace("bluster")
require_namespace("igraph")
if (resolution_mode == "auto") {
  require_namespace("cluster")
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

run_louvain <- function(graph_obj, resolution_value) {
  set.seed(1L)
  cl <- igraph::cluster_louvain(graph_obj, resolution = resolution_value)
  membership <- renumber_membership(as.integer(cl$membership))
  sizes <- table(membership)
  tiny_threshold <- max(merge_min_size_floor, ceiling(merge_min_size_fraction * sum(sizes)))
  tiny_count <- sum(as.integer(sizes) < tiny_threshold)
  tiny_fraction <- if (length(sizes) == 0L) 0 else sum(as.integer(sizes)[as.integer(sizes) < tiny_threshold]) / sum(sizes)
  list(
    resolution = resolution_value,
    membership = membership,
    raw_cluster_count = length(sizes),
    modularity = igraph::modularity(graph_obj, membership = membership),
    silhouette = NA_real_,
    tiny_count = tiny_count,
    tiny_fraction = tiny_fraction,
    tiny_threshold = tiny_threshold,
    score = NA_real_
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

  near_counts <- vapply(near_best, function(x) x$raw_cluster_count, integer(1))
  near_best <- near_best[near_counts == min(near_counts)]

  near_tiny_fraction <- vapply(near_best, function(x) x$tiny_fraction, numeric(1))
  near_best <- near_best[near_tiny_fraction == min(near_tiny_fraction)]

  near_tiny_count <- vapply(near_best, function(x) x$tiny_count, integer(1))
  near_best <- near_best[near_tiny_count == min(near_tiny_count)]

  near_sil <- vapply(near_best, function(x) if (is.finite(x$silhouette)) x$silhouette else -1, numeric(1))
  near_best <- near_best[near_sil == max(near_sil)]

  near_best[[which.max(vapply(near_best, function(x) x$modularity, numeric(1)))]]
}

select_auto_fine <- function(evals, base_best, fine_score_margin) {
  more <- evals[vapply(
    evals,
    function(x) x$raw_cluster_count > base_best$raw_cluster_count && x$score >= (base_best$score - fine_score_margin),
    logical(1)
  )]
  if (length(more) == 0L) {
    more <- evals[vapply(evals, function(x) x$resolution > base_best$resolution, logical(1))]
  }
  if (length(more) == 0L) {
    return(base_best)
  }

  counts <- vapply(more, function(x) x$raw_cluster_count, integer(1))
  more <- more[counts == min(counts)]

  tiny_fraction <- vapply(more, function(x) x$tiny_fraction, numeric(1))
  more <- more[tiny_fraction == min(tiny_fraction)]

  tiny_count <- vapply(more, function(x) x$tiny_count, integer(1))
  more <- more[tiny_count == min(tiny_count)]

  sil <- vapply(more, function(x) if (is.finite(x$silhouette)) x$silhouette else -1, numeric(1))
  more <- more[sil == max(sil)]

  more[[which.max(vapply(more, function(x) x$modularity, numeric(1)))]]
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

vis <- vis[, seq_len(min(2L, ncol(vis))), drop = FALSE]
actual_vis_dims <- ncol(vis)
actual_k <- max(2L, min(as.integer(requested_k), nrow(vis) - 1L))
graph_obj <- bluster::makeSNNGraph(vis, k = actual_k)

if (resolution_mode == "auto") {
  cluster_cap <- preferred_cluster_cap(nrow(vis))
  evals <- lapply(resolution_grid, function(res) {
    out <- run_louvain(graph_obj, res)
    out$silhouette <- mean_silhouette(vis, out$membership)
    sil_term <- if (is.finite(out$silhouette)) out$silhouette else -1
    over_cap <- max(0L, out$raw_cluster_count - cluster_cap)
    out$score <- sil_term +
      0.20 * out$modularity -
      0.08 * out$raw_cluster_count -
      0.60 * out$tiny_fraction -
      0.05 * out$tiny_count -
      0.12 * over_cap
    out$cluster_cap <- cluster_cap
    out
  })
  base_best <- select_auto_best(evals, cluster_cap, score_margin)
  best <- if (cluster_profile == "fine") {
    select_auto_fine(evals, base_best, fine_score_margin)
  } else {
    base_best
  }

  cat(sprintf("[INFO] Clustering uses vis only (actual vis dims=%d). --dim selected file: kodama_full_%d.RData\n", actual_vis_dims, picked$dim))
  cat(sprintf("[INFO] Cluster profile: %s\n", cluster_profile))
  cat(sprintf("[INFO] Auto-resolution grid: %s\n", paste(sprintf("%.3f", resolution_grid), collapse = ", ")))
  cat(sprintf("[INFO] Preferred raw cluster cap for auto mode: %d\n", cluster_cap))
  cat("[INFO] Auto-resolution candidates:\n")
  for (item in evals) {
    cat(sprintf(
      "  - res=%.3f raw_clusters=%d silhouette=%s modularity=%.4f tiny=%d tiny_fraction=%.4f over_cap=%d score=%.4f\n",
      item$resolution,
      item$raw_cluster_count,
      ifelse(is.finite(item$silhouette), sprintf("%.4f", item$silhouette), "NA"),
      item$modularity,
      item$tiny_count,
      item$tiny_fraction,
      max(0L, item$raw_cluster_count - cluster_cap),
      item$score
    ))
  }
  cat(sprintf(
    "[INFO] Selected auto resolution %.3f with raw_clusters=%d (score=%.4f)\n",
    best$resolution,
    best$raw_cluster_count,
    best$score
  ))
} else {
  effective_resolution <- fixed_resolution
  if (cluster_profile == "fine") {
    effective_resolution <- fixed_resolution * fine_resolution_multiplier
  }
  best <- run_louvain(graph_obj, effective_resolution)
  cat(sprintf("[INFO] Clustering uses vis only (actual vis dims=%d). --dim selected file: kodama_full_%d.RData\n", actual_vis_dims, picked$dim))
  cat(sprintf("[INFO] Cluster profile: %s\n", cluster_profile))
  cat(sprintf("[INFO] Fixed resolution %.3f with raw_clusters=%d modularity=%.4f\n", best$resolution, best$raw_cluster_count, best$modularity))
}

raw_membership <- renumber_membership(best$membership)
merge_min_size <- max(merge_min_size_floor, ceiling(merge_min_size_fraction * nrow(vis)))
merged <- merge_small_clusters(vis, raw_membership, merge_min_size)
final_membership <- merged$membership

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
plot(vis, pch = 20, col = as.integer(final_membership), cex = 1)
dev.off()

png_path <- file.path(dirname(out_csv), paste0(sample_id, "_cluster_kodama_membership.png"))
png(filename = png_path, width = 1800, height = 1400, res = 180)
plot(vis, pch = 20, col = as.integer(final_membership), cex = 1)
dev.off()

cat(sprintf("[INFO] Requested --dim=%d loaded kodama_full_%d.RData\n", selected_file_dim, picked$dim))
if (!isTRUE(picked$exact)) {
  cat("[INFO] Requested dim file was not present; nearest lower available file was used.\n")
}
cat(sprintf("[INFO] Requested k=%d actual k=%d\n", requested_k, actual_k))
cat(sprintf("[INFO] Raw clusters=%d final clusters=%d merge_min_size=%d\n", raw_cluster_count, final_cluster_count, merge_min_size))
cat(sprintf("[INFO] Raw cluster sizes=%s\n", raw_cluster_sizes))
cat(sprintf("[INFO] Final cluster sizes=%s\n", final_cluster_sizes))
cat(sprintf("[INFO] Wrote: %s\n", out_csv))
cat(sprintf("[INFO] Wrote: %s\n", summary_path))
cat(sprintf("[INFO] Wrote: %s\n", pdf_path))
cat(sprintf("[INFO] Wrote: %s\n", png_path))
