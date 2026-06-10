#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("Usage: Rscript compare_kodama_clustering_solutions.R <kodama_full_N.RData> <outdir>")
}

kodama_rdata <- args[1]
outdir <- args[2]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

set.seed(1L)

require_namespace <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(sprintf("Required package '%s' is not installed.", pkg))
  }
}

for (pkg in c("Rnanoflann", "cluster")) {
  require_namespace(pkg)
}

load(kodama_rdata)
if (!exists("vis")) {
  stop(sprintf("Variable 'vis' not found in %s", kodama_rdata))
}
vis <- as.matrix(vis)
if (ncol(vis) < 2L) {
  stop("KODAMA 'vis' must have at least two columns.")
}
vis <- vis[, 1:2, drop = FALSE]
storage.mode(vis) <- "double"
if (is.null(rownames(vis))) {
  rownames(vis) <- sprintf("cell_%07d", seq_len(nrow(vis)))
}

cat(sprintf("[INFO] Loaded %d cells from %s\n", nrow(vis), kodama_rdata))

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

random_sample <- function(x, max_total) {
  max_total <- min(as.integer(max_total), nrow(x))
  if (nrow(x) <= max_total) seq_len(nrow(x)) else sort(sample.int(nrow(x), max_total))
}

renumber <- function(x) {
  x <- as.integer(x)
  ids <- sort(unique(x))
  out <- as.integer(match(x, ids))
  names(out) <- names(x)
  out
}

renumber_by_size <- function(x) {
  x <- as.integer(x)
  sizes <- sort(table(x), decreasing = TRUE)
  out <- as.integer(match(as.character(x), names(sizes)))
  names(out) <- names(x)
  out
}

majority_vote <- function(nn_index, ref_labels) {
  votes <- matrix(ref_labels[as.vector(nn_index)], nrow = nrow(nn_index), ncol = ncol(nn_index))
  ids <- sort(unique(as.integer(ref_labels)))
  counts <- vapply(ids, function(id) rowSums(votes == id), numeric(nrow(votes)))
  as.integer(ids[max.col(counts, ties.method = "first")])
}

assign_by_knn <- function(landmark_x, landmark_labels, all_x, k = 5L) {
  k <- max(1L, min(as.integer(k), nrow(landmark_x)))
  nn <- Rnanoflann::nn(data = as.matrix(landmark_x), points = as.matrix(all_x), k = k, method = "euclidean", search = "standard", trans = TRUE)
  nn_index <- matrix(as.integer(nn$indices), nrow = nrow(all_x), ncol = k)
  out <- majority_vote(nn_index, landmark_labels)
  rm(nn)
  gc(FALSE)
  renumber(out)
}

assign_by_centers <- function(centers, all_x) {
  centers <- as.matrix(centers)
  best <- integer(nrow(all_x))
  best_dist <- rep(Inf, nrow(all_x))
  for (i in seq_len(nrow(centers))) {
    d <- (all_x[, 1] - centers[i, 1])^2 + (all_x[, 2] - centers[i, 2])^2
    take <- d < best_dist
    best[take] <- i
    best_dist[take] <- d[take]
  }
  renumber(best)
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
  base <- c(
    "#000000", "#8B0000", "#F2D51B", "#007000", "#1F77B4", "#FF7F0E",
    "#9467BD", "#17BECF", "#7F7F7F", "#E377C2", "#BCBD22", "#8C564B"
  )
  setNames(rep(base, length.out = length(ids)), as.character(ids))
}

plot_solution <- function(name, labels, method_note) {
  png_path <- file.path(outdir, paste0(name, ".png"))
  plot_idx <- if (nrow(vis) > 220000L) sort(sample.int(nrow(vis), 220000L)) else seq_len(nrow(vis))
  cols <- palette_for(labels)
  draw <- function() {
    plot(vis[plot_idx, 1], vis[plot_idx, 2],
      pch = 16, cex = 0.16,
      col = grDevices::adjustcolor("black", alpha.f = 0.20),
      xlab = "KODAMA dimension 1",
      ylab = "KODAMA dimension 2",
      main = sprintf("%s | %s", name, method_note)
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
  }
  png(png_path, width = 1800, height = 1400, res = 180)
  draw()
  dev.off()
  list(png = png_path)
}

write_solution <- function(name, labels, params, elapsed) {
  labels <- renumber_by_size(labels)
  names(labels) <- rownames(vis)
  csv_path <- file.path(outdir, paste0(name, "_clusters.csv"))
  summary_path <- file.path(outdir, paste0(name, "_summary.csv"))
  write.csv(data.frame(label = names(labels), cluster = as.integer(labels)), csv_path, row.names = FALSE, quote = FALSE)
  plot_paths <- plot_solution(name, labels, params$method_note)
  sizes <- sort(table(labels), decreasing = TRUE)
  summary_df <- data.frame(
    solution = name,
    method = params$method,
    method_note = params$method_note,
    cells = nrow(vis),
    final_clusters = length(unique(labels)),
    silhouette_sample = mean_silhouette(vis, labels),
    elapsed_seconds = as.numeric(elapsed[["elapsed"]]),
    cluster_sizes = paste(sprintf("%s:%d", names(sizes), as.integer(sizes)), collapse = ";"),
    png = plot_paths$png,
    csv = csv_path,
    stringsAsFactors = FALSE
  )
  for (nm in setdiff(names(params), c("method", "method_note"))) {
    summary_df[[nm]] <- params[[nm]]
  }
  write.csv(summary_df, summary_path, row.names = FALSE, quote = TRUE)
  summary_df
}

run_graph_solution <- function(name, algorithm, clusters = 4L, resolution = NA_real_, sample_strategy = "grid", landmarks = 100000L, assign_k = 5L, snn_k = 30L) {
  require_namespace("bluster")
  require_namespace("igraph")
  idx <- if (sample_strategy == "random") random_sample(vis, landmarks) else grid_balanced_sample(vis, landmarks)
  landmark_x <- vis[idx, , drop = FALSE]
  graph_k <- max(2L, min(as.integer(snn_k), nrow(landmark_x) - 1L))
  graph <- bluster::makeSNNGraph(landmark_x, k = graph_k)
  if (algorithm == "walktrap") {
    fit <- igraph::cluster_walktrap(graph)
    landmark_labels <- renumber(as.integer(igraph::cut_at(fit, no = as.integer(clusters))))
    note <- sprintf("walktrap %d clusters, %s %d landmarks, %d-NN propagation", clusters, sample_strategy, length(idx), assign_k)
  } else if (algorithm == "louvain") {
    fit <- igraph::cluster_louvain(graph, resolution = resolution)
    landmark_labels <- renumber(as.integer(fit$membership))
    note <- sprintf("louvain res %.3f, %s %d landmarks, %d-NN propagation", resolution, sample_strategy, length(idx), assign_k)
  } else {
    stop(sprintf("Unsupported graph algorithm: %s", algorithm))
  }
  labels <- assign_by_knn(landmark_x, landmark_labels, vis, k = assign_k)
  labels[idx] <- landmark_labels
  rm(graph, landmark_x, landmark_labels)
  gc(FALSE)
  list(
    labels = renumber(labels),
    params = list(
      method = algorithm,
      method_note = note,
      landmarks = length(idx),
      sample_strategy = sample_strategy,
      assign_k = assign_k,
      snn_k = graph_k,
      target_clusters = if (algorithm == "walktrap") clusters else NA_integer_,
      resolution = if (algorithm == "louvain") resolution else NA_real_
    )
  )
}

run_mclust_solution <- function(name, clusters = 4L, model = "VVV", train_n = 50000L, sample_strategy = "grid", scaled = FALSE) {
  require_namespace("mclust")
  suppressPackageStartupMessages(library(mclust))
  idx <- if (sample_strategy == "random") random_sample(vis, train_n) else grid_balanced_sample(vis, train_n, bins = 90L, max_per_bin = 10L)
  x_all <- if (scaled) scale(vis) else vis
  x_train <- x_all[idx, , drop = FALSE]
  init_subset <- if (nrow(x_train) > 5000L) sample.int(nrow(x_train), 5000L) else NULL
  fit <- mclust::Mclust(
    x_train,
    G = as.integer(clusters),
    modelNames = model,
    initialization = if (is.null(init_subset)) NULL else list(subset = init_subset),
    verbose = FALSE
  )
  bic_value <- tryCatch(as.numeric(fit$bic[as.character(clusters), model]), error = function(e) NA_real_)
  if (length(bic_value) != 1L || !is.finite(bic_value)) {
    bic_value <- suppressWarnings(max(as.numeric(fit$bic), na.rm = TRUE))
  }
  pred <- predict(fit, newdata = x_all)
  labels <- renumber(as.integer(pred$classification))
  list(
    labels = labels,
    params = list(
      method = "mclust",
      method_note = sprintf("mclust %s G=%d, %s train n=%d%s", model, clusters, sample_strategy, length(idx), if (scaled) ", scaled" else ""),
      train_n = length(idx),
      sample_strategy = sample_strategy,
      mclust_model = model,
      target_clusters = clusters,
      bic = bic_value,
      scaled = scaled
    )
  )
}

run_kmeans_solution <- function(name, clusters = 4L, train_n = 100000L, sample_strategy = "grid", scaled = FALSE) {
  idx <- if (sample_strategy == "random") random_sample(vis, train_n) else grid_balanced_sample(vis, train_n)
  x_all <- if (scaled) scale(vis) else vis
  x_train <- x_all[idx, , drop = FALSE]
  fit <- kmeans(x_train, centers = as.integer(clusters), iter.max = 100L, nstart = 25L)
  labels <- assign_by_centers(fit$centers, x_all)
  list(
    labels = labels,
    params = list(
      method = "kmeans",
      method_note = sprintf("kmeans K=%d, %s train n=%d%s", clusters, sample_strategy, length(idx), if (scaled) ", scaled" else ""),
      train_n = length(idx),
      sample_strategy = sample_strategy,
      target_clusters = clusters,
      scaled = scaled,
      tot_withinss = fit$tot.withinss
    )
  )
}

run_dbscan_solution <- function(name, min_pts = 30L, eps_quantile = 0.95, train_n = 50000L, sample_strategy = "grid", assign_k = 5L) {
  require_namespace("dbscan")
  idx <- if (sample_strategy == "random") random_sample(vis, train_n) else grid_balanced_sample(vis, train_n, bins = 90L, max_per_bin = 10L)
  landmark_x <- vis[idx, , drop = FALSE]
  knn_dist <- dbscan::kNNdist(landmark_x, k = as.integer(min_pts))
  eps <- as.numeric(stats::quantile(knn_dist, probs = eps_quantile, na.rm = TRUE))
  fit <- dbscan::dbscan(landmark_x, eps = eps, minPts = as.integer(min_pts))
  landmark_labels <- as.integer(fit$cluster)
  if (all(landmark_labels == 0L)) {
    landmark_labels[] <- 1L
  } else {
    noise <- landmark_labels == 0L
    landmark_labels[noise] <- max(landmark_labels) + 1L
  }
  landmark_labels <- renumber(landmark_labels)
  labels <- assign_by_knn(landmark_x, landmark_labels, vis, k = assign_k)
  labels[idx] <- landmark_labels
  list(
    labels = labels,
    params = list(
      method = "dbscan",
      method_note = sprintf("DBSCAN eps=q%.2f %.4f minPts=%d, %s %d landmarks, %d-NN propagation", eps_quantile, eps, min_pts, sample_strategy, length(idx), assign_k),
      train_n = length(idx),
      sample_strategy = sample_strategy,
      min_pts = min_pts,
      eps_quantile = eps_quantile,
      eps = eps,
      assign_k = assign_k
    )
  )
}

solutions <- list(
  list(name = "S01_walktrap4_grid100k_knn5", fun = function() run_graph_solution("S01_walktrap4_grid100k_knn5", "walktrap", clusters = 4L, sample_strategy = "grid", landmarks = 100000L, assign_k = 5L)),
  list(name = "S02_walktrap5_grid100k_knn5", fun = function() run_graph_solution("S02_walktrap5_grid100k_knn5", "walktrap", clusters = 5L, sample_strategy = "grid", landmarks = 100000L, assign_k = 5L)),
  list(name = "S03_walktrap4_random100k_knn5", fun = function() run_graph_solution("S03_walktrap4_random100k_knn5", "walktrap", clusters = 4L, sample_strategy = "random", landmarks = 100000L, assign_k = 5L)),
  list(name = "S04_louvain_res005_grid100k_knn5", fun = function() run_graph_solution("S04_louvain_res005_grid100k_knn5", "louvain", resolution = 0.05, sample_strategy = "grid", landmarks = 100000L, assign_k = 5L)),
  list(name = "S05_louvain_res004_grid100k_knn5", fun = function() run_graph_solution("S05_louvain_res004_grid100k_knn5", "louvain", resolution = 0.04, sample_strategy = "grid", landmarks = 100000L, assign_k = 5L)),
  list(name = "S06_mclust4_VVV_grid50k", fun = function() run_mclust_solution("S06_mclust4_VVV_grid50k", clusters = 4L, model = "VVV", train_n = 50000L, sample_strategy = "grid")),
  list(name = "S07_mclust5_VVV_grid50k", fun = function() run_mclust_solution("S07_mclust5_VVV_grid50k", clusters = 5L, model = "VVV", train_n = 50000L, sample_strategy = "grid")),
  list(name = "S08_mclust4_EEE_grid50k", fun = function() run_mclust_solution("S08_mclust4_EEE_grid50k", clusters = 4L, model = "EEE", train_n = 50000L, sample_strategy = "grid")),
  list(name = "S09_mclust4_VVV_grid50k_scaled", fun = function() run_mclust_solution("S09_mclust4_VVV_grid50k_scaled", clusters = 4L, model = "VVV", train_n = 50000L, sample_strategy = "grid", scaled = TRUE)),
  list(name = "S10_kmeans4_grid100k", fun = function() run_kmeans_solution("S10_kmeans4_grid100k", clusters = 4L, train_n = 100000L, sample_strategy = "grid")),
  list(name = "S11_kmeans5_grid100k", fun = function() run_kmeans_solution("S11_kmeans5_grid100k", clusters = 5L, train_n = 100000L, sample_strategy = "grid")),
  list(name = "S12_dbscan_q95_min30_grid50k_knn5", fun = function() run_dbscan_solution("S12_dbscan_q95_min30_grid50k_knn5", min_pts = 30L, eps_quantile = 0.95, train_n = 50000L, sample_strategy = "grid", assign_k = 5L)),
  list(name = "S13_louvain_res004_grid100k_snn100_knn5", fun = function() run_graph_solution("S13_louvain_res004_grid100k_snn100_knn5", "louvain", resolution = 0.04, sample_strategy = "grid", landmarks = 100000L, assign_k = 5L, snn_k = 100L)),
  list(name = "S14_louvain_res005_grid100k_snn100_knn5", fun = function() run_graph_solution("S14_louvain_res005_grid100k_snn100_knn5", "louvain", resolution = 0.05, sample_strategy = "grid", landmarks = 100000L, assign_k = 5L, snn_k = 100L)),
  list(name = "S15_louvain_res008_grid100k_snn100_knn5", fun = function() run_graph_solution("S15_louvain_res008_grid100k_snn100_knn5", "louvain", resolution = 0.08, sample_strategy = "grid", landmarks = 100000L, assign_k = 5L, snn_k = 100L)),
  list(name = "S16_walktrap4_grid50k_snn100_knn5", fun = function() run_graph_solution("S16_walktrap4_grid50k_snn100_knn5", "walktrap", clusters = 4L, sample_strategy = "grid", landmarks = 50000L, assign_k = 5L, snn_k = 100L)),
  list(name = "S17_walktrap5_grid50k_snn100_knn5", fun = function() run_graph_solution("S17_walktrap5_grid50k_snn100_knn5", "walktrap", clusters = 5L, sample_strategy = "grid", landmarks = 50000L, assign_k = 5L, snn_k = 100L))
)

all_summaries <- list()
for (solution in solutions) {
  name <- solution$name
  existing_summary <- file.path(outdir, paste0(name, "_summary.csv"))
  existing_png <- file.path(outdir, paste0(name, ".png"))
  if (file.exists(existing_summary) && file.exists(existing_png)) {
    existing <- tryCatch(read.csv(existing_summary, stringsAsFactors = FALSE), error = function(e) NULL)
    if (!is.null(existing) && (!("error" %in% names(existing)) || all(is.na(existing$error) | existing$error == ""))) {
      cat(sprintf("[INFO] Skipping completed %s\n", name))
      all_summaries[[name]] <- existing
      next
    }
  }
  cat(sprintf("[INFO] Running %s\n", name))
  flush.console()
  elapsed <- system.time({
    result <- tryCatch(solution$fun(), error = function(e) e)
  })
  if (inherits(result, "error")) {
    cat(sprintf("[ERROR] %s failed: %s\n", name, conditionMessage(result)))
    fail_df <- data.frame(
      solution = name,
      method = NA_character_,
      method_note = NA_character_,
      cells = nrow(vis),
      final_clusters = NA_integer_,
      silhouette_sample = NA_real_,
      elapsed_seconds = as.numeric(elapsed[["elapsed"]]),
      cluster_sizes = NA_character_,
      error = conditionMessage(result),
      stringsAsFactors = FALSE
    )
    write.csv(fail_df, file.path(outdir, paste0(name, "_summary.csv")), row.names = FALSE, quote = TRUE)
    all_summaries[[name]] <- fail_df
    next
  }
  all_summaries[[name]] <- write_solution(name, result$labels, result$params, elapsed)
  rm(result)
  gc(FALSE)
}

all_cols <- unique(unlist(lapply(all_summaries, names), use.names = FALSE))
combined <- do.call(rbind, lapply(all_summaries, function(df) {
  missing <- setdiff(all_cols, names(df))
  for (nm in missing) df[[nm]] <- NA
  df[, all_cols, drop = FALSE]
}))
write.csv(combined, file.path(outdir, "all_solutions_summary.csv"), row.names = FALSE, quote = TRUE)

pngs <- file.path(outdir, paste0(vapply(solutions, `[[`, character(1), "name"), ".png"))
pngs <- pngs[file.exists(pngs)]
if (length(pngs) > 0L && requireNamespace("png", quietly = TRUE) && requireNamespace("abind", quietly = TRUE)) {
  imgs <- lapply(pngs, png::readPNG)
  thumbs <- lapply(imgs, function(img) {
    h <- dim(img)[1]
    w <- dim(img)[2]
    y <- unique(round(seq(1, h, length.out = 350)))
    x <- unique(round(seq(1, w, length.out = 450)))
    img[y, x, , drop = FALSE]
  })
  ncol_panel <- 3L
  nrow_panel <- ceiling(length(thumbs) / ncol_panel)
  blank <- array(1, dim = dim(thumbs[[1]]))
  rows <- vector("list", nrow_panel)
  for (r in seq_len(nrow_panel)) {
    cells <- vector("list", ncol_panel)
    for (c in seq_len(ncol_panel)) {
      idx <- (r - 1L) * ncol_panel + c
      cells[[c]] <- if (idx <= length(thumbs)) thumbs[[idx]] else blank
    }
    rows[[r]] <- do.call(abind::abind, c(cells, along = 2L))
  }
  panel <- do.call(abind::abind, c(rows, along = 1L))
  png::writePNG(panel, file.path(outdir, "all_solutions_contact_sheet.png"))
}

cat(sprintf("[INFO] Wrote comparison summary: %s\n", file.path(outdir, "all_solutions_summary.csv")))
cat("[INFO] Done\n")
