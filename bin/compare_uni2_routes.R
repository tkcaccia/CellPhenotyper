#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7L) {
  stop(paste(
    "Usage: compare_uni2_routes.R <sample_id> <grid_kodama_dir> <cell_kodama_dir>",
    "<grid_objects.csv> <cell_objects.csv> <marker_quant_dir> <output_dir>"
  ))
}

sample_id <- args[1]
grid_kodama_dir <- args[2]
cell_kodama_dir <- args[3]
grid_objects_path <- args[4]
cell_objects_path <- args[5]
marker_quant_dir <- args[6]
output_dir <- args[7]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

load_kodama_vis <- function(directory) {
  files <- list.files(directory, pattern = "^kodama_full_[0-9]+\\.RData$", full.names = TRUE)
  if (!length(files)) stop(sprintf("No kodama_full_*.RData files found in %s", directory))
  dims <- as.integer(sub("^.*kodama_full_([0-9]+)\\.RData$", "\\1", files))
  selected <- files[which.max(dims)]
  env <- new.env(parent = emptyenv())
  load(selected, envir = env)
  if (!exists("vis", envir = env, inherits = FALSE)) {
    stop(sprintf("KODAMA file does not contain vis: %s", selected))
  }
  vis <- as.matrix(env$vis)
  if (ncol(vis) < 2L) stop(sprintf("KODAMA vis must have at least two columns: %s", selected))
  ids <- rownames(vis)
  if (is.null(ids) && exists("common_ids", envir = env, inherits = FALSE)) ids <- env$common_ids
  if (is.null(ids) || length(ids) != nrow(vis)) {
    stop(sprintf("KODAMA vis has no aligned observation identifiers: %s", selected))
  }
  data.frame(
    label = as.character(ids),
    kodama_1 = as.numeric(vis[, 1]),
    kodama_2 = as.numeric(vis[, 2]),
    stringsAsFactors = FALSE
  ) -> out
  list(data = out, file = selected, dim = max(dims))
}

read_objects <- function(path, required) {
  table <- read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  missing <- setdiff(required, colnames(table))
  if (length(missing)) {
    stop(sprintf("%s is missing columns: %s", path, paste(missing, collapse = ", ")))
  }
  table$label <- as.character(table$label)
  table
}

safe_scale <- function(matrix_value) {
  matrix_value <- as.matrix(matrix_value)
  centered <- scale(matrix_value)
  centered[!is.finite(centered)] <- 0
  centered
}

procrustes_align <- function(reference, candidate) {
  cross <- crossprod(candidate, reference)
  decomposition <- svd(cross)
  rotation <- decomposition$u %*% t(decomposition$v)
  aligned <- candidate %*% rotation
  list(
    aligned = aligned,
    correlation = suppressWarnings(cor(as.vector(reference), as.vector(aligned))),
    rmse = sqrt(mean((reference - aligned)^2))
  )
}

neighbor_overlap <- function(first, second, k) {
  if (nrow(first) < 3L) return(NA_real_)
  k <- max(1L, min(as.integer(k), nrow(first) - 1L))
  d_first <- as.matrix(dist(first))
  d_second <- as.matrix(dist(second))
  diag(d_first) <- Inf
  diag(d_second) <- Inf
  scores <- vapply(seq_len(nrow(first)), function(index) {
    first_neighbors <- order(d_first[index, ], method = "radix")[seq_len(k)]
    second_neighbors <- order(d_second[index, ], method = "radix")[seq_len(k)]
    length(intersect(first_neighbors, second_neighbors)) / length(union(first_neighbors, second_neighbors))
  }, numeric(1))
  rm(d_first, d_second)
  mean(scores)
}

cross_validated_r2 <- function(predictors, response, folds = 5L, seed = 1L) {
  valid <- is.finite(response) & apply(predictors, 1L, function(row) all(is.finite(row)))
  predictors <- predictors[valid, , drop = FALSE]
  response <- response[valid]
  if (length(response) < 30L || stats::sd(response) <= .Machine$double.eps) return(NA_real_)
  set.seed(seed)
  fold_id <- sample(rep(seq_len(folds), length.out = length(response)))
  predicted <- rep(NA_real_, length(response))
  design <- cbind(intercept = 1, predictors)
  for (fold in seq_len(folds)) {
    train <- fold_id != fold
    test <- !train
    fit <- tryCatch(stats::lm.fit(design[train, , drop = FALSE], response[train]), error = function(e) NULL)
    if (is.null(fit)) return(NA_real_)
    coefficients <- fit$coefficients
    coefficients[!is.finite(coefficients)] <- 0
    predicted[test] <- as.numeric(design[test, , drop = FALSE] %*% coefficients)
  }
  denominator <- sum((response - mean(response))^2)
  if (denominator <= .Machine$double.eps) return(NA_real_)
  1 - sum((response - predicted)^2) / denominator
}

grid_loaded <- load_kodama_vis(grid_kodama_dir)
cell_loaded <- load_kodama_vis(cell_kodama_dir)
grid_objects <- read_objects(
  grid_objects_path,
  c("label", "x", "y", "core_x0", "core_y0", "core_x1", "core_y1")
)
cell_objects <- read_objects(cell_objects_path, c("label", "x", "y"))

grid_index <- match(grid_loaded$data$label, grid_objects$label)
cell_index <- match(cell_loaded$data$label, cell_objects$label)
grid_valid <- !is.na(grid_index)
cell_valid <- !is.na(cell_index)
grid_data <- cbind(grid_loaded$data[grid_valid, ], grid_objects[grid_index[grid_valid], setdiff(colnames(grid_objects), "label"), drop = FALSE])
cell_data <- cbind(cell_loaded$data[cell_valid, ], cell_objects[cell_index[cell_valid], setdiff(colnames(cell_objects), "label"), drop = FALSE])

x_starts <- sort(unique(as.numeric(grid_data$core_x0)))
y_starts <- sort(unique(as.numeric(grid_data$core_y0)))
x_bin <- findInterval(as.numeric(cell_data$x), x_starts)
y_bin <- findInterval(as.numeric(cell_data$y), y_starts)
candidate_key <- rep(NA_character_, nrow(cell_data))
inside_axis <- x_bin > 0L & y_bin > 0L & x_bin <= length(x_starts) & y_bin <= length(y_starts)
candidate_key[inside_axis] <- paste(x_starts[x_bin[inside_axis]], y_starts[y_bin[inside_axis]], sep = ":")
grid_key <- paste(grid_data$core_x0, grid_data$core_y0, sep = ":")
grid_match <- match(candidate_key, grid_key)
inside_core <- !is.na(grid_match)
inside_core[inside_core] <- (
  as.numeric(cell_data$x[inside_core]) >= as.numeric(grid_data$core_x0[grid_match[inside_core]]) &
  as.numeric(cell_data$x[inside_core]) < as.numeric(grid_data$core_x1[grid_match[inside_core]]) &
  as.numeric(cell_data$y[inside_core]) >= as.numeric(grid_data$core_y0[grid_match[inside_core]]) &
  as.numeric(cell_data$y[inside_core]) < as.numeric(grid_data$core_y1[grid_match[inside_core]])
)
cell_data$grid_label <- NA_character_
cell_data$grid_label[inside_core] <- grid_data$label[grid_match[inside_core]]
matched_cells <- cell_data[!is.na(cell_data$grid_label), , drop = FALSE]
if (nrow(matched_cells) < 10L) stop("Fewer than 10 cell observations matched retained grid cores")

cell_counts <- table(matched_cells$grid_label)
cell_sum_1 <- rowsum(matched_cells$kodama_1, matched_cells$grid_label, reorder = FALSE)
cell_sum_2 <- rowsum(matched_cells$kodama_2, matched_cells$grid_label, reorder = FALSE)
aggregate_labels <- rownames(cell_sum_1)
cell_aggregate <- data.frame(
  label = aggregate_labels,
  kodama_1 = as.numeric(cell_sum_1[, 1]) / as.numeric(cell_counts[aggregate_labels]),
  kodama_2 = as.numeric(cell_sum_2[, 1]) / as.numeric(cell_counts[aggregate_labels]),
  cell_count = as.integer(cell_counts[aggregate_labels]),
  stringsAsFactors = FALSE
)

common_labels <- intersect(grid_data$label, cell_aggregate$label)
if (length(common_labels) < 10L) stop("Fewer than 10 grid cores contain matched cell-centred observations")
grid_common <- grid_data[match(common_labels, grid_data$label), , drop = FALSE]
cell_common <- cell_aggregate[match(common_labels, cell_aggregate$label), , drop = FALSE]
grid_representation <- safe_scale(grid_common[, c("kodama_1", "kodama_2")])
cell_representation <- safe_scale(cell_common[, c("kodama_1", "kodama_2")])
alignment <- procrustes_align(grid_representation, cell_representation)

set.seed(1L)
comparison_index <- seq_along(common_labels)
if (length(comparison_index) > 1500L) comparison_index <- sort(sample(comparison_index, 1500L))
grid_sample <- grid_representation[comparison_index, , drop = FALSE]
cell_sample <- alignment$aligned[comparison_index, , drop = FALSE]
spatial_sample <- safe_scale(grid_common[comparison_index, c("x", "y")])
grid_distances <- as.numeric(dist(grid_sample))
cell_distances <- as.numeric(dist(cell_sample))
spatial_distances <- as.numeric(dist(spatial_sample))
distance_correlation <- suppressWarnings(cor(grid_distances, cell_distances, method = "spearman"))
grid_spatial_coherence <- suppressWarnings(cor(grid_distances, spatial_distances, method = "spearman"))
cell_spatial_coherence <- suppressWarnings(cor(cell_distances, spatial_distances, method = "spearman"))
knn_overlap <- neighbor_overlap(grid_sample, cell_sample, k = 15L)
rm(grid_distances, cell_distances, spatial_distances)

marker_rows <- data.frame(
  marker = character(0),
  grid_route_cv_r2 = numeric(0),
  cell_route_cv_r2 = numeric(0),
  cell_minus_grid_cv_r2 = numeric(0),
  stringsAsFactors = FALSE
)
marker_files <- if (dir.exists(marker_quant_dir)) {
  list.files(marker_quant_dir, pattern = "_nuclei_gigatime_mean_intensity\\.csv$", full.names = TRUE)
} else {
  character(0)
}
if (length(marker_files)) {
  marker_table <- read.csv(marker_files[1], check.names = FALSE, stringsAsFactors = FALSE)
  if ("label_id" %in% colnames(marker_table)) {
    marker_table$label_id <- as.character(marker_table$label_id)
    marker_grid <- matched_cells$grid_label[match(marker_table$label_id, matched_cells$label)]
    metadata_columns <- c(
      "label_id", "mask_name", "area_px", "centroid_y_px", "centroid_x_px",
      "bbox_ymin_px", "bbox_xmin_px", "bbox_ymax_px", "bbox_xmax_px"
    )
    marker_columns <- setdiff(colnames(marker_table), metadata_columns)
    marker_columns <- marker_columns[vapply(marker_table[marker_columns], is.numeric, logical(1))]
    for (marker in marker_columns) {
      valid_marker <- !is.na(marker_grid) & is.finite(marker_table[[marker]])
      if (sum(valid_marker) < 30L) next
      marker_sums <- rowsum(marker_table[[marker]][valid_marker], marker_grid[valid_marker], reorder = FALSE)
      marker_counts <- table(marker_grid[valid_marker])
      marker_means <- as.numeric(marker_sums[, 1]) / as.numeric(marker_counts[rownames(marker_sums)])
      names(marker_means) <- rownames(marker_sums)
      response <- marker_means[common_labels]
      grid_r2 <- cross_validated_r2(grid_representation, response)
      cell_r2 <- cross_validated_r2(cell_representation, response)
      marker_rows <- rbind(marker_rows, data.frame(
        marker = marker,
        grid_route_cv_r2 = grid_r2,
        cell_route_cv_r2 = cell_r2,
        cell_minus_grid_cv_r2 = cell_r2 - grid_r2,
        stringsAsFactors = FALSE
      ))
    }
  }
}

marker_path <- file.path(output_dir, paste0(sample_id, "_uni2_route_marker_endpoints.csv"))
write.csv(marker_rows, marker_path, row.names = FALSE, quote = TRUE)
mean_grid_marker_r2 <- if (nrow(marker_rows)) mean(marker_rows$grid_route_cv_r2, na.rm = TRUE) else NA_real_
mean_cell_marker_r2 <- if (nrow(marker_rows)) mean(marker_rows$cell_route_cv_r2, na.rm = TRUE) else NA_real_

summary <- data.frame(
  metric = c(
    "grid_observations", "cell_observations", "matched_cells", "matched_cell_fraction",
    "common_grid_cores", "procrustes_correlation", "procrustes_rmse",
    "pairwise_distance_spearman", "knn15_neighborhood_jaccard",
    "grid_route_spatial_distance_spearman", "cell_route_spatial_distance_spearman",
    "virtual_marker_endpoints", "grid_route_mean_marker_cv_r2", "cell_route_mean_marker_cv_r2"
  ),
  value = as.character(c(
    nrow(grid_data), nrow(cell_data), nrow(matched_cells), nrow(matched_cells) / nrow(cell_data),
    length(common_labels), alignment$correlation, alignment$rmse,
    distance_correlation, knn_overlap, grid_spatial_coherence, cell_spatial_coherence,
    nrow(marker_rows), mean_grid_marker_r2, mean_cell_marker_r2
  )),
  interpretation = c(
    rep("coverage", 5),
    "cross_route_representation_agreement", "cross_route_representation_agreement",
    "cross_route_representation_agreement", "cross_route_local_structure_agreement",
    "within_route_spatial_coherence", "within_route_spatial_coherence",
    "biological_proxy_coverage", "virtual_marker_cross_validated_prediction", "virtual_marker_cross_validated_prediction"
  ),
  stringsAsFactors = FALSE
)
summary_path <- file.path(output_dir, paste0(sample_id, "_uni2_route_comparison.csv"))
write.csv(summary, summary_path, row.names = FALSE, quote = TRUE)

png_path <- file.path(output_dir, paste0(sample_id, "_uni2_route_comparison.png"))
png(png_path, width = 2000, height = 1500, res = 180)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
palette_values <- grDevices::hcl.colors(100, "Viridis")
color_index <- function(values) {
  breaks <- quantile(values, probs = seq(0, 1, length.out = 101), na.rm = TRUE)
  breaks <- unique(breaks)
  if (length(breaks) < 2L) return(rep(palette_values[50], length(values)))
  palette_values[pmax(1L, pmin(100L, findInterval(values, breaks, all.inside = TRUE)))]
}
plot(grid_common$x, grid_common$y, pch = 16, cex = 0.5, col = color_index(grid_representation[, 1]),
     xlab = "x (px)", ylab = "y (px)", main = "Grid route: KODAMA 1", asp = 1)
plot(grid_common$x, grid_common$y, pch = 16, cex = 0.5, col = color_index(cell_representation[, 1]),
     xlab = "x (px)", ylab = "y (px)", main = "Cell route aggregated to grid", asp = 1)
plot(grid_sample[, 1], grid_sample[, 2], pch = 16, cex = 0.5, col = "#176B87AA",
     xlab = "KODAMA 1", ylab = "KODAMA 2", main = "Grid representation")
plot(cell_sample[, 1], cell_sample[, 2], pch = 16, cex = 0.5, col = "#C44E32AA",
     xlab = "Aligned KODAMA 1", ylab = "Aligned KODAMA 2", main = "Cell representation (Procrustes aligned)")
dev.off()

report_path <- file.path(output_dir, paste0(sample_id, "_uni2_route_comparison.md"))
report <- c(
  paste0("# UNI-2 route comparison: ", sample_id),
  "",
  "## Scope",
  "",
  "Grid observations and cell-centred observations are distinct analysis units. Cell-centred KODAMA coordinates were aggregated within retained grid cores before comparison; independently fitted KODAMA axes were Procrustes-aligned.",
  "",
  "## Predefined endpoints",
  "",
  sprintf("- Matched cells: %d of %d (%.1f%%).", nrow(matched_cells), nrow(cell_data), 100 * nrow(matched_cells) / nrow(cell_data)),
  sprintf("- Common grid cores: %d.", length(common_labels)),
  sprintf("- Procrustes correlation: %.4f; RMSE: %.4f.", alignment$correlation, alignment$rmse),
  sprintf("- Pairwise-distance Spearman correlation: %.4f.", distance_correlation),
  sprintf("- 15-neighbor mean Jaccard overlap: %.4f.", knn_overlap),
  sprintf("- Spatial-distance correlation: grid %.4f; cell aggregated %.4f.", grid_spatial_coherence, cell_spatial_coherence),
  sprintf("- Virtual-marker proxy endpoints available: %d.", nrow(marker_rows)),
  "",
  "## Interpretation constraint",
  "",
  "This report does not select a preferred route automatically. Virtual-marker prediction is a biological proxy derived from the same image, not an independent reference measurement. Route selection requires predefined study-specific endpoints, independent annotations or assays, and blinded review."
)
writeLines(report, report_path)

cat(sprintf("[INFO] Compared %d common grid cores and %d matched cells\n", length(common_labels), nrow(matched_cells)))
cat(sprintf("[INFO] Wrote %s\n", output_dir))
