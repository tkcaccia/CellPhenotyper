#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop(
    paste(
      "Usage: Rscript run_kodama_analysis.R",
      "<rawdata_rdata> <output_dir>",
      "[--embedding-mode tile|nuclei|cyto|inner_square|all|full|tile,inner_square,...]",
      "[--dims-to-run N] [--spark-top N] [--landmarks N] [--kodama-ncomp N] [--n-cores N]"
    )
  )
}

rawdata_path <- args[1]
output_dir <- args[2]

embedding_mode <- "all"
dims_to_run <- 20L
spark_top <- 100L
landmarks <- 1000L
kodama_ncomp <- 2L
n_cores <- 4L

if (length(args) > 2) {
  i <- 3L
  while (i <= length(args)) {
    flag <- args[i]
    if (flag == "--embedding-mode" && i + 1L <= length(args)) {
      embedding_mode <- args[i + 1L]
      i <- i + 2L
      next
    }
    if (flag == "--dims-to-run" && i + 1L <= length(args)) {
      dims_to_run <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--spark-top" && i + 1L <= length(args)) {
      spark_top <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--landmarks" && i + 1L <= length(args)) {
      landmarks <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--kodama-ncomp" && i + 1L <= length(args)) {
      kodama_ncomp <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--n-cores" && i + 1L <= length(args)) {
      n_cores <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    i <- i + 1L
  }
}

if (!is.finite(dims_to_run) || dims_to_run < 2L) {
  dims_to_run <- 20L
}
if (!is.finite(spark_top) || spark_top < 1L) {
  spark_top <- 100L
}
if (!is.finite(landmarks) || landmarks < 1L) {
  landmarks <- 1000L
}
if (!is.finite(kodama_ncomp) || kodama_ncomp < 1L) {
  kodama_ncomp <- 2L
}
if (!is.finite(n_cores) || n_cores < 1L) {
  n_cores <- 1L
}

normalize_modes <- function(mode_string) {
  x <- tolower(trimws(mode_string))
  if (!nzchar(x) || x %in% c("all", "default")) {
    return(c("tile", "inner_square"))
  }
  if (x %in% c("full", "all4", "all_four", "full_stack")) {
    return(c("tile", "nuclei", "cyto", "inner_square"))
  }
  tokens <- unlist(strsplit(gsub("\\+", ",", x), ",", fixed = FALSE), use.names = FALSE)
  tokens <- trimws(tokens)
  tokens <- tokens[nzchar(tokens)]
  mapped <- vapply(tokens, function(tk) {
    if (tk %in% c("tile", "full", "full_tile", "full-tile")) return("tile")
    if (tk %in% c("nuclei", "nucleus", "nuclear", "label", "labels")) return("nuclei")
    if (tk %in% c("cyto", "cytoplasm")) return("cyto")
    if (tk %in% c("inner", "inner_square", "inner-square", "square")) return("inner_square")
    stop(paste("Unknown embedding mode token:", tk))
  }, character(1))
  unique(mapped)
}

library(KODAMA)
library(KODAMAextra)
library(SPARK)
library(data.table)
library(irlba)
library(umap)

plot_max_cells <- 200000L
kodama_exact_max_cells <- 200000L
kodama_projection_max_cells <- 50000L
kodama_projection_min_cells <- 10000L
kodama_projection_neighbors <- 5L

if (!file.exists(rawdata_path)) {
  stop(paste("Raw data RData does not exist:", rawdata_path))
}

rawdata_size_gb <- file.info(rawdata_path)$size / (1024^3)
cat(sprintf("[INFO] Loading rawdata RData: %s (%.2f GiB)\n", rawdata_path, rawdata_size_gb))
flush.console()
load(rawdata_path)
cat("[INFO] Loaded rawdata RData\n")
flush.console()

required_objects <- c("ann", "xy")
missing_required <- required_objects[!vapply(required_objects, exists, logical(1), inherits = TRUE)]
if (length(missing_required) > 0L) {
  stop(paste("Missing required objects in rawdata:", paste(missing_required, collapse = ", ")))
}

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

if (!("label" %in% colnames(ann))) {
  ann$label <- rownames(ann)
}
ann$label <- as.character(ann$label)
ann <- ann[!is.na(ann$label) & nzchar(ann$label), , drop = FALSE]
ann <- ann[!duplicated(ann$label), , drop = FALSE]
rownames(ann) <- ann$label

if (!all(c("x", "y") %in% colnames(ann))) {
  stop("Annotation table in rawdata must contain columns 'x' and 'y'")
}
if (!("polygon_label" %in% colnames(ann))) {
  ann$polygon_label <- "unknown"
}

ann_full <- ann
xy_full <- as.matrix(ann_full[, c("x", "y"), drop = FALSE])
storage.mode(xy_full) <- "double"
rownames(xy_full) <- ann_full$label

select_features <- function(mat, xy_coords, top_n, cores) {
  if (is.null(mat)) {
    return(NULL)
  }
  if (ncol(mat) <= top_n) {
    return(mat)
  }

  idx <- tryCatch(
    multi_SPARKX(
      mat,
      xy_coords,
      as.factor(rep(1, nrow(mat))),
      n.cores = cores
    ),
    error = function(e) NULL
  )

  if (!is.null(idx)) {
    idx <- suppressWarnings(as.integer(idx))
    idx <- idx[is.finite(idx)]
    idx <- idx[idx >= 1L & idx <= ncol(mat)]
    idx <- unique(idx)
    if (length(idx) > 0L) {
      keep <- idx[seq_len(min(top_n, length(idx)))]
      return(mat[, keep, drop = FALSE])
    }
  }

  # Fallback: select highest-variance features when SPARKX is not stable.
  v <- apply(mat, 2, var, na.rm = TRUE)
  ord <- order(v, decreasing = TRUE)
  keep <- ord[seq_len(min(top_n, length(ord)))]
  mat[, keep, drop = FALSE]
}

mode_mats <- if (exists("embeddings_raw") && is.list(embeddings_raw)) {
  list(
    tile = embeddings_raw[["tile"]],
    nuclei = embeddings_raw[["nuclei"]],
    cyto = embeddings_raw[["cyto"]],
    inner_square = embeddings_raw[["inner_square"]]
  )
} else {
  list(
    tile = if (exists("r_tile")) r_tile else NULL,
    nuclei = if (exists("r_nuclei")) r_nuclei else NULL,
    cyto = if (exists("r_cyto")) r_cyto else NULL,
    inner_square = if (exists("r_inner")) r_inner else NULL
  )
}
rm(list = intersect(c("embeddings_raw", "r_tile", "r_full", "r_nuclei", "r_cyto", "r_inner"), ls()))
gc(FALSE)

sanitize_mode_matrix <- function(mat, mode_name) {
  if (is.null(mat)) {
    return(NULL)
  }
  mat <- as.matrix(mat)
  storage.mode(mat) <- "double"
  ids <- rownames(mat)
  if (is.null(ids)) {
    stop(paste("Embedding matrix for mode", mode_name, "is missing row names (cell IDs)."))
  }
  ids <- as.character(ids)
  keep <- !is.na(ids) & nzchar(ids)
  if (!all(keep)) {
    mat <- mat[keep, , drop = FALSE]
    ids <- ids[keep]
  }
  dup <- duplicated(ids)
  if (any(dup)) {
    mat <- mat[!dup, , drop = FALSE]
    ids <- ids[!dup]
  }
  rownames(mat) <- ids
  mat
}

for (m in names(mode_mats)) {
  mode_mats[[m]] <- sanitize_mode_matrix(mode_mats[[m]], m)
}

available_modes <- names(Filter(Negate(is.null), mode_mats))
selected_modes <- normalize_modes(embedding_mode)
missing_modes <- setdiff(selected_modes, available_modes)
if (length(missing_modes) > 0L) {
  stop(paste("Requested embedding mode(s) missing from rawdata:", paste(missing_modes, collapse = ",")))
}
cat(sprintf("[INFO] KODAMA embedding_mode=%s\n", paste(selected_modes, collapse = ",")))

common_ids <- Reduce(
  intersect,
  c(list(ann_full$label), lapply(selected_modes, function(m) rownames(mode_mats[[m]])))
)
common_ids <- sort(unique(common_ids))
if (length(common_ids) < 3L) {
  stop(
    paste(
      "Not enough shared cells between selected embeddings and annotations.",
      "selected_modes=", paste(selected_modes, collapse = ","),
      "shared_cells=", length(common_ids)
    )
  )
}

ann <- ann_full[common_ids, , drop = FALSE]
xy <- xy_full[common_ids, , drop = FALSE]

mats_selected <- list()
for (m in selected_modes) {
  mat <- mode_mats[[m]]
  if (is.null(mat)) {
    stop(paste("Selected mode", m, "is missing in raw data matrix set"))
  }
  mat <- mat[common_ids, , drop = FALSE]
  mat <- select_features(mat, xy_coords = xy, top_n = as.integer(spark_top), cores = as.integer(n_cores))
  colnames(mat) <- paste(m, colnames(mat), sep = "__")
  mats_selected[[m]] <- mat
  mode_mats[[m]] <- NULL
  cat(sprintf("[INFO] mode=%s selected_features=%d\n", m, ncol(mat)))
  rm(mat)
  gc(FALSE)
}

r_tile_selected <- mats_selected[["tile"]]
r_nuclei_selected <- mats_selected[["nuclei"]]
r_cyto_selected <- mats_selected[["cyto"]]
r_inner_selected <- mats_selected[["inner_square"]]

data_parts <- Filter(
  Negate(is.null),
  list(r_tile_selected, r_nuclei_selected, r_cyto_selected, r_inner_selected)
)
data <- if (length(data_parts) > 0L) do.call(cbind, data_parts) else NULL
if (is.null(data) || ncol(data) < 3L) {
  stop("Combined embedding matrix has fewer than 3 features after selection.")
}
rm(
  list = intersect(
    c(
      "data_parts", "mode_mats", "mats_selected",
      "r_tile_selected", "r_nuclei_selected", "r_cyto_selected", "r_inner_selected"
    ),
    ls()
  )
)
gc(FALSE)

requested_pca_components <- as.integer(dims_to_run)
if (!is.finite(requested_pca_components) || requested_pca_components < 2L) {
  requested_pca_components <- 20L
}
max_nv <- min(requested_pca_components, nrow(data) - 1L, ncol(data) - 1L)
if (!is.finite(max_nv) || max_nv < 2L) {
  stop(
    paste(
      "Not enough observations/features for PCA after preprocessing:",
      "nrow=", nrow(data), "ncol=", ncol(data)
    )
  )
}

data_center <- colMeans(data, na.rm = TRUE)
data_scale <- apply(data, 2, sd, na.rm = TRUE)
data_center[!is.finite(data_center)] <- 0
data_scale[!is.finite(data_scale) | data_scale <= 0] <- 1

pca_res <- irlba(A = data, nv = max_nv, center = data_center, scale = data_scale)
pca <- pca_res$u %*% diag(pca_res$d)
rownames(pca) <- common_ids
cat(sprintf("[INFO] PCA components computed=%d requested=%d\n", ncol(pca), requested_pca_components))

lab <- as.factor(ann[, "polygon_label"])

rm(
  list = intersect(
    c(
      "data", "data_parts", "mode_mats", "mats_selected", "data_center", "data_scale",
      "r_tile_selected", "r_nuclei_selected", "r_cyto_selected", "r_inner_selected",
      "embeddings_raw", "r_tile", "r_full", "r_nuclei", "r_cyto", "r_inner",
      "ann_full", "xy_full", "ann_ids", "mode_overlap_with_annotations"
    ),
    ls()
  )
)
gc(FALSE)
cat("[INFO] Released raw embedding matrices after PCA\n")

pca_pdf <- file.path(output_dir, paste0("pca_full_", ncol(pca), ".pdf"))
pdf(pca_pdf)
pca_plot_idx <- seq_len(nrow(pca))
if (length(pca_plot_idx) > plot_max_cells) {
  set.seed(543210)
  pca_plot_idx <- sort(sample.int(nrow(pca), plot_max_cells))
}
plot(pca[pca_plot_idx, 1], pca[pca_plot_idx, 2], pch = 20, col = lab[pca_plot_idx], cex = 0.35)
dev.off()

pca_rdata <- file.path(output_dir, paste0("pca_full_", ncol(pca), ".RData"))
save(pca, xy, file = pca_rdata)

if (nrow(pca) <= plot_max_cells) {
  u <- umap::umap(pca)$layout
  rownames(u) <- common_ids
  umap_pdf <- file.path(output_dir, paste0("umap_full_", ncol(pca), ".pdf"))
  pdf(umap_pdf)
  plot(u, pch = 20, col = lab, cex = 0.5)
  dev.off()

  umap_rdata <- file.path(output_dir, paste0("umap_full_", ncol(pca), ".RData"))
  save(u, xy, ann, common_ids, lab, file = umap_rdata)
} else {
  umap_skip_path <- file.path(output_dir, paste0("umap_full_", ncol(pca), "_skipped.txt"))
  writeLines(
    sprintf(
      "Skipped full pre-KODAMA UMAP preview for %d cells; KODAMA visualization is still generated for all cells.",
      nrow(pca)
    ),
    umap_skip_path
  )
  cat(sprintf("[INFO] Skipped full pre-KODAMA UMAP preview for %d cells\n", nrow(pca)))
}

dims_use <- min(as.integer(dims_to_run), ncol(pca))
if (!is.finite(dims_use) || dims_use < 2L) {
  dims_use <- min(20L, ncol(pca))
}
if (dims_use < 2L) {
  stop("KODAMA requires at least 2 PCA dimensions.")
}

spatial_for_kodama <- as.matrix(xy)
storage.mode(spatial_for_kodama) <- "double"
rownames(spatial_for_kodama) <- common_ids
kodama_landmarks <- min(as.integer(landmarks), nrow(pca))
kodama_ncomp <- min(as.integer(kodama_ncomp), dims_use)
cat(sprintf("[INFO] KODAMA landmarks=%d\n", kodama_landmarks))
cat(sprintf("[INFO] KODAMA internal ncomp=%d\n", kodama_ncomp))

config <- umap::umap.defaults
config$n_threads <- as.integer(n_cores)

if (nrow(pca) <= kodama_exact_max_cells) {
  config$n_neighbors <- min(30L, nrow(pca) - 1L)
  jj <- KODAMA.matrix(
    pca[, seq_len(dims_use), drop = FALSE],
    spatial = spatial_for_kodama,
    landmarks = kodama_landmarks,
    n.cores = as.integer(n_cores),
    seed = 543210,
    ancestry = FALSE,
    ncomp = kodama_ncomp
  )
  vis <- KODAMA.visualization(jj, config = config)
  rownames(vis) <- common_ids
} else {
  if (!requireNamespace("BiocNeighbors", quietly = TRUE)) {
    stop("BiocNeighbors is required for landmark-projected KODAMA on large cell sets.")
  }
  projection_cells <- min(
    nrow(pca),
    kodama_projection_max_cells,
    max(kodama_projection_min_cells, as.integer(kodama_landmarks) * 20L)
  )
  set.seed(543210)
  projection_idx <- sort(sample.int(nrow(pca), projection_cells))
  projection_ids <- common_ids[projection_idx]
  cat(
    sprintf(
      "[INFO] Large cell set (%d cells): running exact KODAMA on %d sampled cells, then projecting all cells with %d-NN in PCA space.\n",
      nrow(pca), projection_cells, min(kodama_projection_neighbors, projection_cells)
    )
  )

  pca_kodama <- pca[projection_idx, seq_len(dims_use), drop = FALSE]
  spatial_kodama <- spatial_for_kodama[projection_idx, , drop = FALSE]
  rownames(pca_kodama) <- projection_ids
  rownames(spatial_kodama) <- projection_ids
  config$n_neighbors <- min(30L, nrow(pca_kodama) - 1L)
  jj <- KODAMA.matrix(
    pca_kodama,
    spatial = spatial_kodama,
    landmarks = min(kodama_landmarks, nrow(pca_kodama)),
    n.cores = as.integer(n_cores),
    seed = 543210,
    ancestry = FALSE,
    ncomp = kodama_ncomp
  )
  vis_kodama <- KODAMA.visualization(jj, config = config)
  rownames(vis_kodama) <- projection_ids

  nn_k <- min(kodama_projection_neighbors, nrow(pca_kodama))
  nn <- BiocNeighbors::queryKNN(
    X = pca_kodama,
    query = pca[, seq_len(dims_use), drop = FALSE],
    k = nn_k,
    BNPARAM = BiocNeighbors::KmknnParam(),
    num.threads = as.integer(n_cores)
  )
  weights <- 1 / pmax(nn$distance, 1e-6)
  weights <- weights / rowSums(weights)

  vis <- matrix(0, nrow = nrow(pca), ncol = ncol(vis_kodama))
  for (j in seq_len(nn_k)) {
    vis <- vis + vis_kodama[nn$index[, j], , drop = FALSE] * weights[, j]
  }
  rownames(vis) <- common_ids
  colnames(vis) <- colnames(vis_kodama)
  rm(pca_kodama, spatial_kodama, vis_kodama, nn, weights)
  gc(FALSE)
}

kodama_pdf <- file.path(output_dir, paste0("kodama_full_", dims_use, ".pdf"))
pdf(kodama_pdf)
vis_plot_idx <- seq_len(nrow(vis))
if (length(vis_plot_idx) > plot_max_cells) {
  set.seed(543210)
  vis_plot_idx <- sort(sample.int(nrow(vis), plot_max_cells))
}
plot(vis[vis_plot_idx, 1], vis[vis_plot_idx, 2], pch = 20, col = lab[vis_plot_idx], cex = 0.35)
dev.off()

kodama_rdata <- file.path(output_dir, paste0("kodama_full_", dims_use, ".RData"))
save(vis, xy, common_ids, ann, lab, selected_modes, file = kodama_rdata)

cat(sprintf("[INFO] Shared cells=%d\n", length(common_ids)))
cat(sprintf("[INFO] Selected modes=%s\n", paste(selected_modes, collapse = ",")))
cat(sprintf("[INFO] Read: %s\n", rawdata_path))
cat(sprintf("[INFO] Wrote: %s\n", kodama_rdata))
