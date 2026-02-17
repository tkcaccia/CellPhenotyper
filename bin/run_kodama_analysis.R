#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop(
    paste(
      "Usage: Rscript run_kodama_analysis.R",
      "<rawdata_rdata> <output_dir>",
      "[--embedding-mode tile|nuclei|cyto|inner_square|all|tile,nuclei,...]",
      "[--dims-to-run N] [--spark-top N] [--n-cores N]"
    )
  )
}

rawdata_path <- args[1]
output_dir <- args[2]

embedding_mode <- "all"
dims_to_run <- 20L
spark_top <- 100L
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
if (!is.finite(n_cores) || n_cores < 1L) {
  n_cores <- 1L
}

normalize_modes <- function(mode_string) {
  x <- tolower(trimws(mode_string))
  if (!nzchar(x) || x == "all") {
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

if (!file.exists(rawdata_path)) {
  stop(paste("Raw data RData does not exist:", rawdata_path))
}

load(rawdata_path)

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
  cat(sprintf("[INFO] mode=%s selected_features=%d\n", m, ncol(mat)))
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

data_scaled <- scale(data)
data_scaled[is.na(data_scaled)] <- 0

max_nv <- min(50L, nrow(data_scaled) - 1L, ncol(data_scaled) - 1L)
if (!is.finite(max_nv) || max_nv < 2L) {
  stop(
    paste(
      "Not enough observations/features for PCA after preprocessing:",
      "nrow=", nrow(data_scaled), "ncol=", ncol(data_scaled)
    )
  )
}

pca_res <- irlba(A = data_scaled, nv = max_nv)
pca <- pca_res$u %*% diag(pca_res$d)
rownames(pca) <- common_ids

lab <- as.factor(ann[, "polygon_label"])

# Keep raw embedding families intact and add analysis-level selected matrices.
r_tile <- mode_mats[["tile"]]
r_nuclei <- mode_mats[["nuclei"]]
r_cyto <- mode_mats[["cyto"]]
r_inner <- mode_mats[["inner_square"]]
r_full <- r_tile
embeddings_raw <- list(
  tile = r_tile,
  nuclei = r_nuclei,
  cyto = r_cyto,
  inner_square = r_inner
)
ann_ids <- ann_full$label
mode_overlap_with_annotations <- vapply(
  available_modes,
  function(m) length(intersect(ann_ids, rownames(embeddings_raw[[m]]))),
  integer(1)
)

# Overwrite rawdata with both full raw inputs and selected analysis objects.
save(
  ann_ids, common_ids, ann_full, xy_full, ann, xy, available_modes, selected_modes,
  embeddings_raw, mode_overlap_with_annotations,
  r_tile, r_full, r_nuclei, r_cyto, r_inner,
  r_tile_selected, r_nuclei_selected, r_cyto_selected, r_inner_selected,
  data, data_scaled, lab,
  file = rawdata_path
)
legacy_path <- file.path(output_dir, "raw_data.RData")
save(
  ann_ids, common_ids, ann_full, xy_full, ann, xy, available_modes, selected_modes,
  embeddings_raw, mode_overlap_with_annotations,
  r_tile, r_full, r_nuclei, r_cyto, r_inner,
  r_tile_selected, r_nuclei_selected, r_cyto_selected, r_inner_selected,
  data, data_scaled, lab,
  file = legacy_path
)

pca_pdf <- file.path(output_dir, paste0("pca_full_", ncol(pca), ".pdf"))
pdf(pca_pdf)
plot(pca, pch = 20, col = lab)
dev.off()

pca_rdata <- file.path(output_dir, paste0("pca_full_", ncol(pca), ".RData"))
save(pca, xy, file = pca_rdata)

u <- umap(pca)$layout
rownames(u) <- common_ids
umap_pdf <- file.path(output_dir, paste0("umap_full_", ncol(pca), ".pdf"))
pdf(umap_pdf)
plot(u, pch = 20, col = lab, cex = 0.5)
dev.off()

umap_rdata <- file.path(output_dir, paste0("umap_full_", ncol(pca), ".RData"))
save(u, xy, ann, common_ids, lab, file = umap_rdata)

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

jj <- KODAMA.matrix.parallel(
  pca[, seq_len(dims_use), drop = FALSE],
  spatial = spatial_for_kodama,
  landmarks = min(1000L, nrow(pca)),
  n.cores = as.integer(n_cores),
  seed = 543210,
  ancestry = FALSE
)

config <- umap.defaults
config$n_neighbors <- min(30L, nrow(pca) - 1L)
config$n_threads <- as.integer(n_cores)
vis <- KODAMA.visualization(jj, config = config)
rownames(vis) <- common_ids

kodama_pdf <- file.path(output_dir, paste0("kodama_full_", dims_use, ".pdf"))
pdf(kodama_pdf)
plot(vis, pch = 20, col = lab, cex = 0.5)
dev.off()

kodama_rdata <- file.path(output_dir, paste0("kodama_full_", dims_use, ".RData"))
save(vis, xy, common_ids, ann, lab, selected_modes, file = kodama_rdata)

cat(sprintf("[INFO] Shared cells=%d\n", length(common_ids)))
cat(sprintf("[INFO] Selected modes=%s\n", paste(selected_modes, collapse = ",")))
cat(sprintf("[INFO] Wrote: %s\n", rawdata_path))
cat(sprintf("[INFO] Wrote: %s\n", kodama_rdata))
