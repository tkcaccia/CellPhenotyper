#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop(
    paste(
      "Usage: Rscript load_kodama_rawdata.R",
      "<tile_embeddings_dir> <cyto_embeddings_dir> <inner_square_embeddings_dir> <nuclei_embeddings_dir>",
      "<objects_assigned_csv> <output_dir>"
    )
  )
}

tile_dir <- args[1]
cyto_dir <- args[2]
inner_square_dir <- args[3]
nuclei_dir <- args[4]
annot_csv <- args[5]
output_dir <- args[6]

library(data.table)

load_embedding_matrix <- function(dir_path, mode_name) {
  if (!dir.exists(dir_path)) {
    stop(paste("Embedding directory does not exist for mode", mode_name, ":", dir_path))
  }

  files <- list.files(
    dir_path,
    recursive = TRUE,
    full.names = TRUE,
    pattern = "\\.csv(\\.gz)?$"
  )
  if (length(files) == 0L) {
    stop(paste("No embedding CSV files found for mode", mode_name, "in", dir_path))
  }

  dt <- rbindlist(lapply(files, fread), fill = TRUE, use.names = TRUE)
  if (!("cell_id" %in% colnames(dt))) {
    stop(paste("Embedding table for mode", mode_name, "is missing 'cell_id' column"))
  }

  feat_cols <- grep("^feat", colnames(dt), value = TRUE)
  if (length(feat_cols) == 0L) {
    stop(paste("Embedding table for mode", mode_name, "has no feature columns starting with 'feat'"))
  }

  dt <- dt[, c("cell_id", feat_cols), with = FALSE]
  dt[, cell_id := as.character(cell_id)]
  dt <- dt[!is.na(cell_id) & nzchar(cell_id)]
  dt <- dt[!duplicated(cell_id)]

  mat <- as.matrix(dt[, ..feat_cols])
  storage.mode(mat) <- "double"
  rownames(mat) <- dt$cell_id

  list(
    matrix = mat,
    n_cells = nrow(mat),
    n_features = ncol(mat)
  )
}

if (!file.exists(annot_csv)) {
  stop(paste("Annotation CSV does not exist:", annot_csv))
}

ann <- read.csv(annot_csv, stringsAsFactors = FALSE)
if (!("label" %in% colnames(ann))) {
  stop("Annotation CSV must contain a 'label' column")
}
if (!all(c("x", "y") %in% colnames(ann))) {
  stop("Annotation CSV must contain 'x' and 'y' columns")
}

ann$label <- as.character(ann$label)
ann$x <- as.numeric(ann$x)
ann$y <- as.numeric(ann$y)
ann <- ann[!is.na(ann$label) & nzchar(ann$label), , drop = FALSE]
ann <- ann[!duplicated(ann$label), , drop = FALSE]
rownames(ann) <- ann$label

if (!("polygon_label" %in% colnames(ann))) {
  ann$polygon_label <- "unknown"
}

mode_dirs <- list(
  tile = tile_dir,
  nuclei = nuclei_dir,
  cyto = cyto_dir,
  inner_square = inner_square_dir
)

available_modes <- c("tile", "nuclei", "cyto", "inner_square")
cat(sprintf("[INFO] Loading embedding families: %s\n", paste(available_modes, collapse = ",")))

loaded <- list()
for (m in available_modes) {
  loaded[[m]] <- load_embedding_matrix(mode_dirs[[m]], m)
  cat(sprintf("[INFO] mode=%s cells=%d features=%d\n", m, loaded[[m]]$n_cells, loaded[[m]]$n_features))
}

ann_ids <- sort(unique(ann$label))
xy <- as.matrix(ann[, c("x", "y"), drop = FALSE])
storage.mode(xy) <- "double"
rownames(xy) <- ann$label

# Keep full embedding matrices as loaded, without forcing a global intersection.
r_tile <- loaded[["tile"]]$matrix
r_full <- r_tile
r_nuclei <- loaded[["nuclei"]]$matrix
r_cyto <- loaded[["cyto"]]$matrix
r_inner <- loaded[["inner_square"]]$matrix
embeddings_raw <- list(
  tile = r_tile,
  nuclei = r_nuclei,
  cyto = r_cyto,
  inner_square = r_inner
)

mode_overlap_with_annotations <- vapply(
  available_modes,
  function(m) length(intersect(ann_ids, rownames(embeddings_raw[[m]]))),
  integer(1)
)
cat(
  sprintf(
    "[INFO] overlap_with_annotations: %s\n",
    paste(sprintf("%s=%d", names(mode_overlap_with_annotations), mode_overlap_with_annotations), collapse = ", ")
  )
)

# Backward-compatible object: global intersection across all loaded modes.
common_ids <- Reduce(
  intersect,
  c(list(ann_ids), lapply(embeddings_raw, rownames))
)
common_ids <- sort(unique(common_ids))
selected_modes <- available_modes

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

rawdata_path <- file.path(output_dir, "rawdata.RData")
save(
  ann_ids, common_ids, ann, xy, available_modes, selected_modes,
  embeddings_raw, mode_overlap_with_annotations,
  r_tile, r_full, r_nuclei, r_cyto, r_inner,
  file = rawdata_path
)

# Keep backward-compatible filename as well.
legacy_path <- file.path(output_dir, "raw_data.RData")
save(
  ann_ids, common_ids, ann, xy, available_modes, selected_modes,
  embeddings_raw, mode_overlap_with_annotations,
  r_tile, r_full, r_nuclei, r_cyto, r_inner,
  file = legacy_path
)

cat(sprintf("[INFO] Annotation cells=%d\n", length(ann_ids)))
cat(sprintf("[INFO] Global shared cells (all modes)=%d\n", length(common_ids)))
cat(sprintf("[INFO] Saved embedding families=%s\n", paste(available_modes, collapse = ",")))
cat(sprintf("[INFO] Wrote: %s\n", rawdata_path))
