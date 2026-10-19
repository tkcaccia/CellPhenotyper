#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("Usage: load_gigatime_kodama_rawdata.R <quantification_dir> <output_dir>")
}

quant_dir <- args[[1L]]
output_dir <- args[[2L]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library(data.table))

files <- list.files(
  quant_dir, recursive = TRUE, full.names = TRUE,
  pattern = "_(nuclei|cyto)_gigatime_quantification\\.csv$"
)
if (length(files) == 0L) {
  stop(sprintf("No GigaTIME per-cell quantification CSVs found under %s", quant_dir))
}

read_compartment <- function(compartment) {
  hit <- files[grepl(sprintf("_%s_gigatime_quantification\\.csv$", compartment), files)]
  if (length(hit) != 1L) {
    stop(sprintf("Expected exactly one %s quantification table; found %d", compartment, length(hit)))
  }
  header <- names(fread(hit[[1L]], nrows = 0L, showProgress = FALSE))
  marker_cols <- grep("__mean$", header, value = TRUE)
  coordinate_cols <- c("label_id", "centroid_x_px", "centroid_y_px")
  missing <- setdiff(coordinate_cols, header)
  if (length(missing) > 0L || length(marker_cols) < 2L) {
    stop(sprintf("Invalid %s GigaTIME table: missing=%s marker_means=%d", compartment, paste(missing, collapse = ","), length(marker_cols)))
  }
  dt <- fread(hit[[1L]], select = c(coordinate_cols, marker_cols), showProgress = FALSE)
  setnames(dt, marker_cols, paste0(compartment, "__", marker_cols))
  dt[, label_id := as.character(label_id)]
  dt <- dt[!is.na(label_id) & nzchar(label_id) & !duplicated(label_id)]
  dt
}

nuclei <- read_compartment("nuclei")
cyto <- read_compartment("cyto")
cyto_feature_cols <- setdiff(names(cyto), c("centroid_x_px", "centroid_y_px"))
joined <- merge(
  nuclei, cyto[, ..cyto_feature_cols],
  by = "label_id", all = FALSE, sort = TRUE
)
if (nrow(joined) < 3L) {
  stop(sprintf("Only %d cells are shared between nuclei and cytoplasm GigaTIME tables", nrow(joined)))
}

feature_cols <- grep("^(nuclei|cyto)__.*__mean$", names(joined), value = TRUE)
feature_matrix <- as.matrix(joined[, ..feature_cols])
storage.mode(feature_matrix) <- "double"
feature_matrix[!is.finite(feature_matrix)] <- NA_real_
for (j in seq_len(ncol(feature_matrix))) {
  column <- feature_matrix[, j]
  replacement <- median(column, na.rm = TRUE)
  if (!is.finite(replacement)) replacement <- 0
  column[is.na(column)] <- replacement
  feature_matrix[, j] <- column
}
rownames(feature_matrix) <- joined$label_id

ann <- data.frame(
  label = joined$label_id,
  x = as.numeric(joined$centroid_x_px),
  y = as.numeric(joined$centroid_y_px),
  polygon_label = "gigatime_markers",
  stringsAsFactors = FALSE
)
rownames(ann) <- ann$label
xy <- as.matrix(ann[, c("x", "y"), drop = FALSE])
rownames(xy) <- ann$label
ann_ids <- ann$label
common_ids <- ann_ids
available_modes <- c("tile")
selected_modes <- c("tile")
embeddings_raw <- list(tile = feature_matrix, nuclei = NULL, cyto = NULL, inner_square = NULL)
mode_overlap_with_annotations <- c(tile = length(ann_ids))

save(
  ann_ids, common_ids, ann, xy, available_modes, selected_modes,
  embeddings_raw, mode_overlap_with_annotations,
  file = file.path(output_dir, "rawdata.RData"), compress = FALSE
)
writeLines(feature_cols, file.path(output_dir, "gigatime_marker_features.txt"))
cat(sprintf("[OK] GigaTIME KODAMA input cells=%d marker_features=%d\n", nrow(feature_matrix), ncol(feature_matrix)))
