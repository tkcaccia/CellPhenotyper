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

strict_marker_read <- function(path, ...) {
  withCallingHandlers(
    fread(path, ..., header = TRUE, check.names = FALSE, strip.white = FALSE,
          na.strings = NULL, encoding = "UTF-8", showProgress = FALSE),
    warning = function(warning) stop("Marker CSV parsing warning: ", conditionMessage(warning))
  )
}

finite_marker_numeric <- function(values, description) {
  if (!is.numeric(values)) stop(description, " must be finite numeric values")
  values <- withCallingHandlers(as.numeric(values),
    warning = function(warning) stop(description, ": ", conditionMessage(warning)))
  if (any(!is.finite(values))) stop(description, " must be finite numeric values")
  values
}

read_compartment <- function(compartment) {
  hit <- files[grepl(sprintf("_%s_gigatime_quantification\\.csv$", compartment), files)]
  if (length(hit) != 1L) {
    stop(sprintf("Expected exactly one %s quantification table; found %d", compartment, length(hit)))
  }
  header <- names(strict_marker_read(hit[[1L]], nrows = 0L))
  if (anyNA(header) || any(!nzchar(header)) || anyDuplicated(header)) {
    stop(sprintf("%s marker table contains empty or duplicate CSV header names", compartment))
  }
  marker_cols <- grep("__mean$", header, value = TRUE)
  marker_cols <- setdiff(marker_cols, c("TRITC__mean", "Cy5__mean"))
  coordinate_cols <- c("label_id", "centroid_x_px", "centroid_y_px")
  missing <- setdiff(coordinate_cols, header)
  if (length(missing) > 0L || length(marker_cols) < 2L) {
    stop(sprintf("Invalid %s GigaTIME table: missing=%s marker_means=%d", compartment, paste(missing, collapse = ","), length(marker_cols)))
  }
  dt <- strict_marker_read(hit[[1L]], select = c(coordinate_cols, marker_cols), colClasses = list(character = "label_id"))
  # IDs are parsed as strings from the start: 001, 1, and the literal NA are
  # distinct observations. Never trim, coerce to a number, or keep-first.
  if (!is.character(dt$label_id) || anyNA(dt$label_id) || any(!nzchar(dt$label_id)) ||
      any(grepl("[[:space:][:cntrl:]\\p{Z}\\p{Cc}]", dt$label_id, perl = TRUE)) || anyDuplicated(dt$label_id)) {
    stop(sprintf("%s marker table contains empty, whitespace/control, or duplicate cell IDs; no rows are silently discarded", compartment))
  }
  # Each mask has its own centroid. Both must be valid, but whole-cell growth
  # may legitimately move its centroid relative to the nucleus.
  for (column in coordinate_cols[-1L]) {
    set(dt, j = column, value = finite_marker_numeric(dt[[column]], paste(compartment, column)))
  }
  for (column in marker_cols) {
    set(dt, j = column, value = finite_marker_numeric(dt[[column]],
      paste("Missing or nonfinite marker features: no silent imputation is permitted for marker KODAMA;", compartment, column)))
  }
  setnames(dt, marker_cols, paste0(compartment, "__", marker_cols))
  attr(dt, "mean_marker_schema") <- marker_cols
  dt
}

nuclei <- read_compartment("nuclei")
cyto <- read_compartment("cyto")
if (!setequal(attr(nuclei, "mean_marker_schema"), attr(cyto, "mean_marker_schema"))) {
  stop("Nuclear and whole-cell biological mean-marker schemas differ; both compartments must quantify the same model channels")
}
if (!setequal(nuclei$label_id, cyto$label_id)) {
  stop("Nuclear and whole-cell marker tables have different cell populations; no intersection/drop is permitted")
}
cyto_feature_cols <- c("label_id", paste0("cyto__", attr(nuclei, "mean_marker_schema")))
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
if (any(!is.finite(feature_matrix))) stop("Missing or nonfinite marker features: no silent imputation is permitted for marker KODAMA")
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
if (any(!is.finite(xy))) stop("Nonfinite marker coordinates")
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
writeLines(c("TRITC__mean", "Cy5__mean"), file.path(output_dir, "gigatime_excluded_background_features.txt"))
cat(sprintf("[OK] GigaTIME KODAMA input cells=%d marker_features=%d\n", nrow(feature_matrix), ncol(feature_matrix)))
