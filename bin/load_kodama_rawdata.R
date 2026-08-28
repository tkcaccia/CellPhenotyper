#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop(
    paste(
      "Usage: Rscript load_kodama_rawdata.R",
      "<tile_embeddings_dir> <cyto_embeddings_dir> <inner_square_embeddings_dir> <nuclei_embeddings_dir>",
      "<objects_assigned_csv> <output_dir> [embedding_modes] [top_features_per_mode]"
    )
  )
}

tile_dir <- args[1]
cyto_dir <- args[2]
inner_square_dir <- args[3]
nuclei_dir <- args[4]
annot_csv <- args[5]
output_dir <- args[6]
embedding_mode <- if (length(args) >= 7) args[7] else "all"
top_features_per_mode <- if (length(args) >= 8) suppressWarnings(as.integer(args[8])) else 100L
if (!is.finite(top_features_per_mode) || top_features_per_mode < 1L) {
  top_features_per_mode <- 100L
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

selected_modes <- normalize_modes(embedding_mode)

library(data.table)

fread_embedding <- function(file_path, ...) {
  if (grepl("\\.gz$", file_path, ignore.case = TRUE)) {
    # data.table otherwise requires the optional R.utils package for gzip files.
    gzip_cmd <- sprintf("gzip -dc -- %s", shQuote(normalizePath(file_path, mustWork = TRUE)))
    return(fread(cmd = gzip_cmd, ...))
  }
  fread(file_path, ...)
}

read_embedding_header <- function(file_path) {
  fread_embedding(file_path, nrows = 0L, showProgress = FALSE)
}

embedding_feature_columns <- function(files, mode_name) {
  header <- read_embedding_header(files[[1L]])
  feat_cols <- grep("^feat", colnames(header), value = TRUE)
  if (length(feat_cols) == 0L) {
    stop(paste("Embedding table for mode", mode_name, "has no feature columns starting with 'feat'"))
  }
  feat_cols
}

select_top_variance_features <- function(files, feat_cols, top_n, mode_name) {
  if (length(feat_cols) <= top_n) {
    return(feat_cols)
  }

  sums <- numeric(length(feat_cols))
  sums_sq <- numeric(length(feat_cols))
  counts <- numeric(length(feat_cols))
  names(sums) <- feat_cols
  names(sums_sq) <- feat_cols
  names(counts) <- feat_cols

  for (fp in files) {
    dt <- fread_embedding(fp, select = feat_cols, showProgress = FALSE)
    mat <- as.matrix(dt)
    storage.mode(mat) <- "double"
    ok <- !is.na(mat)
    vals <- mat
    vals[!ok] <- 0
    sums <- sums + colSums(vals)
    sums_sq <- sums_sq + colSums(vals * vals)
    counts <- counts + colSums(ok)
    rm(dt, mat, ok, vals)
    gc(FALSE)
  }

  means <- sums / pmax(counts, 1)
  variances <- (sums_sq / pmax(counts, 1)) - (means * means)
  variances[!is.finite(variances)] <- -Inf
  ordered <- names(sort(variances, decreasing = TRUE))
  keep <- ordered[seq_len(min(top_n, length(ordered)))]
  cat(sprintf("[INFO] mode=%s preselected_top_variance_features=%d of %d\n", mode_name, length(keep), length(feat_cols)))
  keep
}

load_embedding_matrix <- function(dir_path, mode_name, required = TRUE) {
  if (!required) {
    cat(sprintf("[INFO] mode=%s skipped (not selected)\n", mode_name))
    return(NULL)
  }

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
  files <- sort(files)

  feat_cols_all <- embedding_feature_columns(files, mode_name)
  feat_cols <- select_top_variance_features(
    files,
    feat_cols_all,
    top_features_per_mode,
    mode_name
  )
  read_cols <- c("cell_id", feat_cols)
  dt <- rbindlist(
    lapply(files, function(fp) fread_embedding(fp, select = read_cols, showProgress = FALSE)),
    fill = TRUE,
    use.names = TRUE
  )
  if (!("cell_id" %in% colnames(dt))) {
    stop(paste("Embedding table for mode", mode_name, "is missing 'cell_id' column"))
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
cat(sprintf("[INFO] Selected embedding families: %s\n", paste(selected_modes, collapse = ",")))
cat(sprintf("[INFO] Top features per selected family: %d\n", top_features_per_mode))

loaded <- list()
for (m in available_modes) {
  required_mode <- m %in% selected_modes
  loaded[[m]] <- load_embedding_matrix(mode_dirs[[m]], m, required = required_mode)
  if (!is.null(loaded[[m]])) {
    cat(sprintf("[INFO] mode=%s cells=%d features=%d\n", m, loaded[[m]]$n_cells, loaded[[m]]$n_features))
  }
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
  c(list(ann_ids), lapply(selected_modes, function(m) rownames(embeddings_raw[[m]])))
)
common_ids <- sort(unique(common_ids))

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

rawdata_path <- file.path(output_dir, "rawdata.RData")
save(
  ann_ids, common_ids, ann, xy, available_modes, selected_modes,
  embeddings_raw, mode_overlap_with_annotations,
  file = rawdata_path,
  compress = FALSE
)

# Keep a backward-compatible filename without serializing a second multi-GB copy.
legacy_path <- file.path(output_dir, "raw_data.RData")
if (file.exists(legacy_path)) {
  unlink(legacy_path)
}
if (!file.symlink(basename(rawdata_path), legacy_path)) {
  writeLines(
    "raw_data.RData is intentionally not duplicated; use rawdata.RData.",
    file.path(output_dir, "raw_data.RData.note.txt")
  )
}

cat(sprintf("[INFO] Annotation cells=%d\n", length(ann_ids)))
cat(sprintf("[INFO] Global shared cells (all modes)=%d\n", length(common_ids)))
cat(sprintf("[INFO] Saved embedding families=%s\n", paste(available_modes, collapse = ",")))
cat(sprintf("[INFO] Wrote: %s\n", rawdata_path))
