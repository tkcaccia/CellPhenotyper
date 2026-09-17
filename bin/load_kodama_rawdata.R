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

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_argument) != 1L) stop("Cannot locate the UNI2 binary reader helper")
source(file.path(dirname(normalizePath(sub("^--file=", "", script_argument))), "uni2_embedding_io.R"))
mode_dirs <- list(tile = tile_dir, nuclei = nuclei_dir, cyto = cyto_dir, inner_square = inner_square_dir)
available_modes <- c("tile", "nuclei", "cyto", "inner_square")
input_inventory <- lapply(mode_dirs[selected_modes], discover_uni2_embedding_input)
input_formats <- unique(vapply(input_inventory, `[[`, character(1), "storage"))
if (length(input_formats) != 1L) stop("Mixed legacy CSV and binary UNI2 sources across selected modes are not allowed")
binary_input <- identical(input_formats, "binary")
embedding_input_provenance <- NULL

load_embedding_matrix <- function(dir_path, mode_name, required = TRUE) {
  if (!required) {
    cat(sprintf("[INFO] mode=%s skipped (not selected)\n", mode_name))
    return(NULL)
  }

  if (!dir.exists(dir_path)) {
    stop(paste("Embedding directory does not exist for mode", mode_name, ":", dir_path))
  }
  if (binary_input) return(load_uni2_binary_mode(dir_path, mode_name, top_features_per_mode))
  load_uni2_csv_mode(dir_path, mode_name, top_features_per_mode)
}

if (!file.exists(annot_csv)) {
  stop(paste("Annotation CSV does not exist:", annot_csv))
}

annotation_sha256_before <- uni2_io_sha256(annot_csv)
ann <- as.data.frame(uni2_csv_read(annot_csv, colClasses = list(character = "label")))
uni2_csv_names(names(ann), "annotation column names")
if (!("label" %in% colnames(ann))) {
  stop("Annotation CSV must contain a 'label' column")
}
if (!all(c("x", "y") %in% colnames(ann))) {
  stop("Annotation CSV must contain 'x' and 'y' columns")
}

ann$label <- as.character(ann$label)
uni2_csv_names(ann$label, "annotation labels")
if (!is.numeric(ann$x) || !is.numeric(ann$y) || any(!is.finite(ann$x)) || any(!is.finite(ann$y)))
  stop("UNI2 annotations require numeric finite x/y coordinates")
ann$x <- as.numeric(ann$x)
ann$y <- as.numeric(ann$y)
rownames(ann) <- ann$label

if (!("polygon_label" %in% colnames(ann))) {
  ann$polygon_label <- "unknown"
}

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

{
  analysis_ids <- sort(ann$label)
  source_orders <- list()
  for (mode in selected_modes) {
    source_ids <- rownames(loaded[[mode]]$matrix)
    if (!setequal(source_ids, ann$label)) {
      missing_ids <- setdiff(ann$label, source_ids)
      foreign_ids <- setdiff(source_ids, ann$label)
      stop("UNI2 rows and annotation IDs must match exactly for mode ", mode,
        "; no intersections or foreign IDs are allowed. Missing (", length(missing_ids), "): ",
        paste(head(missing_ids, 10L), collapse = ","), "; foreign (", length(foreign_ids), "): ",
        paste(head(foreign_ids, 10L), collapse = ","))
    }
    metadata_rows <- loaded[[mode]]$rows
    # x/y, when supplied under the same names, are exact source coordinates.
    # Production cx/cy are rounded/clipped extraction centers and are not
    # silently treated as equivalent to floating-point annotation centroids.
    for (coordinate in intersect(c("x", "y"), names(metadata_rows))) {
      values <- suppressWarnings(as.numeric(metadata_rows[[coordinate]]))
      expected <- ann[source_ids, coordinate]
      if (any(!is.finite(values)) || !identical(values, as.numeric(expected)))
        stop("UNI2 row/annotation coordinate mismatch: ", mode, "/", coordinate)
    }
    source_orders[[mode]] <- list(source_observation_ids = source_ids,
      analysis_row_from_source = match(analysis_ids, source_ids))
    loaded[[mode]]$matrix <- loaded[[mode]]$matrix[analysis_ids, , drop = FALSE]
  }
  ann <- ann[analysis_ids, , drop = FALSE]
  if (!identical(uni2_io_sha256(annot_csv), annotation_sha256_before)) stop("Annotation source changed while loading UNI2")
  embedding_input_provenance <- list(format = if (binary_input) "cellphenotyper_uni2_binary_input" else "cellphenotyper_uni2_csv_input", schema_version = "1.0.0",
    storage_integrity = "validated_exact_sizes_sha256_finite_features_and_complete_ids",
    encoder_binding_status = "storage_integrity_does_not_establish_encoder_provenance",
    annotation_path = normalizePath(annot_csv), annotation_sha256 = annotation_sha256_before,
    analysis_observation_ids = analysis_ids, selected_modes = selected_modes,
    analysis_order = "lexical_ID_order_matching_legacy_CSV_policy_without_numeric_ID_coercion",
    source_to_analysis = source_orders, modes = lapply(loaded[selected_modes], `[[`, "provenance"))
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

# Compatibility name; selected modes have already passed exact population checks.
common_ids <- ann_ids

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

rawdata_path <- file.path(output_dir, "rawdata.RData")
save(
  ann_ids, common_ids, ann, xy, available_modes, selected_modes,
  embeddings_raw, mode_overlap_with_annotations,
  embedding_input_provenance,
  file = rawdata_path,
  compress = FALSE
)
jsonlite::write_json(embedding_input_provenance,
  file.path(output_dir, "embedding_input_provenance.json"), auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA)

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
