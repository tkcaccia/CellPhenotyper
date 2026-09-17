#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
`%||%` <- function(x, y) if (is.null(x) || length(x) == 0L) y else x
if (length(args) < 2) {
  stop(
    paste(
      "Usage: Rscript run_kodama_analysis.R",
      "<rawdata_rdata> <output_dir>",
      "[--embedding-mode tile|nuclei|cyto|inner_square|all|full|tile,inner_square,...]",
      "[--dims-to-run N] [--spark-top N] [--landmarks N] [--kodama-ncomp N] [--n-cores N]",
      "[--backend cpu|cuda|metal] [--gpu-device N] [--save-selected-features true|false]",
      "[--export-native-graph true|false]"
    )
  )
}

rawdata_path <- args[1]
output_dir <- args[2]

embedding_mode <- "all"
dims_to_run <- 20L
spark_top <- 100L
landmarks <- 10000L
kodama_ncomp <- 50L
n_cores <- 4L
backend <- "cpu"
gpu_device <- 0L
save_selected_features <- FALSE
export_native_graph <- FALSE

if (length(args) > 2) {
  i <- 3L
  while (i <= length(args)) {
    flag <- args[i]
    if (flag == "--export-native-graph") {
      if (i + 1L > length(args)) stop("--export-native-graph requires true or false")
      value <- tolower(args[i + 1L])
      if (!value %in% c("true", "false")) stop("--export-native-graph must be true or false")
      export_native_graph <- identical(value, "true")
      i <- i + 2L
      next
    }
    if (flag == "--save-selected-features" && i + 1L <= length(args)) {
      value <- tolower(args[i + 1L])
      if (!value %in% c("true", "false")) stop("--save-selected-features must be true or false")
      save_selected_features <- identical(value, "true")
      i <- i + 2L
      next
    }
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
    if (flag == "--backend" && i + 1L <= length(args)) {
      backend <- tolower(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--gpu-device" && i + 1L <= length(args)) {
      gpu_device <- as.integer(args[i + 1L])
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
  landmarks <- 10000L
}
if (!is.finite(kodama_ncomp) || kodama_ncomp < 1L) {
  kodama_ncomp <- 50L
}
if (!is.finite(n_cores) || n_cores < 1L) {
  n_cores <- 1L
}
if (!backend %in% c("cpu", "cuda", "metal")) {
  stop("--backend must be one of: cpu, cuda, metal")
}
if (!is.finite(gpu_device) || gpu_device < 0L) {
  gpu_device <- 0L
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
library(data.table)

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_argument) != 1L) stop("Cannot locate the portable KODAMA graph helper")
script_directory <- dirname(normalizePath(sub("^--file=", "", script_argument)))
source(file.path(script_directory, "kodama_graph_export.R"))
if (export_native_graph) {
  kodama_graph_require_packages()
  if (!"KODAMA.graph.materialize" %in% getNamespaceExports("KODAMA")) {
    stop("Installed KODAMA lacks the verified native graph materialization API")
  }
}

kodama_description <- utils::packageDescription("KODAMA")
kodama_revision <- kodama_description$RemoteSha %||% kodama_description$GithubSHA1 %||% "unknown"
cat(sprintf(
  "[INFO] KODAMA version=%s revision=%s backend=%s gpu_device=%d n_cores=%d\n",
  as.character(utils::packageVersion("KODAMA")), kodama_revision, backend,
  gpu_device, n_cores
))
print(KODAMA.diagnostics())

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
if (!exists("embedding_input_provenance", inherits = FALSE)) embedding_input_provenance <- NULL
verified_uni2_input <- !is.null(embedding_input_provenance)
rawdata_input_sha256 <- NULL
if (verified_uni2_input) {
  if (!is.list(embedding_input_provenance) ||
      !(embedding_input_provenance$format %in% c("cellphenotyper_uni2_binary_input", "cellphenotyper_uni2_csv_input")) ||
      !identical(embedding_input_provenance$schema_version, "1.0.0"))
    stop("Unsupported UNI2 input provenance in rawdata")
  verified_analysis_ids <- embedding_input_provenance$analysis_observation_ids
  if (!is.character(verified_analysis_ids) || !length(verified_analysis_ids) || anyNA(verified_analysis_ids) ||
      any(!nzchar(verified_analysis_ids)) || anyDuplicated(verified_analysis_ids))
    stop("UNI2 input provenance needs unique explicit analysis IDs")
  if (!requireNamespace("digest", quietly = TRUE)) stop("UNI2 provenance requires R package digest")
  rawdata_input_sha256 <- digest::digest(file = rawdata_path, algo = "sha256", serialize = FALSE)
}

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
if (!length(ann$label) || anyNA(ann$label) || any(!nzchar(ann$label)) ||
    any(trimws(ann$label) != ann$label) || any(grepl("[[:cntrl:]]", ann$label)) || anyDuplicated(ann$label))
  stop("Invalid or duplicate annotation IDs; no rows are silently discarded")
if (verified_uni2_input && !identical(ann$label, verified_analysis_ids))
  stop("UNI2 annotation IDs/order differ from the exact rawdata provenance; no dropping/reordering is permitted")
rownames(ann) <- ann$label

if (!all(c("x", "y") %in% colnames(ann))) {
  stop("Annotation table in rawdata must contain columns 'x' and 'y'")
}
if (!is.numeric(ann$x) || !is.numeric(ann$y) || any(!is.finite(ann$x)) || any(!is.finite(ann$y)))
  stop("Annotations require numeric finite coordinates")
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

  selection <- tryCatch(
    KODAMA.spatial.features(
      mat,
      xy_coords,
      n.cores = cores,
      require.nonzero.each.sample = FALSE
    ),
    error = function(e) {
      warning("KODAMA spatial feature selection failed; using variance ranking: ", conditionMessage(e))
      NULL
    }
  )
  idx <- if (is.null(selection)) NULL else selection$ranking

  if (!is.null(idx)) {
    idx <- suppressWarnings(as.integer(idx))
    idx <- idx[is.finite(idx)]
    idx <- idx[idx >= 1L & idx <= ncol(mat)]
    idx <- unique(idx)
    if (length(idx) > 0L) {
      keep <- idx[seq_len(min(top_n, length(idx)))]
      cat(sprintf(
        "[INFO] KODAMA spatial feature selection retained=%d runtime_seconds=%.3f\n",
        length(keep), as.numeric(selection$runtime.seconds %||% NA_real_)
      ))
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
  if (!is.numeric(mat)) stop("Embedding matrix for mode ", mode_name, " must be numeric; no silent imputation")
  storage.mode(mat) <- "double"
  ids <- rownames(mat)
  if (is.null(ids)) {
    stop(paste("Embedding matrix for mode", mode_name, "is missing row names (cell IDs)."))
  }
  ids <- as.character(ids)
  if (!length(ids) || anyNA(ids) || any(!nzchar(ids)) || any(trimws(ids) != ids) ||
      any(grepl("[[:cntrl:]]", ids)) || anyDuplicated(ids))
    stop("Invalid or duplicate embedding cell IDs for mode ", mode_name, "; no rows are silently discarded")
  features <- colnames(mat)
  if (is.null(features) || !length(features) || anyNA(features) || any(!nzchar(features)) || anyDuplicated(features))
    stop("Embedding feature names must be explicit and unique for mode ", mode_name)
  if (any(!is.finite(mat))) stop("Nonfinite embedding features for mode ", mode_name, "; no silent imputation")
  if (verified_uni2_input) {
    if (!identical(ids, verified_analysis_ids))
      stop("UNI2 matrix IDs/order differ from rawdata provenance for mode ", mode_name)
    expected_features <- embedding_input_provenance$modes[[mode_name]]$selected_feature_names
    if (!is.character(expected_features) || !identical(colnames(mat), expected_features))
      stop("UNI2 matrix feature names/order differ from rawdata provenance for mode ", mode_name)
  }
  rownames(mat) <- ids
  mat
}

for (m in names(mode_mats)) {
  mode_mats[[m]] <- sanitize_mode_matrix(mode_mats[[m]], m)
}

available_modes <- names(Filter(Negate(is.null), mode_mats))
selected_modes <- normalize_modes(embedding_mode)
if (verified_uni2_input && !all(selected_modes %in% embedding_input_provenance$selected_modes))
  stop("Requested modes are not covered by UNI2 source provenance")
missing_modes <- setdiff(selected_modes, available_modes)
if (length(missing_modes) > 0L) {
  stop(paste("Requested embedding mode(s) missing from rawdata:", paste(missing_modes, collapse = ",")))
}
cat(sprintf("[INFO] KODAMA embedding_mode=%s\n", paste(selected_modes, collapse = ",")))

for (mode in selected_modes) {
  source_ids <- rownames(mode_mats[[mode]])
  if (!setequal(source_ids, ann_full$label))
    stop("Embedding and annotation IDs must match exactly for mode ", mode,
      "; no intersection/drop is permitted. Missing=", length(setdiff(ann_full$label, source_ids)),
      "; foreign=", length(setdiff(source_ids, ann_full$label)))
}
common_ids <- if (verified_uni2_input) verified_analysis_ids else sort(ann_full$label)
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

pca_res <- KODAMA.pca(
  data,
  ncomp = max_nv,
  center = TRUE,
  scale = TRUE,
  backend = backend,
  n.cores = as.integer(n_cores),
  gpu.device = as.integer(gpu_device),
  seed = 543210L
)
pca <- pca_res$scores
rownames(pca) <- common_ids
pca_feature_names <- colnames(data)
if (is.null(pca_feature_names)) pca_feature_names <- sprintf("selected_feature_%d", seq_len(ncol(data)))
if (is.null(colnames(pca))) colnames(pca) <- sprintf("PC%d", seq_len(ncol(pca)))
pca_model_metadata <- list(
  schema_version = "1.0.0",
  input_feature_names = pca_feature_names,
  input_feature_count = ncol(data),
  selected_embedding_modes = selected_modes,
  requested_components = requested_pca_components,
  computed_components = ncol(pca),
  center = TRUE, scale = TRUE,
  backend = pca_res$backend %||% backend,
  seed = 543210L,
  kodama_package_version = as.character(utils::packageVersion("KODAMA")),
  kodama_package_revision = kodama_revision
)
if (verified_uni2_input) {
  pca_model_metadata$rawdata_input_sha256 <- rawdata_input_sha256
  pca_model_metadata$embedding_input_provenance <- embedding_input_provenance
}
selected_features_path <- if (save_selected_features) file.path(output_dir, "selected_features.rds") else NA_character_
if (save_selected_features) {
  saveRDS(list(
    schema_version = "1.0.0", features = data, observation_ids = common_ids,
    feature_names = pca_feature_names, selected_embedding_modes = selected_modes,
    preprocessing = "raw selected embedding columns, before centered/scaled PCA"
  ), selected_features_path, compress = FALSE)
}
cat(sprintf(
  "[INFO] PCA components computed=%d requested=%d backend=%s runtime_seconds=%.3f\n",
  ncol(pca), requested_pca_components, pca_res$backend %||% backend,
  as.numeric(pca_res$runtime_seconds %||% NA_real_)
))

lab <- as.factor(ann[, "polygon_label"])

rm(
  list = intersect(
    c(
      "data", "data_parts", "mode_mats", "mats_selected",
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
save(pca, xy, common_ids, ann, selected_modes, pca_feature_names, pca_model_metadata, embedding_input_provenance, file = pca_rdata)

if (nrow(pca) <= plot_max_cells) {
  u <- fastEmbedR::umap(
    pca,
    backend = "cpu",
    n.cores = as.integer(n_cores),
    seed = 543210L
  )$layout
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
requested_kodama_ncomp <- as.integer(kodama_ncomp)
effective_kodama_ncomp <- min(requested_kodama_ncomp, dims_use)
if (effective_kodama_ncomp != requested_kodama_ncomp) {
  warning(sprintf(
    "KODAMA requested internal ncomp=%d exceeds its %d input PCA features; effective internal ncomp=%d. These are separate parameters; increase --dims-to-run explicitly if the requested internal rank is required.",
    requested_kodama_ncomp, dims_use, effective_kodama_ncomp
  ))
}
cat(sprintf("[INFO] KODAMA landmarks=%d\n", kodama_landmarks))
cat(sprintf("[INFO] KODAMA internal ncomp requested=%d effective=%d; input PCA dimensions=%d\n", requested_kodama_ncomp, effective_kodama_ncomp, dims_use))

visual_neighbors <- 30L
if (export_native_graph && nrow(pca) > kodama_exact_max_cells) {
  stop("Requested native graph export requires all observations in KODAMA; the outer subset/projection branch cannot supply it")
}

if (nrow(pca) <= kodama_exact_max_cells) {
  visual_neighbors <- min(30L, nrow(pca) - 1L)
  jj <- KODAMA.matrix(
    pca[, seq_len(dims_use), drop = FALSE],
    spatial = spatial_for_kodama,
    landmarks = kodama_landmarks,
    n.cores = as.integer(n_cores),
    seed = 543210,
    ncomp = effective_kodama_ncomp,
    backend = backend,
    visual.init = TRUE,
    return.graph = "handle"
  )
  vis <- KODAMA.visualization(
    jj,
    method = "UMAP",
    k = visual_neighbors,
    backend = backend,
    n.cores = as.integer(n_cores),
    gpu.device = as.integer(gpu_device),
    seed = 543210L
  )
  rownames(vis) <- common_ids
} else {
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
  visual_neighbors <- min(30L, nrow(pca_kodama) - 1L)
  jj <- KODAMA.matrix(
    pca_kodama,
    spatial = spatial_kodama,
    landmarks = min(kodama_landmarks, nrow(pca_kodama)),
    n.cores = as.integer(n_cores),
    seed = 543210,
    ncomp = effective_kodama_ncomp,
    backend = backend,
    visual.init = TRUE,
    return.graph = "handle"
  )
  vis_kodama <- KODAMA.visualization(
    jj,
    method = "UMAP",
    k = visual_neighbors,
    backend = backend,
    n.cores = as.integer(n_cores),
    gpu.device = as.integer(gpu_device),
    seed = 543210L
  )
  rownames(vis_kodama) <- projection_ids

  nn_k <- min(kodama_projection_neighbors, nrow(pca_kodama))
  nn <- fastEmbedR::precompute_query_knn(
    reference = pca_kodama,
    query = pca[, seq_len(dims_use), drop = FALSE],
    k = nn_k,
    backend = "cpu",
    n.cores = as.integer(n_cores)
  )
  weights <- 1 / pmax(nn$distances, 1e-6)
  weights <- weights / rowSums(weights)

  vis <- matrix(0, nrow = nrow(pca), ncol = ncol(vis_kodama))
  for (j in seq_len(nn_k)) {
    vis <- vis + vis_kodama[nn$indices[, j], , drop = FALSE] * weights[, j]
  }
  rownames(vis) <- common_ids
  colnames(vis) <- colnames(vis_kodama)
  rm(pca_kodama, spatial_kodama, vis_kodama, nn, weights)
  gc(FALSE)
}

graph_manifest <- NULL
if (export_native_graph && nrow(pca) <= kodama_exact_max_cells) {
  graph_manifest <- export_portable_kodama_graph(
    jj, observation_ids = common_ids, outdir = output_dir,
    pca_file = pca_rdata, expected_observation_ids = rownames(pca),
    projected = FALSE,
    package_version = as.character(utils::packageVersion("KODAMA")),
    package_revision = kodama_revision
  )
  cat(sprintf("[INFO] Exported all-observation native KODAMA graph: observations=%d directed_edges=%d\n",
    graph_manifest$observation_count, graph_manifest$stored_edges))
} else if (export_native_graph) {
  # Internal landmark optimization still gives an all-input-row graph. This
  # outer sampling branch is different: jj only contains projection_idx rows.
  # Do not publish it as a graph for all common_ids or infer edges from UMAP.
  cat("[INFO] Portable all-observation graph unavailable: outer KODAMA subset/projection branch\n")
}
actual_kodama_classifier <- jj$parameters$classifier %||% "unknown"
kodama_ncomp_applicability <- if (identical(actual_kodama_classifier, "knn")) {
  "not_PLS_rank_for_raw_data_knn_classifier; retained parameter may affect accelerator worker-memory estimates"
} else if (identical(actual_kodama_classifier, "pls_lda")) {
  "PLS_LDA_component_parameter; actual fitted rank may be numerically constrained"
} else "unknown_classifier_applicability"
cat(sprintf("[INFO] Actual KODAMA classifier=%s; ncomp applicability=%s\n",
  actual_kodama_classifier, kodama_ncomp_applicability))

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
representation_metadata <- list(
  schema_version = "1.0.0",
  available_cluster_representations = c("umap2d", "pca", if (!is.null(graph_manifest)) "kodama_graph"),
  pca_file = basename(pca_rdata),
  pca_file_md5 = unname(tools::md5sum(pca_rdata)),
  pca_dimensions = ncol(pca),
  pca_observations = nrow(pca),
  observation_order = "common_ids and rownames(pca), explicitly joined by ID downstream",
  pca_preprocessing = pca_model_metadata,
  rawdata_input_sha256 = rawdata_input_sha256,
  embedding_input_provenance = embedding_input_provenance,
  selected_features_file = if (save_selected_features) basename(selected_features_path) else NULL,
  selected_features_saved = save_selected_features,
  visualization = "KODAMA UMAP; display coordinates, separate from saved PCA feature space",
  visualization_dimensions = ncol(vis),
  visualization_projected = nrow(pca) > kodama_exact_max_cells,
  visualization_exact_observations = if (nrow(pca) > kodama_exact_max_cells) length(projection_idx) else nrow(pca),
  pca_scores_projected = FALSE,
  requested_kodama_ncomp = requested_kodama_ncomp,
  effective_kodama_ncomp = effective_kodama_ncomp,
  actual_kodama_classifier = actual_kodama_classifier,
  kodama_ncomp_applicability = kodama_ncomp_applicability,
  kodama_input_pca_dimensions = dims_use,
  kodama_graph_available = !is.null(graph_manifest),
  kodama_graph_export_requested = export_native_graph,
  kodama_graph_file = if (!is.null(graph_manifest)) graph_manifest$file else NULL,
  kodama_graph_sha256 = if (!is.null(graph_manifest)) graph_manifest$file_sha256 else NULL,
  kodama_graph_manifest_file = if (!is.null(graph_manifest)) "kodama_graph.json" else NULL,
  kodama_graph_reason = if (!is.null(graph_manifest)) "native corrected directed distances; all input observations; no affinity conversion" else if (!export_native_graph)
    "Native graph export was not requested; materialization and storage remain opt-in" else
    "Outer KODAMA optimization used a subset; projected UMAP cannot supply an all-observation native graph"
)
save(vis, xy, common_ids, ann, lab, selected_modes, representation_metadata, file = kodama_rdata)
if (requireNamespace("jsonlite", quietly = TRUE)) {
  jsonlite::write_json(representation_metadata, file.path(output_dir, "clustering_representations.json"), auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null")
} else {
  saveRDS(representation_metadata, file.path(output_dir, "clustering_representations.rds"))
}

timing <- KODAMA.timing(jj)
timing$backend <- backend
timing$package_version <- as.character(utils::packageVersion("KODAMA"))
timing$package_revision <- kodama_revision
data.table::fwrite(timing, file.path(output_dir, "kodama_native_timing.csv"))

cat(sprintf("[INFO] Shared cells=%d\n", length(common_ids)))
cat(sprintf("[INFO] Selected modes=%s\n", paste(selected_modes, collapse = ",")))
cat(sprintf("[INFO] Read: %s\n", rawdata_path))
cat(sprintf("[INFO] Wrote: %s\n", kodama_rdata))
