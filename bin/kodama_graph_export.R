# Portable, all-observation native KODAMA dissimilarity graphs. No package
# handles, visualization coordinates, affinities or inferred edges are saved.

kodama_graph_require_packages <- function() {
  for (package in c("Matrix", "digest", "jsonlite")) {
    if (!requireNamespace(package, quietly = TRUE)) {
      stop("Portable KODAMA graph requires R package: ", package)
    }
  }
}

kodama_graph_sha256 <- function(path) {
  kodama_graph_require_packages()
  if (!file.exists(path) || dir.exists(path)) stop("Graph-bound file is missing: ", path)
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

kodama_graph_ids <- function(ids) {
  if (!is.character(ids) || !length(ids) || anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids)) {
    stop("KODAMA graph observation IDs must be unique nonempty strings")
  }
  enc2utf8(ids)
}

kodama_graph_ids_sha256 <- function(ids) {
  kodama_graph_require_packages()
  bytes <- charToRaw(enc2utf8(as.character(jsonlite::toJSON(
    kodama_graph_ids(ids), auto_unbox = FALSE, pretty = FALSE))))
  digest::digest(bytes, algo = "sha256", serialize = FALSE)
}

kodama_graph_basename <- function(value, field) {
  if (!is.character(value) || length(value) != 1L || is.na(value) ||
      !nzchar(value) || value %in% c(".", "..") ||
      grepl("[/\\\\]", value) || basename(value) != value) {
    stop("Invalid portable graph ", field)
  }
  value
}

validate_portable_kodama_graph <- function(payload, expected_observation_ids) {
  kodama_graph_require_packages()
  expected <- kodama_graph_ids(expected_observation_ids)
  if (!is.list(payload) || !identical(payload$format, "cellphenotyper.kodama_graph") ||
      !identical(payload$schema_version, "1.0.0")) stop("Unsupported portable KODAMA graph schema")
  ids <- kodama_graph_ids(payload$observation_ids)
  if (!identical(ids, expected)) stop("KODAMA graph observation IDs/order must match exactly")
  meta <- payload$metadata
  if (!is.list(meta) || !identical(meta$format, payload$format) ||
      !identical(meta$schema_version, payload$schema_version) ||
      !isTRUE(meta$all_observations) || !identical(meta$projected, FALSE) ||
      !isTRUE(meta$native_corrected) || !isTRUE(meta$directed)) {
    stop("KODAMA graph must be native-corrected, directed and complete, never projected")
  }
  if (!identical(meta$observation_ids_sha256, kodama_graph_ids_sha256(ids)) ||
      length(meta$observation_count) != 1L || meta$observation_count != length(ids)) {
    stop("KODAMA graph observation identity receipt differs")
  }
  graph <- payload$distance_graph
  if (!methods::is(graph, "dgCMatrix") || !identical(dim(graph), rep(length(ids), 2L))) {
    stop("KODAMA distance_graph must be an all-observation dgCMatrix")
  }
  methods::validObject(graph)
  if (!identical(dimnames(graph), list(ids, ids)) || any(!is.finite(graph@x)) || any(graph@x < 0)) {
    stop("KODAMA distance_graph has invalid IDs or finite nonnegative distances")
  }
  if (length(meta$stored_edges) != 1L || meta$stored_edges != length(graph@x)) {
    stop("KODAMA graph stored-edge count differs")
  }
  # Inspect stored entries: sparse diagonal extraction would not distinguish a
  # stored zero self-edge from an absent self-edge.
  columns <- rep.int(seq_len(ncol(graph)), diff(graph@p))
  if (any(graph@i + 1L == columns)) stop("KODAMA graph must omit self edges")
  invisible(payload)
}

export_portable_kodama_graph <- function(fit, observation_ids, outdir, pca_file,
    expected_observation_ids = observation_ids, projected = FALSE,
    package_version = as.character(utils::packageVersion("KODAMA")),
    package_revision = "unknown", materialize = NULL) {
  kodama_graph_require_packages()
  ids <- kodama_graph_ids(observation_ids)
  if (!identical(ids, kodama_graph_ids(expected_observation_ids)) || !identical(projected, FALSE)) {
    stop("Cannot export a projected or subset KODAMA graph as all-observation")
  }
  if (!isTRUE(fit$knn_is_kodama_corrected)) stop("Native KODAMA graph is not confirmed corrected")
  if (!is.matrix(fit$res) || ncol(fit$res) != length(ids)) stop("Native KODAMA result observation count differs")
  if (is.null(materialize)) {
    if (!"KODAMA.graph.materialize" %in% getNamespaceExports("KODAMA")) {
      stop("Installed KODAMA lacks verified native graph materialization API")
    }
    materialize <- getExportedValue("KODAMA", "KODAMA.graph.materialize")
  }
  graph <- materialize(fit)
  indices <- graph$indices
  distances <- graph$distances
  n <- length(ids)
  if (!is.matrix(indices) || !is.numeric(indices) || !is.matrix(distances) ||
      !is.numeric(distances) || !identical(dim(indices), dim(distances)) ||
      nrow(indices) != n || ncol(indices) < 1L) stop("Native graph indices/distances geometry differs")
  if (anyNA(indices) || any(!is.finite(indices)) || any(indices != floor(indices)) ||
      any(indices < 1L | indices > n)) stop("Native graph indices must be valid one-based observation indices")
  if (anyNA(distances) || any(distances < 0)) stop("Native graph distances must be nonnegative, finite or positive infinity")
  from <- rep.int(seq_len(n), ncol(indices))
  to <- as.integer(indices)
  values <- as.vector(distances)
  infinite <- is.infinite(values)
  self <- from == to
  keep <- !infinite & !self
  # Explicit finite zero distances are kept as stored sparse entries. Never
  # drop0(): a zero dissimilarity is not the absence of an edge.
  sparse_graph <- Matrix::sparseMatrix(i = from[keep], j = to[keep], x = values[keep],
    dims = c(n, n), dimnames = list(ids, ids), giveCsparse = TRUE)
  if (length(sparse_graph@x) != sum(keep)) stop("Native graph contains duplicate directed edges")
  parameters <- fit$parameters
  if (is.null(parameters)) parameters <- list()
  classifier <- parameters$classifier
  applicability <- if (identical(classifier, "knn")) {
    "not_PLS_rank_for_raw_data_knn_classifier; retained parameter may affect accelerator worker-memory estimates"
  } else if (identical(classifier, "pls_lda")) {
    "PLS_LDA_component_parameter; actual fitted rank may be numerically constrained"
  } else "unknown_classifier_applicability"
  meta <- list(schema_version = "1.0.0", format = "cellphenotyper.kodama_graph",
    pca_file = basename(pca_file), pca_file_sha256 = kodama_graph_sha256(pca_file),
    pca_file_md5 = unname(tools::md5sum(pca_file)),
    observation_ids_sha256 = kodama_graph_ids_sha256(ids), observation_count = n,
    native_neighbors = ncol(indices), directed = TRUE, all_observations = TRUE,
    projected = FALSE, native_corrected = TRUE, stored_edges = length(sparse_graph@x),
    omitted_infinite_edges = sum(infinite), omitted_self_edges = sum(self & !infinite),
    stored_zero_distance_edges = sum(sparse_graph@x == 0),
    distance_semantics = "native corrected dissimilarity (1 + base distance) / valid-run coassignment_fraction^2; positive infinity omitted, finite zero retained",
    affinity_conversion = "none; consumer must declare weighting and symmetrization",
    native_package_version = package_version, native_package_revision = package_revision,
    native_parameters = parameters, ncomp_applicability = applicability,
    native_index_policy = list(type = fit$graph_index_type, ivf_nlist = fit$graph_ivf_nlist,
      ivf_nprobe = fit$graph_ivf_nprobe, ivf_pilot_recall = fit$graph_ivf_pilot_recall),
    observation_scope = "all input rows; native landmark optimization and within-run label projection retained",
    id_hash_encoding = "SHA256 of compact jsonlite UTF-8 JSON string array, auto_unbox=FALSE")
  payload <- list(format = meta$format, schema_version = meta$schema_version,
    observation_ids = ids, distance_graph = sparse_graph, metadata = meta)
  validate_portable_kodama_graph(payload, ids)
  paths <- file.path(outdir, c("kodama_graph.rds", "kodama_graph.json"))
  if (any(file.exists(paths))) stop("Refusing to overwrite an existing portable KODAMA graph")
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
  saveRDS(payload, paths[1L], compress = FALSE, version = 3)
  # Exact structural readback, including explicit stored zeros and all isolates.
  restored <- readRDS(paths[1L])
  if (!identical(restored, payload)) stop("Portable KODAMA graph RDS readback differs")
  manifest <- c(meta, list(file = "kodama_graph.rds", file_sha256 = kodama_graph_sha256(paths[1L])))
  jsonlite::write_json(manifest, paths[2L], auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null", digits = NA)
  manifest
}

load_portable_kodama_graph <- function(directory, expected_observation_ids,
    expected_graph_sha256 = NULL, pca_file = NULL) {
  kodama_graph_require_packages()
  manifest_path <- file.path(directory, "kodama_graph.json")
  if (!file.exists(manifest_path)) stop("Portable native KODAMA graph manifest is missing")
  manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  filename <- kodama_graph_basename(manifest$file, "filename")
  path <- file.path(directory, filename)
  actual <- kodama_graph_sha256(path)
  if (!identical(actual, manifest$file_sha256) ||
      (!is.null(expected_graph_sha256) && !identical(actual, expected_graph_sha256))) {
    stop("Portable KODAMA graph SHA256 checksum differs")
  }
  payload <- readRDS(path)
  validate_portable_kodama_graph(payload, expected_observation_ids)
  # JSON receipt and RDS metadata must agree; normalize through JSON to avoid
  # R integer/double and list/vector representation differences on decoding.
  expected_meta <- jsonlite::fromJSON(jsonlite::toJSON(payload$metadata,
    auto_unbox = TRUE, null = "null", na = "null", digits = NA), simplifyVector = FALSE)
  supplied_meta <- manifest[setdiff(names(manifest), c("file", "file_sha256"))]
  if (!identical(expected_meta, supplied_meta)) stop("Portable KODAMA graph metadata receipt differs")
  source_name <- kodama_graph_basename(manifest$pca_file, "PCA filename")
  if (is.null(pca_file)) pca_file <- file.path(directory, source_name)
  if (!identical(basename(pca_file), source_name) ||
      !identical(kodama_graph_sha256(pca_file), manifest$pca_file_sha256) ||
      !identical(unname(tools::md5sum(pca_file)), manifest$pca_file_md5)) {
    stop("KODAMA graph source PCA checksum differs")
  }
  payload$manifest <- manifest
  payload
}
