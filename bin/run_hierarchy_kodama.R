#!/usr/bin/env Rscript
# Feature-only native KODAMA inside an already fixed hierarchy parent. No
# geometry, annotation, target K, or nearest-observation assignment enters fit.

hierarchy_sha256 <- function(path) {
  if (!file.exists(path) || dir.exists(path)) stop("Missing input file: ", path)
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

hierarchy_hash_map <- function(paths) {
  paths <- sort(unique(normalizePath(paths, mustWork = TRUE)), method = "radix")
  setNames(lapply(paths, hierarchy_sha256), paths)
}

hierarchy_verify <- function(expected, kind = "Input") {
  if (!identical(hierarchy_hash_map(names(expected)), expected))
    stop(kind, " changed during consumption")
  invisible(TRUE)
}

hierarchy_json_keys <- function(value) {
  if (is.list(value)) {
    if (!is.null(names(value)) && (anyDuplicated(names(value)) || any(!nzchar(names(value)))))
      stop("JSON objects require unique nonempty keys")
    for (child in value) hierarchy_json_keys(child)
  }
  invisible(value)
}

hierarchy_integer <- function(value, field, minimum = 1L) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
      value != floor(value) || value < minimum || value > .Machine$integer.max)
    stop("Invalid integer parameter: ", field)
  as.integer(value)
}

hierarchy_numeric_vector <- function(value, field, integer = FALSE, positive = FALSE) {
  if (!is.list(value) || !length(value) || any(!vapply(value, function(x)
      is.numeric(x) && length(x) == 1L && is.finite(x), logical(1))))
    stop("Invalid numeric array: ", field)
  result <- unlist(value, use.names = FALSE)
  if (anyDuplicated(result) || (positive && any(result <= 0)) ||
      (integer && any(result != floor(result) | result < 0 | result > .Machine$integer.max)))
    stop("Invalid numeric array: ", field)
  if (integer) as.integer(result) else result
}

hierarchy_input_file <- function(record, root, field) {
  path <- record$path
  if (!is.character(path) || length(path) != 1L || is.na(path) || !nzchar(path) ||
      path %in% c(".", "..") || basename(path) != path || grepl("[/\\\\]", path) ||
      grepl("[[:cntrl:]]", path)) stop("Input path must be a contained basename: ", field)
  if (!is.character(record$sha256) || length(record$sha256) != 1L ||
      !grepl("^[0-9a-f]{64}$", record$sha256)) stop("Invalid input SHA256: ", field)
  result <- normalizePath(file.path(root, path), mustWork = TRUE)
  if (!startsWith(result, paste0(root, .Platform$file.sep)) || dir.exists(result))
    stop("Input path escapes manifest directory: ", field)
  if (!identical(hierarchy_sha256(result), record$sha256)) stop("Input SHA256 differs: ", field)
  result
}

hierarchy_read_rows <- function(path) {
  rows <- read.csv(path, colClasses = "character", check.names = FALSE,
    na.strings = character(), row.names = NULL, fill = FALSE, blank.lines.skip = FALSE,
    comment.char = "", strip.white = FALSE, stringsAsFactors = FALSE)
  if (!identical(names(rows), c("label", "x", "y")) || !nrow(rows))
    stop("Rows must contain exactly label,x,y and at least one observation")
  ids <- rows$label
  if (anyNA(ids) || any(!nzchar(ids)) || any(trimws(ids) != ids) ||
      any(grepl("[[:cntrl:]]", ids)) || anyDuplicated(ids))
    stop("Observation IDs must be literal unique nonempty strings without padding/control characters")
  for (field in c("x", "y")) {
    value <- suppressWarnings(as.numeric(rows[[field]]))
    if (any(!nzchar(rows[[field]])) || any(!is.finite(value)))
      stop("Rows coordinates must be finite numeric values")
    rows[[field]] <- value
  }
  rows
}

hierarchy_read_matrix <- function(path, record, ids) {
  shape <- record$shape
  if (!is.list(shape) || length(shape) != 2L) stop("Representation shape must be [N,D]")
  n <- hierarchy_integer(shape[[1]], "shape N")
  d <- hierarchy_integer(shape[[2]], "shape D", minimum = 0L)
  if (n != length(ids)) stop("Representation row count differs from rows.csv")
  if (!is.logical(record$constant) || length(record$constant) != 1L || is.na(record$constant))
    stop("Representation constant must be a Boolean")
  count <- as.double(n) * d
  if (count > .Machine$integer.max || file.info(path)$size != count * 8)
    stop("Representation binary byte size differs from shape")
  connection <- file(path, "rb")
  on.exit(close(connection))
  values <- readBin(connection, "double", n = count, size = 8L, endian = "little")
  if (length(values) != count || any(!is.finite(values)))
    stop("Representation binary must contain exactly finite float64 values")
  value <- matrix(values, nrow = n, ncol = d, byrow = TRUE,
    dimnames = list(ids, if (d) paste0("feature_", seq_len(d)) else NULL))
  constant <- !d || all(vapply(seq_len(d), function(j) all(value[, j] == value[1, j]), logical(1)))
  if (!identical(record$constant, constant)) stop("Representation constant flag differs from payload")
  value
}

hierarchy_write_csv <- function(frame, path, append = FALSE) {
  for (field in names(frame)) if (is.numeric(frame[[field]])) {
    missing <- is.na(frame[[field]])
    frame[[field]] <- sprintf("%.17g", frame[[field]])
    frame[[field]][missing] <- NA_character_
  }
  write.table(frame, path, sep = ",", quote = TRUE, row.names = FALSE,
    col.names = !append, append = append, na = "NA", qmethod = "double")
}

hierarchy_partition <- function(graph_info, representation, resolution, seed) {
  fitted <- run_native_kodama_leiden(graph_info, resolution, objective = "modularity", seed = seed)
  evidence <- kodama_graph_assignment_evidence(graph_info, fitted$membership)
  cluster <- fitted$membership
  cluster[evidence$degree == 0L] <- 0L
  data.frame(representation = representation, resolution = resolution, seed = seed,
    label = graph_info$observation_ids, cluster = as.integer(cluster), degree = evidence$degree,
    affinity_margin = evidence$affinity_margin,
    own_affinity_fraction = evidence$own_community_affinity_fraction,
    stringsAsFactors = FALSE, check.names = FALSE)
}

hierarchy_runtime <- function(packages = loadedNamespaces()) {
  packages <- sort(unique(packages), method = "radix")
  package_records <- setNames(lapply(packages, function(package) {
    root <- normalizePath(find.package(package), mustWork = TRUE)
    files <- c(file.path(root, c("DESCRIPTION", "NAMESPACE")),
      list.files(file.path(root, "R"), full.names = TRUE, recursive = TRUE, all.files = TRUE),
      list.files(file.path(root, "libs"), full.names = TRUE, recursive = TRUE, all.files = TRUE),
      file.path(root, "Meta", "package.rds"))
    files <- files[file.exists(files) & !dir.exists(files)]
    list(version = as.character(utils::packageVersion(package)), path = root,
      files = hierarchy_hash_map(files))
  }), packages)
  dlls <- vapply(getLoadedDLLs(), function(dll) dll[["path"]], character(1))
  executables <- c(file.path(R.home("bin"), "exec", "R"),
    file.path(R.home("bin"), "Rscript"), dlls[nzchar(dlls)])
  executables <- executables[file.exists(executables) & !dir.exists(executables)]
  list(r_version = R.version.string, r_platform = R.version$platform,
    package_identity_scope = "loaded namespace DESCRIPTION/NAMESPACE/R code/libraries/Meta package plus loaded DLLs and R executables; not operating-system transitive libraries",
    packages = package_records, executables_and_loaded_dlls = hierarchy_hash_map(executables))
}

hierarchy_package_closure <- function(packages) {
  result <- character()
  while (length(packages)) {
    package <- packages[1L]; packages <- packages[-1L]
    if (package %in% result) next
    root <- find.package(package)
    description <- read.dcf(file.path(root, "DESCRIPTION"))
    fields <- intersect(c("Depends", "Imports"), colnames(description))
    dependencies <- trimws(sub("[[:space:]]*\\(.*", "",
      unlist(strsplit(paste(description[1L, fields], collapse = ","), ",", fixed = TRUE))))
    result <- c(result, package)
    packages <- c(packages, setdiff(dependencies[nzchar(dependencies)], c("R", result)))
  }
  result
}

hierarchy_runtime_files <- function(runtime) {
  paths <- c(names(runtime$executables_and_loaded_dlls),
    unlist(lapply(runtime$packages, function(x) names(x$files)), use.names = FALSE))
  hierarchy_hash_map(paths)
}

hierarchy_kodama_main <- function(args) {
  if (!(length(args) %in% c(3L, 4L)))
    stop("Usage: run_hierarchy_kodama.R input_manifest.json NEW_outdir expected_manifest_sha256 [native_R_library]")
  options(warn = 2)
  for (package in c("digest", "jsonlite")) if (!requireNamespace(package, quietly = TRUE))
    stop("Required package is not installed: ", package)
  manifest_path <- normalizePath(args[1], mustWork = TRUE)
  if (!grepl("^[0-9a-f]{64}$", args[3]) || !identical(hierarchy_sha256(manifest_path), args[3]))
    stop("Input manifest SHA256 differs")
  if (file.exists(args[2]) || dir.exists(args[2])) stop("Output directory already exists; refusing overwrite")
  manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  hierarchy_json_keys(manifest)
  if (!identical(manifest$format, "cellphenotyper_hierarchy_kodama_input") ||
      !identical(manifest$schema_version, "1.0.0")) stop("Unsupported hierarchy KODAMA input schema")
  if (!is.list(manifest$representations) ||
      !setequal(names(manifest$representations), c("local", "context", "combined")))
    stop("Exactly local, context and combined representations are required")
  parameters <- manifest$parameters
  required <- c("ncomp", "M", "Tcycle", "landmarks", "cores", "neighbors", "seed", "cluster_seeds", "resolutions")
  if (!is.list(parameters) || !setequal(names(parameters), required)) stop("Invalid native parameter keys")
  for (field in setdiff(required, c("cluster_seeds", "resolutions")))
    parameters[[field]] <- hierarchy_integer(parameters[[field]], field, if (field == "seed") 0L else 1L)
  cluster_seeds <- hierarchy_numeric_vector(parameters$cluster_seeds, "cluster_seeds", integer = TRUE)
  resolutions <- hierarchy_numeric_vector(parameters$resolutions, "resolutions", positive = TRUE)
  root <- dirname(manifest_path)
  rows_path <- hierarchy_input_file(manifest$rows, root, "rows")
  representation_paths <- setNames(lapply(c("local", "context", "combined"), function(name)
    hierarchy_input_file(manifest$representations[[name]], root, name)), c("local", "context", "combined"))
  all_paths <- c(manifest_path, rows_path, unlist(representation_paths, use.names = FALSE))
  if (anyDuplicated(all_paths)) stop("Input artifacts must have distinct filenames and payloads")
  source_before <- hierarchy_hash_map(all_paths)
  declared <- setNames(c(list(args[3], manifest$rows$sha256),
    lapply(names(representation_paths), function(name) manifest$representations[[name]]$sha256)), all_paths)
  if (!identical(source_before, declared[sort(names(declared), method = "radix")]))
    stop("Input changed between manifest validation and consumption")
  hierarchy_verify(source_before)
  rows <- hierarchy_read_rows(rows_path)
  # Validate all representations before creating output or starting a native fit.
  for (name in names(representation_paths))
    hierarchy_read_matrix(representation_paths[[name]], manifest$representations[[name]], rows$label)
  hierarchy_verify(source_before)

  if (length(args) == 4L) {
    library <- normalizePath(args[4], mustWork = TRUE)
    if (!dir.exists(file.path(library, "KODAMA"))) stop("Explicit native library lacks KODAMA")
    .libPaths(c(library, .libPaths()))
  }
  # Bind installed native/dependency bytes BEFORE loading their namespaces.
  # digest/jsonlite necessarily bootstrap this checksum verifier and are also
  # included in the full runtime receipt below.
  package_snapshot <- hierarchy_runtime(hierarchy_package_closure(c("KODAMA", "Matrix", "igraph")))
  package_before <- hierarchy_runtime_files(package_snapshot)
  for (package in c("KODAMA", "Matrix", "igraph")) if (!requireNamespace(package, quietly = TRUE))
    stop("Required native runtime package is not installed: ", package)
  hierarchy_verify(package_before, "Native package")
  if (length(args) == 4L && !identical(normalizePath(find.package("KODAMA")),
      normalizePath(file.path(library, "KODAMA")))) stop("Explicit native KODAMA library was not loaded")
  if (!all(c("KODAMA.matrix", "KODAMA.graph.materialize") %in% getNamespaceExports("KODAMA")))
    stop("Installed KODAMA lacks required native handle/materialization API")
  matrix_api <- getExportedValue("KODAMA", "KODAMA.matrix")
  fit_required <- c("data", "ncomp", "M", "Tcycle", "landmarks", "n.cores", "graph.neighbors",
    "classifier", "backend", "seed", "return.graph")
  if (!all(fit_required %in% names(formals(matrix_api)))) stop("Unsupported native KODAMA.matrix API")
  script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(script_argument) != 1L) stop("Cannot identify executed R producer")
  script <- normalizePath(sub("^--file=", "", script_argument), mustWork = TRUE)
  helpers <- file.path(dirname(script), c("kodama_graph_export.R", "kodama_graph_clustering.R"))
  producer_before <- hierarchy_hash_map(c(script, helpers))
  for (helper in helpers) source(helper, local = environment(hierarchy_partition))
  hierarchy_verify(producer_before, "Producer code")
  runtime <- hierarchy_runtime()
  runtime_before <- hierarchy_runtime_files(runtime)
  if (!dir.create(args[2], recursive = TRUE)) stop("Cannot create fresh output directory")
  outdir <- normalizePath(args[2], mustWork = TRUE)
  output <- file.path(outdir, "partitions.csv")
  empty <- data.frame(representation = character(), resolution = numeric(), seed = integer(),
    label = character(), cluster = integer(), degree = integer(), affinity_margin = numeric(),
    own_affinity_fraction = numeric())
  hierarchy_write_csv(empty, output)
  records <- list()
  run_count <- 0L
  xy <- as.matrix(rows[c("x", "y")]); rownames(xy) <- rows$label
  for (name in names(representation_paths)) {
    hierarchy_verify(source_before)
    pca <- hierarchy_read_matrix(representation_paths[[name]], manifest$representations[[name]], rows$label)
    hierarchy_verify(source_before)
    if (isTRUE(manifest$representations[[name]]$constant)) {
      records[[name]] <- list(status = "skipped_constant", observation_count = nrow(pca),
        dimensions = ncol(pca), partition_run_count = 0L)
      next
    }
    fit_args <- list(data = pca, ncomp = parameters$ncomp, M = parameters$M,
      Tcycle = parameters$Tcycle, landmarks = parameters$landmarks, n.cores = parameters$cores,
      graph.neighbors = parameters$neighbors, classifier = "knn", backend = "cpu",
      seed = parameters$seed, return.graph = "handle")
    if ("visual.init" %in% names(formals(matrix_api))) fit_args$visual.init <- FALSE
    if ("progress" %in% names(formals(matrix_api))) fit_args$progress <- FALSE
    fit <- do.call(matrix_api, fit_args)
    if (!is.list(fit$knn) || !identical(fit$knn$storage, "handle") ||
        typeof(fit$knn$handle) != "externalptr" || !isTRUE(fit$knn_is_kodama_corrected) ||
        !identical(fit$parameters$classifier, "knn") || !identical(fit$parameters$backend, "cpu"))
      stop("Native fit did not return a corrected CPU/KNN graph handle")
    hierarchy_verify(source_before)
    hierarchy_verify(producer_before, "Producer code")
    representation_out <- file.path(outdir, name)
    dir.create(representation_out)
    pca_file <- file.path(representation_out, "input_features.RData")
    save(pca, xy, file = pca_file, compress = FALSE, version = 3)
    restored <- new.env(parent = emptyenv()); load(pca_file, envir = restored)
    if (!identical(restored$pca, pca) || !identical(restored$xy, xy)) stop("Feature/coordinate RData readback differs")
    graph_receipt <- export_portable_kodama_graph(fit, rows$label, representation_out, pca_file,
      package_version = as.character(utils::packageVersion("KODAMA")), package_revision = "unknown_installed_source_revision")
    portable <- load_portable_kodama_graph(representation_out, rows$label, graph_receipt$file_sha256, pca_file)
    graph_info <- prepare_kodama_affinity_graph(portable, rows$label)
    for (resolution in resolutions) for (seed in cluster_seeds) {
      frame <- hierarchy_partition(graph_info, name, resolution, seed)
      if (!identical(frame$label, rows$label) || nrow(frame) != nrow(rows)) stop("Partition population/order differs")
      hierarchy_write_csv(frame, output, append = TRUE)
      run_count <- run_count + 1L
    }
    records[[name]] <- list(status = "fit", observation_count = nrow(pca), dimensions = ncol(pca),
      requested_native_parameters = parameters, effective_native_parameters = fit$parameters,
      materialized_neighbors = graph_receipt$native_neighbors, graph_sha256 = graph_receipt$file_sha256,
      graph_manifest_sha256 = hierarchy_sha256(file.path(representation_out, "kodama_graph.json")),
      input_features_sha256 = hierarchy_sha256(pca_file), partition_run_count = length(resolutions) * length(cluster_seeds),
      isolates = sum(graph_info$degree == 0L), graph_builds = fit$graph_builds,
      spatial_graph_builds = fit$spatial_graph_builds, graph_feature_mode = fit$graph_feature_mode,
      graph_index_type = fit$graph_index_type, backend = fit$backend, graph_backend = fit$graph_backend,
      optimization_backend = fit$optimization_backend, dissimilarity_backend = fit$dissimilarity_backend,
      actual_cores = fit$n.cores, native_run_diagnostics = fit$run_diagnostics,
      landmark_represented_strata = fit$landmark_represented_strata,
      ncomp_applicability = graph_receipt$ncomp_applicability)
    rm(pca, restored, fit, portable, graph_info)
  }
  hierarchy_verify(source_before)
  hierarchy_verify(producer_before, "Producer code")
  hierarchy_verify(runtime_before, "Native runtime")
  hierarchy_verify(package_before, "Native package")
  produced <- list.files(outdir, recursive = TRUE, full.names = TRUE, all.files = TRUE)
  produced <- produced[!dir.exists(produced)]
  output_hashes <- setNames(lapply(produced, hierarchy_sha256), substring(produced, nchar(outdir) + 2L))
  summary <- list(format = "cellphenotyper_hierarchy_kodama_runner", schema_version = "1.0.0", status = "pass",
    observation_count = nrow(rows), partition_run_count = run_count, partition_rows = nrow(rows) * run_count,
    representations = records, requested_parameters = parameters,
    feature_protocol = "raw_data_native_retained_graph",
    feature_protocol_description = "native raw-feature KNN corrected graph within an already fixed parent; no coordinates supplied to fitting",
    repeatability_warning = "Native retained graph construction may vary with parallel cores; requested cores are preserved and each realized corrected graph is pinned, not replaced by a fixed-graph protocol",
    coordinates = "physical audit-only x,y saved unchanged; never fit inputs",
    clustering = "native corrected graph affinity 1/(1+d), symmetric max union, Leiden modularity; no forced K or nearest assignment",
    assignment_evidence = "own vs strongest-other weighted community affinity, not biological confidence/probability; isolates cluster 0 with NA evidence",
    scope = manifest$scope, provenance = manifest$provenance,
    sources_before = source_before, sources_after = hierarchy_hash_map(names(source_before)),
    producer_before = producer_before, producer_after = hierarchy_hash_map(names(producer_before)),
    runtime = runtime, runtime_before = runtime_before, runtime_after = hierarchy_hash_map(names(runtime_before)),
    native_package_before_load = package_before,
    native_package_after = hierarchy_hash_map(names(package_before)),
    output_sha256 = output_hashes)
  # This receipt is the completion marker and is intentionally written last.
  jsonlite::write_json(summary, file.path(outdir, "runner_summary.json"), auto_unbox = TRUE,
    pretty = TRUE, null = "null", na = "null", digits = NA)
  invisible(summary)
}

if (sys.nframe() == 0L) hierarchy_kodama_main(commandArgs(TRUE))
