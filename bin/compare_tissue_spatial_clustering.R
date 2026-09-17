#!/usr/bin/env Rscript
# Experimental saved-feature comparison; never a production-default selector.
args <- commandArgs(TRUE)
if (length(args) != 7L) stop("Usage: compare_tissue_spatial_clustering.R inspect|fit source observations.csv output_directory dimensions settings.json input_hashes.json")
script <- normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
for (package in c("digest", "jsonlite")) if (!requireNamespace(package, quietly = TRUE)) stop("Missing package: ", package)
inputs <- jsonlite::read_json(args[7], simplifyVector = FALSE)
if (!is.list(inputs) || !length(inputs) || anyDuplicated(names(inputs)) ||
    any(!vapply(inputs, function(x) is.character(x) && length(x) == 1L && grepl("^[0-9a-f]{64}$", x), logical(1))))
  stop("Invalid producer-bound R input hashes")
verify_inputs <- function() {
  actual <- lapply(names(inputs), function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE))
  names(actual) <- names(inputs)
  if (!identical(actual, inputs)) stop("R input changed during consumption")
}
require_bound <- function(paths) {
  if (!all(normalizePath(paths) %in% names(inputs))) stop("An R-consumed file lacks producer binding")
}
verify_inputs()
require_bound(c(script, file.path(dirname(script), c("kodama_graph_export.R", "kodama_graph_clustering.R", "kodama_spatial_regularization.R")), args[3], args[6]))
source(file.path(dirname(script), "kodama_graph_export.R"))
source(file.path(dirname(script), "kodama_graph_clustering.R"))
source(file.path(dirname(script), "kodama_spatial_regularization.R"))
options(warn = 2)
write_exact_csv <- function(frame, path) {
  # write.table's numeric formatting need not round-trip arbitrary doubles.
  # Format numeric cells explicitly; readers still recover their numeric types.
  for (name in names(frame)) if (is.numeric(frame[[name]])) {
    missing <- is.na(frame[[name]])
    values <- sprintf("%.17g", frame[[name]])
    values[missing] <- NA_character_
    frame[[name]] <- values
  }
  write.csv(frame, path, row.names = FALSE, na = "NA")
}
mode <- args[1]; directory <- normalizePath(args[2]); outdir <- normalizePath(args[4])
dimensions <- as.integer(args[5]); settings <- jsonlite::read_json(args[6], simplifyVector = TRUE)
if (!(mode %in% c("inspect", "fit")) || is.na(dimensions) || dimensions < 1L) stop("Invalid mode/dimensions")
finish_binding <- function(produced) {
  verify_inputs()
  paths <- file.path(outdir, produced)
  output_hashes <- setNames(lapply(paths, kodama_graph_sha256), paths)
  jsonlite::write_json(list(inputs = inputs, outputs = output_hashes),
    file.path(outdir, paste0(mode, "_input_verification.json")), auto_unbox = TRUE, pretty = TRUE)
}
require_bound(file.path(directory, c("kodama_graph.rds", "kodama_graph.json", paste0("kodama_full_", dimensions, ".RData"))))
k <- new.env(); p <- new.env()
load(file.path(directory, paste0("kodama_full_", dimensions, ".RData")), envir = k)
metadata <- k$representation_metadata
if (!is.list(metadata) || !isTRUE(metadata$kodama_graph_available) ||
    !identical(metadata$kodama_graph_file, "kodama_graph.rds") ||
    !identical(metadata$kodama_graph_manifest_file, "kodama_graph.json")) stop("No producer-bound native graph")
pca_name <- kodama_graph_basename(metadata$pca_file, "PCA sidecar")
require_bound(file.path(directory, pca_name))
load(file.path(directory, pca_name), envir = p)
ids <- rownames(p$pca)
if (!is.character(ids) || !length(ids) || anyNA(ids) || anyDuplicated(ids) || any(!nzchar(ids)) ||
    !identical(ids, rownames(k$vis)) || !identical(ids, as.character(k$common_ids)) ||
    !identical(ids, rownames(p$xy)) || !identical(ids, rownames(k$xy)) ||
    !identical(p$xy, k$xy) || !is.matrix(p$pca) || !is.numeric(p$pca) ||
    ncol(p$pca) != dimensions || any(!is.finite(p$pca)) || any(!is.finite(p$xy)) || ncol(p$xy) != 2L)
  stop("Exact PCA/native/visualization observation and coordinate contract failed")
portable <- load_portable_kodama_graph(directory, ids, metadata$kodama_graph_sha256,
                                      file.path(directory, pca_name))
observations <- read.csv(args[3], colClasses = c(label = "character"), check.names = FALSE, na.strings = character())
required <- c("label", "x", "y", "grid_row", "grid_col")
if (!all(required %in% names(observations)) || anyNA(observations$label) ||
    anyDuplicated(observations$label) || !setequal(observations$label, ids)) stop("Grid population differs from saved features")
observations <- observations[match(ids, observations$label), , drop = FALSE]
if (!is.numeric(observations$x) || !is.numeric(observations$y) ||
    !identical(unname(as.matrix(observations[c("x", "y")])), unname(p$xy))) {
  # R integer-versus-double storage is irrelevant; literal numeric values are not.
  if (!isTRUE(all.equal(unname(as.matrix(observations[c("x", "y")])), unname(p$xy), tolerance = 0,
                       check.attributes = FALSE))) stop("Grid coordinates differ from saved PCA coordinates")
}
if (mode == "inspect") {
  write_exact_csv(observations[required], file.path(outdir, "observations.csv"))
  jsonlite::write_json(list(observations = length(ids), dimensions = ncol(p$pca), pca_file = pca_name,
    exact_population_and_coordinates = TRUE, source_grid_reordered_by_literal_id = TRUE,
    graph_sha256 = portable$manifest$file_sha256, requested_kodama_ncomp = metadata$requested_kodama_ncomp,
    effective_kodama_ncomp = metadata$effective_kodama_ncomp,
    actual_kodama_classifier = metadata$actual_kodama_classifier,
    kodama_ncomp_applicability = metadata$kodama_ncomp_applicability,
    native_parameters = portable$metadata,
    feature_provenance = list(legacy_source_scope = metadata$source_scope,
      embedding_input_provenance = metadata$embedding_input_provenance,
      rawdata_input_sha256 = metadata$rawdata_input_sha256, pca_preprocessing = metadata$pca_preprocessing,
      missing_fields = c("embedding_input_provenance", "rawdata_input_sha256", "pca_preprocessing")[
        vapply(c("embedding_input_provenance", "rawdata_input_sha256", "pca_preprocessing"), function(x) is.null(metadata[[x]]), logical(1))]),
    r_version = R.version.string,
    packages = setNames(lapply(c("Matrix", "igraph", "jsonlite", "digest"), function(x) as.character(packageVersion(x))),
                        c("Matrix", "igraph", "jsonlite", "digest"))),
    file.path(outdir, "input_contract.json"), auto_unbox = TRUE, pretty = TRUE, null = "null", digits = NA)
  finish_binding(c("observations.csv", "input_contract.json"))
  quit(status = 0)
}
require_bound(file.path(outdir, c("adjacency.csv", "vertices.csv")))
edges <- read.csv(file.path(outdir, "adjacency.csv"), check.names = FALSE, na.strings = character(),
  colClasses = c(source_index = "numeric", target_index = "numeric", distance_um = "numeric",
                boundary_strength = "numeric", boundary_weight = "numeric"))
vertices <- read.csv(file.path(outdir, "vertices.csv"), colClasses = c(label = "character"), check.names = FALSE, na.strings = character())
if (!identical(vertices$label, ids) || !"in_tissue_support" %in% names(vertices) ||
    anyNA(vertices$in_tissue_support)) stop("Spatial vertices differ from saved features")
vertices$in_tissue_support <- as.logical(vertices$in_tissue_support)
if (anyNA(vertices$in_tissue_support)) stop("Invalid support status")
if (nrow(edges) && any(!vertices$in_tissue_support[c(edges$source_index, edges$target_index)]))
  stop("An added local edge touches an unsupported vertex")
seeds <- settings$seeds; weight <- settings$spatial_weight; resolution <- settings$resolution
target <- settings$target_clusters
if (!is.numeric(seeds) || length(seeds) < 2L || any(!is.finite(seeds)) || any(seeds != as.integer(seeds)) ||
    anyDuplicated(seeds) || !is.numeric(target) || length(target) != 1L || target != 2L)
  stop("Comparison requires at least two distinct integer seeds and explicit sensitivity target K=2")
base <- prepare_kodama_affinity_graph(portable, ids)
variants <- list(baseline = base,
  local_feature = regularize_kodama_graph(base, edges, p$pca, spatial_weight = weight, boundary_aware = FALSE),
  boundary_feature = regularize_kodama_graph(base, edges, p$pca, spatial_weight = weight, boundary_aware = TRUE))
# Exact zero-strength control is mandatory, irrespective of requested weight.
zero <- regularize_kodama_graph(base, edges, p$pca, spatial_weight = 0, boundary_aware = TRUE)
if (!identical(base$affinity, zero$affinity) || !identical(base$graph, zero$graph)) stop("Zero-strength control changed the baseline")
all_assignments <- list(); run_records <- list(); partitions <- list()
for (variant in names(variants)) {
  graph <- variants[[variant]]
  if (!identical(graph$observation_ids, ids)) stop("A graph variant changed vertex identity")
  if (variant != "baseline") write_exact_csv(graph$spatial_local_edges,
    file.path(outdir, paste0(variant, "_edges.csv")))
  partitions[[variant]] <- list()
  for (seed in seeds) {
    start <- proc.time()[["elapsed"]]
    fit <- run_native_kodama_leiden(graph, resolution, "modularity", seed)
    merged <- collapse_kodama_graph_to_target(graph, fit$membership, target)
    labels <- merged$membership
    evidence <- kodama_graph_assignment_evidence(graph, labels)
    within <- if (nrow(edges)) labels[edges$source_index] == labels[edges$target_index] else logical()
    local <- igraph::make_empty_graph(length(ids), directed = FALSE)
    if (any(within)) local <- igraph::add_edges(local, as.vector(t(as.matrix(edges[within, c("source_index", "target_index")]))))
    supported <- which(vertices$in_tissue_support)
    fragments <- if (length(supported)) igraph::components(igraph::induced_subgraph(local, supported))$no else 0L
    partitions[[variant]][[as.character(seed)]] <- labels
    all_assignments[[length(all_assignments) + 1L]] <- data.frame(label = ids, x = observations$x, y = observations$y,
      variant = variant, seed = seed, cluster = labels, raw_cluster = fit$membership,
      in_tissue_support = vertices$in_tissue_support,
      affinity_margin = evidence$affinity_margin, own_affinity_fraction = evidence$own_community_affinity_fraction,
      graph_degree = graph$degree, stringsAsFactors = FALSE)
    run_records[[length(run_records) + 1L]] <- list(variant = variant, seed = seed,
      raw_cluster_count = fit$raw_cluster_count, final_cluster_count = length(unique(labels)),
      target_merge_applied = merged$applied, target_merge_history = merged$merge_history,
      cluster_counts_all = as.list(table(labels)),
      cluster_counts_supported = as.list(table(factor(labels[supported], levels = sort(unique(labels))))),
      supported_local_same_label_components = fragments,
      local_edge_disagreement_fraction = if (length(within)) mean(!within) else NULL,
      low_affinity_margin_count = sum(!is.finite(evidence$affinity_margin) | evidence$affinity_margin < .1),
      graph_connected_components = graph$connected_components,
      graph_isolates = sum(graph$degree == 0), clustering_seconds = proc.time()[["elapsed"]] - start)
    cat(sprintf("Completed %s seed %d: %d raw -> %d target clusters\n", variant, seed, fit$raw_cluster_count, length(unique(labels))))
  }
}
comparisons <- list()
for (variant in names(partitions)) {
  for (pair in combn(as.character(seeds), 2, simplify = FALSE)) comparisons[[length(comparisons) + 1L]] <- list(
    comparison = "within_variant_seed", variant = variant, seed_a = as.integer(pair[1]), seed_b = as.integer(pair[2]),
    ari_all_observations = igraph::compare(partitions[[variant]][[pair[1]]], partitions[[variant]][[pair[2]]], method = "adjusted.rand"))
  if (variant != "baseline") for (seed in as.character(seeds)) comparisons[[length(comparisons) + 1L]] <- list(
    comparison = "same_seed_vs_baseline", variant = variant, seed_a = as.integer(seed), seed_b = as.integer(seed),
    ari_all_observations = igraph::compare(partitions[[variant]][[seed]], partitions$baseline[[seed]], method = "adjusted.rand"))
}
write_exact_csv(do.call(rbind, all_assignments), file.path(outdir, "assignments.csv"))
jsonlite::write_json(list(status = "pass", observation_count = length(ids), zero_weight_control_exact = TRUE,
  target_clusters = target, target_interpretation = "forced_count_sensitivity_not_independent_domain_discovery",
  spatial_weight = weight, resolution = resolution, seeds = seeds, runs = run_records, comparisons = comparisons,
  regularization = lapply(variants[c("local_feature", "boundary_feature")], function(x) x$spatial_regularization),
  geometry_scope = "Only added local edges obey physical/gap constraints; original nonlocal feature affinities remain",
  metric_scope = "Descriptive seed stability and local fragmentation, not independent boundary or histological accuracy",
  margins = "Variant-specific weighted affinity evidence, not calibrated probabilities or directly comparable confidence"),
  file.path(outdir, "clustering_comparison.json"), auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null", digits = NA)
finish_binding(c("assignments.csv", "clustering_comparison.json", "local_feature_edges.csv", "boundary_feature_edges.csv"))
