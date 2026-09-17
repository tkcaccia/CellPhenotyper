# Experimental, additive local affinity for an already verified KODAMA graph.
# Geometry/support provenance belongs to the physical-edge producer. Existing
# nonlocal KODAMA edges are never removed or claimed to avoid tissue gaps.

regularize_kodama_graph <- function(graph_info, edges, pca, spatial_weight = 0.1,
                                    boundary_aware = TRUE) {
  for (package in c("Matrix", "igraph")) {
    if (!requireNamespace(package, quietly = TRUE)) stop("Required package is unavailable: ", package)
  }
  numeric_finite <- function(x) is.numeric(x) && !is.complex(x) && all(is.finite(x))
  if (!numeric_finite(spatial_weight) || length(spatial_weight) != 1L || spatial_weight < 0)
    stop("spatial_weight must be one finite nonnegative number")
  if (!is.logical(boundary_aware) || length(boundary_aware) != 1L || is.na(boundary_aware))
    stop("boundary_aware must be one nonmissing logical value")
  if (!is.list(graph_info) || !is.null(graph_info$spatial_regularization) ||
      !is.null(graph_info$spatial_local_edges)) stop("Expected an unregularized graph_info list")
  ids <- graph_info$observation_ids
  if (!is.character(ids) || !length(ids) || anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids))
    stop("Graph observation IDs must be unique nonempty strings")
  n <- length(ids)
  affinity <- graph_info$affinity
  if (!inherits(affinity, "dgCMatrix") || !identical(dim(affinity), c(n, n)) ||
      !identical(dimnames(affinity), list(ids, ids))) stop("Affinity shape and ordered IDs must match exactly")
  methods::validObject(affinity)
  if (any(!is.finite(affinity@x)) || any(affinity@x <= 0))
    stop("Stored original affinities must be finite and strictly positive")
  columns <- rep.int(seq_len(n), diff(affinity@p))
  if (any(affinity@i + 1L == columns)) stop("Original affinity must not contain self edges")
  delta <- affinity - Matrix::t(affinity)
  if (any(delta@x != 0)) stop("Original affinity must be exactly symmetric")
  graph <- graph_info$graph
  if (!igraph::is_igraph(graph) || igraph::is_directed(graph) || !igraph::is_simple(graph) ||
      igraph::vcount(graph) != n || !identical(igraph::V(graph)$name, ids))
    stop("Graph vertices and ordered IDs must match the undirected simple affinity graph")
  graph_weights <- igraph::E(graph)$weight
  if ((igraph::ecount(graph) > 0L && !numeric_finite(graph_weights)) ||
      length(graph_weights) != igraph::ecount(graph) || any(graph_weights <= 0))
    stop("Graph edge weights must be finite and positive")
  # igraph omits the weight attribute for a genuinely empty prepared graph.
  reconstructed <- if (igraph::ecount(graph)) igraph::as_adjacency_matrix(graph, attr = "weight", sparse = TRUE) else
    igraph::as_adjacency_matrix(graph, sparse = TRUE)
  if (any((affinity - reconstructed)@x != 0)) stop("Graph edge weights differ from the affinity matrix")
  degree <- as.integer(igraph::degree(graph))
  strength <- as.numeric(Matrix::rowSums(affinity))
  components <- as.integer(igraph::components(graph)$no)
  if (!identical(graph_info$degree, degree) || !identical(graph_info$strength, strength) ||
      !identical(graph_info$connected_components, components))
    stop("Graph degree, strength or connected-component metadata differs")
  if (!is.character(graph_info$affinity_transform) || length(graph_info$affinity_transform) != 1L ||
      is.na(graph_info$affinity_transform) || !nzchar(graph_info$affinity_transform))
    stop("Original affinity transform must be explicitly identified")
  if (!is.matrix(pca) || !numeric_finite(pca) || nrow(pca) != n || ncol(pca) < 1L ||
      !identical(rownames(pca), ids)) stop("PCA must be a finite numeric matrix with exact graph row IDs/order")
  required <- c("source_index", "target_index", "distance_um", "boundary_weight")
  if (!is.data.frame(edges) || anyDuplicated(names(edges)) || !all(required %in% names(edges)) ||
      !all(vapply(edges[required], numeric_finite, logical(1))))
    stop("Physical edges require finite numeric source_index, target_index, distance_um and boundary_weight")
  i <- edges$source_index
  j <- edges$target_index
  if (any(i != floor(i)) || any(j != floor(j)) || any(i < 1) || any(j > n) || any(i >= j) ||
      anyDuplicated(edges[c("source_index", "target_index")]))
    stop("Physical edges must be unique unordered integer pairs with 1 <= source_index < target_index <= N")
  if (any(edges$distance_um <= 0) || any(edges$boundary_weight < 0 | edges$boundary_weight > 1))
    stop("Physical distance_um must be positive and boundary_weight must be in [0,1]")
  added_columns <- c("pca_distance", "feature_kernel", "unscaled_local_weight", "added_affinity_weight")
  if (any(added_columns %in% names(edges))) stop("Physical edges already contain reserved regularization columns")

  # Stable Euclidean norms use O(E) vectors, never an E-by-D or N-by-N matrix.
  # The PCA coordinates are feature coordinates, not slide/UMAP coordinates.
  distances <- numeric(nrow(edges))
  for (axis in seq_len(ncol(pca))) {
    difference <- abs(pca[i, axis] - pca[j, axis])
    if (any(!is.finite(difference))) stop("Adjacent PCA differences overflow float64")
    scale <- pmax(distances, difference)
    nonzero <- scale > 0
    distances[nonzero] <- scale[nonzero] * sqrt((distances[nonzero] / scale[nonzero])^2 +
                                                (difference[nonzero] / scale[nonzero])^2)
  }
  if (any(!is.finite(distances))) stop("Adjacent PCA distances overflow float64")
  positive <- distances[distances > 0]
  sigma <- if (length(positive)) stats::median(positive) else 0
  kernel <- if (sigma > 0) exp(-0.5 * (distances / sigma)^2) else rep(1, length(distances))
  local <- kernel * if (boundary_aware) edges$boundary_weight else 1
  feature_mass <- 2 * sum(kernel)
  local_mass <- 2 * sum(local)
  original_mass <- sum(affinity@x)
  target_mass <- spatial_weight * original_mass
  if (!all(is.finite(c(feature_mass, local_mass, original_mass, target_mass))))
    stop("Original or requested affinity mass overflowed float64")
  reason <- if (spatial_weight == 0) "spatial_weight_zero" else if (!nrow(edges)) "no_physical_edges" else
    if (original_mass == 0) "original_zero_mass" else if (feature_mass == 0) "feature_kernel_zero_mass" else
      if (local_mass == 0) "local_zero_mass" else
      if (target_mass == 0) "requested_mass_underflow" else "applied"
  added <- numeric(nrow(edges))
  if (reason == "applied") {
    # Both comparison variants share the UNGATED feature-only denominator.
    # Boundary penalties can only reduce the added edge weights/mass; never
    # renormalize away their suppression. Normalize before multiplying to
    # avoid overflowing a target/feature_mass intermediate.
    added <- (target_mass / 2) * (kernel / sum(kernel))
    if (boundary_aware) added <- added * edges$boundary_weight
    if (!any(added > 0)) reason <- "added_weights_underflow"
  }
  result <- graph_info
  added_strength <- numeric(n)
  if (reason == "applied") {
    keep <- added > 0
    local_affinity <- Matrix::sparseMatrix(i = c(i[keep], j[keep]), j = c(j[keep], i[keep]),
      x = rep(added[keep], 2L), dims = c(n, n), dimnames = list(ids, ids), giveCsparse = TRUE)
    added_strength <- as.numeric(Matrix::rowSums(local_affinity))
    result$affinity <- affinity + local_affinity
    if (any(!is.finite(result$affinity@x))) stop("Combined affinity overflowed float64")
    result$graph <- igraph::graph_from_adjacency_matrix(result$affinity, mode = "undirected", weighted = TRUE, diag = FALSE)
    if (!identical(igraph::V(result$graph)$name, ids)) stop("Regularization changed graph vertex identities")
    result$degree <- as.integer(igraph::degree(result$graph))
    result$strength <- as.numeric(Matrix::rowSums(result$affinity))
    result$connected_components <- as.integer(igraph::components(result$graph)$no)
    result$affinity_transform <- paste0(graph_info$affinity_transform, "+experimental_mass_normalized_local_PCA_affinity")
  }
  existing_candidate <- if (nrow(edges)) as.numeric(affinity[cbind(i, j)]) > 0 else logical()
  native_positive <- strength > 0
  strength_ratio <- added_strength[native_positive] / strength[native_positive]
  finite_ratio <- strength_ratio[is.finite(strength_ratio)]
  ratio_names <- c("min", "p25", "median", "p75", "p95", "p99", "max")
  ratio_quantiles <- if (length(finite_ratio)) unname(stats::quantile(finite_ratio,
    probs = c(0, .25, .5, .75, .95, .99, 1), names = FALSE)) else rep(NA_real_, length(ratio_names))
  result$spatial_local_edges <- edges
  result$spatial_local_edges$pca_distance <- distances
  result$spatial_local_edges$feature_kernel <- kernel
  result$spatial_local_edges$unscaled_local_weight <- local
  result$spatial_local_edges$added_affinity_weight <- added
  result$spatial_regularization <- list(
    schema_version = "1.0.0", method = "additive_feature_mass_normalized_boundary_gated_physical_PCA_affinity",
    experimental = TRUE, applied = reason == "applied", reason = reason,
    spatial_weight = spatial_weight, boundary_aware = boundary_aware,
    observation_count = n, pca_dimensions = ncol(pca), candidate_edge_count = nrow(edges),
    positive_added_edge_count = sum(added > 0), sigma_pca = sigma,
    candidate_existing_edge_count = sum(existing_candidate), candidate_new_edge_count = sum(!existing_candidate),
    positive_added_existing_edge_count = sum(added > 0 & existing_candidate),
    positive_added_new_edge_count = sum(added > 0 & !existing_candidate),
    sigma_policy = "median_strictly_positive_adjacent_PCA_distance; all_zero_distances_use_kernel_one",
    feature_kernel = "exp(-PCA_distance^2/(2*sigma_pca^2))",
    physical_distance_role = "validated positive audit field; no additional distance decay",
    original_affinity_mass = original_mass, unscaled_local_affinity_mass = local_mass,
    feature_only_local_affinity_mass = feature_mass,
    normalization_denominator = "feature_only_local_affinity_mass; shared by gated and ungated variants",
    boundary_gate_normalization = "applied after feature-only normalization; never renormalized",
    requested_added_affinity_mass = target_mass, added_affinity_mass = 2 * sum(added),
    actual_added_to_original_mass_fraction = if (original_mass > 0) 2 * sum(added) / original_mass else 0,
    added_native_strength_ratio = list(
      scope = "added physical affinity divided by original native strength; zero-native-strength nodes reported separately",
      native_positive_node_count = sum(native_positive), native_zero_node_count = sum(!native_positive),
      native_zero_with_added_strength_node_count = sum(!native_positive & added_strength > 0),
      ratio_overflow_node_count = sum(!is.finite(strength_ratio)),
      finite_ratio_quantiles = as.list(stats::setNames(ratio_quantiles, ratio_names)),
      max = if (!length(strength_ratio) || any(!is.finite(strength_ratio))) NA_real_ else max(strength_ratio)),
    resulting_affinity_mass = sum(result$affinity@x), mass_convention = "sum_of_both_directed_entries_of_symmetric_affinity",
    original_affinity_transform = graph_info$affinity_transform,
    original_nonlocal_edges_retained = TRUE, original_weights_not_subtracted = TRUE,
    geometry_verification = "external_edge_producer; this helper validates indices and numeric fields, not image/support provenance",
    gap_constraint_scope = "added physical edges only; existing nonlocal KODAMA edges remain unchanged/additive",
    pca_provenance_scope = "caller_supplied_exact_row_bound_PCA; no source-file provenance is inferred",
    scientific_claim = "experimental feature-plus-local-affinity graph; no biological accuracy or calibrated confidence claim",
    memory_policy = "sparse O(N+E) graph structures plus O(E) edge vectors; no dense N-by-N or E-by-D matrix",
    numerical_policy = "float64; mass normalization and addition may round; no bitwise claim for positive weight")
  result
}
