# Graph-native clustering helpers. No visualization/PCA coordinates are accepted
# by these functions, and absent sparse entries are never interpreted as zero
# distances. The portable receipt is validated by kodama_graph_export.R first.

kodama_graph_membership <- function(membership, observations) {
  if (length(membership) != observations || anyNA(membership) ||
      any(!is.finite(membership)) || any(membership != as.integer(membership)) ||
      any(membership < 1L)) stop("Graph membership must contain one positive integer per observation")
  as.integer(match(membership, sort(unique(membership))))
}

prepare_kodama_affinity_graph <- function(portable, expected_observation_ids = NULL) {
  for (package in c("Matrix", "igraph")) {
    if (!requireNamespace(package, quietly = TRUE)) stop(sprintf("Required package '%s' is not installed", package))
  }
  ids <- portable$observation_ids
  distances <- portable$distance_graph
  if (!is.character(ids) || !length(ids) || anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids))
    stop("Portable graph needs unique nonempty observation IDs")
  if (!inherits(distances, "dgCMatrix") || !identical(dim(distances), rep(length(ids), 2L)))
    stop("Portable graph must be a square dgCMatrix matching all observation IDs")
  methods::validObject(distances)
  if (any(!is.finite(distances@x)) || any(distances@x < 0))
    stop("Stored graph distances must be finite and nonnegative")
  columns <- rep.int(seq_along(ids), diff(distances@p))
  if (any(distances@i + 1L == columns)) stop("Portable graph must not contain self edges")
  # Transform STORED entries before any sparse operations: explicitly stored
  # distance zero is a genuine edge with affinity one; absent entries stay absent.
  # base::pmax on Matrix objects can silently allocate dense N-by-N arrays.
  # Aggregate only stored edges by their unordered integer endpoint pair.
  # Sorting two integer columns avoids lossy floating-point encoded pair keys.
  rows <- distances@i + 1L
  low <- pmin(rows, columns)
  high <- pmax(rows, columns)
  weights <- 1 / (1 + distances@x)
  if (length(weights)) {
    ordering <- order(low, high, -weights, method = "radix")
    low <- low[ordering]
    high <- high[ordering]
    weights <- weights[ordering]
    first_pair <- c(TRUE, low[-1L] != low[-length(low)] | high[-1L] != high[-length(high)])
    low <- low[first_pair]
    high <- high[first_pair]
    weights <- weights[first_pair]
  }
  affinity <- Matrix::sparseMatrix(i = c(low, high), j = c(high, low), x = rep(weights, 2L),
    dims = rep(length(ids), 2L), dimnames = list(ids, ids), giveCsparse = TRUE)
  if (!is.null(expected_observation_ids)) {
    if (!is.character(expected_observation_ids) || anyNA(expected_observation_ids) ||
        anyDuplicated(expected_observation_ids) || !setequal(ids, expected_observation_ids))
      stop("Graph and expected observation IDs must match exactly")
    affinity <- affinity[expected_observation_ids, expected_observation_ids, drop = FALSE]
    ids <- expected_observation_ids
  }
  graph <- igraph::graph_from_adjacency_matrix(affinity, mode = "undirected", weighted = TRUE, diag = FALSE)
  if (igraph::vcount(graph) != length(ids) || !identical(igraph::V(graph)$name, ids))
    stop("Affinity graph lost or reordered observations")
  list(affinity = affinity, graph = graph, observation_ids = ids,
       degree = as.integer(igraph::degree(graph)),
       strength = as.numeric(Matrix::rowSums(affinity)),
       connected_components = as.integer(igraph::components(graph)$no),
       affinity_transform = "stored_distance_to_1_over_1_plus_d_symmetric_max_union")
}

run_native_kodama_leiden <- function(graph_info, resolution, objective = "modularity", seed = 1L) {
  if (length(resolution) != 1L || !is.finite(resolution) || resolution <= 0)
    stop("Native graph Leiden requires a positive fixed resolution")
  if (!(objective %in% c("modularity", "cpm"))) stop("Native graph Leiden objective must be modularity or cpm")
  if (length(seed) != 1L || !is.finite(seed) || seed != as.integer(seed)) stop("Invalid graph clustering seed")
  n <- length(graph_info$observation_ids)
  membership <- integer(n)
  supported <- which(graph_info$degree > 0L)
  isolated <- which(graph_info$degree == 0L)
  set.seed(as.integer(seed))
  if (length(supported)) {
    supported_graph <- igraph::induced_subgraph(graph_info$graph, supported)
    fitted <- igraph::cluster_leiden(supported_graph,
      objective_function = if (objective == "cpm") "CPM" else "modularity",
      resolution = resolution, weights = igraph::E(supported_graph)$weight)
    membership[supported] <- as.integer(fitted$membership)
  }
  # An isolate has no graph-supported community assignment. Retain its vertex
  # and a distinct raw cluster, then expose missing affinity evidence downstream.
  if (length(isolated)) membership[isolated] <- max(membership) + seq_along(isolated)
  membership <- kodama_graph_membership(membership, n)
  names(membership) <- graph_info$observation_ids
  list(membership = membership, final_membership = membership,
       resolution = resolution, raw_cluster_count = length(unique(membership)),
       final_cluster_count = length(unique(membership)),
       modularity = if (igraph::ecount(graph_info$graph)) igraph::modularity(graph_info$graph, membership,
         weights = igraph::E(graph_info$graph)$weight) else NA_real_,
       silhouette = NA_real_, tiny_count = NA_integer_, tiny_fraction = NA_real_,
       tiny_threshold = NA_integer_, score = NA_real_,
       landmark_algorithm = "native_kodama_graph_leiden", landmark_cells_used = n,
       landmark_assignment_mode = "native_graph", landmark_assign_k_used = 0L,
       landmark_assignment_knn_backend = NA_character_)
}

collapse_kodama_graph_to_target <- function(graph_info, membership, target) {
  n <- length(graph_info$observation_ids)
  membership <- kodama_graph_membership(membership, n)
  if (length(target) != 1L || !is.finite(target) || target != as.integer(target) || target < 0L || target == 1L)
    stop("Graph target must be 0 (disabled) or an integer >=2")
  initial_count <- length(unique(membership))
  if (target == 0L || initial_count == target)
    return(list(membership = membership, applied = FALSE, initial_count = initial_count, merge_history = ""))
  if (initial_count < target) stop("Native graph produced fewer communities than the requested target; no unsupported splitting is performed")
  history <- character()
  while (length(unique(membership)) > target) {
    count <- length(unique(membership))
    indicator <- Matrix::sparseMatrix(i = seq_len(n), j = membership, x = 1, dims = c(n, count))
    between <- Matrix::crossprod(indicator, graph_info$affinity %*% indicator)
    edges <- Matrix::summary(between)
    edges <- edges[edges$i < edges$j & edges$x > 0, , drop = FALSE]
    if (!nrow(edges)) stop("Cannot force target: remaining graph communities have no supported affinity merge (disconnected components/isolates are preserved)")
    # Maximum total cross-community affinity, with stable numeric-ID tie breaks.
    # This is explicitly a forced-count sensitivity operation, not a discovery.
    selected <- edges[order(-edges$x, edges$i, edges$j)[1L], , drop = FALSE]
    keep <- selected$i[1L]
    drop <- selected$j[1L]
    history <- c(history, sprintf("%d+%d@affinity_sum=%.9g", keep, drop, selected$x[1L]))
    membership[membership == drop] <- keep
    membership <- kodama_graph_membership(membership, n)
  }
  list(membership = membership, applied = TRUE, initial_count = initial_count,
       merge_history = paste(history, collapse = ";"))
}

kodama_graph_assignment_evidence <- function(graph_info, membership) {
  n <- length(graph_info$observation_ids)
  membership <- kodama_graph_membership(membership, n)
  indicator <- Matrix::sparseMatrix(i = seq_len(n), j = membership, x = 1,
                                    dims = c(n, length(unique(membership))))
  by_community <- graph_info$affinity %*% indicator
  own <- as.numeric(by_community[cbind(seq_len(n), membership)])
  alternatives <- Matrix::summary(by_community)
  alternatives <- alternatives[alternatives$j != membership[alternatives$i], , drop = FALSE]
  strongest_other <- numeric(n)
  if (nrow(alternatives)) {
    alternatives <- alternatives[order(alternatives$i, -alternatives$x), , drop = FALSE]
    alternatives <- alternatives[!duplicated(alternatives$i), , drop = FALSE]
    strongest_other[alternatives$i] <- alternatives$x
  }
  supported <- graph_info$strength > 0
  own_fraction <- margin <- rep(NA_real_, n)
  own_fraction[supported] <- own[supported] / graph_info$strength[supported]
  margin[supported] <- (own[supported] - strongest_other[supported]) / graph_info$strength[supported]
  list(own_community_affinity_fraction = own_fraction, affinity_margin = margin,
       degree = graph_info$degree, strength = graph_info$strength,
       is_isolated = !supported,
       semantics = "own_vs_strongest_other_community_weighted_affinity_after_target_merge_not_probability")
}
