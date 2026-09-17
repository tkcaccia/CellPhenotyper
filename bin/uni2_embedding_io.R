# Strict reader for Python-produced, hash-bound UNI2 binary storage shards.
# Storage validation does not establish encoder revision or calibration.

uni2_io_require <- function() {
  for (package in c("data.table", "jsonlite", "digest"))
    if (!requireNamespace(package, quietly = TRUE)) stop("Binary UNI2 requires R package: ", package)
}

uni2_io_sha256 <- function(path) {
  uni2_io_require()
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

uni2_io_names <- function(values, field, allow_empty = FALSE) {
  if (is.list(values) && all(vapply(values, function(x) is.character(x) && length(x) == 1L, logical(1))))
    values <- unlist(values, use.names = FALSE)
  if (!is.character(values) || (!allow_empty && !length(values)) || anyNA(values) ||
      any(!nzchar(values)) || any(trimws(values) != values) || any(grepl("[[:cntrl:]]", values)) || anyDuplicated(values))
    stop("Invalid or duplicate binary UNI2 ", field)
  values
}

uni2_io_no_duplicate_keys <- function(value) {
  if (is.list(value)) {
    if (!is.null(names(value)) && anyDuplicated(names(value))) stop("Duplicate binary UNI2 manifest key")
    invisible(lapply(value, uni2_io_no_duplicate_keys))
  }
}

uni2_io_integer <- function(value, field, maximum = 2^53) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
      value < 1 || value != floor(value) || value > maximum) stop("Invalid binary UNI2 ", field)
  as.numeric(value)
}

uni2_io_payload <- function(manifest_path, filename, expected_name) {
  if (!is.character(filename) || length(filename) != 1L || is.na(filename) ||
      !identical(filename, expected_name) || basename(filename) != filename || grepl("[/\\\\]", filename))
    stop("Binary UNI2 payload must use its shard stem in the same directory")
  root <- dirname(normalizePath(manifest_path, mustWork = TRUE))
  path <- file.path(root, filename)
  if (!file.exists(path) || dir.exists(path)) stop("Binary UNI2 payload is missing: ", filename)
  actual <- normalizePath(path, mustWork = TRUE)
  if (!identical(dirname(actual), root)) stop("Binary UNI2 payload escapes its manifest directory")
  actual
}

read_uni2_binary_manifest <- function(path, expected_mode = NULL) {
  uni2_io_require()
  path <- normalizePath(path, mustWork = TRUE)
  if (!grepl("\\.embedding\\.json$", path) || file.info(path)$size > 8 * 1024^2)
    stop("Invalid or oversized binary UNI2 manifest")
  manifest <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  uni2_io_no_duplicate_keys(manifest)
  if (!is.list(manifest) || !identical(manifest$format, "cellphenotyper_uni2_binary") ||
      !identical(manifest$schema_version, "1.0.0")) stop("Unsupported binary UNI2 format/schema")
  if (!is.character(manifest$dtype) || length(manifest$dtype) != 1L ||
      !(manifest$dtype %in% c("<f4", "<f8")) || !identical(manifest$order, "C"))
    stop("Binary UNI2 requires little-endian float32/float64 in C order")
  if (!is.list(manifest$shape) || length(manifest$shape) != 2L) stop("Binary UNI2 shape must be [rows, features]")
  shape <- vapply(manifest$shape, uni2_io_integer, numeric(1), field = "shape dimension", maximum = .Machine$integer.max)
  features <- uni2_io_names(manifest$feature_names, "feature names")
  columns <- uni2_io_names(manifest$row_columns, "row columns")
  if (length(features) != shape[2L] || !("cell_id" %in% columns) || length(intersect(features, columns)))
    stop("Binary UNI2 feature/row schema mismatch")
  if (!is.null(manifest$embedding_mode)) {
    mode <- manifest$embedding_mode
    if (!is.character(mode) || length(mode) != 1L || !(mode %in% c("tile", "nuclei", "cyto", "inner_square")) ||
        (!is.null(expected_mode) && !identical(mode, expected_mode))) stop("Binary UNI2 embedding mode mismatch")
  }
  stem <- sub("\\.embedding\\.json$", "", basename(path))
  paths <- list()
  for (kind in c("features", "rows")) {
    suffix <- if (kind == "features") ".features.bin" else ".rows.csv"
    paths[[kind]] <- uni2_io_payload(path, manifest[[paste0(kind, "_file")]], paste0(stem, suffix))
    expected_size <- uni2_io_integer(manifest[[paste0(kind, "_size_bytes")]], paste(kind, "byte size"))
    hash <- manifest[[paste0(kind, "_sha256")]]
    if (!is.character(hash) || length(hash) != 1L || is.na(hash) || !grepl("^[0-9a-f]{64}$", hash))
      stop("Binary UNI2 requires SHA256 payload hashes")
    if (file.info(paths[[kind]])$size != expected_size) stop("Binary UNI2 ", kind, " byte-size mismatch")
    if (!identical(uni2_io_sha256(paths[[kind]]), hash)) stop("Binary UNI2 ", kind, " SHA256 mismatch")
  }
  bytes <- if (manifest$dtype == "<f4") 4L else 8L
  if (prod(shape) * bytes > 2^53 || manifest$features_size_bytes != prod(shape) * bytes)
    stop("Binary UNI2 shape/dtype does not match exact feature byte size")
  rows <- withCallingHandlers(data.table::fread(paths$rows, check.names = FALSE,
    colClasses = list(character = "cell_id"), na.strings = NULL, encoding = "UTF-8", showProgress = FALSE),
    warning = function(warning) stop("Binary UNI2 rows parsing failed: ", conditionMessage(warning)))
  if (!identical(names(rows), columns) || nrow(rows) != shape[1L]) stop("Binary UNI2 metadata dimensions/schema mismatch")
  ids <- uni2_io_names(rows$cell_id, "cell_id values")
  list(path = path, manifest = manifest, manifest_sha256 = uni2_io_sha256(path),
       features_path = paths$features, rows_path = paths$rows, rows = rows, ids = ids,
       shape = as.integer(shape), bytes = bytes, feature_names = features)
}

discover_uni2_embedding_input <- function(directory) {
  if (!dir.exists(directory)) stop("Embedding directory does not exist: ", directory)
  paths <- sort(list.files(directory, recursive = TRUE, full.names = TRUE))
  binary <- paths[grepl("\\.embedding\\.json$", paths)]
  metadata <- paths[grepl("\\.rows\\.csv$", paths)]
  payload <- paths[grepl("\\.features\\.bin$", paths)]
  legacy <- paths[grepl("\\.csv(\\.gz)?$", paths) & !grepl("\\.rows\\.csv$", paths)]
  if (length(binary) && length(legacy)) stop("Mixed legacy CSV and binary UNI2 shards are not allowed")
  expected_rows <- sub("\\.embedding\\.json$", ".rows.csv", binary)
  expected_features <- sub("\\.embedding\\.json$", ".features.bin", binary)
  if (!setequal(metadata, expected_rows) || !setequal(payload, expected_features))
    stop("Unowned or incomplete binary UNI2 payload inventory")
  if (!length(binary) && !length(legacy)) stop("No embedding shards found in: ", directory)
  list(storage = if (length(binary)) "binary" else "legacy_csv", files = if (length(binary)) binary else legacy)
}

uni2_io_read_block <- function(connection, row_count, feature_count, bytes) {
  expected <- as.double(row_count) * feature_count
  values <- readBin(connection, what = "double", n = expected, size = bytes, endian = "little")
  if (length(values) != expected) stop("Binary UNI2 truncated feature stream")
  if (any(!is.finite(values))) stop("Binary UNI2 feature vectors must be finite")
  matrix(values, nrow = row_count, ncol = feature_count, byrow = TRUE)
}

uni2_io_unchanged <- function(info) {
  if (!identical(uni2_io_sha256(info$path), info$manifest_sha256) ||
      !identical(uni2_io_sha256(info$features_path), info$manifest$features_sha256) ||
      !identical(uni2_io_sha256(info$rows_path), info$manifest$rows_sha256))
    stop("Binary UNI2 source changed while loading")
}

uni2_io_receipt_json <- function(path) {
  if (file.info(path)$size > 64 * 1024^2) stop("Oversized binary UNI2 extraction receipt")
  value <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  uni2_io_no_duplicate_keys(value)
  if (!is.list(value) || is.null(names(value))) stop("Invalid binary UNI2 extraction receipt")
  value
}

uni2_io_receipt_count <- function(value, field, minimum = 0) {
  if (!is.numeric(value) || length(value) != 1L || !is.finite(value) ||
      value < minimum || value != floor(value) || value > 2^53)
    stop("Invalid binary UNI2 receipt ", field)
  as.numeric(value)
}

uni2_io_receipt_path <- function(root, relative) {
  if (!is.character(relative) || length(relative) != 1L || is.na(relative) || !nzchar(relative) ||
      grepl("^/|\\\\|[[:cntrl:]]", relative) ||
      any(strsplit(relative, "/", fixed = TRUE)[[1L]] %in% c("", ".", "..")))
    stop("Unsafe binary UNI2 receipt payload path")
  path <- file.path(root, relative)
  if (!file.exists(path) || dir.exists(path)) stop("Missing binary UNI2 receipt payload: ", relative)
  path <- normalizePath(path, mustWork = TRUE)
  if (!startsWith(path, paste0(root, "/"))) stop("Binary UNI2 receipt payload escapes its root")
  path
}

# Completion receipts bind stored files and declared extraction identity. Their
# SHA strings are compared, not recomputed by reserializing Python JSON in R.
# Neither this internal consistency check nor an unsigned receipt independently
# verifies the original image, encoder state, or biological validity.
validate_uni2_binary_receipts <- function(directory, shards, mode_name) {
  root <- normalizePath(directory, mustWork = TRUE)
  files <- sort(list.files(root, recursive = TRUE, full.names = TRUE, all.files = TRUE))
  roots <- files[grepl("_embedding_complete\\.json$", files)]
  grids <- files[grepl("_grid_complete\\.json$", files)]
  if (!length(roots) && !length(grids))
    return(list(receipts = list(), status = "legacy_unverified_no_encoder_receipt",
                storage_receipts = "absent", cache_contract_digest = "absent"))
  if (length(roots) != 1L || dirname(roots[[1L]]) != root || !length(grids))
    stop("Binary UNI2 extraction requires one root completion and all grid receipts")
  root_name <- basename(roots[[1L]])
  if (!grepl("^\\..+_embedding_complete\\.json$", root_name)) stop("Invalid binary UNI2 completion receipt name")
  tag <- sub("_embedding_complete\\.json$", "", substring(root_name, 2L))
  if (any(basename(grids) != paste0(".", tag, "_grid_complete.json")))
    stop("Binary UNI2 grid receipt tag differs from root completion")
  completion <- uni2_io_receipt_json(roots[[1L]])
  contract_hash <- completion$cache_contract_sha256
  if (!is.character(contract_hash) || length(contract_hash) != 1L ||
      !grepl("^[0-9a-f]{64}$", contract_hash) ||
      !identical(completion$cache_contract$schema_version, "2.0.0") ||
      !identical(completion$cache_contract$parameters$embedding_storage, "binary"))
    stop("Binary UNI2 completion requires its declared extraction cache contract")
  contract_mode <- completion$cache_contract$parameters$embedding_mode
  paired_secondary <- identical(mode_name, "inner_square") && identical(contract_mode, "tile") &&
    isTRUE(completion$cache_contract$parameters$paired_inner_square_enabled)
  if (!is.null(contract_mode) && !identical(contract_mode, mode_name) && !paired_secondary)
    stop("Binary UNI2 cache contract embedding mode mismatch")
  observation_type <- completion$observation_type
  if (!is.character(observation_type) || length(observation_type) != 1L ||
      !(observation_type %in% c("cell", "grid")) ||
      !identical(completion$cache_contract$parameters$observation_type, observation_type))
    stop("Binary UNI2 completion observation type mismatch")
  check_identity <- function(record, description) {
    if (!identical(record$embedding_storage, "binary") || !identical(record$embedding_mode, mode_name))
      stop("Binary UNI2 ", description, " storage/embedding mode mismatch or omission")
    if (!is.null(record$tag) && !identical(record$tag, tag)) stop("Binary UNI2 receipt tag mismatch")
  }
  check_identity(completion, "completion")
  for (info in shards) {
    if (!identical(info$manifest$embedding_mode, mode_name))
      stop("Binary UNI2 shard embedding mode missing or inconsistent with extraction receipt")
    if (!("observation_type" %in% names(info$rows)) ||
        anyNA(info$rows$observation_type) || any(info$rows$observation_type != observation_type))
      stop("Binary UNI2 shard observation type differs from extraction receipt")
  }
  shard_paths <- vapply(shards, `[[`, character(1), "path")
  # Reuse hashes already read from actual payload bytes; no extra whole-feature
  # hash pass is needed for each nested inventory.
  actual <- list()
  for (info in shards) {
    for (path in c(info$path, info$rows_path, info$features_path)) {
      if (!startsWith(path, paste0(root, "/")) || !is.null(actual[[path]]))
        stop("Duplicate or escaping binary UNI2 extraction payload")
      hash <- if (path == info$path) info$manifest_sha256 else if (path == info$rows_path)
        info$manifest$rows_sha256 else info$manifest$features_sha256
      actual[[path]] <- list(sha256 = hash, size_bytes = file.info(path)$size)
    }
  }
  receipt_sources <- lapply(c(roots, grids), function(path) list(path = path, sha256 = uni2_io_sha256(path)))
  for (record in receipt_sources[-1L])
    actual[[record$path]] <- list(sha256 = record$sha256, size_bytes = file.info(record$path)$size)
  check_inventory <- function(records, directory, key, expected) {
    if (!is.list(records) || !length(records) || !is.null(names(records)))
      stop("Missing or malformed binary UNI2 receipt inventory")
    found <- character(length(records))
    for (i in seq_along(records)) {
      record <- records[[i]]
      if (!is.list(record)) stop("Malformed binary UNI2 receipt inventory record")
      relative <- record[[key]]
      if (key == "name" && (!is.character(relative) || length(relative) != 1L ||
                           basename(relative) != relative)) stop("Unsafe binary UNI2 grid payload name")
      path <- uni2_io_receipt_path(directory, relative)
      if (!(path %in% expected) || path %in% found) stop("Unknown or duplicate binary UNI2 receipt payload")
      if (uni2_io_receipt_count(record$size_bytes, "payload byte size", 1) != actual[[path]]$size_bytes ||
          !identical(record$sha256, actual[[path]]$sha256)) stop("Binary UNI2 receipt payload checksum/size mismatch")
      found[[i]] <- path
    }
    if (!setequal(found, expected)) stop("Binary UNI2 receipt does not cover the exact payload inventory")
    found
  }
  check_inventory(completion$payload_inventory, root, "path", names(actual))
  covered <- character()
  intervals <- matrix(0, nrow = length(grids), ncol = 2L)
  for (i in seq_along(grids)) {
    marker <- uni2_io_receipt_json(grids[[i]])
    check_identity(marker, "grid receipt")
    if (!identical(marker$cache_contract_sha256, contract_hash) ||
        !identical(marker$observation_type, observation_type)) stop("Binary UNI2 grid receipt contract/type mismatch")
    expected <- shard_paths[dirname(shard_paths) == dirname(grids[[i]])]
    if (!length(expected) || uni2_io_receipt_count(marker$shards, "shard count", 1) != length(expected))
      stop("Binary UNI2 grid receipt shard count mismatch")
    found <- check_inventory(marker$shard_files, dirname(grids[[i]]), "name", expected)
    if (length(intersect(found, covered))) stop("Binary UNI2 logical shard covered by multiple grid receipts")
    covered <- c(covered, found)
    for (j in seq_along(found)) {
      info <- shards[[match(found[[j]], shard_paths)]]
      record <- marker$shard_files[[j]]
      if (!identical(record$storage, "binary")) stop("Binary UNI2 grid logical shard storage mismatch")
      check_inventory(record$payload_files, dirname(found[[j]]), "name", c(info$rows_path, info$features_path))
    }
    intervals[i, ] <- c(uni2_io_receipt_count(marker$index_start, "index_start"),
                       uni2_io_receipt_count(marker$index_end, "index_end", 1))
    rows <- sum(vapply(shards[match(found, shard_paths)], function(info) info$shape[1L], numeric(1)))
    if (uni2_io_receipt_count(marker$rows_written, "grid rows", 1) != rows ||
        intervals[i, 2L] - intervals[i, 1L] != rows) stop("Binary UNI2 grid row coverage mismatch")
  }
  total <- sum(vapply(shards, function(info) info$shape[1L], numeric(1)))
  intervals <- intervals[order(intervals[, 1L]), , drop = FALSE]
  if (!setequal(covered, shard_paths) || intervals[1L, 1L] != 0 || tail(intervals[, 2L], 1L) != total ||
      (nrow(intervals) > 1L && any(intervals[-1L, 1L] != intervals[-nrow(intervals), 2L])))
    stop("Binary UNI2 extraction receipts have missing, overlapping or unlisted row coverage")
  for (key in c("expected_observations", "rows_written"))
    if (uni2_io_receipt_count(completion[[key]], key, 1) != total) stop("Binary UNI2 completion row coverage mismatch")
  if (uni2_io_receipt_count(completion$completed_grids, "completed_grids", 1) != length(grids) ||
      (!is.null(completion$expected_cells) && uni2_io_receipt_count(completion$expected_cells, "expected_cells", 1) != total) ||
      (!is.null(completion$missing_cells) && uni2_io_receipt_count(completion$missing_cells, "missing_cells") != 0))
    stop("Binary UNI2 completion grid/row coverage mismatch")
  list(receipts = receipt_sources, status = "receipt_consistency_verified_encoder_inputs_not_independently_verified",
       storage_receipts = "verified_exact_root_grid_payload_inventory_hashes_sizes_modes_and_row_coverage",
       cache_contract_sha256 = contract_hash,
       cache_contract_digest = "declared_sha256_agrees_across_receipts_not_recomputed_from_python_canonical_json")
}

load_uni2_binary_mode <- function(directory, mode_name, top_features, block_bytes = 8 * 1024^2) {
  inventory <- discover_uni2_embedding_input(directory)
  if (inventory$storage != "binary") stop("Expected binary UNI2 input")
  shards <- lapply(inventory$files, read_uni2_binary_manifest, expected_mode = mode_name)
  feature_names <- shards[[1L]]$feature_names
  if (!all(vapply(shards, function(shard) identical(shard$feature_names, feature_names), logical(1))))
    stop("Binary UNI2 feature names/order differ between shards")
  ids <- uni2_io_names(unlist(lapply(shards, `[[`, "ids"), use.names = FALSE), "cross-shard cell_id values")
  extraction <- validate_uni2_binary_receipts(directory, shards, mode_name)
  # Bounded first pass retains the existing population-variance ranking policy.
  sums <- sums_sq <- numeric(length(feature_names))
  select_features <- length(feature_names) > top_features
  if (select_features) {
    for (info in shards) {
      connection <- file(info$features_path, "rb")
      tryCatch({
        step <- max(1L, as.integer(floor(block_bytes / (8 * info$shape[2L]))))
        for (start in seq.int(1L, info$shape[1L], by = step)) {
          count <- min(step, info$shape[1L] - start + 1L)
          values <- uni2_io_read_block(connection, count, info$shape[2L], info$bytes)
          sums <- sums + colSums(values)
          sums_sq <- sums_sq + colSums(values * values)
        }
        if (length(readBin(connection, "raw", n = 1L))) stop("Binary UNI2 has trailing feature bytes")
      }, finally = close(connection))
    }
    variance <- sums_sq / length(ids) - (sums / length(ids))^2
    variance[!is.finite(variance)] <- -Inf
    selected <- order(-variance, seq_along(variance))[seq_len(min(top_features, length(variance)))]
  } else selected <- seq_along(feature_names)
  result <- matrix(NA_real_, nrow = length(ids), ncol = length(selected), dimnames = list(ids, feature_names[selected]))
  offset <- 0L
  for (info in shards) {
    connection <- file(info$features_path, "rb")
    tryCatch({
      step <- max(1L, as.integer(floor(block_bytes / (8 * info$shape[2L]))))
      for (start in seq.int(1L, info$shape[1L], by = step)) {
        count <- min(step, info$shape[1L] - start + 1L)
        values <- uni2_io_read_block(connection, count, info$shape[2L], info$bytes)
        index <- offset + seq.int(start, length.out = count)
        result[index, ] <- values[, selected, drop = FALSE]
      }
      if (length(readBin(connection, "raw", n = 1L))) stop("Binary UNI2 has trailing feature bytes")
    }, finally = close(connection))
    uni2_io_unchanged(info)
    offset <- offset + info$shape[1L]
  }
  rows <- data.table::rbindlist(lapply(shards, `[[`, "rows"), fill = FALSE, use.names = TRUE)
  if (!identical(rows$cell_id, ids)) stop("Binary UNI2 metadata order changed during assembly")
  receipts <- lapply(shards, function(info) list(
    manifest_path = info$path, manifest_sha256 = info$manifest_sha256,
    manifest = info$manifest, mode_binding = if (is.null(info$manifest$embedding_mode)) "caller_mode_unverified_in_legacy_conversion" else "declared_mode_matches_caller"))
  for (receipt in extraction$receipts)
    if (!identical(uni2_io_sha256(receipt$path), receipt$sha256)) stop("Binary UNI2 extraction receipt changed while loading")
  list(matrix = result, n_cells = nrow(result), n_features = ncol(result), rows = rows,
       provenance = list(storage = "binary", format = "cellphenotyper_uni2_binary", schema_version = "1.0.0",
         observation_order = "sorted_shard_paths_then_exact_stored_row_order", source_feature_names = feature_names,
         selected_feature_names = feature_names[selected], preselection = "population_variance_existing_policy",
         shards = receipts, encoder_receipts = extraction$receipts,
         extraction_receipt_validation = extraction[setdiff(names(extraction), c("receipts", "status"))],
         encoder_binding_status = extraction$status))
}

# CSV remains a supported storage format, not permission to repair a damaged
# cell population. Validate all columns before variance preselection, including
# features that will not reach KODAMA. These hashes bind this read, not an encoder.
uni2_csv_read <- function(path, ...) {
  withCallingHandlers({
    if (grepl("\\.gz$", path, ignore.case = TRUE)) {
      command <- sprintf("gzip -dc -- %s", shQuote(normalizePath(path, mustWork = TRUE)))
      data.table::fread(cmd = command, check.names = FALSE, na.strings = NULL,
        strip.white = FALSE, encoding = "UTF-8", showProgress = FALSE, ...)
    } else data.table::fread(path, check.names = FALSE, na.strings = NULL,
      strip.white = FALSE, encoding = "UTF-8", showProgress = FALSE, ...)
  }, warning = function(w) stop("UNI2 CSV parsing failed: ", conditionMessage(w)))
}

uni2_csv_names <- function(values, field) {
  if (!is.character(values) || !length(values) || anyNA(values) ||
      any(!nzchar(values)) || any(trimws(values) != values) ||
      any(grepl("[[:cntrl:]]", values)) || anyDuplicated(values))
    stop("Invalid or duplicate UNI2 CSV ", field, "; no rows are silently discarded")
  values
}

load_uni2_csv_mode <- function(directory, mode_name, top_features) {
  uni2_io_require()
  inventory <- discover_uni2_embedding_input(directory)
  if (inventory$storage != "legacy_csv") stop("Expected CSV UNI2 input")
  files <- inventory$files
  receipts <- lapply(files, function(path) {
    receipt <- list(path = normalizePath(path), sha256 = uni2_io_sha256(path), size_bytes = file.info(path)$size)
    # fread(cmd=...) does not reliably report a gzip CRC/trailer failure.
    if (grepl("\\.gz$", path, ignore.case = TRUE)) {
      result <- suppressWarnings(system2("gzip", c("-t", "--", shQuote(normalizePath(path))),
        stdout = FALSE, stderr = TRUE))
      if (!is.null(attr(result, "status")) && attr(result, "status") != 0L)
        stop("UNI2 CSV gzip integrity check failed: ", path)
    }
    receipt
  })
  columns <- uni2_csv_names(names(uni2_csv_read(files[[1L]], nrows = 0L)), "column names")
  if (!("cell_id" %in% columns)) stop("UNI2 CSV is missing cell_id")
  features <- grep("^feat", columns, value = TRUE)
  if (!length(features)) stop("UNI2 CSV has no feature columns starting with feat")
  metadata_columns <- setdiff(columns, features)
  sums <- sums_sq <- numeric(length(features))
  metadata <- vector("list", length(files))
  selected <- seq_along(features)
  single_pass <- length(features) <= top_features
  retained <- if (single_pass) vector("list", length(files)) else NULL
  for (index in seq_along(files)) {
    table <- uni2_csv_read(files[[index]], colClasses = list(character = "cell_id"))
    if (!identical(names(table), columns)) stop("UNI2 CSV column/feature names/order differ between shards")
    if (!nrow(table)) stop("UNI2 CSV contains an empty shard")
    uni2_csv_names(table$cell_id, paste(mode_name, "cell_id values"))
    if (!all(vapply(table[, ..features], is.numeric, logical(1))))
      stop("UNI2 CSV feature vectors must be numeric and finite; no silent imputation")
    values <- as.matrix(table[, ..features])
    storage.mode(values) <- "double"
    if (any(!is.finite(values))) stop("UNI2 CSV feature vectors must be finite; no silent imputation")
    metadata[[index]] <- table[, ..metadata_columns]
    receipts[[index]]$row_count <- nrow(table)
    if (single_pass) retained[[index]] <- values else {
      sums <- sums + colSums(values)
      sums_sq <- sums_sq + colSums(values * values)
    }
    rm(table, values)
  }
  rows <- data.table::rbindlist(metadata, fill = FALSE, use.names = TRUE)
  ids <- uni2_csv_names(rows$cell_id, paste(mode_name, "cross-shard cell_id values"))
  if (!single_pass) {
    variance <- sums_sq / length(ids) - (sums / length(ids))^2
    variance[!is.finite(variance)] <- -Inf
    selected <- order(-variance, seq_along(variance))[seq_len(min(top_features, length(variance)))]
  }
  selected_features <- features[selected]
  result <- matrix(NA_real_, length(ids), length(selected), dimnames = list(ids, selected_features))
  offset <- 0L
  for (index in seq_along(files)) {
    count <- receipts[[index]]$row_count
    if (single_pass) values <- retained[[index]] else {
      table <- uni2_csv_read(files[[index]], select = c("cell_id", selected_features),
        colClasses = list(character = "cell_id"))
      if (!identical(table$cell_id, metadata[[index]]$cell_id)) stop("UNI2 CSV source row order changed while loading")
      values <- as.matrix(table[, ..selected_features])
      storage.mode(values) <- "double"
      if (any(!is.finite(values))) stop("UNI2 CSV selected features must be finite")
    }
    result[offset + seq_len(count), ] <- values
    offset <- offset + count
    if (!identical(uni2_io_sha256(files[[index]]), receipts[[index]]$sha256))
      stop("UNI2 CSV source changed while loading")
  }
  if (!identical(discover_uni2_embedding_input(directory)$files, files))
    stop("UNI2 CSV shard inventory changed while loading")
  list(matrix = result, rows = rows, n_cells = nrow(result), n_features = ncol(result),
    provenance = list(storage = "legacy_csv", format = "cellphenotyper_uni2_csv", schema_version = "1.0.0",
      observation_order = "sorted_shard_paths_then_exact_stored_row_order",
      source_feature_names = features, selected_feature_names = selected_features,
      preselection = "population_variance_existing_policy", shards = receipts,
      encoder_binding_status = "unverified_csv_read_hashes_do_not_establish_encoder_provenance"))
}
