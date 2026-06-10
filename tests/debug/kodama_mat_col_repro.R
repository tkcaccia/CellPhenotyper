#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

usage <- function() {
  cat(
    "Usage:\n",
    "  Rscript tests/debug/kodama_mat_col_repro.R --mode real_pca --pca-rdata <pca_full_20.RData> --outdir <dir> [options]\n",
    "  Rscript tests/debug/kodama_mat_col_repro.R --mode synthetic --outdir <dir> [options]\n\n",
    "Options:\n",
    "  --sample-cells N       Number of cells for KODAMA.matrix input [20000]\n",
    "  --dims N               PCA/features dimensions to use [20]\n",
    "  --landmarks N          KODAMA landmarks [1000]\n",
    "  --ncomp N              KODAMA internal PLS components [KODAMA default]\n",
    "  --n-cores N            KODAMA cores [4]\n",
    "  --seed N               Random seed [543210]\n",
    "  --synthetic-kind K     normal|duplicated_spatial|low_rank [normal]\n",
    "  --no-visualization     Skip KODAMA.visualization()\n",
    "  --help                 Show this help\n",
    sep = ""
  )
}

get_arg <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0L || idx[1L] == length(args)) return(default)
  args[idx[1L] + 1L]
}

has_flag <- function(flag) any(args == flag)

if (has_flag("--help") || length(args) == 0L) {
  usage()
  quit(status = 0L)
}

mode <- get_arg("--mode", "real_pca")
pca_rdata <- get_arg("--pca-rdata", "")
outdir <- get_arg("--outdir", "kodama_mat_col_repro")
sample_cells <- suppressWarnings(as.integer(get_arg("--sample-cells", "20000")))
dims <- suppressWarnings(as.integer(get_arg("--dims", "20")))
landmarks <- suppressWarnings(as.integer(get_arg("--landmarks", "1000")))
ncomp_arg <- get_arg("--ncomp", "")
ncomp <- if (nzchar(ncomp_arg)) suppressWarnings(as.integer(ncomp_arg)) else NA_integer_
n_cores <- suppressWarnings(as.integer(get_arg("--n-cores", "4")))
seed <- suppressWarnings(as.integer(get_arg("--seed", "543210")))
synthetic_kind <- get_arg("--synthetic-kind", "normal")
run_visualization <- !has_flag("--no-visualization")

if (!(mode %in% c("real_pca", "synthetic"))) stop("--mode must be real_pca or synthetic")
if (!is.finite(sample_cells) || sample_cells < 10L) stop("--sample-cells must be >= 10")
if (!is.finite(dims) || dims < 2L) stop("--dims must be >= 2")
if (!is.finite(landmarks) || landmarks < 1L) stop("--landmarks must be >= 1")
if (nzchar(ncomp_arg) && (!is.finite(ncomp) || ncomp < 1L)) stop("--ncomp must be >= 1")
if (!is.finite(n_cores) || n_cores < 1L) n_cores <- 1L
if (!is.finite(seed)) seed <- 543210L

if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(outdir, sprintf("kodama_mat_col_repro_%s_n%d_d%d_lm%d.log", mode, sample_cells, dims, landmarks))

log_con <- file(log_file, open = "wt")
sink(log_con, type = "output", split = TRUE)
sink(log_con, type = "message")
cleanup_done <- FALSE
cleanup <- function() {
  if (isTRUE(cleanup_done)) return(invisible(NULL))
  cleanup_done <<- TRUE
  try(sink(type = "message"), silent = TRUE)
  while (sink.number(type = "output") > 0L) {
    try(sink(type = "output"), silent = TRUE)
  }
  try(close(log_con), silent = TRUE)
  invisible(NULL)
}
on.exit(cleanup(), add = TRUE)

options(warn = 1)
cat(sprintf("[INFO] start=%s\n", format(Sys.time(), usetz = TRUE)))
cat(sprintf("[INFO] mode=%s sample_cells=%d dims=%d landmarks=%d n_cores=%d seed=%d visualization=%s\n",
            mode, sample_cells, dims, landmarks, n_cores, seed, run_visualization))
cat(sprintf("[INFO] output_dir=%s\n", normalizePath(outdir, mustWork = FALSE)))

container_r_lib <- "/opt/micromamba/envs/stardist/lib/R/library"
if (dir.exists(container_r_lib)) {
  .libPaths(container_r_lib)
}
cat(sprintf("[INFO] R library paths=%s\n", paste(.libPaths(), collapse = " | ")))

required <- c("KODAMA", "KODAMAextra", "umap")
missing <- required[!vapply(required, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing) > 0L) stop(sprintf("Missing packages: %s", paste(missing, collapse = ", ")))

suppressPackageStartupMessages({
  library(KODAMA)
  library(KODAMAextra)
  library(umap)
})

cat(sprintf("[INFO] KODAMA package version=%s\n", as.character(utils::packageVersion("KODAMA"))))
cat(sprintf("[INFO] KODAMAextra package version=%s\n", as.character(utils::packageVersion("KODAMAextra"))))
cat(sprintf("[INFO] R version=%s\n", R.version.string))

validate_input <- function(feature_mat, spatial_mat) {
  if (!is.matrix(feature_mat)) stop("feature input is not a matrix")
  if (!is.matrix(spatial_mat)) stop("spatial input is not a matrix")
  if (nrow(feature_mat) != nrow(spatial_mat)) stop("feature and spatial row counts differ")
  if (ncol(feature_mat) < 2L) stop("feature matrix must have >= 2 columns")
  if (ncol(spatial_mat) < 2L) stop("spatial matrix must have >= 2 columns")
  if (anyNA(feature_mat) || any(!is.finite(feature_mat))) stop("feature matrix contains NA/Inf")
  if (anyNA(spatial_mat) || any(!is.finite(spatial_mat))) stop("spatial matrix contains NA/Inf")
  if (is.null(rownames(feature_mat))) rownames(feature_mat) <- sprintf("cell_%06d", seq_len(nrow(feature_mat)))
  if (is.null(rownames(spatial_mat))) rownames(spatial_mat) <- rownames(feature_mat)
  common <- intersect(rownames(feature_mat), rownames(spatial_mat))
  if (length(common) != nrow(feature_mat)) stop("feature/spatial row names are not aligned")
  invisible(TRUE)
}

make_real_pca_input <- function() {
  if (!nzchar(pca_rdata) || !file.exists(pca_rdata)) stop("--pca-rdata is required for --mode real_pca")
  cat(sprintf("[INFO] loading pca_rdata=%s size=%.2f MiB\n", pca_rdata, file.info(pca_rdata)$size / (1024^2)))
  load(pca_rdata)
  if (!exists("pca")) stop("pca object not found in pca RData")
  if (!exists("xy")) stop("xy object not found in pca RData")
  pca <- as.matrix(pca)
  xy <- as.matrix(xy)
  storage.mode(pca) <- "double"
  storage.mode(xy) <- "double"
  if (is.null(rownames(pca)) || is.null(rownames(xy))) stop("pca and xy need row names")
  common <- intersect(rownames(pca), rownames(xy))
  pca <- pca[common, , drop = FALSE]
  xy <- xy[common, , drop = FALSE]
  keep <- stats::complete.cases(pca) & stats::complete.cases(xy)
  keep <- keep & apply(pca, 1L, function(z) all(is.finite(z)))
  keep <- keep & apply(xy, 1L, function(z) all(is.finite(z)))
  pca <- pca[keep, , drop = FALSE]
  xy <- xy[keep, , drop = FALSE]
  used_dims <- seq_len(min(dims, ncol(pca)))
  set.seed(seed)
  idx <- sort(sample.int(nrow(pca), min(sample_cells, nrow(pca))))
  feature_mat <- pca[idx, used_dims, drop = FALSE]
  spatial_mat <- xy[idx, seq_len(min(2L, ncol(xy))), drop = FALSE]
  colnames(feature_mat) <- sprintf("PC%02d", used_dims)
  colnames(spatial_mat) <- c("x", "y")[seq_len(ncol(spatial_mat))]
  list(feature = feature_mat, spatial = spatial_mat)
}

make_synthetic_input <- function() {
  set.seed(seed)
  n <- sample_cells
  d <- dims
  feature_mat <- matrix(stats::rnorm(n * d), nrow = n, ncol = d)
  spatial_mat <- cbind(x = stats::rnorm(n), y = stats::rnorm(n))
  if (synthetic_kind == "duplicated_spatial") {
    spatial_mat <- round(spatial_mat, 1L)
  } else if (synthetic_kind == "low_rank") {
    latent <- matrix(stats::rnorm(n * 3L), nrow = n, ncol = 3L)
    weights <- matrix(stats::rnorm(3L * d), nrow = 3L, ncol = d)
    feature_mat <- latent %*% weights + matrix(stats::rnorm(n * d, sd = 0.001), nrow = n)
  } else if (synthetic_kind != "normal") {
    stop("--synthetic-kind must be normal, duplicated_spatial, or low_rank")
  }
  rownames(feature_mat) <- sprintf("cell_%06d", seq_len(n))
  rownames(spatial_mat) <- rownames(feature_mat)
  colnames(feature_mat) <- sprintf("feat_%02d", seq_len(d))
  list(feature = feature_mat, spatial = spatial_mat)
}

input <- if (mode == "real_pca") make_real_pca_input() else make_synthetic_input()
feature_mat <- input$feature
spatial_mat <- input$spatial
validate_input(feature_mat, spatial_mat)

cat(sprintf("[INFO] feature_dim=%d x %d spatial_dim=%d x %d\n", nrow(feature_mat), ncol(feature_mat), nrow(spatial_mat), ncol(spatial_mat)))
cat(sprintf("[INFO] feature_summary min=%.6f median=%.6f max=%.6f\n", min(feature_mat), stats::median(feature_mat), max(feature_mat)))
cat(sprintf("[INFO] spatial_summary x=[%.3f, %.3f] y=[%.3f, %.3f]\n", min(spatial_mat[, 1]), max(spatial_mat[, 1]), min(spatial_mat[, 2]), max(spatial_mat[, 2])))

saveRDS(
  list(feature = feature_mat, spatial = spatial_mat, mode = mode, seed = seed, sample_cells = sample_cells, dims = dims, landmarks = landmarks),
  file.path(outdir, sprintf("kodama_mat_col_input_%s_n%d_d%d_lm%d.rds", mode, nrow(feature_mat), ncol(feature_mat), min(landmarks, nrow(feature_mat))))
)

cat("[INFO] calling KODAMA.matrix()\n")
flush.console()
t0 <- proc.time()[["elapsed"]]
kodama_args <- list(
  data = feature_mat,
  spatial = spatial_mat,
  landmarks = min(landmarks, nrow(feature_mat)),
  n.cores = as.integer(n_cores),
  seed = seed,
  ancestry = FALSE
)
if (nzchar(ncomp_arg)) {
  kodama_args$ncomp <- as.integer(ncomp)
}
result <- tryCatch(
  do.call(KODAMA.matrix, kodama_args),
  error = function(e) {
    cat(sprintf("[ERROR] KODAMA.matrix failed: %s\n", conditionMessage(e)))
    structure(list(error = conditionMessage(e)), class = "kodama_repro_error")
  }
)
cat(sprintf("[INFO] KODAMA.matrix elapsed_sec=%.2f\n", proc.time()[["elapsed"]] - t0))

if (!inherits(result, "kodama_repro_error")) {
  saveRDS(result, file.path(outdir, sprintf("kodama_matrix_result_%s_n%d.rds", mode, nrow(feature_mat))))
  cat(sprintf("[INFO] saved KODAMA.matrix result class=%s\n", paste(class(result), collapse = ",")))
  if (run_visualization) {
    cat("[INFO] calling KODAMA.visualization()\n")
    flush.console()
    config <- umap::umap.defaults
    config$n_neighbors <- min(30L, nrow(feature_mat) - 1L)
    config$n_threads <- as.integer(n_cores)
    t1 <- proc.time()[["elapsed"]]
    vis <- tryCatch(
      KODAMA.visualization(result, config = config),
      error = function(e) {
        cat(sprintf("[ERROR] KODAMA.visualization failed: %s\n", conditionMessage(e)))
        NULL
      }
    )
    cat(sprintf("[INFO] KODAMA.visualization elapsed_sec=%.2f\n", proc.time()[["elapsed"]] - t1))
    if (!is.null(vis)) {
      saveRDS(vis, file.path(outdir, sprintf("kodama_visualization_%s_n%d.rds", mode, nrow(feature_mat))))
      png(file.path(outdir, sprintf("kodama_visualization_%s_n%d.png", mode, nrow(feature_mat))), width = 1600, height = 1600, res = 200)
      plot(vis[, 1], vis[, 2], pch = 16, cex = 0.08, col = grDevices::adjustcolor("black", alpha.f = 0.35), xlab = "KODAMA 1", ylab = "KODAMA 2")
      dev.off()
      cat(sprintf("[INFO] saved visualization rows=%d cols=%d\n", nrow(vis), ncol(vis)))
    }
  }
}

cat(sprintf("[INFO] end=%s\n", format(Sys.time(), usetz = TRUE)))
cat("[INFO] sessionInfo follows\n")
print(utils::sessionInfo())

# Close sinks before scanning the log for the target message.
cleanup()
on.exit(NULL, add = FALSE)
log_lines <- readLines(log_file, warn = FALSE)
mat_col_hits <- grep("Mat::col\\(\\): index out of bounds", log_lines, value = TRUE)
summary_path <- file.path(outdir, sprintf("kodama_mat_col_repro_%s_n%d_summary.txt", mode, sample_cells))
writeLines(
  c(
    sprintf("mode=%s", mode),
    sprintf("sample_cells=%d", sample_cells),
    sprintf("dims=%d", dims),
    sprintf("landmarks=%d", landmarks),
    sprintf("ncomp=%s", if (nzchar(ncomp_arg)) as.character(ncomp) else "KODAMA_default"),
    sprintf("n_cores=%d", n_cores),
    sprintf("seed=%d", seed),
    sprintf("mat_col_index_out_of_bounds_count=%d", length(mat_col_hits)),
    sprintf("log_file=%s", normalizePath(log_file, mustWork = FALSE))
  ),
  summary_path
)
cat(sprintf("[SUMMARY] Mat::col index-out-of-bounds count: %d\n", length(mat_col_hits)))
cat(sprintf("[SUMMARY] summary_file=%s\n", summary_path))
