args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Rcode_Clustering.R <kodama_dir> <out_csv> [--dim N] [--k N] [--resolution X]")
}

kodama_dir <- args[1]
out_csv <- args[2]

target_dim <- 20L
snn_k <- 100L
resolution <- 0.2

if (length(args) > 2) {
  i <- 3L
  while (i <= length(args)) {
    flag <- args[i]
    if (flag == "--dim" && i + 1L <= length(args)) {
      target_dim <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--k" && i + 1L <= length(args)) {
      snn_k <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--resolution" && i + 1L <= length(args)) {
      resolution <- as.numeric(args[i + 1L])
      i <- i + 2L
      next
    }
    i <- i + 1L
  }
}

files <- list.files(kodama_dir, pattern = "^kodama_full_[0-9]+\\.RData$", full.names = TRUE)
if (length(files) == 0) {
  stop(paste("No kodama_full_*.RData files found in:", kodama_dir))
}

dims <- as.integer(sub("^.*kodama_full_([0-9]+)\\.RData$", "\\1", files))
if (any(is.na(dims))) {
  stop("Unable to parse embedding dimensions from kodama_full_*.RData filenames.")
}

select_file <- function(target, candidate_dims, candidate_files) {
  if (target %in% candidate_dims) {
    return(candidate_files[which(candidate_dims == target)[1]])
  }
  lower <- which(candidate_dims < target)
  if (length(lower) > 0) {
    return(candidate_files[lower[which.max(candidate_dims[lower])]])
  }
  candidate_files[which.min(candidate_dims)]
}

chosen <- select_file(target_dim, dims, files)
load(chosen)

if (!exists("vis")) {
  stop(paste("Variable 'vis' not found in:", chosen))
}

vis <- as.matrix(vis)
if (is.null(rownames(vis))) {
  rownames(vis) <- sprintf("cell_%05d", seq_len(nrow(vis)))
}

if (nrow(vis) < 3) {
  stop(paste("Need at least 3 cells for clustering. Found:", nrow(vis)))
}

membership <- NULL
if (requireNamespace("bluster", quietly = TRUE) && requireNamespace("igraph", quietly = TRUE)) {
  k_use <- max(2L, min(as.integer(snn_k), nrow(vis) - 1L))
  g <- bluster::makeSNNGraph(vis, k = k_use)
  clu <- igraph::cluster_louvain(g, resolution = resolution)
  membership <- as.integer(clu$membership)
} else {
  kmeans_k <- max(2L, min(8L, floor(sqrt(nrow(vis)))))
  km <- stats::kmeans(scale(vis), centers = kmeans_k, nstart = 20)
  membership <- as.integer(km$cluster)
}

da <- data.frame(label = rownames(vis), cluster = membership)
write.csv(da, out_csv, row.names = FALSE, quote = FALSE)

cat(paste0("[INFO] RData source: ", chosen, "\n"))
cat(paste0("[INFO] Cells clustered: ", nrow(da), "\n"))
cat(paste0("[INFO] Wrote: ", out_csv, "\n"))





