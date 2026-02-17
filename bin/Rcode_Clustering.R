args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript Rcode_Clustering.R <kodama_dir> <out_csv> [--dim N] [--k N] [--resolution X]")
}

kodama_dir <- args[1]
out_csv <- args[2]

target_dim <- 20L
snn_k <- 10L
resolution_mode <- "fixed"
resolution <- 0.2
resolution_grid <- c(0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)

target_n <- 0L
annotation_csv <- NULL
annotation_col <- "polygon_label"
annotation_ari_margin <- 0.01
annotation_only <- FALSE

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
      res_arg <- tolower(args[i + 1L])
      if (res_arg == "auto") {
        resolution_mode <- "auto"
      } else {
        resolution_mode <- "fixed"
        resolution <- as.numeric(args[i + 1L])
      }
      i <- i + 2L
      next
    }
    if (flag == "--resolution-grid" && i + 1L <= length(args)) {
      grid_vals <- strsplit(args[i + 1L], ",", fixed = TRUE)[[1]]
      parsed <- suppressWarnings(as.numeric(trimws(grid_vals)))
      parsed <- parsed[is.finite(parsed) & parsed > 0]
      if (length(parsed) > 0) {
        resolution_grid <- sort(unique(parsed))
      }
      i <- i + 2L
      next
    }
    if (flag == "--target-n" && i + 1L <= length(args)) {
      target_n <- as.integer(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--annotation-csv" && i + 1L <= length(args)) {
      annotation_csv <- args[i + 1L]
      i <- i + 2L
      next
    }
    if (flag == "--annotation-col" && i + 1L <= length(args)) {
      annotation_col <- args[i + 1L]
      i <- i + 2L
      next
    }
    if (flag == "--annotation-ari-margin" && i + 1L <= length(args)) {
      annotation_ari_margin <- as.numeric(args[i + 1L])
      i <- i + 2L
      next
    }
    if (flag == "--annotation-only") {
      annotation_only <- TRUE
      i <- i + 1L
      next
    }
    i <- i + 1L
  }
}

if (resolution_mode == "fixed" && (!is.finite(resolution) || resolution <= 0)) {
  stop(paste("Invalid --resolution value:", resolution))
}
if (!is.finite(annotation_ari_margin) || annotation_ari_margin < 0) {
  annotation_ari_margin <- 0.01
}
if (!is.finite(target_n) || target_n < 0) {
  target_n <- 0L
}

calc_ari <- function(truth, pred) {
  keep <- !is.na(truth) & !is.na(pred) & nzchar(as.character(truth)) & nzchar(as.character(pred))
  truth <- as.character(truth[keep])
  pred <- as.character(pred[keep])
  if (length(truth) < 2L || length(unique(truth)) < 2L || length(unique(pred)) < 2L) {
    return(NA_real_)
  }

  tab <- table(truth, pred)
  n <- sum(tab)
  if (n < 2L) {
    return(NA_real_)
  }
  comb2 <- function(x) x * (x - 1) / 2
  sum_nij <- sum(comb2(as.numeric(tab)))
  a_sum <- rowSums(tab)
  b_sum <- colSums(tab)
  sum_ai <- sum(comb2(as.numeric(a_sum)))
  sum_bj <- sum(comb2(as.numeric(b_sum)))
  total_pairs <- comb2(n)
  if (total_pairs == 0) {
    return(NA_real_)
  }
  expected <- (sum_ai * sum_bj) / total_pairs
  max_index <- 0.5 * (sum_ai + sum_bj)
  denom <- max_index - expected
  if (denom == 0) {
    return(NA_real_)
  }
  (sum_nij - expected) / denom
}

build_annotation_df <- function(path, colname) {
  if (is.null(path) || !nzchar(path) || !file.exists(path)) {
    return(NULL)
  }
  ann <- tryCatch(
    read.csv(path, stringsAsFactors = FALSE),
    error = function(e) NULL
  )
  if (is.null(ann)) {
    cat(sprintf("[WARN] Could not read annotation CSV: %s\n", path))
    return(NULL)
  }
  if (!("label" %in% colnames(ann)) || !(colname %in% colnames(ann))) {
    cat(sprintf("[WARN] Annotation CSV missing required columns: label and %s\n", colname))
    return(NULL)
  }
  out <- ann[, c("label", colname)]
  colnames(out) <- c("label", "truth")
  out$label <- trimws(as.character(out$label))
  out$truth <- trimws(as.character(out$truth))
  out <- out[!is.na(out$label) & nzchar(out$label) & !is.na(out$truth) & nzchar(out$truth), , drop = FALSE]
  if (nrow(out) == 0L) {
    cat("[WARN] Annotation CSV has no usable rows after filtering empty labels.\n")
    return(NULL)
  }
  out <- out[!duplicated(out$label), , drop = FALSE]
  if (length(unique(out$truth)) < 2L) {
    cat("[WARN] Annotation column has <2 classes; annotation-guided selection disabled.\n")
    return(NULL)
  }
  out
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
vis_all <- vis

annotation_df <- build_annotation_df(annotation_csv, annotation_col)
if (!is.null(annotation_df)) {
  cat(sprintf("[INFO] Annotation guidance active: %s (rows=%d, classes=%d)\n",
              annotation_col, nrow(annotation_df), length(unique(annotation_df$truth))))
  if (isTRUE(annotation_only)) {
    keep_labels <- intersect(rownames(vis), annotation_df$label)
    if (length(keep_labels) >= 3L) {
      truth_keep <- annotation_df$truth[match(keep_labels, annotation_df$label)]
      if (length(unique(truth_keep)) >= 2L) {
        vis <- vis[keep_labels, , drop = FALSE]
        annotation_df <- annotation_df[annotation_df$label %in% keep_labels, , drop = FALSE]
        cat(sprintf("[INFO] Annotation-only clustering enabled: retained %d cells with labels.\n", nrow(vis)))
      } else {
        cat("[WARN] Annotation-only requested but retained cells have <2 classes; using all cells.\n")
      }
    } else {
      cat("[WARN] Annotation-only requested but <3 labeled cells overlap embedding; using all cells.\n")
    }
  }
}

if (nrow(vis) < 3) {
  stop(paste("Need at least 3 cells for clustering. Found:", nrow(vis)))
}

membership <- NULL
if (requireNamespace("bluster", quietly = TRUE) && requireNamespace("igraph", quietly = TRUE)) {
  k_use <- max(2L, min(as.integer(snn_k), nrow(vis) - 1L))
  g <- bluster::makeSNNGraph(vis, k = k_use)

  run_louvain <- function(res_value) {
    set.seed(1L)
    clu <- igraph::cluster_louvain(g, resolution = res_value)
    memb <- as.integer(clu$membership)
    list(
      resolution = res_value,
      membership = memb,
      n_clusters = length(unique(memb)),
      modularity = igraph::modularity(g, membership = memb),
      silhouette = NA_real_,
      annotation_ari = NA_real_,
      annotation_n = 0L
    )
  }

  if (resolution_mode == "auto") {
    evals <- lapply(resolution_grid, run_louvain)
    max_clusters <- max(2L, min(20L, floor(nrow(vis) / 3L)))
    valid_idx <- which(vapply(evals, function(e) {
      e$n_clusters >= 2L && e$n_clusters <= max_clusters
    }, logical(1)))
    if (length(valid_idx) == 0L) {
      valid_idx <- seq_along(evals)
    }
    if (target_n > 0L) {
      exact_target <- valid_idx[vapply(evals[valid_idx], function(e) e$n_clusters == target_n, logical(1))]
      if (length(exact_target) > 0L) {
        valid_idx <- exact_target
        cat(sprintf("[INFO] Auto-resolution constrained to target_n=%d (candidate_count=%d)\n", target_n, length(valid_idx)))
      }
    }

    apply_target_preference <- function(indices) {
      if (length(indices) <= 1L || target_n <= 0L) {
        return(indices)
      }
      dist_to_target <- vapply(evals[indices], function(e) abs(e$n_clusters - target_n), integer(1))
      indices[which(dist_to_target == min(dist_to_target))]
    }

    silhouette_margin <- 0.02
    max_silhouette_cells <- 2000L
    use_silhouette <- requireNamespace("cluster", quietly = TRUE) && nrow(vis) >= 4L

    if (use_silhouette) {
      set.seed(1L)
      sample_idx <- seq_len(nrow(vis))
      if (nrow(vis) > max_silhouette_cells) {
        sample_idx <- sort(sample(sample_idx, max_silhouette_cells))
      }
      vis_eval <- scale(vis[sample_idx, , drop = FALSE])
      dist_eval <- stats::dist(vis_eval)

      for (ii in valid_idx) {
        memb_eval <- evals[[ii]]$membership[sample_idx]
        if (length(unique(memb_eval)) >= 2L) {
          sil_obj <- tryCatch(cluster::silhouette(memb_eval, dist_eval), error = function(e) NULL)
          evals[[ii]]$silhouette <- if (is.null(sil_obj)) NA_real_ else summary(sil_obj)$avg.width
        }
      }
    }

    if (!is.null(annotation_df)) {
      for (ii in valid_idx) {
        pred_df <- data.frame(
          label = rownames(vis),
          pred = as.character(evals[[ii]]$membership),
          stringsAsFactors = FALSE
        )
        merged <- merge(annotation_df, pred_df, by = "label")
        evals[[ii]]$annotation_n <- nrow(merged)
        if (nrow(merged) >= 4L) {
          evals[[ii]]$annotation_ari <- calc_ari(merged$truth, merged$pred)
        }
      }
    }

    ari_values <- vapply(evals[valid_idx], function(e) e$annotation_ari, numeric(1))
    if (any(is.finite(ari_values))) {
      best_ari <- max(ari_values, na.rm = TRUE)
      near_ari <- valid_idx[which(ari_values >= (best_ari - annotation_ari_margin))]
      near_ari <- apply_target_preference(near_ari)

      sil_near <- vapply(evals[near_ari], function(e) e$silhouette, numeric(1))
      if (any(is.finite(sil_near))) {
        best_sil_near <- max(sil_near, na.rm = TRUE)
        near_sil <- near_ari[which(sil_near >= (best_sil_near - silhouette_margin))]
      } else {
        near_sil <- near_ari
      }

      near_clusters <- vapply(evals[near_sil], function(e) e$n_clusters, integer(1))
      min_clusters <- min(near_clusters)
      simplest <- near_sil[which(near_clusters == min_clusters)]
      best_pos <- simplest[which.max(vapply(evals[simplest], function(e) e$modularity, numeric(1)))]
      auto_reason <- sprintf(
        "annotation_ari+silhouette+parsimony (ari_margin=%.3f, sil_margin=%.2f)",
        annotation_ari_margin, silhouette_margin
      )
    } else {
      sil_values <- vapply(evals[valid_idx], function(e) e$silhouette, numeric(1))
      if (any(is.finite(sil_values))) {
        best_sil <- max(sil_values, na.rm = TRUE)
        near_best <- valid_idx[which(sil_values >= (best_sil - silhouette_margin))]
        near_best <- apply_target_preference(near_best)
        near_clusters <- vapply(evals[near_best], function(e) e$n_clusters, integer(1))
        min_clusters <- min(near_clusters)
        simplest <- near_best[which(near_clusters == min_clusters)]
        best_pos <- simplest[which.max(vapply(evals[simplest], function(e) e$modularity, numeric(1)))]
        auto_reason <- sprintf("silhouette+parsimony (margin=%.2f)", silhouette_margin)
      } else {
        score_values <- vapply(evals[valid_idx], function(e) {
          target_penalty <- if (target_n > 0L) 0.10 * abs(e$n_clusters - target_n) else 0
          e$modularity - 0.08 * max(0, e$n_clusters - 2L) - target_penalty
        }, numeric(1))
        best_pos <- valid_idx[which.max(score_values)]
        auto_reason <- if (target_n > 0L) {
          "modularity-penalized fallback (+target_n penalty)"
        } else {
          "modularity-penalized fallback"
        }
      }
    }

    best <- evals[[best_pos]]
    membership <- best$membership

    cat(sprintf(
      "[INFO] Louvain auto-resolution selected: %.3f (clusters=%d, modularity=%.4f, silhouette=%s, annotation_ari=%s, reason=%s)\n",
      best$resolution, best$n_clusters, best$modularity,
      ifelse(is.finite(best$silhouette), sprintf("%.4f", best$silhouette), "NA"),
      ifelse(is.finite(best$annotation_ari), sprintf("%.4f", best$annotation_ari), "NA"),
      auto_reason
    ))
    cat("[INFO] Candidate resolutions:\n")
    for (e in evals) {
      cat(sprintf(
        "  - res=%.3f clusters=%d modularity=%.4f silhouette=%s annotation_ari=%s matched=%d\n",
        e$resolution, e$n_clusters, e$modularity,
        ifelse(is.finite(e$silhouette), sprintf("%.4f", e$silhouette), "NA"),
        ifelse(is.finite(e$annotation_ari), sprintf("%.4f", e$annotation_ari), "NA"),
        as.integer(e$annotation_n)
      ))
    }
  } else {
    best <- run_louvain(resolution)
    if (!is.null(annotation_df)) {
      pred_df <- data.frame(
        label = rownames(vis),
        pred = as.character(best$membership),
        stringsAsFactors = FALSE
      )
      merged <- merge(annotation_df, pred_df, by = "label")
      best$annotation_n <- nrow(merged)
      if (nrow(merged) >= 4L) {
        best$annotation_ari <- calc_ari(merged$truth, merged$pred)
      }
    }
    membership <- best$membership
    cat(sprintf(
      "[INFO] Louvain fixed resolution: %.3f (clusters=%d, modularity=%.4f, annotation_ari=%s)\n",
      best$resolution, best$n_clusters, best$modularity,
      ifelse(is.finite(best$annotation_ari), sprintf("%.4f", best$annotation_ari), "NA")
    ))
  }
} else {
  kmeans_k <- if (target_n > 1L) {
    max(2L, min(as.integer(target_n), nrow(vis) - 1L))
  } else {
    max(2L, min(8L, floor(sqrt(nrow(vis)))))
  }
  km <- stats::kmeans(scale(vis), centers = kmeans_k, nstart = 20)
  membership <- as.integer(km$cluster)
  cat(sprintf("[INFO] bluster/igraph unavailable; fallback kmeans centers=%d\n", kmeans_k))
}

output_labels <- rownames(vis)
output_membership <- as.integer(membership)
if (isTRUE(annotation_only) && nrow(vis_all) > nrow(vis)) {
  missing_labels <- setdiff(rownames(vis_all), rownames(vis))
  if (length(missing_labels) > 0L && length(unique(output_membership)) >= 1L) {
    cluster_ids <- sort(unique(output_membership))
    centroids <- do.call(rbind, lapply(cluster_ids, function(cid) {
      colMeans(vis[output_membership == cid, , drop = FALSE])
    }))
    rownames(centroids) <- as.character(cluster_ids)

    missing_pred <- integer(length(missing_labels))
    for (ii in seq_along(missing_labels)) {
      v <- vis_all[missing_labels[ii], , drop = FALSE]
      diff <- centroids - matrix(v, nrow = nrow(centroids), ncol = ncol(centroids), byrow = TRUE)
      dist2 <- rowSums(diff * diff)
      missing_pred[ii] <- as.integer(rownames(centroids)[which.min(dist2)])
    }

    full_membership <- setNames(rep(NA_integer_, nrow(vis_all)), rownames(vis_all))
    full_membership[rownames(vis)] <- output_membership
    full_membership[missing_labels] <- missing_pred
    output_labels <- names(full_membership)
    output_membership <- as.integer(full_membership)
    cat(sprintf("[INFO] Assigned %d unlabeled cells to nearest cluster centroids.\n", length(missing_labels)))
  }
}

da <- data.frame(label = output_labels, cluster = output_membership)
write.csv(da, out_csv, row.names = FALSE, quote = FALSE)

cat(paste0("[INFO] RData source: ", chosen, "\n"))
cat(paste0("[INFO] Cells clustered: ", nrow(da), "\n"))
cat(paste0("[INFO] Wrote: ", out_csv, "\n"))

membership_named <- setNames(output_membership, output_labels)
plot_vis <- vis_all
plot_membership <- as.integer(membership_named[rownames(plot_vis)])
plot_membership[is.na(plot_membership)] <- 0L

pdf_name <- paste0(tools::file_path_sans_ext(basename(out_csv)), "_kodama_membership.pdf")
pdf_path <- file.path(dirname(out_csv), pdf_name)
pdf(pdf_path)
plot(plot_vis, pch = 20, col = plot_membership, cex = 1)
dev.off()
cat(paste0("[INFO] Wrote: ", pdf_path, "\n"))
