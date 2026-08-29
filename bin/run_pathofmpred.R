#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(PathoFMPred)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
read_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx)) return(default)
  if (idx == length(args)) stop("Missing value for ", flag)
  args[[idx + 1L]]
}

input <- read_arg("--features")
cancer <- toupper(read_arg("--cancer", ""))
patient_id <- read_arg("--patient-id", "sample")
outdir <- read_arg("--outdir", "pathofmpred")
report_format <- read_arg("--report-format", "html")
include_limited <- identical(tolower(read_arg("--include-limited-evidence", "false")), "true")

if (is.null(input) || !nzchar(input)) stop("--features is required")
if (!nzchar(cancer)) stop("--cancer is required and must be a TCGA cancer code")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

features <- read.csv(input, check.names = FALSE, stringsAsFactors = FALSE)
expected <- sprintf("titan_%03d", 0:767)
missing <- setdiff(expected, names(features))
if (length(missing)) stop("Missing TITAN columns: ", paste(head(missing, 10L), collapse = ", "))
x <- features[, expected, drop = FALSE]
if (ncol(x) != 768L || any(!is.finite(as.matrix(x)))) {
  stop("PathoFMPred requires exactly 768 finite TITAN values")
}

predictions <- predict_titan(
  cancer = cancer,
  features = x,
  patient_id = patient_id,
  include_limited_evidence = include_limited
)
write.csv(predictions, file.path(outdir, "pathofmpred_predictions.csv"), row.names = FALSE)

radar <- plot_titan_radar(predictions)
ggsave(file.path(outdir, "pathofmpred_continuous_radar.png"), radar,
       width = 11, height = 9, units = "in", dpi = 200, bg = "white")
binary <- plot_titan_binary(predictions)
ggsave(file.path(outdir, "pathofmpred_binary_predictions.png"), binary,
       width = 11, height = 9, units = "in", dpi = 200, bg = "white")

# rmarkdown writes intermediate files beside its input document. Render from a
# writable copy because the protected package library is mounted read-only.
template_root <- system.file(
  "rmarkdown", "templates", "titan-report", package = "PathoFMPred"
)
if (!nzchar(template_root)) stop("PathoFMPred report template is unavailable")
render_parent <- tempfile("pathofmpred_report_")
dir.create(render_parent, recursive = TRUE)
template_copy_root <- file.path(render_parent, "titan-report")
dir.create(template_copy_root, recursive = TRUE)
template_entries <- list.files(
  template_root, recursive = TRUE, full.names = TRUE, all.files = TRUE,
  include.dirs = TRUE, no.. = TRUE
)
relative_entries <- substring(template_entries, nchar(template_root) + 2L)
directory_entries <- dir.exists(template_entries)
for (relative_dir in relative_entries[directory_entries]) {
  dir.create(file.path(template_copy_root, relative_dir), recursive = TRUE)
}
source_files <- template_entries[!directory_entries]
destination_files <- file.path(template_copy_root, relative_entries[!directory_entries])
copy_ok <- file.copy(source_files, destination_files, overwrite = TRUE)
if (!length(copy_ok) || !all(copy_ok)) {
  stop("Could not copy the PathoFMPred report template to a writable directory")
}
template <- file.path(template_copy_root, "skeleton", "skeleton.Rmd")
if (!file.exists(template)) stop("Writable PathoFMPred report template is missing")

report_formats <- if (report_format == "both") c("html", "pdf") else report_format
if (!all(report_formats %in% c("html", "pdf"))) {
  stop("--report-format must be html, pdf, or both")
}
outdir_abs <- normalizePath(outdir, mustWork = TRUE)
for (extension in report_formats) {
  rmarkdown::render(
    template,
    output_format = if (extension == "html") "html_document" else "pdf_document",
    output_file = paste0("pathofmpred_research_report.", extension),
    output_dir = outdir_abs,
    params = list(
      predictions = predictions,
      sample_id = patient_id,
      cancer = cancer,
      clinical_context = NULL
    ),
    envir = new.env(parent = globalenv()),
    quiet = TRUE
  )
}

writeLines(c(
  paste0("PathoFMPred_version=", as.character(packageVersion("PathoFMPred"))),
  paste0("fastPLS_version=", as.character(packageVersion("fastPLS"))),
  paste0("cancer=", cancer),
  paste0("patient_id=", patient_id),
  paste0("prediction_rows=", nrow(predictions)),
  "scope=TCGA discovery models; no independent external validation; not for clinical use"
), file.path(outdir, "pathofmpred_runtime.txt"))
