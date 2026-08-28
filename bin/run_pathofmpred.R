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

titan_sample_report(
  cancer = cancer,
  features = x,
  patient_id = patient_id,
  output_file = file.path(outdir, "pathofmpred_research_report"),
  format = report_format,
  include_limited_evidence = include_limited,
  quiet = TRUE
)

writeLines(c(
  paste0("PathoFMPred_version=", as.character(packageVersion("PathoFMPred"))),
  paste0("fastPLS_version=", as.character(packageVersion("fastPLS"))),
  paste0("cancer=", cancer),
  paste0("patient_id=", patient_id),
  paste0("prediction_rows=", nrow(predictions)),
  "scope=TCGA discovery models; no independent external validation; not for clinical use"
), file.path(outdir, "pathofmpred_runtime.txt"))
