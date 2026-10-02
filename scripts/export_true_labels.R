#!/usr/bin/env Rscript
# Export the finalized ground-truth .rds files to UTF-8 CSVs so the Python
# pipeline can read them without an R dependency at run time.
#
#   Rscript scripts/export_true_labels.R
#
# Inputs  (data/true_labels/):
#   09_data_final.rds          500 studies x 56 cols  (consolidated gold, 1 row/study)
#   01_study_data_long_raw.rds 750 rows   x 79 cols   (per-rater codings, 1 row/rater/study)
# Outputs (data/true_labels/):
#   09_data_final.csv
#   01_study_data_long_raw.csv
#
# All columns are written as-is (character); MC cells that contain commas are
# quoted by write.csv and round-trip correctly through pandas.read_csv.

# Robust project-root detection (works whether run from repo root or scripts/)
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
if (length(script_path) == 1 && nzchar(script_path)) {
  root <- normalizePath(file.path(dirname(script_path), ".."))
} else {
  root <- normalizePath(getwd())
}

tl <- file.path(root, "data", "true_labels")

export_one <- function(rds_name, csv_name) {
  src <- file.path(tl, rds_name)
  dst <- file.path(tl, csv_name)
  if (!file.exists(src)) {
    stop(sprintf("Input not found: %s", src))
  }
  d <- readRDS(src)
  # Coerce every column to character so numeric codes (e.g. "-96") and free
  # text are written verbatim, with no scientific notation or type coercion.
  d[] <- lapply(d, function(col) {
    v <- as.character(col)
    v[is.na(v)] <- ""
    v
  })
  write.csv(d, dst, row.names = FALSE, fileEncoding = "UTF-8", na = "")
  cat(sprintf("wrote %s  (%d rows x %d cols)\n", dst, nrow(d), ncol(d)))
}

export_one("09_data_final.rds", "09_data_final.csv")
export_one("01_study_data_long_raw.rds", "01_study_data_long_raw.csv")
cat("done.\n")
