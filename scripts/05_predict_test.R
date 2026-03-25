source("src/helpers.R")
source("src/models.R")

cat("=== Test Set Prediction (two CV-selected models) ===\n\n")

test_path  <- "data/processed/test.csv"
models_dir <- "outputs/models"
output_dir <- "outputs/results"
sel_path   <- file.path(models_dir, "selected_models.rds")

if (!file.exists(test_path)) {
  stop("test.csv not found at ", test_path,
       "\nPlace the file there and re-run this script.")
}
if (!file.exists(sel_path)) {
  stop("selected_models.rds not found at ", sel_path,
       "\nRun `make cv` or Rscript scripts/02_cv.R first.")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

selected <- readRDS(sel_path)
m_best  <- get_model_from_registry(selected$best_overall)
m_learn <- get_model_from_registry(selected$best_non_tree)

# ---- Load test data and apply saved preprocessing --------------------------

scale_params <- readRDS(file.path(models_dir, "scale_params.rds"))
test_raw     <- read.csv(test_path, sep = ";", check.names = FALSE)
proc_test    <- apply_preprocess(test_raw, scale_params)

raw <- load_train_data()
lc  <- if (!is.null(scale_params$log_chlorides)) scale_params$log_chlorides else TRUE
proc <- preprocess(raw, log_chlorides = lc)

# ---- Predict with the two selected models only -----------------------------

predict_one <- function(m) {
  res <- tryCatch(
    m$fn(proc$X, proc$y, proc_test$X),
    error = function(e) {
      cat("[error:", conditionMessage(e), "] ")
      NULL
    }
  )
  if (is.null(res)) {
    stop("Prediction failed for model: ", m$name)
  }
  preds <- res$predictions
  preds_raw <- as.numeric(preds)
  if (m$is_reg) {
    preds_int <- as.integer(pmin(pmax(round(preds), 3L), 9L))
  } else {
    preds_int <- as.integer(preds)
  }
  list(raw = preds_raw, int = preds_int)
}

cat("Predicting:", m_best$name, " (Best Performance Model) ... ")
out_best <- predict_one(m_best)
cat("done\n")

cat("Predicting:", m_learn$name, " (Best Performance Model We Have Learned) ... ")
out_learn <- predict_one(m_learn)
cat("done\n")

all_preds <- data.frame(row = seq_len(nrow(proc_test$X)))
all_preds[["Best Performance Model"]] <- out_best$int
all_preds[["Best Performance Model We Have Learned"]] <- out_learn$int

has_test_labels <- !is.null(proc_test$y)
test_metrics <- NULL
if (has_test_labels) {
  y <- proc_test$y
  test_metrics <- data.frame(
    model    = c(m_best$name, m_learn$name),
    accuracy = c(
      compute_accuracy(y, out_best$raw),
      compute_accuracy(y, out_learn$raw)
    ),
    rmse = c(
      compute_rmse(y, out_best$raw),
      compute_rmse(y, out_learn$raw)
    ),
    stringsAsFactors = FALSE
  )
} else {
  cat("Note: no `quality` column in test.csv — skipping test accuracy/RMSE.\n\n")
}

# ---- Save -------------------------------------------------------------------

write.csv(all_preds, file.path(output_dir, "test_predictions.csv"),
          row.names = FALSE)
cat("\nTest predictions saved to",
    file.path(output_dir, "test_predictions.csv"), "\n")

labels_path <- file.path(output_dir, "test_predictions_labels.txt")
lines <- c(
  paste0("Best Performance Model: ", selected$best_overall),
  paste0("Best Performance Model We Have Learned: ", selected$best_non_tree),
  "",
  paste0("(CV) Best overall — accuracy=", format(selected$best_overall_cv_accuracy, digits = 6),
         ", RMSE=", format(selected$best_overall_cv_rmse, digits = 6)),
  paste0("(CV) Best non-Random Forest — accuracy=", format(selected$best_non_tree_cv_accuracy, digits = 6),
         ", RMSE=", format(selected$best_non_tree_cv_rmse, digits = 6))
)
writeLines(lines, labels_path)
cat("Labels saved to", labels_path, "\n")

if (!is.null(test_metrics)) {
  metrics_path <- file.path(output_dir, "test_metrics.csv")
  write.csv(test_metrics, metrics_path, row.names = FALSE)
  cat("Test metrics saved to", metrics_path, "\n")
  cat("\n--- Test set metrics ---\n")
  print(test_metrics, row.names = FALSE, digits = 4)
}
