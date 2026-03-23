source("src/helpers.R")
source("src/models.R")

cat("=== Test Set Prediction ===\n\n")

test_path  <- "data/processed/test.csv"
models_dir <- "outputs/models"
output_dir <- "outputs/results"

if (!file.exists(test_path)) {
  stop("test.csv not found at ", test_path,
       "\nPlace the test file there and re-run this script.")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Load test data and apply saved preprocessing --------------------------

scale_params <- readRDS(file.path(models_dir, "scale_params.rds"))
test_raw     <- read.csv(test_path, sep = ";", check.names = FALSE)
proc_test    <- apply_preprocess(test_raw, scale_params)

# Also preprocess full training set to refit models
raw  <- load_train_data()
proc <- preprocess(raw)

# ---- Predict with each model -----------------------------------------------

all_preds <- data.frame(row = seq_len(nrow(proc_test$X)))

for (m in MODEL_REGISTRY) {
  cat("Predicting:", m$name, "... ")

  res <- tryCatch(
    m$fn(proc$X, proc$y, proc_test$X),
    error = function(e) {
      cat("[error:", conditionMessage(e), "] ")
      NULL
    }
  )

  if (is.null(res)) {
    cat("SKIPPED\n")
    next
  }

  preds <- res$predictions
  if (m$is_reg) {
    preds <- pmin(pmax(round(preds), 3L), 9L)
  }

  safe_name <- gsub("[^A-Za-z0-9_]", "_", m$name)
  all_preds[[safe_name]] <- as.integer(preds)
  cat("done\n")
}

# ---- Save -------------------------------------------------------------------

write.csv(all_preds, file.path(output_dir, "test_predictions.csv"),
          row.names = FALSE)
cat("\nTest predictions saved to",
    file.path(output_dir, "test_predictions.csv"), "\n")
