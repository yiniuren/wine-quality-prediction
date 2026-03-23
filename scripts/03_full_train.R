source("src/helpers.R")
source("src/models.R")

cat("=== Full Training Set Evaluation ===\n\n")

output_dir  <- "outputs/results"
models_dir  <- "outputs/models"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Preprocess full training set -------------------------------------------

raw  <- load_train_data()
proc <- preprocess(raw)
X <- proc$X
y <- proc$y

saveRDS(proc$scale_params, file.path(models_dir, "scale_params.rds"))
cat("Saved scale_params.rds\n")

# ---- Fit each model on full data, evaluate in-sample ------------------------

results <- data.frame()

for (m in MODEL_REGISTRY) {
  cat("Fitting:", m$name, "... ")

  res <- tryCatch(
    m$fn(X, y, X),
    error = function(e) {
      cat("[error:", conditionMessage(e), "] ")
      NULL
    }
  )

  if (is.null(res)) {
    cat("SKIPPED\n")
    next
  }

  acc  <- compute_accuracy(y, res$predictions)
  rmse <- compute_rmse(y, res$predictions)

  safe_name <- gsub("[^A-Za-z0-9_]", "_", m$name)
  saveRDS(res$model, file.path(models_dir, paste0(safe_name, ".rds")))

  results <- rbind(results, data.frame(
    model          = m$name,
    train_accuracy = acc,
    train_rmse     = rmse,
    stringsAsFactors = FALSE
  ))
  cat(sprintf("Acc=%.4f  RMSE=%.4f\n", acc, rmse))
}

# ---- Save -------------------------------------------------------------------

write.csv(results, file.path(output_dir, "full_train_results.csv"),
          row.names = FALSE)
cat("\nFull-train results saved to",
    file.path(output_dir, "full_train_results.csv"), "\n")
print(results)
