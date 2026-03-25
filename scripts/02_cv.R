source("src/helpers.R")
source("src/models.R")

cat("=== Cross-Validation ===\n\n")

output_dir <- "outputs/results"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Load and prepare fold data (preprocess per fold) -----------------------

raw <- load_train_data()
folds <- create_cv_folds(raw$quality, k = 5, seed = SEED)

fold_data <- list()
for (i in seq_along(folds)) {
  test_idx   <- folds[[i]]
  train_fold <- raw[-test_idx, ]
  test_fold  <- raw[ test_idx, ]

  proc_train <- preprocess(train_fold)
  proc_test  <- apply_preprocess(test_fold, proc_train$scale_params)

  fold_data[[i]] <- list(
    X_train = proc_train$X, y_train = proc_train$y,
    X_test  = proc_test$X,  y_test  = proc_test$y
  )
}

# ---- Run each model across folds -------------------------------------------

results <- data.frame()

for (m in MODEL_REGISTRY) {
  cat("Fitting:", m$name, "... ")
  fold_acc  <- numeric(length(folds))
  fold_rmse <- numeric(length(folds))
  skip <- FALSE

  for (i in seq_along(fold_data)) {
    fd <- fold_data[[i]]

    res <- tryCatch(
      m$fn(fd$X_train, fd$y_train, fd$X_test),
      error = function(e) {
        cat("[fold", i, "error:", conditionMessage(e), "] ")
        NULL
      }
    )

    if (is.null(res)) { skip <- TRUE; break }

    fold_acc[i]  <- compute_accuracy(fd$y_test, res$predictions)
    fold_rmse[i] <- compute_rmse(fd$y_test, res$predictions)
  }

  if (skip) {
    cat("SKIPPED\n")
    next
  }

  results <- rbind(results, data.frame(
    model            = m$name,
    mean_cv_accuracy = mean(fold_acc),
    sd_cv_accuracy   = sd(fold_acc),
    mean_cv_rmse     = mean(fold_rmse),
    sd_cv_rmse       = sd(fold_rmse),
    stringsAsFactors = FALSE
  ))
  cat(sprintf("Acc=%.4f  RMSE=%.4f\n",
              mean(fold_acc), mean(fold_rmse)))
}

# ---- Save -------------------------------------------------------------------

write.csv(results, file.path(output_dir, "cv_results.csv"), row.names = FALSE)
cat("\nCV results saved to", file.path(output_dir, "cv_results.csv"), "\n")
print(results)

# ---- Select two models for test prediction (best overall + best non-Random-Forest) ----

best_overall_row <- pick_best_by_cv_accuracy(results)
non_rf <- results[!vapply(results$model, is_random_forest_model, logical(1)), , drop = FALSE]
if (nrow(non_rf) == 0) {
  stop("No non-Random-Forest models in cv results — cannot select best_non_tree")
}
best_non_tree_row <- pick_best_by_cv_accuracy(non_rf)

selected <- list(
  best_overall              = best_overall_row$model[1],
  best_non_tree             = best_non_tree_row$model[1],
  best_overall_cv_accuracy  = best_overall_row$mean_cv_accuracy[1],
  best_overall_cv_rmse      = best_overall_row$mean_cv_rmse[1],
  best_non_tree_cv_accuracy = best_non_tree_row$mean_cv_accuracy[1],
  best_non_tree_cv_rmse     = best_non_tree_row$mean_cv_rmse[1]
)

models_dir <- "outputs/models"
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)
saveRDS(selected, file.path(models_dir, "selected_models.rds"))
cat("\nSelected for test prediction:\n")
cat("  Best overall:           ", selected$best_overall,
    sprintf(" (CV acc=%.4f, RMSE=%.4f)\n", selected$best_overall_cv_accuracy, selected$best_overall_cv_rmse))
cat("  Best non-Random Forest: ", selected$best_non_tree,
    sprintf(" (CV acc=%.4f, RMSE=%.4f)\n", selected$best_non_tree_cv_accuracy, selected$best_non_tree_cv_rmse))
cat("Saved selected_models.rds\n")

sel_csv <- data.frame(
  role       = c("best_overall", "best_non_tree"),
  model      = c(selected$best_overall, selected$best_non_tree),
  mean_cv_accuracy = c(selected$best_overall_cv_accuracy, selected$best_non_tree_cv_accuracy),
  mean_cv_rmse     = c(selected$best_overall_cv_rmse, selected$best_non_tree_cv_rmse),
  stringsAsFactors = FALSE
)
write.csv(sel_csv, file.path(output_dir, "selected_models.csv"), row.names = FALSE)
cat("Saved selected_models.csv\n")

# ---- Save preprocessing scaling from full training set (for test prediction) ----

proc_full <- preprocess(raw)
saveRDS(proc_full$scale_params, file.path(models_dir, "scale_params.rds"))
cat("Saved scale_params.rds (full training preprocessing)\n")
