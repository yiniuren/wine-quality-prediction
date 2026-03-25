suppressPackageStartupMessages(library(caret))

SEED <- 42

load_train_data <- function(path = "data/processed/train.csv") {
  if (!file.exists(path)) stop("File not found: ", path)
  read.csv(path, sep = ";", check.names = FALSE)
}

preprocess <- function(df, scale = TRUE, log_chlorides = TRUE) {
  if (log_chlorides && "chlorides" %in% colnames(df)) {
    df$chlorides <- log1p(df$chlorides)
  }

  y <- df$quality
  X <- df[, setdiff(colnames(df), "quality"), drop = FALSE]

  scale_params <- list(log_chlorides = log_chlorides, means = NULL, sds = NULL)

  if (scale) {
    cont_cols <- setdiff(colnames(X), "is_red")
    means <- colMeans(X[, cont_cols, drop = FALSE])
    sds   <- apply(X[, cont_cols, drop = FALSE], 2, sd)
    sds[sds == 0] <- 1
    X[, cont_cols] <- scale(X[, cont_cols, drop = FALSE],
                            center = means, scale = sds)
    scale_params$means <- means
    scale_params$sds   <- sds
  }

  list(X = X, y = y, scale_params = scale_params)
}

apply_preprocess <- function(df, scale_params) {
  if (scale_params$log_chlorides && "chlorides" %in% colnames(df)) {
    df$chlorides <- log1p(df$chlorides)
  }

  y <- if ("quality" %in% colnames(df)) df$quality else NULL
  X <- df[, setdiff(colnames(df), "quality"), drop = FALSE]

  if (!is.null(scale_params$means)) {
    cont_cols <- names(scale_params$means)
    X[, cont_cols] <- scale(X[, cont_cols, drop = FALSE],
                            center = scale_params$means,
                            scale  = scale_params$sds)
  }

  list(X = X, y = y)
}

# Append squared terms for every continuous column (all except is_red).
# Apply the same transformation to train and test for a given model.
add_quadratic_features <- function(X) {
  if (!is.data.frame(X)) {
    X <- as.data.frame(X)
  }
  cont_cols <- setdiff(colnames(X), "is_red")
  out <- X
  for (col in cont_cols) {
    out[[paste0(col, "_sq")]] <- X[[col]]^2
  }
  out
}

create_cv_folds <- function(y, k = 5, seed = SEED) {
  set.seed(seed)
  createFolds(factor(y), k = k, list = TRUE, returnTrain = FALSE)
}

compute_accuracy <- function(actual, predicted) {
  pred_int <- round(predicted)
  pred_int <- pmin(pmax(pred_int, 3L), 9L)
  mean(actual == pred_int)
}

compute_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

# Tree ensembles excluded when picking "models we have learned" (non-RF best).
is_random_forest_model <- function(name) {
  grepl("^Random Forest", name)
}

# One row from cv_results: max mean_cv_accuracy, tie-break lower mean_cv_rmse, then model name.
pick_best_by_cv_accuracy <- function(cv_df) {
  if (nrow(cv_df) == 0) {
    stop("pick_best_by_cv_accuracy: empty data frame")
  }
  ord <- order(-cv_df$mean_cv_accuracy, cv_df$mean_cv_rmse, cv_df$model)
  cv_df[ord[1], , drop = FALSE]
}
