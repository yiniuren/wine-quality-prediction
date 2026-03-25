suppressPackageStartupMessages({
  library(nnet)
  library(glmnet)
  library(randomForest)
  library(e1071)
})

# ---------------------------------------------------------------------------
# 1. OLS
# ---------------------------------------------------------------------------
train_and_predict_ols <- function(X_train, y_train, X_test) {
  df <- data.frame(quality = y_train, X_train, check.names = FALSE)
  fit <- lm(quality ~ ., data = df)
  preds <- predict(fit, newdata = data.frame(X_test, check.names = FALSE))
  list(predictions = as.numeric(preds), model = fit)
}

# ---------------------------------------------------------------------------
# 2. Multinomial logistic regression
# ---------------------------------------------------------------------------
train_and_predict_multinom <- function(X_train, y_train, X_test) {
  df <- data.frame(quality = factor(y_train), X_train, check.names = FALSE)
  fit <- suppressWarnings(multinom(quality ~ ., data = df, trace = FALSE, MaxNWts = 5000))
  preds <- predict(fit, newdata = data.frame(X_test, check.names = FALSE))
  list(predictions = as.integer(as.character(preds)), model = fit)
}

# ---------------------------------------------------------------------------
# 3. Ridge logistic (glmnet, alpha = 0)
# ---------------------------------------------------------------------------
train_and_predict_ridge_logistic <- function(X_train, y_train, X_test) {
  X_mat <- as.matrix(X_train)
  y_fac <- factor(y_train)
  set.seed(42)
  cv_fit <- cv.glmnet(X_mat, y_fac, family = "multinomial",
                       alpha = 0, type.measure = "class")
  preds <- predict(cv_fit, newx = as.matrix(X_test),
                   s = "lambda.min", type = "class")
  list(predictions = as.integer(preds), model = cv_fit)
}

# ---------------------------------------------------------------------------
# 4. Lasso logistic (glmnet, alpha = 1)
# ---------------------------------------------------------------------------
train_and_predict_lasso_logistic <- function(X_train, y_train, X_test) {
  X_mat <- as.matrix(X_train)
  y_fac <- factor(y_train)
  set.seed(42)
  cv_fit <- cv.glmnet(X_mat, y_fac, family = "multinomial",
                       alpha = 1, type.measure = "class")
  preds <- predict(cv_fit, newx = as.matrix(X_test),
                   s = "lambda.min", type = "class")
  list(predictions = as.integer(preds), model = cv_fit)
}

# ---------------------------------------------------------------------------
# 5. Elastic Net logistic (glmnet, alpha = 0.5)
# ---------------------------------------------------------------------------
train_and_predict_enet_logistic <- function(X_train, y_train, X_test) {
  X_mat <- as.matrix(X_train)
  y_fac <- factor(y_train)
  set.seed(42)
  cv_fit <- cv.glmnet(X_mat, y_fac, family = "multinomial",
                       alpha = 0.5, type.measure = "class")
  preds <- predict(cv_fit, newx = as.matrix(X_test),
                   s = "lambda.min", type = "class")
  list(predictions = as.integer(preds), model = cv_fit)
}

# ---------------------------------------------------------------------------
# 6. KNN (internal 3-fold CV to pick k from {5, 7, 11})
# ---------------------------------------------------------------------------
train_and_predict_knn <- function(X_train, y_train, X_test) {
  X_tr <- as.matrix(X_train)
  X_te <- as.matrix(X_test)
  y_fac <- factor(y_train)

  n <- nrow(X_tr)
  set.seed(123)
  fold_ids <- sample(rep(1:3, length.out = n))

  best_k   <- 5
  best_acc <- 0
  for (k_try in c(5, 7, 11)) {
    accs <- numeric(3)
    for (f in 1:3) {
      idx <- fold_ids == f
      p <- class::knn(X_tr[!idx, , drop = FALSE],
                      X_tr[ idx, , drop = FALSE],
                      y_fac[!idx], k = k_try)
      accs[f] <- mean(p == y_fac[idx])
    }
    if (mean(accs) > best_acc) {
      best_acc <- mean(accs)
      best_k   <- k_try
    }
  }

  preds <- class::knn(X_tr, X_te, y_fac, k = best_k)
  list(predictions = as.integer(as.character(preds)),
       model = list(k = best_k))
}

# ---------------------------------------------------------------------------
# 7. Random Forest classifier
# ---------------------------------------------------------------------------
train_and_predict_rf_class <- function(X_train, y_train, X_test) {
  set.seed(42)
  fit <- randomForest(x = X_train, y = factor(y_train), ntree = 500)
  preds <- predict(fit, newdata = X_test)
  list(predictions = as.integer(as.character(preds)), model = fit)
}

# ---------------------------------------------------------------------------
# 8. Naive Bayes (Gaussian)
# ---------------------------------------------------------------------------
train_and_predict_naive_bayes <- function(X_train, y_train, X_test) {
  # Gaussian Naive Bayes is a poor fit for binary is_red; use chemistry-only features.
  nb_cols <- setdiff(colnames(X_train), "is_red")
  X_tr <- X_train[, nb_cols, drop = FALSE]
  X_te <- X_test[, nb_cols, drop = FALSE]
  fit <- naiveBayes(x = as.data.frame(X_tr), y = factor(y_train))
  preds <- predict(fit, newdata = as.data.frame(X_te))
  list(predictions = as.integer(as.character(preds)), model = fit)
}

# ---------------------------------------------------------------------------
# 9. Ridge regression (glmnet, alpha = 0, gaussian)
# ---------------------------------------------------------------------------
train_and_predict_ridge_reg <- function(X_train, y_train, X_test) {
  X_mat <- as.matrix(X_train)
  set.seed(42)
  cv_fit <- cv.glmnet(X_mat, y_train, family = "gaussian", alpha = 0)
  preds <- predict(cv_fit, newx = as.matrix(X_test), s = "lambda.min")
  list(predictions = as.numeric(preds), model = cv_fit)
}

# ---------------------------------------------------------------------------
# 10. Lasso regression (glmnet, alpha = 1, gaussian)
# ---------------------------------------------------------------------------
train_and_predict_lasso_reg <- function(X_train, y_train, X_test) {
  X_mat <- as.matrix(X_train)
  set.seed(42)
  cv_fit <- cv.glmnet(X_mat, y_train, family = "gaussian", alpha = 1)
  preds <- predict(cv_fit, newx = as.matrix(X_test), s = "lambda.min")
  list(predictions = as.numeric(preds), model = cv_fit)
}

# ---------------------------------------------------------------------------
# 11. Random Forest regressor
# ---------------------------------------------------------------------------
train_and_predict_rf_reg <- function(X_train, y_train, X_test) {
  set.seed(42)
  fit <- randomForest(x = X_train, y = y_train, ntree = 500)
  preds <- predict(fit, newdata = X_test)
  list(predictions = as.numeric(preds), model = fit)
}

# ---------------------------------------------------------------------------
# Separate red / white submodels (no is_red column in each subfit)
# ---------------------------------------------------------------------------
strip_stratified_features <- function(X) {
  if (!"is_red" %in% colnames(X)) {
    return(X)
  }
  X[, setdiff(colnames(X), "is_red"), drop = FALSE]
}

wrap_stratified_by_wine_type <- function(base_fn) {
  function(X_train, y_train, X_test) {
    if (!"is_red" %in% colnames(X_train) || !"is_red" %in% colnames(X_test)) {
      stop("is_red column required for stratified-by-wine-type models")
    }
    red   <- which(X_train$is_red == 1)
    white <- which(X_train$is_red == 0)
    if (length(red) == 0L || length(white) == 0L) {
      return(base_fn(X_train, y_train, X_test))
    }
    Xtr_s <- strip_stratified_features(X_train)
    Xte_s <- strip_stratified_features(X_test)
    out_r <- base_fn(Xtr_s[red, , drop = FALSE], y_train[red], Xte_s)
    out_w <- base_fn(Xtr_s[white, , drop = FALSE], y_train[white], Xte_s)
    if (is.null(out_r) || is.null(out_w)) {
      return(NULL)
    }
    ir_te <- X_test$is_red == 1
    preds <- ifelse(ir_te, out_r$predictions, out_w$predictions)
    list(
      predictions = preds,
      model = list(red = out_r$model, white = out_w$model)
    )
  }
}

# ---------------------------------------------------------------------------
# MODEL_REGISTRY — 11 global + 11 by wine type
# ---------------------------------------------------------------------------
BASE_MODEL_REGISTRY <- list(
  list(name = "OLS",                  fn = train_and_predict_ols,             is_reg = TRUE),
  list(name = "Multinomial Logistic", fn = train_and_predict_multinom,       is_reg = FALSE),
  list(name = "Ridge Logistic",       fn = train_and_predict_ridge_logistic,  is_reg = FALSE),
  list(name = "Lasso Logistic",       fn = train_and_predict_lasso_logistic,  is_reg = FALSE),
  list(name = "Elastic Net Logistic", fn = train_and_predict_enet_logistic,   is_reg = FALSE),
  list(name = "KNN",                  fn = train_and_predict_knn,             is_reg = FALSE),
  list(name = "Random Forest (clf)",  fn = train_and_predict_rf_class,        is_reg = FALSE),
  list(name = "Naive Bayes",          fn = train_and_predict_naive_bayes,      is_reg = FALSE),
  list(name = "Ridge Regression",     fn = train_and_predict_ridge_reg,        is_reg = TRUE),
  list(name = "Lasso Regression",     fn = train_and_predict_lasso_reg,        is_reg = TRUE),
  list(name = "Random Forest (reg)",  fn = train_and_predict_rf_reg,           is_reg = TRUE)
)

STRATIFIED_MODEL_REGISTRY <- lapply(BASE_MODEL_REGISTRY, function(m) {
  list(
    name   = paste0(m$name, " (by wine type)"),
    fn     = wrap_stratified_by_wine_type(m$fn),
    is_reg = m$is_reg
  )
})

MODEL_REGISTRY <- c(BASE_MODEL_REGISTRY, STRATIFIED_MODEL_REGISTRY)
