library(ggplot2)
library(reshape2)

output_dir <- "outputs/results"

cv_path    <- file.path(output_dir, "cv_results.csv")
train_path <- file.path(output_dir, "full_train_results.csv")

if (!file.exists(cv_path)) {
  stop("Run scripts/02_cv.R first: ", cv_path, " not found")
}
if (!file.exists(train_path)) {
  stop("Run scripts/03_full_train.R first: ", train_path, " not found")
}

cv    <- read.csv(cv_path,    stringsAsFactors = FALSE)
ftrain <- read.csv(train_path, stringsAsFactors = FALSE)

# ---- Build long-format data for faceted plot --------------------------------

cv_long <- data.frame(
  model   = rep(cv$model, 2),
  metric  = rep(c("Accuracy", "RMSE"), each = nrow(cv)),
  setting = "Cross-Validation",
  value   = c(cv$mean_cv_accuracy, cv$mean_cv_rmse),
  sd      = c(cv$sd_cv_accuracy,   cv$sd_cv_rmse),
  stringsAsFactors = FALSE
)

train_long <- data.frame(
  model   = rep(ftrain$model, 2),
  metric  = rep(c("Accuracy", "RMSE"), each = nrow(ftrain)),
  setting = "Full Training Set",
  value   = c(ftrain$train_accuracy, ftrain$train_rmse),
  sd      = 0,
  stringsAsFactors = FALSE
)

all_data <- rbind(cv_long, train_long)

all_data$setting <- factor(all_data$setting,
                           levels = c("Cross-Validation", "Full Training Set"))
all_data$metric  <- factor(all_data$metric, levels = c("Accuracy", "RMSE"))

model_order <- cv$model
all_data$model <- factor(all_data$model, levels = model_order)

# ---- 2x2 comparison figure -------------------------------------------------

p <- ggplot(all_data, aes(x = model, y = value)) +
  geom_col(fill = "steelblue", width = 0.7) +
  geom_errorbar(aes(ymin = pmax(value - sd, 0), ymax = value + sd),
                width = 0.25, linewidth = 0.4) +
  facet_grid(setting ~ metric, scales = "free_y") +
  labs(title = "Model Comparison — Accuracy & RMSE",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x  = element_text(angle = 45, hjust = 1, size = 7),
        strip.text    = element_text(face = "bold"),
        plot.title    = element_text(hjust = 0.5))

ggsave(file.path(output_dir, "model_comparison.png"),
       p, width = 14, height = 8, dpi = 300)
cat("Saved model_comparison.png to", output_dir, "\n")
