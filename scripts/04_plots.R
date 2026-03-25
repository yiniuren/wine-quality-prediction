library(ggplot2)

output_dir <- "outputs/results"

cv_path <- file.path(output_dir, "cv_results.csv")

if (!file.exists(cv_path)) {
  stop("Run scripts/02_cv.R first: ", cv_path, " not found")
}

cv <- read.csv(cv_path, stringsAsFactors = FALSE)

cv_long <- data.frame(
  model  = rep(cv$model, 2),
  metric = rep(c("Accuracy", "RMSE"), each = nrow(cv)),
  value  = c(cv$mean_cv_accuracy, cv$mean_cv_rmse),
  sd     = c(cv$sd_cv_accuracy,   cv$sd_cv_rmse),
  stringsAsFactors = FALSE
)
cv_long$metric <- factor(cv_long$metric, levels = c("Accuracy", "RMSE"))
cv_long$model <- factor(cv_long$model, levels = cv$model)

p <- ggplot(cv_long, aes(x = model, y = value)) +
  geom_col(fill = "steelblue", width = 0.7) +
  geom_errorbar(aes(ymin = pmax(value - sd, 0), ymax = value + sd),
                width = 0.25, linewidth = 0.4) +
  facet_wrap(~ metric, nrow = 1, scales = "free_y") +
  labs(title = "Model comparison — 5-fold CV (mean ± SD)",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        strip.text  = element_text(face = "bold"),
        plot.title  = element_text(hjust = 0.5))

ggsave(file.path(output_dir, "model_comparison.png"),
       p, width = 14, height = 5, dpi = 300)
cat("Saved model_comparison.png to", output_dir, "\n")
