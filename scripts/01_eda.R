library(ggplot2)
library(reshape2)

input_path <- "data/processed/train.csv"
output_dir <- "outputs/eda"

if (!file.exists(input_path)) stop("Input file not found: ", input_path)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

train <- read.csv(input_path, sep = ";", check.names = FALSE)

# ---- 1. Correlation heatmap ------------------------------------------------

cor_mat  <- cor(train, use = "complete.obs")
cor_long <- melt(cor_mat, varnames = c("Var1", "Var2"), value.name = "corr")

p_corr <- ggplot(cor_long, aes(x = Var1, y = Var2, fill = corr)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", corr)), size = 2.2) +
  scale_fill_gradient2(low = "steelblue", mid = "white", high = "firebrick",
                       midpoint = 0, limits = c(-1, 1)) +
  labs(title = "Correlation Heatmap", x = NULL, y = NULL) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        axis.text.y = element_text(size = 7))

ggsave(file.path(output_dir, "correlation_heatmap.png"),
       p_corr, width = 10, height = 8, dpi = 300)
cat("Saved correlation_heatmap.png\n")

# ---- 2. Boxplots of each feature grouped by quality ------------------------

feature_cols <- setdiff(colnames(train), c("quality", "is_red"))
long_box <- melt(train[, c(feature_cols, "quality")],
                 id.vars = "quality",
                 variable.name = "feature",
                 value.name = "value")

p_box <- ggplot(long_box, aes(x = factor(quality), y = value)) +
  geom_boxplot(fill = "steelblue", alpha = 0.6, outlier.size = 0.5) +
  facet_wrap(~ feature, scales = "free_y", ncol = 3) +
  labs(title = "Feature Distributions by Quality Level",
       x = "quality", y = NULL) +
  theme_minimal() +
  theme(strip.text = element_text(size = 8))

ggsave(file.path(output_dir, "boxplots_by_quality.png"),
       p_box, width = 12, height = 10, dpi = 300)
cat("Saved boxplots_by_quality.png\n")

# ---- 3. Red vs. white wine comparison (density) ----------------------------

features_rw <- c("alcohol", "volatile.acidity", "sulphates",
                 "residual.sugar", "pH", "density")

long_rw <- melt(train[, c(features_rw, "is_red")],
                id.vars = "is_red",
                variable.name = "feature",
                value.name = "value")
long_rw$wine_type <- ifelse(long_rw$is_red == 1, "Red", "White")

p_rw <- ggplot(long_rw, aes(x = value, fill = wine_type)) +
  geom_density(alpha = 0.45) +
  facet_wrap(~ feature, scales = "free", ncol = 3) +
  scale_fill_manual(values = c(Red = "firebrick", White = "gold3")) +
  labs(title = "Red vs. White Wine — Feature Densities",
       x = NULL, y = "Density", fill = NULL) +
  theme_minimal()

ggsave(file.path(output_dir, "red_vs_white.png"),
       p_rw, width = 12, height = 7, dpi = 300)
cat("Saved red_vs_white.png\n")

# ---- 4. Chlorides: raw vs. log1p comparison --------------------------------

comp <- data.frame(
  Raw      = train$chlorides,
  Log1p    = log1p(train$chlorides)
)
long_comp <- melt(comp, variable.name = "Transform", value.name = "value")

p_chl <- ggplot(long_comp, aes(x = value)) +
  geom_histogram(bins = 40, fill = "darkorange", color = "black") +
  facet_wrap(~ Transform, scales = "free_x") +
  labs(title = "Chlorides — Raw vs. log1p Transform",
       x = NULL, y = "Count") +
  theme_minimal()

ggsave(file.path(output_dir, "chlorides_log_comparison.png"),
       p_chl, width = 10, height = 4.5, dpi = 300)
cat("Saved chlorides_log_comparison.png\n")

cat("\nAll EDA plots saved to:", output_dir, "\n")
