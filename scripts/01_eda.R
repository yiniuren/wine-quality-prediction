library(ggplot2)
library(reshape2)

input_path <- "data/processed/train.csv"
output_dir <- "outputs/eda"

if (!file.exists(input_path)) stop("Input file not found: ", input_path)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

train <- read.csv(input_path, sep = ";", check.names = FALSE)

eda_theme <- function() {
  theme_minimal(base_size = 11) +
    theme(plot.title = element_text(hjust = 0.5))
}

# ---- 1. Quality distribution table + bar; per-variable marginals -----------

quality_counts <- as.data.frame(table(train$quality))
colnames(quality_counts) <- c("quality", "count")
quality_counts <- quality_counts[order(as.numeric(as.character(quality_counts$quality))), ]
quality_counts$proportion <- quality_counts$count / sum(quality_counts$count)

write.csv(quality_counts, file.path(output_dir, "quality_distribution.csv"), row.names = FALSE)

cat("Distribution of y-variable `quality`:\n")
print(quality_counts)

quality_plot <- ggplot(train, aes(x = factor(quality))) +
  geom_bar(fill = "steelblue", color = "black") +
  labs(
    title = "Distribution of quality",
    x = "quality",
    y = "Count"
  ) +
  eda_theme()

ggsave(
  filename = file.path(output_dir, "quality_distribution_bar.png"),
  plot = quality_plot,
  width = 8,
  height = 5,
  dpi = 300
)

for (col in colnames(train)) {
  if (col == "quality") {
    next
  }

  plot_path <- file.path(output_dir, paste0(gsub("[^A-Za-z0-9_]+", "_", col), "_distribution.png"))
  values <- train[[col]]

  if (col == "is_red" || !is.numeric(values)) {
    p <- ggplot(train, aes(x = factor(.data[[col]]))) +
      geom_bar(fill = "steelblue", color = "black") +
      labs(
        title = paste("Distribution of", col),
        x = col,
        y = "Count"
      ) +
      eda_theme()
  } else {
    p <- ggplot(train, aes(x = .data[[col]])) +
      geom_density(fill = "steelblue", alpha = 0.35, color = "black", linewidth = 0.4) +
      labs(
        title = paste("Distribution of", col),
        x = col,
        y = "Density"
      ) +
      eda_theme()
  }

  ggsave(filename = plot_path, plot = p, width = 8, height = 5, dpi = 300)
}

# ---- 2. Correlation heatmap ------------------------------------------------

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

# ---- 3. Boxplots of each feature grouped by quality ------------------------

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

# ---- 4. Red vs. white — all predictors (overlaid densities) ----------------

features_all <- setdiff(colnames(train), c("quality", "is_red"))
long_rw_all <- melt(train[, c(features_all, "is_red"), drop = FALSE],
                    id.vars = "is_red",
                    variable.name = "feature",
                    value.name = "value")
long_rw_all$wine_type <- ifelse(long_rw_all$is_red == 1, "Red", "White")

p_rw_all <- ggplot(long_rw_all, aes(x = value, fill = wine_type)) +
  geom_density(alpha = 0.45) +
  facet_wrap(~ feature, scales = "free", ncol = 3) +
  scale_fill_manual(values = c(Red = "firebrick", White = "gold3")) +
  labs(title = "Red vs. White Wine — All Feature Densities",
       x = NULL, y = "Density", fill = NULL) +
  theme_minimal() +
  theme(strip.text = element_text(size = 8))

ggsave(file.path(output_dir, "red_vs_white_all_features.png"),
       p_rw_all, width = 14, height = 13, dpi = 300)
cat("Saved red_vs_white_all_features.png\n")

# ---- 5. Red vs. white — quality (proportional bars, semi-transparent) -----

quality_levels <- sort(unique(train$quality))
prop_rows <- list()
for (wt in c("White", "Red")) {
  sub <- train[if (wt == "Red") train$is_red == 1 else train$is_red == 0, ]
  tab <- table(factor(sub$quality, levels = quality_levels))
  n_tot <- sum(tab)
  prop_rows[[wt]] <- data.frame(
    quality    = quality_levels,
    proportion = as.numeric(tab) / n_tot,
    wine_type  = wt,
    stringsAsFactors = FALSE
  )
}
df_q <- do.call(rbind, prop_rows)
df_q$x <- df_q$quality + ifelse(df_q$wine_type == "Red", -0.22, 0.22)

p_q_rw <- ggplot(df_q, aes(x = x, y = proportion, fill = wine_type)) +
  geom_col(alpha = 0.55, width = 0.38, color = NA) +
  scale_x_continuous(breaks = quality_levels, labels = quality_levels) +
  scale_fill_manual(values = c(Red = "firebrick", White = "gold3")) +
  labs(
    title = "Quality distribution: Red vs. White wine",
    subtitle = "Proportion within each wine type; bars offset slightly for overlap",
    x = "quality",
    y = "Proportion",
    fill = NULL
  ) +
  theme_minimal()

ggsave(file.path(output_dir, "red_vs_white_quality.png"),
       p_q_rw, width = 9, height = 5, dpi = 300)
cat("Saved red_vs_white_quality.png\n")

# ---- 6. Chlorides: raw vs. log1p comparison --------------------------------

comp <- data.frame(
  Raw      = train$chlorides,
  Log1p    = log1p(train$chlorides)
)
long_comp <- melt(comp, variable.name = "Transform", value.name = "value")

p_chl <- ggplot(long_comp, aes(x = value)) +
  geom_density(fill = "steelblue", alpha = 0.35, color = "black", linewidth = 0.4) +
  facet_wrap(~ Transform, scales = "free_x") +
  labs(title = "Chlorides — Raw vs. log1p Transform",
       x = NULL, y = "Density") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5))

ggsave(file.path(output_dir, "chlorides_log_comparison.png"),
       p_chl, width = 10, height = 4.5, dpi = 300)
cat("Saved chlorides_log_comparison.png\n")

cat("\nAll EDA plots saved to:", output_dir, "\n")
