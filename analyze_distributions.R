library(ggplot2)

input_path <- "data/processed/train.csv"
output_dir <- "outputs/distributions"

if (!file.exists(input_path)) {
  stop(paste("Input file not found:", input_path))
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

train <- read.csv(input_path, sep = ";", check.names = FALSE)

# Quality distribution table
quality_counts <- as.data.frame(table(train$quality))
colnames(quality_counts) <- c("quality", "count")
quality_counts <- quality_counts[order(as.numeric(as.character(quality_counts$quality))), ]
quality_counts$proportion <- quality_counts$count / sum(quality_counts$count)

write.csv(quality_counts, file.path(output_dir, "quality_distribution.csv"), row.names = FALSE)

cat("Distribution of y-variable `quality`:\n")
print(quality_counts)

# Quality bar chart
quality_plot <- ggplot(train, aes(x = factor(quality))) +
  geom_bar(fill = "steelblue", color = "black") +
  labs(
    title = "Distribution of quality",
    x = "quality",
    y = "Count"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(output_dir, "quality_distribution_bar.png"),
  plot = quality_plot,
  width = 8,
  height = 5,
  dpi = 300
)

# Distribution plots for all variables
for (col in colnames(train)) {
  values <- train[[col]]
  plot_path <- file.path(output_dir, paste0(gsub("[^A-Za-z0-9_]+", "_", col), "_distribution.png"))

  if (is.numeric(values)) {
    p <- ggplot(train, aes(x = .data[[col]])) +
      geom_histogram(bins = 30, fill = "darkorange", color = "black") +
      labs(
        title = paste("Distribution of", col),
        x = col,
        y = "Count"
      ) +
      theme_minimal()
  } else {
    p <- ggplot(train, aes(x = .data[[col]])) +
      geom_bar(fill = "darkorange", color = "black") +
      labs(
        title = paste("Distribution of", col),
        x = col,
        y = "Count"
      ) +
      theme_minimal()
  }

  ggsave(filename = plot_path, plot = p, width = 8, height = 5, dpi = 300)
}

cat("\nSaved distribution outputs to:", output_dir, "\n")
