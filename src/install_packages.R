cat("R version:", R.version.string, "\n")
if (as.numeric(R.version$major) < 4) {
  warning("R >= 4.0 is recommended. You are running ", R.version.string)
}

options(repos = c(CRAN = "https://cloud.r-project.org"))

cran_packages <- c(
  "ggplot2", "reshape2",
  "nnet", "glmnet", "class", "randomForest",
  "e1071", "caret"
)

for (pkg in cran_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing", pkg, "...\n")
    install.packages(pkg, dependencies = TRUE)
  } else {
    cat(pkg, "already installed.\n")
  }
}

cat("\n--- Package verification ---\n")
all_ok <- TRUE
for (pkg in cran_packages) {
  ok <- requireNamespace(pkg, quietly = TRUE)
  cat(sprintf("  %-15s %s\n", pkg, if (ok) "OK" else "MISSING"))
  if (!ok) all_ok <- FALSE
}

if (!all_ok) {
  stop("Some required CRAN packages are missing. Check output above.")
}
cat("\nAll required packages are available.\n")
