# Wine Quality Prediction

Predict wine quality (score 3–9) from physicochemical properties using a
suite of classification and regression models, then compare their accuracy
and RMSE side by side. Models whose names end with **`(by wine type)`** fit
one submodel on red wines only and one on white wines only (no `is_red`
feature inside each subfit); predictions use the submodel that matches each
row’s `is_red`.

There are **11** base model types and **11** stratified-by-wine-type variants
(**22** entries in the registry total).

**Feature sets:** After the usual preprocessing (log-chlorides, z-scoring of continuous inputs), **OLS**, **multinomial logistic**, **glmnet** models (logistic and regression), and **KNN** each add **squared terms** for every continuous predictor (`*_sq`). **Random Forest** and **Naive Bayes** use the preprocessed features only (no quadratic expansion).

## Dataset

Based on the Portuguese "Vinho Verde" wine dataset
([Cortez et al., 2009](http://dx.doi.org/10.1016/j.dss.2009.05.016)).
The training set (`data/processed/train.csv`) contains 5,198 observations
with 11 chemical input features, a binary `is_red` indicator, and the
`quality` target (integer 3–9).

## Prerequisites

- **R >= 4.0** (tested on R 4.x)
- **GNU Make** (optional but recommended)

Install all R packages with:

```bash
make deps
# or equivalently:
Rscript src/install_packages.R
```

### R Packages

| Package | Purpose |
|---------|---------|
| ggplot2, reshape2 | Plotting and data reshaping |
| nnet | Multinomial logistic regression |
| glmnet | Ridge / Lasso / Elastic Net (classification & regression) |
| class | K-Nearest Neighbors |
| randomForest | Random Forest (classifier & regressor) |
| e1071 | Naive Bayes |
| caret | Stratified fold creation (`createFolds`) |

## Usage

Run the full pipeline (EDA through comparison plots):

```bash
make all
```

Or run individual steps:

```bash
make eda          # distribution plots + correlation heatmap + boxplots
make cv           # 5-fold stratified CV + save scale_params.rds for test prep
make plots        # CV accuracy & RMSE (single-row faceted figure)
make predict      # generate predictions on test.csv (when available)
make clean        # remove generated results and saved preprocessing
```

Each step can also be invoked directly:

```bash
Rscript analyze_distributions.R
Rscript scripts/01_eda.R
Rscript scripts/02_cv.R
Rscript scripts/04_plots.R
Rscript scripts/05_predict_test.R
```

## Project Layout

```
wine-quality-prediction/
├── data/
│   ├── raw/winequality.names       # Dataset documentation
│   └── processed/
│       ├── train.csv               # Training data
│       └── test.csv                # Test data (to be provided)
├── src/
│   ├── install_packages.R          # One-time dependency install
│   ├── helpers.R                   # Shared utilities (preprocess, metrics)
│   └── models.R                    # All model train/predict functions
├── scripts/
│   ├── 01_eda.R                    # Exploratory data analysis
│   ├── 02_cv.R                     # Cross-validation runner
│   ├── 04_plots.R                  # Comparison figures (CV only)
│   └── 05_predict_test.R           # Test set prediction
├── outputs/
│   ├── eda/                        # All EDA & distribution plots
│   ├── models/                     # scale_params.rds (from end of CV)
│   └── results/                    # cv_results.csv & comparison figure
├── analyze_distributions.R         # Per-variable distributions → outputs/eda
├── Makefile
├── PROJECT_PLAN.md
├── AI_USAGE.md
└── README.md
```

## Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **Accuracy** | Fraction of exact matches (rounded predictions for regressors) |
| **RMSE** | Root mean squared error on the 3–9 quality scale |

Metrics are reported for **5-fold stratified cross-validation** (mean ± SD across folds). Preprocessing parameters for applying the same scaling to `test.csv` are saved as `outputs/models/scale_params.rds` at the end of `scripts/02_cv.R`.

## Citation

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
*Modeling wine preferences by data mining from physicochemical properties.*
Decision Support Systems, Elsevier, 47(4):547–553, 2009.
