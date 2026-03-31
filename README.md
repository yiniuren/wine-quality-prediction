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

## Usage

### EDA, cross-validation, and plots

```bash
make eda          # distribution plots + correlation heatmap + boxplots
make cv           # 5-fold CV; saves cv_results.csv, scale_params.rds, and selected_models.rds (two winners — see below)
make plots        # CV accuracy & RMSE (single-row faceted figure)
make clean        # remove generated results and saved preprocessing
```

### Test-set prediction

Requires **`make cv`** first (for `selected_models.rds` and preprocessing parameters). Runs only the two CV-selected models on `test.csv`.

```bash
make predict
```

### Full pipeline (EDA through comparison plots)

Runs **`eda`**, **`cv`**, and **`plots`**. Does **not** run **`predict`**.

```bash
make all
```

### Direct `Rscript` invocations

```bash
Rscript scripts/01_eda.R
Rscript scripts/02_cv.R
Rscript scripts/04_plots.R
Rscript scripts/05_predict_test.R
```

## Project layout

```
wine-quality-prediction/
├── data/
│   ├── raw/
│   │   └── winequality.names
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── src/
│   ├── install_packages.R
│   ├── helpers.R              # preprocess, metrics
│   └── models.R               # train/predict for all model types
├── scripts/
│   ├── 01_eda.R
│   ├── 02_cv.R
│   ├── 04_plots.R
│   └── 05_predict_test.R
├── outputs/
│   ├── eda/                   # EDA figures and quality_distribution.csv
│   ├── models/                # scale_params.rds, selected_models.rds, per-model *.rds from CV
│   └── results/               # cv_results.csv, selected_models.csv, model_comparison.png;
│                              # test_predictions*.csv/txt, test_metrics.csv after predict
├── Makefile
├── PROJECT_PLAN.md
├── .gitignore
└── README.md
```

## Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **Accuracy** | Fraction of exact matches (rounded predictions for regressors) |
| **RMSE** | Root mean squared error on the 3–9 quality scale |

**RMSE** is reported for reference only; **accuracy** is the main metric because quality is treated as a **classification** problem (discrete levels 3–9), not a continuous regression target.

Metrics are reported for **5-fold stratified cross-validation** (mean ± SD across folds). Preprocessing parameters for applying the same scaling to `test.csv` are saved as `outputs/models/scale_params.rds` at the end of `scripts/02_cv.R`.

### CV selection for test prediction

After CV, **`scripts/02_cv.R`** picks two models from **`mean_cv_accuracy`** (tie-break: lower **`mean_cv_rmse`**), saves **`outputs/models/selected_models.rds`** and **`outputs/results/selected_models.csv`**:

1. **Best overall** — best among all models.
2. **Best non–Random Forest** — best among models whose name does not start with `Random Forest` (tree ensembles excluded from this slot).

**`make predict`** (or `Rscript scripts/05_predict_test.R`) requires **`selected_models.rds`** from a prior **`make cv`**. It refits only those two models and writes:

- **`outputs/results/test_predictions.csv`** — columns `row`, **`Best Performance Model`**, **`Best Performance Model We Have Learned`** (integer predictions).
- **`outputs/results/test_predictions_labels.txt`** — human-readable model names and CV metrics for the two picks.

If `data/processed/test.csv` includes **`quality`**, **`outputs/results/test_metrics.csv`** contains **two rows** (accuracy and RMSE for those models only).
