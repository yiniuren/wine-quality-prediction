# Wine Quality Prediction — Project Plan

## 1. Problem Overview

| Item | Detail |
|------|--------|
| **Goal** | Predict the `quality` score (0–10, discrete) of wine from physicochemical features |
| **Dataset** | 5,198 training observations, 13 columns (11 chemical inputs + `is_red` indicator + `quality` target) |
| **Evaluation** | Model will be scored on a held-out `test.csv` |
| **Nature** | Ordinal / multi-class classification (7 observed classes: 3–9, heavily imbalanced toward 5 and 6) |

### Target Distribution (Training Set)

| Quality | Count | Proportion |
|---------|------:|----------:|
| 3       |    20 |   0.4%    |
| 4       |   178 |   3.4%    |
| 5       | 1,701 |  32.7%    |
| 6       | 2,269 |  43.7%    |
| 7       |   867 |  16.7%    |
| 8       |   158 |   3.0%    |
| 9       |     5 |   0.1%    |

### Feature List

| # | Feature | Type |
|---|---------|------|
| 1 | fixed acidity | continuous |
| 2 | volatile acidity | continuous |
| 3 | citric acid | continuous |
| 4 | residual sugar | continuous |
| 5 | chlorides | continuous |
| 6 | free sulfur dioxide | continuous |
| 7 | total sulfur dioxide | continuous |
| 8 | density | continuous |
| 9 | pH | continuous |
| 10 | sulphates | continuous |
| 11 | alcohol | continuous |
| 12 | is_red | binary (0/1) |

---

## 2. Candidate Models

Most models treat `quality` as a **class label** (multi-class classification). **OLS** and the **regression-then-round** models treat `quality` as numeric, then (where needed) round to integers for classification-style metrics.

### 2.1 Baseline / Simple Models

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **OLS (Ordinary Least Squares)** | Linear regression with `quality` as a continuous outcome | Simplest linear structure; interpretable coefficients; use rounded predictions when reporting accuracy / confusion matrix |
| **Multinomial logistic regression** | Linear decision boundaries via softmax over quality classes | Strong discrete baseline; directly targets multi-class labels |

### 2.2 Regularized Linear Models (Classification)

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Ridge (L2) logistic regression** | Penalizes large coefficients with L2 norm | Helps when features are correlated (e.g., density, alcohol, SO₂-related variables) |
| **Lasso (L1) logistic regression** | L1 penalty; can zero out coefficients | Feature selection; shows which inputs matter most |
| **Elastic Net logistic regression** | Mix of L1 + L2 | Balances stability with sparsity when predictors are correlated |

### 2.3 Nearest-Neighbor Methods

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **K-Nearest Neighbors (KNN)** | Majority vote among the *k* closest training points | Non-parametric; captures local structure; needs scaled features |

### 2.4 Tree-Based Models

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Random Forest (classifier)** | Many de-correlated trees via bagging + random splits | Strong default for tabular data; handles nonlinearity and interactions |
| **XGBoost (classifier)** | Gradient boosted trees with regularization | Often very accurate on structured data; tune depth/learning rate modestly |
| **CatBoost (classifier)** | Gradient boosting with ordered boosting / categorical handling | Works well with minimal tuning; `is_red` fits naturally |

### 2.5 Probabilistic Models

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Naive Bayes (Gaussian)** | Class-conditional Gaussian features, independence assumption | Fast baseline; can work when linear separability is rough |

### 2.6 Ordinal-Aware Models

`quality` is **ordinal** (3 < 4 < … < 9). One model that respects ordering:

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Ordinal logistic regression (proportional odds)** | One coefficient vector + ordered cutpoints between classes | Treats quality as ordered categories instead of unrelated labels |

### 2.7 Regression-then-Round

Treat `quality` as continuous, predict with regression, then **round** to the nearest integer (clip to valid range if needed) for discrete metrics.

| Model | Key Idea | Why Try It |
|-------|----------|------------|
| **Ridge / Lasso regression → round** | Penalized linear regression, then discretize | Same “numeric quality” idea as OLS but with multicollinearity control |
| **Random Forest regressor → round** | RF predicts a continuous score, then round | Compare tree ensemble as regressor vs. as classifier |
| **XGBoost regressor → round** | Boosted trees for continuous outcome, then round | Same comparison for boosting |

*(OLS in §2.1 is the unpenalized linear regression baseline; use rounding when you want classification-style evaluation.)*

---

## 3. Project Structure

```
wine-quality-prediction/
├── data/
│   ├── raw/                        # Original data files
│   │   └── winequality.names       # Dataset documentation
│   └── processed/
│       ├── train.csv               # Training data (provided)
│       └── test.csv                # Test data (to be provided)
├── src/
│   ├── install_packages.R          # One-time R dependency install
│   ├── helpers.R                   # Preprocessing, CV folds, metrics
│   └── models.R                    # Model train/predict functions + registry
├── scripts/
│   ├── 01_eda.R                    # Exploratory data analysis (extra plots)
│   ├── 02_cv.R                     # Cross-validation
│   ├── 03_full_train.R             # Full training set fit & metrics
│   ├── 04_plots.R                  # Model comparison figure
│   └── 05_predict_test.R           # Test set predictions (when test.csv exists)
├── outputs/
│   ├── distributions/              # EDA plots & tables
│   ├── eda/                        # Correlation heatmap, boxplots, etc.
│   ├── models/                     # Saved model objects
│   └── results/                    # Comparison tables, final predictions
├── analyze_distributions.R         # Distribution plots for all variables
├── Makefile
├── PROJECT_PLAN.md                 # ← This file
├── AI_USAGE.md                     # AI usage documentation
└── .gitignore
```

---

## 4. Workflow (aligned with the reduced model list)

A compact path from EDA through model fitting to **side-by-side comparison** (no need to declare a single best model yet).

### Step 1 — EDA (mostly done)

- **Already in place:** distribution plots and `quality` bar chart under `outputs/distributions/`.
- **Still useful:** correlation heatmap, boxplots of predictors by `quality`, and red vs. white comparisons if not already explored.

**Chlorides and log transform:** Chlorides is typically **right-skewed** (long right tail, many small values). Logging (e.g. `log(chlorides)` or `log1p(chlorides)` if any zeros) can pull in the tail and make the variable easier for **OLS**, **Naive Bayes (Gaussian)**, and other methods that benefit from more symmetric inputs. It does **not** guarantee better test performance—**compare in cross-validation** with and without the log version (or replace raw chlorides with log-chlorides in the feature set, not both, to avoid collinearity). If you add it, apply the **same** definition on `train.csv` and `test.csv` and document it in the write-up.

### Step 2 — Preprocessing

- **Scaling:** Use consistent scaling (e.g. z-scores) for **KNN** and any implementation that is distance-sensitive; tree models are scale-invariant.
- **Optional:** Include `log1p(chlorides)` (or similar) as one engineered column if Step 1 supports it.
- **Validation:** Use **stratified k-fold CV** (e.g. k = 5 or 10) so rare quality levels appear in each fold.

### Step 3 — Fit and lightly tune candidates

Work through the families in §2 in a sensible order: **OLS** and **multinomial logistic** first, then **Ridge/Lasso/Elastic Net** logistic, **ordinal logistic**, **Naive Bayes**, **KNN** (tune `k`), **Random Forest / XGBoost / CatBoost** (small grid or defaults + a bit of tuning), then **regression-then-round** variants (**Ridge/Lasso**, **RF reg**, **XGBoost reg**) with rounding for discrete metrics.

- Keep a **single table**: model name, key settings, **CV accuracy**, **CV RMSE** (see §5 for definitions).

### Step 4 — Full training set metrics and comparison figures

- **Fit each model** on the **entire** training set (same hyperparameters you used in CV).
- On that same full training set, compute **training-set accuracy** and **training-set RMSE** for each model (these are in-sample / optimistic; they complement CV, not replace it).
- **Do not** need to pick a “winner” yet—goal is a clear visual comparison across models.

**Comparison plot(s):** Build figure(s) so each model can be compared on **accuracy** and **RMSE** in both settings:

| Setting | What to plot |
|--------|----------------|
| **Cross-validation** | Per model: CV accuracy (e.g. mean across folds) and CV RMSE (mean across folds). |
| **Full training set** | Per model: accuracy and RMSE computed on the full training data after refitting on all rows. |

Use a layout that keeps both metrics visible—for example **two panels** (one for CV, one for full training), each with **paired bars or points** per model: one series for **accuracy**, one for **RMSE**. Because accuracy (often 0–1 or 0–100%) and RMSE (error scale) live on different scales, use **dual y-axes** *or* **normalize** RMSE for display *or* **two separate subplots per panel** (accuracy vs. RMSE) with models on the x-axis—choose whichever is clearest for your report. The point is to **see every model’s accuracy and RMSE together** for CV and again for the full training run.

When you later have `test.csv`, apply the same preprocessing and save predictions; test metrics can follow the same accuracy + RMSE convention if labels are available.

---

## 5. Evaluation metrics (accuracy and RMSE only)

Use **only** these two metrics everywhere (CV folds, full training set, and test if applicable).

| Metric | Definition (for this project) |
|--------|--------------------------------|
| **Accuracy** | Fraction of observations where predicted `quality` equals true `quality`. For regressors and “regression → round,” use **rounded** predictions (clipped to valid scores) when computing accuracy. |
| **RMSE** | \(\sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}\) where \(y_i\) is true `quality` (numeric). Use **continuous** predictions \(\hat{y}_i\) for OLS and regression models. For **classifiers** (predicted integer class), \(\hat{y}_i\) is that predicted class treated as a number—so RMSE still measures typical error magnitude on the 0–10 quality scale. |

**CV:** Report **mean** accuracy and **mean** RMSE across folds (and optionally fold-wise SD in a table if useful).  
**Full training set:** Report a single accuracy and a single RMSE per model after refitting on all training rows.

---

## 6. Reminders

- **Imbalance:** Most wines are 5–6; accuracy alone can look high even when rare classes are weak—still useful for comparing models on the same scale.
- **Correlated predictors:** Regularized logistic and tree methods handle this better than plain OLS for stability.
- **Reproducibility:** Fix random seeds; save the exact preprocessing + model for test-time use.
