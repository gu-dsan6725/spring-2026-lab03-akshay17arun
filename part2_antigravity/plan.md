# Wine Classification Pipeline Plan

## Goal
Build a robust machine learning pipeline for classifying Wine varieties using the Wine dataset from scikit-learn. The pipeline will include Exploratory Data Analysis (EDA), Feature Engineering, XGBoost training with cross-validation, and comprehensive evaluation.

## Directory Structure
```
part2_antigravity/
├── plan.md
├── pyproject.toml
├── uv.lock
├── src/
│   ├── __init__.py
│   ├── eda.py          # EDA scripts
│   ├── features.py     # Feature engineering
│   ├── train.py        # Model training
│   └── evaluate.py     # Evaluation and reporting
└── output/
    ├── plots/          # Generated plots
    └── report.txt      # Final evaluation report
```

## Step 1: Environment Setup
- Initialize `uv` project.
- Add dependencies: `polars`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `ruff`, `pytest`, `pandas` (if needed for plotting compatibility, though polars is preferred or conversion).

## Step 2: Exploratory Data Analysis (EDA)
**Script:** `src/eda.py`
- **Load Data:** Use `sklearn.datasets.load_wine(as_frame=True)` or convert numpy to Polars DataFrame.
- **Summary Statistics:** Compute mean, median, std, min, max for all features.
- **Visualizations:**
    - Distribution plots (histograms) for all features -> `output/distributions.png`.
    - Correlation heatmap -> `output/correlation_heatmap.png`.
    - Class balance bar chart -> `output/class_balance.png`.
- **Outlier Detection:** use IQR or Z-score to identify potential outliers and log them.

## Step 3: Feature Engineering
**Script:** `src/features.py`
- **Derived Features:**
    1. `proline_magnesium_ratio = proline / magnesium`
    2. `phenols_flavanoids_ratio = total_phenols / flavanoids`
    3. `color_hue_ratio = color_intensity / hue`
- **Preprocessing:**
    - Standard Scaling (Z-score normalization) for continuous variables.
- **Splitting:**
    - Stratified Train/Test split (e.g., 80/20).

## Step 4: Model Training
**Script:** `src/train.py`
- **Model:** XGBoost Classifier (`XGBClassifier`).
- **Validation:** 5-fold Stratified Cross-Validation.
- **Metrics:** Track Accuracy, Precision (weighted), Recall (weighted), F1 (weighted) during CV.

## Step 5: Evaluation & Reporting
**Script:** `src/evaluate.py`
- **Final Evaluation:** Train on full training set, evaluate on held-out test set.
- **Confusion Matrix:** Generate and save `output/confusion_matrix.png`.
- **Feature Importance:** Plot Gain/Weight importance -> `output/feature_importance.png`.
- **Report:** Compile all metrics, best/worst features, and recommendations into `output/report.txt`.

## Usage
Run the pipeline via:
```bash
uv run python src/eda.py
uv run python src/features.py
uv run python src/train.py
uv run python src/evaluate.py
```
