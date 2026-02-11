import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
PROCESSED_DATA_DIR: Path = Path("output/processed")
MODEL_DIR: Path = Path("output/models")


def _setup_directories():
    """Ensure output directories exist."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _load_train_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load processed training data."""
    X_train = pl.read_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    y_train = pl.read_parquet(PROCESSED_DATA_DIR / "y_train.parquet")
    logging.info(
        f"Loaded training data: X_train shape={X_train.shape}, y_train shape={y_train.shape}"
    )
    return X_train, y_train


def _perform_cross_validation(X: pl.DataFrame, y: pl.DataFrame) -> Dict[str, float]:
    """
    Perform 5-fold Stratified Cross-Validation using XGBoost.
    Returns dictionary of mean metrics.
    """
    clf = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=42)

    # Define metrics
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="weighted"),
        "recall": make_scorer(recall_score, average="weighted"),
        "f1": make_scorer(f1_score, average="weighted"),
    }

    # Convert to pandas/numpy for sklearn compatibility if needed?
    # XGBClassifier handles Polars, but cross_validate splits might input numpy to it?
    # Actually sklearn cross_validate passes pandas/numpy slices.
    # It's safer to pass pandas to cross_validate.

    cv_results = cross_validate(
        clf,
        X.to_pandas(),
        y.to_pandas().values.ravel(),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=scoring,
    )

    results = {}
    for metric in scoring:
        key = f"test_{metric}"
        mean_score = np.mean(cv_results[key])
        std_score = np.std(cv_results[key])
        results[metric] = mean_score
        logging.info(f"CV {metric.capitalize()}: {mean_score:.4f} (+/- {std_score:.4f})")

    return results


def _train_final_model(X: pl.DataFrame, y: pl.DataFrame) -> None:
    """Train model on full training set and save it."""
    clf = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False, random_state=42)
    clf.fit(X.to_pandas(), y.to_pandas().values.ravel())

    # Save model
    model_path = MODEL_DIR / "xgboost_model.json"
    clf.save_model(model_path)
    logging.info(f"Saved trained model to {model_path}")


def run_training() -> None:
    """Execute training pipeline."""
    _setup_directories()
    X_train, y_train = _load_train_data()

    logging.info("Starting 5-Fold Cross-Validation...")
    cv_metrics = _perform_cross_validation(X_train, y_train)

    logging.info("Training final model on full training set...")
    _train_final_model(X_train, y_train)

    logging.info("Training pipeline completed.")
    logging.info(f"CV Results: {json.dumps(cv_metrics, indent=2)}")


if __name__ == "__main__":
    run_training()
