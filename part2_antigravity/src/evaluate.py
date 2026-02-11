import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
PROCESSED_DATA_DIR: Path = Path("output/processed")
MODEL_DIR: Path = Path("output/models")
OUTPUT_DIR: Path = Path("output")
PLOTS_DIR: Path = OUTPUT_DIR / "plots"


def _setup_directories():
    """Ensure output directories exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_test_data() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load processed test data."""
    X_test = pl.read_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
    y_test = pl.read_parquet(PROCESSED_DATA_DIR / "y_test.parquet")
    logging.info(f"Loaded test data: X_test shape={X_test.shape}, y_test shape={y_test.shape}")
    return X_test, y_test


def _load_model() -> XGBClassifier:
    """Load the trained XGBoost model."""
    model_path = MODEL_DIR / "xgboost_model.json"
    clf = XGBClassifier()
    clf.load_model(model_path)
    logging.info(f"Loaded model from {model_path}")
    return clf


def _evaluate_model(
    clf: XGBClassifier, X_test: pl.DataFrame, y_test: pl.DataFrame
) -> Dict[str, float]:
    """Evaluate model on test set and return metrics."""
    y_pred = clf.predict(X_test.to_pandas())
    y_true = y_test.to_pandas().values.ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }

    logging.info(f"Test Set Metrics: {json.dumps(metrics, indent=2)}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()
    logging.info("Saved confusion matrix to output/plots/confusion_matrix.png")

    return metrics


def _plot_feature_importance(clf: XGBClassifier, feature_names: List[str]) -> None:
    """Plot feature importance."""
    # XGBoost stores feature importance by default, but labels might be f0, f1... unless set?
    # When loading from JSON, features names might be preserved if saved correctly,
    # but safe to set them or map them.
    # Actually, we can just use the booster's feature importance.

    importance = clf.feature_importances_

    # Create DataFrame for plotting
    fi_df = pl.DataFrame({"feature": feature_names, "importance": importance}).sort(
        "importance", descending=True
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=fi_df.to_pandas())
    plt.title("Feature Importance (Gain/Weight)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png")
    plt.close()

    logging.info("Saved feature importance plot to output/plots/feature_importance.png")

    # Save text report of best features
    logging.info(f"Top 5 Features:\n{fi_df.head(5)}")


def _generate_report(
    metrics: Dict[str, float], clf: XGBClassifier, feature_names: List[str]
) -> None:
    """Generate final text report."""
    importance = clf.feature_importances_
    fi_df = pl.DataFrame({"feature": feature_names, "importance": importance}).sort(
        "importance", descending=True
    )

    report = []
    report.append("Wine Classification Report")
    report.append("==========================")
    report.append("\nMetrics:")
    for k, v in metrics.items():
        report.append(f"{k.capitalize()}: {v:.4f}")

    report.append("\nTop 5 Features:")
    for row in fi_df.head(5).iter_rows(named=True):
        report.append(f"- {row['feature']}: {row['importance']:.4f}")

    report.append("\nRecommendations:")
    report.append("- Focus on collecting high-quality data for the top features.")
    report.append("- The Proline/Magnesium ratio and other derived features should be monitored.")

    with open(OUTPUT_DIR / "report.txt", "w") as f:
        f.write("\n".join(report))
    logging.info("Saved final report to output/report.txt")


def run_evaluation() -> None:
    """Execute evaluation pipeline."""
    _setup_directories()
    X_test, y_test = _load_test_data()
    clf = _load_model()

    metrics = _evaluate_model(clf, X_test, y_test)
    _plot_feature_importance(clf, X_test.columns)
    _generate_report(metrics, clf, X_test.columns)

    logging.info("Evaluation completed successfully.")


if __name__ == "__main__":
    run_evaluation()
