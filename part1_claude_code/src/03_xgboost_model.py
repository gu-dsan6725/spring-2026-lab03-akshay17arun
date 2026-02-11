import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output")
FIGURE_DPI: int = 150
RANDOM_STATE: int = 42
N_SPLITS: int = 5
N_ITER: int = 20
CLASS_NAMES: list[str] = ["Class_0", "Class_1", "Class_2"]

PARAM_DISTRIBUTIONS: dict[str, list] = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [round(0.01 + i * 0.01, 2) for i in range(29)],
    "subsample": [round(0.60 + i * 0.02, 2) for i in range(21)],
    "colsample_bytree": [round(0.60 + i * 0.02, 2) for i in range(21)],
    "min_child_weight": [1, 3, 5],
}


def _native(value: object) -> object:
    """Convert numpy scalars to Python native types for JSON serialization."""
    if hasattr(value, "item"):
        return value.item()
    return value


def _tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[dict, float]:
    """Run RandomizedSearchCV; save results to output/tuning_results.json."""
    base_model = XGBClassifier(
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=False,
    )
    search.fit(X_train, y_train)

    raw = search.cv_results_
    iter_results = [
        {
            "rank": int(raw["rank_test_score"][i]),
            "mean_f1_macro": round(float(raw["mean_test_score"][i]), 6),
            "std_f1_macro": round(float(raw["std_test_score"][i]), 6),
            "params": {k: _native(v) for k, v in raw["params"][i].items()},
        }
        for i in range(N_ITER)
    ]
    best_params = {k: _native(v) for k, v in search.best_params_.items()}
    best_score = float(search.best_score_)

    tuning_output = {
        "best_params": best_params,
        "best_cv_score_f1_macro": round(best_score, 6),
        "cv_results": sorted(iter_results, key=lambda x: x["rank"]),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "tuning_results.json"
    with open(path, "w") as f:
        json.dump(tuning_output, f, indent=2, default=str)

    logging.info(
        "Tuning complete. Best params:\n%s",
        json.dumps(best_params, indent=2, default=str),
    )
    logging.info("Best CV F1 macro: %.4f", best_score)
    logging.info("Tuning results saved to %s", path)
    return best_params, best_score


def _plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> None:
    """Save normalized confusion matrix to output/confusion_matrix.png."""
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title("Normalized Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logging.info("Saved confusion matrix to %s", path)


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> tuple[XGBClassifier, dict]:
    """Tune hyperparameters, run 5-fold CV, train final model, evaluate on test set."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_params, best_cv_score = _tune_hyperparameters(X_train, y_train)

    cv_model = XGBClassifier(
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        **best_params,
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        cv_model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro",
        },
        return_train_score=False,
    )

    cv_metrics: dict[str, dict[str, float]] = {
        metric: {
            "mean": round(float(cv_scores[f"test_{metric}"].mean()), 4),
            "std": round(float(cv_scores[f"test_{metric}"].std()), 4),
        }
        for metric in ["accuracy", "precision", "recall", "f1"]
    }
    logging.info(
        "5-fold CV metrics (best params):\n%s",
        json.dumps(cv_metrics, indent=2, default=str),
    )

    final_model = XGBClassifier(
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        verbosity=0,
        **best_params,
    )
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    test_metrics: dict[str, float] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="macro")), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="macro")), 4),
        "f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
    }
    logging.info(
        "Test set metrics:\n%s",
        json.dumps(test_metrics, indent=2, default=str),
    )

    _plot_confusion_matrix(y_test, y_pred, OUTPUT_DIR)

    metrics = {
        "best_params": best_params,
        "best_cv_score_f1_macro": round(best_cv_score, 4),
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
    }
    return final_model, metrics
