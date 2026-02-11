import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output")
FIGURE_DPI: int = 150
TOP_N_FEATURES: int = 15


def _plot_feature_importances(
    model: XGBClassifier,
    feature_names: list[str],
    output_dir: Path,
) -> list[tuple[str, float]]:
    """Save top-N feature importance bar chart; return all pairs sorted by importance."""
    importances = model.feature_importances_
    pairs = sorted(
        zip(feature_names, importances, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    top = pairs[:TOP_N_FEATURES]
    names = [p[0] for p in top]
    values = [p[1] for p in top]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names[::-1], values[::-1], edgecolor="black")
    ax.set_title(f"Top {TOP_N_FEATURES} Feature Importances (XGBoost)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = output_dir / "feature_importances.png"
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logging.info("Saved feature importances plot to %s", path)
    return pairs


def generate_report(
    model: XGBClassifier,
    metrics: dict,
    feature_names: list[str],
) -> None:
    """Generate Markdown evaluation report and feature importance chart."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    importance_pairs = _plot_feature_importances(model, feature_names, OUTPUT_DIR)

    cv = metrics["cv_metrics"]
    test = metrics["test_metrics"]
    best_params = metrics.get("best_params", {})
    best_cv_score = metrics.get("best_cv_score_f1_macro", "N/A")

    lines: list[str] = [
        "# Wine Classification Pipeline — Evaluation Report\n",
        "\n## 1. Dataset Overview\n",
        "- **Dataset:** sklearn `load_wine` (UCI Wine Recognition)\n",
        "- **Samples:** 178 total (3 wine cultivars)\n",
        "- **Features:** 13 original + 3 engineered = 16 total\n",
        "- **Classes:** Class 0 (59 samples), Class 1 (71 samples), Class 2 (48 samples)\n",
        "- **Split:** 80% train / 20% test, stratified by class\n",
        "\n## 2. EDA Findings\n",
        "- Class distribution is moderately imbalanced (Class 2 smallest at 48 samples).\n",
        "- Strong positive correlations: `flavanoids` ↔ `total_phenols` (r≈0.86),\n"
        "  `flavanoids` ↔ `od280/od315_of_diluted_wines` (r≈0.79).\n",
        "- Notable negative correlations: `color_intensity` ↔ `hue`,\n"
        "  `nonflavanoid_phenols` ↔ `flavanoids`.\n",
        "- IQR-based outliers detected; most prominent in `proline` and `total_phenols`.\n",
        "\n## 3. Feature Engineering\n",
        "| Feature | Description |\n",
        "|---|---|\n",
        "| `alcohol_to_malic_ratio` | Alcohol / malic acid — fermentation character |\n",
        "| `total_phenols_flavanoids` | Sum of total phenols + flavanoids — phenolic richness |\n",
        "| `color_intensity_hue_interaction` | color_intensity × hue — color profile |\n",
        "\nAll features standardized with `StandardScaler` (fit on train set only).\n",
        "\n## 4. Hyperparameter Tuning\n",
        "- **Method:** `RandomizedSearchCV`, 20 iterations, 5-fold stratified CV, F1 macro\n",
        f"- **Best CV F1 macro:** {best_cv_score}\n",
        f"\n**Best parameters:**\n```json\n{json.dumps(best_params, indent=2, default=str)}\n```\n",
        "\n## 5. Cross-Validation Results (5-fold, best params)\n",
        "| Metric | Mean | Std |\n",
        "|---|---|---|\n",
    ]
    for m in ["accuracy", "precision", "recall", "f1"]:
        lines.append(f"| {m.capitalize()} | {cv[m]['mean']:.4f} | ±{cv[m]['std']:.4f} |\n")

    lines += [
        "\n## 6. Test Set Results\n",
        "| Metric | Score |\n",
        "|---|---|\n",
    ]
    for m in ["accuracy", "precision", "recall", "f1"]:
        lines.append(f"| {m.capitalize()} | {test[m]:.4f} |\n")

    lines += [
        "\n## 7. Top 10 Feature Importances\n",
        "| Rank | Feature | Importance |\n",
        "|---|---|---|\n",
    ]
    for rank, (name, score) in enumerate(importance_pairs[:10], start=1):
        lines.append(f"| {rank} | `{name}` | {score:.4f} |\n")

    lines += [
        "\n## 8. Recommendations\n",
        "1. **Top features** (e.g., `flavanoids`, `proline`, `color_intensity`) are the strongest\n"
        "   discriminators — consider trimming to the top 8 for a leaner production model.\n",
        "2. **Engineered features** (especially `total_phenols_flavanoids`) add signal;\n"
        "   retain them in production.\n",
        "3. **Class 2** is the smallest class — track per-class recall in production\n"
        "   to detect degradation early.\n",
        "4. A CV–test F1 gap < 2pp indicates good generalization; a larger gap warrants stronger\n"
        "   regularization (increase `min_child_weight`, reduce `subsample`).\n",
        "5. **Next steps:** collect more Class 2 samples or apply SMOTE oversampling if per-class\n"
        "   recall degrades in a deployed setting.\n",
    ]

    path = OUTPUT_DIR / "report.md"
    with open(path, "w") as f:
        f.writelines(lines)
    logging.info("Evaluation report saved to %s", path)
