"""Standalone analyze-data skill: full EDA on the Wine dataset."""

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output")
FIGURE_DPI: int = 150
N_COLS: int = 4
HIST_BINS: int = 20


def _summary_stats(
    df: pl.DataFrame,
    feature_cols: list[str],
) -> dict[str, dict[str, float]]:
    """Return mean/median/std/min/max for each numeric feature."""
    stats: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        s = df[col]
        stats[col] = {
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
        }
    return stats


def _missing_values(
    df: pl.DataFrame,
) -> tuple[dict[str, dict], int]:
    """Return per-column null counts/percentages and total null count."""
    missing = {
        col: {
            "count": int(df[col].null_count()),
            "pct": round(df[col].null_count() / df.height * 100, 2),
        }
        for col in df.columns
    }
    total = sum(v["count"] for v in missing.values())
    return missing, total


def _iqr_outliers(
    df: pl.DataFrame,
    feature_cols: list[str],
) -> dict[str, int]:
    """Count IQR-based outliers per feature."""
    outliers: dict[str, int] = {}
    for col in feature_cols:
        s = df[col]
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        n = s.filter((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).len()
        outliers[col] = int(n)
    return outliers


def _plot_distributions(
    df: pl.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    n_rows = (len(feature_cols) + N_COLS - 1) // N_COLS
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(16, n_rows * 3))
    axes_flat = np.array(axes).flatten()
    for i, col in enumerate(feature_cols):
        axes_flat[i].hist(df[col].to_numpy(), bins=HIST_BINS, edgecolor="black")
        axes_flat[i].set_title(col, fontsize=9)
    for j in range(len(feature_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Feature Distributions — Wine Dataset", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "eda_distributions.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logging.info("Saved eda_distributions.png")


def _plot_correlation_heatmap(
    df: pl.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> np.ndarray:
    corr = np.corrcoef(df.select(feature_cols).to_numpy().T)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        xticklabels=feature_cols,
        yticklabels=feature_cols,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Heatmap — Wine Dataset", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "eda_correlation_heatmap.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logging.info("Saved eda_correlation_heatmap.png")
    return corr


def run_analysis() -> None:
    """Execute all 8 steps of the analyze-data skill on the Wine dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    wine = load_wine()
    feature_cols: list[str] = list(wine.feature_names)
    df = pl.DataFrame({col: wine.data[:, i] for i, col in enumerate(feature_cols)}).with_columns(
        pl.Series("target", wine.target)
    )
    logging.info("Shape: %d rows × %d cols", df.height, df.width)

    # 2. Summary statistics
    stats = _summary_stats(df, feature_cols)
    logging.info("Summary statistics:\n%s", json.dumps(stats, indent=2))

    # 3. Missing values
    missing, total_missing = _missing_values(df)
    logging.info("Missing values (total=%d):\n%s", total_missing, json.dumps(missing, indent=2))

    # 4. Duplicate rows
    n_dupes = df.height - df.unique().height
    logging.info("Duplicate rows: %d", n_dupes)

    # 5. Distribution plots
    _plot_distributions(df, feature_cols, OUTPUT_DIR)

    # 6. Correlation heatmap
    corr = _plot_correlation_heatmap(df, feature_cols, OUTPUT_DIR)

    # 7. IQR outliers
    outliers = _iqr_outliers(df, feature_cols)
    logging.info("IQR outlier counts:\n%s", json.dumps(outliers, indent=2))

    # 8. Key findings summary
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            pairs.append((feature_cols[i], feature_cols[j], round(float(corr[i, j]), 3)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    class_counts = df["target"].value_counts().sort("target")
    findings = {
        "n_samples": df.height,
        "n_features": len(feature_cols),
        "missing_values": total_missing,
        "duplicate_rows": n_dupes,
        "class_counts": dict(
            zip(
                class_counts["target"].to_list(),
                class_counts["count"].to_list(),
                strict=False,
            )
        ),
        "total_outliers": sum(outliers.values()),
        "top_5_correlations": [{"feature_a": a, "feature_b": b, "r": r} for a, b, r in pairs[:5]],
        "most_outliers_feature": max(outliers, key=lambda k: outliers[k]),
    }
    logging.info("Key findings:\n%s", json.dumps(findings, indent=2, default=str))


if __name__ == "__main__":
    run_analysis()
