import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

OUTPUT_DIR: Path = Path("output")
FIGURE_DPI: int = 150
HIST_BINS: int = 20
N_COLS: int = 4


def _compute_iqr_outliers(
    df: pl.DataFrame,
    feature_cols: list[str],
) -> dict[str, int]:
    """Count IQR-based outliers per feature column."""
    outlier_counts: dict[str, int] = {}
    for col in feature_cols:
        series = df[col]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = series.filter((series < lower) | (series > upper)).len()
        outlier_counts[col] = int(n_outliers)
    return outlier_counts


def _plot_distributions(
    df: pl.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    """Save histogram grid to output/distributions.png."""
    n_rows = (len(feature_cols) + N_COLS - 1) // N_COLS
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(16, n_rows * 3))
    axes_flat = np.array(axes).flatten()
    for i, col in enumerate(feature_cols):
        axes_flat[i].hist(df[col].to_numpy(), bins=HIST_BINS, edgecolor="black")
        axes_flat[i].set_title(col, fontsize=9)
    for j in range(len(feature_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Feature Distributions", fontsize=14, y=1.01)
    plt.tight_layout()
    path = output_dir / "distributions.png"
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logging.info("Saved distributions plot to %s", path)


def _plot_correlation_heatmap(
    df: pl.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    """Save Pearson correlation heatmap to output/correlation_heatmap.png."""
    matrix = np.corrcoef(df.select(feature_cols).to_numpy().T)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        matrix,
        xticklabels=feature_cols,
        yticklabels=feature_cols,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Pearson Correlation Heatmap", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = output_dir / "correlation_heatmap.png"
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logging.info("Saved correlation heatmap to %s", path)


def _plot_class_balance(
    df: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Save class balance bar chart to output/class_balance.png."""
    counts = df["target"].value_counts().sort("target")
    labels = [f"Class {v}" for v in counts["target"].to_list()]
    values = counts["count"].to_list()
    logging.info(
        "Class balance:\n%s",
        json.dumps(dict(zip(labels, values, strict=False)), indent=2),
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, edgecolor="black")
    ax.set_title("Class Balance (Wine Cultivars)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    path = output_dir / "class_balance.png"
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logging.info("Saved class balance plot to %s", path)


def _plot_boxplots(
    df: pl.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    """Save box plot grid to output/boxplots.png."""
    n_rows = (len(feature_cols) + N_COLS - 1) // N_COLS
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(16, n_rows * 3))
    axes_flat = np.array(axes).flatten()
    for i, col in enumerate(feature_cols):
        axes_flat[i].boxplot(df[col].to_numpy(), patch_artist=True)
        axes_flat[i].set_title(col, fontsize=9)
    for j in range(len(feature_cols), len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle("Feature Box Plots (Outlier Detection)", fontsize=14, y=1.01)
    plt.tight_layout()
    path = output_dir / "boxplots.png"
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logging.info("Saved boxplots to %s", path)


def run_eda() -> pl.DataFrame:
    """Load wine dataset, run full EDA, save all plots, and return Polars DataFrame."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wine = load_wine()
    feature_cols: list[str] = list(wine.feature_names)
    df = pl.DataFrame({col: wine.data[:, i] for i, col in enumerate(feature_cols)}).with_columns(
        pl.Series("target", wine.target)
    )

    logging.info("Loaded Wine dataset: %d rows, %d features", df.height, len(feature_cols))

    stats: dict[str, dict] = {}
    for col in feature_cols:
        s = df[col]
        stats[col] = {
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "q25": round(float(s.quantile(0.25)), 4),
            "q50": round(float(s.quantile(0.50)), 4),
            "q75": round(float(s.quantile(0.75)), 4),
        }
    logging.info("Summary statistics:\n%s", json.dumps(stats, indent=2, default=str))

    outlier_counts = _compute_iqr_outliers(df, feature_cols)
    logging.info(
        "IQR outlier counts per feature:\n%s",
        json.dumps(outlier_counts, indent=2, default=str),
    )

    _plot_distributions(df, feature_cols, OUTPUT_DIR)
    _plot_correlation_heatmap(df, feature_cols, OUTPUT_DIR)
    _plot_class_balance(df, OUTPUT_DIR)
    _plot_boxplots(df, feature_cols, OUTPUT_DIR)

    return df
