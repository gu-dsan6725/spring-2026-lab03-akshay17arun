import logging
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.datasets import load_wine

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR: Path = Path("output")
PLOTS_DIR: Path = OUTPUT_DIR / "plots"


def _setup_directories():
    """Ensure output directories exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> pl.DataFrame:
    """Load Wine dataset and convert to Polars DataFrame."""
    wine = load_wine(as_frame=True)
    df = pl.from_pandas(wine.frame)
    logging.info(f"Loaded Wine dataset with shape: {df.shape}")
    return df


def _save_summary_stats(df: pl.DataFrame) -> None:
    """Compute and log summary statistics."""
    summary = df.describe()
    logging.info("Summary Statistics:\n%s", summary)

    # Save to file for record
    with open(OUTPUT_DIR / "summary_statistics.txt", "w") as f:
        f.write(str(summary))


def _plot_distributions(df: pl.DataFrame) -> None:
    """Generate distribution plots for all features."""
    features = [col for col in df.columns if col != "target"]

    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(features, 1):
        plt.subplot(4, 4, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f"Distribution of {feature}")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "distributions.png")
    plt.close()
    logging.info("Saved distribution plots to output/plots/distributions.png")


def _plot_correlation_heatmap(df: pl.DataFrame) -> None:
    """Generate correlation heatmap."""
    # Compute correlation matrix
    corr_matrix = df.corr()

    # Convert to pandas for seaborn
    # (Polars support in seaborn is improving but pandas is safer for heatmaps)
    # Using .to_pandas() as seaborn expects numpy/pandas for heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix.to_pandas(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()
    logging.info("Saved correlation heatmap to output/plots/correlation_heatmap.png")


def _plot_class_balance(df: pl.DataFrame) -> None:
    """Generate class balance plot."""
    target_counts = df["target"].value_counts().sort("target")

    plt.figure(figsize=(8, 6))
    sns.barplot(x=target_counts["target"], y=target_counts["count"])
    plt.title("Class Balance")
    plt.xlabel("Wine Class")
    plt.ylabel("Count")
    plt.savefig(PLOTS_DIR / "class_balance.png")
    plt.close()
    logging.info("Saved class balance plot to output/plots/class_balance.png")


def _detect_outliers(df: pl.DataFrame) -> None:
    """Detect outliers using IQR method and log them."""
    outliers_report = []
    features = [col for col in df.columns if col != "target"]

    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df.filter((pl.col(feature) < lower_bound) | (pl.col(feature) > upper_bound))
        num_outliers = outliers.height

        if num_outliers > 0:
            outliers_report.append(
                f"{feature}: {num_outliers} outliers detected ("
                f"Limits: {lower_bound:.2f}, {upper_bound:.2f})"
            )

    logging.info("Outlier Detection Report:\n" + "\n".join(outliers_report))

    with open(OUTPUT_DIR / "outliers.txt", "w") as f:
        f.write("\n".join(outliers_report))


def run_eda() -> None:
    """Execute the full EDA pipeline."""
    _setup_directories()
    df = _load_data()

    _save_summary_stats(df)
    _plot_distributions(df)
    _plot_correlation_heatmap(df)
    _plot_class_balance(df)
    _detect_outliers(df)
    logging.info("EDA completed successfully.")


if __name__ == "__main__":
    run_eda()
