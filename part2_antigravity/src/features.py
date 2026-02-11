import logging
from pathlib import Path

import polars as pl
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

# Constants
OUTPUT_DIR: Path = Path("output")
PROCESSED_DATA_DIR: Path = OUTPUT_DIR / "processed"


def _setup_directories():
    """Ensure output directories exist."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> pl.DataFrame:
    """
    Load Wine dataset and convert to Polars DataFrame.
    """
    wine = load_wine(as_frame=True)
    df = pl.from_pandas(wine.frame)
    logging.info(f"Loaded Wine dataset with shape: {df.shape}")
    return df


def _create_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create derived features:
    1. proline_magnesium_ratio = proline / magnesium
    2. phenols_flavanoids_ratio = total_phenols / flavanoids
    3. color_hue_ratio = color_intensity / hue
    """
    df = df.with_columns(
        (pl.col("proline") / pl.col("magnesium")).alias("proline_magnesium_ratio"),
        (pl.col("total_phenols") / pl.col("flavanoids")).alias("phenols_flavanoids_ratio"),
        (pl.col("color_intensity") / pl.col("hue")).alias("color_hue_ratio"),
    )
    logging.info(f"Created derived features. New shape: {df.shape}")
    return df


def _perform_splitting_and_scaling(df: pl.DataFrame) -> None:
    """
    Split data into stratified train/test sets and apply standard scaling.
    Saves processed datasets to output/processed/.
    """
    target_col = "target"
    feature_cols = [col for col in df.columns if col != target_col]

    X = df.select(feature_cols)
    y = df.select(target_col)

    # Split using sklearn (convert to pandas/numpy for compatibility if needed,
    # but polars works with train_test_split effectively by passing numpy representation or wrapper)
    # Using to_pandas() for full compatibility with sklearn split
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_pandas(), y.to_pandas(), test_size=0.2, stratify=y.to_pandas(), random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to Polars
    X_train_pl = pl.from_numpy(X_train_scaled, schema=feature_cols)
    X_test_pl = pl.from_numpy(X_test_scaled, schema=feature_cols)
    y_train_pl = pl.from_pandas(y_train)
    y_test_pl = pl.from_pandas(y_test)

    # Save processed data
    X_train_pl.write_parquet(PROCESSED_DATA_DIR / "X_train.parquet")
    X_test_pl.write_parquet(PROCESSED_DATA_DIR / "X_test.parquet")
    y_train_pl.write_parquet(PROCESSED_DATA_DIR / "y_train.parquet")
    y_test_pl.write_parquet(PROCESSED_DATA_DIR / "y_test.parquet")

    logging.info("Saved processed datasets to output/processed/")
    logging.info(f"Train shape: {X_train_pl.shape}, Test shape: {X_test_pl.shape}")


def run_features() -> None:
    """Execute the feature engineering pipeline."""
    _setup_directories()
    df = _load_data()
    df = _create_derived_features(df)
    _perform_splitting_and_scaling(df)
    logging.info("Feature Engineering completed successfully.")


if __name__ == "__main__":
    run_features()
