import logging

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2


def _add_derived_features(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Add 3 engineered features capturing key wine chemistry interactions."""
    return df.with_columns(
        [
            (pl.col("alcohol") / pl.col("malic_acid")).alias("alcohol_to_malic_ratio"),
            (pl.col("total_phenols") + pl.col("flavanoids")).alias("total_phenols_flavanoids"),
            (pl.col("color_intensity") * pl.col("hue")).alias("color_intensity_hue_interaction"),
        ]
    )


def build_features(
    df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Engineer features, scale, and return stratified train/test splits."""
    df_eng = _add_derived_features(df)

    feature_cols: list[str] = [c for c in df_eng.columns if c != "target"]
    X = df_eng.select(feature_cols).to_numpy()
    y = df_eng["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logging.info(
        "Feature engineering complete: %d features, train=%d, test=%d",
        len(feature_cols),
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test, feature_cols
