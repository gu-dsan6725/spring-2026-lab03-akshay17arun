import importlib.util
import logging
from pathlib import Path
from types import ModuleType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

SRC_DIR: Path = Path(__file__).parent

# Maps logical module names to their filenames in src/.
# Update values here if files are renamed.
MODULE_FILES: dict[str, str] = {
    "eda": "01_eda.py",
    "features": "02_feature_engineering.py",
    "train": "03_xgboost_model.py",
    "report": "report.py",
}


def _load_module(
    module_name: str,
    filename: str,
) -> ModuleType:
    """Load a module from a file path, supporting numeric-prefixed filenames."""
    path = SRC_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def main() -> None:
    """Run the full Wine classification pipeline end-to-end."""
    eda_mod = _load_module("eda", MODULE_FILES["eda"])
    features_mod = _load_module("features", MODULE_FILES["features"])
    train_mod = _load_module("train", MODULE_FILES["train"])
    report_mod = _load_module("report", MODULE_FILES["report"])

    logging.info("=== Step 1: EDA ===")
    df = eda_mod.run_eda()

    logging.info("=== Step 2: Feature Engineering ===")
    X_train, X_test, y_train, y_test, feature_names = features_mod.build_features(df)

    logging.info("=== Step 3: Hyperparameter Tuning + Model Training ===")
    model, metrics = train_mod.train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    )

    logging.info("=== Step 4: Evaluation Report ===")
    report_mod.generate_report(model, metrics, feature_names)

    logging.info("Pipeline complete. All outputs saved to output/")


if __name__ == "__main__":
    main()
