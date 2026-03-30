import numpy as np
import pandas as pd
import os
import sys
import yaml
import logging
import pickle
from typing import Tuple
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer


# ---------------- LOGGING CONFIG ---------------- #
def setup_logger(name: str = "build_feature") -> logging.Logger:
    """Set up and return a configured logger."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("app.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()
# ------------------------------------------------ #


def load_params(params_path: Path | str) -> int:
    """
    Load feature engineering parameters from a YAML file.

    Args:
        params_path: Path to the YAML configuration file.

    Returns:
        max_features value for CountVectorizer.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If required keys are missing.
        ValueError: If max_features is not a positive integer.
    """
    logger.info("Loading parameters from: %s", params_path)

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    try:
        max_features = params["build_features"]["max_features"]
    except KeyError as e:
        logger.error("Missing key in YAML file: %s", e)
        raise

    if not isinstance(max_features, int) or max_features <= 0:
        raise ValueError(
            f"max_features must be a positive integer, got: {max_features}"
        )

    logger.debug("Loaded max_features=%d", max_features)
    return max_features


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load processed train and test CSV files.

    Args:
        train_path: Path to the processed training CSV.
        test_path: Path to the processed test CSV.

    Returns:
        A tuple of (train_df, test_df).

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If required columns are missing.
    """
    logger.info("Loading processed train data from: %s", train_path)
    train_df = pd.read_csv(train_path)

    logger.info("Loading processed test data from: %s", test_path)
    test_df = pd.read_csv(test_path)

    for name, df in [("train", train_df), ("test", test_df)]:
        missing = {"content", "sentiment"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing column(s) in {name} data: {missing}")

    logger.debug(
        "Loaded train shape: %s, test shape: %s", train_df.shape, test_df.shape
    )
    return train_df, test_df


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature arrays and labels from DataFrames.

    Fills NaN content values with empty strings before extraction.

    Args:
        train_df: Processed training DataFrame.
        test_df: Processed test DataFrame.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as numpy arrays.
    """
    # Fill NaN in content only — avoid silently masking label issues
    train_df = train_df.copy()
    test_df = test_df.copy()

    null_train = train_df["content"].isna().sum()
    null_test = test_df["content"].isna().sum()
    if null_train or null_test:
        logger.warning(
            "Filling %d train and %d test NaN content values with empty strings.",
            null_train,
            null_test,
        )

    train_df["content"] = train_df["content"].fillna("")
    test_df["content"] = test_df["content"].fillna("")

    X_train = train_df["content"].values
    y_train = train_df["sentiment"].values
    X_test = test_df["content"].values
    y_test = test_df["sentiment"].values

    logger.debug(
        "Feature arrays ready. Train: %d samples, Test: %d samples.",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test


def build_bow_features(
    X_train: np.ndarray, X_test: np.ndarray, max_features: int
) -> Tuple[np.ndarray, np.ndarray, CountVectorizer]:
    """
    Fit a Bag-of-Words CountVectorizer on training data and transform both splits.

    The vectorizer is fit only on training data to prevent data leakage.

    Args:
        X_train: Training text array.
        X_test: Test text array.
        max_features: Maximum vocabulary size for CountVectorizer.

    Returns:
        Tuple of (X_train_bow, X_test_bow, fitted_vectorizer).
    """
    logger.info("Fitting CountVectorizer with max_features=%d ...", max_features)

    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(X_train).toarray()
    X_test_bow = vectorizer.transform(X_test).toarray()

    logger.debug(
        "BoW feature matrix shapes — Train: %s, Test: %s",
        X_train_bow.shape,
        X_test_bow.shape,
    )
    return X_train_bow, X_test_bow, vectorizer


def save_features(
    data_path: str,
    X_train_bow: np.ndarray,
    y_train: np.ndarray,
    X_test_bow: np.ndarray,
    y_test: np.ndarray,
    vectorizer: CountVectorizer,
) -> None:
    """
    Save BoW feature DataFrames and the fitted vectorizer to disk.

    Args:
        data_path: Directory to save output files.
        X_train_bow: Training BoW feature matrix.
        y_train: Training labels.
        X_test_bow: Test BoW feature matrix.
        y_test: Test labels.
        vectorizer: Fitted CountVectorizer to persist.

    Raises:
        OSError: If the directory cannot be created or files cannot be written.
    """
    logger.info("Saving feature data to: %s", data_path)
    os.makedirs(data_path, exist_ok=True)

    train_df = pd.DataFrame(X_train_bow)
    train_df["label"] = y_train

    test_df = pd.DataFrame(X_test_bow)
    test_df["label"] = y_test

    train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)

    # Persist the fitted vectorizer so it can be reused at inference time
    vectorizer_path = os.path.join(data_path, "vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    logger.info("Feature data and vectorizer saved successfully.")
    logger.debug(
        "Train shape: %s, Test shape: %s, Vectorizer: %s",
        train_df.shape,
        test_df.shape,
        vectorizer_path,
    )


def main() -> None:
    """Run the end-to-end feature engineering pipeline."""
    logger.info("Starting feature engineering pipeline...")

    try:
        # Get project root directory
        project_root = Path(__file__).resolve().parents[2]

        params_path = project_root / "params.yaml"
        max_features = load_params(params_path)

        train_path = project_root / "data" / "processed" / "train_processed.csv"
        test_path = project_root / "data" / "processed" / "test_processed.csv"

        train_data, test_data = load_data(
            train_path=str(train_path),
            test_path=str(test_path),
        )

        X_train, X_test, y_train, y_test = prepare_features(train_data, test_data)

        X_train_bow, X_test_bow, vectorizer = build_bow_features(
            X_train, X_test, max_features
        )

        feature_path = project_root / "data" / "feature"
        save_features(
            str(feature_path),
            X_train_bow,
            y_train,
            X_test_bow,
            y_test,
            vectorizer,
        )

        logger.info("Pipeline completed successfully.")

    except Exception:
        logger.exception("Pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()