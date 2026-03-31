import numpy as np
import pandas as pd
import pickle
import yaml
import logging
import sys
import os
from typing import Tuple
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier


# ---------------- LOGGING CONFIG ---------------- #
def setup_logger(name: str = "model_building") -> logging.Logger:
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


def load_params(params_path: str) -> Tuple[int, float]:
    """
    Load model building hyperparameters from a YAML file.

    Args:
        params_path: Path to the YAML configuration file.

    Returns:
        A tuple of (n_estimators, learning_rate).

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        KeyError: If required keys are missing.
        ValueError: If parameter values are invalid.
    """
    logger.info("Loading parameters from: %s", params_path)

    params_path = Path(params_path)

    if not params_path.exists():
        raise FileNotFoundError(f"{params_path} does not exist")

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    try:
        n_estimators = params["model_building"]["n_estimators"]
        learning_rate = params["model_building"]["learning_rate"]
    except KeyError as e:
        logger.error("Missing key in YAML file: %s", e)
        raise

    if not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError(
            f"n_estimators must be a positive integer, got: {n_estimators}"
        )
    if not isinstance(learning_rate, (int, float)) or not (0 < learning_rate <= 1):
        raise ValueError(
            f"learning_rate must be a float in (0, 1], got: {learning_rate}"
        )

    logger.debug(
        "Loaded n_estimators=%d, learning_rate=%s", n_estimators, learning_rate
    )
    return n_estimators, learning_rate


def load_data(train_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the training feature matrix and labels from a CSV file.

    Expects all columns except the last to be features, and the last
    column to be the label.

    Args:
        train_path: Path to the training CSV file.

    Returns:
        A tuple of (X_train, y_train) as numpy arrays.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has fewer than 2 columns.
    """
    logger.info("Loading training data from: %s", train_path)
    train_df = pd.read_csv(train_path)

    if train_df.shape[1] < 2:
        raise ValueError(
            f"Training data must have at least 2 columns (features + label), "
            f"got: {train_df.shape[1]}"
        )

    if train_df.isnull().any().any():
        null_count = train_df.isnull().sum().sum()
        logger.warning(
            "%d NaN value(s) found in training data. Filling with 0.", null_count
        )
        train_df = train_df.fillna(0)

    if "label" not in train_df.columns:
       raise ValueError("Expected 'label' column in training data")

    X_train = train_df.drop(columns=["label"]).values
    y_train = train_df["label"].values

    logger.debug(
        "Training data loaded. Features shape: %s, Labels shape: %s",
        X_train.shape,
        y_train.shape,
    )
    return X_train, y_train


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int,
    learning_rate: float,
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting Classifier on the provided data.

    Args:
        X_train: Feature matrix.
        y_train: Target labels.
        n_estimators: Number of boosting stages.
        learning_rate: Shrinkage rate for each tree's contribution.

    Returns:
        Fitted GradientBoostingClassifier instance.
    """
    logger.info(
        "Training GradientBoostingClassifier — n_estimators=%d, learning_rate=%s",
        n_estimators,
        learning_rate,
    )

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42,
        max_depth= 5,  
    )
    clf.fit(X_train, y_train)

    train_accuracy = clf.score(X_train, y_train)
    logger.info("Training complete. Train accuracy: %.4f", train_accuracy)

    return clf


def save_model(clf: GradientBoostingClassifier, model_path: str) -> None:
    """
    Persist a trained model to disk using pickle.

    Args:
        clf: Fitted classifier to save.
        model_path: Destination file path (e.g. 'models/model.pkl').

    Raises:
        OSError: If the directory cannot be created or file cannot be written.
    """
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    logger.info("Model saved to: %s", model_path)


def main() -> None:
    """Run the end-to-end model building pipeline."""
    logger.info("Starting model building pipeline...")

    try:
        project_root = Path(__file__).resolve().parents[2]
        params_path = project_root / "params.yaml"

        n_estimators, learning_rate = load_params(str(params_path))

        X_train, y_train = load_data(
            train_path=os.path.join("data", "feature", "train_tfidf.csv")
        )

        clf = train_model(X_train, y_train, n_estimators, learning_rate)

        save_model(clf, model_path=os.path.join("models", "model.pkl"))

        logger.info("Pipeline completed successfully.")

    except FileNotFoundError as e:
        logger.error("Input file not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Validation error: %s", e)
        sys.exit(1)
    except Exception:
        logger.exception("Pipeline failed due to an unexpected error.")
        sys.exit(1)


if __name__ == "__main__":
    main()