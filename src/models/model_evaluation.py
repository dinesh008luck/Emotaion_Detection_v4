import numpy as np
import pandas as pd
import pickle
import json
import logging
import sys
import os
from typing import Tuple, Dict
import yaml
from dvclive import Live

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import GradientBoostingClassifier
from src.visualization.visualize import save_confusion_matrix


# ---------------- LOGGING CONFIG ---------------- #
def setup_logger(name: str = "model_evaluation") -> logging.Logger:
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

def load_params(params_path: str = "params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)["model_evaluation"]

def load_model(model_path: str) -> GradientBoostingClassifier:
    """
    Load a trained model from a pickle file.

    Args:
        model_path: Path to the saved .pkl model file.

    Returns:
        Loaded classifier object.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the loaded object is not a sklearn estimator.
    """
    logger.info("Loading model from: %s", model_path)

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    if not hasattr(clf, "predict"):
        raise ValueError(
            f"Loaded object from '{model_path}' is not a valid sklearn estimator."
        )

    logger.debug("Model loaded successfully: %s", type(clf).__name__)
    return clf


def load_data(test_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the test feature matrix and labels from a CSV file.

    Expects all columns except the last to be features, and the last
    column to be the label.

    Args:
        test_path: Path to the test CSV file.

    Returns:
        A tuple of (X_test, y_test) as numpy arrays.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has fewer than 2 columns or contains NaNs.
    """
    logger.info("Loading test data from: %s", test_path)
    test_df = pd.read_csv(test_path)

    if test_df.shape[1] < 2:
        raise ValueError(
            f"Test data must have at least 2 columns (features + label), "
            f"got: {test_df.shape[1]}"
        )

    if test_df.isnull().any().any():
        null_count = test_df.isnull().sum().sum()
        logger.warning(
            "%d NaN value(s) found in test data. Filling with 0.", null_count
        )
        test_df = test_df.fillna(0)

    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    logger.debug(
        "Test data loaded. Features shape: %s, Labels shape: %s",
        X_test.shape,
        y_test.shape,
    )
    return X_test, y_test


def evaluate_model(
    clf: GradientBoostingClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Generate predictions and compute evaluation metrics.

    Metrics computed: accuracy, precision, recall, f1, AUC-ROC.
    Also logs the confusion matrix and full classification report.

    Args:
        clf: Fitted classifier with predict and predict_proba methods.
        X_test: Test feature matrix.
        y_test: True test labels.

    Returns:
        Dictionary of metric names to rounded float values.

    Raises:
        ValueError: If the classifier does not support predict_proba (needed for AUC).
    """
    logger.info("Running predictions on test data...")

    if not hasattr(clf, "predict_proba"):
        raise ValueError(
            f"{type(clf).__name__} does not support predict_proba, "
            "which is required for AUC-ROC computation."
        )

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    metrics: Dict[str, float] = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_pred_proba), 4),
    }

    # Log a human-readable summary
    logger.info("--- Evaluation Metrics ---")
    for name, value in metrics.items():
        logger.info("  %-12s: %.4f", name, value)

    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:\n%s", cm)

    report = classification_report(y_test, y_pred, zero_division=0)
    logger.debug("Classification Report:\n%s", report)

    np.save("reports/confusion_matrix.npy", cm)
    logger.info("Confusion matrix array saved.")
    
    return metrics


def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics: Dictionary of metric names and values.
        output_path: Destination file path for the JSON output.

    Raises:
        OSError: If the directory cannot be created or the file cannot be written.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info("Metrics saved to: %s", output_path)


def main() -> None:
    """Run the end-to-end model evaluation pipeline."""
    logger.info("Starting model evaluation pipeline...")

    try:
        
        params = load_params()

        clf = load_model(params["model_path"])
        X_test, y_test = load_data(params["test_data_path"])

        metrics = evaluate_model(clf, X_test, y_test)

        save_metrics(metrics, params["metrics_path"])

        # Dvclive
        with Live () as live:
            for key, value in metrics.items():
                live.log_metric(key, value)

        logger.info("Pipeline completed successfully.")

    except FileNotFoundError as e:
        logger.error("Required file not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Validation error: %s", e)
        sys.exit(1)
    except Exception:
        logger.exception("Pipeline failed due to an unexpected error.")
        sys.exit(1)

if __name__ == "__main__":
    main()