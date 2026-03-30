import numpy as np
import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple
from pathlib import Path

# ---------------- LOGGING CONFIG ---------------- #
def setup_logger(log_file: str = "app.log") -> logging.Logger:
    logger = logging.getLogger("data_injection")

    # Prevent duplicate logs if script runs multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()

# ------------------------------------------------ #


def load_params(params_path: str) -> Tuple[float, int]:
    try:
        logger.info("Loading parameters from YAML file...")

                # Convert to Path object
        params_path = Path(params_path)

        # If file not found in current directory,
        # try locating it from project root
        if not params_path.exists():
            project_root = Path(__file__).resolve().parents[2]
            params_path = project_root / params_path

        if not params_path.exists():
            raise FileNotFoundError(f"{params_path} does not exist.")

        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        test_size = params["data_preprocess"]["test_size"]
        random_state = params["data_preprocess"]["random_state"]

        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        logger.debug(f"Parameters loaded: test_size={test_size}, random_state={random_state}")
        return test_size, random_state

    except Exception:
        logger.exception("Error in load_params")
        raise


def data_read(url: str) -> pd.DataFrame:
    try:
        logger.info("Reading dataset...")

        df = pd.read_csv(url)

        if df.empty:
            raise ValueError("Dataset is empty.")

        logger.debug(f"Dataset loaded successfully with shape: {df.shape}")
        return df

    except Exception:
        logger.exception("Error reading data")
        raise


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("Processing data...")

        required_columns = {"tweet_id", "sentiment"}

        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()

        # Drop unnecessary column
        df.drop(columns=["tweet_id"], inplace=True)

        # Filter required sentiments
        final_df = df.loc[df["sentiment"].isin(["happiness", "sadness"])].copy()

        if final_df.empty:
            raise ValueError("No matching sentiment rows found.")

        # Encoding
        final_df["sentiment"] = (
                final_df["sentiment"]
                .map({"happiness": 1, "sadness": 0})
                .astype("int64")
                )

        logger.debug(f"Processed dataset shape: {final_df.shape}")
        return final_df

    except Exception:
        logger.exception("Error in process_data")
        raise


def save_data(data_path: str,
              train_data: pd.DataFrame,
              test_data: pd.DataFrame) -> None:
    try:
        logger.info("Saving train and test data...")

        os.makedirs(data_path, exist_ok=True)

        train_path = os.path.join(data_path, "train.csv")
        test_path = os.path.join(data_path, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.info("Data saved successfully.")
        logger.debug(f"Train shape: {train_data.shape}")
        logger.debug(f"Test shape: {test_data.shape}")

    except Exception:
        logger.exception("Error saving data")
        raise


from pathlib import Path

def main() -> None:
    try:
        logger.info("Starting data injection pipeline...")

        test_size, random_state = load_params("params.yaml")

        df = data_read(
            "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
        )

        final_df = process_data(df)

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=random_state,
            stratify=final_df["sentiment"]
        )

        logger.info("Train-test split completed.")

        # Always resolve project root
        project_root = Path(__file__).resolve().parents[2]
        data_path = project_root / "data" / "raw"

        save_data(str(data_path), train_data, test_data)

        logger.info("Pipeline completed successfully.")

    except Exception:
        logger.exception("Pipeline failed")
        raise

if __name__ == "__main__":
    main()