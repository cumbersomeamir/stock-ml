"""I/O utilities for data persistence."""

import json
import pickle
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from trading_lab.config.settings import get_settings


def save_dataframe(df: pd.DataFrame, filepath: Path, format: str = "parquet") -> None:
    """
    Save DataFrame to disk.

    Args:
        df: DataFrame to save
        filepath: Output file path
        format: File format ('parquet', 'csv', 'pkl')
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        df.to_parquet(filepath, index=True)
    elif format == "csv":
        df.to_csv(filepath, index=True)
    elif format == "pkl":
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(filepath: Path, format: Optional[str] = None) -> pd.DataFrame:
    """
    Load DataFrame from disk.

    Args:
        filepath: Input file path
        format: File format ('parquet', 'csv', 'pkl'). If None, inferred from extension.

    Returns:
        Loaded DataFrame
    """
    if format is None:
        format = filepath.suffix[1:].lower()  # Remove dot

    if format == "parquet":
        return pd.read_parquet(filepath)
    elif format == "csv":
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif format == "pkl":
        return pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_model(model: Any, filepath: Path) -> None:
    """
    Save a model using joblib.

    Args:
        model: Model object to save
        filepath: Output file path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path) -> Any:
    """
    Load a model using joblib.

    Args:
        filepath: Input file path

    Returns:
        Loaded model
    """
    return joblib.load(filepath)


def save_json(data: dict, filepath: Path) -> None:
    """
    Save data as JSON.

    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Path) -> dict:
    """
    Load data from JSON.

    Args:
        filepath: Input file path

    Returns:
        Loaded dictionary
    """
    with open(filepath, "r") as f:
        return json.load(f)

