"""Data validation and quality checks."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_lab.common.validation")


def check_data_quality(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    check_missing: bool = True,
    check_duplicates: bool = True,
    check_outliers: bool = False,
    outlier_threshold: float = 3.0,
) -> dict:
    """
    Perform data quality checks on a DataFrame.

    Args:
        df: DataFrame to check
        required_columns: List of required column names
        check_missing: Check for missing values
        check_duplicates: Check for duplicate rows
        check_outliers: Check for outliers using z-score
        outlier_threshold: Z-score threshold for outlier detection

    Returns:
        Dictionary with quality check results
    """
    results = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "stats": {},
    }

    # Check if empty
    if df.empty:
        results["is_valid"] = False
        results["issues"].append("DataFrame is empty")
        return results

    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            results["is_valid"] = False
            results["issues"].append(f"Missing required columns: {missing_cols}")

    # Check for missing values
    if check_missing:
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if len(missing_cols) > 0:
            results["warnings"].append(f"Missing values found in columns: {missing_cols.to_dict()}")
            results["stats"]["missing_values"] = missing_cols.to_dict()

    # Check for duplicates
    if check_duplicates:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            results["warnings"].append(f"Found {duplicate_count} duplicate rows")
            results["stats"]["duplicate_count"] = int(duplicate_count)

    # Check for outliers in numeric columns
    if check_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > outlier_threshold
            outlier_count = outliers.sum()
            if outlier_count > 0:
                outlier_counts[col] = int(outlier_count)

        if outlier_counts:
            results["warnings"].append(f"Potential outliers found: {outlier_counts}")
            results["stats"]["outlier_counts"] = outlier_counts

    # Basic statistics
    results["stats"]["shape"] = df.shape
    results["stats"]["memory_usage_mb"] = float(df.memory_usage(deep=True).sum() / 1024**2)

    return results


def validate_price_data(df: pd.DataFrame) -> dict:
    """
    Validate price data quality.

    Args:
        df: Price DataFrame

    Returns:
        Validation results dictionary
    """
    required_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    results = check_data_quality(df, required_columns=required_cols, check_outliers=True)

    if not results["is_valid"]:
        return results

    # Additional price-specific checks
    issues = []

    # Check price relationships
    if "high" in df.columns and "low" in df.columns:
        invalid_high_low = df["high"] < df["low"]
        if invalid_high_low.any():
            issues.append(f"Found {invalid_high_low.sum()} rows where high < low")

    if "open" in df.columns and "close" in df.columns:
        # Check for extreme price movements (potential data errors)
        price_change_pct = abs((df["close"] - df["open"]) / df["open"])
        extreme_moves = price_change_pct > 0.5  # 50% move in one day
        if extreme_moves.any():
            results["warnings"].append(
                f"Found {extreme_moves.sum()} rows with extreme price movements (>50%)"
            )

    # Check for zero or negative prices
    price_cols = ["open", "high", "low", "close", "adj_close"]
    for col in price_cols:
        if col in df.columns:
            invalid_prices = df[col] <= 0
            if invalid_prices.any():
                issues.append(f"Found {invalid_prices.sum()} rows with invalid {col} (<=0)")

    # Check for zero or negative volume
    if "volume" in df.columns:
        invalid_volume = df["volume"] < 0
        if invalid_volume.any():
            issues.append(f"Found {invalid_volume.sum()} rows with negative volume")

    if issues:
        results["is_valid"] = False
        results["issues"].extend(issues)

    return results


def clean_price_data(df: pd.DataFrame, remove_invalid: bool = True) -> pd.DataFrame:
    """
    Clean price data by removing invalid rows.

    Args:
        df: Price DataFrame
        remove_invalid: If True, remove invalid rows instead of raising error

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    original_len = len(df)

    # Remove rows where high < low
    if "high" in df.columns and "low" in df.columns:
        df = df[df["high"] >= df["low"]]

    # Remove rows with invalid prices
    price_cols = ["open", "high", "low", "close", "adj_close"]
    for col in price_cols:
        if col in df.columns:
            df = df[df[col] > 0]

    # Remove rows with negative volume
    if "volume" in df.columns:
        df = df[df["volume"] >= 0]

    removed = original_len - len(df)
    if removed > 0:
        logger.warning(f"Removed {removed} invalid rows from price data ({removed/original_len*100:.2f}%)")

    return df.reset_index(drop=True)
