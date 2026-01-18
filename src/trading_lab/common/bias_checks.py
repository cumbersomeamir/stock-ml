"""Look-ahead bias detection and prevention."""

import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger("trading_lab.bias_checks")


def check_for_lookahead_bias(
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    max_correlation_threshold: float = 0.95,
) -> dict:
    """
    Check for potential look-ahead bias by detecting suspiciously high correlations.
    
    Args:
        features_df: Features DataFrame
        targets_df: Targets DataFrame
        max_correlation_threshold: Correlation threshold for warning
        
    Returns:
        Dictionary with bias check results
    """
    results = {
        "has_potential_bias": False,
        "warnings": [],
        "suspicious_features": [],
    }
    
    # Merge features and targets
    data = features_df.merge(targets_df[["date", "ticker", "y_class", "y_reg"]], on=["date", "ticker"], how="inner")
    
    # Get feature columns
    feature_cols = [col for col in data.columns if col not in ["date", "ticker", "y_class", "y_reg"]]
    
    # Check correlation with target
    for target_col in ["y_class", "y_reg"]:
        if target_col not in data.columns:
            continue
        
        # Remove NaN values
        valid_data = data[[target_col] + feature_cols].dropna()
        
        if len(valid_data) < 10:
            continue
        
        # Calculate correlations
        correlations = valid_data[feature_cols].corrwith(valid_data[target_col]).abs()
        
        # Find suspiciously high correlations
        suspicious = correlations[correlations > max_correlation_threshold]
        
        if len(suspicious) > 0:
            results["has_potential_bias"] = True
            results["warnings"].append(
                f"High correlation with {target_col}: {suspicious.to_dict()}"
            )
            results["suspicious_features"].extend(suspicious.index.tolist())
    
    return results


def validate_time_series_split(
    data: pd.DataFrame,
    train_end_date: pd.Timestamp,
    test_start_date: pd.Timestamp,
    date_column: str = "date",
) -> Tuple[bool, str]:
    """
    Validate that time series split doesn't have overlap.
    
    Args:
        data: DataFrame with date column
        train_end_date: End of training period
        test_start_date: Start of test period
        date_column: Name of date column
        
    Returns:
        Tuple of (is_valid, message)
    """
    if train_end_date >= test_start_date:
        return False, f"Training period overlaps with test period: train_end={train_end_date}, test_start={test_start_date}"
    
    # Check for data leakage (train data should not contain future info)
    train_data = data[data[date_column] < train_end_date]
    test_data = data[(data[date_column] >= test_start_date)]
    
    if train_data.empty:
        return False, "Training data is empty"
    
    if test_data.empty:
        return False, "Test data is empty"
    
    return True, "Time series split is valid"


def detect_future_data_leakage(
    features_df: pd.DataFrame,
    lookback_days: int,
    date_column: str = "date",
) -> dict:
    """
    Detect if features might contain future data leakage.
    
    Checks if feature values change retroactively (which would indicate
    features are using future information).
    
    Args:
        features_df: Features DataFrame
        lookback_days: Expected lookback window
        date_column: Name of date column
        
    Returns:
        Dictionary with leakage detection results
    """
    results = {
        "has_potential_leakage": False,
        "issues": [],
    }
    
    # Check if features are stable (don't change when new data is added)
    # This is a heuristic check - real implementation would require
    # re-calculating features with different end dates and comparing
    
    # For now, check for forward-looking values (values that exist before
    # enough lookback data is available)
    features_df = features_df.copy()
    features_df[date_column] = pd.to_datetime(features_df[date_column])
    
    if "ticker" in features_df.columns:
        # Check per ticker
        for ticker in features_df["ticker"].unique():
            ticker_df = features_df[features_df["ticker"] == ticker].sort_values(date_column)
            
            # Features should be NaN for the first lookback_days
            first_n_rows = ticker_df.head(lookback_days)
            
            feature_cols = [col for col in ticker_df.columns if col not in ["date", "ticker"]]
            non_null_counts = first_n_rows[feature_cols].notna().sum()
            
            # If more than half of the features have values in the first lookback period,
            # this might indicate look-ahead bias
            suspicious_features = non_null_counts[non_null_counts > lookback_days * 0.5]
            
            if len(suspicious_features) > 0:
                results["has_potential_leakage"] = True
                results["issues"].append(
                    f"Ticker {ticker}: Features have values in first {lookback_days} days: {suspicious_features.to_dict()}"
                )
    
    return results


def add_embargo_period(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embargo_days: int,
    date_column: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add embargo period between train and test to prevent leakage.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        embargo_days: Number of days to embargo
        date_column: Name of date column
        
    Returns:
        Tuple of (cleaned_train_df, test_df)
    """
    if embargo_days <= 0:
        return train_df, test_df
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df[date_column] = pd.to_datetime(train_df[date_column])
    test_df[date_column] = pd.to_datetime(test_df[date_column])
    
    test_start = test_df[date_column].min()
    embargo_cutoff = test_start - pd.Timedelta(days=embargo_days)
    
    # Remove training data within embargo period
    train_df = train_df[train_df[date_column] < embargo_cutoff]
    
    logger.info(f"Applied {embargo_days}-day embargo period. Train ends at {embargo_cutoff.date()}")
    
    return train_df, test_df
