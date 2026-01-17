"""Data manipulation and analysis utilities."""

from typing import List, Optional

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """
    Safely divide two series, handling division by zero.

    Args:
        numerator: Numerator series
        denominator: Denominator series
        fill_value: Value to use when denominator is zero

    Returns:
        Result series
    """
    return numerator.div(denominator.replace(0, np.nan)).fillna(fill_value)


def calculate_rolling_zscore(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    """
    Calculate rolling z-score of a series.

    Args:
        series: Input series
        window: Rolling window size
        min_periods: Minimum periods required for calculation

    Returns:
        Z-score series
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    
    return (series - rolling_mean) / (rolling_std + 1e-8)


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers from specified columns using IQR or z-score method.

    Args:
        df: Input DataFrame
        columns: Columns to process
        method: Method to use ('iqr' or 'zscore')
        factor: Factor for outlier detection (1.5 for IQR, typically 3 for z-score)

    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    mask = pd.Series([True] * len(df), index=df.index)

    for col in columns:
        if col not in df.columns:
            continue

        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            col_mask = z_scores <= factor
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")

        mask = mask & col_mask

    return df[mask].reset_index(drop=True)


def ensure_datetime_index(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Ensure DataFrame has a datetime index.

    Args:
        df: Input DataFrame
        date_column: Name of date column to use as index

    Returns:
        DataFrame with datetime index
    """
    df = df.copy()
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(date_column)
    return df


def resample_dataframe(
    df: pd.DataFrame,
    freq: str = "D",
    method: str = "last",
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Resample DataFrame to a different frequency.

    Args:
        df: Input DataFrame with date column
        freq: Target frequency (e.g., 'D', 'W', 'M')
        method: Aggregation method ('last', 'first', 'mean', 'sum')
        date_column: Name of date column

    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")

    df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date as index if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(date_column)
    
    # Group by ticker if present, then resample
    if "ticker" in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        agg_dict = {col: method for col in numeric_cols if col != "ticker"}
        
        resampled = df.groupby("ticker").resample(freq).agg(agg_dict)
        resampled = resampled.reset_index()
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        agg_dict = {col: method for col in numeric_cols}
        resampled = df.resample(freq).agg(agg_dict)
        resampled = resampled.reset_index()

    return resampled.reset_index(drop=True)


def forward_fill_missing_dates(
    df: pd.DataFrame,
    date_column: str = "date",
    ticker_column: Optional[str] = "ticker",
    freq: str = "D",
) -> pd.DataFrame:
    """
    Forward fill missing dates in a time series DataFrame.

    Args:
        df: Input DataFrame
        date_column: Name of date column
        ticker_column: Name of ticker column (optional, for multi-ticker data)
        freq: Expected frequency

    Returns:
        DataFrame with missing dates filled
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    if ticker_column and ticker_column in df.columns:
        # Handle multiple tickers
        result_dfs = []
        for ticker in df[ticker_column].unique():
            ticker_df = df[df[ticker_column] == ticker].copy()
            ticker_df = ticker_df.set_index(date_column)
            
            # Create complete date range
            date_range = pd.date_range(
                start=ticker_df.index.min(),
                end=ticker_df.index.max(),
                freq=freq
            )
            ticker_df = ticker_df.reindex(date_range)
            ticker_df[ticker_column] = ticker
            ticker_df = ticker_df.ffill().reset_index()
            ticker_df = ticker_df.rename(columns={"index": date_column})
            
            result_dfs.append(ticker_df)
        
        result = pd.concat(result_dfs, ignore_index=True)
    else:
        # Single time series
        df = df.set_index(date_column)
        date_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )
        df = df.reindex(date_range)
        df = df.ffill().reset_index()
        df = df.rename(columns={"index": date_column})

    return result.sort_values([date_column] + ([ticker_column] if ticker_column and ticker_column in df.columns else [])).reset_index(drop=True)


def calculate_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Calculate correlation matrix for specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to include (if None, uses all numeric columns)
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Correlation matrix DataFrame
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter to only include columns that exist
    columns = [col for col in columns if col in df.columns]
    
    if len(columns) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation")
    
    return df[columns].corr(method=method)
