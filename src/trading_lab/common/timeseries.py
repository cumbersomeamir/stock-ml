"""Time-series alignment and gap handling utilities.

Fundamental operations for ensuring data integrity across time series.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_lab.timeseries")


def align_time_series(
    *dataframes: pd.DataFrame,
    date_column: str = "date",
    ticker_column: Optional[str] = "ticker",
    method: str = "inner",
    fill_method: Optional[str] = None,
) -> List[pd.DataFrame]:
    """
    Align multiple time series to the same date index.
    
    Critical for preventing look-ahead bias and ensuring all data points
    have corresponding values across all datasets.
    
    Args:
        dataframes: Variable number of DataFrames to align
        date_column: Name of date column
        ticker_column: Name of ticker column (None for single series)
        method: Alignment method ('inner', 'outer', 'left')
        fill_method: Method to fill missing values ('ffill', 'bfill', None)
        
    Returns:
        List of aligned DataFrames
    """
    if len(dataframes) < 2:
        return list(dataframes)
    
    aligned = []
    
    if ticker_column:
        # Multi-series alignment (per ticker)
        # Get all unique tickers
        all_tickers = set()
        for df in dataframes:
            if ticker_column in df.columns:
                all_tickers.update(df[ticker_column].unique())
        
        aligned_dfs = [[] for _ in dataframes]
        
        for ticker in all_tickers:
            # Extract per-ticker data
            ticker_dfs = []
            for df in dataframes:
                if ticker_column in df.columns:
                    ticker_df = df[df[ticker_column] == ticker].copy()
                    ticker_dfs.append(ticker_df)
                else:
                    ticker_dfs.append(df.copy())
            
            # Align dates
            if method == "inner":
                # Only keep dates present in all series
                common_dates = set(ticker_dfs[0][date_column])
                for df in ticker_dfs[1:]:
                    common_dates &= set(df[date_column])
                
                for i, df in enumerate(ticker_dfs):
                    aligned_df = df[df[date_column].isin(common_dates)].copy()
                    aligned_dfs[i].append(aligned_df)
            
            elif method == "outer":
                # Keep all dates from all series
                all_dates = set()
                for df in ticker_dfs:
                    all_dates.update(df[date_column])
                all_dates = sorted(all_dates)
                
                for i, df in enumerate(ticker_dfs):
                    # Reindex to all dates
                    df_indexed = df.set_index(date_column)
                    df_reindexed = df_indexed.reindex(all_dates)
                    
                    if fill_method:
                        df_reindexed = df_reindexed.fillna(method=fill_method)
                    
                    df_reindexed = df_reindexed.reset_index()
                    df_reindexed[date_column] = df_reindexed["index"]
                    df_reindexed = df_reindexed.drop("index", axis=1)
                    
                    if ticker_column:
                        df_reindexed[ticker_column] = ticker
                    
                    aligned_dfs[i].append(df_reindexed)
        
        # Concatenate all tickers
        aligned = [pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame() 
                   for dfs in aligned_dfs]
    
    else:
        # Single series alignment
        if method == "inner":
            common_dates = set(dataframes[0][date_column])
            for df in dataframes[1:]:
                common_dates &= set(df[date_column])
            
            aligned = [df[df[date_column].isin(common_dates)].copy() 
                      for df in dataframes]
        
        elif method == "outer":
            all_dates = set()
            for df in dataframes:
                all_dates.update(df[date_column])
            all_dates = sorted(all_dates)
            
            aligned = []
            for df in dataframes:
                df_indexed = df.set_index(date_column)
                df_reindexed = df_indexed.reindex(all_dates)
                
                if fill_method:
                    df_reindexed = df_reindexed.fillna(method=fill_method)
                
                df_reindexed = df_reindexed.reset_index()
                aligned.append(df_reindexed)
    
    # Log alignment results
    original_lengths = [len(df) for df in dataframes]
    aligned_lengths = [len(df) for df in aligned]
    logger.info(
        f"Aligned time series: {original_lengths} -> {aligned_lengths} "
        f"(method={method})"
    )
    
    return aligned


def detect_gaps(
    df: pd.DataFrame,
    date_column: str = "date",
    ticker_column: Optional[str] = "ticker",
    expected_frequency: str = "B",  # Business days
    max_gap_days: int = 5,
) -> pd.DataFrame:
    """
    Detect gaps in time series data.
    
    Args:
        df: DataFrame to check
        date_column: Name of date column
        ticker_column: Name of ticker column (None for single series)
        expected_frequency: Expected frequency ('D', 'B', 'W', etc.)
        max_gap_days: Maximum acceptable gap in days
        
    Returns:
        DataFrame with gap information (ticker, gap_start, gap_end, gap_days)
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    gaps = []
    
    if ticker_column:
        for ticker in df[ticker_column].unique():
            ticker_df = df[df[ticker_column] == ticker].sort_values(date_column)
            ticker_gaps = _detect_gaps_single_series(
                ticker_df, date_column, expected_frequency, max_gap_days
            )
            for gap in ticker_gaps:
                gap["ticker"] = ticker
            gaps.extend(ticker_gaps)
    else:
        gaps = _detect_gaps_single_series(
            df, date_column, expected_frequency, max_gap_days
        )
    
    if gaps:
        logger.warning(f"Detected {len(gaps)} gaps in time series data")
    
    return pd.DataFrame(gaps)


def _detect_gaps_single_series(
    df: pd.DataFrame,
    date_column: str,
    expected_frequency: str,
    max_gap_days: int,
) -> List[dict]:
    """Detect gaps in a single time series."""
    df = df.sort_values(date_column).reset_index(drop=True)
    dates = pd.to_datetime(df[date_column])
    
    gaps = []
    
    for i in range(len(dates) - 1):
        current_date = dates.iloc[i]
        next_date = dates.iloc[i + 1]
        
        # Calculate expected next date
        if expected_frequency == "B":
            expected_range = pd.bdate_range(current_date, periods=2, freq="B")
        else:
            expected_range = pd.date_range(current_date, periods=2, freq=expected_frequency)
        
        expected_next = expected_range[1]
        
        # Check if there's a gap
        actual_gap_days = (next_date - current_date).days
        expected_gap_days = (expected_next - current_date).days
        
        if actual_gap_days > max_gap_days or actual_gap_days > expected_gap_days * 2:
            gaps.append({
                "gap_start": current_date,
                "gap_end": next_date,
                "gap_days": actual_gap_days,
                "expected_days": expected_gap_days,
            })
    
    return gaps


def fill_gaps(
    df: pd.DataFrame,
    date_column: str = "date",
    ticker_column: Optional[str] = "ticker",
    frequency: str = "B",
    method: str = "ffill",
    limit: Optional[int] = 5,
) -> pd.DataFrame:
    """
    Fill gaps in time series data.
    
    Args:
        df: DataFrame with gaps
        date_column: Name of date column
        ticker_column: Name of ticker column
        frequency: Frequency to fill ('D', 'B', etc.)
        method: Fill method ('ffill', 'bfill', 'interpolate')
        limit: Maximum number of consecutive NaNs to fill
        
    Returns:
        DataFrame with filled gaps
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    if ticker_column:
        filled_dfs = []
        for ticker in df[ticker_column].unique():
            ticker_df = df[df[ticker_column] == ticker].copy()
            ticker_filled = _fill_gaps_single_series(
                ticker_df, date_column, frequency, method, limit
            )
            ticker_filled[ticker_column] = ticker
            filled_dfs.append(ticker_filled)
        
        result = pd.concat(filled_dfs, ignore_index=True)
    else:
        result = _fill_gaps_single_series(df, date_column, frequency, method, limit)
    
    original_len = len(df)
    filled_len = len(result)
    logger.info(f"Filled gaps: {original_len} -> {filled_len} rows (+{filled_len - original_len})")
    
    return result


def _fill_gaps_single_series(
    df: pd.DataFrame,
    date_column: str,
    frequency: str,
    method: str,
    limit: Optional[int],
) -> pd.DataFrame:
    """Fill gaps in a single time series."""
    df = df.sort_values(date_column).reset_index(drop=True)
    
    # Create complete date range
    if frequency == "B":
        date_range = pd.bdate_range(
            df[date_column].min(),
            df[date_column].max(),
            freq="B"
        )
    else:
        date_range = pd.date_range(
            df[date_column].min(),
            df[date_column].max(),
            freq=frequency
        )
    
    # Set date as index and reindex
    df_indexed = df.set_index(date_column)
    df_reindexed = df_indexed.reindex(date_range)
    
    # Fill missing values
    if method == "ffill":
        df_filled = df_reindexed.fillna(method="ffill", limit=limit)
    elif method == "bfill":
        df_filled = df_reindexed.fillna(method="bfill", limit=limit)
    elif method == "interpolate":
        df_filled = df_reindexed.interpolate(method="time", limit=limit)
    else:
        raise ValueError(f"Unknown fill method: {method}")
    
    # Reset index
    df_filled = df_filled.reset_index()
    df_filled.rename(columns={"index": date_column}, inplace=True)
    
    return df_filled


def validate_temporal_ordering(
    df: pd.DataFrame,
    date_column: str = "date",
    ticker_column: Optional[str] = "ticker",
) -> Tuple[bool, List[str]]:
    """
    Validate that data is properly ordered in time.
    
    Args:
        df: DataFrame to validate
        date_column: Name of date column
        ticker_column: Name of ticker column
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    if ticker_column:
        for ticker in df[ticker_column].unique():
            ticker_df = df[df[ticker_column] == ticker]
            
            # Check if sorted
            if not ticker_df[date_column].is_monotonic_increasing:
                errors.append(f"Ticker {ticker}: Dates not in ascending order")
            
            # Check for duplicates
            duplicates = ticker_df[ticker_df.duplicated(subset=[date_column], keep=False)]
            if len(duplicates) > 0:
                errors.append(
                    f"Ticker {ticker}: {len(duplicates)} duplicate dates found"
                )
    else:
        # Check if sorted
        if not df[date_column].is_monotonic_increasing:
            errors.append("Dates not in ascending order")
        
        # Check for duplicates
        duplicates = df[df.duplicated(subset=[date_column], keep=False)]
        if len(duplicates) > 0:
            errors.append(f"{len(duplicates)} duplicate dates found")
    
    is_valid = len(errors) == 0
    
    if not is_valid:
        logger.warning(f"Temporal ordering validation failed: {errors}")
    
    return is_valid, errors


def resample_to_common_frequency(
    df: pd.DataFrame,
    target_frequency: str = "D",
    date_column: str = "date",
    ticker_column: Optional[str] = "ticker",
    aggregation: str = "last",
) -> pd.DataFrame:
    """
    Resample time series to a common frequency.
    
    Args:
        df: DataFrame to resample
        target_frequency: Target frequency ('D', 'W', 'M', etc.)
        date_column: Name of date column
        ticker_column: Name of ticker column
        aggregation: Aggregation method ('last', 'first', 'mean', 'sum')
        
    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    if ticker_column:
        resampled_dfs = []
        for ticker in df[ticker_column].unique():
            ticker_df = df[df[ticker_column] == ticker].copy()
            ticker_resampled = _resample_single_series(
                ticker_df, target_frequency, date_column, aggregation
            )
            ticker_resampled[ticker_column] = ticker
            resampled_dfs.append(ticker_resampled)
        
        result = pd.concat(resampled_dfs, ignore_index=True)
    else:
        result = _resample_single_series(df, target_frequency, date_column, aggregation)
    
    logger.info(f"Resampled to {target_frequency}: {len(df)} -> {len(result)} rows")
    
    return result


def _resample_single_series(
    df: pd.DataFrame,
    target_frequency: str,
    date_column: str,
    aggregation: str,
) -> pd.DataFrame:
    """Resample a single time series."""
    df = df.set_index(date_column)
    
    if aggregation == "last":
        resampled = df.resample(target_frequency).last()
    elif aggregation == "first":
        resampled = df.resample(target_frequency).first()
    elif aggregation == "mean":
        resampled = df.resample(target_frequency).mean()
    elif aggregation == "sum":
        resampled = df.resample(target_frequency).sum()
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return resampled.reset_index()
