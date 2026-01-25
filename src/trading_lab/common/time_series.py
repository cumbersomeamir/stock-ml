"""Time-series alignment and gap handling utilities."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_lab.time_series")


def align_time_series(
    dataframes: List[pd.DataFrame],
    date_column: str = "date",
    ticker_column: str = "ticker",
    method: str = "outer",
    fill_method: Optional[str] = "ffill",
) -> pd.DataFrame:
    """
    Align multiple time-series DataFrames to a common date/ticker index.
    
    This ensures all data sources have the same temporal structure,
    preventing misalignment issues in downstream processing.
    
    Args:
        dataframes: List of DataFrames to align
        date_column: Name of date column
        ticker_column: Name of ticker column
        method: Merge method ('inner', 'outer', 'left', 'right')
        fill_method: How to fill missing values ('ffill', 'bfill', 'interpolate', None)
        
    Returns:
        Aligned DataFrame with consistent date/ticker structure
    """
    if not dataframes:
        raise ValueError("No dataframes provided")
    
    if len(dataframes) == 1:
        return dataframes[0].copy()
    
    logger.info(f"Aligning {len(dataframes)} time-series DataFrames")
    
    # Get all unique dates and tickers
    all_dates = set()
    all_tickers = set()
    
    for df in dataframes:
        if df.empty:
            continue
        all_dates.update(pd.to_datetime(df[date_column]).unique())
        all_tickers.update(df[ticker_column].unique())
    
    if not all_dates or not all_tickers:
        logger.warning("No dates or tickers found in dataframes")
        return pd.DataFrame()
    
    # Create complete index
    all_dates = sorted(all_dates)
    all_tickers = sorted(all_tickers)
    
    # Create MultiIndex
    complete_index = pd.MultiIndex.from_product(
        [all_dates, all_tickers],
        names=[date_column, ticker_column]
    )
    
    # Align each dataframe
    aligned_dfs = []
    for i, df in enumerate(dataframes):
        if df.empty:
            continue
        
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Set index
        df_indexed = df.set_index([date_column, ticker_column])
        
        # Reindex to complete index
        df_aligned = df_indexed.reindex(complete_index)
        
        # Fill missing values
        if fill_method == "ffill":
            df_aligned = df_aligned.groupby(level=ticker_column).ffill()
        elif fill_method == "bfill":
            df_aligned = df_aligned.groupby(level=ticker_column).bfill()
        elif fill_method == "interpolate":
            df_aligned = df_aligned.groupby(level=ticker_column).interpolate(method="linear")
        
        aligned_dfs.append(df_aligned)
    
    # Merge aligned dataframes
    if not aligned_dfs:
        return pd.DataFrame()
    
    result = aligned_dfs[0]
    for df in aligned_dfs[1:]:
        if method == "inner":
            result = result.join(df, how="inner")
        elif method == "outer":
            result = result.join(df, how="outer")
        elif method == "left":
            result = result.join(df, how="left")
        elif method == "right":
            result = result.join(df, how="right")
    
    result = result.reset_index()
    
    logger.info(f"Aligned to {len(result)} rows across {len(all_tickers)} tickers")
    
    return result


def detect_gaps(
    df: pd.DataFrame,
    date_column: str = "date",
    ticker_column: str = "ticker",
    expected_frequency: str = "D",
) -> pd.DataFrame:
    """
    Detect gaps in time-series data.
    
    Args:
        df: DataFrame with time-series data
        date_column: Name of date column
        ticker_column: Name of ticker column
        expected_frequency: Expected frequency ('D' for daily, 'B' for business days)
        
    Returns:
        DataFrame with gap information (ticker, gap_start, gap_end, gap_days)
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values([ticker_column, date_column])
    
    gaps = []
    
    for ticker in df[ticker_column].unique():
        ticker_data = df[df[ticker_column] == ticker].copy()
        ticker_data = ticker_data.sort_values(date_column)
        
        dates = ticker_data[date_column].values
        
        if len(dates) < 2:
            continue
        
        # Calculate expected dates
        date_range = pd.date_range(
            start=dates[0],
            end=dates[-1],
            freq=expected_frequency
        )
        
        # Find missing dates
        missing_dates = set(date_range) - set(dates)
        
        if missing_dates:
            # Group consecutive missing dates
            missing_sorted = sorted(missing_dates)
            gap_start = missing_sorted[0]
            gap_end = missing_sorted[-1]
            gap_days = len(missing_sorted)
            
            gaps.append({
                ticker_column: ticker,
                "gap_start": gap_start,
                "gap_end": gap_end,
                "gap_days": gap_days,
            })
    
    if gaps:
        gaps_df = pd.DataFrame(gaps)
        logger.warning(f"Detected {len(gaps_df)} gaps in time-series data")
        return gaps_df
    else:
        return pd.DataFrame(columns=[ticker_column, "gap_start", "gap_end", "gap_days"])


def fill_gaps(
    df: pd.DataFrame,
    date_column: str = "date",
    ticker_column: str = "ticker",
    method: str = "forward_fill",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fill gaps in time-series data.
    
    Args:
        df: DataFrame with time-series data
        date_column: Name of date column
        ticker_column: Name of ticker column
        method: Fill method ('forward_fill', 'backward_fill', 'interpolate', 'zero')
        limit: Maximum number of consecutive gaps to fill
        
    Returns:
        DataFrame with gaps filled
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values([ticker_column, date_column])
    
    # Create complete date range per ticker
    complete_dfs = []
    
    for ticker in df[ticker_column].unique():
        ticker_data = df[df[ticker_column] == ticker].copy()
        
        # Create complete date range
        date_range = pd.date_range(
            start=ticker_data[date_column].min(),
            end=ticker_data[date_column].max(),
            freq="D"
        )
        
        # Reindex
        ticker_data = ticker_data.set_index(date_column)
        ticker_data = ticker_data.reindex(date_range)
        ticker_data[ticker_column] = ticker
        
        # Fill missing values
        numeric_cols = ticker_data.select_dtypes(include=[np.number]).columns
        
        if method == "forward_fill":
            ticker_data[numeric_cols] = ticker_data[numeric_cols].ffill(limit=limit)
        elif method == "backward_fill":
            ticker_data[numeric_cols] = ticker_data[numeric_cols].bfill(limit=limit)
        elif method == "interpolate":
            ticker_data[numeric_cols] = ticker_data[numeric_cols].interpolate(method="linear", limit=limit)
        elif method == "zero":
            ticker_data[numeric_cols] = ticker_data[numeric_cols].fillna(0)
        
        # Reset index
        ticker_data = ticker_data.reset_index()
        ticker_data = ticker_data.rename(columns={"index": date_column})
        
        complete_dfs.append(ticker_data)
    
    result = pd.concat(complete_dfs, ignore_index=True)
    result = result.sort_values([ticker_column, date_column]).reset_index(drop=True)
    
    logger.info(f"Filled gaps using method '{method}'")
    
    return result


def validate_temporal_consistency(
    df: pd.DataFrame,
    date_column: str = "date",
    ticker_column: str = "ticker",
    check_ordering: bool = True,
    check_duplicates: bool = True,
    check_gaps: bool = False,
) -> dict:
    """
    Validate temporal consistency of time-series data.
    
    Args:
        df: DataFrame to validate
        date_column: Name of date column
        ticker_column: Name of ticker column
        check_ordering: Check if dates are properly ordered
        check_duplicates: Check for duplicate date/ticker combinations
        check_gaps: Check for gaps in time-series
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
    }
    
    if df.empty:
        results["is_valid"] = False
        results["issues"].append("DataFrame is empty")
        return results
    
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Check for duplicates
    if check_duplicates:
        duplicates = df.duplicated(subset=[date_column, ticker_column], keep=False)
        if duplicates.any():
            results["is_valid"] = False
            n_duplicates = duplicates.sum()
            results["issues"].append(f"Found {n_duplicates} duplicate date/ticker combinations")
    
    # Check ordering
    if check_ordering:
        for ticker in df[ticker_column].unique():
            ticker_data = df[df[ticker_column] == ticker].sort_values(date_column)
            if not ticker_data[date_column].is_monotonic_increasing:
                results["warnings"].append(f"Ticker {ticker} has non-monotonic dates")
    
    # Check for gaps
    if check_gaps:
        gaps = detect_gaps(df, date_column, ticker_column)
        if not gaps.empty:
            results["warnings"].append(f"Found {len(gaps)} gaps in time-series")
    
    return results


def resample_time_series(
    df: pd.DataFrame,
    frequency: str,
    date_column: str = "date",
    ticker_column: str = "ticker",
    aggregation: str = "last",
) -> pd.DataFrame:
    """
    Resample time-series data to a different frequency.
    
    Args:
        df: DataFrame with time-series data
        frequency: Target frequency ('D', 'W', 'M', 'Q', 'Y')
        date_column: Name of date column
        ticker_column: Name of ticker column
        aggregation: Aggregation method ('last', 'first', 'mean', 'sum', 'ohlc')
        
    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index([date_column, ticker_column])
    
    resampled_dfs = []
    
    for ticker in df.index.get_level_values(ticker_column).unique():
        ticker_data = df.xs(ticker, level=ticker_column)
        
        if aggregation == "last":
            resampled = ticker_data.resample(frequency).last()
        elif aggregation == "first":
            resampled = ticker_data.resample(frequency).first()
        elif aggregation == "mean":
            resampled = ticker_data.resample(frequency).mean()
        elif aggregation == "sum":
            resampled = ticker_data.resample(frequency).sum()
        elif aggregation == "ohlc":
            # For OHLC data
            if "open" in ticker_data.columns:
                resampled = ticker_data.resample(frequency).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "adj_close": "last",
                    "volume": "sum",
                })
            else:
                resampled = ticker_data.resample(frequency).last()
        else:
            resampled = ticker_data.resample(frequency).last()
        
        resampled[ticker_column] = ticker
        resampled_dfs.append(resampled)
    
    result = pd.concat(resampled_dfs, ignore_index=False)
    result = result.reset_index()
    
    logger.info(f"Resampled to frequency '{frequency}' using '{aggregation}' aggregation")
    
    return result
