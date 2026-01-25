"""Tests for time-series alignment and gap handling."""

import pandas as pd
import pytest

from trading_lab.common.time_series import (
    align_time_series,
    detect_gaps,
    fill_gaps,
    resample_time_series,
    validate_temporal_consistency,
)


def test_align_time_series():
    """Test time-series alignment."""
    df1 = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "ticker": ["AAPL"] * 5,
        "value1": [1, 2, 3, 4, 5],
    })
    
    df2 = pd.DataFrame({
        "date": pd.date_range("2020-01-03", periods=5, freq="D"),
        "ticker": ["AAPL"] * 5,
        "value2": [10, 20, 30, 40, 50],
    })
    
    aligned = align_time_series([df1, df2], method="outer", fill_method="ffill")
    
    assert len(aligned) > 0
    assert "value1" in aligned.columns
    assert "value2" in aligned.columns


def test_detect_gaps():
    """Test gap detection."""
    # Create data with gaps
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    # Remove some dates to create gaps
    dates_with_gaps = dates[[0, 1, 2, 5, 6, 7, 9]]
    
    df = pd.DataFrame({
        "date": dates_with_gaps,
        "ticker": ["AAPL"] * len(dates_with_gaps),
        "value": range(len(dates_with_gaps)),
    })
    
    gaps = detect_gaps(df, expected_frequency="D")
    
    assert not gaps.empty
    assert "gap_days" in gaps.columns


def test_fill_gaps():
    """Test gap filling."""
    # Create data with gaps
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    dates_with_gaps = dates[[0, 2, 4]]  # Missing days 1 and 3
    
    df = pd.DataFrame({
        "date": dates_with_gaps,
        "ticker": ["AAPL"] * len(dates_with_gaps),
        "value": [1, 3, 5],
    })
    
    filled = fill_gaps(df, method="forward_fill")
    
    # Should have all dates
    assert len(filled) >= len(dates)
    assert filled["value"].notna().any()


def test_validate_temporal_consistency():
    """Test temporal consistency validation."""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10, freq="D"),
        "ticker": ["AAPL"] * 10,
        "value": range(10),
    })
    
    validation = validate_temporal_consistency(df, check_ordering=True, check_duplicates=True)
    
    assert validation["is_valid"]
    assert len(validation["issues"]) == 0


def test_resample_time_series():
    """Test time-series resampling."""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=30, freq="D"),
        "ticker": ["AAPL"] * 30,
        "value": range(30),
    })
    
    resampled = resample_time_series(df, frequency="W", aggregation="last")
    
    assert len(resampled) < len(df)  # Weekly should have fewer rows
    assert "value" in resampled.columns
