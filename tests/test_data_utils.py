"""Tests for data utilities."""

import numpy as np
import pandas as pd
import pytest

from trading_lab.common.data_utils import (
    calculate_correlation_matrix,
    calculate_rolling_zscore,
    ensure_datetime_index,
    forward_fill_missing_dates,
    remove_outliers,
    resample_dataframe,
    safe_divide,
)


def test_safe_divide():
    """Test safe division."""
    a = pd.Series([1, 2, 3, 4])
    b = pd.Series([2, 0, 1, 2])
    result = safe_divide(a, b)
    assert result.iloc[1] == 0.0  # Division by zero should return 0
    assert result.iloc[0] == 0.5
    assert result.iloc[2] == 3.0


def test_calculate_rolling_zscore():
    """Test rolling z-score calculation."""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    zscore = calculate_rolling_zscore(series, window=5)
    assert len(zscore) == len(series)
    # Z-scores should be centered around 0
    assert abs(zscore.iloc[-1]) < 5  # Reasonable z-score


def test_remove_outliers_iqr():
    """Test outlier removal using IQR method."""
    df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10]})
    result = remove_outliers(df, ["values"], method="iqr")
    # Outlier 100 should be removed
    assert 100 not in result["values"].values
    assert len(result) < len(df)


def test_remove_outliers_zscore():
    """Test outlier removal using z-score method."""
    df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10]})
    result = remove_outliers(df, ["values"], method="zscore", factor=2.0)
    # Outlier 100 should be removed with z-score > 2
    assert 100 not in result["values"].values or len(result) < len(df)


def test_ensure_datetime_index():
    """Test datetime index conversion."""
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "value": [1, 2, 3]
    })
    result = ensure_datetime_index(df, date_column="date")
    assert isinstance(result.index, pd.DatetimeIndex)


def test_resample_dataframe():
    """Test DataFrame resampling."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["AAPL"] * 10,
        "value": range(10)
    })
    result = resample_dataframe(df, freq="2D", method="last")
    assert len(result) <= len(df)
    assert "date" in result.columns


def test_forward_fill_missing_dates():
    """Test forward filling missing dates."""
    dates = pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05"])
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["AAPL"] * 3,
        "value": [1, 2, 3]
    })
    result = forward_fill_missing_dates(df, freq="D")
    assert len(result) > len(df)  # Should have more dates after filling


def test_calculate_correlation_matrix():
    """Test correlation matrix calculation."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [2, 4, 6, 8, 10],
        "c": [1, 1, 1, 1, 1]
    })
    corr = calculate_correlation_matrix(df)
    assert isinstance(corr, pd.DataFrame)
    assert "a" in corr.columns
    assert "b" in corr.columns
    # a and b should be highly correlated
    assert corr.loc["a", "b"] > 0.9
