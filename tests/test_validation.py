"""Tests for data validation utilities."""

import pandas as pd
import pytest

from trading_lab.common.validation import (
    check_data_quality,
    clean_price_data,
    validate_price_data,
)


def test_check_data_quality_success():
    """Test successful data quality check."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    results = check_data_quality(df, required_columns=["col1", "col2"])
    assert results["is_valid"] is True
    assert len(results["issues"]) == 0


def test_check_data_quality_missing_columns():
    """Test data quality check with missing columns."""
    df = pd.DataFrame({"col1": [1, 2, 3]})
    results = check_data_quality(df, required_columns=["col1", "col2"])
    assert results["is_valid"] is False
    assert "Missing required columns" in str(results["issues"])


def test_check_data_quality_missing_values():
    """Test data quality check with missing values."""
    df = pd.DataFrame({"col1": [1, 2, None], "col2": [4, None, 6]})
    results = check_data_quality(df, check_missing=True)
    assert len(results["warnings"]) > 0
    assert "missing_values" in results["stats"]


def test_validate_price_data_valid():
    """Test validation of valid price data."""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5),
        "ticker": ["AAPL"] * 5,
        "open": [100, 101, 102, 103, 104],
        "high": [105, 106, 107, 108, 109],
        "low": [95, 96, 97, 98, 99],
        "close": [102, 103, 104, 105, 106],
        "adj_close": [102, 103, 104, 105, 106],
        "volume": [1000000] * 5,
    })
    results = validate_price_data(df)
    assert results["is_valid"] is True


def test_validate_price_data_invalid_high_low():
    """Test validation with invalid high/low relationship."""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=3),
        "ticker": ["AAPL"] * 3,
        "open": [100, 101, 102],
        "high": [105, 95, 107],  # Second row: high < low
        "low": [95, 96, 97],
        "close": [102, 103, 104],
        "adj_close": [102, 103, 104],
        "volume": [1000000] * 3,
    })
    results = validate_price_data(df)
    assert results["is_valid"] is False
    assert any("high < low" in issue for issue in results["issues"])


def test_clean_price_data():
    """Test cleaning price data."""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5),
        "ticker": ["AAPL"] * 5,
        "open": [100, 101, -1, 103, 104],  # Invalid price
        "high": [105, 95, 107, 108, 109],  # Invalid high < low
        "low": [95, 96, 97, 98, 99],
        "close": [102, 103, 104, 105, 106],
        "adj_close": [102, 103, 104, 105, 106],
        "volume": [1000000] * 5,
    })
    cleaned = clean_price_data(df)
    assert len(cleaned) < len(df)
    assert all(cleaned["open"] > 0)
    assert all(cleaned["high"] >= cleaned["low"])
