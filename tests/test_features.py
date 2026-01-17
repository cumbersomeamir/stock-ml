"""Tests for feature engineering."""

import pandas as pd
import pytest

from trading_lab.features.feature_defs import calculate_price_features


def test_calculate_price_features_basic():
    """Test basic price feature calculation."""
    # Create sample data
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "ticker": ["AAPL"] * 100 + ["MSFT"] * 100,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "adj_close": 100.0,
            "volume": 1000000,
        }
    )

    result = calculate_price_features(df, lookback_days=20)

    assert isinstance(result, pd.DataFrame)
    assert "return_1d" in result.columns
    assert "volatility_5d" in result.columns
    assert len(result) == len(df)


def test_calculate_price_features_required_columns():
    """Test that required columns are present."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "ticker": ["AAPL"] * 50,
        "open": range(100, 150),
        "high": range(105, 155),
        "low": range(95, 145),
        "close": range(100, 150),
        "adj_close": range(100, 150),
        "volume": [1000000] * 50,
    })
    
    result = calculate_price_features(df, lookback_days=20)
    
    # Check that required columns are in result
    assert "date" in result.columns
    assert "ticker" in result.columns
    assert "return_1d" in result.columns
    assert "rsi_14" in result.columns or "volatility_5d" in result.columns


def test_calculate_price_features_empty_input():
    """Test handling of empty input."""
    df = pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])
    result = calculate_price_features(df, lookback_days=20)
    
    assert isinstance(result, pd.DataFrame)
    # Should return DataFrame with at least date and ticker columns
    assert "date" in result.columns or len(result) == 0

