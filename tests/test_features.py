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

