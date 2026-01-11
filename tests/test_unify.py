"""Tests for data unification."""

import pandas as pd
import pytest

from trading_lab.unify.unify_prices import unify_prices


def test_unify_prices_empty():
    """Test unifying empty price data."""
    # This should return empty DataFrame with correct columns
    result = unify_prices(force_refresh=True)
    assert isinstance(result, pd.DataFrame)
    expected_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "currency", "exchange"]
    assert all(col in result.columns for col in expected_cols)

