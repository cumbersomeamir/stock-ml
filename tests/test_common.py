"""Tests for common utilities."""

import pandas as pd
import pytest

from trading_lab.common.schemas import validate_dataframe_columns


def test_validate_dataframe_columns_success():
    """Test successful validation."""
    df = pd.DataFrame({"date": [1, 2], "ticker": ["A", "B"], "value": [10, 20]})
    # Should not raise
    validate_dataframe_columns(df, ["date", "ticker"], "test")


def test_validate_dataframe_columns_empty():
    """Test validation with empty DataFrame."""
    df = pd.DataFrame(columns=["date", "ticker"])
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_dataframe_columns(df, ["date", "ticker"], "test")


def test_validate_dataframe_columns_missing():
    """Test validation with missing columns."""
    df = pd.DataFrame({"date": [1, 2], "value": [10, 20]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe_columns(df, ["date", "ticker"], "test")
