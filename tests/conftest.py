"""Pytest configuration and shared fixtures."""

from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from trading_lab.config.settings import get_settings


@pytest.fixture(scope="session")
def temp_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, None, None]:
    """Create a temporary data directory for tests."""
    temp_dir = tmp_path_factory.mktemp("data")
    settings = get_settings()
    original_data_dir = settings.data_dir
    
    # Temporarily change data directory
    settings.data_dir = temp_dir
    yield temp_dir
    
    # Restore original
    settings.data_dir = original_data_dir


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "ticker": ["AAPL"] * 100 + ["MSFT"] * 100,
            "open": [100.0 + i * 0.1 for i in range(200)],
            "high": [105.0 + i * 0.1 for i in range(200)],
            "low": [95.0 + i * 0.1 for i in range(200)],
            "close": [100.0 + i * 0.1 for i in range(200)],
            "adj_close": [100.0 + i * 0.1 for i in range(200)],
            "volume": [1000000 + i * 1000 for i in range(200)],
            "currency": "USD",
            "exchange": "NASDAQ",
        }
    )


@pytest.fixture
def sample_signals_data() -> pd.DataFrame:
    """Create sample signals data for testing."""
    dates = pd.date_range("2020-01-02", periods=50, freq="D")
    return pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "ticker": ["AAPL"] * 50 + ["MSFT"] * 50,
            "prob_class": [0.6 + (i % 10) * 0.04 for i in range(100)],
            "pred_class": [1 if i % 3 == 0 else 0 for i in range(100)],
            "pred_reg": [0.02 + (i % 5) * 0.001 for i in range(100)],
        }
    )


@pytest.fixture
def sample_features_data() -> pd.DataFrame:
    """Create sample features data for testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "ticker": ["AAPL"] * 100 + ["MSFT"] * 100,
            "return_1d": [0.01 * (i % 10 - 5) / 5 for i in range(200)],
            "volatility_5d": [0.02 + (i % 5) * 0.001 for i in range(200)],
            "rsi": [50.0 + (i % 20 - 10) for i in range(200)],
        }
    )
