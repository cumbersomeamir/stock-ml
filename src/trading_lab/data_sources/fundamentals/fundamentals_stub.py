"""Fundamentals data stub (optional)."""

import logging
from typing import List

import pandas as pd

from trading_lab.config.settings import get_settings

logger = logging.getLogger("trading_lab.data_sources.fundamentals")


class FundamentalsStub:
    """Fundamentals data fetcher stub (optional, requires API keys)."""

    def __init__(self):
        """Initialize fundamentals stub."""
        self.settings = get_settings()
        self.fundamentals_dir = self.settings.get_raw_data_dir() / "fundamentals"
        self.fundamentals_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch fundamentals data for given tickers (stub).

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, re-download even if cached

        Returns:
            Empty DataFrame (stub implementation)
        """
        logger.warning("Fundamentals integration is a stub. Implement with FMP or Alpha Vantage if needed.")
        return pd.DataFrame(columns=["date", "ticker", "pe", "pb", "dividend_yield"])

