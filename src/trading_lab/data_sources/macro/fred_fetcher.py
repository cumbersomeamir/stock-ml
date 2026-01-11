"""FRED API fetcher (optional)."""

import logging
from typing import List

import pandas as pd

from trading_lab.config.settings import get_settings

logger = logging.getLogger("trading_lab.data_sources.macro")


class FREDFetcher:
    """Fetch macro data from FRED (optional, requires API key)."""

    def __init__(self):
        """Initialize FRED fetcher."""
        self.settings = get_settings()
        self.macro_dir = self.settings.get_raw_data_dir() / "macro" / "fred"
        self.macro_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = self.settings.fred_api_key

        if not self.api_key:
            logger.warning("FRED API key not set. FRED fetching will be disabled.")

    def fetch(
        self,
        series_ids: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch FRED data for given series (stub).

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, re-download even if cached

        Returns:
            Empty DataFrame (stub implementation)
        """
        if not self.api_key:
            logger.warning("FRED API key not set. Returning empty DataFrame.")
            return pd.DataFrame()

        logger.warning("FRED integration is a stub. Implement full integration if needed.")
        return pd.DataFrame()

