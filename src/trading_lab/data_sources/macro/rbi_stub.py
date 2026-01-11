"""RBI data stub (optional)."""

import logging
from typing import List

import pandas as pd

from trading_lab.config.settings import get_settings

logger = logging.getLogger("trading_lab.data_sources.macro")


class RBIStub:
    """RBI data fetcher stub (optional)."""

    def __init__(self):
        """Initialize RBI stub."""
        self.settings = get_settings()
        self.macro_dir = self.settings.get_raw_data_dir() / "macro" / "rbi"
        self.macro_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        series_ids: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch RBI data (stub).

        Args:
            series_ids: List of series IDs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, re-download even if cached

        Returns:
            Empty DataFrame (stub implementation)
        """
        logger.warning("RBI integration is a stub. Implement full integration if needed.")
        return pd.DataFrame()

