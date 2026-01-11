"""Reddit fetcher (optional)."""

import logging
from typing import List

import pandas as pd

from trading_lab.config.settings import get_settings

logger = logging.getLogger("trading_lab.data_sources.social")


class RedditFetcher:
    """Fetch Reddit data (optional, requires API credentials)."""

    def __init__(self):
        """Initialize Reddit fetcher."""
        self.settings = get_settings()
        self.reddit_dir = self.settings.get_raw_data_dir() / "social" / "reddit"
        self.reddit_dir.mkdir(parents=True, exist_ok=True)
        self.client_id = self.settings.reddit_client_id
        self.client_secret = self.settings.reddit_client_secret
        self.user_agent = self.settings.reddit_user_agent

        if not all([self.client_id, self.client_secret]):
            logger.warning("Reddit credentials not set. Reddit fetching will be disabled.")

    def fetch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch Reddit data for given tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, re-download even if cached

        Returns:
            DataFrame with columns: date, ticker, source, score, count, meta_json
        """
        if not all([self.client_id, self.client_secret]):
            logger.warning("Reddit credentials not set. Returning empty DataFrame.")
            return pd.DataFrame(columns=["date", "ticker", "source", "score", "count", "meta_json"])

        # This is a stub implementation
        logger.warning("Reddit integration is a stub. Implement full integration if needed.")

        return pd.DataFrame(columns=["date", "ticker", "source", "score", "count", "meta_json"])

