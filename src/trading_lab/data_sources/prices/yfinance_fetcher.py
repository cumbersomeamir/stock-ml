"""yfinance price data fetcher."""

import logging
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

from trading_lab.common.io import load_dataframe, save_dataframe
from trading_lab.common.resilience import retry
from trading_lab.config.settings import get_settings

logger = logging.getLogger("trading_lab.data_sources.prices")


class YFinanceFetcher:
    """Fetch price data from yfinance with caching."""

    def __init__(self):
        """Initialize yfinance fetcher."""
        self.settings = get_settings()
        self.prices_dir = self.settings.get_prices_dir()

    def fetch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
        rate_limit_delay: float = 0.1,
    ) -> pd.DataFrame:
        """
        Fetch price data for given tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: If True, re-download even if cached
            rate_limit_delay: Delay between requests in seconds

        Returns:
            DataFrame with columns: date, ticker, open, high, low, close, adj_close, volume
        """
        all_data = []

        for ticker in tickers:
            logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")

            # Check cache
            cache_file = self.prices_dir / f"{ticker}_{start_date}_{end_date}.parquet"

            if cache_file.exists() and not force_refresh:
                logger.info(f"Loading cached data for {ticker}")
                try:
                    df = load_dataframe(cache_file)
                    all_data.append(df)
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load cache for {ticker}: {e}, re-downloading")

            # Download data with retry logic
            @retry(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(Exception,))
            def _fetch_ticker_data():
                ticker_obj = yf.Ticker(ticker)
                # Use auto_adjust=True to get adjusted prices
                df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
                return df
            
            try:
                df = _fetch_ticker_data()

                if df.empty:
                    logger.warning(f"No data found for {ticker}")
                    continue

                # Reset index to make date a column
                df = df.reset_index()
                df["Date"] = pd.to_datetime(df["Date"]).dt.date

                # Standardize column names
                # When auto_adjust=True, Close is already adjusted
                df = df.rename(
                    columns={
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )

                # Use close as adj_close when auto_adjust=True
                df["adj_close"] = df["close"]

                # Add ticker column
                df["ticker"] = ticker

                # Select and order columns
                df = df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]

                # Save to cache
                save_dataframe(df, cache_file)

                all_data.append(df)

                # Rate limiting
                time.sleep(rate_limit_delay)

            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                continue

        if not all_data:
            logger.warning("No data fetched for any ticker")
            return pd.DataFrame()

        # Combine all data
        result = pd.concat(all_data, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])

        logger.info(f"Fetched data for {len(result['ticker'].unique())} tickers, {len(result)} rows")

        return result

