"""Unify price data from different sources into standard schema."""

import logging
from pathlib import Path

import pandas as pd

from trading_lab.common.io import load_dataframe, save_dataframe
from trading_lab.config.settings import get_settings

logger = logging.getLogger("trading_lab.unify")


def unify_prices(force_refresh: bool = False) -> pd.DataFrame:
    """
    Unify price data from all sources into standard schema.

    Standard schema:
    - date: datetime
    - ticker: str
    - open: float
    - high: float
    - low: float
    - close: float
    - adj_close: float
    - volume: float
    - currency: str (default: USD)
    - exchange: str (inferred from ticker)

    Args:
        force_refresh: If True, regenerate even if unified file exists

    Returns:
        Unified price DataFrame
    """
    settings = get_settings()
    unified_dir = settings.get_unified_dir()
    unified_file = unified_dir / "prices_unified.parquet"

    if unified_file.exists() and not force_refresh:
        logger.info("Loading existing unified price data")
        return load_dataframe(unified_file)

    logger.info("Unifying price data from all sources")

    prices_dir = settings.get_prices_dir()
    all_dfs = []

    # Load all price files from yfinance
    if prices_dir.exists():
        for price_file in prices_dir.glob("*.parquet"):
            try:
                df = load_dataframe(price_file)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {price_file}: {e}")

    if not all_dfs:
        logger.warning("No price data found to unify")
        return pd.DataFrame(
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "currency", "exchange"]
        )

    # Combine all data
    unified = pd.concat(all_dfs, ignore_index=True)

    # Ensure date is datetime
    unified["date"] = pd.to_datetime(unified["date"])

    # Infer exchange and currency from ticker
    def infer_exchange_currency(ticker: str) -> tuple[str, str]:
        """Infer exchange and currency from ticker symbol."""
        if ticker.endswith(".NS"):
            return "NSE", "INR"
        elif ticker.endswith(".BO"):
            return "BSE", "INR"
        else:
            return "NYSE", "USD"  # Default assumption

    if "exchange" not in unified.columns:
        unified[["exchange", "currency"]] = unified["ticker"].apply(
            lambda x: pd.Series(infer_exchange_currency(x))
        )
    elif "currency" not in unified.columns:
        unified["currency"] = unified["exchange"].map({"NSE": "INR", "BSE": "INR", "NYSE": "USD", "NASDAQ": "USD"}).fillna("USD")

    # Ensure required columns exist
    required_cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "currency", "exchange"]
    for col in required_cols:
        if col not in unified.columns:
            if col == "currency":
                unified[col] = "USD"
            elif col == "exchange":
                unified[col] = ""
            else:
                logger.warning(f"Missing required column: {col}")

    # Select and order columns
    unified = unified[required_cols]

    # Sort by date and ticker
    unified = unified.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Remove duplicates
    unified = unified.drop_duplicates(subset=["date", "ticker"], keep="last")

    logger.info(f"Unified {len(unified)} rows for {unified['ticker'].nunique()} tickers")

    # Save unified data
    save_dataframe(unified, unified_file)

    return unified

