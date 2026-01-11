"""Unify event data (news, social, etc.) into standard schema."""

import json
import logging
from pathlib import Path

import pandas as pd

from trading_lab.common.io import load_dataframe, save_dataframe
from trading_lab.config.settings import get_settings

logger = logging.getLogger("trading_lab.unify")


def unify_events(force_refresh: bool = False) -> pd.DataFrame:
    """
    Unify event data from all sources into standard schema.

    Standard schema:
    - date: datetime
    - ticker: str
    - source: str (e.g., 'newsapi', 'reddit')
    - score: float (sentiment score, optional)
    - count: int (number of events)
    - meta_json: dict (additional metadata as JSON)

    Args:
        force_refresh: If True, regenerate even if unified file exists

    Returns:
        Unified event DataFrame
    """
    settings = get_settings()
    unified_dir = settings.get_unified_dir()
    unified_file = unified_dir / "events_unified.parquet"

    if unified_file.exists() and not force_refresh:
        logger.info("Loading existing unified event data")
        return load_dataframe(unified_file)

    logger.info("Unifying event data from all sources")

    raw_data_dir = settings.get_raw_data_dir()
    all_dfs = []

    # Load news data
    news_dir = raw_data_dir / "news" / "newsapi"
    if news_dir.exists():
        for news_file in news_dir.glob("*.parquet"):
            try:
                df = load_dataframe(news_file)
                if not df.empty and "source" not in df.columns:
                    df["source"] = "newsapi"
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logger.debug(f"No news data file {news_file}: {e}")

    # Load social data
    social_dir = raw_data_dir / "social" / "reddit"
    if social_dir.exists():
        for social_file in social_dir.glob("*.parquet"):
            try:
                df = load_dataframe(social_file)
                if not df.empty and "source" not in df.columns:
                    df["source"] = "reddit"
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logger.debug(f"No social data file {social_file}: {e}")

    if not all_dfs:
        logger.info("No event data found to unify")
        return pd.DataFrame(columns=["date", "ticker", "source", "score", "count", "meta_json"])

    # Combine all data
    unified = pd.concat(all_dfs, ignore_index=True)

    # Ensure date is datetime
    if "date" in unified.columns:
        unified["date"] = pd.to_datetime(unified["date"])

    # Ensure required columns exist
    if "score" not in unified.columns:
        unified["score"] = None
    if "count" not in unified.columns:
        unified["count"] = 1
    if "meta_json" not in unified.columns:
        unified["meta_json"] = "{}"

    # Convert meta_json to string if it's a dict
    if unified["meta_json"].dtype == object:
        unified["meta_json"] = unified["meta_json"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else (x if isinstance(x, str) else "{}")
        )

    # Select and order columns
    required_cols = ["date", "ticker", "source", "score", "count", "meta_json"]
    for col in required_cols:
        if col not in unified.columns:
            logger.warning(f"Missing required column: {col}")

    unified = unified[required_cols]

    # Sort by date and ticker
    unified = unified.sort_values(["date", "ticker", "source"]).reset_index(drop=True)

    logger.info(f"Unified {len(unified)} event rows")

    # Save unified data
    save_dataframe(unified, unified_file)

    return unified

