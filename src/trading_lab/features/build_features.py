"""Build features from unified data."""

import logging
from pathlib import Path

import pandas as pd

from trading_lab.common.io import load_dataframe
from trading_lab.config.settings import get_settings
from trading_lab.features.feature_defs import (
    calculate_event_features,
    calculate_fundamentals_features,
    calculate_price_features,
)
from trading_lab.features.feature_store import FeatureStore
from trading_lab.unify import unify_events, unify_prices

logger = logging.getLogger("trading_lab.features")


def build_features(force_refresh: bool = False, lookback_days: int = 60) -> pd.DataFrame:
    """
    Build features from unified data.

    Args:
        force_refresh: If True, rebuild even if features exist
        lookback_days: Lookback window for rolling features

    Returns:
        DataFrame with all features
    """
    settings = get_settings()
    feature_store = FeatureStore()

    # Check if features already exist
    if not force_refresh:
        try:
            features = feature_store.load_features()
            logger.info("Loaded existing features")
            return features
        except FileNotFoundError:
            logger.info("No existing features found, building new features")

    logger.info("Building features from unified data")

    # Load unified price data
    price_df = unify_prices(force_refresh=False)
    if price_df.empty:
        raise ValueError("No price data available. Run download-prices first.")

    # Calculate price features
    logger.info("Calculating price features")
    price_features = calculate_price_features(price_df, lookback_days=lookback_days)

    # Calculate event features (optional)
    logger.info("Calculating event features")
    try:
        events_df = unify_events(force_refresh=False)
        event_features = calculate_event_features(events_df, price_df)
    except Exception as e:
        logger.warning(f"Could not calculate event features: {e}")
        event_features = pd.DataFrame({"date": price_df["date"], "ticker": price_df["ticker"]})
        event_features["event_sentiment_score"] = 0.0
        event_features["event_count"] = 0

    # Calculate fundamentals features (optional, stub)
    logger.info("Calculating fundamentals features")
    try:
        fundamentals_df = pd.DataFrame()  # Stub
        fundamental_features = calculate_fundamentals_features(fundamentals_df, price_df)
    except Exception as e:
        logger.warning(f"Could not calculate fundamental features: {e}")
        fundamental_features = pd.DataFrame({"date": price_df["date"], "ticker": price_df["ticker"]})
        fundamental_features["fundamental_pe"] = pd.NA
        fundamental_features["fundamental_pb"] = pd.NA
        fundamental_features["fundamental_dividend_yield"] = pd.NA

    # Merge all features
    logger.info("Merging all features")
    features = price_features.merge(event_features, on=["date", "ticker"], how="left")
    features = features.merge(fundamental_features, on=["date", "ticker"], how="left")

    # Fill NaN values
    numeric_cols = features.select_dtypes(include=[float, int]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0.0)

    # Sort by date and ticker
    features = features.sort_values(["date", "ticker"]).reset_index(drop=True)

    logger.info(f"Built {len(features)} feature rows with {len(features.columns) - 2} features")

    # Save features
    feature_store.save_features(features)

    return features

