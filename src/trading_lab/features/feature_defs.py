"""Feature definitions and calculations."""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_lab.features")


def calculate_price_features(df: pd.DataFrame, lookback_days: int = 60) -> pd.DataFrame:
    """
    Calculate price-based features with proper time-shifting to avoid leakage.

    Args:
        df: DataFrame with columns: date, ticker, open, high, low, close, adj_close, volume
        lookback_days: Number of days to look back for rolling features

    Returns:
        DataFrame with feature columns added
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    features_df = pd.DataFrame(index=df.index)

    # Use adj_close for returns
    price_col = "adj_close"

    # Returns (log and simple)
    df["return_1d"] = df.groupby("ticker")[price_col].pct_change(1)
    df["return_5d"] = df.groupby("ticker")[price_col].pct_change(5)
    df["return_20d"] = df.groupby("ticker")[price_col].pct_change(20)
    df["log_return_1d"] = df.groupby("ticker")[price_col].transform(lambda x: np.log(x / x.shift(1)))

    # Rolling volatility
    df["volatility_5d"] = df.groupby("ticker")["return_1d"].transform(lambda x: x.rolling(5, min_periods=2).std())
    df["volatility_20d"] = df.groupby("ticker")["return_1d"].transform(lambda x: x.rolling(20, min_periods=5).std())

    # Moving averages
    df["ma_5"] = df.groupby("ticker")[price_col].transform(lambda x: x.rolling(5, min_periods=2).mean())
    df["ma_20"] = df.groupby("ticker")[price_col].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["ma_50"] = df.groupby("ticker")[price_col].transform(lambda x: x.rolling(50, min_periods=10).mean())

    # Moving average gaps (price relative to MA)
    df["ma_gap_5"] = (df[price_col] - df["ma_5"]) / df["ma_5"]
    df["ma_gap_20"] = (df[price_col] - df["ma_20"]) / df["ma_20"]
    df["ma_gap_50"] = (df[price_col] - df["ma_50"]) / df["ma_50"]

    # Momentum
    df["momentum_5d"] = df["return_5d"]
    df["momentum_20d"] = df["return_20d"]

    # RSI (simplified - 14 period)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df["rsi_14"] = df.groupby("ticker")[price_col].transform(lambda x: calculate_rsi(x, 14))

    # MACD (simplified)
    ema_12 = df.groupby("ticker")[price_col].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_26 = df.groupby("ticker")[price_col].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df.groupby("ticker")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # ATR proxy (using high-low range)
    df["high_low_range"] = (df["high"] - df["low"]) / df[price_col]
    df["atr_proxy_5d"] = df.groupby("ticker")["high_low_range"].transform(lambda x: x.rolling(5, min_periods=2).mean())
    df["atr_proxy_20d"] = df.groupby("ticker")["high_low_range"].transform(lambda x: x.rolling(20, min_periods=5).mean())

    # Volume features
    if "volume" in df.columns and df["volume"].notna().any():
        df["volume_ma_5"] = df.groupby("ticker")["volume"].transform(lambda x: x.rolling(5, min_periods=2).mean())
        df["volume_ma_20"] = df.groupby("ticker")["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
        df["volume_ratio_5"] = df["volume"] / (df["volume_ma_5"] + 1e-8)
        df["volume_ratio_20"] = df["volume"] / (df["volume_ma_20"] + 1e-8)
        # Volume z-score
        df["volume_zscore"] = df.groupby("ticker")["volume"].transform(
            lambda x: (x - x.rolling(20, min_periods=5).mean()) / (x.rolling(20, min_periods=5).std() + 1e-8)
        )

    # Feature selection (only include calculated features)
    feature_cols = [
        "return_1d",
        "return_5d",
        "return_20d",
        "log_return_1d",
        "volatility_5d",
        "volatility_20d",
        "ma_gap_5",
        "ma_gap_20",
        "ma_gap_50",
        "momentum_5d",
        "momentum_20d",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_proxy_5d",
        "atr_proxy_20d",
    ]

    if "volume_ratio_5" in df.columns:
        feature_cols.extend(["volume_ratio_5", "volume_ratio_20", "volume_zscore"])

    # Combine original columns with features
    result = df[["date", "ticker"] + feature_cols].copy()

    return result


def calculate_event_features(
    events_df: pd.DataFrame, price_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate event-based features (news sentiment, social mentions, etc.).

    Args:
        events_df: Unified event DataFrame
        price_df: Price DataFrame to join events to

    Returns:
        DataFrame with event features added to price dates
    """
    if events_df.empty:
        logger.info("No event data available, skipping event features")
        price_df["event_sentiment_score"] = 0.0
        price_df["event_count"] = 0
        return price_df[["date", "ticker", "event_sentiment_score", "event_count"]]

    # Aggregate events by date and ticker
    events_agg = events_df.groupby(["date", "ticker"]).agg(
        event_sentiment_score=pd.NamedAgg(column="score", aggfunc="mean"),
        event_count=pd.NamedAgg(column="count", aggfunc="sum"),
    ).reset_index()

    # Forward fill event features (events affect future days)
    events_agg = events_agg.sort_values(["ticker", "date"])
    events_agg["event_sentiment_score"] = events_agg.groupby("ticker")["event_sentiment_score"].ffill(limit=5)
    events_agg["event_count"] = events_agg.groupby("ticker")["event_count"].fillna(0)

    # Merge with price dates
    result = price_df[["date", "ticker"]].merge(
        events_agg, on=["date", "ticker"], how="left"
    )
    result["event_sentiment_score"] = result["event_sentiment_score"].fillna(0.0)
    result["event_count"] = result["event_count"].fillna(0)

    return result[["date", "ticker", "event_sentiment_score", "event_count"]]


def calculate_fundamentals_features(
    fundamentals_df: pd.DataFrame, price_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate fundamentals-based features.

    Args:
        fundamentals_df: Fundamentals DataFrame (stub for now)
        price_df: Price DataFrame to join fundamentals to

    Returns:
        DataFrame with fundamental features added
    """
    # Stub implementation - would join PE, PB, dividend yield if available
    price_df["fundamental_pe"] = np.nan
    price_df["fundamental_pb"] = np.nan
    price_df["fundamental_dividend_yield"] = np.nan

    return price_df[["date", "ticker", "fundamental_pe", "fundamental_pb", "fundamental_dividend_yield"]]

