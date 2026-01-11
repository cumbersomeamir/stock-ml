"""Generate target labels for supervised learning."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from trading_lab.config.constants import DEFAULT_COST_BUFFER, DEFAULT_RETURN_THRESHOLD
from trading_lab.unify import unify_prices

logger = logging.getLogger("trading_lab.labeling")


def generate_targets(
    features_df: pd.DataFrame,
    return_threshold: float = DEFAULT_RETURN_THRESHOLD,
    cost_buffer: Optional[float] = DEFAULT_COST_BUFFER,
    horizon_days: int = 1,
) -> pd.DataFrame:
    """
    Generate target labels for classification and regression.

    Classification target (y_class):
    - 1 if next-day return > threshold
    - 0 if next-day return < -threshold
    - NaN if abs(return) <= threshold (can be dropped or kept as neutral)

    Regression target (y_reg):
    - Next-day realized volatility (absolute return or rolling std proxy)

    Args:
        features_df: Features DataFrame (must have date and ticker columns)
        return_threshold: Minimum absolute return to label as positive/negative
        cost_buffer: Optional cost buffer for transaction-cost-aware labeling
        horizon_days: Prediction horizon in days (default: 1)

    Returns:
        DataFrame with y_class and y_reg columns added
    """
    logger.info(f"Generating targets with threshold={return_threshold}, horizon={horizon_days}")

    # Load price data to calculate returns
    price_df = unify_prices(force_refresh=False)
    if price_df.empty:
        raise ValueError("No price data available for target generation")

    # Calculate forward returns
    price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    price_df["forward_return"] = price_df.groupby("ticker")["adj_close"].pct_change(-horizon_days) * -1
    price_df["forward_volatility"] = price_df["forward_return"].abs()

    # Merge with features
    targets_df = features_df[["date", "ticker"]].merge(
        price_df[["date", "ticker", "forward_return", "forward_volatility"]],
        on=["date", "ticker"],
        how="left",
    )

    # Generate classification target
    if cost_buffer is not None:
        effective_threshold = return_threshold + cost_buffer
    else:
        effective_threshold = return_threshold

    targets_df["y_class"] = np.where(
        targets_df["forward_return"] > effective_threshold,
        1,
        np.where(targets_df["forward_return"] < -effective_threshold, 0, np.nan),
    )

    # Generate regression target (volatility)
    targets_df["y_reg"] = targets_df["forward_volatility"]

    # Drop rows where forward_return is NaN (end of data)
    targets_df = targets_df.dropna(subset=["forward_return"])

    logger.info(
        f"Generated targets: {targets_df['y_class'].notna().sum()} valid classification labels, "
        f"{targets_df['y_class'].sum():.0f} positive, {(targets_df['y_class'] == 0).sum():.0f} negative"
    )

    return targets_df[["date", "ticker", "y_class", "y_reg"]]

