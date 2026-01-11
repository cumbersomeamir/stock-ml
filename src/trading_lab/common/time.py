"""Time and date utilities."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


def parse_date(date_str: str) -> pd.Timestamp:
    """
    Parse a date string to pandas Timestamp.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        pandas Timestamp
    """
    return pd.Timestamp(date_str)


def get_trading_days(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.DatetimeIndex:
    """
    Get trading days between start and end dates.

    Args:
        start_date: Start date
        end_date: End date
        calendar: Optional trading calendar. If None, uses business days.

    Returns:
        DatetimeIndex of trading days
    """
    if calendar is None:
        # Use business days as proxy for trading days
        return pd.bdate_range(start_date, end_date, inclusive="both")
    else:
        mask = (calendar >= start_date) & (calendar <= end_date)
        return calendar[mask]


def add_business_days(date: pd.Timestamp, days: int) -> pd.Timestamp:
    """
    Add business days to a date.

    Args:
        date: Start date
        days: Number of business days to add

    Returns:
        New date
    """
    return pd.bdate_range(date, periods=days + 1, freq="B")[-1]


def subtract_business_days(date: pd.Timestamp, days: int) -> pd.Timestamp:
    """
    Subtract business days from a date.

    Args:
        date: Start date
        days: Number of business days to subtract

    Returns:
        New date
    """
    return pd.bdate_range(end=date, periods=days + 1, freq="B")[0]

