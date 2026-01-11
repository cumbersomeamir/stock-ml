"""Data schemas and types."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PriceData(BaseModel):
    """Schema for unified price data."""

    date: datetime
    ticker: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float
    currency: str = "USD"
    exchange: str = ""

    class Config:
        """Pydantic config."""

        from_attributes = True


class EventData(BaseModel):
    """Schema for unified event data (news, social, etc.)."""

    date: datetime
    ticker: str
    source: str
    score: Optional[float] = None  # Sentiment score
    count: int = 1  # Number of events
    meta_json: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic config."""

        from_attributes = True


class FeatureData(BaseModel):
    """Schema for feature data."""

    date: datetime
    ticker: str
    features: Dict[str, float]
    target_class: Optional[int] = None
    target_reg: Optional[float] = None

    class Config:
        """Pydantic config."""

        from_attributes = True


class BacktestResult(BaseModel):
    """Schema for backtest results."""

    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    turnover: float
    metrics: Dict[str, float] = Field(default_factory=dict)

    class Config:
        """Pydantic config."""

        from_attributes = True

