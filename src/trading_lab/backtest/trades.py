"""Trade tracking and analysis for backtesting."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd


@dataclass
class Trade:
    """Represents a single trade."""

    ticker: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float  # Fraction of capital
    direction: int  # 1 for long, -1 for short
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    holding_days: Optional[int] = None
    entry_capital: Optional[float] = None
    exit_capital: Optional[float] = None


class TradeTracker:
    """Track trades during backtesting."""

    def __init__(self):
        """Initialize trade tracker."""
        self.trades: List[Trade] = []
        self.open_trades: dict[str, Trade] = {}  # ticker -> Trade

    def open_trade(
        self,
        ticker: str,
        date: datetime,
        price: float,
        position_size: float,
        direction: int,
        capital: float,
    ) -> None:
        """
        Record opening a trade.

        Args:
            ticker: Ticker symbol
            date: Entry date
            price: Entry price
            position_size: Position size as fraction of capital
            direction: 1 for long, -1 for short
            capital: Current capital
        """
        trade = Trade(
            ticker=ticker,
            entry_date=date,
            exit_date=None,
            entry_price=price,
            exit_price=None,
            position_size=position_size,
            direction=direction,
            entry_capital=capital,
        )
        self.open_trades[ticker] = trade

    def close_trade(
        self,
        ticker: str,
        date: datetime,
        price: float,
        capital: float,
    ) -> Optional[Trade]:
        """
        Record closing a trade.

        Args:
            ticker: Ticker symbol
            date: Exit date
            price: Exit price
            capital: Current capital

        Returns:
            Closed trade if it existed, None otherwise
        """
        if ticker not in self.open_trades:
            return None

        trade = self.open_trades.pop(ticker)
        trade.exit_date = date
        trade.exit_price = price
        trade.exit_capital = capital

        # Calculate PnL
        if trade.direction == 1:  # Long
            trade.pnl_pct = (price - trade.entry_price) / trade.entry_price
        else:  # Short
            trade.pnl_pct = (trade.entry_price - price) / trade.entry_price

        trade.pnl = trade.pnl_pct * trade.position_size * trade.entry_capital
        trade.holding_days = (date - trade.entry_date).days

        self.trades.append(trade)
        return trade

    def update_position(
        self,
        ticker: str,
        date: datetime,
        price: float,
        new_position_size: float,
        new_direction: int,
        capital: float,
    ) -> None:
        """
        Update position (close old, open new if needed).

        Args:
            ticker: Ticker symbol
            date: Date of update
            price: Current price
            new_position_size: New position size
            new_direction: New direction (1 or -1)
            capital: Current capital
        """
        # Close existing position if direction changed or size went to zero
        if ticker in self.open_trades:
            existing_trade = self.open_trades[ticker]
            if (
                existing_trade.direction != new_direction
                or abs(new_position_size) < 1e-6
            ):
                self.close_trade(ticker, date, price, capital)

        # Open new position if size is non-zero
        if abs(new_position_size) > 1e-6:
            self.open_trade(ticker, date, price, abs(new_position_size), new_direction, capital)

    def get_trades_df(self) -> pd.DataFrame:
        """
        Get all trades as a DataFrame.

        Returns:
            DataFrame with trade information
        """
        if not self.trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.trades:
            trades_data.append({
                "ticker": trade.ticker,
                "entry_date": trade.entry_date,
                "exit_date": trade.exit_date,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "position_size": trade.position_size,
                "direction": trade.direction,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "holding_days": trade.holding_days,
                "entry_capital": trade.entry_capital,
                "exit_capital": trade.exit_capital,
            })

        return pd.DataFrame(trades_data)

    def get_open_trades_df(self) -> pd.DataFrame:
        """
        Get open trades as a DataFrame.

        Returns:
            DataFrame with open trade information
        """
        if not self.open_trades:
            return pd.DataFrame()

        trades_data = []
        for trade in self.open_trades.values():
            trades_data.append({
                "ticker": trade.ticker,
                "entry_date": trade.entry_date,
                "entry_price": trade.entry_price,
                "position_size": trade.position_size,
                "direction": trade.direction,
                "entry_capital": trade.entry_capital,
            })

        return pd.DataFrame(trades_data)

    def calculate_trade_statistics(self) -> dict:
        """
        Calculate trade statistics.

        Returns:
            Dictionary with trade statistics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_holding_days": 0.0,
            }

        trades_df = self.get_trades_df()
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] <= 0]

        return {
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0.0,
            "avg_pnl": float(trades_df["pnl"].mean()),
            "avg_win": float(winning_trades["pnl"].mean()) if len(winning_trades) > 0 else 0.0,
            "avg_loss": float(losing_trades["pnl"].mean()) if len(losing_trades) > 0 else 0.0,
            "largest_win": float(trades_df["pnl"].max()),
            "largest_loss": float(trades_df["pnl"].min()),
            "avg_holding_days": float(trades_df["holding_days"].mean()),
            "profit_factor": abs(winning_trades["pnl"].sum() / losing_trades["pnl"].sum())
            if len(losing_trades) > 0 and losing_trades["pnl"].sum() != 0
            else float("inf") if len(winning_trades) > 0 else 0.0,
        }
