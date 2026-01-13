"""Backtesting engine."""

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from trading_lab.backtest.costs import calculate_transaction_cost
from trading_lab.backtest.metrics import calculate_metrics
from trading_lab.backtest.risk import apply_position_limits, check_circuit_breaker
from trading_lab.common.schemas import validate_dataframe_columns
from trading_lab.config.settings import get_settings
from trading_lab.unify import unify_prices

logger = logging.getLogger("trading_lab.backtest")


class BacktestEngine:
    """Backtesting engine with position management, costs, and risk controls."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_per_asset: Optional[float] = None,
        max_gross_exposure: Optional[float] = None,
        transaction_cost_bps: Optional[float] = None,
        slippage_bps: Optional[float] = None,
        max_drawdown_threshold: Optional[float] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Initial capital
            max_position_per_asset: Maximum position per asset (fraction of capital)
            max_gross_exposure: Maximum gross exposure (sum of absolute positions)
            transaction_cost_bps: Transaction cost in basis points
            slippage_bps: Slippage in basis points
            max_drawdown_threshold: Maximum drawdown threshold (circuit breaker)
        """
        self.settings = get_settings()

        self.initial_capital = initial_capital
        self.max_position_per_asset = max_position_per_asset or self.settings.max_position_per_asset
        self.max_gross_exposure = max_gross_exposure or self.settings.max_gross_exposure
        self.transaction_cost_bps = transaction_cost_bps or self.settings.transaction_cost_bps
        self.slippage_bps = slippage_bps or self.settings.slippage_bps
        self.max_drawdown_threshold = max_drawdown_threshold or self.settings.max_drawdown_threshold

    def run(
        self,
        signals_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        signal_function: Optional[Callable[[pd.Series], float]] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Run backtest.

        Args:
            signals_df: DataFrame with columns: date, ticker, and signal columns (e.g., prob_class, pred_class)
            price_df: Price DataFrame. If None, loads from unified data.
            signal_function: Optional function to convert signals to positions. If None, uses default.

        Returns:
            DataFrame with equity curve and metrics
        """
        logger.info("Starting backtest")

        # Validate signals DataFrame
        validate_dataframe_columns(signals_df, ["date", "ticker"], "backtest signals")

        # Load prices if not provided
        if price_df is None:
            price_df = unify_prices(force_refresh=False)

        if price_df.empty:
            raise ValueError("No price data available for backtesting")

        # Validate price DataFrame
        validate_dataframe_columns(price_df, ["date", "ticker", "adj_close"], "backtest prices")

        # Merge signals with prices
        data = signals_df.merge(price_df[["date", "ticker", "adj_close"]], on=["date", "ticker"], how="inner")
        
        if data.empty:
            raise ValueError("No overlapping data between signals and prices. Check date ranges and tickers.")
        
        data = data.sort_values(["date", "ticker"]).reset_index(drop=True)

        # Initialize state
        capital = self.initial_capital
        positions: Dict[str, float] = {}  # ticker -> position (fraction of capital)
        entry_prices: Dict[str, float] = {}  # ticker -> entry price
        peak_capital = self.initial_capital

        equity_curve = []

        # Default signal function: use prob_class if available, else pred_class
        if signal_function is None:

            def default_signal_function(row):
                if "prob_class" in row.index and pd.notna(row["prob_class"]):
                    # Use probability: long if > 0.6, short if < 0.4
                    if row["prob_class"] > 0.6:
                        return 0.1  # Long
                    elif row["prob_class"] < 0.4:
                        return -0.1  # Short
                    else:
                        return 0.0  # Neutral
                elif "pred_class" in row.index and pd.notna(row["pred_class"]):
                    return 0.1 if row["pred_class"] == 1 else -0.1 if row["pred_class"] == 0 else 0.0
                else:
                    return 0.0

            signal_function = default_signal_function

        # Process each date
        dates = sorted(data["date"].unique())
        total_dates = len(dates)

        logger.info(f"Processing {total_dates} trading days")

        circuit_breaker_triggered = False
        date_iterator = tqdm(dates, desc="Backtesting", disable=not show_progress) if show_progress else dates

        for date in date_iterator:
            date_data = data[data["date"] == date].copy()

            # Check circuit breaker
            if check_circuit_breaker(capital, peak_capital, self.max_drawdown_threshold):
                logger.warning(f"Circuit breaker triggered on {date}")
                circuit_breaker_triggered = True
                break

            # Update capital from previous positions
            for ticker in list(positions.keys()):
                if ticker in date_data["ticker"].values:
                    current_price = date_data[date_data["ticker"] == ticker]["adj_close"].iloc[0]
                    if ticker in entry_prices:
                        # Calculate PnL
                        position_value = capital * positions[ticker]
                        pnl_pct = (current_price - entry_prices[ticker]) / entry_prices[ticker]
                        pnl = position_value * pnl_pct
                        capital += pnl

            # Update peak capital
            peak_capital = max(peak_capital, capital)

            # Generate signals and update positions
            total_turnover = 0.0

            for _, row in date_data.iterrows():
                ticker = row["ticker"]
                price = row["adj_close"]

                # Get signal
                target_position = signal_function(row)

                # Apply position limits
                target_position = apply_position_limits(
                    target_position,
                    self.max_position_per_asset,
                    self.max_gross_exposure,
                    positions,
                )

                # Get current position
                current_position = positions.get(ticker, 0.0)
                position_change = target_position - current_position

                # Calculate transaction costs
                if abs(position_change) > 1e-6:
                    position_value = capital * abs(position_change)
                    cost = calculate_transaction_cost(
                        position_change * capital,  # Position change in currency
                        price,
                        self.transaction_cost_bps,
                        self.slippage_bps,
                    )
                    capital -= cost
                    total_turnover += abs(position_change)

                    # Update position
                    if abs(target_position) < 1e-6:
                        positions.pop(ticker, None)
                        entry_prices.pop(ticker, None)
                    else:
                        positions[ticker] = target_position
                        # Update entry price (with slippage in basis points)
                        slippage_multiplier = self.slippage_bps / 10000.0
                        if target_position > 0:
                            entry_prices[ticker] = price * (1 + slippage_multiplier)
                        else:
                            entry_prices[ticker] = price * (1 - slippage_multiplier)

            # Record equity curve point
            equity_curve.append(
                {
                    "date": date,
                    "capital": capital,
                    "turnover": total_turnover,
                }
            )

            if circuit_breaker_triggered:
                break

        # Calculate final PnL for open positions
        if dates and not circuit_breaker_triggered:
            last_date = dates[-1]
            last_data = data[data["date"] == last_date]
            for ticker, position in positions.items():
                if ticker in last_data["ticker"].values:
                    final_price = last_data[last_data["ticker"] == ticker]["adj_close"].iloc[0]
                    if ticker in entry_prices:
                        position_value = capital * position
                        pnl_pct = (final_price - entry_prices[ticker]) / entry_prices[ticker]
                        pnl = position_value * pnl_pct
                        capital += pnl

        # Create equity curve DataFrame
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df["date"] = pd.to_datetime(equity_df["date"])

        # Calculate metrics
        metrics = calculate_metrics(equity_df, self.initial_capital)

        logger.info(f"Backtest complete: Final capital={capital:.2f}, Return={metrics.get('total_return', 0)*100:.2f}%")

        return equity_df

