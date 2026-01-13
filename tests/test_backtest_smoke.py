"""Smoke tests for backtesting."""

import pandas as pd
import pytest

from trading_lab.backtest.costs import calculate_transaction_cost
from trading_lab.backtest.engine import BacktestEngine
from trading_lab.backtest.risk import apply_position_limits, check_circuit_breaker


def test_calculate_transaction_cost():
    """Test transaction cost calculation."""
    cost = calculate_transaction_cost(100.0, 50.0, 10.0, 5.0)
    assert cost > 0
    assert cost < 1.0  # Should be reasonable


def test_check_circuit_breaker():
    """Test circuit breaker logic."""
    # Normal case
    assert not check_circuit_breaker(100000, 100000, 0.2)

    # Drawdown case
    assert check_circuit_breaker(70000, 100000, 0.2)  # 30% drawdown > 20% threshold

    # Edge case
    assert not check_circuit_breaker(100000, 0, 0.2)


def test_apply_position_limits():
    """Test position limit application."""
    # Normal case
    pos = apply_position_limits(0.05, 0.1, 1.0, {})
    assert abs(pos) <= 0.1

    # Limit exceeded
    pos = apply_position_limits(0.2, 0.1, 1.0, {})
    assert abs(pos) <= 0.1

    # Gross exposure limit
    current_positions = {"TICKER1": 0.8, "TICKER2": 0.1}
    pos = apply_position_limits(0.2, 0.1, 1.0, current_positions)
    assert abs(pos) <= 0.1  # Only 0.1 available


def test_backtest_engine_basic(sample_signals_data: pd.DataFrame, sample_price_data: pd.DataFrame):
    """Test basic backtest engine functionality."""
    engine = BacktestEngine(initial_capital=100000.0)
    
    # Run backtest
    equity_curve = engine.run(
        signals_df=sample_signals_data,
        price_df=sample_price_data,
        show_progress=False
    )
    
    assert isinstance(equity_curve, pd.DataFrame)
    assert "date" in equity_curve.columns
    assert "capital" in equity_curve.columns
    assert len(equity_curve) > 0
    assert equity_curve["capital"].iloc[-1] > 0


def test_backtest_engine_circuit_breaker(sample_signals_data: pd.DataFrame, sample_price_data: pd.DataFrame):
    """Test circuit breaker functionality."""
    engine = BacktestEngine(
        initial_capital=100000.0,
        max_drawdown_threshold=0.1  # Very low threshold
    )
    
    # Modify price data to cause large drawdown
    price_data = sample_price_data.copy()
    price_data.loc[price_data["date"] > "2020-02-01", "adj_close"] *= 0.5  # 50% drop
    
    equity_curve = engine.run(
        signals_df=sample_signals_data,
        price_df=price_data,
        show_progress=False
    )
    
    # Should stop early due to circuit breaker
    assert len(equity_curve) < len(sample_signals_data["date"].unique())

