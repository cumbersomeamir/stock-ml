"""Smoke tests for backtesting."""

import pandas as pd
import pytest

from trading_lab.backtest.costs import calculate_transaction_cost
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

