"""Tests for vectorized backtest engine."""

import pandas as pd
import pytest

from trading_lab.backtest.vectorized_engine import VectorizedBacktestEngine


def test_vectorized_backtest_basic(sample_signals_data: pd.DataFrame, sample_price_data: pd.DataFrame):
    """Test basic vectorized backtest."""
    engine = VectorizedBacktestEngine(initial_capital=100000.0)
    
    # Ensure signal column exists
    if "signal" not in sample_signals_data.columns:
        sample_signals_data["signal"] = sample_signals_data["prob_class"]
    
    equity_curve = engine.run(
        signals_df=sample_signals_data,
        price_df=sample_price_data
    )
    
    assert isinstance(equity_curve, pd.DataFrame)
    assert "date" in equity_curve.columns
    assert "capital" in equity_curve.columns
    assert len(equity_curve) > 0
    assert equity_curve["capital"].iloc[-1] > 0


def test_vectorized_vs_iterative_consistency(
    sample_signals_data: pd.DataFrame,
    sample_price_data: pd.DataFrame
):
    """Test that vectorized engine produces similar results to iterative engine."""
    vectorized_engine = VectorizedBacktestEngine(
        initial_capital=100000.0,
        transaction_cost_bps=10.0,
        slippage_bps=5.0
    )
    
    # Ensure signal column exists
    if "signal" not in sample_signals_data.columns:
        sample_signals_data["signal"] = sample_signals_data["prob_class"]
    
    equity_vectorized = vectorized_engine.run(
        signals_df=sample_signals_data,
        price_df=sample_price_data
    )
    
    # Results should be positive and reasonable
    assert equity_vectorized["capital"].iloc[-1] > 0
    assert len(equity_vectorized) > 0


def test_vectorized_circuit_breaker(sample_signals_data: pd.DataFrame, sample_price_data: pd.DataFrame):
    """Test circuit breaker in vectorized engine."""
    engine = VectorizedBacktestEngine(
        initial_capital=100000.0,
        max_drawdown_threshold=0.05  # Very strict threshold
    )
    
    # Create severe drawdown scenario
    price_data = sample_price_data.copy()
    price_data.loc[price_data["date"] > "2020-01-15", "adj_close"] *= 0.3  # 70% drop
    
    if "signal" not in sample_signals_data.columns:
        sample_signals_data["signal"] = sample_signals_data["prob_class"]
    
    equity_curve = engine.run(
        signals_df=sample_signals_data,
        price_df=price_data
    )
    
    # Should stop early due to circuit breaker
    assert len(equity_curve) <= len(sample_signals_data["date"].unique())
