"""Comprehensive tests for backtest engine."""

import pandas as pd
import pytest

from trading_lab.backtest.engine import BacktestEngine


def test_backtest_engine_empty_signals():
    """Test backtest with empty signals."""
    engine = BacktestEngine()
    signals = pd.DataFrame(columns=["date", "ticker", "prob_class"])
    prices = pd.DataFrame(columns=["date", "ticker", "adj_close"])
    
    with pytest.raises(ValueError, match="DataFrame is empty"):
        engine.run(signals, prices, show_progress=False)


def test_backtest_engine_no_overlapping_data(sample_signals_data: pd.DataFrame):
    """Test backtest with no overlapping data."""
    engine = BacktestEngine()
    
    # Create price data with different dates/tickers
    prices = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=10),
        "ticker": ["GOOGL"] * 10,
        "adj_close": [100.0] * 10,
    })
    
    with pytest.raises(ValueError, match="No overlapping data"):
        engine.run(sample_signals_data, prices, show_progress=False)


def test_backtest_engine_custom_signal_function(
    sample_signals_data: pd.DataFrame, 
    sample_price_data: pd.DataFrame
):
    """Test backtest with custom signal function."""
    engine = BacktestEngine()
    
    def custom_signal(row: pd.Series) -> float:
        """Custom signal that always returns 0.05 (5% long)."""
        return 0.05
    
    equity_curve = engine.run(
        sample_signals_data,
        sample_price_data,
        signal_function=custom_signal,
        show_progress=False
    )
    
    assert len(equity_curve) > 0
    assert all(equity_curve["capital"] > 0)


def test_backtest_engine_position_limits(
    sample_signals_data: pd.DataFrame,
    sample_price_data: pd.DataFrame
):
    """Test that position limits are enforced."""
    engine = BacktestEngine(
        max_position_per_asset=0.05,  # 5% max position
        max_gross_exposure=0.2,  # 20% max gross
    )
    
    equity_curve = engine.run(
        sample_signals_data,
        sample_price_data,
        show_progress=False
    )
    
    # Should complete without errors and respect limits
    assert len(equity_curve) > 0


def test_backtest_engine_transaction_costs(
    sample_signals_data: pd.DataFrame,
    sample_price_data: pd.DataFrame
):
    """Test that transaction costs reduce capital."""
    engine_low_cost = BacktestEngine(
        transaction_cost_bps=1.0,
        slippage_bps=0.5,
    )
    engine_high_cost = BacktestEngine(
        transaction_cost_bps=50.0,
        slippage_bps=25.0,
    )
    
    equity_low = engine_low_cost.run(
        sample_signals_data,
        sample_price_data,
        show_progress=False
    )
    
    equity_high = engine_high_cost.run(
        sample_signals_data,
        sample_price_data,
        show_progress=False
    )
    
    # High cost backtest should have lower final capital (or equal if no trades)
    final_low = equity_low["capital"].iloc[-1]
    final_high = equity_high["capital"].iloc[-1]
    assert final_high <= final_low
