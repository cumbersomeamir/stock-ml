"""Backtest performance metrics."""

import numpy as np
import pandas as pd


def calculate_metrics(equity_curve: pd.DataFrame, initial_capital: float) -> dict:
    """
    Calculate performance metrics from equity curve.

    Args:
        equity_curve: DataFrame with columns: date, capital, returns
        initial_capital: Initial capital

    Returns:
        Dictionary with performance metrics
    """
    if len(equity_curve) == 0:
        return {}

    equity_curve = equity_curve.copy()
    equity_curve = equity_curve.sort_values("date").reset_index(drop=True)

    # Returns
    equity_curve["returns"] = equity_curve["capital"].pct_change().fillna(0)
    total_return = (equity_curve["capital"].iloc[-1] - initial_capital) / initial_capital

    # CAGR
    days = (equity_curve["date"].iloc[-1] - equity_curve["date"].iloc[0]).days
    years = days / 365.25
    if years > 0:
        cagr = ((equity_curve["capital"].iloc[-1] / initial_capital) ** (1 / years)) - 1
    else:
        cagr = 0.0

    # Sharpe Ratio (annualized)
    if equity_curve["returns"].std() > 0:
        sharpe = np.sqrt(252) * equity_curve["returns"].mean() / equity_curve["returns"].std()
    else:
        sharpe = 0.0

    # Sortino Ratio (annualized, using downside deviation)
    downside_returns = equity_curve["returns"][equity_curve["returns"] < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino = np.sqrt(252) * equity_curve["returns"].mean() / downside_returns.std()
    else:
        sortino = 0.0

    # Maximum Drawdown
    equity_curve["cummax"] = equity_curve["capital"].cummax()
    equity_curve["drawdown"] = (equity_curve["capital"] - equity_curve["cummax"]) / equity_curve["cummax"]
    max_drawdown = equity_curve["drawdown"].min()

    # Win Rate (from trades if available in attrs, otherwise from trade_pnl column)
    if hasattr(equity_curve, "attrs") and "trade_stats" in equity_curve.attrs:
        win_rate = equity_curve.attrs["trade_stats"].get("win_rate", np.nan)
    elif "trade_pnl" in equity_curve.columns:
        trades = equity_curve[equity_curve["trade_pnl"] != 0]
        if len(trades) > 0:
            win_rate = (trades["trade_pnl"] > 0).sum() / len(trades)
        else:
            win_rate = 0.0
    else:
        win_rate = np.nan

    # Turnover (if available)
    if "turnover" in equity_curve.columns:
        turnover = equity_curve["turnover"].sum()
    else:
        turnover = np.nan

    metrics = {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate) if not np.isnan(win_rate) else None,
        "turnover": float(turnover) if not np.isnan(turnover) else None,
        "final_capital": float(equity_curve["capital"].iloc[-1]),
        "initial_capital": float(initial_capital),
    }

    return metrics

