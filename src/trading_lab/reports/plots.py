"""Plotting utilities for reports."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("trading_lab.reports")


def plot_equity_curve(equity_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot equity curve.

    Args:
        equity_df: Equity curve DataFrame with columns: date, capital
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    equity_df = equity_df.sort_values("date")
    ax.plot(equity_df["date"], equity_df["capital"], linewidth=2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Capital")
    ax.set_title("Equity Curve")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved equity curve plot to {output_path}")


def plot_drawdown(equity_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot drawdown chart.

    Args:
        equity_df: Equity curve DataFrame with columns: date, capital
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    equity_df = equity_df.sort_values("date").copy()
    equity_df["cummax"] = equity_df["capital"].cummax()
    equity_df["drawdown"] = (equity_df["capital"] - equity_df["cummax"]) / equity_df["cummax"] * 100

    ax.fill_between(equity_df["date"], equity_df["drawdown"], 0, alpha=0.3, color="red")
    ax.plot(equity_df["date"], equity_df["drawdown"], linewidth=2, color="red")

    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Drawdown Chart")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved drawdown plot to {output_path}")


def plot_monthly_returns(equity_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot monthly returns heatmap.

    Args:
        equity_df: Equity curve DataFrame with columns: date, capital
        output_path: Output file path
    """
    equity_df = equity_df.sort_values("date").copy()
    equity_df["returns"] = equity_df["capital"].pct_change()
    equity_df["year"] = pd.to_datetime(equity_df["date"]).dt.year
    equity_df["month"] = pd.to_datetime(equity_df["date"]).dt.month

    # Calculate monthly returns
    monthly_returns = equity_df.groupby(["year", "month"])["returns"].sum() * 100

    # Create pivot table
    returns_pivot = monthly_returns.reset_index().pivot(
        index="year", columns="month", values="returns"
    )
    returns_pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    fig, ax = plt.subplots(figsize=(12, max(6, len(returns_pivot) * 0.5)))
    im = ax.imshow(returns_pivot.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

    # Set ticks
    ax.set_xticks(range(len(returns_pivot.columns)))
    ax.set_yticks(range(len(returns_pivot.index)))
    ax.set_xticklabels(returns_pivot.columns)
    ax.set_yticklabels(returns_pivot.index)

    # Add text annotations
    for i in range(len(returns_pivot.index)):
        for j in range(len(returns_pivot.columns)):
            value = returns_pivot.iloc[i, j]
            if not np.isnan(value):
                text_color = "white" if abs(value) > 5 else "black"
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color, fontsize=8)

    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_title("Monthly Returns Heatmap (%)")
    plt.colorbar(im, ax=ax, label="Return (%)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved monthly returns plot to {output_path}")


def plot_trade_distribution(trades_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot trade PnL distribution.

    Args:
        trades_df: Trades DataFrame with columns: pnl, pnl_pct
        output_path: Output file path
    """
    if trades_df.empty or "pnl" not in trades_df.columns:
        logger.warning("No trades data available for distribution plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PnL distribution
    axes[0].hist(trades_df["pnl"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("PnL ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Trade PnL Distribution")
    axes[0].grid(True, alpha=0.3)

    # PnL % distribution
    if "pnl_pct" in trades_df.columns:
        axes[1].hist(trades_df["pnl_pct"] * 100, bins=50, edgecolor="black", alpha=0.7)
        axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
        axes[1].set_xlabel("PnL (%)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Trade PnL % Distribution")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved trade distribution plot to {output_path}")


def plot_cumulative_returns(equity_df: pd.DataFrame, output_path: Path, benchmark: Optional[pd.DataFrame] = None) -> None:
    """
    Plot cumulative returns.

    Args:
        equity_df: Equity curve DataFrame with columns: date, capital
        output_path: Output file path
        benchmark: Optional benchmark equity curve for comparison
    """
    equity_df = equity_df.sort_values("date").copy()
    equity_df["cumulative_return"] = (equity_df["capital"] / equity_df["capital"].iloc[0] - 1) * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(equity_df["date"], equity_df["cumulative_return"], linewidth=2, label="Strategy")

    if benchmark is not None and not benchmark.empty:
        benchmark = benchmark.sort_values("date")
        benchmark["cumulative_return"] = (benchmark["capital"] / benchmark["capital"].iloc[0] - 1) * 100
        ax.plot(benchmark["date"], benchmark["cumulative_return"], linewidth=2, label="Benchmark", alpha=0.7)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Cumulative Returns")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved cumulative returns plot to {output_path}")

