"""Plotting utilities for reports."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
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

