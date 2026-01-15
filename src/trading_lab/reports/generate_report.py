"""Generate backtest reports."""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from trading_lab.common.io import load_dataframe, load_json, save_json
from trading_lab.config.settings import get_settings
from trading_lab.reports.plots import (
    plot_cumulative_returns,
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_returns,
    plot_trade_distribution,
)

logger = logging.getLogger("trading_lab.reports")


def generate_report(strategy: Optional[str] = None) -> Dict:
    """
    Generate backtest report with metrics and plots.

    Args:
        strategy: Strategy name. If None, uses the most recent backtest.

    Returns:
        Dictionary with report data
    """
    logger.info("Generating backtest report")

    settings = get_settings()
    backtests_dir = settings.get_backtests_dir()
    reports_dir = settings.get_reports_dir()

    # Find most recent backtest
    if strategy is None:
        equity_files = list(backtests_dir.glob("equity_curve_*.parquet"))
        if not equity_files:
            raise FileNotFoundError("No backtest results found. Run backtest first.")
        equity_file = max(equity_files, key=lambda p: p.stat().st_mtime)
        strategy = equity_file.stem.replace("equity_curve_", "").split("_")[0]
    else:
        equity_files = list(backtests_dir.glob(f"equity_curve_{strategy}_*.parquet"))
        if not equity_files:
            raise FileNotFoundError(f"No backtest results found for strategy: {strategy}")
        equity_file = max(equity_files, key=lambda p: p.stat().st_mtime)

    # Load equity curve
    equity_df = load_dataframe(equity_file)
    logger.info(f"Loaded equity curve: {len(equity_df)} rows")

    # Load metrics
    metrics_files = list(backtests_dir.glob(f"metrics_{strategy}_*.json"))
    if metrics_files:
        metrics_file = max(metrics_files, key=lambda p: p.stat().st_mtime)
        metrics = load_json(metrics_file)
    else:
        logger.warning("No metrics file found, calculating from equity curve")
        from trading_lab.backtest.metrics import calculate_metrics

        initial_capital = equity_df["capital"].iloc[0] if not equity_df.empty else 100000.0
        metrics = calculate_metrics(equity_df, initial_capital)

    # Generate plots
    report_id = equity_file.stem.replace("equity_curve_", "")
    equity_plot_path = reports_dir / f"equity_curve_{report_id}.png"
    drawdown_plot_path = reports_dir / f"drawdown_{report_id}.png"
    monthly_returns_path = reports_dir / f"monthly_returns_{report_id}.png"
    cumulative_returns_path = reports_dir / f"cumulative_returns_{report_id}.png"
    trade_dist_path = reports_dir / f"trade_distribution_{report_id}.png"

    plot_equity_curve(equity_df, equity_plot_path)
    plot_drawdown(equity_df, drawdown_plot_path)
    plot_monthly_returns(equity_df, monthly_returns_path)
    plot_cumulative_returns(equity_df, cumulative_returns_path)

    # Plot trade distribution if available
    if hasattr(equity_df, "attrs") and "trades" in equity_df.attrs:
        trades_df = equity_df.attrs["trades"]
        if not trades_df.empty:
            plot_trade_distribution(trades_df, trade_dist_path)

    # Add monthly returns statistics
    equity_df_sorted = equity_df.sort_values("date").copy()
    equity_df_sorted["returns"] = equity_df_sorted["capital"].pct_change()
    monthly_returns_stats = {}
    if not equity_df_sorted.empty:
        equity_df_sorted["year_month"] = pd.to_datetime(equity_df_sorted["date"]).dt.to_period("M")
        monthly_returns = equity_df_sorted.groupby("year_month")["returns"].sum()
        monthly_returns_stats = {
            "best_month": float(monthly_returns.max() * 100),
            "worst_month": float(monthly_returns.min() * 100),
            "avg_monthly_return": float(monthly_returns.mean() * 100),
            "positive_months": int((monthly_returns > 0).sum()),
            "total_months": len(monthly_returns),
        }

    # Create report
    report = {
        "strategy": strategy,
        "metrics": metrics,
        "plots": {
            "equity_curve": str(equity_plot_path),
            "drawdown": str(drawdown_plot_path),
            "monthly_returns": str(monthly_returns_path),
            "cumulative_returns": str(cumulative_returns_path),
        },
        "summary": {
            "total_return": f"{metrics.get('total_return', 0)*100:.2f}%",
            "cagr": f"{metrics.get('cagr', 0)*100:.2f}%",
            "sharpe_ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
            "max_drawdown": f"{metrics.get('max_drawdown', 0)*100:.2f}%",
            "total_trades": metrics.get("total_trades", "N/A"),
            "win_rate": f"{metrics.get('win_rate', 0)*100:.2f}%" if metrics.get('win_rate') is not None else "N/A",
        },
        "monthly_stats": monthly_returns_stats,
    }

    # Add trade distribution plot if available
    if hasattr(equity_df, "attrs") and "trades" in equity_df.attrs:
        trades_df = equity_df.attrs["trades"]
        if not trades_df.empty:
            report["plots"]["trade_distribution"] = str(trade_dist_path)

    # Save report
    report_path = reports_dir / f"report_{report_id}.json"
    save_json(report, report_path)

    logger.info(f"Report generated: {report_path}")
    logger.info(f"  Total Return: {report['summary']['total_return']}")
    logger.info(f"  CAGR: {report['summary']['cagr']}")
    logger.info(f"  Sharpe Ratio: {report['summary']['sharpe_ratio']}")
    logger.info(f"  Max Drawdown: {report['summary']['max_drawdown']}")

    return report

