"""System status and health check utilities."""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from trading_lab.config.settings import get_settings
from trading_lab.common.validation import check_data_quality, validate_price_data

logger = logging.getLogger("trading_lab.cli")


def check_system_status() -> Dict:
    """
    Check system status including data availability and quality.

    Returns:
        Dictionary with status information
    """
    settings = get_settings()
    status = {
        "directories": {},
        "data_availability": {},
        "data_quality": {},
        "issues": [],
        "warnings": [],
    }

    # Check directories
    dirs_to_check = {
        "data_dir": settings.data_dir,
        "raw_data": settings.get_raw_data_dir(),
        "processed_data": settings.get_processed_data_dir(),
        "artifacts": settings.get_artifacts_dir(),
        "prices": settings.get_prices_dir(),
        "unified": settings.get_unified_dir(),
        "features": settings.get_features_dir(),
        "models": settings.get_models_dir(),
        "backtests": settings.get_backtests_dir(),
        "reports": settings.get_reports_dir(),
        "logs": settings.get_logs_dir(),
    }

    for name, path in dirs_to_check.items():
        exists = path.exists()
        status["directories"][name] = {
            "path": str(path),
            "exists": exists,
            "writable": path.parent.exists() and Path(path.parent).is_dir(),
        }
        if not exists:
            status["warnings"].append(f"Directory {name} does not exist: {path}")

    # Check data availability
    try:
        from trading_lab.unify import unify_prices
        from trading_lab.common.io import load_dataframe

        # Check price data
        price_file = settings.get_unified_dir() / "prices_unified.parquet"
        if price_file.exists():
            price_df = load_dataframe(price_file)
            status["data_availability"]["prices"] = {
                "available": True,
                "rows": len(price_df),
                "tickers": price_df["ticker"].nunique() if not price_df.empty else 0,
                "date_range": {
                    "start": str(price_df["date"].min()) if not price_df.empty else None,
                    "end": str(price_df["date"].max()) if not price_df.empty else None,
                },
            }
            # Validate price data
            validation = validate_price_data(price_df)
            status["data_quality"]["prices"] = validation
            if not validation["is_valid"]:
                status["issues"].extend([f"Price data: {issue}" for issue in validation["issues"]])
        else:
            status["data_availability"]["prices"] = {"available": False}
            status["warnings"].append("No unified price data found")

        # Check features
        from trading_lab.features.feature_store import FeatureStore

        feature_store = FeatureStore()
        try:
            features_df = feature_store.load_features()
            status["data_availability"]["features"] = {
                "available": True,
                "rows": len(features_df),
                "feature_count": len(features_df.columns) - 2,  # Exclude date and ticker
            }
            quality = check_data_quality(features_df, check_missing=True, check_duplicates=True)
            status["data_quality"]["features"] = quality
            if quality["warnings"]:
                status["warnings"].extend([f"Features: {w}" for w in quality["warnings"]])
        except FileNotFoundError:
            status["data_availability"]["features"] = {"available": False}
            status["warnings"].append("No features data found")

        # Check models
        models_dir = settings.get_models_dir()
        model_files = list(models_dir.glob("classifier_*.joblib"))
        status["data_availability"]["models"] = {
            "available": len(model_files) > 0,
            "count": len(model_files),
            "models": [f.stem.replace("classifier_", "") for f in model_files],
        }
        if len(model_files) == 0:
            status["warnings"].append("No trained models found")

        # Check backtests
        backtests_dir = settings.get_backtests_dir()
        backtest_files = list(backtests_dir.glob("equity_curve_*.parquet"))
        status["data_availability"]["backtests"] = {
            "available": len(backtest_files) > 0,
            "count": len(backtest_files),
        }
        if len(backtest_files) == 0:
            status["warnings"].append("No backtest results found")

    except Exception as e:
        status["issues"].append(f"Error checking data: {e}")

    return status


def print_status_report(status: Dict) -> None:
    """Print formatted status report."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Overall status
    has_issues = len(status["issues"]) > 0
    has_warnings = len(status["warnings"]) > 0

    if has_issues:
        console.print("[bold red]⚠️ System has issues[/bold red]")
    elif has_warnings:
        console.print("[bold yellow]⚠️ System has warnings[/bold yellow]")
    else:
        console.print("[bold green]✓ System status OK[/bold green]")

    # Data availability
    console.print("\n[bold]Data Availability:[/bold]")
    data_table = Table(show_header=True, header_style="bold")
    data_table.add_column("Data Type")
    data_table.add_column("Status")
    data_table.add_column("Details")

    for data_type, info in status["data_availability"].items():
        if info.get("available"):
            details = []
            if "rows" in info:
                details.append(f"Rows: {info['rows']:,}")
            if "tickers" in info:
                details.append(f"Tickers: {info['tickers']}")
            if "feature_count" in info:
                details.append(f"Features: {info['feature_count']}")
            if "count" in info:
                details.append(f"Count: {info['count']}")
            data_table.add_row(data_type.title(), "[green]Available[/green]", ", ".join(details))
        else:
            data_table.add_row(data_type.title(), "[red]Not Available[/red]", "")

    console.print(data_table)

    # Issues and warnings
    if status["issues"]:
        console.print("\n[bold red]Issues:[/bold red]")
        for issue in status["issues"]:
            console.print(f"  • {issue}")

    if status["warnings"]:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in status["warnings"]:
            console.print(f"  • {warning}")
