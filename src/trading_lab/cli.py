"""Command-line interface for trading lab."""

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from trading_lab.backtest.walk_forward import walk_forward_backtest
from trading_lab.cli_info import print_data_summary, print_feature_statistics
from trading_lab.cli_status import check_system_status, print_status_report
from trading_lab.common.logging import setup_logging
from trading_lab.data_sources.prices.yfinance_fetcher import YFinanceFetcher
from trading_lab.features.build_features import build_features as build_features_fn
from trading_lab.models.supervised.train_supervised import train_supervised
from trading_lab.models.unsupervised.regime_detection import detect_regimes
from trading_lab.reports.generate_report import generate_report
from trading_lab.unify import unify_events, unify_prices

app = typer.Typer(help="Trading Lab - ML-driven trading research system")
console = Console()

# Set up logging
logger = setup_logging()


@app.command()
def download_prices(
    tickers: str = typer.Option(..., "--tickers", "-t", help="Comma-separated ticker symbols"),
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    force_refresh: bool = typer.Option(False, "--force", "-f", help="Force re-download"),
):
    """
    Download price data for given tickers.

    Examples:
        trading-lab download-prices --tickers "AAPL,MSFT" --start 2018-01-01 --end 2024-12-31
        trading-lab download-prices --tickers "RELIANCE.NS,TCS.NS" --start 2018-01-01 --end 2024-12-31
    """
    from datetime import datetime

    # Validate date formats
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
    except ValueError as e:
        console.print(f"[bold red]Invalid date format. Use YYYY-MM-DD format. Error: {e}[/bold red]")
        raise typer.Exit(1)

    # Validate date range
    if start_date >= end_date:
        console.print("[bold red]Start date must be before end date[/bold red]")
        raise typer.Exit(1)

    # Validate tickers
    ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        console.print("[bold red]No valid tickers provided[/bold red]")
        raise typer.Exit(1)

    console.print("[bold green]Downloading price data...[/bold green]")

    fetcher = YFinanceFetcher()
    df = fetcher.fetch(ticker_list, start, end, force_refresh=force_refresh)

    if df.empty:
        console.print("[bold red]No data downloaded[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]✓ Downloaded data for {df['ticker'].nunique()} tickers, {len(df)} rows[/bold green]")


@app.command()
def build_features(
    force_refresh: bool = typer.Option(False, "--force", "-f", help="Force rebuild"),
    lookback_days: int = typer.Option(60, "--lookback", "-l", help="Lookback window for features"),
):
    """
    Build features from unified data.

    Examples:
        trading-lab build-features
        trading-lab build-features --force --lookback 90
    """
    console.print("[bold green]Building features...[/bold green]")

    try:
        df = build_features_fn(force_refresh=force_refresh, lookback_days=lookback_days)
        console.print(f"[bold green]✓ Built {len(df)} feature rows with {len(df.columns) - 2} features[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error building features: {e}[/bold red]")
        logger.exception("Error building features")
        raise typer.Exit(1)


@app.command(name="train-supervised")
def train_supervised_cmd(
    model_name: str = typer.Option(
        "gradient_boosting", "--model", "-m", help="Model name (logistic, random_forest, gradient_boosting, lightgbm)"
    ),
    force_refresh: bool = typer.Option(False, "--force", "-f", help="Force retrain"),
    test_size: float = typer.Option(0.2, "--test-size", "-t", help="Test set proportion"),
):
    """
    Train supervised models for classification and regression.

    Examples:
        trading-lab train-supervised
        trading-lab train-supervised --model lightgbm --test-size 0.3
    """
    # Validate test_size
    if not 0.0 < test_size < 1.0:
        console.print("[bold red]test-size must be between 0.0 and 1.0[/bold red]")
        raise typer.Exit(1)

    # Validate model_name
    valid_models = ["logistic", "random_forest", "gradient_boosting", "lightgbm"]
    if model_name not in valid_models:
        console.print(f"[bold red]Invalid model name. Must be one of: {', '.join(valid_models)}[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Training supervised models: {model_name}...[/bold green]")

    try:
        metrics = train_supervised(model_name=model_name, force_refresh=force_refresh, test_size=test_size)
        console.print("[bold green]✓ Training complete[/bold green]")
        console.print(f"  Classification Accuracy: {metrics['classification']['accuracy']:.4f}")
        if metrics['classification']['auc']:
            console.print(f"  Classification AUC: {metrics['classification']['auc']:.4f}")
        console.print(f"  Regression RMSE: {metrics['regression']['rmse']:.6f}")
    except Exception as e:
        console.print(f"[bold red]Error training models: {e}[/bold red]")
        logger.exception("Error training models")
        raise typer.Exit(1)


@app.command()
def train_regime_detection(
    n_regimes: int = typer.Option(4, "--n-regimes", "-n", help="Number of regimes"),
    method: str = typer.Option("kmeans", "--method", "-m", help="Method (kmeans, gmm)"),
    window: int = typer.Option(20, "--window", "-w", help="Rolling window"),
    force_refresh: bool = typer.Option(False, "--force", "-f", help="Force redetection"),
):
    """
    Detect market regimes using unsupervised learning.

    Examples:
        trading-lab train-regime-detection
        trading-lab train-regime-detection --n-regimes 5 --method gmm
    """
    console.print(f"[bold green]Detecting regimes: {method}, n={n_regimes}...[/bold green]")

    try:
        df = detect_regimes(n_regimes=n_regimes, method=method, window=window, force_refresh=force_refresh)
        console.print(f"[bold green]✓ Regime detection complete[/bold green]")
        console.print(f"  Regime distribution: {df['regime'].value_counts().to_dict()}")
    except Exception as e:
        console.print(f"[bold red]Error detecting regimes: {e}[/bold red]")
        logger.exception("Error detecting regimes")
        raise typer.Exit(1)


@app.command()
def backtest(
    strategy: str = typer.Option(
        "supervised_prob_threshold", "--strategy", "-s", help="Strategy name"
    ),
    model_name: str = typer.Option("gradient_boosting", "--model", "-m", help="Model name"),
    train_window_years: int = typer.Option(None, "--train-window-years", help="Training window (years)"),
    test_window_months: int = typer.Option(None, "--test-window-months", help="Test window (months)"),
):
    """
    Run walk-forward backtest.

    Examples:
        trading-lab backtest
        trading-lab backtest --strategy supervised_prob_threshold --model lightgbm
    """
    console.print(f"[bold green]Running backtest: {strategy}...[/bold green]")

    try:
        results = walk_forward_backtest(
            train_window_years=train_window_years,
            test_window_months=test_window_months,
            strategy=strategy,
            model_name=model_name,
        )
        console.print("[bold green]✓ Backtest complete[/bold green]")
        metrics = results.get("metrics", {})
        if metrics:
            console.print(f"  Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            console.print(f"  CAGR: {metrics.get('cagr', 0)*100:.2f}%")
            console.print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            console.print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
    except Exception as e:
        console.print(f"[bold red]Error running backtest: {e}[/bold red]")
        logger.exception("Error running backtest")
        raise typer.Exit(1)


@app.command()
def report(
    strategy: Optional[str] = typer.Option(None, "--strategy", "-s", help="Strategy name (uses most recent if not specified)"),
    export_csv: bool = typer.Option(False, "--export-csv", help="Export equity curve and trades to CSV"),
):
    """
    Generate backtest report with metrics and plots.

    Examples:
        trading-lab report
        trading-lab report --strategy supervised_prob_threshold
        trading-lab report --export-csv
    """
    console.print("[bold green]Generating report...[/bold green]")

    try:
        report_data = generate_report(strategy=strategy)
        console.print("[bold green]✓ Report generated[/bold green]")
        console.print("\n[bold]Summary:[/bold]")
        for key, value in report_data["summary"].items():
            console.print(f"  {key}: {value}")
        
        if report_data.get("monthly_stats"):
            console.print("\n[bold]Monthly Statistics:[/bold]")
            monthly = report_data["monthly_stats"]
            console.print(f"  Best Month: {monthly.get('best_month', 0):.2f}%")
            console.print(f"  Worst Month: {monthly.get('worst_month', 0):.2f}%")
            console.print(f"  Average Monthly Return: {monthly.get('avg_monthly_return', 0):.2f}%")
            console.print(f"  Positive Months: {monthly.get('positive_months', 0)}/{monthly.get('total_months', 0)}")
        
        console.print(f"\nReport saved to: {report_data['plots']['equity_curve']}")
        
        # Export CSV if requested
        if export_csv:
            from trading_lab.config.settings import get_settings
            from trading_lab.common.io import load_dataframe
            from pathlib import Path
            
            settings = get_settings()
            reports_dir = settings.get_reports_dir()
            
            # Find the equity file used in the report
            equity_files = list(settings.get_backtests_dir().glob(f"equity_curve_{strategy or '*'}_*.parquet"))
            if not equity_files:
                equity_files = list(settings.get_backtests_dir().glob("equity_curve_*.parquet"))
            
            if equity_files:
                # Use the most recent one or the one matching strategy
                if strategy:
                    matching = [f for f in equity_files if strategy in f.stem]
                    equity_file = max(matching, key=lambda p: p.stat().st_mtime) if matching else max(equity_files, key=lambda p: p.stat().st_mtime)
                else:
                    equity_file = max(equity_files, key=lambda p: p.stat().st_mtime)
                
                equity_df = load_dataframe(equity_file)
                
                # Export equity curve
                report_id = equity_file.stem.replace("equity_curve_", "")
                csv_path = reports_dir / f"equity_curve_{report_id}.csv"
                equity_df.to_csv(csv_path, index=False)
                console.print(f"[green]✓ Exported equity curve to: {csv_path}[/green]")
                
                # Export trades if available
                if hasattr(equity_df, "attrs") and "trades" in equity_df.attrs:
                    trades_df = equity_df.attrs["trades"]
                    if not trades_df.empty:
                        trades_csv = reports_dir / f"trades_{report_id}.csv"
                        trades_df.to_csv(trades_csv, index=False)
                        console.print(f"[green]✓ Exported trades to: {trades_csv}[/green]")
            else:
                console.print("[yellow]Warning: Could not find equity curve file to export[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error generating report: {e}[/bold red]")
        logger.exception("Error generating report")
        raise typer.Exit(1)


@app.command(name="status")
def status_cmd():
    """
    Check system status and data availability.

    Examples:
        trading-lab status
    """
    console.print("[bold green]Checking system status...[/bold green]")
    
    try:
        status = check_system_status()
        print_status_report(status)
    except Exception as e:
        console.print(f"[bold red]Error checking status: {e}[/bold red]")
        logger.exception("Error checking status")
        raise typer.Exit(1)


@app.command(name="info")
def info_cmd(
    features: bool = typer.Option(False, "--features", help="Show feature statistics"),
):
    """
    Show data information and statistics.

    Examples:
        trading-lab info
        trading-lab info --features
    """
    try:
        print_data_summary()
        
        if features:
            print_feature_statistics()
    except Exception as e:
        console.print(f"[bold red]Error getting info: {e}[/bold red]")
        logger.exception("Error in info command")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

