"""Command-line interface for trading lab."""

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from trading_lab.backtest.walk_forward import walk_forward_backtest
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
    console.print("[bold green]Downloading price data...[/bold green]")
    ticker_list = [t.strip() for t in tickers.split(",")]

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
):
    """
    Generate backtest report with metrics and plots.

    Examples:
        trading-lab report
        trading-lab report --strategy supervised_prob_threshold
    """
    console.print("[bold green]Generating report...[/bold green]")

    try:
        report_data = generate_report(strategy=strategy)
        console.print("[bold green]✓ Report generated[/bold green]")
        console.print("\n[bold]Summary:[/bold]")
        for key, value in report_data["summary"].items():
            console.print(f"  {key}: {value}")
        console.print(f"\nReport saved to: {report_data['plots']['equity_curve']}")
    except Exception as e:
        console.print(f"[bold red]Error generating report: {e}[/bold red]")
        logger.exception("Error generating report")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

