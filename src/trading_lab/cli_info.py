"""Information and statistics commands."""

import logging
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from trading_lab.config.settings import get_settings
from trading_lab.features.feature_store import FeatureStore
from trading_lab.unify import unify_prices

logger = logging.getLogger("trading_lab.cli")
console = Console()


def print_data_summary() -> None:
    """Print summary of available data."""
    settings = get_settings()
    
    console.print("\n[bold]Data Summary[/bold]")
    
    # Price data
    try:
        price_df = unify_prices(force_refresh=False)
        if not price_df.empty:
            console.print(f"\n[green]Price Data:[/green]")
            console.print(f"  Rows: {len(price_df):,}")
            console.print(f"  Tickers: {price_df['ticker'].nunique()}")
            console.print(f"  Date Range: {price_df['date'].min().date()} to {price_df['date'].max().date()}")
            console.print(f"  Days: {(price_df['date'].max() - price_df['date'].min()).days}")
        else:
            console.print("\n[yellow]Price Data:[/yellow] No data available")
    except Exception as e:
        console.print(f"\n[yellow]Price Data:[/yellow] Error loading: {e}")
    
    # Features
    try:
        feature_store = FeatureStore()
        features_df = feature_store.load_features()
        if not features_df.empty:
            console.print(f"\n[green]Features:[/green]")
            console.print(f"  Rows: {len(features_df):,}")
            console.print(f"  Features: {len(features_df.columns) - 2}")  # Exclude date and ticker
            console.print(f"  Date Range: {features_df['date'].min().date()} to {features_df['date'].max().date()}")
        else:
            console.print("\n[yellow]Features:[/yellow] No data available")
    except FileNotFoundError:
        console.print("\n[yellow]Features:[/yellow] Not available (run build-features)")
    except Exception as e:
        console.print(f"\n[yellow]Features:[/yellow] Error loading: {e}")
    
    # Models
    models_dir = settings.get_models_dir()
    model_files = list(models_dir.glob("classifier_*.joblib"))
    if model_files:
        console.print(f"\n[green]Models:[/green]")
        for model_file in model_files:
            model_name = model_file.stem.replace("classifier_", "")
            console.print(f"  • {model_name}")
    else:
        console.print("\n[yellow]Models:[/yellow] No trained models found")
    
    # Backtests
    backtests_dir = settings.get_backtests_dir()
    backtest_files = list(backtests_dir.glob("equity_curve_*.parquet"))
    if backtest_files:
        console.print(f"\n[green]Backtests:[/green] {len(backtest_files)} results found")
        # Show most recent 3
        sorted_files = sorted(backtest_files, key=lambda p: p.stat().st_mtime, reverse=True)[:3]
        for bf in sorted_files:
            console.print(f"  • {bf.stem.replace('equity_curve_', '')}")
    else:
        console.print("\n[yellow]Backtests:[/yellow] No backtest results found")


def print_feature_statistics() -> None:
    """Print statistics about features."""
    try:
        feature_store = FeatureStore()
        features_df = feature_store.load_features()
        
        if features_df.empty:
            console.print("[yellow]No features available[/yellow]")
            return
        
        # Get numeric feature columns (exclude date and ticker)
        numeric_cols = features_df.select_dtypes(include=[float, int]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ["date", "ticker"]]
        
        if not numeric_cols:
            console.print("[yellow]No numeric features found[/yellow]")
            return
        
        console.print(f"\n[bold]Feature Statistics (showing top 10 features by variance)[/bold]")
        
        # Calculate variance and sort
        variances = features_df[numeric_cols].var().sort_values(ascending=False)
        top_features = variances.head(10).index.tolist()
        
        table = Table(show_header=True, header_style="bold")
        table.add_column("Feature", style="cyan")
        table.add_column("Mean", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Missing %", justify="right")
        
        for feature in top_features:
            stats = features_df[feature].describe()
            missing_pct = (features_df[feature].isna().sum() / len(features_df)) * 100
            
            table.add_row(
                feature,
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['min']:.4f}",
                f"{stats['max']:.4f}",
                f"{missing_pct:.2f}%"
            )
        
        console.print(table)
        
    except FileNotFoundError:
        console.print("[yellow]Features not found. Run build-features first.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error loading features: {e}[/red]")
        logger.exception("Error in print_feature_statistics")
