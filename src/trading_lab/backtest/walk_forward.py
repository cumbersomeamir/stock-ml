"""Walk-forward backtesting with rolling train/test windows."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from trading_lab.backtest.engine import BacktestEngine
from trading_lab.backtest.metrics import calculate_metrics
from trading_lab.common.io import save_dataframe, save_json
from trading_lab.config.settings import get_settings
from trading_lab.features.feature_store import FeatureStore
from trading_lab.models.supervised.predict_supervised import predict_supervised

logger = logging.getLogger("trading_lab.backtest")


def walk_forward_backtest(
    train_window_years: Optional[int] = None,
    test_window_months: Optional[int] = None,
    step_months: Optional[int] = None,
    strategy: str = "supervised_prob_threshold",
    model_name: str = "gradient_boosting",
) -> Dict:
    """
    Run walk-forward backtest with rolling windows.

    Args:
        train_window_years: Training window in years
        test_window_months: Test window in months
        step_months: Step size in months
        strategy: Strategy name ('supervised_prob_threshold')
        model_name: Model name for supervised strategies

    Returns:
        Dictionary with backtest results
    """
    settings = get_settings()
    train_window_years = train_window_years or settings.train_window_years
    test_window_months = test_window_months or settings.test_window_months
    step_months = step_months or settings.step_months

    logger.info(
        f"Walk-forward backtest: train={train_window_years}y, test={test_window_months}m, step={step_months}m"
    )

    # Load features
    feature_store = FeatureStore()
    features_df = feature_store.load_features()

    if features_df.empty:
        raise ValueError("No features available. Run build-features first.")

    # Get date range
    dates = pd.to_datetime(features_df["date"]).sort_values()
    start_date = dates.min()
    end_date = dates.max()

    logger.info(f"Data range: {start_date.date()} to {end_date.date()}")

    # Generate walk-forward windows
    windows = []
    current_start = start_date
    train_delta = timedelta(days=365 * train_window_years)
    test_delta = timedelta(days=30 * test_window_months)
    step_delta = timedelta(days=30 * step_months)

    while current_start + train_delta + test_delta <= end_date:
        train_start = current_start
        train_end = train_start + train_delta
        test_start = train_end
        test_end = min(test_start + test_delta, end_date)

        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

        current_start += step_delta

    logger.info(f"Generated {len(windows)} walk-forward windows")

    # Run backtest for each window
    all_equity_curves = []
    all_metrics = []

    for i, window in enumerate(windows):
        logger.info(
            f"Window {i+1}/{len(windows)}: Test {window['test_start'].date()} to {window['test_end'].date()}"
        )

        # Get test period data
        test_mask = (features_df["date"] >= window["test_start"]) & (features_df["date"] < window["test_end"])
        test_features = features_df[test_mask].copy()

        if test_features.empty:
            logger.warning(f"Empty test period for window {i+1}")
            continue

        # Generate predictions (model should be trained on train period)
        # For simplicity, we use the globally trained model
        # In production, you would retrain on train period for each window
        try:
            predictions = predict_supervised(test_features, model_name=model_name)
        except Exception as e:
            logger.warning(f"Failed to generate predictions for window {i+1}: {e}")
            continue

        # Create signals based on strategy
        if strategy == "supervised_prob_threshold":
            signals_df = predictions[["date", "ticker", "prob_class"]].copy()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Run backtest
        engine = BacktestEngine()
        equity_curve = engine.run(signals_df)

        if not equity_curve.empty:
            equity_curve["window"] = i
            all_equity_curves.append(equity_curve)

            # Calculate metrics
            metrics = calculate_metrics(equity_curve, engine.initial_capital)
            metrics["window"] = i
            metrics["test_start"] = window["test_start"]
            metrics["test_end"] = window["test_end"]
            all_metrics.append(metrics)

    # Combine results
    if not all_equity_curves:
        logger.warning("No backtest results generated")
        return {}

    combined_equity = pd.concat(all_equity_curves, ignore_index=True)
    combined_equity = combined_equity.sort_values("date").reset_index(drop=True)

    # Calculate overall metrics
    overall_metrics = calculate_metrics(combined_equity, engine.initial_capital)
    overall_metrics["strategy"] = strategy
    overall_metrics["model_name"] = model_name
    overall_metrics["n_windows"] = len(windows)

    # Save results
    backtests_dir = settings.get_backtests_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    equity_path = backtests_dir / f"equity_curve_{strategy}_{timestamp}.parquet"
    metrics_path = backtests_dir / f"metrics_{strategy}_{timestamp}.json"

    save_dataframe(combined_equity, equity_path)
    save_json(overall_metrics, metrics_path)

    logger.info(f"Walk-forward backtest complete. Results saved to {backtests_dir}")

    return {
        "equity_curve": combined_equity,
        "metrics": overall_metrics,
        "window_metrics": all_metrics,
    }

