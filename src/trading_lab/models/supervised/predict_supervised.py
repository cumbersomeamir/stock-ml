"""Predict using trained supervised models."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from trading_lab.common.io import load_model
from trading_lab.config.settings import get_settings
from trading_lab.features.feature_store import FeatureStore

logger = logging.getLogger("trading_lab.models.supervised")


def predict_supervised(
    features_df: Optional[pd.DataFrame] = None,
    model_name: str = "gradient_boosting",
    return_proba: bool = True,
) -> pd.DataFrame:
    """
    Generate predictions using trained supervised models.

    Args:
        features_df: Features DataFrame. If None, loads from feature store.
        model_name: Model name to use for prediction
        return_proba: If True, return probabilities for classification

    Returns:
        DataFrame with predictions (prob_class, pred_class, pred_reg)
    """
    settings = get_settings()
    models_dir = settings.get_models_dir()
    feature_store = FeatureStore()

    # Load models
    classifier_path = models_dir / f"classifier_{model_name}.joblib"
    regressor_path = models_dir / f"regressor_{model_name}.joblib"

    if not classifier_path.exists() or not regressor_path.exists():
        raise FileNotFoundError(f"Models not found. Train models first with model_name={model_name}")

    classifier = load_model(classifier_path)
    regressor = load_model(regressor_path)

    # Load metrics to get feature columns
    from trading_lab.common.io import load_json
    metrics_path = models_dir / f"metrics_{model_name}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}. Train models first.")
    
    metrics = load_json(metrics_path)
    if "feature_cols" not in metrics:
        raise ValueError(f"Metrics file missing 'feature_cols' key: {metrics_path}")
    
    feature_cols = metrics["feature_cols"]

    # Load features if not provided
    if features_df is None:
        features_df = feature_store.load_features()

    # Validate required columns
    required_cols = ["date", "ticker"] + feature_cols
    missing_cols = [col for col in required_cols if col not in features_df.columns]
    if missing_cols:
        raise ValueError(
            f"Features DataFrame missing required columns: {missing_cols}. "
            f"Available columns: {list(features_df.columns)}"
        )

    # Prepare features
    X = features_df[feature_cols].fillna(0).values

    # Generate predictions
    if return_proba and hasattr(classifier, "predict_proba"):
        prob_class = classifier.predict_proba(X)[:, 1]
        pred_class = (prob_class > 0.5).astype(int)
    else:
        pred_class = classifier.predict(X)
        prob_class = None

    pred_reg = regressor.predict(X)

    # Create results DataFrame
    results = features_df[["date", "ticker"]].copy()
    if prob_class is not None:
        results["prob_class"] = prob_class
    results["pred_class"] = pred_class
    results["pred_reg"] = pred_reg

    logger.info(f"Generated predictions for {len(results)} samples")

    return results

