"""Train supervised models for classification and regression."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from trading_lab.common.bias_checks import check_for_lookahead_bias
from trading_lab.common.io import load_dataframe, save_json, save_model
from trading_lab.common.performance import profile_function
from trading_lab.common.reproducibility import set_random_seeds
from trading_lab.config.settings import get_settings
from trading_lab.features.feature_store import FeatureStore
from trading_lab.labeling.targets import generate_targets
from trading_lab.models.supervised.model_zoo import get_classifier, get_regressor

logger = logging.getLogger("trading_lab.models.supervised")


@profile_function
def train_supervised(
    model_name: str = "gradient_boosting",
    force_refresh: bool = False,
    test_size: float = 0.2,
    random_seed: Optional[int] = 42,
    check_bias: bool = True,
) -> Dict:
    """
    Train supervised models for classification and regression.

    Args:
        model_name: Model name (see model_zoo for options)
        force_refresh: If True, retrain even if model exists
        test_size: Proportion of data to use for testing
        random_seed: Random seed for reproducibility (None to disable)
        check_bias: Whether to check for look-ahead bias

    Returns:
        Dictionary with training results and metrics
    """
    logger.info(f"Training supervised models: {model_name}")
    
    # Set random seeds for reproducibility
    if random_seed is not None:
        set_random_seeds(random_seed)

    settings = get_settings()
    models_dir = settings.get_models_dir()
    feature_store = FeatureStore()

    # Check if models already exist
    classifier_path = models_dir / f"classifier_{model_name}.joblib"
    regressor_path = models_dir / f"regressor_{model_name}.joblib"
    metrics_path = models_dir / f"metrics_{model_name}.json"

    if not force_refresh and classifier_path.exists() and regressor_path.exists():
        logger.info("Models already exist, loading metrics")
        from trading_lab.common.io import load_json
        return load_json(metrics_path)

    # Load features
    features_df = feature_store.load_features()
    if features_df.empty:
        raise ValueError("No features available. Run build-features first.")
    logger.info(f"Loaded {len(features_df)} feature rows")

    # Generate targets
    targets_df = generate_targets(features_df)
    if targets_df.empty:
        raise ValueError("No targets generated. Check price data availability.")
    logger.info(f"Generated {len(targets_df)} target rows")
    
    # Check for look-ahead bias if enabled
    if check_bias:
        bias_results = check_for_lookahead_bias(features_df, targets_df)
        if bias_results["has_potential_bias"]:
            logger.warning("⚠️ Potential look-ahead bias detected!")
            for warning in bias_results["warnings"]:
                logger.warning(f"  {warning}")
            logger.warning("This may indicate features are using future information.")

    # Merge features and targets
    data = features_df.merge(targets_df, on=["date", "ticker"], how="inner")
    data = data.sort_values("date").reset_index(drop=True)

    # Select feature columns (exclude date, ticker, targets)
    feature_cols = [col for col in data.columns if col not in ["date", "ticker", "y_class", "y_reg"]]

    # Prepare data
    X = data[feature_cols].fillna(0).values
    y_class = data["y_class"].values
    y_reg = data["y_reg"].values

    # Remove rows with NaN targets
    valid_class = ~np.isnan(y_class)
    valid_reg = ~np.isnan(y_reg)

    X_class = X[valid_class]
    y_class_clean = y_class[valid_class].astype(int)

    X_reg = X[valid_reg]
    y_reg_clean = y_reg[valid_reg]

    logger.info(f"Training samples: classification={len(X_class)}, regression={len(X_reg)}")

    # Time-series split (use simple split for now, can be improved with purged CV)
    n_train = int(len(X_class) * (1 - test_size))
    if n_train < 5:
        raise ValueError(f"Training set too small: {n_train} samples. Increase data or reduce test_size.")
    X_train_class, X_test_class = X_class[:n_train], X_class[n_train:]
    y_train_class, y_test_class = y_class_clean[:n_train], y_class_clean[n_train:]

    n_train_reg = int(len(X_reg) * (1 - test_size))
    if n_train_reg < 5:
        raise ValueError(f"Regression training set too small: {n_train_reg} samples.")
    X_train_reg, X_test_reg = X_reg[:n_train_reg], X_reg[n_train_reg:]
    y_train_reg, y_test_reg = y_reg_clean[:n_train_reg], y_reg_clean[n_train_reg:]

    # Train classifier
    logger.info("Training classifier")
    classifier = get_classifier(model_name)
    try:
        classifier.fit(X_train_class, y_train_class)
        y_pred_class = classifier.predict(X_test_class)
        y_pred_proba = classifier.predict_proba(X_test_class)[:, 1] if hasattr(classifier, "predict_proba") else None
    except Exception as e:
        logger.error(f"Error training classifier: {e}")
        raise ValueError(f"Failed to train classifier: {e}") from e

    # Train regressor
    logger.info("Training regressor")
    regressor = get_regressor(model_name)
    try:
        regressor.fit(X_train_reg, y_train_reg)
        y_pred_reg = regressor.predict(X_test_reg)
    except Exception as e:
        logger.error(f"Error training regressor: {e}")
        raise ValueError(f"Failed to train regressor: {e}") from e

    # Calculate metrics
    accuracy = accuracy_score(y_test_class, y_pred_class)
    auc = roc_auc_score(y_test_class, y_pred_proba) if y_pred_proba is not None else None
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)

    # Get feature importance if available
    feature_importance = None
    if hasattr(classifier, "feature_importances_"):
        feature_importance = {
            feature_cols[i]: float(importance)
            for i, importance in enumerate(classifier.feature_importances_)
        }
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

    metrics = {
        "model_name": model_name,
        "classification": {
            "accuracy": float(accuracy),
            "auc": float(auc) if auc is not None else None,
            "n_train": int(len(X_train_class)),
            "n_test": int(len(X_test_class)),
        },
        "regression": {
            "mse": float(mse),
            "rmse": float(rmse),
            "n_train": int(len(X_train_reg)),
            "n_test": int(len(X_test_reg)),
        },
        "feature_cols": feature_cols,
        "feature_importance": feature_importance,
    }

    logger.info(f"Classification accuracy: {accuracy:.4f}, AUC: {auc:.4f if auc else 'N/A'}")
    logger.info(f"Regression RMSE: {rmse:.6f}")

    # Save models
    save_model(classifier, classifier_path)
    save_model(regressor, regressor_path)
    save_json(metrics, metrics_path)

    logger.info(f"Saved models to {models_dir}")

    return metrics

