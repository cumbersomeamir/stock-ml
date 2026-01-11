"""Model zoo with various ML models."""

import logging
from typing import Any, Optional

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge

logger = logging.getLogger("trading_lab.models")


def get_classifier(model_name: str, **kwargs) -> Any:
    """
    Get a classifier model by name.

    Args:
        model_name: Model name ('logistic', 'random_forest', 'gradient_boosting', 'lightgbm')
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Classifier model instance
    """
    if model_name == "logistic":
        return LogisticRegression(random_state=42, max_iter=1000, **kwargs)
    elif model_name == "random_forest":
        return RandomForestClassifier(random_state=42, n_estimators=100, **kwargs)
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42, n_estimators=100, **kwargs)
    elif model_name == "lightgbm":
        try:
            import lightgbm as lgb
            return lgb.LGBMClassifier(random_state=42, n_estimators=100, verbose=-1, **kwargs)
        except ImportError:
            logger.warning("LightGBM not installed, falling back to GradientBoosting")
            return GradientBoostingClassifier(random_state=42, n_estimators=100, **kwargs)
    else:
        raise ValueError(f"Unknown classifier model: {model_name}")


def get_regressor(model_name: str, **kwargs) -> Any:
    """
    Get a regressor model by name.

    Args:
        model_name: Model name ('ridge', 'lasso', 'gradient_boosting', 'lightgbm')
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Regressor model instance
    """
    if model_name == "ridge":
        return Ridge(random_state=42, **kwargs)
    elif model_name == "lasso":
        return Lasso(random_state=42, max_iter=1000, **kwargs)
    elif model_name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=42, n_estimators=100, **kwargs)
    elif model_name == "lightgbm":
        try:
            import lightgbm as lgb
            return lgb.LGBMRegressor(random_state=42, n_estimators=100, verbose=-1, **kwargs)
        except ImportError:
            logger.warning("LightGBM not installed, falling back to GradientBoosting")
            return GradientBoostingRegressor(random_state=42, n_estimators=100, **kwargs)
    else:
        raise ValueError(f"Unknown regressor model: {model_name}")

