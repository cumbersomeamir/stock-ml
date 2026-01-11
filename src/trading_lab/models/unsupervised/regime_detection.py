"""Market regime detection using unsupervised learning."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from trading_lab.common.io import load_dataframe, save_dataframe, save_model
from trading_lab.config.settings import get_settings
from trading_lab.features.feature_store import FeatureStore

logger = logging.getLogger("trading_lab.models.unsupervised")


def detect_regimes(
    n_regimes: int = 4,
    method: str = "kmeans",
    window: int = 20,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Detect market regimes using unsupervised learning.

    Args:
        n_regimes: Number of regimes to detect
        method: Clustering method ('kmeans' or 'gmm')
        window: Rolling window for regime features
        force_refresh: If True, re-detect even if results exist

    Returns:
        DataFrame with regime labels added
    """
    logger.info(f"Detecting regimes: n_regimes={n_regimes}, method={method}")

    settings = get_settings()
    models_dir = settings.get_models_dir()
    feature_store = FeatureStore()

    # Check if regimes already detected
    regime_file = settings.get_features_dir() / f"regimes_{method}_{n_regimes}.parquet"
    if regime_file.exists() and not force_refresh:
        logger.info("Loading existing regime labels")
        return load_dataframe(regime_file)

    # Load features
    features_df = feature_store.load_features()
    logger.info(f"Loaded {len(features_df)} feature rows")

    # Select features for regime detection (returns and volatility)
    regime_features = ["return_1d", "volatility_20d"]
    available_features = [f for f in regime_features if f in features_df.columns]

    if not available_features:
        logger.warning("No suitable features for regime detection")
        features_df["regime"] = 0
        return features_df

    # Calculate rolling features for regime detection
    features_df = features_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    regime_data = []

    for ticker in features_df["ticker"].unique():
        ticker_data = features_df[features_df["ticker"] == ticker].copy()
        ticker_data = ticker_data.sort_values("date").reset_index(drop=True)

        # Rolling statistics
        for feature in available_features:
            ticker_data[f"{feature}_rolling_mean"] = ticker_data[feature].rolling(window, min_periods=5).mean()
            ticker_data[f"{feature}_rolling_std"] = ticker_data[feature].rolling(window, min_periods=5).std()

        regime_data.append(ticker_data)

    regime_df = pd.concat(regime_data, ignore_index=True)

    # Select features for clustering
    cluster_features = [f"{f}_rolling_mean" for f in available_features] + [
        f"{f}_rolling_std" for f in available_features
    ]
    cluster_features = [f for f in cluster_features if f in regime_df.columns]

    # Prepare data
    X = regime_df[cluster_features].fillna(0).values
    valid_mask = ~np.isnan(X).any(axis=1)
    X_clean = X[valid_mask]

    if len(X_clean) < n_regimes:
        logger.warning("Not enough data for regime detection")
        regime_df["regime"] = 0
        save_dataframe(regime_df[["date", "ticker", "regime"]], regime_file)
        return regime_df

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Cluster
    if method == "kmeans":
        model = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    elif method == "gmm":
        model = GaussianMixture(n_components=n_regimes, random_state=42, max_iter=100)
    else:
        raise ValueError(f"Unknown method: {method}")

    labels = model.fit_predict(X_scaled)

    # Assign labels
    regime_df.loc[valid_mask, "regime"] = labels
    regime_df["regime"] = regime_df["regime"].fillna(0).astype(int)

    logger.info(f"Detected regimes: {regime_df['regime'].value_counts().to_dict()}")

    # Save model and scaler
    save_model(model, models_dir / f"regime_model_{method}_{n_regimes}.joblib")
    save_model(scaler, models_dir / f"regime_scaler_{method}_{n_regimes}.joblib")

    # Save regime labels
    save_dataframe(regime_df[["date", "ticker", "regime"]], regime_file)

    # Add regime to features
    features_df = features_df.merge(regime_df[["date", "ticker", "regime"]], on=["date", "ticker"], how="left")
    features_df["regime"] = features_df["regime"].fillna(0).astype(int)

    return features_df

