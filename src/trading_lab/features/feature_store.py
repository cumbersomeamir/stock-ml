"""Feature store for managing feature data."""

import logging
from pathlib import Path

import pandas as pd

from trading_lab.common.io import load_dataframe, save_dataframe
from trading_lab.config.settings import get_settings

logger = logging.getLogger("trading_lab.features")


class FeatureStore:
    """Store and retrieve feature data."""

    def __init__(self):
        """Initialize feature store."""
        self.settings = get_settings()
        self.features_dir = self.settings.get_features_dir()

    def save_features(self, features_df: pd.DataFrame, name: str = "features") -> Path:
        """
        Save features DataFrame to disk.

        Args:
            features_df: Features DataFrame
            name: Feature set name

        Returns:
            Path to saved file
        """
        filepath = self.features_dir / f"{name}.parquet"
        save_dataframe(features_df, filepath)
        logger.info(f"Saved features to {filepath}")
        return filepath

    def load_features(self, name: str = "features") -> pd.DataFrame:
        """
        Load features DataFrame from disk.

        Args:
            name: Feature set name

        Returns:
            Features DataFrame
        """
        filepath = self.features_dir / f"{name}.parquet"
        if not filepath.exists():
            raise FileNotFoundError(f"Feature file not found: {filepath}")
        return load_dataframe(filepath)

    def list_features(self) -> list[str]:
        """
        List available feature sets.

        Returns:
            List of feature set names
        """
        if not self.features_dir.exists():
            return []
        return [f.stem for f in self.features_dir.glob("*.parquet")]

