"""Application settings and configuration."""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Data directory
    data_dir: Path = Path("./data")

    # API Keys (optional)
    newsapi_key: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "trading-lab/0.1.0"
    fmp_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    fred_api_key: Optional[str] = None
    rbi_api_key: Optional[str] = None

    # Logging
    log_level: str = "INFO"

    # Trading parameters
    max_position_per_asset: float = 0.1
    max_gross_exposure: float = 1.0
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    max_drawdown_threshold: float = 0.2

    # Model parameters
    train_window_years: int = 2
    test_window_months: int = 3
    step_months: int = 1

    # Feature engineering
    feature_lookback_days: int = 60
    min_price_change_threshold: float = 0.0005

    def get_raw_data_dir(self) -> Path:
        """Get raw data directory path."""
        return self.data_dir / "raw"

    def get_processed_data_dir(self) -> Path:
        """Get processed data directory path."""
        return self.data_dir / "processed"

    def get_artifacts_dir(self) -> Path:
        """Get artifacts directory path."""
        return self.data_dir / "artifacts"

    def get_prices_dir(self) -> Path:
        """Get prices data directory path."""
        return self.get_raw_data_dir() / "prices" / "yfinance"

    def get_unified_dir(self) -> Path:
        """Get unified data directory path."""
        return self.get_processed_data_dir() / "unified"

    def get_features_dir(self) -> Path:
        """Get features directory path."""
        return self.get_processed_data_dir() / "features"

    def get_models_dir(self) -> Path:
        """Get models directory path."""
        return self.get_artifacts_dir() / "models"

    def get_backtests_dir(self) -> Path:
        """Get backtests directory path."""
        return self.get_artifacts_dir() / "backtests"

    def get_reports_dir(self) -> Path:
        """Get reports directory path."""
        return self.get_artifacts_dir() / "reports"


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        # Create directories
        _settings.data_dir.mkdir(parents=True, exist_ok=True)
        _settings.get_raw_data_dir().mkdir(parents=True, exist_ok=True)
        _settings.get_processed_data_dir().mkdir(parents=True, exist_ok=True)
        _settings.get_artifacts_dir().mkdir(parents=True, exist_ok=True)
        _settings.get_prices_dir().mkdir(parents=True, exist_ok=True)
        _settings.get_unified_dir().mkdir(parents=True, exist_ok=True)
        _settings.get_features_dir().mkdir(parents=True, exist_ok=True)
        _settings.get_models_dir().mkdir(parents=True, exist_ok=True)
        _settings.get_backtests_dir().mkdir(parents=True, exist_ok=True)
        _settings.get_reports_dir().mkdir(parents=True, exist_ok=True)
    return _settings

