"""Data quality checks at ingestion point.

Ensures data quality from the moment it enters the system.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_lab.data_quality")


class DataQualityReport:
    """Report on data quality issues."""
    
    def __init__(self):
        """Initialize quality report."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.metrics: Dict[str, float] = {}
    
    def add_error(self, message: str) -> None:
        """Add an error."""
        self.errors.append(message)
        logger.error(f"Data quality error: {message}")
    
    def add_warning(self, message: str) -> None:
        """Add a warning."""
        self.warnings.append(message)
        logger.warning(f"Data quality warning: {message}")
    
    def add_info(self, message: str) -> None:
        """Add informational message."""
        self.info.append(message)
        logger.info(f"Data quality info: {message}")
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a quality metric."""
        self.metrics[name] = value
    
    @property
    def has_errors(self) -> bool:
        """Check if there are errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return len(self.warnings) > 0
    
    def __str__(self) -> str:
        """String representation."""
        lines = ["Data Quality Report"]
        lines.append("=" * 50)
        
        if self.errors:
            lines.append(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  ❌ {error}")
        
        if self.warnings:
            lines.append(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
        
        if self.info:
            lines.append(f"\nINFO ({len(self.info)}):")
            for info_msg in self.info:
                lines.append(f"  ℹ️  {info_msg}")
        
        if self.metrics:
            lines.append("\nMETRICS:")
            for name, value in self.metrics.items():
                lines.append(f"  {name}: {value:.4f}")
        
        return "\n".join(lines)


def check_price_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """
    Comprehensive quality checks for price data.
    
    Args:
        df: Price DataFrame
        
    Returns:
        DataQualityReport with findings
    """
    report = DataQualityReport()
    
    # Check required columns
    required_cols = ["date", "ticker", "adj_close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        report.add_error(f"Missing required columns: {missing_cols}")
        return report
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        try:
            df["date"] = pd.to_datetime(df["date"])
            report.add_info("Converted date column to datetime")
        except Exception as e:
            report.add_error(f"Cannot convert date column to datetime: {e}")
    
    # Check for empty data
    if len(df) == 0:
        report.add_error("DataFrame is empty")
        return report
    
    # Check for missing values
    missing_pct = df.isnull().sum() / len(df) * 100
    for col in df.columns:
        if missing_pct[col] > 0:
            if missing_pct[col] > 50:
                report.add_error(f"Column '{col}' has {missing_pct[col]:.1f}% missing values")
            elif missing_pct[col] > 10:
                report.add_warning(f"Column '{col}' has {missing_pct[col]:.1f}% missing values")
            else:
                report.add_info(f"Column '{col}' has {missing_pct[col]:.1f}% missing values")
    
    report.add_metric("missing_data_pct", float(missing_pct.mean()))
    
    # Check for duplicates
    duplicates = df.duplicated(subset=["date", "ticker"], keep=False)
    if duplicates.any():
        dup_count = duplicates.sum()
        report.add_error(f"Found {dup_count} duplicate (date, ticker) pairs")
        report.add_metric("duplicate_rows", float(dup_count))
    
    # Check price validity
    price_cols = ["open", "high", "low", "close", "adj_close"]
    available_price_cols = [col for col in price_cols if col in df.columns]
    
    for col in available_price_cols:
        # Check for non-positive prices
        invalid_prices = (df[col] <= 0) | df[col].isnull()
        if invalid_prices.any():
            invalid_count = invalid_prices.sum()
            report.add_warning(
                f"Column '{col}' has {invalid_count} non-positive or null values"
            )
            report.add_metric(f"{col}_invalid_pct", float(invalid_count / len(df) * 100))
        
        # Check for extreme values
        if col in df.columns and not df[col].isnull().all():
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            extreme = (df[col] > q99 * 10) | (df[col] < q01 / 10)
            if extreme.any():
                report.add_warning(
                    f"Column '{col}' has {extreme.sum()} extreme outliers"
                )
    
    # Check high/low consistency
    if "high" in df.columns and "low" in df.columns:
        invalid_hl = df["high"] < df["low"]
        if invalid_hl.any():
            report.add_error(
                f"Found {invalid_hl.sum()} rows where high < low"
            )
    
    # Check OHLC consistency
    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        invalid_ohlc = (
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )
        if invalid_ohlc.any():
            report.add_warning(
                f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships"
            )
    
    # Check volume
    if "volume" in df.columns:
        negative_volume = df["volume"] < 0
        if negative_volume.any():
            report.add_error(f"Found {negative_volume.sum()} rows with negative volume")
        
        zero_volume = df["volume"] == 0
        if zero_volume.any():
            zero_pct = zero_volume.sum() / len(df) * 100
            if zero_pct > 5:
                report.add_warning(f"{zero_pct:.1f}% of rows have zero volume")
    
    # Check date range
    date_range = (df["date"].max() - df["date"].min()).days
    report.add_metric("date_range_days", float(date_range))
    report.add_info(f"Date range: {df['date'].min()} to {df['date'].max()} ({date_range} days)")
    
    # Check number of tickers
    n_tickers = df["ticker"].nunique()
    report.add_metric("n_tickers", float(n_tickers))
    report.add_info(f"Number of unique tickers: {n_tickers}")
    
    # Check data completeness per ticker
    if n_tickers > 1:
        rows_per_ticker = df.groupby("ticker").size()
        completeness_std = rows_per_ticker.std() / rows_per_ticker.mean()
        report.add_metric("ticker_completeness_cv", float(completeness_std))
        
        if completeness_std > 0.5:
            report.add_warning(
                f"High variability in data completeness across tickers (CV={completeness_std:.2f})"
            )
    
    # Final summary
    if not report.has_errors and not report.has_warnings:
        report.add_info("✅ All quality checks passed")
    
    return report


def check_feature_data_quality(df: pd.DataFrame, feature_cols: List[str]) -> DataQualityReport:
    """
    Quality checks for feature data.
    
    Args:
        df: Feature DataFrame
        feature_cols: List of feature column names
        
    Returns:
        DataQualityReport with findings
    """
    report = DataQualityReport()
    
    # Check for infinite values
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        inf_values = np.isinf(df[col])
        if inf_values.any():
            report.add_error(f"Feature '{col}' has {inf_values.sum()} infinite values")
    
    # Check for constant features
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        if df[col].nunique() <= 1:
            report.add_warning(f"Feature '{col}' is constant (no variation)")
    
    # Check for high correlation (potential redundancy)
    numeric_features = [col for col in feature_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if len(numeric_features) > 1:
        corr_matrix = df[numeric_features].corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                    )
        
        if high_corr_pairs:
            report.add_warning(
                f"Found {len(high_corr_pairs)} pairs of highly correlated features (>0.95)"
            )
            for feat1, feat2, corr in high_corr_pairs[:5]:  # Show first 5
                report.add_info(f"  {feat1} <-> {feat2}: {corr:.3f}")
    
    # Check for missing values
    missing_pct = df[feature_cols].isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 20]
    if len(high_missing) > 0:
        report.add_warning(f"{len(high_missing)} features have >20% missing values")
        for feat in high_missing.index[:5]:
            report.add_info(f"  {feat}: {missing_pct[feat]:.1f}% missing")
    
    report.add_metric("avg_missing_pct", float(missing_pct.mean()))
    
    return report


def suggest_fixes(report: DataQualityReport) -> List[str]:
    """
    Suggest fixes for data quality issues.
    
    Args:
        report: DataQualityReport
        
    Returns:
        List of suggested actions
    """
    suggestions = []
    
    if any("duplicate" in err.lower() for err in report.errors):
        suggestions.append("Remove duplicates: df.drop_duplicates(subset=['date', 'ticker'])")
    
    if any("missing" in warn.lower() for warn in report.warnings):
        suggestions.append("Handle missing values: df.fillna(method='ffill') or df.dropna()")
    
    if any("high < low" in err.lower() for err in report.errors):
        suggestions.append("Fix invalid OHLC data: Swap high/low values or remove invalid rows")
    
    if any("infinite" in err.lower() for err in report.errors):
        suggestions.append("Remove infinite values: df.replace([np.inf, -np.inf], np.nan)")
    
    if any("constant" in warn.lower() for warn in report.warnings):
        suggestions.append("Remove constant features: They provide no predictive value")
    
    if any("correlation" in warn.lower() for warn in report.warnings):
        suggestions.append("Remove highly correlated features: Keep one from each correlated pair")
    
    return suggestions
