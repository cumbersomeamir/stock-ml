"""Common utilities and helpers."""

from trading_lab.common.bias_checks import (
    add_embargo_period,
    check_for_lookahead_bias,
    detect_future_data_leakage,
    validate_time_series_split,
)
from trading_lab.common.cache import file_cache
from trading_lab.common.data_utils import (
    calculate_correlation_matrix,
    calculate_rolling_zscore,
    ensure_datetime_index,
    forward_fill_missing_dates,
    remove_outliers,
    resample_dataframe,
    safe_divide,
)
from trading_lab.common.io import load_dataframe, load_json, load_model, save_dataframe, save_json, save_model
from trading_lab.common.performance import (
    estimate_memory_usage,
    get_performance_monitor,
    optimize_dataframe_memory,
    profile_function,
    timing,
)
from trading_lab.common.pipeline import Pipeline, PipelineTask, TaskStatus, create_standard_pipeline
from trading_lab.common.reproducibility import ExperimentTracker, set_random_seeds
from trading_lab.common.resilience import (
    GracefulDegradation,
    circuit_breaker,
    retry,
    safe_execute,
    validate_and_fix,
)
from trading_lab.common.schemas import validate_dataframe_columns
from trading_lab.common.time import add_business_days, get_trading_days, parse_date, subtract_business_days
from trading_lab.common.time_series import (
    align_time_series,
    detect_gaps,
    fill_gaps,
    resample_time_series,
    validate_temporal_consistency,
)
from trading_lab.common.validation import check_data_quality, clean_price_data, validate_price_data
from trading_lab.common.error_handling import (
    BacktestError,
    ConfigurationError,
    DataError,
    ModelError,
    TradingLabError,
    ValidationError,
    format_error_message,
    handle_errors,
    log_error_with_context,
)

__all__ = [
    # Bias checks
    "check_for_lookahead_bias",
    "validate_time_series_split",
    "detect_future_data_leakage",
    "add_embargo_period",
    # Cache
    "file_cache",
    # Data utils
    "safe_divide",
    "calculate_rolling_zscore",
    "remove_outliers",
    "ensure_datetime_index",
    "resample_dataframe",
    "forward_fill_missing_dates",
    "calculate_correlation_matrix",
    # IO
    "load_dataframe",
    "save_dataframe",
    "load_model",
    "save_model",
    "load_json",
    "save_json",
    # Performance
    "profile_function",
    "timing",
    "get_performance_monitor",
    "estimate_memory_usage",
    "optimize_dataframe_memory",
    # Pipeline
    "Pipeline",
    "PipelineTask",
    "TaskStatus",
    "create_standard_pipeline",
    # Reproducibility
    "set_random_seeds",
    "ExperimentTracker",
    # Resilience
    "retry",
    "circuit_breaker",
    "safe_execute",
    "validate_and_fix",
    "GracefulDegradation",
    # Schemas
    "validate_dataframe_columns",
    # Time
    "parse_date",
    "get_trading_days",
    "add_business_days",
    "subtract_business_days",
    # Time series
    "align_time_series",
    "detect_gaps",
    "fill_gaps",
    "resample_time_series",
    "validate_temporal_consistency",
    # Validation
    "check_data_quality",
    "validate_price_data",
    "clean_price_data",
    # Error handling
    "TradingLabError",
    "DataError",
    "ValidationError",
    "ConfigurationError",
    "ModelError",
    "BacktestError",
    "handle_errors",
    "format_error_message",
    "log_error_with_context",
]
