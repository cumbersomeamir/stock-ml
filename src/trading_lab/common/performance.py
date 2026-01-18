"""Performance monitoring and profiling utilities."""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("trading_lab.performance")


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        """Initialize performance monitor."""
        self.timings: Dict[str, list[float]] = {}
        self.memory_usage: Dict[str, list[float]] = {}

    def record_timing(self, operation: str, duration: float) -> None:
        """
        Record timing for an operation.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
        """
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(duration)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance summary.
        
        Returns:
            Dictionary with performance statistics
        """
        summary = {}
        
        for operation, times in self.timings.items():
            if times:
                summary[operation] = {
                    "count": len(times),
                    "total_time": sum(times),
                    "mean_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                }
        
        return summary

    def print_summary(self) -> None:
        """Print performance summary."""
        summary = self.get_summary()
        
        if not summary:
            logger.info("No performance data collected")
            return
        
        logger.info("Performance Summary:")
        for operation, stats in summary.items():
            logger.info(
                f"  {operation}: "
                f"count={stats['count']}, "
                f"total={stats['total_time']:.2f}s, "
                f"mean={stats['mean_time']:.4f}s"
            )


# Global monitor instance
_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _monitor


@contextmanager
def timing(operation: str, log_result: bool = True):
    """
    Context manager for timing operations.
    
    Args:
        operation: Operation name
        log_result: Whether to log the result
        
    Usage:
        with timing("data_load"):
            data = load_large_file()
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        _monitor.record_timing(operation, duration)
        if log_result:
            logger.debug(f"{operation} took {duration:.4f}s")


def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile function execution time.
    
    Usage:
        @profile_function
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        operation_name = f"{func.__module__}.{func.__name__}"
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            _monitor.record_timing(operation_name, duration)
            logger.debug(f"{operation_name} took {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            _monitor.record_timing(f"{operation_name}_failed", duration)
            raise
    
    return wrapper


def estimate_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate memory usage of a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with memory usage statistics
    """
    import pandas as pd
    
    memory_usage = df.memory_usage(deep=True)
    
    return {
        "total_mb": float(memory_usage.sum() / 1024**2),
        "per_column_mb": {col: float(memory_usage[col] / 1024**2) for col in df.columns},
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def optimize_dataframe_memory(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        aggressive: If True, use more aggressive optimization
        
    Returns:
        Optimized DataFrame
    """
    df = df.copy()
    
    # Downcast integers
    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    
    # Downcast floats
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        if aggressive:
            df[col] = pd.to_numeric(df[col], downcast="float")
        else:
            # Use float32 if values fit
            if df[col].abs().max() < 3.4e38:  # float32 max
                df[col] = df[col].astype("float32")
    
    # Convert object columns to category if beneficial
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype("category")
    
    original_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Optimized DataFrame memory: {original_mb:.2f} MB")
    
    return df
