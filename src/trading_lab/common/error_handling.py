"""Enhanced error handling and user-friendly error messages."""

import logging
import traceback
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("trading_lab.error_handling")

T = TypeVar("T")


class TradingLabError(Exception):
    """Base exception for trading lab errors."""
    pass


class DataError(TradingLabError):
    """Error related to data loading or processing."""
    pass


class ValidationError(TradingLabError):
    """Error related to data validation."""
    pass


class ConfigurationError(TradingLabError):
    """Error related to configuration."""
    pass


class ModelError(TradingLabError):
    """Error related to model training or prediction."""
    pass


class BacktestError(TradingLabError):
    """Error related to backtesting."""
    pass


def handle_errors(
    default_return: Optional[Any] = None,
    log_traceback: bool = True,
    reraise: bool = False,
):
    """
    Decorator to handle errors gracefully with user-friendly messages.
    
    Args:
        default_return: Value to return on error (if None, raises)
        log_traceback: Whether to log full traceback
        reraise: Whether to re-raise the exception
        
    Usage:
        @handle_errors(default_return=0)
        def risky_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except TradingLabError as e:
                # Re-raise our custom errors
                if reraise:
                    raise
                if default_return is not None:
                    logger.error(f"{func.__name__} failed: {e}")
                    return default_return
                raise
            except Exception as e:
                error_msg = f"Error in {func.__name__}: {str(e)}"
                
                if log_traceback:
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                else:
                    logger.error(error_msg)
                
                if reraise:
                    raise
                
                if default_return is not None:
                    return default_return
                
                # Convert to our error types
                if "data" in str(e).lower() or "file" in str(e).lower():
                    raise DataError(error_msg) from e
                elif "validation" in str(e).lower() or "invalid" in str(e).lower():
                    raise ValidationError(error_msg) from e
                elif "config" in str(e).lower() or "setting" in str(e).lower():
                    raise ConfigurationError(error_msg) from e
                elif "model" in str(e).lower() or "train" in str(e).lower():
                    raise ModelError(error_msg) from e
                elif "backtest" in str(e).lower():
                    raise BacktestError(error_msg) from e
                else:
                    raise TradingLabError(error_msg) from e
        
        return wrapper
    return decorator


def format_error_message(error: Exception, context: Optional[str] = None) -> str:
    """
    Format error message in a user-friendly way.
    
    Args:
        error: Exception to format
        context: Optional context information
        
    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    if context:
        return f"[{context}] {error_type}: {error_msg}"
    else:
        return f"{error_type}: {error_msg}"


def log_error_with_context(
    error: Exception,
    context: dict,
    level: int = logging.ERROR,
):
    """
    Log error with additional context.
    
    Args:
        error: Exception to log
        context: Dictionary with context information
        level: Logging level
    """
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    error_msg = format_error_message(error, context_str)
    logger.log(level, error_msg)
