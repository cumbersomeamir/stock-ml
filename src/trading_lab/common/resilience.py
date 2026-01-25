"""Resilience and error recovery utilities."""

import logging
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("trading_lab.resilience")

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_failure: Optional[Callable] = None,
):
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry
        on_failure: Optional callback on final failure
        
    Usage:
        @retry(max_attempts=3, delay=1.0)
        def fetch_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        if on_failure:
                            on_failure(e, attempt)
            
            raise last_exception
        
        return wrapper
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    timeout: float = 60.0,
    exceptions: tuple = (Exception,),
):
    """
    Circuit breaker pattern to prevent cascading failures.
    
    After failure_threshold failures, the circuit opens and
    all calls fail immediately until timeout expires.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        timeout: Time in seconds before attempting to close circuit
        exceptions: Exceptions that count as failures
        
    Usage:
        @circuit_breaker(failure_threshold=5, timeout=60.0)
        def risky_operation():
            ...
    """
    state = {
        "failures": 0,
        "last_failure_time": None,
        "circuit_open": False,
    }
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Check if circuit should be closed
            if state["circuit_open"]:
                if state["last_failure_time"]:
                    elapsed = time.time() - state["last_failure_time"]
                    if elapsed > timeout:
                        logger.info(f"Circuit breaker for {func.__name__} attempting to close")
                        state["circuit_open"] = False
                        state["failures"] = 0
                    else:
                        raise RuntimeError(
                            f"Circuit breaker is OPEN for {func.__name__}. "
                            f"Try again in {timeout - elapsed:.1f}s"
                        )
                else:
                    raise RuntimeError(f"Circuit breaker is OPEN for {func.__name__}")
            
            # Try to execute
            try:
                result = func(*args, **kwargs)
                # Reset on success
                state["failures"] = 0
                return result
            except exceptions as e:
                state["failures"] += 1
                state["last_failure_time"] = time.time()
                
                if state["failures"] >= failure_threshold:
                    state["circuit_open"] = True
                    logger.error(
                        f"Circuit breaker OPENED for {func.__name__} "
                        f"after {state['failures']} failures"
                    )
                
                raise
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable[..., T],
    default: Optional[T] = None,
    log_errors: bool = True,
    *args: Any,
    **kwargs: Any,
) -> Optional[T]:
    """
    Safely execute a function, returning default on error.
    
    Args:
        func: Function to execute
        default: Default value to return on error
        log_errors: Whether to log errors
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error in {func.__name__}: {e}")
        return default


def validate_and_fix(
    validator: Callable[[Any], bool],
    fixer: Optional[Callable[[Any], Any]] = None,
    raise_on_failure: bool = True,
):
    """
    Decorator to validate and optionally fix function inputs/outputs.
    
    Args:
        validator: Function that returns True if value is valid
        fixer: Optional function to fix invalid values
        raise_on_failure: Whether to raise exception if validation fails and no fixer
        
    Usage:
        @validate_and_fix(
            validator=lambda x: x > 0,
            fixer=lambda x: abs(x)
        )
        def process_value(x):
            return x * 2
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Validate inputs
            for arg in args:
                if not validator(arg):
                    if fixer:
                        arg = fixer(arg)
                    elif raise_on_failure:
                        raise ValueError(f"Invalid input to {func.__name__}: {arg}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Validate output
            if not validator(result):
                if fixer:
                    result = fixer(result)
                elif raise_on_failure:
                    raise ValueError(f"Invalid output from {func.__name__}: {result}")
            
            return result
        
        return wrapper
    return decorator


class GracefulDegradation:
    """
    Context manager for graceful degradation on errors.
    
    Usage:
        with GracefulDegradation(default_value=0):
            result = risky_operation()
    """
    
    def __init__(self, default_value: Any = None, log_errors: bool = True):
        """
        Initialize graceful degradation context.
        
        Args:
            default_value: Value to return on error
            log_errors: Whether to log errors
        """
        self.default_value = default_value
        self.log_errors = log_errors
        self.error_occurred = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_occurred = True
            if self.log_errors:
                logger.warning(f"Error occurred, using default value: {exc_val}")
            return True  # Suppress exception
    
    def get_result(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute function with graceful degradation.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result or default value
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.error_occurred = True
            if self.log_errors:
                logger.warning(f"Error in {func.__name__}, using default: {e}")
            return self.default_value
