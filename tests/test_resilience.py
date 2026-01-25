"""Tests for resilience and error recovery utilities."""

import pytest

from trading_lab.common.resilience import (
    GracefulDegradation,
    circuit_breaker,
    retry,
    safe_execute,
    validate_and_fix,
)


def test_retry_success():
    """Test retry decorator on successful function."""
    call_count = [0]
    
    @retry(max_attempts=3)
    def successful_func():
        call_count[0] += 1
        return "success"
    
    result = successful_func()
    assert result == "success"
    assert call_count[0] == 1


def test_retry_failure():
    """Test retry decorator on failing function."""
    call_count = [0]
    
    @retry(max_attempts=3, delay=0.01)
    def failing_func():
        call_count[0] += 1
        raise ValueError("Always fails")
    
    with pytest.raises(ValueError):
        failing_func()
    
    assert call_count[0] == 3  # Should retry 3 times


def test_circuit_breaker():
    """Test circuit breaker pattern."""
    call_count = [0]
    
    @circuit_breaker(failure_threshold=2, timeout=0.1)
    def failing_func():
        call_count[0] += 1
        raise ValueError("Fails")
    
    # First failure
    with pytest.raises(ValueError):
        failing_func()
    
    # Second failure should open circuit
    with pytest.raises(ValueError):
        failing_func()
    
    # Third call should hit circuit breaker
    with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
        failing_func()


def test_safe_execute():
    """Test safe execution."""
    def failing_func():
        raise ValueError("Error")
    
    result = safe_execute(failing_func, default=42)
    assert result == 42


def test_validate_and_fix():
    """Test validation and fixing."""
    @validate_and_fix(
        validator=lambda x: x > 0,
        fixer=lambda x: abs(x)
    )
    def process_value(x):
        return x * 2
    
    # Should fix negative value
    result = process_value(-5)
    assert result == 10  # abs(-5) * 2


def test_graceful_degradation():
    """Test graceful degradation context manager."""
    with GracefulDegradation(default_value=0) as gd:
        def risky_operation():
            raise ValueError("Error")
        
        result = gd.get_result(risky_operation)
        assert result == 0
        assert gd.error_occurred
