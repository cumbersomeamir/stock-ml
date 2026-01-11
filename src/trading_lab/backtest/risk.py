"""Risk management and controls."""

from typing import Dict


def check_circuit_breaker(
    current_capital: float,
    peak_capital: float,
    max_drawdown_threshold: float,
) -> bool:
    """
    Check if circuit breaker should trigger (stop trading due to drawdown).

    Args:
        current_capital: Current capital
        peak_capital: Peak capital reached
        max_drawdown_threshold: Maximum allowed drawdown (e.g., 0.2 for 20%)

    Returns:
        True if circuit breaker should trigger (stop trading)
    """
    if peak_capital == 0:
        return False

    drawdown = (peak_capital - current_capital) / peak_capital
    return drawdown > max_drawdown_threshold


def apply_position_limits(
    target_position: float,
    max_position_per_asset: float,
    max_gross_exposure: float,
    current_positions: Dict[str, float],
) -> float:
    """
    Apply position limits.

    Args:
        target_position: Target position size (as fraction of capital)
        max_position_per_asset: Maximum position per asset
        max_gross_exposure: Maximum gross exposure (sum of absolute positions)

    Returns:
        Adjusted position size
    """
    # Limit per asset
    target_position = max(-max_position_per_asset, min(max_position_per_asset, target_position))

    # Limit gross exposure
    current_gross = sum(abs(p) for p in current_positions.values())
    if current_gross + abs(target_position) > max_gross_exposure:
        # Reduce position to stay within limits
        available = max_gross_exposure - current_gross
        if target_position > 0:
            target_position = min(target_position, available)
        else:
            target_position = max(target_position, -available)

    return target_position

