"""Transaction costs and slippage calculations."""

from typing import Dict


def calculate_transaction_cost(
    position_change: float,
    price: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> float:
    """
    Calculate total transaction cost including fees and slippage.

    Args:
        position_change: Change in position (absolute value)
        price: Current price
        transaction_cost_bps: Transaction cost in basis points
        slippage_bps: Slippage in basis points
        position_size: Position size (absolute value)

    Returns:
        Total cost (fees + slippage)
    """
    if position_change == 0:
        return 0.0

    transaction_cost = abs(position_change) * price * (transaction_cost_bps / 10000.0)
    slippage_cost = abs(position_change) * price * (slippage_bps / 10000.0)

    return transaction_cost + slippage_cost

