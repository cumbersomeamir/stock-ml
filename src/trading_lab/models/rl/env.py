"""Trading environment for reinforcement learning (optional, requires gymnasium)."""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_lab.models.rl")

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    logger.warning("gymnasium not available. RL module is disabled.")


if GYMNASIUM_AVAILABLE:

    class TradingEnv(gym.Env):
        """
        Trading environment for reinforcement learning.

        Observation: Recent features window
        Action: Position {-1, 0, +1}
        Reward: PnL - costs - risk_penalty
        """

        def __init__(
            self,
            features_df: pd.DataFrame,
            price_df: pd.DataFrame,
            initial_capital: float = 100000.0,
            transaction_cost_bps: float = 10.0,
            slippage_bps: float = 5.0,
            window_size: int = 20,
            max_position: float = 0.1,
        ):
            """
            Initialize trading environment.

            Args:
                features_df: Features DataFrame
                price_df: Price DataFrame with adj_close
                initial_capital: Initial capital
                transaction_cost_bps: Transaction cost in basis points
                slippage_bps: Slippage in basis points
                window_size: Size of observation window
                max_position: Maximum position size as fraction of capital
            """
            super().__init__()

            self.features_df = features_df.sort_values(["ticker", "date"]).reset_index(drop=True)
            self.price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)
            self.initial_capital = initial_capital
            self.transaction_cost_bps = transaction_cost_bps / 10000.0
            self.slippage_bps = slippage_bps / 10000.0
            self.window_size = window_size
            self.max_position = max_position

            # Get feature columns (exclude date, ticker, targets)
            self.feature_cols = [
                col
                for col in self.features_df.columns
                if col not in ["date", "ticker", "y_class", "y_reg", "regime"]
            ]

            # Merge features and prices
            self.data = self.features_df.merge(
                self.price_df[["date", "ticker", "adj_close"]], on=["date", "ticker"], how="inner"
            )
            self.data = self.data.sort_values(["ticker", "date"]).reset_index(drop=True)

            # Create date index
            self.dates = self.data["date"].unique()
            self.current_idx = 0

            # Observation space: flattened feature window
            n_features = len(self.feature_cols)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(window_size * n_features,), dtype=np.float32
            )

            # Action space: position {-1, 0, +1}
            self.action_space = spaces.Discrete(3)  # -1, 0, +1

            # State
            self.reset()

        def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> tuple:
            """Reset environment."""
            super().reset(seed=seed)

            self.current_idx = self.window_size  # Start after window
            self.capital = self.initial_capital
            self.positions = {}  # ticker -> position (-1, 0, +1)
            self.entry_prices = {}  # ticker -> entry price
            self.max_drawdown = 0.0
            self.peak_capital = self.initial_capital

            observation = self._get_observation()
            info = {}

            return observation, info

        def step(self, action: int) -> tuple:
            """
            Execute one step.

            Args:
                action: Position action (0=-1, 1=0, 2=+1)

            Returns:
                observation, reward, terminated, truncated, info
            """
            if self.current_idx >= len(self.dates):
                return self._get_observation(), 0.0, True, False, {}

            current_date = self.dates[self.current_idx]
            date_data = self.data[self.data["date"] == current_date]

            # Convert action to position
            position_action = action - 1  # -1, 0, +1

            # Update positions for each ticker
            total_pnl = 0.0
            total_cost = 0.0

            for _, row in date_data.iterrows():
                ticker = row["ticker"]
                price = row["adj_close"]

                # Get current position
                current_pos = self.positions.get(ticker, 0)

                # Calculate PnL from previous position
                if ticker in self.entry_prices:
                    pnl = current_pos * (price - self.entry_prices[ticker]) / self.entry_prices[ticker]
                    total_pnl += pnl * self.capital * abs(current_pos) * self.max_position

                # Update position
                new_pos = position_action  # Simplified: same action for all tickers
                if new_pos != current_pos:
                    # Transaction cost and slippage
                    cost = abs(new_pos - current_pos) * self.transaction_cost_bps
                    slippage = abs(new_pos - current_pos) * self.slippage_bps
                    total_cost += (cost + slippage) * self.capital * self.max_position

                    if new_pos != 0:
                        self.entry_prices[ticker] = price * (1 + slippage if new_pos > 0 else 1 - slippage)
                    else:
                        self.entry_prices.pop(ticker, None)

                    self.positions[ticker] = new_pos

            # Update capital
            self.capital = self.capital + total_pnl - total_cost
            self.peak_capital = max(self.peak_capital, self.capital)
            drawdown = (self.peak_capital - self.capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, drawdown)

            # Reward: PnL - costs - risk penalty
            reward = total_pnl - total_cost - 0.1 * drawdown  # Risk penalty

            # Move to next date
            self.current_idx += 1
            terminated = self.current_idx >= len(self.dates)
            truncated = False

            observation = self._get_observation()
            info = {
                "capital": self.capital,
                "pnl": total_pnl,
                "cost": total_cost,
                "drawdown": drawdown,
            }

            return observation, reward, terminated, truncated, info

        def _get_observation(self) -> np.ndarray:
            """Get current observation (feature window)."""
            if self.current_idx < self.window_size:
                # Pad with zeros
                observation = np.zeros((self.window_size, len(self.feature_cols)))
                available = self.data[self.data["date"] <= self.dates[self.current_idx]]
                if len(available) > 0:
                    recent = available.tail(self.window_size)
                    n_available = len(recent)
                    observation[-n_available:] = recent[self.feature_cols].fillna(0).values
            else:
                date_idx = self.dates[self.current_idx - self.window_size : self.current_idx]
                window_data = self.data[self.data["date"].isin(date_idx)]
                if len(window_data) > 0:
                    observation = window_data[self.feature_cols].fillna(0).values
                    # Flatten and pad if needed
                    observation = observation.reshape(-1, len(self.feature_cols))
                    if len(observation) < self.window_size:
                        padding = np.zeros((self.window_size - len(observation), len(self.feature_cols)))
                        observation = np.vstack([padding, observation])
                else:
                    observation = np.zeros((self.window_size, len(self.feature_cols)))

            return observation.flatten().astype(np.float32)

else:
    # Stub class if gymnasium is not available
    class TradingEnv:
        """Stub class when gymnasium is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium is required for RL. Install with: pip install gymnasium stable-baselines3")

