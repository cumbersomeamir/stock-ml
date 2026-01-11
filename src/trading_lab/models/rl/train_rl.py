"""Train reinforcement learning agent (optional, requires stable-baselines3)."""

import logging
from pathlib import Path

from trading_lab.config.settings import get_settings
from trading_lab.features.feature_store import FeatureStore
from trading_lab.models.rl.env import GYMNASIUM_AVAILABLE, TradingEnv
from trading_lab.unify import unify_prices

logger = logging.getLogger("trading_lab.models.rl")


def train_rl_agent(
    algorithm: str = "PPO",
    total_timesteps: int = 10000,
    force_refresh: bool = False,
) -> None:
    """
    Train RL agent (optional, requires stable-baselines3).

    ⚠️ WARNING: This is experimental. Only use after supervised baselines work well.

    Args:
        algorithm: RL algorithm ('PPO', 'A2C', 'SAC')
        total_timesteps: Number of training timesteps
        force_refresh: If True, retrain even if model exists
    """
    if not GYMNASIUM_AVAILABLE:
        raise ImportError(
            "gymnasium is required for RL. Install with: pip install '.[rl]' or pip install gymnasium stable-baselines3"
        )

    try:
        from stable_baselines3 import A2C, PPO, SAC
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required for RL. Install with: pip install '.[rl]' or pip install stable-baselines3"
        )

    logger.warning("⚠️ RL training is experimental. Ensure supervised baselines work first.")

    settings = get_settings()
    models_dir = settings.get_models_dir()
    feature_store = FeatureStore()

    # Load data
    features_df = feature_store.load_features()
    price_df = unify_prices(force_refresh=False)

    # Create environment
    env = TradingEnv(features_df=features_df, price_df=price_df)

    # Create model
    model_path = models_dir / f"rl_agent_{algorithm}.zip"

    if model_path.exists() and not force_refresh:
        logger.info(f"Loading existing RL model: {model_path}")
        if algorithm == "PPO":
            model = PPO.load(model_path, env=env)
        elif algorithm == "A2C":
            model = A2C.load(model_path, env=env)
        elif algorithm == "SAC":
            model = SAC.load(model_path, env=env)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    else:
        logger.info(f"Training new RL model: {algorithm}")
        if algorithm == "PPO":
            model = PPO("MlpPolicy", env, verbose=1)
        elif algorithm == "A2C":
            model = A2C("MlpPolicy", env, verbose=1)
        elif algorithm == "SAC":
            model = SAC("MlpPolicy", env, verbose=1)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Train
        model.learn(total_timesteps=total_timesteps)

        # Save
        model.save(model_path)
        logger.info(f"Saved RL model to {model_path}")

    logger.info("RL training complete")

