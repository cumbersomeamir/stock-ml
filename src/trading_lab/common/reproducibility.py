"""Reproducibility utilities for experiments."""

import hashlib
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("trading_lab.reproducibility")


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to set seeds for common ML libraries
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logger.info(f"Set random seeds to {seed}")


class ExperimentTracker:
    """
    Track experiments for reproducibility and comparison.
    
    Records configuration, code version, data version, and results.
    """

    def __init__(self, experiment_dir: Optional[Path] = None):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory to store experiment logs
        """
        if experiment_dir is None:
            from trading_lab.config.settings import get_settings
            settings = get_settings()
            experiment_dir = settings.get_artifacts_dir() / "experiments"
        
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment: Optional[Dict] = None

    def start_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        description: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> str:
        """
        Start a new experiment.
        
        Args:
            name: Experiment name
            config: Configuration dictionary
            description: Optional description
            tags: Optional tags for categorization
            
        Returns:
            Experiment ID
        """
        experiment_id = self._generate_experiment_id(name, config)
        
        self.current_experiment = {
            "id": experiment_id,
            "name": name,
            "description": description,
            "config": config,
            "tags": tags or [],
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "results": {},
            "metrics": {},
            "artifacts": {},
            "status": "running",
        }
        
        logger.info(f"Started experiment: {experiment_id} ({name})")
        return experiment_id

    def log_metric(self, key: str, value: Any) -> None:
        """Log a metric for the current experiment."""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment["metrics"][key] = value

    def log_artifact(self, key: str, path: Path) -> None:
        """Log an artifact path for the current experiment."""
        if self.current_experiment is None:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        self.current_experiment["artifacts"][key] = str(path)

    def end_experiment(self, status: str = "completed", results: Optional[Dict] = None) -> None:
        """
        End the current experiment.
        
        Args:
            status: Experiment status ('completed', 'failed', 'cancelled')
            results: Optional results dictionary
        """
        if self.current_experiment is None:
            raise ValueError("No active experiment to end.")
        
        self.current_experiment["end_time"] = datetime.now().isoformat()
        self.current_experiment["status"] = status
        
        if results:
            self.current_experiment["results"] = results
        
        # Save experiment log
        experiment_file = self.experiment_dir / f"{self.current_experiment['id']}.json"
        with open(experiment_file, "w") as f:
            json.dump(self.current_experiment, f, indent=2, default=str)
        
        logger.info(f"Ended experiment: {self.current_experiment['id']} ({status})")
        self.current_experiment = None

    def list_experiments(self, limit: int = 10) -> list[Dict]:
        """
        List recent experiments.
        
        Args:
            limit: Maximum number of experiments to return
            
        Returns:
            List of experiment dictionaries
        """
        experiment_files = sorted(
            self.experiment_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]
        
        experiments = []
        for file in experiment_files:
            with open(file, "r") as f:
                experiments.append(json.load(f))
        
        return experiments

    def compare_experiments(self, experiment_ids: list[str], metrics: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: List of metrics to compare (if None, uses all metrics)
            
        Returns:
            DataFrame comparing experiments
        """
        comparison_data = []
        
        for exp_id in experiment_ids:
            exp_file = self.experiment_dir / f"{exp_id}.json"
            if not exp_file.exists():
                logger.warning(f"Experiment not found: {exp_id}")
                continue
            
            with open(exp_file, "r") as f:
                exp = json.load(f)
            
            row = {
                "experiment_id": exp_id,
                "name": exp.get("name"),
                "status": exp.get("status"),
                "start_time": exp.get("start_time"),
            }
            
            # Add metrics
            exp_metrics = exp.get("metrics", {})
            if metrics is None:
                row.update(exp_metrics)
            else:
                for metric in metrics:
                    row[metric] = exp_metrics.get(metric)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)

    @staticmethod
    def _generate_experiment_id(name: str, config: Dict) -> str:
        """Generate unique experiment ID from name and config."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{name}_{timestamp}_{config_hash}"
