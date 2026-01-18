"""Tests for reproducibility utilities."""

import numpy as np
import pytest

from trading_lab.common.reproducibility import ExperimentTracker, set_random_seeds


def test_set_random_seeds():
    """Test that random seeds are set correctly."""
    set_random_seeds(42)
    
    # Generate random numbers
    np_rand1 = np.random.rand(5)
    
    # Reset and generate again
    set_random_seeds(42)
    np_rand2 = np.random.rand(5)
    
    # Should be identical
    assert np.allclose(np_rand1, np_rand2)


def test_experiment_tracker_lifecycle(tmp_path):
    """Test experiment tracker lifecycle."""
    tracker = ExperimentTracker(experiment_dir=tmp_path)
    
    # Start experiment
    exp_id = tracker.start_experiment(
        name="test_exp",
        config={"param1": 1, "param2": "value"},
        description="Test experiment"
    )
    
    assert exp_id is not None
    assert tracker.current_experiment is not None
    
    # Log metrics
    tracker.log_metric("accuracy", 0.95)
    tracker.log_metric("loss", 0.05)
    
    # End experiment
    tracker.end_experiment(status="completed", results={"final_score": 0.9})
    
    assert tracker.current_experiment is None
    
    # Check file was created
    exp_files = list(tmp_path.glob("*.json"))
    assert len(exp_files) == 1


def test_experiment_tracker_list_experiments(tmp_path):
    """Test listing experiments."""
    tracker = ExperimentTracker(experiment_dir=tmp_path)
    
    # Create multiple experiments
    for i in range(3):
        exp_id = tracker.start_experiment(
            name=f"exp_{i}",
            config={"param": i}
        )
        tracker.log_metric("score", i * 0.1)
        tracker.end_experiment()
    
    # List experiments
    experiments = tracker.list_experiments(limit=10)
    assert len(experiments) == 3
    assert all("name" in exp for exp in experiments)


def test_experiment_tracker_compare(tmp_path):
    """Test comparing experiments."""
    tracker = ExperimentTracker(experiment_dir=tmp_path)
    
    # Create experiments
    exp_ids = []
    for i in range(2):
        exp_id = tracker.start_experiment(
            name=f"exp_{i}",
            config={"param": i}
        )
        tracker.log_metric("accuracy", 0.9 + i * 0.05)
        tracker.log_metric("loss", 0.1 - i * 0.02)
        tracker.end_experiment()
        exp_ids.append(exp_id)
    
    # Compare experiments
    comparison = tracker.compare_experiments(exp_ids, metrics=["accuracy", "loss"])
    
    assert len(comparison) == 2
    assert "accuracy" in comparison.columns
    assert "loss" in comparison.columns
