#!/usr/bin/env python3
"""
test_split_mnist.py — TDD tests for Task 71: Split-MNIST accuracy >80%.

Tests the redesigned Split-MNIST benchmark with:
  - Full image (784 pixels) as input (n_in=784)
  - Reservoir steps per image (n_steps=10)
  - Spectral radius rescaling
  - Multiple epochs for convergence
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from visao.bench.split_mnist import (
    load_split_mnist,
    evaluate_task,
    train_on_task,
    run_split_mnist_experiment,
    run_experiment_for_config,
    create_brain,
)
from visao.brain import VisaoBrain


class TestDataLoading:
    """Test MNIST data loading and splitting."""
    
    def test_load_split_mnist_returns_5_tasks(self):
        """Split-MNIST should produce 5 binary tasks."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=100)
        assert len(tasks) == 5
    
    def test_tasks_have_correct_digits(self):
        """Each task should have the correct digit pair."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=100)
        expected_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        for task, (a, b) in zip(tasks, expected_pairs):
            assert task["digit_a"] == a
            assert task["digit_b"] == b
    
    def test_tasks_have_train_test_split(self):
        """Each task should have both train and test data."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=100)
        for task in tasks:
            assert task["X_train"].shape[0] > 0
            assert task["X_test"].shape[0] > 0
            assert task["X_train"].shape[1] == 784  # 28*28
            assert task["X_test"].shape[1] == 784
    
    def test_labels_are_binary(self):
        """Labels should be 0 or 1."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=100)
        for task in tasks:
            assert set(np.unique(task["y_train"])).issubset({0.0, 1.0})
            assert set(np.unique(task["y_test"])).issubset({0.0, 1.0})
    
    def test_data_is_normalized(self):
        """Pixel values should be in [0, 1]."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=100)
        for task in tasks:
            assert task["X_train"].min() >= 0.0
            assert task["X_train"].max() <= 1.0


class TestBrainCreation:
    """Test brain creation with spectral radius rescaling."""
    
    def test_create_brain_returns_visao_brain(self):
        """create_brain should return a VisaoBrain instance."""
        brain = create_brain(n_in=784, n_hidden=128, n_out=1, seed=42)
        assert isinstance(brain, VisaoBrain)
    
    def test_brain_has_correct_dimensions(self):
        """Brain should have correct input/output dimensions."""
        brain = create_brain(n_in=784, n_hidden=128, n_out=1, seed=42)
        assert brain.n_in == 784
        assert brain.n_hidden == 128
        assert brain.n_out == 1
    
    def test_spectral_radius_is_set(self):
        """W_rec should be rescaled to target spectral radius."""
        brain = create_brain(n_in=784, n_hidden=128, n_out=1, seed=42,
                            spectral_radius=0.95)
        eigs = np.linalg.eigvals(brain.cell.W_rec)
        actual_sr = np.max(np.abs(eigs))
        assert abs(actual_sr - 0.95) < 0.05  # within tolerance


class TestEvaluateTask:
    """Test task evaluation."""
    
    def test_evaluate_returns_accuracy_in_range(self):
        """Accuracy should be between 0 and 1."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=100)
        brain = create_brain(n_in=784, n_hidden=128, n_out=1, seed=42)
        acc = evaluate_task(brain, tasks[0])
        assert 0.0 <= acc <= 1.0
    
    def test_evaluate_untrained_brain_is_random(self):
        """Untrained brain should have ~50% accuracy on binary task."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=100)
        brain = create_brain(n_in=784, n_hidden=128, n_out=1, seed=42)
        acc = evaluate_task(brain, tasks[0])
        # Random guessing should be around 50% (with some tolerance)
        assert 0.3 <= acc <= 0.7


class TestTrainOnTask:
    """Test training on a single task."""
    
    def test_training_changes_weights(self):
        """Training should change the brain's weights."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=100)
        brain = create_brain(n_in=784, n_hidden=128, n_out=1, seed=42)
        
        w_before = brain.learner.W_out.copy()
        train_on_task(brain, tasks[0], n_epochs=1)
        w_after = brain.learner.W_out
        
        assert not np.allclose(w_before, w_after)
    
    def test_training_improves_accuracy(self):
        """After training, accuracy should improve significantly."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=200)
        brain = create_brain(n_in=784, n_hidden=256, n_out=1, seed=42, lr=0.1)
        
        acc_before = evaluate_task(brain, tasks[0])
        train_on_task(brain, tasks[0], n_epochs=5, n_steps=10)
        acc_after = evaluate_task(brain, tasks[0])
        
        # Should improve significantly (target: >80%)
        assert acc_after > acc_before
        assert acc_after >= 0.70  # At least 70% after training


class TestContinualExperiment:
    """Test the full continual learning experiment."""
    
    def test_experiment_returns_correct_structure(self):
        """Experiment should return all required fields."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=100)
        brain = create_brain(n_in=784, n_hidden=128, n_out=1, seed=42)
        result = run_split_mnist_experiment(brain, tasks, n_epochs=1)
        
        assert "accuracy_matrix" in result
        assert "mean_accuracy" in result
        assert "forgetting" in result
        assert "forgetting_per_task" in result
        assert "final_accuracies" in result
    
    def test_accuracy_matrix_shape(self):
        """Accuracy matrix should be n_tasks x n_tasks."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=100)
        brain = create_brain(n_in=784, n_hidden=128, n_out=1, seed=42)
        result = run_split_mnist_experiment(brain, tasks, n_epochs=1)
        
        matrix = np.array(result["accuracy_matrix"])
        assert matrix.shape == (5, 5)
    
    def test_mean_accuracy_in_range(self):
        """Mean accuracy should be between 0 and 1."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=100)
        brain = create_brain(n_in=784, n_hidden=128, n_out=1, seed=42)
        result = run_split_mnist_experiment(brain, tasks, n_epochs=1)
        
        assert 0.0 <= result["mean_accuracy"] <= 1.0


class TestConfigComparison:
    """Test different brain configurations."""
    
    def test_visao_config_runs(self):
        """Full VisaoBrain config should run."""
        result = run_experiment_for_config("visao", n_seeds=2, max_per_task=100,
                                          n_epochs=1)
        assert result["config"] == "visao"
        assert result["mean_accuracy"] is not None
    
    def test_naive_config_runs(self):
        """Naive config (no consolidation) should run."""
        result = run_experiment_for_config("naive", n_seeds=2, max_per_task=100,
                                          n_epochs=1)
        assert result["config"] == "naive"
        assert result["mean_accuracy"] is not None
    
    def test_ewc_only_config_runs(self):
        """EWC-only config should run."""
        result = run_experiment_for_config("ewc_only", n_seeds=2, max_per_task=100,
                                          n_epochs=1)
        assert result["config"] == "ewc_only"
        assert result["mean_accuracy"] is not None
    
    def test_visao_forgets_less_than_naive(self):
        """VisaoBrain should forget less than naive baseline."""
        result_visao = run_experiment_for_config("visao", n_seeds=2, max_per_task=100,
                                                n_epochs=1)
        result_naive = run_experiment_for_config("naive", n_seeds=2, max_per_task=100,
                                                n_epochs=1)
        
        # VisaoBrain should have less forgetting (with some tolerance)
        assert result_visao["mean_forgetting"] <= result_naive["mean_forgetting"] + 0.10


class TestAccuracyTarget:
    """Test that accuracy target (>80%) is achievable."""
    
    def test_single_task_accuracy_above_80(self):
        """After training on a single task, accuracy should exceed 80%."""
        tasks = load_split_mnist("visao/bench/data", max_per_task=200)
        brain = create_brain(n_in=784, n_hidden=256, n_out=1, seed=42, lr=0.1)
        
        # Train on first task
        train_on_task(brain, tasks[0], n_epochs=5, n_steps=10)
        acc = evaluate_task(brain, tasks[0])
        
        assert acc >= 0.80, f"Accuracy {acc:.3f} < 0.80 target"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
