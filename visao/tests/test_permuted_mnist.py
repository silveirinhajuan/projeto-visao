#!/usr/bin/env python3
"""
test_permuted_mnist.py — TDD tests for Task 73: Permuted-MNIST benchmark.

Tests the Permuted-MNIST benchmark with:
  - 20 tasks with different pixel permutations
  - Multi-class classification (10 classes)
  - Comparison against literature baselines (EWC, SI, LwF)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from visao.bench.permuted_mnist import (
    load_permuted_mnist,
    generate_permutations,
    evaluate_task,
    train_on_task,
    run_permuted_mnist_experiment,
    run_experiment_for_config,
    create_brain,
)
from visao.brain import VisaoBrain


class TestPermutationGeneration:
    """Test permutation generation for Permuted-MNIST."""
    
    def test_generate_permutations_returns_n_tasks(self):
        """Should generate exactly n_tasks permutations."""
        perms = generate_permutations(n_tasks=20, n_pixels=784, seed=42)
        assert len(perms) == 20
    
    def test_permutations_are_valid(self):
        """Each permutation should be a valid shuffling of 0..783."""
        perms = generate_permutations(n_tasks=5, n_pixels=784, seed=42)
        for perm in perms:
            assert len(perm) == 784
            assert set(perm) == set(range(784))
    
    def test_permutations_are_different(self):
        """Different tasks should have different permutations."""
        perms = generate_permutations(n_tasks=5, n_pixels=784, seed=42)
        for i in range(len(perms)):
            for j in range(i + 1, len(perms)):
                assert not np.array_equal(perms[i], perms[j])
    
    def test_permutations_are_reproducible(self):
        """Same seed should produce same permutations."""
        perms1 = generate_permutations(n_tasks=5, n_pixels=784, seed=42)
        perms2 = generate_permutations(n_tasks=5, n_pixels=784, seed=42)
        for p1, p2 in zip(perms1, perms2):
            assert np.array_equal(p1, p2)


class TestDataLoading:
    """Test MNIST data loading and permutation."""
    
    def test_load_permuted_mnist_returns_n_tasks(self):
        """Should produce exactly n_tasks tasks."""
        tasks = load_permuted_mnist("visao/bench/data", n_tasks=20,
                                    max_per_task=100)
        assert len(tasks) == 20
    
    def test_tasks_have_correct_shape(self):
        """Each task should have 784-dimensional inputs."""
        tasks = load_permuted_mnist("visao/bench/data", n_tasks=5,
                                    max_per_task=100)
        for task in tasks:
            assert task["X_train"].shape[1] == 784
            assert task["X_test"].shape[1] == 784
    
    def test_tasks_have_labels(self):
        """Labels should be 0-9 (multi-class)."""
        tasks = load_permuted_mnist("visao/bench/data", n_tasks=5,
                                    max_per_task=100)
        for task in tasks:
            assert set(np.unique(task["y_train"])).issubset(set(range(10)))
            assert set(np.unique(task["y_test"])).issubset(set(range(10)))
    
    def test_data_is_normalized(self):
        """Pixel values should be in [0, 1]."""
        tasks = load_permuted_mnist("visao/bench/data", n_tasks=5,
                                    max_per_task=100)
        for task in tasks:
            assert task["X_train"].min() >= 0.0
            assert task["X_train"].max() <= 1.0
    
    def test_different_tasks_have_different_permutations(self):
        """Different tasks should apply different permutations."""
        tasks = load_permuted_mnist("visao/bench/data", n_tasks=5,
                                    max_per_task=100)
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                assert not np.array_equal(tasks[i]["permutation"],
                                          tasks[j]["permutation"])


class TestBrainCreation:
    """Test brain creation for multi-class classification."""
    
    def test_create_brain_returns_visao_brain(self):
        """create_brain should return a VisaoBrain instance."""
        brain = create_brain(n_in=784, n_hidden=128, n_out=10, seed=42)
        assert isinstance(brain, VisaoBrain)
    
    def test_brain_has_correct_output_dim(self):
        """Brain should have 10 outputs for 10-class classification."""
        brain = create_brain(n_in=784, n_hidden=128, n_out=10, seed=42)
        assert brain.n_out == 10


class TestExperiment:
    """Test the continual learning experiment."""
    
    def test_run_experiment_returns_metrics(self):
        """Experiment should return accuracy and forgetting."""
        tasks = load_permuted_mnist("visao/bench/data", n_tasks=3,
                                    max_per_task=100)
        brain = create_brain(n_in=784, n_hidden=64, n_out=10, seed=42)
        result = run_permuted_mnist_experiment(brain, tasks, n_epochs=2,
                                               n_steps=5)
        
        assert "mean_accuracy" in result
        assert "forgetting" in result
        assert "accuracy_matrix" in result
        assert "final_accuracies" in result
    
    def test_accuracy_matrix_shape(self):
        """Accuracy matrix should be n_tasks x n_tasks."""
        tasks = load_permuted_mnist("visao/bench/data", n_tasks=3,
                                    max_per_task=100)
        brain = create_brain(n_in=784, n_hidden=64, n_out=10, seed=42)
        result = run_permuted_mnist_experiment(brain, tasks, n_epochs=2,
                                               n_steps=5)
        
        matrix = np.array(result["accuracy_matrix"])
        assert matrix.shape == (3, 3)
    
    def test_forgetting_is_non_negative(self):
        """Forgetting should be non-negative (we can't gain from forgetting)."""
        tasks = load_permuted_mnist("visao/bench/data", n_tasks=3,
                                    max_per_task=100)
        brain = create_brain(n_in=784, n_hidden=64, n_out=10, seed=42)
        result = run_permuted_mnist_experiment(brain, tasks, n_epochs=2,
                                               n_steps=5)
        
        assert result["forgetting"] >= 0.0
    
    def test_visao_vs_naive(self):
        """VISÃO should have lower forgetting than naive baseline."""
        tasks = load_permuted_mnist("visao/bench/data", n_tasks=3,
                                    max_per_task=100)
        
        # VISÃO
        brain_visao = create_brain(n_in=784, n_hidden=64, n_out=10, seed=42)
        result_visao = run_permuted_mnist_experiment(brain_visao, tasks,
                                                     n_epochs=2, n_steps=5)
        
        # Naive (no consolidation)
        brain_naive = create_brain(n_in=784, n_hidden=64, n_out=10, seed=42)
        brain_naive.learner.consolidation = 0.0
        brain_naive.learner.surprise_gain = 0.0
        brain_naive.learner.oja_lr = 0.0
        result_naive = run_permuted_mnist_experiment(brain_naive, tasks,
                                                     n_epochs=2, n_steps=5)
        
        # VISÃO should forget less (or equal) than naive
        assert result_visao["forgetting"] <= result_naive["forgetting"] + 0.05


class TestConfigRunner:
    """Test the configuration runner."""
    
    def test_run_config_returns_aggregated_metrics(self):
        """Config runner should return mean and std across seeds."""
        result = run_experiment_for_config(
            "visao",
            n_seeds=2,
            n_tasks=3,
            max_per_task=100,
            n_epochs=2,
            n_steps=5,
        )
        
        assert "mean_accuracy" in result
        assert "mean_accuracy_std" in result
        assert "mean_forgetting" in result
        assert "mean_forgetting_std" in result
        assert result["n_seeds"] == 2
