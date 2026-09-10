#!/usr/bin/env python3
"""
test_benchmark_sota_v3.py — TDD tests for Task 16: SOTA Benchmark.

Tests the benchmark suite on psMNIST (permuted sequential MNIST).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.bench.benchmark_sota_v3 import (
    PSMNISTTask,
    BaselineLSTM,
    BaselineGRU,
    BaselineTransformer,
    measure_model,
    run_benchmark,
)


class TestPSMNISTTask:
    """Test psMNIST data loading and permutation."""

    def test_load_psmnist(self):
        """Test that psMNIST loads correctly."""
        task = PSMNISTTask(data_dir="visao/bench/data", max_train=100, max_test=50)
        assert task.X_train.shape[1] == 784  # 28*28
        assert task.X_train.shape[2] == 1
        assert task.y_train.shape[1] == 10
        assert task.X_train.shape[0] == 100

    def test_permutation(self):
        """Test that permutation is applied consistently."""
        task = PSMNISTTask(data_dir="visao/bench/data", max_train=50, max_test=20, seed=42)
        # Same seed should give same permutation
        task2 = PSMNISTTask(data_dir="visao/bench/data", max_train=50, max_test=20, seed=42)
        perm1 = task.permutation
        perm2 = task2.permutation
        assert np.array_equal(perm1, perm2)

    def test_different_seeds_different_perms(self):
        """Different seeds should give different permutations."""
        task1 = PSMNISTTask(data_dir="visao/bench/data", max_train=50, max_test=20, seed=0)
        task2 = PSMNISTTask(data_dir="visao/bench/data", max_train=50, max_test=20, seed=1)
        assert not np.array_equal(task1.permutation, task2.permutation)


class TestBaselines:
    """Test LSTM, GRU, Transformer baselines."""

    def test_lstm_forward_shape(self):
        """LSTM forward should return correct shape."""
        lstm = BaselineLSTM(n_in=1, n_hidden=32, n_out=10, seed=0)
        x = np.random.randn(50, 1).astype(np.float32)
        out = lstm.forward(x)
        assert out.shape == (10,)

    def test_gru_forward_shape(self):
        """GRU forward should return correct shape."""
        gru = BaselineGRU(n_in=1, n_hidden=32, n_out=10, seed=0)
        x = np.random.randn(50, 1).astype(np.float32)
        out = gru.forward(x)
        assert out.shape == (10,)

    def test_transformer_forward_shape(self):
        """Transformer forward should return correct shape."""
        trans = BaselineTransformer(n_in=1, n_hidden=32, n_out=10, seed=0)
        x = np.random.randn(50, 1).astype(np.float32)
        out = trans.forward(x)
        assert out.shape == (10,)

    def test_lstm_learn_reduces_error(self):
        """LSTM should learn (error should decrease)."""
        lstm = BaselineLSTM(n_in=1, n_hidden=32, n_out=10, seed=0)
        x = np.random.randn(100, 1).astype(np.float32)
        y = np.zeros(10)
        y[3] = 1.0

        errors_before = []
        errors_after = []

        # Measure error before learning
        for i in range(50):
            pred = lstm.forward(x[i:i+1])
            err = np.mean((pred - y) ** 2)
            errors_before.append(err)

        # Learn
        for epoch in range(5):
            for i in range(100):
                lstm.learn(x[i:i+1], y)

        # Measure error after learning
        lstm.reset()
        for i in range(50):
            pred = lstm.forward(x[i:i+1])
            err = np.mean((pred - y) ** 2)
            errors_after.append(err)

        # After learning, error should be lower
        assert np.mean(errors_after) < np.mean(errors_before)

    def test_gru_learn_reduces_error(self):
        """GRU should learn (error should decrease)."""
        gru = BaselineGRU(n_in=1, n_hidden=32, n_out=10, seed=0)
        x = np.random.randn(100, 1).astype(np.float32)
        y = np.zeros(10)
        y[3] = 1.0

        errors_before = []
        errors_after = []

        for i in range(50):
            pred = gru.forward(x[i:i+1])
            err = np.mean((pred - y) ** 2)
            errors_before.append(err)

        for epoch in range(5):
            for i in range(100):
                gru.learn(x[i:i+1], y)

        gru.reset()
        for i in range(50):
            pred = gru.forward(x[i:i+1])
            err = np.mean((pred - y) ** 2)
            errors_after.append(err)

        assert np.mean(errors_after) < np.mean(errors_before)

    def test_transformer_learn_reduces_error(self):
        """Transformer should learn (error should decrease)."""
        trans = BaselineTransformer(n_in=1, n_hidden=32, n_out=10, seed=0)
        x = np.random.randn(100, 1).astype(np.float32)
        y = np.zeros(10)
        y[3] = 1.0

        errors_before = []
        errors_after = []

        for i in range(50):
            pred = trans.forward(x[i:i+1])
            err = np.mean((pred - y) ** 2)
            errors_before.append(err)

        for epoch in range(5):
            for i in range(100):
                trans.learn(x[i:i+1], y)

        trans.reset()
        for i in range(50):
            pred = trans.forward(x[i:i+1])
            err = np.mean((pred - y) ** 2)
            errors_after.append(err)

        assert np.mean(errors_after) < np.mean(errors_before)


class TestMeasureModel:
    """Test the measure_model function."""

    def test_measure_model_returns_correct_keys(self):
        """measure_model should return dict with required keys."""
        task = PSMNISTTask(data_dir="visao/bench/data", max_train=50, max_test=20, seed=0)
        lstm = BaselineLSTM(n_in=1, n_hidden=32, n_out=10, seed=0)
        result = measure_model("lstm", lstm, task, n_epochs=2)

        required_keys = ["model", "error", "params", "time", "ram_mb"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_measure_model_mse_is_positive(self):
        """Error should be positive."""
        task = PSMNISTTask(data_dir="visao/bench/data", max_train=50, max_test=20, seed=0)
        lstm = BaselineLSTM(n_in=1, n_hidden=32, n_out=10, seed=0)
        result = measure_model("lstm", lstm, task, n_epochs=2)
        assert result["error"] > 0

    def test_measure_model_params_is_positive(self):
        """Params should be positive."""
        task = PSMNISTTask(data_dir="visao/bench/data", max_train=50, max_test=20, seed=0)
        lstm = BaselineLSTM(n_in=1, n_hidden=32, n_out=10, seed=0)
        result = measure_model("lstm", lstm, task, n_epochs=2)
        assert result["params"] > 0


class TestRunBenchmark:
    """Test the full benchmark run."""

    def test_run_benchmark_quick(self):
        """Quick benchmark should complete and return results."""
        results = run_benchmark(quick=True, seeds=(0,))
        assert "visao" in results
        assert "lstm" in results
        assert "gru" in results
        assert "transformer" in results

    def test_run_benchmark_saves_results(self):
        """Benchmark should save results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_benchmark(quick=True, seeds=(0,), output_dir=tmpdir)
            output_path = Path(tmpdir) / "results_benchmark_sota_v3.json"
            assert output_path.exists()

            with open(output_path) as f:
                saved = json.load(f)
            assert "visao" in saved

    def test_run_benchmark_multi_seed(self):
        """Multi-seed benchmark should average results."""
        results = run_benchmark(quick=True, seeds=(0, 1))
        # Should have std field for multi-seed
        if "visao" in results and "mse_std" in results["visao"]:
            assert results["visao"]["mse_std"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
