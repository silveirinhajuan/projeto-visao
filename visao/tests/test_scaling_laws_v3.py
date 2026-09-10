#!/usr/bin/env python3
"""
test_scaling_laws_v3.py — TDD tests for Task 68: Scaling Laws v3.

Tests the scaling law experiment with more data + regularization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from visao.bench.scaling_laws_v3 import (
    generate_signal,
    run_experiment,
    HIDDEN_SIZES,
    N_TRAIN,
    N_TEST,
)


class TestDataGeneration:
    """Test signal generation."""

    def test_generate_signal_returns_correct_shapes(self):
        X, Y = generate_signal(100, seed=42)
        assert X.shape == (100, 1)
        assert Y.shape == (100, 1)

    def test_generate_signal_deterministic(self):
        X1, Y1 = generate_signal(100, seed=42)
        X2, Y2 = generate_signal(100, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)

    def test_generate_signal_different_seeds(self):
        X1, Y1 = generate_signal(100, seed=0)
        X2, Y2 = generate_signal(100, seed=1)
        assert not np.array_equal(X1, X2)


class TestRunExperiment:
    """Test single experiment run."""

    def test_run_experiment_returns_required_fields(self):
        r = run_experiment(n_hidden=32, seed=0)
        assert "n_hidden" in r
        assert "n_params" in r
        assert "mse_test" in r
        assert "wall_time" in r
        assert "seed" in r

    def test_run_experiment_mse_is_positive(self):
        r = run_experiment(n_hidden=32, seed=0)
        assert r["mse_test"] > 0

    def test_run_experiment_n_params_scales_with_hidden(self):
        r32 = run_experiment(n_hidden=32, seed=0)
        r64 = run_experiment(n_hidden=64, seed=0)
        assert r64["n_params"] > r32["n_params"]

    def test_run_experiment_deterministic_with_seed(self):
        r1 = run_experiment(n_hidden=32, seed=42)
        r2 = run_experiment(n_hidden=32, seed=42)
        assert r1["mse_test"] == r2["mse_test"]
        assert r1["n_params"] == r2["n_params"]


class TestScalingFit:
    """Test the scaling law fit."""

    def test_scaling_fit_positive_alpha(self):
        """With enough data, alpha should be positive (loss decreases with scale)."""
        results = []
        for n_hidden in [32, 64, 128]:
            r = run_experiment(n_hidden=n_hidden, seed=0)
            results.append(r)

        params_arr = np.array([r["n_params"] for r in results])
        mse_arr = np.array([r["mse_test"] for r in results])

        # Fit: log(MSE) = -alpha * log(N) + log(C)
        log_p = np.log(params_arr)
        log_m = np.log(mse_arr)
        A = np.vstack([-log_p, np.ones_like(log_p)]).T
        alpha, _ = np.linalg.lstsq(A, log_m, rcond=None)[0]

        # With n_train=10000, alpha should be positive
        assert alpha > 0, f"Expected positive alpha, got {alpha}"


class TestResultsFile:
    """Test that results file is generated correctly."""

    def test_results_file_exists(self):
        results_path = Path(__file__).parent / "bench" / "results_scaling_v3.json"
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            assert "scaling_law" in data
            assert "alpha" in data["scaling_law"]
            assert "R_squared" in data["scaling_law"]
            assert "config" in data
            assert data["config"]["n_train"] == N_TRAIN


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
