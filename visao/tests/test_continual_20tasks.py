#!/usr/bin/env python3
"""
test_continual_20tasks.py — Testes TDD para o experimento de 20+ tarefas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from experiments.continual_20tasks import (
    make_task,
    generate_task_suite,
    run_continual_experiment,
    run_experiment_for_config,
)
from visao.brain import VisaoBrain


class TestTaskGeneration:
    def test_make_task_returns_correct_shapes(self):
        u_tr, y_tr, u_te, y_te = make_task("sine", n=100, seed=0)
        assert u_tr.shape == (70, 2)
        assert y_tr.shape == (70, 1)
        assert u_te.shape == (30, 2)
        assert y_te.shape == (30, 1)

    def test_make_task_different_seeds_different_data(self):
        u1, y1, _, _ = make_task("sine", n=100, seed=0)
        u2, y2, _, _ = make_task("sine", n=100, seed=1)
        assert not np.allclose(u1, u2)

    def test_generate_task_suite_20_tasks(self):
        tasks = generate_task_suite(20, seed=42)
        assert len(tasks) == 20
        assert all("u_tr" in t for t in tasks)
        assert all("y_tr" in t for t in tasks)
        assert all("u_te" in t for t in tasks)
        assert all("y_te" in t for t in tasks)

    def test_tasks_have_distinct_distributions(self):
        tasks = generate_task_suite(20, seed=42)
        # Verificar que tarefas têm diferentes "kinds"
        kinds = set(t["kind"] for t in tasks)
        assert len(kinds) > 1


class TestContinualExperiment:
    def test_experiment_runs_without_error(self):
        tasks = generate_task_suite(5, seed=42)
        brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)
        result = run_continual_experiment(brain, tasks)
        assert "mean_forgetting" in result
        assert "forgetting_per_task" in result
        assert len(result["forgetting_per_task"]) == 5

    def test_experiment_forgetting_is_finite(self):
        tasks = generate_task_suite(5, seed=42)
        brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)
        result = run_continual_experiment(brain, tasks)
        assert np.isfinite(result["mean_forgetting"])

    def test_full_config_runs(self):
        result = run_experiment_for_config("full", n_tasks=5, n_seeds=2, seed_start=42)
        assert result["config"] == "full"
        assert result["mean_forgetting"] is not None
        assert np.isfinite(result["mean_forgetting"])

    def test_naive_config_runs(self):
        result = run_experiment_for_config("naive", n_tasks=5, n_seeds=2, seed_start=42)
        assert result["config"] == "naive"
        assert np.isfinite(result["mean_forgetting"])

    def test_ewc_only_config_runs(self):
        result = run_experiment_for_config("ewc_only", n_tasks=5, n_seeds=2, seed_start=42)
        assert result["config"] == "ewc_only"
        assert np.isfinite(result["mean_forgetting"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
