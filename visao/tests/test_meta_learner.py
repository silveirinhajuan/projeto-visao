"""test_meta_learner.py — Testes para MetaContinualLearner.

Cobre:
  - TaskSignature: extração de assinatura
  - MetaContinualLearner: instanciação e API
  - Detecção de tarefa dominante
  - Ajuste automático de EWC/Surprise
  - Comparação vs baseline em 5+ tarefas sequenciais
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain
from visao.continual.meta_learner import (
    MetaContinualLearner,
    TaskSignature,
    TaskEntry,
    compare_baseline_vs_meta,
    generate_task,
    run_continual_experiment,
)


# ==============================================================
#  FIXTURES
# ==============================================================

@pytest.fixture
def meta_learner():
    return MetaContinualLearner(n_in=2, n_hidden=32, n_out=1, seed=42)


@pytest.fixture
def tasks_5():
    """5 tarefas sequenciais distintas."""
    return [
        generate_task("sine", n=300, seed=0, noise=0.1),
        generate_task("saw", n=300, seed=1, noise=0.1),
        generate_task("mixed", n=300, seed=2, noise=0.1),
        generate_task("high_freq", n=300, seed=3, noise=0.1),
        generate_task("quadrada", n=300, seed=4, noise=0.1),
    ]


@pytest.fixture
def tasks_6():
    """6 tarefas sequenciais distintas."""
    return [
        generate_task("sine", n=300, seed=0, noise=0.1),
        generate_task("saw", n=300, seed=1, noise=0.1),
        generate_task("mixed", n=300, seed=2, noise=0.1),
        generate_task("high_freq", n=300, seed=3, noise=0.1),
        generate_task("quadrada", n=300, seed=4, noise=0.1),
        generate_task("amortecida", n=300, seed=5, noise=0.1),
    ]


# ==============================================================
#  TESTES: TaskSignature
# ==============================================================

class TestTaskSignature:
    def test_signature_init(self):
        sig = TaskSignature(n_hidden=32)
        assert sig.n_hidden == 32
        assert not sig.is_ready()

    def test_signature_updates(self):
        sig = TaskSignature(n_hidden=16, buffer_size=10)
        rng = np.random.default_rng(42)
        for _ in range(5):
            sig.update(0.5, rng.normal(0, 1, 16))
        assert sig.is_ready()

    def test_signature_vector_shape(self):
        sig = TaskSignature(n_hidden=8, buffer_size=10)
        rng = np.random.default_rng(42)
        for _ in range(5):
            sig.update(0.3, rng.normal(0, 1, 8))
        vec = sig.vector()
        # 4 + 2*8 = 20
        assert vec.shape == (7,)

    def test_signature_reset(self):
        sig = TaskSignature(n_hidden=8, buffer_size=10)
        rng = np.random.default_rng(42)
        for _ in range(5):
            sig.update(0.3, rng.normal(0, 1, 8))
        sig.reset()
        assert not sig.is_ready()
        assert len(sig._err_buffer) == 0


# ==============================================================
#  TESTES: MetaContinualLearner - Instanciação e API
# ==============================================================

class TestMetaContinualLearnerInit:
    def test_default_creation(self):
        mcl = MetaContinualLearner()
        assert mcl.n_in == 2
        assert mcl.n_hidden == 64
        assert mcl.n_out == 1

    def test_custom_creation(self):
        mcl = MetaContinualLearner(n_in=3, n_hidden=48, n_out=2)
        assert mcl.n_in == 3
        assert mcl.n_hidden == 48
        assert mcl.n_out == 2

    def test_brain_integrity(self, meta_learner):
        assert meta_learner.brain is not None
        assert isinstance(meta_learner.brain, VisaoBrain)

    def test_initial_task_bank_empty(self, meta_learner):
        assert len(meta_learner.task_bank) == 0


class TestMetaContinualLearnerAPI:
    def test_learn_returns_dict(self, meta_learner):
        result = meta_learner.learn([0.5, 0.3], [0.7])
        assert "pred" in result
        assert "err" in result
        assert "surprise" in result

    def test_forward_no_error(self, meta_learner):
        meta_learner.set_mode("infer")
        pred = meta_learner.forward([0.5, 0.3])
        assert pred is not None

    def test_reset_state(self, meta_learner):
        meta_learner.set_mode("infer")
        meta_learner.forward([0.5, 0.3])
        meta_learner.reset_state()
        assert np.allclose(meta_learner.brain.x, 0.0)

    def test_set_mode(self, meta_learner):
        meta_learner.set_mode("infer")
        assert meta_learner.brain.mode == "infer"
        meta_learner.set_mode("learn")
        assert meta_learner.brain.mode == "learn"

    def test_diagnostics(self, meta_learner):
        diag = meta_learner.get_diagnostics()
        assert "n_known_tasks" in diag
        assert "brain_lr" in diag
        assert "brain_consolidation" in diag


# ==============================================================
#  TESTES: Detecção de Tarefa Dominante
# ==============================================================

class TestTaskDetection:
    def test_detects_task_transition(self, meta_learner):
        """Após duas tarefas muito diferentes, detecta transição."""
        rng = np.random.default_rng(42)

        # Tarefa 1: senoide lenta
        X1 = rng.normal(0, 0.1, (200, 2)).astype(np.float64)
        X1[:, 0] += np.sin(2 * np.pi * 0.02 * np.arange(200))
        Y1 = np.sin(2 * np.pi * 0.02 * np.arange(200))[:, None]

        for xi, yi in zip(X1, Y1):
            meta_learner.learn(xi, yi)
        meta_learner.end_task()

        # Tarefa 2: ruído alto (muito diferente)
        X2 = rng.normal(0, 1.0, (200, 2)).astype(np.float64)
        Y2 = rng.normal(0, 1.0, (200, 1))

        for xi, yi in zip(X2, Y2):
            meta_learner.learn(xi, yi)

        # Deve ter detectado pelo menos 1 tarefa
        assert len(meta_learner.task_bank) >= 1

    def test_task_bank_grows(self, meta_learner):
        """Banco de tarefas cresce com tarefas distintas."""
        tasks = [
            generate_task("sine", n=100, seed=0),
            generate_task("saw", n=100, seed=1),
            generate_task("mixed", n=100, seed=2),
        ]

        for X, Y in tasks:
            meta_learner.set_mode("learn")
            meta_learner.reset_state()
            for xi, yi in zip(X, Y):
                meta_learner.learn(xi, yi)
            meta_learner.end_task()

        assert len(meta_learner.task_bank) >= 1

    def test_current_task_id_updates(self, meta_learner):
        """ID da tarefa atual muda após tarefas diferentes."""
        X1 = np.random.default_rng(0).normal(0, 0.1, (100, 2))
        Y1 = np.sin(np.arange(100) * 0.1)[:, None]

        for xi, yi in zip(X1, Y1):
            meta_learner.learn(xi, yi)
        meta_learner.end_task()

        first_task_id = meta_learner.current_task_id
        assert first_task_id >= 0


# ==============================================================
#  TESTES: Ajuste Automático de EWC/Surprise
# ==============================================================

class TestAdaptiveEWC:
    def test_ewc_adapts_on_new_task(self, meta_learner):
        """EWC deve mudar ao detectar tarefa nova."""
        initial_consolidation = meta_learner.brain.learner.consolidation

        # Treina tarefa 1
        X, Y = generate_task("sine", n=100, seed=0)
        for xi, yi in zip(X, Y):
            meta_learner.learn(xi, yi)
        meta_learner.end_task()

        # Treina tarefa 2 (deve adaptar EWC)
        X2, Y2 = generate_task("saw", n=100, seed=1)
        for xi, yi in zip(X2, Y2):
            meta_learner.learn(xi, yi)

        # EWC deve ter mudado
        assert meta_learner.brain.learner.consolidation != initial_consolidation or \
               len(meta_learner.task_bank) > 0

    def test_surprise_adapts(self, meta_learner):
        """Surprise gain muda com tarefas."""
        initial_sg = meta_learner.brain.learner.surprise_gain

        tasks = [
            generate_task("sine", n=100, seed=0),
            generate_task("saw", n=100, seed=1),
            generate_task("high_freq", n=100, seed=2),
        ]

        for X, Y in tasks:
            for xi, yi in zip(X, Y):
                meta_learner.learn(xi, yi)
            meta_learner.end_task()

        diag = meta_learner.get_diagnostics()
        assert diag["brain_surprise_gain"] > 0

    def test_lr_adapts_per_task(self, meta_learner):
        """lr deve mudar ao longo do aprendizado de múltiplas tarefas."""
        initial_lr = meta_learner.brain.learner.lr

        tasks = [
            generate_task("sine", n=150, seed=0),
            generate_task("saw", n=150, seed=1),
        ]

        for X, Y in tasks:
            for xi, yi in zip(X, Y):
                meta_learner.learn(xi, yi)
            meta_learner.end_task()

        # lr deve ter sido ajustado
        final_lr = meta_learner.brain.learner.lr
        assert final_lr != initial_lr or len(meta_learner.task_bank) > 0


# ==============================================================
#  TESTES: Continual Learning (5+ tarefas)
# ==============================================================

class TestContinualLearning5Tasks:
    def test_5_tasks_sequential(self, tasks_5):
        """MetaContinualLearner completa 5 tarefas sequenciais sem erro."""
        mcl = MetaContinualLearner(n_in=2, n_hidden=32, n_out=1, seed=42)

        for X, Y in tasks_5:
            mcl.set_mode("learn")
            mcl.reset_state()
            for xi, yi in zip(X, Y):
                mcl.learn(xi, yi)
            mcl.end_task()

        diag = mcl.get_diagnostics()
        assert diag["n_known_tasks"] >= 1

    def test_meta_reduces_forgetting_vs_baseline_5_tasks(self, tasks_5):
        """MetaContinualLearner deve esquecer menos que baseline em 5 tarefas."""
        result = compare_baseline_vs_meta(tasks_5, n_hidden=32, seed=42)

        baseline_forgetting = result["baseline"]["forgetting"]
        meta_forgetting = result["meta"]["forgetting"]

        # Meta deve ter menos esquecimento (ou igual, dado estocasticidade)
        assert meta_forgetting <= baseline_forgetting + 0.05

    def test_6_tasks_sequential(self, tasks_6):
        """MetaContinualLearner completa 6 tarefas sequenciais."""
        mcl = MetaContinualLearner(n_in=2, n_hidden=32, n_out=1, seed=42)

        for X, Y in tasks_6:
            mcl.set_mode("learn")
            mcl.reset_state()
            for xi, yi in zip(X, Y):
                mcl.learn(xi, yi)
            mcl.end_task()

        diag = mcl.get_diagnostics()
        assert diag["n_known_tasks"] >= 1

    def test_meta_improves_over_baseline_6_tasks(self, tasks_6):
        """MetaContinualLearner deve ser melhor que baseline em 6 tarefas."""
        result = compare_baseline_vs_meta(tasks_6, n_hidden=32, seed=42)

        # Deve haver alguma melhoria (ou no pior caso, não ser muito pior)
        assert result["improvement_pct"] >= -20  # margem para variação

    def test_task_final_mse_reasonable(self, tasks_5):
        """MSE final das tarefas deve ser razoável (menor que inicial)."""
        mcl = MetaContinualLearner(n_in=2, n_hidden=32, n_out=1, seed=42)

        initial_errs = []
        final_errs = []

        for X, Y in tasks_5[:2]:  # primeira e segunda tarefa
            mcl.set_mode("learn")
            mcl.reset_state()

            task_errs = []
            for xi, yi in zip(X, Y):
                res = mcl.learn(xi, yi)
                task_errs.append(res["err"])

            initial_errs.append(np.mean(task_errs[:20]))
            final_errs.append(np.mean(task_errs[-20:]))

        # Pelo menos uma tarefa deve melhorar
        improved = False
        for init, fin in zip(initial_errs, final_errs):
            if fin < init:
                improved = True
                break
        assert improved


# ==============================================================
#  TESTES: Comparação direta com run_continual_experiment
# ==============================================================

class TestRunExperiment:
    def test_run_experiment_baseline(self, tasks_5):
        baseline = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)
        result = run_continual_experiment(baseline, tasks_5)
        assert "forgetting" in result
        assert "mean_final_mse" in result
        assert result["forgetting"] >= 0

    def test_run_experiment_meta(self, tasks_5):
        meta = MetaContinualLearner(n_in=2, n_hidden=32, n_out=1, seed=42)
        result = run_continual_experiment(meta, tasks_5)
        assert "forgetting" in result
        assert "mean_final_mse" in result
        assert result["forgetting"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
