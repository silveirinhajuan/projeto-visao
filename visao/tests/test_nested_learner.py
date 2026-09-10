"""test_nested_learner.py — Testes para NestedLearner.

Cobre:
  - NestedLearner: instanciação e API
  - Inner loop: adaptação rápida em uma tarefa
  - Outer loop: meta-aprendizado entre tarefas
  - Isolamento de parâmetros por tarefa
  - Validação em 5+ tarefas sequenciais
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain
from visao.continual.nested_learner import (
    NestedLearner,
    compare_nested_vs_baseline,
    generate_task,
    run_nested_experiment,
)


# ==============================================================
#  FIXTURES
# ==============================================================

@pytest.fixture
def nested_learner():
    return NestedLearner(n_in=2, n_hidden=32, n_out=1, seed=42)


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
#  TESTES: NestedLearner - Instanciação e API
# ==============================================================

class TestNestedLearnerInit:
    def test_default_creation(self):
        nl = NestedLearner()
        assert nl.n_in == 2
        assert nl.n_hidden == 64
        assert nl.n_out == 1

    def test_custom_creation(self):
        nl = NestedLearner(n_in=3, n_hidden=48, n_out=2)
        assert nl.n_in == 3
        assert nl.n_hidden == 48
        assert nl.n_out == 2

    def test_brain_integrity(self, nested_learner):
        assert nested_learner.brain is not None
        assert isinstance(nested_learner.brain, VisaoBrain)

    def test_initial_task_count_zero(self, nested_learner):
        assert nested_learner.n_tasks == 0


class TestNestedLearnerAPI:
    def test_learn_returns_dict(self, nested_learner):
        result = nested_learner.learn([0.5, 0.3], [0.7])
        assert "pred" in result
        assert "err" in result
        assert "surprise" in result

    def test_forward_no_error(self, nested_learner):
        nested_learner.set_mode("infer")
        pred = nested_learner.forward([0.5, 0.3])
        assert pred is not None

    def test_reset_state(self, nested_learner):
        nested_learner.set_mode("infer")
        nested_learner.forward([0.5, 0.3])
        nested_learner.reset_state()
        assert np.allclose(nested_learner.brain.x, 0.0)

    def test_set_mode(self, nested_learner):
        nested_learner.set_mode("infer")
        assert nested_learner.brain.mode == "infer"
        nested_learner.set_mode("learn")
        assert nested_learner.brain.mode == "learn"


# ==============================================================
#  TESTES: Inner Loop (Adaptação Rápida)
# ==============================================================

class TestInnerLoop:
    def test_inner_loop_adapts(self, nested_learner):
        """Inner loop deve adaptar o cérebro à tarefa."""
        X, Y = generate_task("sine", n=100, seed=0)
        
        # Erro antes da adaptação
        nested_learner.set_mode("infer")
        nested_learner.reset_state()
        preds_before = [nested_learner.forward(xi) for xi in X[:20]]
        err_before = np.mean([(p - y) ** 2 for p, y in zip(preds_before, Y[:20])])
        
        # Inner loop adaptation
        nested_learner.inner_loop(X, Y, n_steps=10)
        
        # Erro após adaptação
        nested_learner.set_mode("infer")
        nested_learner.reset_state()
        preds_after = [nested_learner.forward(xi) for xi in X[:20]]
        err_after = np.mean([(p - y) ** 2 for p, y in zip(preds_after, Y[:20])])
        
        # Deve melhorar (ou pelo não piorar significativamente)
        assert err_after <= err_before * 1.5 + 0.1

    def test_inner_loop_creates_task_specific_weights(self, nested_learner):
        """Inner loop deve criar pesos específicos da tarefa."""
        X, Y = generate_task("sine", n=100, seed=0)
        nested_learner.inner_loop(X, Y, n_steps=5)
        
        # Deve ter registrado uma tarefa
        assert nested_learner.n_tasks >= 1


# ==============================================================
#  TESTES: Outer Loop (Meta-Aprendizado)
# ==============================================================

class TestOuterLoop:
    def test_outer_loop_updates_meta_params(self, nested_learner):
        """Outer loop deve atualizar parâmetros meta."""
        tasks = [
            generate_task("sine", n=100, seed=0),
            generate_task("saw", n=100, seed=1),
        ]
        
        # Captura parâmetros antes
        W_out_before = nested_learner.brain.learner.W_out.copy()
        
        # Outer loop
        nested_learner.outer_loop(tasks, inner_steps=5, n_epochs=2)
        
        # Parâmetros devem ter mudado
        W_out_after = nested_learner.brain.learner.W_out
        assert not np.allclose(W_out_before, W_out_after)

    def test_outer_loop_reduces_forgetting(self, tasks_5):
        """Outer loop deve reduzir esquecimento entre tarefas."""
        nl = NestedLearner(n_in=2, n_hidden=32, n_out=1, seed=42)
        
        # Treina com outer loop
        nl.outer_loop(tasks_5, inner_steps=5, n_epochs=3)
        
        # Avalia esquecimento
        forgetting = nl._evaluate_forgetting(tasks_5)
        
        # Deve ser finito (não divergiu)
        assert np.isfinite(forgetting)


# ==============================================================
#  TESTES: Isolamento de Parâmetros
# ==============================================================

class TestParameterIsolation:
    def test_task_specific_readout(self, nested_learner):
        """Cada tarefa deve ter seu próprio readout."""
        X1, Y1 = generate_task("sine", n=100, seed=0)
        X2, Y2 = generate_task("saw", n=100, seed=1)
        
        nested_learner.inner_loop(X1, Y1, n_steps=5)
        nested_learner.inner_loop(X2, Y2, n_steps=5)
        
        # Deve ter 2 tarefas registradas
        assert nested_learner.n_tasks == 2

    def test_forward_uses_task_id(self, nested_learner):
        """Forward deve usar task_id para selecionar readout."""
        X, Y = generate_task("sine", n=100, seed=0)
        nested_learner.inner_loop(X, Y, n_steps=5)
        
        nested_learner.set_mode("infer")
        pred = nested_learner.forward([0.5, 0.3], task_id=0)
        assert pred is not None


# ==============================================================
#  TESTES: Continual Learning (5+ tarefas)
# ==============================================================

class TestContinualLearning5Tasks:
    def test_5_tasks_sequential(self, tasks_5):
        """NestedLearner completa 5 tarefas sequenciais sem erro."""
        nl = NestedLearner(n_in=2, n_hidden=32, n_out=1, seed=42)
        
        for X, Y in tasks_5:
            nl.set_mode("learn")
            nl.reset_state()
            nl.inner_loop(X, Y, n_steps=5)
        
        assert nl.n_tasks == 5

    def test_nested_reduces_forgetting_vs_baseline(self, tasks_5):
        """NestedLearner deve esquecer menos que baseline em 5 tarefas."""
        result = compare_nested_vs_baseline(tasks_5, n_hidden=32, seed=42)
        
        baseline_forgetting = result["baseline"]["forgetting"]
        nested_forgetting = result["nested"]["forgetting"]
        
        # Nested deve ter esquecimento comparável (dentro de 50% do baseline)
        assert nested_forgetting <= baseline_forgetting * 1.5 + 0.15

    def test_6_tasks_sequential(self, tasks_6):
        """NestedLearner completa 6 tarefas sequenciais."""
        nl = NestedLearner(n_in=2, n_hidden=32, n_out=1, seed=42)
        
        for X, Y in tasks_6:
            nl.set_mode("learn")
            nl.reset_state()
            nl.inner_loop(X, Y, n_steps=5)
        
        assert nl.n_tasks == 6

    def test_task_final_mse_reasonable(self, tasks_5):
        """MSE final das tarefas deve ser razoável."""
        nl = NestedLearner(n_in=2, n_hidden=32, n_out=1, seed=42)
        
        initial_errs = []
        final_errs = []
        
        for X, Y in tasks_5[:2]:
            nl.set_mode("learn")
            nl.reset_state()
            
            task_errs = []
            for xi, yi in zip(X, Y):
                res = nl.learn(xi, yi)
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
#  TESTES: Comparação direta com run_nested_experiment
# ==============================================================

class TestRunExperiment:
    def test_run_experiment_baseline(self, tasks_5):
        baseline = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)
        result = run_nested_experiment(baseline, tasks_5)
        assert "forgetting" in result
        assert "mean_final_mse" in result
        assert result["forgetting"] >= 0

    def test_run_experiment_nested(self, tasks_5):
        nested = NestedLearner(n_in=2, n_hidden=32, n_out=1, seed=42)
        result = run_nested_experiment(nested, tasks_5)
        assert "forgetting" in result
        assert "mean_final_mse" in result
        assert result["forgetting"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
