"""
test_task_free.py — Testes para task-free learning (Tarefa 8.5).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


def make_regime_stream(n: int = 1200, seed: int = 0):
    """Gera stream com mudanças de regime."""
    rng = np.random.default_rng(seed)
    u = np.zeros((n, 2))
    y = np.zeros((n, 1))

    regimes = [
        ("sine_slow", 0, 400, 0.02),
        ("sine_fast", 400, 800, 0.08),
        ("saw", 800, 1200, 0.0),
    ]

    for kind, start, end, freq in regimes:
        length = end - start
        tt = np.arange(length)
        if kind == "sine_slow":
            signal = np.sin(2 * np.pi * freq * tt)
        elif kind == "sine_fast":
            signal = np.sin(2 * np.pi * freq * tt)
        else:
            signal = 2.0 * (tt % 30) / 30.0 - 1.0

        u[start:end, 0] += signal
        y[start:end, 0] = np.convolve(signal, np.ones(5) / 5, mode="same")[:length]

    u += rng.normal(0, 0.03, u.shape)
    return u.astype(np.float64), y.astype(np.float64)


class TestTaskFree:
    def test_brain_adapts_to_regime_change(self):
        """Cérebro deve se adaptar a mudanças de regime."""
        brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)
        u, y = make_regime_stream(n=1200, seed=0)

        errs = []
        for ui, yi in zip(u, y):
            result = brain.learn(ui, yi)
            errs.append(result["err"])

        # Erro deve ser finito em todos os regimes
        assert np.all(np.isfinite(errs))

        # Erro na última metade deve ser razoável (não explode)
        assert np.mean(errs[600:]) < 5.0

    def test_surprise_triggers_adaptation(self):
        """Surpresa alta deve indicar mudança de regime."""
        brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)

        # Primeiro: aprender padrão A
        rng = np.random.default_rng(0)
        u_a = rng.normal(0, 0.1, (100, 2))
        y_a = np.sin(u_a[:, 0])[:, None]
        for ui, yi in zip(u_a, y_a):
            brain.learn(ui, yi)

        # Segundo: padrão B muito diferente
        u_b = rng.normal(5, 0.1, (100, 2))  # mudança drástica
        y_b = np.cos(u_b[:, 0])[:, None]

        surprises = []
        for ui, yi in zip(u_b, y_b):
            result = brain.learn(ui, yi)
            surprises.append(result["surprise"])

        # Surpresa deve ser alta no início do padrão B
        assert np.mean(surprises[:20]) > np.mean(surprises[80:])

    def test_brain_diverges_without_plasticity(self):
        """Cérebro sem plasticidade deve ter mais dificuldade."""
        brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, consolidation=0.0, surprise_gain=0.0, seed=42)
        u, y = make_regime_stream(n=1200, seed=0)

        errs = []
        for ui, yi in zip(u, y):
            result = brain.learn(ui, yi)
            errs.append(result["err"])

        # Sem plasticidade, erro pode ser maior, mas não deve explodir
        assert np.all(np.isfinite(errs))

    def test_task_free_with_meta_learn(self):
        """Meta-learning deve ajudar em task-free."""
        brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, meta_learn=True, seed=42)
        u, y = make_regime_stream(n=1200, seed=0)

        errs = []
        for ui, yi in zip(u, y):
            result = brain.learn(ui, yi)
            errs.append(result["err"])

        # lr deve ter sido ajustado
        assert brain.lr != brain.lr_base or brain._err_trend != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
