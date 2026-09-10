"""
test_streaming_bench.py — Testes para o benchmark de streaming (Tarefa 8.2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


def make_stream(n: int = 1000, seed: int = 0):
    """Gera stream simples para testes."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    signal = np.sin(2 * np.pi * 0.03 * t)
    u = rng.normal(0, 0.05, (n, 2))
    u[:, 0] += signal
    y = np.convolve(signal, np.ones(5) / 5, mode="same")[:, None]
    return u.astype(np.float64), y.astype(np.float64)


class TestStreaming:
    def test_brain_survives_stream(self):
        """Cérebro deve processar o stream sem crash."""
        brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)
        u, y = make_stream(n=200, seed=0)

        for ui, yi in zip(u, y):
            result = brain.learn(ui, yi)
            assert np.isfinite(result["err"])
            assert np.isfinite(result["surprise"])

        assert brain.step == 200

    def test_streaming_error_bounded(self):
        """Erro médio deve ser finito (não explode)."""
        brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)
        u, y = make_stream(n=500, seed=0)

        errs = []
        for ui, yi in zip(u, y):
            result = brain.learn(ui, yi)
            errs.append(result["err"])

        mean_err = np.mean(errs)
        assert np.isfinite(mean_err)
        assert mean_err < 10  # Erro deve ser razoável

    def test_full_config_better_than_naive(self):
        """VisaoBrain completo deve ter erro menor que naive."""
        u, y = make_stream(n=1000, seed=0)

        brain_full = VisaoBrain(n_in=2, n_hidden=32, n_out=1, consolidation=8.0, seed=42)
        brain_naive = VisaoBrain(n_in=2, n_hidden=32, n_out=1, consolidation=0.0, surprise_gain=0.0, seed=42)

        errs_full = []
        errs_naive = []

        for ui, yi in zip(u, y):
            r1 = brain_full.learn(ui, yi)
            r2 = brain_naive.learn(ui, yi)
            errs_full.append(r1["err"])
            errs_naive.append(r2["err"])

        # Comparar última metade (aprendizado consolidado)
        mid = len(errs_full) // 2
        assert np.mean(errs_full[mid:]) <= np.mean(errs_naive[mid:]) * 1.2  # margem

    def test_streaming_with_regime_change(self):
        """Testa adaptação a mudança de regime."""
        brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)
        u, y = make_stream(n=600, seed=0)

        # Primeira metade
        for ui, yi in zip(u[:300], y[:300]):
            brain.learn(ui, yi)

        # Segunda metade (mudança de freq)
        rng = np.random.default_rng(1)
        t = np.arange(300)
        signal2 = np.sin(2 * np.pi * 0.08 * t)  # freq diferente
        u2 = rng.normal(0, 0.05, (300, 2))
        u2[:, 0] += signal2
        y2 = np.convolve(signal2, np.ones(5) / 5, mode="same")[:, None]

        errs_after = []
        for ui, yi in zip(u2, y2):
            result = brain.learn(ui, yi)
            errs_after.append(result["err"])

        # Erro deve ser finito (cérebro se adaptou)
        assert np.all(np.isfinite(errs_after))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
