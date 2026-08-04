"""
test_predictive.py — Tarefa 1.6: predictive coding multicamada aprende localmente.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from visao.core.cfc import CfCCell                       # noqa: E402
from visao.core.predictive_coder import PredictiveCoder  # noqa: E402


def _make_io(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 32))
    # alvo linear simples em 4 classes
    logits = x @ rng.normal(0, 1, (32, 4))
    y = np.argmax(logits, axis=1)
    return x, y


class TestPredictiveCoder:
    def test_erro_cai_com_treino(self):
        x, y = _make_io()
        n_hidden = 48
        cell = CfCCell(n_in=32, n_hidden=n_hidden, dt=0.1, seed=1)
        coder = PredictiveCoder(n_hidden, 4, n_mid=24, lr=0.001, rng=np.random.default_rng(2))

        states = np.array([np.asarray(cell.rollout(xx[None, :])[0][-1]) for xx in x])
        states = states / (np.std(states, axis=0, keepdims=True) + 1e-8)  # normaliza

        err0 = np.mean([coder.update(s, np.eye(4)[yy]) for s, yy in zip(states, y)])
        for _ in range(15):
            for s, yy in zip(states, y):
                coder.update(s, np.eye(4)[yy])
        err1 = np.mean([coder.update(s, np.eye(4)[yy]) for s, yy in zip(states, y)])
        assert np.isfinite(err1), f"erro divergiu: {err1}"
        assert err1 < err0, f"erro não caiu: {err0:.4f} -> {err1:.4f}"

    def test_inference_retorna_logits(self):
        cell = CfCCell(n_in=8, n_hidden=24, dt=0.1, seed=3)
        coder = PredictiveCoder(24, 3, n_mid=12, rng=np.random.default_rng(4))
        x = np.zeros(8)
        out = coder.predict(np.asarray(cell.rollout(x[None, :])[0][-1]))
        assert out.shape == (3,)
        assert np.all(np.isfinite(out))

    def test_parametros_contaveis(self):
        n_h, n_mid, n_out = 64, 32, 10
        coder = PredictiveCoder(n_h, n_out, n_mid=n_mid, rng=np.random.default_rng(5))
        p = coder.params()
        assert p > 0
        # 2 camadas (W_up+W_down cada) + W_bridge(n_mid*n_h) + W_out(n_out*n_mid)
        esperado = 4 * (n_mid * n_mid) + (n_mid * n_h) + (n_out * n_mid)
        assert p == esperado
