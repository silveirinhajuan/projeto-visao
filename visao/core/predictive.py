"""
predictive.py — Predictive coding multicamada (tarefa 1.6).

Camada valor/erro (Whittington & Bogacz 2017, simplificado para dimensoes
congruentes). Cada camada tem n_in == n_up == n_down: a camada preve a
atividade da camada vizinha (valor/erro simetricos). Isso garante consistencia
de broadcast: W_up (n_in x n_in) e o erro da camada (n_in,) casam no update.

A saida final fica num W_out externo ao PredictiveCoder (projeta o value topo
para n_out). Regras locais, sem backprop global (coerente com Fase 1).
"""

from __future__ import annotations

import numpy as np


class PredictiveLayer:
    def __init__(self, n: int, lr: float = 0.005, rng=None):
        rng = rng or np.random.default_rng(0)
        self.n = n
        self.lr = lr
        # W_up projeta value (n,) -> predicao para a camada acima (n,)
        self.W_up = rng.normal(0, 1.0 / np.sqrt(n), (n, n)).astype(np.float64)
        # W_down recebe a predicao da camada acima (n,) -> correcao top-down
        self.W_down = rng.normal(0, 1.0 / np.sqrt(n), (n, n)).astype(np.float64)
        self.value = np.zeros(n)
        self.pred_up = np.zeros(n)
        self.error = np.zeros(n)

    def forward(self, bottom_up, top_down=None):
        """bottom_up (n,): entrada de baixo. top_down (n,) ou None.
        Retorna pred_up (n,)."""
        if top_down is not None:
            td = self.W_down @ top_down
        else:
            td = 0.0
        self.error = bottom_up - td
        self.value = np.clip(self.value + self.lr * self.error, -2.0, 2.0)
        self.pred_up = self.W_up @ self.value
        return self.pred_up

    def correct_top(self, delta):
        """Camada TOPO: recebe erro de saida (target - predicao) ja calculado
        externamente (dim n,). Ajusta value/W_up por ele."""
        self.error = delta
        self.value = np.clip(self.value + self.lr * delta, -2.0, 2.0)

    def update(self):
        self.W_up += self.lr * np.outer(self.error, self.value)
        self.W_down += self.lr * np.outer(self.value, self.error)
        np.clip(self.W_up, -3.0, 3.0, out=self.W_up)
        np.clip(self.W_down, -3.0, 3.0, out=self.W_down)
