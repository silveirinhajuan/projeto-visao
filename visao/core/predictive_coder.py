"""
predictive_coder.py — Predictive coding multicamada (tarefa 1.6).

Hierarquia valor/erro sobre o estado líquido do CfCCell. Duas camadas de mesma
dimensionalidade interna (n_mid): a baixa recebe x (n_hidden) e o projeta para
n_mid; a alta recebe o valor da baixa e emite a saida via W_out (n_out x n_mid).

A entrada do CfCCell (n_hidden) e a dimensao interna (n_mid) sao diferentes, entao
a baixa faz a ponte: W_bridge (n_mid x n_hidden) projeta x -> value da baixa.
Regras locais, sem backprop global (coerente com Fase 1).
"""

from __future__ import annotations

import numpy as np

from visao.core.predictive import PredictiveLayer


class PredictiveCoder:
    def __init__(self, n_hidden: int, n_out: int, n_mid: int = 32, lr: float = 0.005, rng=None):
        rng = rng or np.random.default_rng(0)
        self.lr = lr
        self.n_hidden = n_hidden
        self.n_mid = n_mid
        self.n_out = n_out
        self.low = PredictiveLayer(n_mid, lr=lr, rng=rng)
        self.high = PredictiveLayer(n_mid, lr=lr, rng=rng)
        # ponte da entrada do cell (n_hidden) para a dimensao interna (n_mid)
        self.W_bridge = rng.normal(0, 1.0 / np.sqrt(n_hidden), (n_mid, n_hidden)).astype(np.float64)
        self.W_bridge = np.clip(self.W_bridge, -3.0, 3.0)
        # saida final: value da alta (n_mid) -> n_out
        self.W_out = rng.normal(0, 1.0 / np.sqrt(n_mid), (n_out, n_mid)).astype(np.float64)
        self.W_out = np.clip(self.W_out, -3.0, 3.0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        xm = self.W_bridge @ x
        pred_mid = self.low.forward(xm)
        _ = self.high.forward(pred_mid)
        return self.W_out @ self.high.value

    def update(self, x: np.ndarray, target: np.ndarray) -> float:
        xm = self.W_bridge @ x
        pred_mid = self.low.forward(xm)
        _ = self.high.forward(pred_mid)
        out = self.W_out @ self.high.value
        err = float(np.mean((target - out) ** 2))
        delta = target - out
        delta_mid = self.W_out.T @ delta   # projeta erro de saida -> dim n_mid
        # alto corrige contra o erro projetado (topo)
        self.high.correct_top(delta_mid)
        self.high.update()
        # baixa recebe a predicao da alta como top_down
        self.low.forward(xm, top_down=self.high.pred_up)
        self.low.update()
        # W_out e W_bridge ajustam por gradiente local da saida
        self.W_out += self.lr * np.outer(delta, self.high.value)
        np.clip(self.W_out, -3.0, 3.0, out=self.W_out)
        self.W_bridge += self.lr * np.outer(self.low.error, x)
        np.clip(self.W_bridge, -3.0, 3.0, out=self.W_bridge)
        return err

    def params(self) -> int:
        return int(self.low.W_up.size + self.low.W_down.size +
                   self.high.W_up.size + self.high.W_down.size +
                   self.W_bridge.size + self.W_out.size)
