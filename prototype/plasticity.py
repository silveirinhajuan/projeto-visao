"""
plasticity.py — Regras de aprendizado LOCAIS (sem backpropagation).

Por que sem backprop: backprop exige um passe global reverso, guarda o grafo
inteiro de ativações e é a razão estrutural do esquecimento catastrófico em
aprendizado contínuo. Regras locais só usam o que está fisicamente disponível
na sinapse (pré-sináptico, pós-sináptico, erro local, neuromodulador).

Referências:
  Oja (1982) — Hebbian normalizado, estável
  Whittington & Bogacz (2017) — predictive coding com plasticidade Hebbiana local
  Hinton (2022) — Forward-Forward
  Kirkpatrick et al. (2017) — consolidação sináptica elástica (EWC)

Aqui usamos três mecanismos que se combinam:
  1. Readout com erro local modulado (delta rule = predictive coding de 1 camada)
  2. Oja no recorrente (auto-organização não supervisionada do reservatório)
  3. Consolidação por importância + gate de surpresa (anti-esquecimento)
"""

from __future__ import annotations

import numpy as np


class LocalLearner:
    """Aprendizado local para a saída do reservatório líquido.

    O readout é linear sobre o estado líquido. O que impede o esquecimento
    catastrófico não é o readout — é o par (importância sináptica, gate de surpresa):

      - importância: média móvel de |erro * pré-sináptico|. Sinapse importante
        para tarefas passadas fica RÍGIDA (lr efetivo dividido por 1+omega).
      - surpresa: se o erro está dentro do esperado, aprende pouco. Se explode
        (mudança de distribuição), abre o portão e aprende rápido.
        Isso imita neuromodulação (dopamina/acetilcolina) e é o que dá
        estabilidade-plasticidade sem replay de dados antigos.
    """

    def __init__(
        self,
        n_hidden: int,
        n_out: int,
        lr: float = 0.05,
        oja_lr: float = 0.002,
        consolidation: float = 1.0,
        surprise_gain: float = 3.0,
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng(0)
        self.W_out = rng.normal(0, 0.1, (n_out, n_hidden))
        self.b_out = np.zeros(n_out)
        self.lr = lr
        self.oja_lr = oja_lr
        self.consolidation = consolidation
        self.surprise_gain = surprise_gain

        self.omega = np.zeros((n_out, n_hidden))  # importância sináptica
        self.err_ema = 1.0                        # média móvel do erro (baseline)
        self.err_var = 1.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.W_out @ x + self.b_out

    def surprise(self, err_mag: float) -> float:
        """Quão inesperado é este erro? ~1.0 = esperado, >1 = mudou o mundo."""
        z = (err_mag - self.err_ema) / (np.sqrt(self.err_var) + 1e-8)
        return float(1.0 + self.surprise_gain * max(0.0, np.tanh(z)))

    def update(self, x: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        """Um passo de aprendizado local. Retorna (erro_quadratico, surpresa)."""
        pred = self.predict(x)
        err = target - pred
        err_mag = float(np.abs(err).mean())

        s = self.surprise(err_mag)

        # lr efetivo: aberto pela surpresa, fechado pela importância acumulada
        eff = (self.lr * s) / (1.0 + self.consolidation * self.omega)

        delta = np.outer(err, x)          # regra delta = puramente local
        self.W_out += eff * delta
        self.b_out += self.lr * s * err

        # importância cresce onde a sinapse fez trabalho útil
        self.omega += 0.01 * np.abs(delta)

        # baselines de surpresa (Welford simplificado)
        d = err_mag - self.err_ema
        self.err_ema += 0.02 * d
        self.err_var += 0.02 * (d * d - self.err_var)

        return float((err ** 2).mean()), s

    def oja_update(self, cell, x_pre: np.ndarray, x_post: np.ndarray) -> None:
        """Oja no recorrente: Hebb com decaimento normalizador.

            dW_ij = eta * y_i * (x_j - y_i * W_ij)

        Mantém o reservatório auto-organizado e com norma estável, sem
        supervisão nenhuma. É o componente "não supervisionado" da plasticidade.
        """
        dW = np.outer(x_post, x_pre) - (x_post ** 2)[:, None] * cell.W_rec
        cell.W_rec += self.oja_lr * dW * cell.mask
        np.clip(cell.W_rec, -2.0, 2.0, out=cell.W_rec)

    # ------------------------------------------------------------------ estado
    def state(self) -> dict:
        return {"W_out": self.W_out.copy(), "b_out": self.b_out.copy(), "omega": self.omega.copy()}

    def load(self, st: dict) -> None:
        self.W_out = st["W_out"].copy()
        self.b_out = st["b_out"].copy()
        self.omega = st["omega"].copy()


class NaiveLearner(LocalLearner):
    """Baseline de controle: mesma regra delta, SEM consolidação e SEM surpresa.

    Existe para provar que o ganho vem dos mecanismos, não do reservatório.
    """

    def __init__(self, n_hidden: int, n_out: int, lr: float = 0.05, rng=None):
        super().__init__(n_hidden, n_out, lr=lr, oja_lr=0.0,
                         consolidation=0.0, surprise_gain=0.0, rng=rng)
