"""
liquid.py — Célula Liquid Time-Constant (LTC) / Closed-form Continuous-time (CfC)
em numpy puro.

Base teórica:
  Hasani et al., "Liquid Time-constant Networks" (AAAI 2021)
  Hasani et al., "Closed-form Continuous-time Neural Networks" (Nature MI, 2022)

A ideia central: o "time constant" de cada neurônio NÃO é fixo. Ele depende da
entrada atual. Isso é o análogo computacional da neuroplasticidade de curto prazo:
a dinâmica do neurônio muda em função do que ele está vendo AGORA, sem precisar
reescrever pesos.

    dx/dt = -[1/tau + f(x, I)] * x + f(x, I) * A

    tau_efetivo(t) = tau / (1 + tau * f(x, I))   <- varia por neurônio, por passo

Solver fundido (fused explicit/implicit Euler), estável para dt grande:

    x_{t+1} = (x_t + dt * f * A) / (1 + dt * (1/tau + f))
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


class LiquidCell:
    """Célula LTC com constantes de tempo dependentes da entrada.

    Parameters
    ----------
    n_in : dimensão da entrada
    n_hidden : número de neurônios líquidos
    sparsity : fração de sinapses recorrentes zeradas (esparsidade tipo NCP)
    tau_min, tau_max : faixa das constantes de tempo base (heterogeneidade)
    dt : passo de integração
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        sparsity: float = 0.5,
        tau_min: float = 0.4,
        tau_max: float = 4.0,
        dt: float = 0.1,
        rng: np.random.Generator | None = None,
    ):
        self.rng = rng or np.random.default_rng(0)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.dt = dt

        scale_in = 1.0 / np.sqrt(max(n_in, 1))
        scale_rec = 1.0 / np.sqrt(max(n_hidden, 1))

        self.W_in = self.rng.normal(0, scale_in, (n_hidden, n_in))
        self.W_rec = self.rng.normal(0, scale_rec, (n_hidden, n_hidden))
        self.b = np.zeros(n_hidden)

        # Máscara de esparsidade fixa (topologia). Neuroplasticidade age nos
        # pesos, não cria sinapses do nada — igual ao cérebro adulto.
        mask = self.rng.random((n_hidden, n_hidden)) > sparsity
        np.fill_diagonal(mask, False)  # sem auto-sinapse: o vazamento já faz isso
        self.mask = mask.astype(np.float64)
        self.W_rec *= self.mask

        # tau heterogêneo: neurônios rápidos capturam transientes,
        # lentos carregam contexto. Isso é o que dá memória multiescala.
        self.tau = np.exp(self.rng.uniform(np.log(tau_min), np.log(tau_max), n_hidden))

        # A = potencial de reversão sináptico (para onde o neurônio é puxado)
        self.A = self.rng.normal(0, 1.0, n_hidden)

    # ---------------------------------------------------------------- dinâmica
    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Não-linearidade sináptica -> vira a taxa de acoplamento."""
        return sigmoid(self.W_in @ u + self.W_rec @ x + self.b)

    def step(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Um passo do solver fundido. Retorna (novo_estado, f_ativacao)."""
        fx = self.f(x, u)
        num = x + self.dt * fx * self.A
        den = 1.0 + self.dt * (1.0 / self.tau + fx)
        return num / den, fx

    def tau_effective(self, fx: np.ndarray) -> np.ndarray:
        """Constante de tempo instantânea — a métrica de 'liquidez'."""
        return self.tau / (1.0 + self.tau * fx)

    def rollout(self, seq: np.ndarray, x0: np.ndarray | None = None):
        """Roda uma sequência (T, n_in). Retorna estados (T, n_hidden) e f (T, n_hidden)."""
        x = np.zeros(self.n_hidden) if x0 is None else x0.copy()
        states = np.empty((len(seq), self.n_hidden))
        acts = np.empty((len(seq), self.n_hidden))
        for t, u in enumerate(seq):
            x, fx = self.step(x, u)
            states[t] = x
            acts[t] = fx
        return states, acts

    # -------------------------------------------------------------- utilidades
    def params(self) -> dict[str, np.ndarray]:
        return {"W_in": self.W_in, "W_rec": self.W_rec, "b": self.b, "A": self.A}

    def flat(self) -> np.ndarray:
        return np.concatenate([p.ravel() for p in self.params().values()])

    def load_flat(self, vec: np.ndarray) -> None:
        i = 0
        for p in self.params().values():
            n = p.size
            p[...] = vec[i : i + n].reshape(p.shape)
            i += n
        self.W_rec *= self.mask  # topologia é invariante

    def clone(self) -> "LiquidCell":
        import copy

        return copy.deepcopy(self)
