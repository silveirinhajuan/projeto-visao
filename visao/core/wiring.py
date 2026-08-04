"""
wiring.py — Topologia Neural Circuit Policy (NCP).

Lechner et al., "Neural Circuit Policies Enabling Untrained Feedforward
Control in Animals", Nature Machine Intelligence 2020.

A NCP organiza n neurônios em quatro grupos funcionais, conectados de
forma esparsa e estritamente direcionada:

    sensory ─┬─▶ inter ─┬─▶ command ──▶ motor
             │          ├─▶ command
             │          └─▶ motor
             └─▶ command

Regras de conexão (source → target):
    sensory → inter      entrada sensorial projeta para inter
    sensory → command    atalho sensorial direto
    inter   → inter      recorrência local do grupo inter
    inter   → command
    inter   → motor
    command → motor

Bloqueios (nunca conectam):
    motor   não tem saída          (é a camada de saída)
    command não alimenta inter
    command não se auto-alimenta
    sensory não recebe recorrência (só entrada externa)
    diagonal = 0                   (sem auto-sinapse)

A esparsidade é aplicada DENTRO dos blocos permitidos: a topologia é
invariante. A plasticidade só ajusta os pesos nas sinapses já existentes;
`enforce` reaplica a máscara e mata qualquer peso que o update criou fora dela.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# pares (source, target) permitidos — a assinatura da NCP
_ALLOWED = frozenset(
    {
        ("sensory", "inter"),
        ("sensory", "command"),
        ("inter", "inter"),
        ("inter", "command"),
        ("inter", "motor"),
        ("command", "motor"),
    }
)

_GROUPS = ("sensory", "inter", "command", "motor")


@dataclass
class NCPWiring:
    n_sensory: int
    n_inter: int
    n_command: int
    n_motor: int
    sparsity: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        if not all(
            isinstance(n, int) and n >= 0
            for n in (self.n_sensory, self.n_inter, self.n_command, self.n_motor)
        ):
            raise ValueError("contagens de neurônios devem ser inteiros >= 0")
        if self.sparsity < 0 or self.sparsity >= 1:
            raise ValueError("sparsity deve estar em [0, 1)")

        self.groups = {
            "sensory": (0, self.n_sensory),
            "inter": (self.n_sensory, self.n_sensory + self.n_inter),
            "command": (
                self.n_sensory + self.n_inter,
                self.n_sensory + self.n_inter + self.n_command,
            ),
            "motor": (
                self.n_sensory + self.n_inter + self.n_command,
                self.n_sensory + self.n_inter + self.n_command + self.n_motor,
            ),
        }
        self.n_total = (
            self.n_sensory + self.n_inter + self.n_command + self.n_motor
        )
        rng = np.random.default_rng(self.seed)
        self.mask = self._build_mask(rng)

    def _build_mask(self, rng: np.random.Generator) -> np.ndarray:
        mask = np.zeros((self.n_total, self.n_total), dtype=np.float64)
        for src, dst in _ALLOWED:
            j0, j1 = self.groups[src]  # colunas (source)
            i0, i1 = self.groups[dst]  # linhas (target)
            block = rng.random((i1 - i0, j1 - j0)) > self.sparsity
            mask[i0:i1, j0:j1] = block
        np.fill_diagonal(mask, 0.0)
        return mask

    def enforce(self, weights: np.ndarray) -> np.ndarray:
        """Aplica a topologia: zera qualquer peso fora da máscara NCP."""
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != self.mask.shape:
            raise ValueError(
                f"weights {w.shape} incompatível com mask {self.mask.shape}"
            )
        return w * self.mask

    def random_weights(self, scale: float = 1.0, rng=None) -> np.ndarray:
        """Pesos aleatórios já respeitando a topologia (bloco permitido, esparsificado)."""
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        return self.enforce(rng.normal(0, scale, self.mask.shape))
