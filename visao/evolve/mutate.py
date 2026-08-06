"""mutate.py — loop de mutação (tarefa 3.2, FASE3_EVOLUCAO.md §2).

Permitidas: topologia de wiring, faixa de tau, hiperparâmetros de
plasticidade, código das regras locais.
Proibidas (R3): camada de governança, política de rede, chaves
criptográficas — FORA do espaço de mutação. Qualquer código que referencie
essas zonas levanta ``ForbiddenMutation`` e NUNCA entra no arquivo.

O ``containment.py`` (visao/governance) é imutável e verificado à parte
(tarefa 4.1, 39 testes). Aqui a gente só garante que o loop de mutação em
si jamais produz nem propaga DNA proibido.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from visao.evolve.archive import Archive

# Marcadores: se aparecerem no código de uma variante, ela está fora do
# espaço de mutação permitido (governança / rede / chaves).
FORBIDDEN_MARKERS = (
    "visao.governance",
    "visao/governance",
    "network.policy",
    "containment",
    "chaves_cripto",
    "PRIVATE_KEY",
)

# Operadores de mutação que o loop pode aplicar: todos em zona permitida.
ALLOWED_KINDS = ("tau", "plasticity", "code")


class ForbiddenMutation(ValueError):
    """Mutação em zona proibida (governança/rede/chaves, R3)."""


def _contains_forbidden(code: str) -> bool:
    """True se ``code`` referencia alguma zona proibida (R3)."""
    low = code.lower()
    return any(marker.lower() in low for marker in FORBIDDEN_MARKERS)


def mutate_tau(weights: dict, *, rng, scale: float = 0.1) -> dict:
    """Perturba a faixa de tau (constante de tempo) — zona permitida."""
    new = dict(weights)
    tau = np.asarray(weights.get("tau", 1.0), dtype=float)
    factor = 1.0 + scale * (rng.random() * 2 - 1)
    new["tau"] = np.maximum(1e-3, tau * factor)
    return new


def mutate_plasticity(weights: dict, *, rng, scale: float = 0.1) -> dict:
    """Perturba taxas de plasticidade local — zona permitida."""
    new = dict(weights)
    for key in ("oja_lr", "surprise_lr", "consolidation_lr"):
        if key in weights:
            val = float(weights[key])
            factor = 1.0 + scale * (rng.random() * 2 - 1)
            new[key] = max(0.0, val * factor)
    return new


def mutate_code(code: str, *, rng) -> str:
    """Aplica micro-mutação no código de regra local — zona permitida.

    Por ora anexa uma tag determinística por rng (varia o hash da variante
    sem alterar o comportamento), mantendo o código funcional. A porta de
    substituição por mutação estrutural de regra fica para 3.3/3.4.
    """
    tag = f"# local_rule_mut:{rng.randint(0, 1_000_000_000)}"
    return code + "\n" + tag


class MutationLoop:
    """Loop de mutação: gera filhos em zona permitida; bloqueia zona proibida."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def _propose(self, parent, kind: str):
        weights = dict(parent.weights) if parent.weights else {}
        if kind == "tau":
            return parent.code, mutate_tau(weights, rng=self.rng)
        if kind == "plasticity":
            return parent.code, mutate_plasticity(weights, rng=self.rng)
        if kind == "code":
            return mutate_code(parent.code, rng=self.rng), weights
        raise ValueError(f"kind de mutação desconhecido: {kind!r}")

    def mutate(self, archive: Archive, parent_id: str, kind: Optional[str] = None) -> str:
        """Cria um filho mutado de ``parent_id`` e o registra no arquivo.

        Levanta ``ForbiddenMutation`` se o DNA resultante (ou o pai) tocar
        zona proibida — nesse caso NADA é adicionado ao arquivo.
        """
        parent = archive.get(parent_id)
        if _contains_forbidden(parent.code):
            raise ForbiddenMutation(
                "Variante pai já referencia zona proibida (R3); mutação bloqueada."
            )
        if kind is None:
            kind = self.rng.choice(ALLOWED_KINDS)
        code, weights = self._propose(parent, kind)
        if _contains_forbidden(code):
            raise ForbiddenMutation(
                f"Mutacao '{kind}' produziria DNA proibido (R3); bloqueada."
            )
        return archive.add(code, weights, parent_id=parent_id)
