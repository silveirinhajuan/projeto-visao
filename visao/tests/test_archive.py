"""Testes TDD da tarefa 3.1 — arquivo de variantes (grafo de linhagem).

Cada nó carrega (código, pesos, fitness, pai). Regra invariável da Fase 3:
NUNCA podar por fitness baixa — os becos sem saída são de onde saem os saltos
(DGM: sem arquivo aberto, o desempenho CAI). O arquivo é o "DNA" evolutivo.
"""
from __future__ import annotations

import numpy as np
import pytest

from visao.evolve.archive import Archive, PruningForbidden, Variant


def _cell_state(n=3):
    """Peso realista no formato de um estado de CfCCell (save_brain)."""
    return {
        "W_in": np.zeros((n, 1)),
        "W_rec": np.eye(n),
        "b": np.ones(n),
        "A": np.arange(n, dtype=float),
        "mask": np.ones((n, n)),
        "tau": np.linspace(0.4, 4.0, n),
    }


def test_add_root_variant_has_no_parent():
    arch = Archive()
    vid = arch.add(code="baseline", weights=_cell_state())

    assert vid in arch
    v = arch.get(vid)
    assert v.parent_id is None
    assert v.code == "baseline"
    assert np.allclose(v.weights["W_rec"], np.eye(3))


def test_child_links_to_parent():
    arch = Archive()
    root = arch.add(code="r")
    child = arch.add(code="c", parent_id=root)

    assert arch.get(child).parent_id == root
    assert arch.children(root) == [child]


def test_lineage_returns_full_root_to_node_path():
    arch = Archive()
    a = arch.add(code="a")
    b = arch.add(code="b", parent_id=a)
    c = arch.add(code="c", parent_id=b)

    assert arch.lineage(c) == [a, b, c]
    assert arch.ancestors(c) == [a, b]


def test_unknown_parent_rejected():
    arch = Archive()
    with pytest.raises(KeyError):
        arch.add(code="orphan", parent_id="v999")


def test_low_fitness_variant_is_never_pruned():
    arch = Archive()
    good = arch.add(code="good", fitness=0.9)
    dead = arch.add(code="dead_end", fitness=-999.0)

    # O arquivo guarda o beco sem saída, independente do fitness:
    assert dead in arch
    assert arch.get(good).fitness > arch.get(dead).fitness

    # E qualquer tentativa de podar por fitness é PROIBIDA por construção:
    with pytest.raises(PruningForbidden):
        arch.prune_by_fitness(threshold=0.0)

    assert len(arch) == 2


def test_set_fitness_updates_value():
    arch = Archive()
    v = arch.add(code="x")
    assert arch.get(v).fitness is None

    arch.set_fitness(v, 0.42)
    assert arch.get(v).fitness == 0.42


def test_best_selects_without_removing_others():
    arch = Archive()
    lo = arch.add(code="lo", fitness=-5.0)
    hi = arch.add(code="hi", fitness=0.95)
    mid = arch.add(code="mid", fitness=0.5)

    assert arch.best(1) == [hi]
    assert arch.best() == [hi, mid, lo]
    # selecionar NÃO é podar:
    assert len(arch) == 3
