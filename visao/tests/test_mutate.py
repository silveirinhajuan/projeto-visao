"""Testes TDD da tarefa 3.2 — loop de mutação (permitidas/proibidas).

Permitidas: topologia de wiring, faixa de tau, hiperparâmetros de
plasticidade, código das regras locais.
Proibidas (R3): camada de governança, política de rede, chaves
criptográficas — fora do espaço de mutação. Qualquer tentativa levanta
ForbiddenMutation e NÃO entra no arquivo (3.1 guarda tudo; aqui a gente
impede que o DNA proibido se propague).
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from visao.evolve.archive import Archive
from visao.evolve.mutate import (
    FORBIDDEN_MARKERS,
    ForbiddenMutation,
    MutationLoop,
    _contains_forbidden,
    mutate_code,
    mutate_plasticity,
    mutate_tau,
)


def _cell_state(n=4):
    """Peso realista no formato de um estado de CfCCell (save_brain)."""
    return {
        "W_in": np.zeros((n, 1)),
        "W_rec": np.eye(n),
        "b": np.ones(n),
        "A": np.arange(n, dtype=float),
        "mask": np.ones((n, n)),
        "tau": np.linspace(0.4, 4.0, n),
        "oja_lr": 0.01,
        "surprise_lr": 0.05,
        "consolidation_lr": 0.02,
    }


def test_mutate_tau_altera_tau_e_preserva_outros_campos():
    r = random.Random(0)
    w = _cell_state()
    w2 = mutate_tau(w, rng=r)
    assert not np.allclose(np.asarray(w2["tau"]), np.asarray(w["tau"]))
    # demais campos inalterados (inclui as taxas de plasticidade)
    for k in ("W_in", "W_rec", "b", "A", "mask", "oja_lr", "surprise_lr", "consolidation_lr"):
        assert np.allclose(np.asarray(w2[k]), np.asarray(w[k]))
    # o original jamais é tocado
    assert np.allclose(np.asarray(w["tau"]), np.linspace(0.4, 4.0, 4))


def test_mutate_plasticity_perturba_taxas_mantendo_pesos():
    r = random.Random(1)
    w = _cell_state()
    w2 = mutate_plasticity(w, rng=r)
    for k in ("oja_lr", "surprise_lr", "consolidation_lr"):
        assert w2[k] != w[k]
    # tau e pesos estruturais ficam intactos
    assert np.allclose(np.asarray(w2["tau"]), np.asarray(w["tau"]))
    assert np.allclose(np.asarray(w2["W_rec"]), np.asarray(w["W_rec"]))


def test_mutate_code_anexa_tag_sem_tocar_pesos():
    r = random.Random(2)
    code = "def local_rule(x):\n    return x"
    new = mutate_code(code, rng=r)
    assert new.startswith(code)
    assert "# local_rule_mut:" in new
    assert new != code


def test_mutation_loop_gera_filho_com_parent_id():
    r = random.Random(3)
    arch = Archive()
    root = arch.add(code="baseline", weights=_cell_state())
    loop = MutationLoop(rng=r)
    child = loop.mutate(arch, root)
    assert arch.get(child).parent_id == root
    assert child in arch
    assert len(arch) == 2


def test_forbidden_markers_detectados_e_codigo_liquido_nao():
    for marker in FORBIDDEN_MARKERS:
        assert _contains_forbidden(f"import {marker}") is True
    # código de célula líquida comum NÃO é proibido
    assert _contains_forbidden("import visao.core.cfc") is False
    assert _contains_forbidden("def local_rule(state):\n    return state * 0.5") is False


def test_mutation_loop_rejeita_variante_com_codigo_proibido():
    r = random.Random(4)
    arch = Archive()
    # DNA que já referencia governança: não deve se propagar por mutação
    bad = arch.add(code="import visao.governance.containment", weights=_cell_state())
    loop = MutationLoop(rng=r)
    with pytest.raises(ForbiddenMutation):
        loop.mutate(arch, bad)
    # nada foi acrescentado além da própria raiz proibida
    assert len(arch) == 1
