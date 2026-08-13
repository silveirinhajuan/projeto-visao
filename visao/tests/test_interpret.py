"""
test_interpret.py — Tarefa 5.9 do backlog: interpretabilidade ao vivo.

Dado um estado do reservatório líquido, explicar quais neurônios / τ
responderam e POR QUÊ. A base de atribuição de importância vem da tarefa
1.8 (estabilidade): o jacobiano local da dinâmica (jacobian_step) mede
quanto uma perturbação em cada neurônio propaga para o próximo estado.

Critério de portão (BACKLOG 5.9): explicar 1 decisão real. Aqui
demonstramos a capacidade sobre uma decisão do reservatório (readout
treinado por mínimos quadrados sobre tarefa de rotina sintética).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

jax = pytest.importorskip("jax", reason="tarefa 1.1 instala jax[cpu]")
jnp = pytest.importorskip("jax.numpy")

from visao.core.cfc import CfCCell  # noqa: E402
from visao.analysis.stability import jacobian_step  # noqa: E402
from visao.interpret.explain import (  # noqa: E402
    explain_decision,
    explain_state,
    jacobian_importance,
)


def _cell(n_hidden=16, seed=0):
    return CfCCell(n_in=2, n_hidden=n_hidden, dt=0.1, seed=seed)


def test_explain_state_retorna_metricas_e_ordena_por_ativacao():
    """Para cada neurônio: ativação, τ efetivo e importância jacobiana;
    top_neurons ordenado por |ativação| decrescente (quem 'respondeu')."""
    cell = _cell(16)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.5, 16)
    u = rng.normal(0, 1.0, 2)

    exp = explain_state(cell, x, u)

    f = np.asarray(cell.f(x, u))
    tau_eff = np.asarray(cell.tau_effective(f))
    assert np.allclose(exp["effective_tau"], tau_eff, atol=1e-6)
    assert exp["activation"].shape == (16,)
    assert exp["jacobian_importance"].shape == (16,)
    assert len(exp["top_neurons"]) == 16

    acts = [n["activation"] for n in exp["top_neurons"]]
    assert acts == sorted(acts, key=abs, reverse=True)
    # cada entrada traz τ efetivo e importância
    for n in exp["top_neurons"]:
        assert set(n.keys()) >= {"index", "activation", "effective_tau", "jacobian_importance"}


def test_jacobian_importance_reusa_base_da_tarefa_1_8():
    """Importância = norma-L2 da coluna do jacobiano local (estabilidade 1.8)."""
    cell = _cell(16)
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.5, 16)
    u = rng.normal(0, 1.0, 2)

    imp = jacobian_importance(cell, x, u)
    J = jacobian_step(cell, x, u)
    esperado = np.linalg.norm(J, axis=0)

    assert imp.shape == (16,)
    assert np.allclose(imp, esperado, atol=1e-10)


def test_explain_decision_atribui_contribuicao_a_saida_escolhida():
    """A soma das contribuições top reproduz a saída escolhida (argmax)."""
    cell = _cell(16)
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.5, 16)
    u = rng.normal(0, 1.0, 2)
    W = rng.normal(0, 1.0, (3, 16))   # 3 saídas (readout)
    b = rng.normal(0, 0.1, 3)

    exp = explain_decision(cell, x, u, W, b)

    y = W @ x + b
    chosen = int(np.argmax(y))
    assert exp["chosen_output"] == chosen
    assert np.allclose(exp["predicted"], y, atol=1e-6)

    contribs = np.array([n["contribution"] for n in exp["top_neurons"]])
    assert np.isclose(contribs.sum(), y[chosen] - b[chosen], atol=1e-6)


def test_portao_explica_decisao_real_do_reservatorio():
    """Portão 5.9: treina um readout e explica a decisão que o reservatório
    tomou, atribuindo-a a neurônios concretos com τ efetivo medido."""
    cell = _cell(24, seed=5)
    rng = np.random.default_rng(7)
    T = 200
    seq = rng.normal(0, 1.0, (T, 2))

    # alvo sintético de 'rotina': combinação linear estável dos estados
    states, _ = cell.rollout(seq)
    W_true = rng.normal(0, 1.0, (1, 24))
    y = states @ W_true.T + 0.01 * rng.normal(0, 1, (T, 1))

    # readout por mínimos quadrados (igual LocalLearner, porém direto)
    W_out, _, _, _ = np.linalg.lstsq(states, y, rcond=None)
    W_out = W_out.T  # (1, n_hidden)

    # decisão = último instante
    x = np.asarray(states[-1])
    u = seq[-1]
    exp = explain_decision(cell, x, u, W_out, top_k=5)

    assert exp["chosen_output"] == 0
    top = exp["top_neurons"]
    assert len(top) == 5
    # os neurônios que dirigiram a decisão têm τ efetivo finito e não-nulo
    taus = [n["effective_tau"] for n in top]
    assert all(np.isfinite(t) and t > 0 for t in taus)
    # contribuição dominante é a de maior |peso*estado|
    contribs = [abs(n["contribution"]) for n in top]
    assert contribs == sorted(contribs, reverse=True)
