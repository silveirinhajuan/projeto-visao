"""
explain.py — interpretabilidade ao vivo do reservatório líquido (tarefa 5.9).

Dado um estado, explica quais neurônios / τ responderam e POR QUÊ.

A "base de atribuição de importância" é a tarefa 1.8 (estabilidade): o
jacobiano local da dinâmica (jacobian_step, de visao.analysis.stability)
mede quanto uma perturbação em cada neurônio propaga para o próximo
estado. Reutilizar esse jacobiano é exatamente o que o BACKLOG 5.9 pede
("Depende da estabilidade medida em 1.8 [...] como base de atribuição de
importância").

API
---
jacobian_importance(cell, x, u)
    Por neurônio: norma-L2 da coluna do jacobiano local = sensibilidade da
    dinâmica àquele neurônio. (Rede sub-caótica => valores < ~1 por coluna.)

explain_state(cell, x, u, top_k=None)
    Métricas por neurônio (ativação f, τ efetivo, importância jacobiana) e
    ranking dos que mais "responderam" (|ativação| decrescente).

explain_decision(cell, x, u, readout_W, readout_b=None, top_k=None)
    Dado um readout linear (W_out, b_out), explica a decisão tomada no
    estado x: saída prevista, índice escolhido (argmax) e quais neurônios
    mais contribuíram para essa saída — com seu τ efetivo.
"""

from __future__ import annotations

import numpy as np

from visao.analysis.stability import jacobian_step


def jacobian_importance(cell, x, u) -> np.ndarray:
    """Importância por neurônio = norma-L2 da coluna do jacobiano local (tarefa 1.8).

    ||J[:, i]|| responde: "se eu perturbar o neurônio i, quanto o próximo
    estado inteiro se mexe?". É a métrica de atribuição de importância que
    a estabilidade mediu.
    """
    J = jacobian_step(cell, x, u)
    return np.linalg.norm(np.asarray(J), axis=0)


def explain_state(cell, x, u, top_k: int | None = None) -> dict:
    """Métricas por neurônio num estado, ranqueadas por quem mais respondeu."""
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)

    f = np.asarray(cell.f(x, u))
    tau_eff = np.asarray(cell.tau_effective(f))
    imp = jacobian_importance(cell, x, u)

    order = np.argsort(-np.abs(f))          # quem "respondeu" (ativação)
    top_idx = order if top_k is None else order[:top_k]
    top_neurons = [
        {
            "index": int(i),
            "activation": float(f[i]),
            "effective_tau": float(tau_eff[i]),
            "jacobian_importance": float(imp[i]),
        }
        for i in top_idx
    ]

    return {
        "activation": f,
        "effective_tau": tau_eff,
        "jacobian_importance": imp,
        "top_neurons": top_neurons,
    }


def explain_decision(cell, x, u, readout_W, readout_b=None,
                     top_k: int | None = None) -> dict:
    """Explica a decisão (readout linear) tomada no estado x.

    Retorna a saída prevista, o índice escolhido (argmax) e os neurônios
    que mais contribuíram para ESSA saída, com seu τ efetivo. A soma das
    contribuições top reproduz (previsto - viés) da saída escolhida.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    W = np.asarray(readout_W, dtype=float)
    b = np.zeros(W.shape[0]) if readout_b is None else np.asarray(readout_b, dtype=float)

    f = np.asarray(cell.f(x, u))
    tau_eff = np.asarray(cell.tau_effective(f))

    y = W @ x + b
    chosen = int(np.argmax(y))

    contrib = W * x[None, :]               # (n_out, n_hidden): W[o,i]*x[i]
    c = contrib[chosen]

    order = np.argsort(-np.abs(c))
    top_idx = order if top_k is None else order[:top_k]
    top_neurons = [
        {
            "index": int(i),
            "contribution": float(c[i]),
            "activation": float(x[i]),
            "effective_tau": float(tau_eff[i]),
        }
        for i in top_idx
    ]

    return {
        "predicted": y,
        "chosen_output": chosen,
        "top_neurons": top_neurons,
    }
