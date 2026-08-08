"""TDD — Fase 2.1: nó local dois-processos via socket local.

Prova vertical: dois processos-worker treinam em shards distintos e trocam
deltas de gradiente por socket local. O modelo distribuído deve convergir
para o MESMO estado (tolerância 5%) que o treino single-node nos dados
completos. Sem isto, "enxame" é só uma palavra bonita.
"""
from __future__ import annotations

import multiprocessing as mp
import numpy as np
import pytest

from visao.swarm.coordinator import run_coordinator
from visao.swarm.node import train_single_node, worker_process
from visao.swarm.transport import free_port

RNG = np.random.default_rng(0)
D = 4
X = RNG.standard_normal((200, D))
W_TRUE = RNG.standard_normal((D, 1))
Y = X @ W_TRUE + 0.01 * RNG.standard_normal((200, 1))
INIT = {"w": np.zeros((D, 1)), "b": np.zeros((1, 1))}


@pytest.mark.parametrize("lr,K,n_rounds", [(0.01, 5, 20)])
def test_two_node_local_converges_within_5pct_of_single(lr, K, n_rounds):
    # Single-node de referência: treino full-batch nos dados COMPLETOS.
    w_single = train_single_node(INIT, X, Y, lr, K * n_rounds)

    # Distribuído: 2 processos-worker em shards IGUAIS (FedAvg == single p/ igual).
    X1, y1 = X[:100], Y[:100]
    X2, y2 = X[100:], Y[100:]
    addr = ("127.0.0.1", free_port())
    ctx = mp.get_context("fork")
    result = ctx.Queue()

    coord = run_coordinator(addr, n_workers=2, n_rounds=n_rounds, result=result)
    coord.start()
    coord.ready.wait()

    w1 = ctx.Process(target=worker_process, args=(addr, (X1, y1), lr, K, n_rounds, INIT, 1))
    w2 = ctx.Process(target=worker_process, args=(addr, (X2, y2), lr, K, n_rounds, INIT, 2))
    w1.start(); w2.start()
    w1.join(); w2.join(); coord.join()

    global_dist = result.get()
    denom = np.linalg.norm(np.concatenate([w_single["w"].ravel(), w_single["b"].ravel()]))
    num = np.linalg.norm(np.concatenate([global_dist["w"].ravel() - w_single["w"].ravel(),
                                         global_dist["b"].ravel() - w_single["b"].ravel()]))
    rel = num / denom
    assert rel < 0.05, f"modelo distribuido divergiu {rel:.3f} do single-node (>5%)"
