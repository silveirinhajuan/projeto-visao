"""No worker do enxame (Fase 2.1) — treino local + troca de deltas.

Arquitetura-agnostico: parametros sao um dict numpy (ex.: {"w","b"}). Hoje
usamos regressao linear como prova vertical; o CfCCell pluga depois como
qualquer dict de pesos. Protocolo: conexao persistente por worker; cada round
envia o delta local e recebe o modelo global agregado pelo coordenador.
"""
from __future__ import annotations

import pickle
import socket
import struct

import numpy as np

HEADER = struct.Struct("!I")


def _send(f, payload: bytes) -> None:
    f.write(HEADER.pack(len(payload)))
    f.write(payload)
    f.flush()


def _recv(f) -> bytes:
    size = HEADER.unpack(f.read(4))[0]
    return f.read(size)


def _grad(params: dict, X: np.ndarray, Y: np.ndarray) -> dict:
    w = params["w"]
    b = params["b"]
    n = X.shape[0]
    err = (X @ w + b) - Y
    return {
        "w": (2.0 / n) * (X.T @ err),
        "b": (2.0 / n) * err.sum(axis=0, keepdims=True),
    }


def train_single_node(params: dict, X: np.ndarray, Y: np.ndarray,
                      lr: float, steps: int) -> dict:
    """Referencia single-node: full-batch GD nos dados COMPLETOS."""
    p = {k: v.copy() for k, v in params.items()}
    for _ in range(steps):
        g = _grad(p, X, Y)
        p["w"] -= lr * g["w"]
        p["b"] -= lr * g["b"]
    return p


def _local_steps(params: dict, X: np.ndarray, Y: np.ndarray,
                 lr: float, K: int) -> dict:
    p = {k: v.copy() for k, v in params.items()}
    for _ in range(K):
        g = _grad(p, X, Y)
        p["w"] -= lr * g["w"]
        p["b"] -= lr * g["b"]
    return p


def worker_process(addr, data, lr: float, K: int, n_rounds: int,
                   init: dict, worker_id: int) -> None:
    X, Y = data
    current = {k: v.copy() for k, v in init.items()}
    s = socket.create_connection(addr)
    f = s.makefile("rwb")
    try:
        for _ in range(n_rounds):
            updated = _local_steps(current, X, Y, lr, K)
            delta = {k: updated[k] - current[k] for k in current}
            _send(f, pickle.dumps(delta))
            current = pickle.loads(_recv(f))
    finally:
        f.close()
        s.close()
