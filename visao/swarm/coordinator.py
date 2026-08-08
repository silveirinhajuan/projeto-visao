"""Coordenador FedAvg local (Fase 2.1) — mediacao por socket TCP.

Mantem uma conexao persistente por worker. Cada round: recebe o delta de
cada worker, faz a MEDIA (FedAvg; shards iguais == single-node para K=1),
aplica ao modelo global e devolve o modelo agregado. Ao fim, coloca o modelo
global na fila de resultado.
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
import socket
import struct
from typing import Optional

import numpy as np

from visao.swarm.node import HEADER, _recv, _send

HEADER = struct.Struct("!I")


class Coordinator(mp.Process):
    def __init__(self, addr, n_workers: int, n_rounds: int, result):
        super().__init__()
        self.addr = addr
        self.n_workers = n_workers
        self.n_rounds = n_rounds
        self.result = result
        self.ready = mp.Event()

    def run(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(self.addr)
        srv.listen(self.n_workers)
        self.ready.set()

        conns = [srv.accept()[0] for _ in range(self.n_workers)]
        fs = [c.makefile("rwb") for c in conns]
        global_params: Optional[dict] = None
        try:
            for _ in range(self.n_rounds):
                deltas = [pickle.loads(_recv(fs[i])) for i in range(self.n_workers)]
                avg = {k: sum(d[k] for d in deltas) / len(deltas)
                       for k in deltas[0]}
                if global_params is None:
                    global_params = {k: avg[k] for k in avg}
                else:
                    global_params = {k: global_params[k] + avg[k]
                                     for k in global_params}
                for i in range(self.n_workers):
                    _send(fs[i], pickle.dumps(global_params))
        finally:
            for c in conns:
                c.close()
            srv.close()

        self.result.put(global_params)


def run_coordinator(addr, n_workers: int, n_rounds: int, result):
    return Coordinator(addr, n_workers, n_rounds, result)
