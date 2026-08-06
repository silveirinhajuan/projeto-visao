"""transfer.py — Teste de transferência (tarefa 3.4, FASE3_EVOLUCAO.md §4).

Uma variante só é promovida se o ganho no conjunto de otimização se sustenta
num conjunto de TRANSFERÊNCIA (fora do conjunto de otimização). Sem este
portão, o sistema aprende a trapacear o benchmark: maximiza a métrica de
treino e regride silenciosamente em tudo mais. O Darwin Gödel Machine validou
o padrão inverso — melhorias em Python transferiram para Rust/C++/Go —, o que
só se mede COM um teste de transferência.

Este módulo é o decisor: recebe o fitness medido na tarefa de transferência
(e.g. via ``Sandbox``) e decide se a variante generaliza.
"""
from __future__ import annotations


class SameTaskError(ValueError):
    """A tarefa de transferência deve ser FORA do conjunto de otimização."""


class TransferGate:
    """Portão de generalização (3.4).

    ``accepts`` retorna ``True`` somente se a variante não regrediu na
    tarefa de transferência além de ``transfer_tolerance``. Se a tarefa de
    transferência for a MESMA do conjunto de otimização, levanta
    ``SameTaskError`` — o teste só existe para cobrir o que fica FORA.
    """

    def __init__(self, transfer_tolerance: float = 0.0):
        # Tolerância de regressão permitida na tarefa de transferência.
        self.transfer_tolerance = float(transfer_tolerance)

    def accepts(
        self,
        *,
        transfer_fitness: float,
        baseline_transfer_fitness: float,
        transfer_task_id,
        opt_task_id,
    ) -> bool:
        if transfer_task_id == opt_task_id:
            raise SameTaskError(
                "tarefa de transferência deve ser FORA do conjunto de "
                "otimização (recebeu a mesma tarefa)"
            )
        if transfer_fitness < baseline_transfer_fitness - self.transfer_tolerance:
            return False
        return True
