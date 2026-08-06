"""Testes TDD da tarefa 3.4 — Teste de transferência (FASE3_EVOLUCAO.md §4).

Uma melhoria só é aceita se generaliza para tarefas FORA do conjunto de
otimização. Sem este portão, o sistema aprende a trapacear o benchmark:
otimiza a métrica de treino e regride silenciosamente em tudo mais.
"""
from __future__ import annotations

import pytest

from visao.evolve.archive import Archive
from visao.evolve.sandbox import Sandbox
from visao.evolve.transfer import TransferGate, SameTaskError


def test_aceita_quando_ganho_transfere():
    gate = TransferGate()
    # melhorou na otimização E se sustenta (ou melhora) na transferência
    assert gate.accepts(
        transfer_fitness=0.90,
        baseline_transfer_fitness=0.80,
        transfer_task_id="transfer_A",
        opt_task_id="opt_X",
    ) is True


def test_rejeita_quando_trapaceia_benchmark():
    gate = TransferGate()
    # ganhou muito na otimização, mas PIOROU na transferência => trapaça
    assert gate.accepts(
        transfer_fitness=0.55,
        baseline_transfer_fitness=0.80,
        transfer_task_id="transfer_A",
        opt_task_id="opt_X",
    ) is False


def test_tolerancia_admite_regressao_pequena():
    gate = TransferGate(transfer_tolerance=0.1)
    # regressão de 0.05 está dentro da tolerância
    assert gate.accepts(
        transfer_fitness=0.75,
        baseline_transfer_fitness=0.80,
        transfer_task_id="transfer_A",
        opt_task_id="opt_X",
    ) is True


def test_tarefa_transferencia_deve_ser_fora_do_conjunto():
    gate = TransferGate()
    with pytest.raises(SameTaskError):
        gate.accepts(
            transfer_fitness=0.90,
            baseline_transfer_fitness=0.80,
            transfer_task_id="opt_X",
            opt_task_id="opt_X",
        )


def test_portao_usa_fitness_medido_no_sandbox(tmp_path):
    # End-to-end com código real: a variante "trapaceia" o opt (fitness alto)
    # mas não generaliza (fitness baixo na transferência). O portão deve
    # rejeitá-la usando fitness medido de verdade no Sandbox (subprocess).
    archive = Archive()
    vid = archive.add(code="# variante", weights={"tau": 1.0})
    sandbox = Sandbox(cpu_seconds=10, mem_mb=256, tmp_root=str(tmp_path))

    opt_script = "report_fitness(0.98)"          # trapça no conjunto de treino
    transfer_script = "report_fitness(0.40)"      # não generaliza

    opt_fit = sandbox.evaluate(archive.get(vid), opt_script)
    trans_fit = sandbox.evaluate(archive.get(vid), transfer_script)

    gate = TransferGate()
    accepted = gate.accepts(
        transfer_fitness=trans_fit,
        baseline_transfer_fitness=0.80,
        transfer_task_id="transfer_A",
        opt_task_id="opt_X",
    )
    assert opt_fit == 0.98
    assert accepted is False  # ganhou no opt mas não generaliza => rejeitada
