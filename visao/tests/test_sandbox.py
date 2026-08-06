"""Testes TDD da tarefa 3.3 — Sandbox de fitness (FASE3_EVOLUCAO.md §3).

Contrato: toda variante roda AQUI antes de promover. Sem rede, sem
persistência, com limites de CPU/RAM. Backend de isolamento real (gVisor/
Firecracker) trocável; o CONTRATO é o que estes testes exigem.
"""
from __future__ import annotations

import os
import tempfile

from visao.evolve.sandbox import (
    Sandbox,
    ForbiddenNetworkAccess,
    NetworkEscape,
    ResourceExceeded,
    SandboxError,
)
from visao.evolve.archive import Archive, Variant


def _variant(code="baseline", weights=None, vid="v1"):
    return Variant(id=vid, code=code, weights=weights)


def _count_sb_dirs():
    d = tempfile.gettempdir()
    return len([n for n in os.listdir(d) if n.startswith("visao_sb_")])


def test_evaluate_returns_fitness():
    sb = Sandbox()
    fit = sb.evaluate(_variant(), "report_fitness(0.73)")
    assert fit == 0.73


def test_no_persistence_temp_cleaned():
    before = _count_sb_dirs()
    sb = Sandbox()
    fit = sb.evaluate(
        _variant(),
        "import os\nopen('side_effect.txt', 'w').write('x')\nreport_fitness(1.0)",
    )
    after = _count_sb_dirs()
    assert fit == 1.0
    # nada deixa rastro no disco fora do sandbox:
    assert after == before


def test_static_forbids_explicit_socket():
    sb = Sandbox()
    with __import__("pytest").raises(ForbiddenNetworkAccess):
        sb.evaluate(_variant(), "import socket\nsocket.socket()\nreport_fitness(0.0)")


def test_runtime_blocks_socket_use():
    sb = Sandbox(static_scan=False)
    with __import__("pytest").raises(NetworkEscape):
        sb.evaluate(
            _variant(),
            "import socket\nsocket.create_connection(('1.2.3.4', 80))\nreport_fitness(0.0)",
        )


def test_cpu_limit_kills_runaway():
    sb = Sandbox(cpu_seconds=1, mem_mb=512)
    with __import__("pytest").raises(ResourceExceeded):
        sb.evaluate(_variant(), "x=0\nwhile True:\n    x+=1\nreport_fitness(0.0)")


def test_mem_limit_kills_allocation():
    sb = Sandbox(cpu_seconds=10, mem_mb=128)
    with __import__("pytest").raises(ResourceExceeded):
        sb.evaluate(_variant(), "bytearray(1_000_000_000)\nreport_fitness(0.0)")


def test_eval_must_call_report():
    sb = Sandbox()
    with __import__("pytest").raises(SandboxError):
        sb.evaluate(_variant(), "y = 1 + 1")


def test_evaluate_records_into_archive():
    arch = Archive()
    vid = arch.add(code="baseline")
    sb = Sandbox()
    fit = sb.evaluate(arch.get(vid), "report_fitness(0.91)")
    assert fit == 0.91
    # conveniência: grava no arquivo (quem chama decide promover)
    arch.set_fitness(vid, fit)
    assert arch.get(vid).fitness == 0.91
