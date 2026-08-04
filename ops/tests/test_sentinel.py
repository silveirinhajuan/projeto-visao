"""Testes do próprio sentinela (OPS.1).

O sentinela decide se o agente autônomo pode agir. Estes testes cobrem a
lógica de decisão pura: detecção de suíte vermelha, governança violada, RAM
baixa e a escolha da próxima tarefa desbloqueada num backlog sintético.

Nada aqui toca o sistema real: usamos tmp_path e backlogs falsos, e testamos
funções puras (sem subprocess, sem I/O de rede).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OPS_DIR = Path(__file__).resolve().parents[1]
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

import sentinel  # noqa: E402


# ───────────────────────────────────────────────────── detecção de estado

def test_detecta_suite_vermelha():
    # falha de teste (failed > 0) => não verde
    assert sentinel.suite_green(exit_code=0, failed=3) is False
    # exit code != 0 => não verde mesmo sem falhas contadas
    assert sentinel.suite_green(exit_code=1, failed=0) is False
    # suíte limpa
    assert sentinel.suite_green(exit_code=0, failed=0) is True


def test_detecta_governanca_violada():
    # fronteira diverge do manifesto => não intacta
    assert sentinel.governance_intact(code=1, out="BOUNDARY BREACHED") is False
    # comando nem roda => não intacta
    assert sentinel.governance_intact(code=124, out="TIMEOUT") is False
    # selo confere
    assert sentinel.governance_intact(code=0, out="INTACTA") is True


def test_detecta_ram_baixa():
    # memória livre abaixo do piso (1.5GB)
    assert sentinel.resources_safe(mem_available_gb=1.0, swap_used_pct=50.0) is False
    # no limite exato ainda é seguro
    assert sentinel.resources_safe(mem_available_gb=1.5, swap_used_pct=50.0) is True
    # swap acima do teto (85%)
    assert sentinel.resources_safe(mem_available_gb=5.0, swap_used_pct=90.0) is False
    # leitura ausente não deve dar falso positivo de segurança
    assert sentinel.resources_safe(mem_available_gb=None, swap_used_pct=None) is False


# ──────────────────────────────────────────────── escolha de próxima tarefa

def _write_backlog(tmp_path: Path, tasks: list[dict]) -> Path:
    p = tmp_path / "BACKLOG.json"
    p.write_text(json.dumps({"tasks": tasks}))
    return p


def test_next_task_escolhe_primeira_desbloqueada(tmp_path):
    bl = _write_backlog(tmp_path, [
        {"id": "A", "status": "pending", "deps": []},
        {"id": "B", "status": "pending", "deps": []},
    ])
    t = sentinel.next_task(bl)
    assert t is not None
    assert t["id"] == "A"


def test_next_task_pula_dependencia_pendente(tmp_path):
    # A depende de algo não-done; B está livre => deve escolher B
    bl = _write_backlog(tmp_path, [
        {"id": "A", "status": "pending", "deps": ["X"]},
        {"id": "B", "status": "pending", "deps": []},
        {"id": "X", "status": "in_progress", "deps": []},
    ])
    t = sentinel.next_task(bl)
    assert t["id"] == "B"


def test_next_task_ignora_done_e_blocked(tmp_path):
    bl = _write_backlog(tmp_path, [
        {"id": "A", "status": "done", "deps": []},
        {"id": "B", "status": "blocked", "deps": []},
        {"id": "C", "status": "pending", "deps": []},
    ])
    t = sentinel.next_task(bl)
    assert t["id"] == "C"


def test_next_task_retorna_none_quando_nenhuma_desbloqueada(tmp_path):
    bl = _write_backlog(tmp_path, [
        {"id": "A", "status": "pending", "deps": ["Z"]},
        {"id": "Z", "status": "blocked", "deps": []},
    ])
    assert sentinel.next_task(bl) is None


def test_next_task_retorna_none_com_backlog_vazio(tmp_path):
    bl = _write_backlog(tmp_path, [])
    assert sentinel.next_task(bl) is None
