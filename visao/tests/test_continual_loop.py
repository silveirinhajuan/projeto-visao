"""
test_continual_loop.py — Teste TDD para o loop contínuo single-node (Tarefa 5.8).

Garante que o ContinualLoop em visao/continual/loop.py:
1. Recebe tarefas em sequência (A -> B -> C -> D -> E) sem replay de dados antigos.
2. Registra o desempenho imediatamente após o treino de cada tarefa.
3. Avalia o esquecimento acumulado de tarefas passadas após cada nova tarefa.
4. Mede e reporta se o critério do portão (esquecimento < 5% em tarefas antigas) é satisfeito.
5. Permite salvar e carregar o estado do loop contínuo (histórico + modelo).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.continual.loop import ContinualLoop, generate_synthetic_task  # noqa: E402


def test_continual_loop_sequence_and_forgetting():
    """Testa uma sequência de 3 tarefas e verifica o cálculo de esquecimento."""
    rng = np.random.default_rng(42)
    loop = ContinualLoop(n_in=2, n_hidden=64, seed=42)

    # Cria 3 tarefas sintéticas
    task_a = generate_synthetic_task("A", n_steps=1000, rng=rng)
    task_b = generate_synthetic_task("B", n_steps=1000, rng=rng)
    task_c = generate_synthetic_task("C", n_steps=1000, rng=rng)

    # 1. Treina tarefa A
    res_a = loop.add_and_train_task("A", task_a["u_tr"], task_a["y_tr"], task_a["u_te"], task_a["y_te"])
    assert "A" in loop.history
    assert res_a["eval_error"] >= 0.0

    # 2. Treina tarefa B
    loop.add_and_train_task("B", task_b["u_tr"], task_b["y_tr"], task_b["u_te"], task_b["y_te"])
    assert "B" in loop.history

    # 3. Treina tarefa C
    loop.add_and_train_task("C", task_c["u_tr"], task_c["y_tr"], task_c["u_te"], task_c["y_te"])
    assert "C" in loop.history

    # Mede esquecimento de A e B após C
    forgetting = loop.get_forgetting()
    assert "A" in forgetting
    assert "B" in forgetting
    assert "C" not in forgetting  # C é a tarefa atual, não há esquecimento medido para ela

    # Média de esquecimento
    mean_forg = loop.get_mean_forgetting()
    assert isinstance(mean_forg, float)


def test_continual_loop_gate_criterion():
    """Testa se o método gate_passed verifica a meta de esquecimento < 5%."""
    loop = ContinualLoop(n_in=2, n_hidden=64, seed=123)
    rng = np.random.default_rng(123)

    for name in ["T1", "T2", "T3"]:
        t = generate_synthetic_task(name, n_steps=800, rng=rng)
        loop.add_and_train_task(name, t["u_tr"], t["y_tr"], t["u_te"], t["y_te"])

    status = loop.check_gate_status(threshold_pct=0.05, min_tasks=3)
    assert "passed" in status
    assert "task_count" in status
    assert status["task_count"] == 3


def test_continual_loop_persistence():
    """Testa se o estado do loop (histórico e pesos) pode ser salvo e recarregado."""
    rng = np.random.default_rng(99)
    loop = ContinualLoop(n_in=2, n_hidden=48, seed=99)

    t1 = generate_synthetic_task("T1", n_steps=500, rng=rng)
    loop.add_and_train_task("T1", t1["u_tr"], t1["y_tr"], t1["u_te"], t1["y_te"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "loop_state.json"
        loop.save_state(path)
        assert path.exists()

        loop2 = ContinualLoop.load_state(path)
        assert "T1" in loop2.history
        assert loop2.history["T1"]["initial_error"] == loop.history["T1"]["initial_error"]
