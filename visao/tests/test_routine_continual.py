"""
test_routine_continual.py — Tarefa 5.6 do backlog: problema REAL contínuo.

Substitui o psMNIST (benchmarks sintéticos de permutação de pixels) por um
problema de SÉRIE TEMPORAL REAL da rotina/estudos do autor:

  Tarefa R1 "aulas"  : ritmo circadian semanal suave (semestre de aulas)
  Tarefa R2 "provas" : rajadas curtas e intensas (semana de provas)

O modelo (reservatório líquido + aprendizado local) treina R1 e depois R2,
SEM replay. A métrica é o esquecimento de R1 após aprender R2:

  forgetting(R1) = erro_teste(R1, pós-R2) - erro_teste(R1, logo-após-R1)

Demonstra-se o "nunca-esquecer": o modelo PLÁSTICO (consolidação + surpresa
+ Oja) esquece MENOS que o NAIVE (só regra delta, sem proteção).

Usa grade enxuta (2 seeds, n_steps curto) para caber no tempo de teste; o
experimento completo de 5 seeds está em results_routine_continual.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visao.analysis.routine_continual import (  # noqa: E402
    make_routine_task,
    run_experiment,
    save_results,
)


def test_rotina_tarefas_sao_distintas():
    """R1 (aulas) e R2 (provas) são regimes genuinamente diferentes.

    Se fossem o mesmo sinal disfarçado, o 'problema real' seria uma farsa e
    o esquecimento não mediria nada. Exigimos baixa correlação entre os alvos.
    """
    u_a, y_a = make_routine_task("aulas", 1500, 1)
    u_p, y_p = make_routine_task("provas", 1500, 1)

    # não-degenerados
    assert np.std(y_a) > 0
    assert np.std(y_p) > 0

    # regimes distintos: correlação de Pearson dos alvos abaixo de 0.7
    ya = y_a[:, 0] - y_a[:, 0].mean()
    yp = y_p[:, 0] - y_p[:, 0].mean()
    corr = float(np.dot(ya, yp) / (np.linalg.norm(ya) * np.linalg.norm(yp) + 1e-12))
    assert corr < 0.7


def test_plastico_esquece_menos_que_naive():
    """NÚCLEO do 5.6: plástico esquece MENOS que o baseline ao aprender R2."""
    exp = run_experiment(seeds=(1, 2), n_steps=800, n_hidden=64)

    assert np.isfinite(exp["plastic_mean_forgetting"])
    assert np.isfinite(exp["naive_mean_forgetting"])
    # nunca-esquecer: proteção anti-catastrofe funciona neste problema real
    assert exp["plastic_mean_forgetting"] < exp["naive_mean_forgetting"]


def test_experimento_salva_json_valido(tmp_path):
    """save_results grava JSON com chaves certas e valores finitos."""
    exp = run_experiment(seeds=(1, 2), n_steps=800, n_hidden=64)
    out = tmp_path / "results_routine_continual.json"
    save_results(exp, out)

    data = json.loads(out.read_text())
    assert "plastic_mean_forgetting" in data
    assert "naive_mean_forgetting" in data
    assert np.isfinite(data["plastic_mean_forgetting"])
    assert np.isfinite(data["naive_mean_forgetting"])
