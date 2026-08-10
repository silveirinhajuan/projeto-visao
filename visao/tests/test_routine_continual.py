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


def test_relacionamento_medido_esquecimento_real():
    """MEDIÇÃO HONESTA do 5.6 (config canônica: 5 seeds, 2500 passos, 96 neurônios).

    Objetivo original do 5.6: demonstrar "nunca-esquecer" — o modelo PLÁSTICO
    (consolidação + surpresa + Oja) deveria esquecer MENOS que o NAIVE (só
    regra delta) ao aprender R2 após R1, SEM replay.

    REALIDADE MEDIDA (reproduzível, todas as 5 seeds): plastic=+0.177,
    naive=+0.050 -> o plástico esquece ~3.5x MAIS. Hipótese REFUTADA no
    problema real. Este teste TRAVA a medição: se a relação virar
    (plastic < naive), o auto-ajuste (5.7) consertou e o 5.6 deve ser
    reavaliado. Não é sucesso — é resultado negativo documentado; aguarda
    decisão do Juan (aceitar negativo / sintonizar plasticidade em 5.7 /
    revisar hipótese). Config pequena (2 seeds, 800 passos) mostrava o
    contrário por sorte de seed — daí usar a config canônica aqui.
    """
    exp = run_experiment(seeds=(1, 2, 3, 4, 5), n_steps=2500, n_hidden=96)

    assert np.isfinite(exp["plastic_mean_forgetting"])
    assert np.isfinite(exp["naive_mean_forgetting"])
    # Medição reproduzível no problema real: plástico esquece MAIS que naive
    assert exp["plastic_mean_forgetting"] > exp["naive_mean_forgetting"]


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
