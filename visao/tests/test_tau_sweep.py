"""
test_tau_sweep.py — Tarefa 1.10 do backlog: sensibilidade a tau.

Garante que o sweep (1) roda e produz o ponto de controle uniforme, (2) o ponto
uniforme tem std finito (ruído estimado), e (3) o veredito classifica
corretamente quando o efeito está DENTRO do ruído (decoracao) vs FORA (real).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.analysis.tau_sweep import sweep, analyze  # noqa: E402


def test_sweep_uniforme_eh_controle():
    """Sweep enxuto (2 ratios: uniforme + 1 largo, 2 seeds). Confere que o
    ponto uniforme existe e tem forgetting finito."""
    res = sweep(ratios=(1.0, 3.0), seeds=(1, 2), n_steps=800, verbose=False)
    assert len(res) == 2
    uni = [r for r in res if r["uniform"]]
    assert len(uni) == 1
    assert np.isfinite(uni[0]["mean_forgetting"])
    assert np.isfinite(uni[0]["std_forgetting"])
    assert uni[0]["std_forgetting"] >= 0.0


def test_veredito_dentro_do_ruido_e_classificado_como_decoracao():
    """Quando |delta| < std do uniforme, o veredito deve dizer DECORAÇÃO,
    não inflar um ganho que é ruído. Usa dados sintéticos controlados."""
    fake = [
        {"ratio": 1.0, "uniform": True, "mean_forgetting": 0.0135, "std_forgetting": 0.0103,
         "tau_min": 1.0, "tau_max": 1.0, "per_seed": [0.01, 0.017]},
        {"ratio": 3.0, "uniform": False, "mean_forgetting": 0.0125, "std_forgetting": 0.0106,
         "tau_min": 0.577, "tau_max": 1.732, "per_seed": [0.009, 0.016]},
        {"ratio": 10.0, "uniform": False, "mean_forgetting": 0.0131, "std_forgetting": 0.0106,
         "tau_min": 0.316, "tau_max": 3.162, "per_seed": [0.01, 0.016]},
    ]
    ana = analyze(fake)
    assert "DECORAÇÃO" in ana["verdict"]
    assert abs(ana["delta_best_vs_uniform"]) < ana["uniform_std"]


def test_veredito_fora_do_ruido_e_classificado_como_ajuda():
    """Se o delta for claramente maior que o ruído, classifica como AJUDA."""
    fake = [
        {"ratio": 1.0, "uniform": True, "mean_forgetting": 0.20, "std_forgetting": 0.01,
         "tau_min": 1.0, "tau_max": 1.0, "per_seed": [0.19, 0.21]},
        {"ratio": 10.0, "uniform": False, "mean_forgetting": 0.05, "std_forgetting": 0.01,
         "tau_min": 0.316, "tau_max": 3.162, "per_seed": [0.04, 0.06]},
    ]
    ana = analyze(fake)
    assert "AJUDA" in ana["verdict"]
    assert ana["delta_best_vs_uniform"] < 0


def test_resultados_completos_existem_e_sao_honestos():
    """O sweep de 5 ratios x5 seeds foi salvo; o veredito não pode ser
    'AJUDA' se o delta estiver dentro do ruído (anti-maquiagem)."""
    path = Path(__file__).resolve().parents[1] / "analysis" / "results_tau_sweep.json"
    if not path.exists():
        pytest.skip("results_tau_sweep.json ainda não gerado (rode tau_sweep.py)")
    data = json.loads(path.read_text())
    assert "sweep" in data and "analysis" in data
    ana = data["analysis"]
    # se o veredito diz AJUDA, o delta deve ser < -std (fora do ruído)
    if "AJUDA" in ana["verdict"]:
        assert ana["delta_best_vs_uniform"] < -ana["uniform_std"]
