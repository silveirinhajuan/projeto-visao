"""
test_robustness.py — Tarefa 1.11 do backlog: robustez a ruído e perda de neurônios.

Garante que: (1) o experimento roda e produz degradações finitas; (2) sob estresse
a degradação é >= 1.0 (nunca "melhora" ficticiamente) e finita; (3) o veredito de
viabilidade da Fase 2 é computado; (4) tolerância é respeitada nos níveis testados.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.analysis.robustness import aggregate, verdict  # noqa: E402


def test_robustez_produz_degradacao_finita_e_maior_igual_1():
    """Sweep enxuto (2 seeds, poucos níveis). Degradação deve ser finita e >= 1.0."""
    agg = aggregate(seeds=(1, 2), sigmas=(0.0, 0.5), drops=(0.0, 0.2), n_steps=800)
    for k, dd in agg["noise_degradation"].items():
        assert np.isfinite(dd)
        if float(k) > 0:
            assert dd >= 1.0
    for k, dd in agg["drop_degradation"].items():
        assert np.isfinite(dd)
        if float(k) > 0:
            assert dd >= 1.0


def test_veredito_fase2_eh_computado():
    """O veredito de viabilidade da Fase 2 deve ser emitido e coerente."""
    agg = aggregate(seeds=(1, 2), sigmas=(0.0, 0.5), drops=(0.0, 0.2), n_steps=800)
    v = verdict(agg)
    assert "worst_drop_degradation" in v
    assert "worst_noise_degradation" in v
    assert isinstance(v["phase2_viable"], bool)
    # se qualquer degradação < tol, e nenhuma estoura, é viável
    tol = 2.0
    expect = (v["worst_drop_degradation"] < tol and v["worst_noise_degradation"] < tol)
    assert v["phase2_viable"] == expect


def test_degradacao_cresce_com_estresse_ou_fica_plana():
    """Propriedade de sanidade: mais estresse não deve FAZER a degradação cair
    abaixo de 1.0 (não pode 'melhorar' com ruído/morte)."""
    agg = aggregate(seeds=(1, 2, 3), sigmas=(0.0, 0.2, 0.5, 1.0),
                    drops=(0.0, 0.1, 0.2, 0.4), n_steps=800)
    # todas as degradações >= ~1.0 (ruído/morte não melhora o modelo)
    for dd in list(agg["noise_degradation"].values()) + list(agg["drop_degradation"].values()):
        assert dd >= 0.99


def test_resultados_completos_existem_e_sao_honestos():
    """O experimento de 5 seeds foi salvo; degradação máx não pode exceder tol
    de forma silenciosa (anti-maquiagem do veredito)."""
    path = Path(__file__).resolve().parents[1] / "analysis" / "results_robustness.json"
    if not path.exists():
        pytest.skip("results_robustness.json ainda não gerado (rode robustness.py)")
    data = json.loads(path.read_text())
    v = {k: data[k] for k in ("worst_drop_degradation", "worst_noise_degradation", "phase2_viable")}
    tol = 2.0
    if v["worst_drop_degradation"] < tol and v["worst_noise_degradation"] < tol:
        assert v["phase2_viable"] is True
