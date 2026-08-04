"""
test_ablation.py — Tarefa 1.9 do backlog: qual componente carrega os 97,2%?

Valida que o experimento de ablação 2x2x2 (1) roda sem quebrar, (2) mede a
contribuição isolada dos mecanismos para as células ESTÁVEIS, e (3) detecta
corretamente as células instáveis (surpresa ON sem consolidação => divergência),
que NÃO devem ser contadas como "redução de esquecimento".

Usa uma grade enxuta (2 seeds, n_steps curto) para caber no tempo de teste; o
experimento completo de 5 seeds está em results_ablation.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.analysis.ablation import run_grid, attribution  # noqa: E402


def test_ablation_grid_produz_atribuicao_estavel():
    """Grade enxuta: 2 seeds, n_steps=1200. Confere que as células estáveis
    dão forgetting finito e a atribuição é computável."""
    grid = run_grid(seeds=(1, 2), n_steps=1200, verbose=False)

    # 8 células
    assert len(grid) == 8

    # baseline (---) e completo (CSO) devem ser finitos
    baseline = next(r for r in grid
                    if not r["config"]["consolidation"]
                    and not r["config"]["surprise"]
                    and not r["config"]["oja"])
    full = next(r for r in grid
                if r["config"]["consolidation"]
                and r["config"]["surprise"]
                and r["config"]["oja"])
    assert np.isfinite(baseline["mean_forgetting"])
    assert np.isfinite(full["mean_forgetting"])

    # completo reduz esquecimento vs baseline
    assert full["mean_forgetting"] < baseline["mean_forgetting"]

    # atribuição para consolidação e oja (células estáveis envolvidas) é finita
    attr = attribution(grid)
    assert np.isfinite(attr["consolidation"])
    assert np.isfinite(attr["oja"])


def test_celulas_instaveis_sao_marcadas():
    """Se uma célula diverge (surpresa ON sem consolidação é a candidata natural),
    ela DEVE ser marcada diverged e NÃO reportar forgetting finito. E as células
    estáveis (baseline --- e C--) NUNCA devem divergir.

    Nota: a divergência depende da semente; em seed única curta a instável pode
    ou não disparar. O contrato testado é: (1) estável nunca diverge; (2) SE
    divergir, é marcado e não silenciado como 'redução'."""
    grid = run_grid(seeds=(1, 2, 3), n_steps=1500, verbose=False)
    for r in grid:
        cfg = r["config"]
        if (not cfg["surprise"]) and (not cfg["oja"]):
            # baseline (---) e C-- : sempre estáveis
            assert r["diverged"] is False
            assert np.isfinite(r["mean_forgetting"])
        if not np.isfinite(r["mean_forgetting"]):
            # forgetting inf só pode vir de divergência (nunca silenciado)
            assert r["diverged"] is True


def test_resultados_completos_existem_e_sao_finitos():
    """O experimento completo de 5 seeds foi salvo; as células estáveis têm
    forgetting finito (as instáveis são reportadas, não silenciadas)."""
    path = Path(__file__).resolve().parents[1] / "analysis" / "results_ablation.json"
    if not path.exists():
        pytest.skip("results_ablation.json ainda não gerado (rode ablation.py)")
    data = json.loads(path.read_text())
    assert "grid" in data and "attribution" in data
    assert data["baseline_forgetting"] > 0
    # redução total da Fase 0 (~97%) deve aparecer no completo
    assert data["full_forgetting"] < data["baseline_forgetting"]
    # pelo menos uma célula instável documentada
    assert "-S-" in data["diverged_configs"]
