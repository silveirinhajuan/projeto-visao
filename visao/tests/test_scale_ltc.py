"""
test_scale_ltc.py — tarefa 5.1 (Escala LTC p/ notebook).

Valida que a célula líquida escala para n_hidden em {512, 1024} DENTRO do
orçamento do notebook do autor (7.6GB RAM, CPU Ryzen, single-node): mede o
pico de RAM de processo e o tempo por passo de integração, e exige que caibam
no envelope. Sem isto, "roda no notebook" é uma afirmação, não um fato.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visao.analysis.scale_ltc import (
    SCALE_RESULT_PATH,
    measure_cell,
    run_scale,
)


def test_measure_cell_returns_real_metrics():
    """measure_cell devolve métricas finitas e estritamente positivas."""
    r = measure_cell(n_hidden=512, n_in=2, T=200, seed=0)
    assert r["n_hidden"] == 512
    assert r["peak_ram_mb"] > 0
    assert r["time_per_step_s"] > 0
    assert np.isfinite(r["peak_ram_mb"])
    assert np.isfinite(r["time_per_step_s"])
    for key in ("peak_ram_mb", "time_per_step_s", "rss_before_mb", "rss_after_mb"):
        assert key in r


def test_scales_to_1024_within_notebook_budget():
    """n=512 e n=1024 devem caber em 7.6GB de RAM de processo (pico)."""
    BUDGET_MB = 7.6 * 1024  # envelope do notebook Ryzen
    for n in (512, 1024):
        r = measure_cell(n_hidden=n, n_in=2, T=200, seed=0)
        assert r["peak_ram_mb"] < BUDGET_MB, (
            f"n={n} usou {r['peak_ram_mb']:.0f}MB > orçamento {BUDGET_MB:.0f}MB"
        )


def test_run_scale_writes_results():
    """run_scale mede as duas escalas e emite o JSON de resultado real."""
    out = Path(SCALE_RESULT_PATH)
    if out.exists():
        out.unlink()
    d = run_scale()
    assert out.exists(), "run_scale deve emitir results_scale_ltc.json"
    loaded = json.loads(out.read_text())
    assert "n_512" in loaded and "n_1024" in loaded
    assert d["n_512"]["n_hidden"] == 512
    assert d["n_1024"]["n_hidden"] == 1024
