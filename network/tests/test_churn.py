"""
test_churn.py — O dashboard de churn mede a métrica que mata projetos.

O Folding@home perdeu 98,6% do pico justamente por descobrir o churn tarde.
Este módulo prova que medimos nós ativos em 7d, churn mensal e N* observado
vs previsto — com a matemática validada por uma série sintética antes de
existirem dados reais.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from network.churn_dashboard import (  # noqa: E402
    active_in_window,
    compare_nstar,
    init_db,
    monthly_churn_rate,
    record_heartbeat,
    seed_synthetic,
    steady_state_predicted,
)


def _fresh():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


class TestJanela7Dias:
    def test_conta_nos_distintos_na_janela(self):
        conn = _fresh()
        ref = date(2025, 6, 10)
        for d in (date(2025, 6, 4), date(2025, 6, 8), date(2025, 6, 10)):
            for n in ("a", "b", "c"):
                record_heartbeat(conn, n, d)
        record_heartbeat(conn, "z", date(2025, 6, 1))  # fora da janela de 7d
        assert active_in_window(conn, ref, window_days=7) == 3

    def test_sem_atividade_retorna_zero(self):
        conn = _fresh()
        assert active_in_window(conn, date(2025, 6, 10), window_days=7) == 0

    def test_heartbeat_duplicado_nao_dobra_contagem(self):
        conn = _fresh()
        record_heartbeat(conn, "a", date(2025, 6, 10))
        record_heartbeat(conn, "a", date(2025, 6, 10))
        assert active_in_window(conn, date(2025, 6, 10)) == 1


class TestChurnMensal:
    def test_churn_zero_quando_todos_continuam(self):
        conn = _fresh()
        for n in ("a", "b", "c"):
            record_heartbeat(conn, n, date(2025, 1, 15))
            record_heartbeat(conn, n, date(2025, 2, 15))
        assert monthly_churn_rate(conn, 2025, 1) == pytest.approx(0.0)

    def test_churn_um_terco_quando_um_de_tres_sai(self):
        conn = _fresh()
        for n in ("a", "b", "c"):
            record_heartbeat(conn, n, date(2025, 1, 15))
        for n in ("a", "b"):  # c sumiu em fevereiro
            record_heartbeat(conn, n, date(2025, 2, 15))
        assert monthly_churn_rate(conn, 2025, 1) == pytest.approx(1 / 3)

    def test_mes_sem_nos_retorna_zero(self):
        conn = _fresh()
        assert monthly_churn_rate(conn, 2025, 1) == 0.0


class TestEstadoDeEquilibrio:
    def test_steady_state_fechado_e_entrada_sobre_churn(self):
        assert steady_state_predicted(200, 0.2) == pytest.approx(1000.0)
        assert steady_state_predicted(100, 0.0) == float("inf")

    def test_serie_sintetica_recupera_parametros_e_nstar(self):
        conn = _fresh()
        # Decai de 2000 para o equilíbrio de 1000 (inflow 200 / churn 0,2).
        seed_synthetic(conn, initial=2000, monthly_inflow=200,
                       monthly_churn=0.2, months=60)
        res = compare_nstar(conn, 2025, 1, n_months=60)
        assert res["monthly_churn"] == pytest.approx(0.2, abs=1e-2)
        assert res["monthly_inflow"] == pytest.approx(200.0, abs=5.0)
        assert res["predicted"] == pytest.approx(1000.0, abs=1.0)
        assert res["observed"] == pytest.approx(1000.0, abs=1.0)
        assert res["ratio"] == pytest.approx(1.0, abs=1e-2)
