"""
test_param_count.py — Tarefa 1.12: tabela de parâmetros CfC vs LSTM/GRU é honesta.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.bench import param_count  # noqa: E402


def test_conta_e_gera_tabela():
    # roda a funcao principal capturando o JSON escrito
    param_count.main()
    out = Path(__file__).resolve().parents[1] / "bench" / "results_param_count.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["parametros_treinaveis"]["CfC"] > 0
    assert data["parametros_treinaveis"]["LSTM"] > data["parametros_treinaveis"]["CfC"]
    # razao deve ser >= ~8x (alegacao ~10x)
    assert data["razao_LSTM_sobre_CfC"] >= 8.0


def test_cfc_tem_menos_parametros_que_lstm_gru():
    ts = param_count._load_timeseries()
    cfc = ts.count_params(ts.cfc_params(cfc_hidden=64, seed=0))
    lstm = ts.count_params(ts.lstm_params(lstm_hidden=128, seed=1))
    assert cfc < lstm
    # CfC h=64 ja bate LSTM h=128 em parametros (arquitetura mais eficiente)
    assert lstm / cfc >= 10.0
