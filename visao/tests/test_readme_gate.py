"""
test_readme_gate.py — Tarefa 1.7 (PORTÃO DA FASE 1).
O README só abre a Fase 2 se reportar números MEDIDOS e reprodutíveis.
Este teste amarra o README aos arquivos de evidência reais do projeto.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _fase0_reducao_pct() -> float:
    cont = json.loads((ROOT / "prototype" / "results_continual.json").read_text())
    plastic = cont["summary"]["plastic"]["mean_forgetting"]
    naive = cont["summary"]["naive"]["mean_forgetting"]
    return (naive - plastic) / naive * 100


def _param_ratios() -> dict:
    return json.loads(
        (ROOT / "visao" / "bench" / "results_param_count.json").read_text()
    )


def test_readme_existe_com_numeros_da_fase0():
    readme = ROOT / "README.md"
    assert readme.exists(), (
        "README.md ausente — portão da Fase 1 não abre sem números medidos."
    )
    text = readme.read_text()
    reduc = _fase0_reducao_pct()
    assert f"{reduc:.1f}%" in text, (
        f"README deve reportar a redução medida de esquecimento ({reduc:.1f}%)."
    )


def test_readme_contem_razao_de_parametros():
    readme = ROOT / "README.md"
    text = readme.read_text()
    pc = _param_ratios()
    assert f"{pc['razao_LSTM_sobre_CfC']:.2f}x" in text, (
        "README deve citar a razão LSTM/CfC medida (claim ~10x menos parâmetros)."
    )
    assert f"{pc['razao_GRU_sobre_CfC']:.2f}x" in text, (
        "README deve citar a razão GRU/CfC medida."
    )
