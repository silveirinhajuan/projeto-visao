"""
param_count.py — Tarefa 1.12: tabela honesta de parâmetros treináveis.

Compara o número de parâmetros do CfC (arquitetura líquida) contra LSTM e GRU
nas mesmas dimensões usadas no benchmark da 1.5 (timeseries.py). A alegação do
projeto é '~10x menos parâmetros'. Este script NÃO treina — só conta, então é
independente do resultado da 1.5 (que roda no Colab). Usa as funções de
inicialização de timeseries.py para garantir que os números batem com o benchmark.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_timeseries():
    """Importa as funcoes de inicializacao de parametros do timeseries.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "timeseries", Path(__file__).resolve().parent / "timeseries.py")
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    return ts


def main():
    ts = _load_timeseries()

    # dimensoes usadas no benchmark 1.5 (timeseries.benchmark_smnist defaults)
    CFC_H = 64
    LSTM_H = 128
    GRU_H = 128   # GRU tipicamente comparado na mesma largura do LSTM
    N_IN = 1      # psMNIST: um pixel por passo

    cfc = ts.cfc_params(cfc_hidden=CFC_H, seed=0)
    lstm = ts.lstm_params(lstm_hidden=LSTM_H, seed=1)
    # GRU: 3 portas (z,r,h) -> (3*H, H+1); readout (10, H)
    import numpy as np
    rng = np.random.default_rng(2)
    h = GRU_H
    s = 1.0 / np.sqrt(h)
    gru_w = rng.normal(0, s, (3 * h, h + 1)).astype(np.float32)
    gru_wout = rng.normal(0, 1.0 / np.sqrt(h), (10, h)).astype(np.float32)
    gru_params = int(gru_w.size + gru_wout.size)

    cfc_p = ts.count_params(cfc)
    lstm_p = ts.count_params(lstm)

    tabela = {
        "dimensoes": {
            "n_in": N_IN, "cfc_hidden": CFC_H, "lstm_hidden": LSTM_H, "gru_hidden": GRU_H,
            "tarefa": "psMNIST (permutado) — entrada 1 pixel/passo, T=784",
        },
        "parametros_treinaveis": {
            "CfC": cfc_p,
            "LSTM": lstm_p,
            "GRU": gru_params,
        },
        "razao_LSTM_sobre_CfC": round(lstm_p / cfc_p, 2),
        "razao_GRU_sobre_CfC": round(gru_params / cfc_p, 2),
        "alegacao_projeto_10x_menos": True,
        "alegacao_sustentada_CfC_vs_LSTM": bool(lstm_p / cfc_p >= 8.0),
        "note": "Contagem independente do treino (1.5). A validacao de ACURACIA "
                "da 1.5 roda no Colab; aqui so contamos parametros, que e determinístico.",
    }

    out = Path(__file__).resolve().parent / "results_param_count.json"
    out.write_text(json.dumps(tabela, indent=2))
    print("=" * 56)
    print("CONTAGEM DE PARÂMETROS — CfC vs LSTM/GRU (psMNIST)")
    print("=" * 56)
    print(f"  CfC  (h={CFC_H}): {cfc_p:>7} params")
    print(f"  LSTM (h={LSTM_H}): {lstm_p:>7} params")
    print(f"  GRU  (h={GRU_H}): {gru_params:>7} params")
    print(f"  Razão LSTM/CfC : {tabela['razao_LSTM_sobre_CfC']}x")
    print(f"  Razão GRU/CfC  : {tabela['razao_GRU_sobre_CfC']}x")
    print(f"  Alegação '~10x menos' sustentada (LSTM): "
          f"{'SIM' if tabela['alegacao_sustentada_CfC_vs_LSTM'] else 'NÃO'}")
    print(f"\n[ok] -> {out}")


if __name__ == "__main__":
    main()
