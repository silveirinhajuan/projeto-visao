"""Driver de benchmark Tarefa 1.5 — comparação CfC vs LSTM vs GRU em psMNIST.

Treina os três modelos por N épocas e registra a curva de acurácia de
TREINO por época + acurácia de TESTE final, junto com a contagem de
parâmetros. Objetivo: dar à 1.5 uma comparação de acurácia honesta
(dentro do orçamento de CPU desta máquina) e fechar a tarefa in_progress.

Limite assumido: subamostra (n_train=2000, n_test=500). psMNIST completo
(60k) exige ~200 épocas e está fora do orçamento de um ciclo em CPU — isto
é declarado na evidência, não mascarado.
"""
from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from visao.bench import timeseries as ts

N_TRAIN = 2000
N_TEST = 500
N_EPOCHS = 60
BATCH = 128
DT = 0.1


def _curve(forward, params_init, x, y, lr):
    """Treina e devolve (params, fwd_jit, train_curve)."""
    params = params_init
    opt = ts.adam_init(params)
    n = x.shape[0]
    rng = np.random.default_rng(123)
    grad_fn = jax.jit(
        jax.grad(lambda p, xb, yb: ts._xent(
            jax.vmap(forward, in_axes=(None, 0))(p, xb), yb))
    )
    fwd_jit = jax.jit(lambda p, xb: jax.vmap(forward, in_axes=(None, 0))(p, xb))
    curve = []
    for _ in range(N_EPOCHS):
        order = rng.permutation(n)
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            g = grad_fn(params, x[idx], y[idx])
            params, opt = ts.adam_update(params, g, opt, lr=lr)
        curve.append(float(ts._accuracy(fwd_jit, params, x, y)))
    return params, fwd_jit, curve


def main():
    x_tr, y_tr, x_te, y_te = ts.load_smnist_permuted(
        n_train=N_TRAIN, n_test=N_TEST, perm_seed=42
    )
    y_tr = jnp.asarray(y_tr)
    y_te = jnp.asarray(y_te)

    out = {
        "config": {
            "n_train": N_TRAIN, "n_test": N_TEST,
            "n_epochs": N_EPOCHS, "batch": BATCH, "dt": DT,
            "dataset": "psMNIST (perm_seed=42, subamostra)",
        },
        "models": {},
    }

    # CfC (dt=0.1, lr=1e-2 — melhor config da 1.5 original)
    t0 = time.time()
    cfc_p = ts.cfc_params(cfc_hidden=64, seed=0)
    cfc_fwd = ts.make_cfc_forward(dt=DT)
    cfc_p, cfc_jit, cfc_curve = _curve(cfc_fwd, cfc_p, x_tr, y_tr, lr=1e-2)
    cfc_test = float(ts._accuracy(cfc_jit, cfc_p, x_te, y_te))
    out["models"]["cfc"] = {
        "params": ts.count_params(cfc_p),
        "test_acc": cfc_test,
        "train_curve": cfc_curve,
        "lr": 1e-2, "hidden": 64, "seconds": round(time.time() - t0, 1),
    }

    # LSTM (lr=1e-3)
    t0 = time.time()
    lstm_p = ts.lstm_params(lstm_hidden=128, seed=1)
    lstm_p, lstm_jit, lstm_curve = _curve(ts.lstm_forward, lstm_p, x_tr, y_tr, lr=1e-3)
    lstm_test = float(ts._accuracy(lstm_jit, lstm_p, x_te, y_te))
    out["models"]["lstm"] = {
        "params": ts.count_params(lstm_p),
        "test_acc": lstm_test,
        "train_curve": lstm_curve,
        "lr": 1e-3, "hidden": 128, "seconds": round(time.time() - t0, 1),
    }

    # GRU (lr=1e-3)
    t0 = time.time()
    gru_p = ts.gru_params(gru_hidden=128, seed=2)
    gru_p, gru_jit, gru_curve = _curve(ts.gru_forward, gru_p, x_tr, y_tr, lr=1e-3)
    gru_test = float(ts._accuracy(gru_jit, gru_p, x_te, y_te))
    out["models"]["gru"] = {
        "params": ts.count_params(gru_p),
        "test_acc": gru_test,
        "train_curve": gru_curve,
        "lr": 1e-3, "hidden": 128, "seconds": round(time.time() - t0, 1),
    }

    # razões de parâmetros
    cfc_n = out["models"]["cfc"]["params"]
    out["param_ratios"] = {
        "lstm_over_cfc": round(out["models"]["lstm"]["params"] / cfc_n, 2),
        "gru_over_cfc": round(out["models"]["gru"]["params"] / cfc_n, 2),
    }
    # veredito honesto
    best = max(out["models"].items(), key=lambda kv: kv[1]["test_acc"])
    out["verdict"] = {
        "best_model": best[0],
        "best_test_acc": best[1]["test_acc"],
        "cfc_vs_best_gap": round(best[1]["test_acc"] - out["models"]["cfc"]["test_acc"], 4),
        "note": "Subamostra 2k; psMNIST completo (60k, ~200ep) fora do orcamento CPU.",
    }

    here = os.path.dirname(__file__)
    path = os.path.join(here, "results_1_5_full.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("DONE", path)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "train_curve"}
                      for k, v in out["models"].items()}, indent=2))
    print("RATIOS", out["param_ratios"])
    print("VERDICT", out["verdict"])


if __name__ == "__main__":
    main()
