"""
streaming_bench.py — Tarefa 8.2: Benchmark de streaming com VisaoBrain.

Testa o cérebro em regime de streaming puro:
  - Dados chegam um a um (sem replay, sem batch)
  - Sem fronteira de tarefa (task-free)
  - Métrica: erro em janela deslizante

Compara três configurações:
  1. VisaoBrain completo (EWC-temporal + surpresa + Oja + meta)
  2. VisaoBrain sem meta-learning
  3. Baseline naive (sem consolidacao, sem surpresa)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "visao" / "bench" / "results_streaming.json"


def make_stream(n: int = 5000, seed: int = 0):
    """Gera um stream não-estacionário (mudança de regime embutida)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    u = np.zeros((n, 2))
    y = np.zeros((n, 1))

    # 3 regimes diferentes (sem aviso de fronteira)
    regimes = [
        ("sine", 0, 1500),
        ("saw", 1500, 3000),
        ("mixed", 3000, 5000),
    ]

    for kind, start, end in regimes:
        length = end - start
        tt = np.arange(length)
        if kind == "sine":
            signal = np.sin(2 * np.pi * 0.03 * tt)
        elif kind == "saw":
            signal = 2.0 * (tt % 40) / 40.0 - 1.0
        else:
            signal = np.sin(2 * np.pi * 0.02 * tt) + 0.5 * np.cos(2 * np.pi * 0.05 * tt)

        u[start:end, 0] += signal
        u[start:end, 1] += 0.3 * np.roll(signal, 1)
        y[start:end, 0] = np.convolve(signal, np.ones(5) / 5, mode="same")[:length]

    u += rng.normal(0, 0.05, u.shape)
    return u.astype(np.float64), y.astype(np.float64)


def run_streaming_brain(brain: VisaoBrain, u: np.ndarray, y: np.ndarray, window: int = 100) -> dict:
    """Roda o cérebro no stream e coleta métricas."""
    errors = []
    surprises = []
    lrs = []

    for i in range(len(u)):
        result = brain.learn(u[i], y[i])
        errors.append(result["err"])
        surprises.append(result["surprise"])
        lrs.append(brain.lr)

    # Métricas
    errors = np.array(errors)
    mse_total = float(np.mean(errors ** 2))
    mae_total = float(np.mean(errors))

    # Janela deslizante (últimos 500)
    mse_final = float(np.mean(errors[-500:] ** 2))
    mae_final = float(np.mean(errors[-500:]))

    # Por regime
    regime_mses = {}
    for name, start, end in [("R1_sine", 0, 1500), ("R2_saw", 1500, 3000), ("R3_mixed", 3000, 5000)]:
        regime_mses[name] = float(np.mean(errors[start:end] ** 2))

    return {
        "mse_total": mse_total,
        "mae_total": mae_total,
        "mse_final": mse_final,
        "mae_final": mae_final,
        "regime_mses": regime_mses,
        "mean_surprise": float(np.mean(surprises)),
        "final_lr": float(brain.lr),
    }


def run_benchmark(seeds: tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict:
    """Roda o benchmark completo."""
    configs = {
        "visao_full": {"consolidation": 8.0, "surprise_gain": 3.0, "meta_learn": True, "lambda_decay": 0.0005},
        "visao_no_meta": {"consolidation": 8.0, "surprise_gain": 3.0, "meta_learn": False, "lambda_decay": 0.0005},
        "naive": {"consolidation": 0.0, "surprise_gain": 0.0, "meta_learn": False, "lambda_decay": 0.0},
    }

    results = {name: [] for name in configs}

    for seed in seeds:
        u, y = make_stream(n=5000, seed=seed)

        for name, cfg in configs.items():
            brain = VisaoBrain(n_in=2, n_hidden=64, n_out=1, seed=seed, **cfg)
            metrics = run_streaming_brain(brain, u, y)
            results[name].append(metrics)

    # Agregar
    aggregated = {}
    for name, runs in results.items():
        aggregated[name] = {
            "mse_total": float(np.mean([r["mse_total"] for r in runs])),
            "mse_total_std": float(np.std([r["mse_total"] for r in runs])),
            "mse_final": float(np.mean([r["mse_final"] for r in runs])),
            "mse_final_std": float(np.std([r["mse_final"] for r in runs])),
            "regime_mses": {
                reg: float(np.mean([r["regime_mses"][reg] for r in runs]))
                for reg in ["R1_sine", "R2_saw", "R3_mixed"]
            },
        }

    return aggregated


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 8.2 — Benchmark de Streaming")
    print("Stream não-estacionário: sine → saw → mixed (sem fronteira)")
    print("=" * 70)

    results = run_benchmark()

    print("\nResultados (5 seeds):")
    print(f"{'Config':<18} {'MSE total':>12} {'MSE final':>12} {'R1':>8} {'R2':>8} {'R3':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<18} {r['mse_total']:>12.4f} {r['mse_final']:>12.4f} "
              f"{r['regime_mses']['R1_sine']:>8.4f} {r['regime_mses']['R2_saw']:>8.4f} "
              f"{r['regime_mses']['R3_mixed']:>8.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] -> {OUT}")
