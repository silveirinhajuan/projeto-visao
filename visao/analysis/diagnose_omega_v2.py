"""
diagnose_omega_v2.py — Tarefa A (expandida): isolamento de causa.

Hipótese revisada: o problema pode não ser Ω em si, mas a combinação
Oja (muda W_rec) + consolidação (trava W_out). Se a representação do
reservatório muda durante provas, a consolidação no readout não compensa.

Plastic (C+S+O): forgetting = 0.1766
Naive (nada):    forgetting = 0.0504

Testa 5 configurações para isolar o culpado:
  A. C+S+O (full plastic)
  B. C+S   (sem Oja)
  C. S+O   (sem consolidação)
  D. C+O   (sem surpresa)
  E. Naive (baseline)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
from liquid import LiquidCell
from plasticity import LocalLearner

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "visao" / "analysis" / "results_omega_diagnosis_v2.json"


def make_routine_task(kind: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    noise = rng.normal(0, 1, (n, 2)) * 0.2

    if kind == "aulas":
        weekly = np.sin(2 * np.pi * t / 7.0)
        signal = 0.5 + 0.35 * weekly
        u = noise.copy()
        u[:, 0] += signal
        u[:, 1] += 0.3 * np.cos(2 * np.pi * t / 30.0)
        y = np.convolve(signal, np.ones(7) / 7, mode="same")[:, None]
    elif kind == "provas":
        burst = (np.sin(2 * np.pi * t / 3.0) > 0.4).astype(float)
        signal = burst * (1.0 + 0.2 * np.sin(2 * np.pi * t))
        u = noise.copy()
        u[:, 0] += signal
        u[:, 1] += 0.5 * np.roll(burst, 1)
        y = np.convolve(signal, np.ones(3) / 3, mode="same")[:, None]
    else:
        raise ValueError(kind)
    return u, y.astype(np.float64)


def evaluate(cell, learner, u, y, warmup: int = 50) -> float:
    x = np.zeros(cell.n_hidden)
    errs = []
    for i, (ui, yi) in enumerate(zip(u, y)):
        x, _ = cell.step(x, ui)
        if i >= warmup:
            pred = learner.predict(x)
            e = float(np.clip(((yi - pred) ** 2).mean(), 0.0, 1e6))
            if not np.isfinite(e):
                e = 1e6
            errs.append(e)
    return float(np.mean(errs))


def train_task(cell, learner, u, y, use_oja: bool, warmup: int = 50):
    x = np.zeros(cell.n_hidden)
    for i, (ui, yi) in enumerate(zip(u, y)):
        x_prev = x
        x, _ = cell.step(x, ui)
        if i >= warmup:
            learner.update(x, yi)
            if use_oja:
                learner.oja_update(cell, x_prev, x)


def run_config(seed: int, consolidation: float, surprise_gain: float,
               oja_lr: float, n_steps: int = 2500, n_hidden: int = 96,
               n_in: int = 2, warmup: int = 50):
    """Roda uma config específica. Retorna forgetting."""
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=n_in, n_hidden=n_hidden, sparsity=0.6, dt=0.15, rng=rng)
    learner = LocalLearner(n_hidden, 1, lr=0.02, oja_lr=oja_lr,
                           consolidation=consolidation,
                           surprise_gain=surprise_gain, rng=rng)

    use_oja = oja_lr > 0

    kind_offset = {"aulas": 1, "provas": 2}
    tasks = {}
    for k in ("aulas", "provas"):
        o = kind_offset[k]
        u_tr, y_tr = make_routine_task(k, n_steps, seed * 10 + o)
        u_te, y_te = make_routine_task(k, 1500, seed * 10 + o + 500)
        tasks[k] = (u_tr, y_tr, u_te, y_te)

    order = ["aulas", "provas"]
    right_after = {}
    for k in order:
        u_tr, y_tr, _, _ = tasks[k]
        train_task(cell, learner, u_tr, y_tr, use_oja, warmup)
        _, _, u_te, y_te = tasks[k]
        right_after[k] = evaluate(cell, learner, u_te, y_te, warmup)

    _, _, u_te, y_te = tasks["aulas"]
    final_aulas = evaluate(cell, learner, u_te, y_te, warmup)
    forgetting = final_aulas - right_after["aulas"]

    return float(forgetting)


CONFIGS = {
    "A_plastic_CSO": {"consolidation": 8.0, "surprise_gain": 4.0, "oja_lr": 0.0015},
    "B_no_Oja_CS":  {"consolidation": 8.0, "surprise_gain": 4.0, "oja_lr": 0.0},
    "C_no_cons_SO": {"consolidation": 0.0, "surprise_gain": 4.0, "oja_lr": 0.0015},
    "D_no_surprise_CO": {"consolidation": 8.0, "surprise_gain": 0.0, "oja_lr": 0.0015},
    "E_naive":       {"consolidation": 0.0, "surprise_gain": 0.0, "oja_lr": 0.0},
}


def run_experiment(seeds=(1, 2, 3, 4, 5)):
    results = {}
    for name, cfg in CONFIGS.items():
        fgs = []
        for seed in seeds:
            fg = run_config(seed, **cfg)
            fgs.append(fg)
        mean_fg = float(np.mean(fgs))
        std_fg = float(np.std(fgs))
        results[name] = {"mean_forgetting": mean_fg, "std": std_fg, "seeds": fgs}
        print(f"{name:20s}: forgetting = {mean_fg:+.4f} ± {std_fg:.4f}  {fgs}")
    return results


if __name__ == "__main__":
    print("TAREFA A (v2) — isolamento de causa")
    print("R1=aulas -> R2=provas (sem replay)")
    print("=" * 70)
    results = run_experiment()

    print("\n" + "=" * 70)
    print("RANQUEAMENTO (menor forgetting = melhor):")
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_forgetting"])
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i}. {name:20s}: {r['mean_forgetting']:+.4f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] -> {OUT}")
