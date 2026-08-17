"""
ablation_2.py — Tarefa 7.3: Re-ablação 2.0 com configuração ótima (sem surpresa).

Grade 2x2:
  - Consolidação: on/off
  - Oja: on/off

Protocolo idêntico à Fase 0 / 5.6:
  R1=aulas (suave) -> R2=provas (rajadas), sem replay
  forgetting(R1) = erro_teste(R1, pós-R2) - erro_teste(R1, logo-após-R1)

Re-surge a pergunta: qual componente carrega o ganho medido em 7.1?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
from liquid import LiquidCell
from plasticity import LocalLearner, NaiveLearner

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "visao" / "analysis" / "results_ablation_2.json"


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


def run_two_tasks(
    seed: int,
    use_consolidation: bool,
    use_oja: bool,
    n_steps: int = 2500,
    n_hidden: int = 96,
    n_in: int = 2,
    warmup: int = 50,
):
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=n_in, n_hidden=n_hidden, sparsity=0.6, dt=0.15, rng=rng)

    if use_consolidation or use_oja:
        learner = LocalLearner(
            n_hidden, 1, lr=0.02,
            oja_lr=0.0015 if use_oja else 0.0,
            consolidation=8.0 if use_consolidation else 0.0,
            surprise_gain=0.0,  # sem surpresa (ótimo 7.1)
            rng=rng,
        )
    else:
        learner = NaiveLearner(n_hidden, 1, lr=0.02, rng=rng)

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
        train_task(cell, learner, u_tr, y_tr, use_oja=use_oja, warmup=warmup)
        _, _, u_te, y_te = tasks[k]
        right_after[k] = evaluate(cell, learner, u_te, y_te, warmup)

    _, _, u_te, y_te = tasks["aulas"]
    final_aulas = evaluate(cell, learner, u_te, y_te, warmup)
    forgetting = final_aulas - right_after["aulas"]

    return float(forgetting)


def run_ablation(seeds=(1, 2, 3, 4, 5)):
    """Grade 2x2: consolidação x Oja."""
    configs = {
        "naive":       {"use_consolidation": False, "use_oja": False},
        "consol_only": {"use_consolidation": True,  "use_oja": False},
        "oja_only":    {"use_consolidation": False, "use_oja": True},
        "consol+oja":  {"use_consolidation": True,  "use_oja": True},
    }

    results = {}
    for name, cfg in configs.items():
        fgs = []
        for seed in seeds:
            fg = run_two_tasks(seed, **cfg)
            fgs.append(fg)
        mean_fg = float(np.mean(fgs))
        std_fg = float(np.std(fgs))
        results[name] = {
            "mean_forgetting": mean_fg,
            "std": std_fg,
            "seeds": fgs,
            "cfg": cfg,
        }
        print(f"  {name:15s}: forgetting = {mean_fg:+.4f} ± {std_fg:.4f}  {fgs}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 7.3 — Re-ablação 2.0 (consolidação x Oja, sem surpresa)")
    print("R1=aulas -> R2=provas (sem replay)")
    print("=" * 70)
    results = run_ablation()

    print("\n" + "=" * 70)
    print("RANQUEAMENTO (menor forgetting = melhor):")
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_forgetting"])
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i}. {name:15s}: {r['mean_forgetting']:+.4f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] -> {OUT}")
