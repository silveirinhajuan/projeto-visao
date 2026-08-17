"""
ewc_temporal.py — Tarefa 7.2: EWC com decaimento temporal.

Hipótese: Elastic Weight Consolidation clássico penaliza desvio de pesos
importantes igualmente para todas as tarefas antigas. Mas tarefas mais
velhas deveriam ter peso de consolidação decaído (memórias se desvanecem
naturalmente).

EWC-temporal: F_i(t) = F_i(0) * exp(-λt)
  - λ=0: EWC clássico (sem decaimento)
  - λ>0: importância decai exponencialmente com o tempo
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
OUT = ROOT / "visao" / "analysis" / "results_ewc_temporal.json"


class EWCTemporalLearner(LocalLearner):
    """Learner com EWC (consolidação por importância) + decaimento temporal.
    
    Herda LocalLearner mas:
    - omega decai exponencialmente: omega *= exp(-lambda_decay * dt)
    - dt é medido em "passos de tempo" (cada update incrementa self.time)
    """

    def __init__(self, *args, lambda_decay: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_decay = lambda_decay
        self.time = 0  # contador de passos (para decaimento temporal)

    def update(self, x: np.ndarray, target: np.ndarray):
        pred = self.predict(x)
        err = target - pred
        err_mag = float(np.abs(err).mean())

        s = self.surprise(err_mag)

        # lr efetivo: fixo (sem amplificação de surpresa, conforme 7.1)
        eff = self.lr / (1.0 + self.consolidation * self.omega)

        delta = np.outer(err, x)
        self.W_out += eff * delta
        self.b_out += self.lr * err

        # importância cresce onde a sinapse fez trabalho útil
        self.omega += 0.01 * np.abs(delta)

        # DECAIMENTO TEMPORAL: omega decai exponencialmente
        if self.lambda_decay > 0:
            self.omega *= np.exp(-self.lambda_decay)

        # baselines de surpresa
        d = err_mag - self.err_ema
        self.err_ema += 0.02 * d
        self.err_var += 0.02 * (d * d - self.err_var)

        self.time += 1

        return float((err ** 2).mean()), s


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
    lambda_decay: float,
    n_steps: int = 2500,
    n_hidden: int = 96,
    n_in: int = 2,
    warmup: int = 50,
):
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=n_in, n_hidden=n_hidden, sparsity=0.6, dt=0.15, rng=rng)
    learner = EWCTemporalLearner(
        n_hidden, 1,
        lr=0.02,
        oja_lr=0.0015,
        consolidation=8.0,
        surprise_gain=0.0,  # sem surpresa (ótimo 7.1)
        lambda_decay=lambda_decay,
        rng=rng,
    )

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
        train_task(cell, learner, u_tr, y_tr, use_oja=True, warmup=warmup)
        _, _, u_te, y_te = tasks[k]
        right_after[k] = evaluate(cell, learner, u_te, y_te, warmup)

    _, _, u_te, y_te = tasks["aulas"]
    final_aulas = evaluate(cell, learner, u_te, y_te, warmup)
    forgetting = final_aulas - right_after["aulas"]

    return float(forgetting)


def run_sweep(
    lambda_values: list[float] | None = None,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    n_steps: int = 2500,
    n_hidden: int = 96,
) -> dict:
    if lambda_values is None:
        lambda_values = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01]

    results = {}
    for ld in lambda_values:
        fgs = []
        for seed in seeds:
            fg = run_two_tasks(seed, ld, n_steps, n_hidden)
            fgs.append(fg)
        mean_fg = float(np.mean(fgs))
        std_fg = float(np.std(fgs))
        results[f"lambda={ld:.4f}"] = {
            "lambda_decay": ld,
            "mean_forgetting": mean_fg,
            "std": std_fg,
            "seeds": fgs,
        }
        print(f"  lambda={ld:7.4f}: forgetting = {mean_fg:+.4f} ± {std_fg:.4f}  {fgs}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 7.2 — EWC-temporal (decaimento exponencial de omega)")
    print("R1=aulas -> R2=provas (sem replay)")
    print("=" * 70)
    results = run_sweep()

    print("\n" + "=" * 70)
    print("RANQUEAMENTO (menor forgetting = melhor):")
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_forgetting"])
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i}. {name:14s}: {r['mean_forgetting']:+.4f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] -> {OUT}")
