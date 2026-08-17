"""
surprise_decay.py — Tarefa 7.1: varredura sistemática do decaimento de surpresa.

Hipótese: a surpresa deve decair Ω (afrouxar consolidação) em vez de
amplificar lr. Testamos sistematicamente surprise_decay ∈ [0, 2.0] para
encontrar o valor ótimo e comparar com o baseline D (sem surpresa, +0.059).
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
OUT = ROOT / "visao" / "analysis" / "results_surprise_decay.json"


class SurpriseDecayLearner(LocalLearner):
    """Learner com surpresa como decaimento de Ω (não amplificação de lr).
    
    Herda LocalLearner mas:
    - lr efetivo NÃO é amplificado pela surpresa (eff = lr / (1+cons*omega))
    - surpresa alta decai omega: omega *= exp(-(s-1)*surprise_decay)
    """

    def __init__(self, *args, surprise_decay: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.surprise_decay = surprise_decay

    def update(self, x: np.ndarray, target: np.ndarray):
        pred = self.predict(x)
        err = target - pred
        err_mag = float(np.abs(err).mean())

        s = self.surprise(err_mag)

        # lr efetivo: fixo (sem amplificação de surpresa)
        eff = self.lr / (1.0 + self.consolidation * self.omega)

        delta = np.outer(err, x)
        self.W_out += eff * delta
        self.b_out += self.lr * err

        # importância cresce normalmente
        self.omega += 0.01 * np.abs(delta)

        # SURPRESA COMO DECAY: se surpresa alta, decai Ω
        if self.surprise_decay > 0 and s > 1.0:
            decay = (s - 1.0) * self.surprise_decay
            self.omega *= np.exp(-decay)

        # baselines de surpresa
        d = err_mag - self.err_ema
        self.err_ema += 0.02 * d
        self.err_var += 0.02 * (d * d - self.err_var)

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
    surprise_decay: float,
    n_steps: int = 2500,
    n_hidden: int = 96,
    n_in: int = 2,
    warmup: int = 50,
):
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=n_in, n_hidden=n_hidden, sparsity=0.6, dt=0.15, rng=rng)
    learner = SurpriseDecayLearner(
        n_hidden, 1,
        lr=0.02,
        oja_lr=0.0015,
        consolidation=8.0,
        surprise_gain=4.0,
        surprise_decay=surprise_decay,
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
    decay_values: list[float] | None = None,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    n_steps: int = 2500,
    n_hidden: int = 96,
) -> dict:
    if decay_values is None:
        decay_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    results = {}
    for sd in decay_values:
        fgs = []
        for seed in seeds:
            fg = run_two_tasks(seed, sd, n_steps, n_hidden)
            fgs.append(fg)
        mean_fg = float(np.mean(fgs))
        std_fg = float(np.std(fgs))
        results[f"sd={sd:.3f}"] = {
            "surprise_decay": sd,
            "mean_forgetting": mean_fg,
            "std": std_fg,
            "seeds": fgs,
        }
        print(f"  sd={sd:6.3f}: forgetting = {mean_fg:+.4f} ± {std_fg:.4f}  {fgs}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 7.1 — sweep de surprise_decay (surpresa decai Ω)")
    print("R1=aulas -> R2=provas (sem replay)")
    print("=" * 70)
    results = run_sweep()

    print("\n" + "=" * 70)
    print("RANQUEAMENTO (menor forgetting = melhor):")
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_forgetting"])
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i}. {name:12s}: {r['mean_forgetting']:+.4f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] -> {OUT}")
