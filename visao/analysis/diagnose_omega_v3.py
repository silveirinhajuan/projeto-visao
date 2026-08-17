"""
diagnose_omega_v3.py — Tarefa A (v3): redesign da surpresa.

Hipótese: a surpresa não deve amplificar lr (isso causa overshoot quando
a distribuição muda). Em vez disso, a surpresa deve DECAIR Ω — ou seja,
"afrouxar" a consolidação em sinapses que o sistema está usando errado,
permitindo que se adaptem.

Nova mecânica proposta:
  - surpresa alta → decaimento de Ω (esquece o que não serve)
  - surpresa baixa → crescimento normal de Ω (protege o que é importante)

Testa 4 configurações:
  D.  Consolidação+Oja (sem surpresa) — o melhor até agora
  F.  Surpresa como decaimento de Ω (nova)
  G.  Surpresa como decaimento + lr fixo (sem amplificação)
  H.  Surpresa como decaimento + lr adaptativo (original, mas com decaimento)
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
OUT = ROOT / "visao" / "analysis" / "results_omega_diagnosis_v3.json"


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


class AdaptiveLearner(LocalLearner):
    """Learner com surpresa como decaimento de Ω (não amplificação de lr)."""

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

        # SURPRESA COMO DECAY: se surpresa alta, decai Ω (afrouxa consolidação)
        if self.surprise_decay > 0 and s > 1.0:
            decay = (s - 1.0) * self.surprise_decay
            self.omega *= np.exp(-decay)

        # baselines de surpresa
        d = err_mag - self.err_ema
        self.err_ema += 0.02 * d
        self.err_var += 0.02 * (d * d - self.err_var)

        return float((err ** 2).mean()), s


class HybridLearner(LocalLearner):
    """Learner com surpresa como decaimento + lr adaptativo (original)."""

    def __init__(self, *args, surprise_decay: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.surprise_decay = surprise_decay

    def update(self, x: np.ndarray, target: np.ndarray):
        pred = self.predict(x)
        err = target - pred
        err_mag = float(np.abs(err).mean())

        s = self.surprise(err_mag)

        # lr efetivo: amplificado pela surpresa (original)
        eff = (self.lr * s) / (1.0 + self.consolidation * self.omega)

        delta = np.outer(err, x)
        self.W_out += eff * delta
        self.b_out += self.lr * s * err

        # importância cresce normalmente
        self.omega += 0.01 * np.abs(delta)

        # SURPRETA COMO DECAY: se surpresa alta, decai Ω
        if self.surprise_decay > 0 and s > 1.0:
            decay = (s - 1.0) * self.surprise_decay
            self.omega *= np.exp(-decay)

        # baselines de surpresa
        d = err_mag - self.err_ema
        self.err_ema += 0.02 * d
        self.err_var += 0.02 * (d * d - self.err_var)

        return float((err ** 2).mean()), s


def train_task(cell, learner, u, y, use_oja: bool, warmup: int = 50):
    x = np.zeros(cell.n_hidden)
    for i, (ui, yi) in enumerate(zip(u, y)):
        x_prev = x
        x, _ = cell.step(x, ui)
        if i >= warmup:
            learner.update(x, yi)
            if use_oja:
                learner.oja_update(cell, x_prev, x)


def run_config(seed: int, learner_cls, learner_kwargs: dict,
               n_steps: int = 2500, n_hidden: int = 96,
               n_in: int = 2, warmup: int = 50):
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=n_in, n_hidden=n_hidden, sparsity=0.6, dt=0.15, rng=rng)
    learner = learner_cls(n_hidden, 1, **learner_kwargs)

    use_oja = learner_kwargs.get("oja_lr", 0) > 0

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
    "D_no_surprise": {
        "cls": LocalLearner,
        "kwargs": {"consolidation": 8.0, "surprise_gain": 0.0, "oja_lr": 0.0015},
    },
    "F_adaptive_no_amplify": {
        "cls": AdaptiveLearner,
        "kwargs": {"consolidation": 8.0, "surprise_gain": 4.0, "oja_lr": 0.0015,
                   "surprise_decay": 0.05},
    },
    "G_adaptive_no_amplify_strong": {
        "cls": AdaptiveLearner,
        "kwargs": {"consolidation": 8.0, "surprise_gain": 4.0, "oja_lr": 0.0015,
                   "surprise_decay": 0.1},
    },
    "H_hybrid_amplify_and_decay": {
        "cls": HybridLearner,
        "kwargs": {"consolidation": 8.0, "surprise_gain": 4.0, "oja_lr": 0.0015,
                   "surprise_decay": 0.05},
    },
}


def run_experiment(seeds=(1, 2, 3, 4, 5)):
    results = {}
    for name, cfg in CONFIGS.items():
        fgs = []
        for seed in seeds:
            fg = run_config(seed, cfg["cls"], cfg["kwargs"])
            fgs.append(fg)
        mean_fg = float(np.mean(fgs))
        std_fg = float(np.std(fgs))
        results[name] = {"mean_forgetting": mean_fg, "std": std_fg, "seeds": fgs}
        print(f"{name:30s}: forgetting = {mean_fg:+.4f} ± {std_fg:.4f}  {fgs}")
    return results


if __name__ == "__main__":
    print("TAREFA A (v3) — redesign da surpresa")
    print("R1=aulas -> R2=provas (sem replay)")
    print("=" * 70)
    results = run_experiment()

    print("\n" + "=" * 70)
    print("RANQUEAMENTO (menor forgetting = melhor):")
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_forgetting"])
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i}. {name:30s}: {r['mean_forgetting']:+.4f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] -> {OUT}")
