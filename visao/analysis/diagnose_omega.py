"""
diagnose_omega.py — Tarefa A: por que a consolidação (Ω) falha no problema real?

Hipótese: Ω cresce monotonicamente (nunca decai) e trava pesos W_out que
deveriam se adaptar quando a distribuição muda (aulas -> provas).

Protocolo: idêntico a routine_continual.py (R1=aulas, R2=provas, sem replay).
Aqui logamos por seed/timestep: omega, eff_lr, surpresa, erro, para identificar
o momento em que a consolidação vira sabotagem.
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
OUT = ROOT / "visao" / "analysis" / "results_omega_diagnosis.json"


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


def run_with_logs(seed: int, n_steps: int = 2500, n_hidden: int = 96,
                  n_in: int = 2, warmup: int = 50, log_every: int = 100):
    """Roda R1->R2 loggando estado interno do learner a cada log_every passos."""
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=n_in, n_hidden=n_hidden, sparsity=0.6, dt=0.15, rng=rng)
    learner = LocalLearner(n_hidden, 1, lr=0.02, oja_lr=0.0015,
                           consolidation=8.0, surprise_gain=4.0, rng=rng)

    kind_offset = {"aulas": 1, "provas": 2}
    tasks = {}
    for k in ("aulas", "provas"):
        o = kind_offset[k]
        u_tr, y_tr = make_routine_task(k, n_steps, seed * 10 + o)
        u_te, y_te = make_routine_task(k, 1500, seed * 10 + o + 500)
        tasks[k] = (u_tr, y_tr, u_te, y_te)

    order = ["aulas", "provas"]
    right_after = {}
    logs = []

    for k in order:
        u_tr, y_tr, _, _ = tasks[k]
        x = np.zeros(cell.n_hidden)
        for i in range(len(u_tr)):
            x_prev = x
            x, _ = cell.step(x, u_tr[i])
            if i >= warmup:
                _, s = learner.update(x, y_tr[i])
                learner.oja_update(cell, x_prev, x)
            if i % log_every == 0 and i >= warmup:
                pred = learner.predict(x)
                err = float(np.clip(((y_tr[i] - pred) ** 2).mean(), 0.0, 1e6))
                logs.append({
                    "task": k,
                    "step": i,
                    "omega_mean": float(learner.omega.mean()),
                    "omega_max": float(learner.omega.max()),
                    "omega_std": float(learner.omega.std()),
                    "eff_lr_mean": float(
                        (learner.lr * s) / (1.0 + learner.consolidation * learner.omega.mean())
                    ),
                    "surprise": float(s),
                    "err_ema": float(learner.err_ema),
                    "error": err,
                })

        _, _, u_te, y_te = tasks[k]
        right_after[k] = evaluate(cell, learner, u_te, y_te, warmup)

    _, _, u_te, y_te = tasks["aulas"]
    final_aulas = evaluate(cell, learner, u_te, y_te, warmup)
    forgetting = final_aulas - right_after["aulas"]

    return {
        "forgetting": float(forgetting),
        "right_after_aulas": float(right_after["aulas"]),
        "right_after_provas": float(right_after["provas"]),
        "final_aulas": float(final_aulas),
        "logs": logs,
    }


def run_experiment(seeds=(1, 2, 3, 4, 5)):
    results = []
    for seed in seeds:
        r = run_with_logs(seed)
        results.append(r)
        print(f"seed {seed}: forgetting={r['forgetting']:+.4f}  "
              f"after_aulas={r['right_after_aulas']:.4f}  "
              f"after_provas={r['right_after_provas']:.4f}")

    forgettings = [r["forgetting"] for r in results]
    print(f"\nMEAN forgetting: {np.mean(forgettings):+.4f}  std: {np.std(forgettings):.4f}")
    return results


def analyze(results):
    """Analisa logs: quando Ω cresce e como isso afeta o lr efetivo?"""
    print("\n" + "=" * 70)
    print("ANÁLISE: Ω vs lr efetivo ao longo do tempo")
    print("=" * 70)

    for r in results:
        logs = r["logs"]
        # logs de aulas (primeiro bloco) vs provas (segundo bloco)
        aulas_logs = [l for l in logs if l["task"] == "aulas"]
        provas_logs = [l for l in logs if l["task"] == "provas"]

        if not aulas_logs or not provas_logs:
            continue

        # Último terço de aulas (Ω acumulado no fim de R1)
        tail_aulas = aulas_logs[-len(aulas_logs)//3:]
        # Primeiro terço de provas (quando a mudança de regime acontece)
        head_provas = provas_logs[:len(provas_logs)//3]

        o_a = np.mean([l["omega_mean"] for l in tail_aulas])
        o_p = np.mean([l["omega_mean"] for l in head_provas])
        e_a = np.mean([l["eff_lr_mean"] for l in tail_aulas])
        e_p = np.mean([l["eff_lr_mean"] for l in head_provas])
        s_p = np.mean([l["surprise"] for l in head_provas])
        err_p = np.mean([l["error"] for l in head_provas])

        print(f"\nforgetting={r['forgetting']:+.4f}")
        print(f"  FIM aulas   : Ω={o_a:.4f}  eff_lr={e_a:.6f}")
        print(f"  INÍCIO provas: Ω={o_p:.4f}  eff_lr={e_p:.6f}  surprise={s_p:.3f}  err={err_p:.4f}")
        print(f"  Ω cresceu {(o_p/o_a - 1)*100:.1f}%  |  eff_lr caiu {(1 - e_p/e_a)*100:.1f}%")


if __name__ == "__main__":
    print("TAREFA A — diagnóstico de Ω")
    print("R1=aulas -> R2=provas (sem replay)")
    print("=" * 70)
    results = run_experiment()
    analyze(results)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] -> {OUT}")
