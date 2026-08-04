"""
tau_sweep.py — Tarefa 1.10 do backlog: sensibilidade a tau, heterogeneidade importa?

HIPÓTESE do projeto: a heterogeneidade das constantes de tempo (cada neurônio
com seu próprio tau, de uma faixa log-uniforme tau_min..tau_max) é o que confere
memória multiescala à arquitetura líquida. Se um reservatório com tau UNIFORME
( todos os neurônios no mesmo tau) tiver esquecimento idêntico, a heterogeneidade
é DECORAÇÃO — e o backlog deve dizer isso, não fingir.

PROTOCOLO:
  - Mesmo aprendizado contínuo A->B->C da Fase 0 / ablação 1.9 (sem replay).
  - Varre-se a LARGURA da faixa de tau: de estreita (tau_min~tau_max, quase
    uniforme) a larga (10x de escala). Em cada ponto mede-se o esquecimento.
  - Em tau estritamente uniforme (variância zero) serve como controle direto.
  - 5 seeds por ponto.

MÉTRICA:
  mean_forgetting (A,B,C em sequência, sem rever dados) por configuração.
  delta = forgetting(heterog) - forgetting(uniform)  -> negativo = heterogeneidade ajuda.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

sys_mod = __import__("sys")
sys_mod.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
from liquid import LiquidCell  # noqa: E402
from plasticity import LocalLearner  # noqa: E402


def make_task(kind: str, n: int, rng: np.random.Generator):
    t = np.arange(n)
    if kind == "A":
        u = rng.normal(0, 1, (n, 2)) * 0.3
        u[:, 1] += np.sin(t * 0.05)
        y = np.convolve(u[:, 0], np.ones(30) / 30, mode="same")[:, None]
    elif kind == "B":
        u = rng.normal(0, 1, (n, 2)) * 0.3
        u[:, 1] += np.sin(t * 0.31 + 1.0)
        d = np.diff(u[:, 1], prepend=u[0, 1])
        y = np.tanh(d * 4.0)[:, None]
    elif kind == "C":
        u = rng.normal(0, 1, (n, 2)) * 0.3
        u[:, 0] += np.sin(t * 0.11 + 2.0)
        lag = 12
        a = np.roll(u[:, 0], lag)
        a[:lag] = 0
        y = (a * u[:, 1])[:, None] * 2.0
    else:
        raise ValueError(kind)
    return u, y


def evaluate(cell, learner, u, y, warmup: int = 50) -> float:
    x = np.zeros(cell.n_hidden)
    errs = []
    for i, (ui, yi) in enumerate(zip(u, y)):
        x, _ = cell.step(x, ui)
        if i >= warmup:
            e = float(np.clip(((yi - learner.predict(x)) ** 2).mean(), 0.0, 1e6))
            if not np.isfinite(e):
                e = 1e6
            errs.append(e)
    return float(np.mean(errs))


def train_task(cell, learner, u, y, warmup: int = 50):
    x = np.zeros(cell.n_hidden)
    for i, (ui, yi) in enumerate(zip(u, y)):
        x_prev = x
        x, fx = cell.step(x, ui)
        if i >= warmup:
            learner.update(x, yi)
            learner.oja_update(cell, x_prev, x)


def run_continual(tau_min, tau_max, seed, n_hidden=96, n_steps=2500):
    """Aprendizado contínuo A->B->C. Retorna forgetting médio (sem replay)."""
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=2, n_hidden=n_hidden, sparsity=0.6, dt=0.15, rng=rng)
    # aplica a faixa de tau solicitada (sobrescreve a heterogênea do __init__)
    if tau_min == tau_max:
        cell.tau = np.full(n_hidden, tau_min)
    else:
        cell.tau = np.exp(rng.uniform(np.log(tau_min), np.log(tau_max), n_hidden))

    learner = LocalLearner(n_hidden, 1, lr=0.02, oja_lr=0.0015,
                           consolidation=8.0, surprise_gain=4.0, rng=rng)
    tasks = {}
    for k in "ABC":
        u_tr, y_tr = make_task(k, n_steps, np.random.default_rng(seed * 10 + ord(k)))
        u_te, y_te = make_task(k, 1500, np.random.default_rng(seed * 10 + ord(k) + 500))
        tasks[k] = (u_tr, y_tr, u_te, y_te)

    order = list("ABC")
    matrix = []
    right_after = {}
    for k in order:
        u_tr, y_tr, _, _ = tasks[k]
        train_task(cell, learner, u_tr, y_tr)
        row = []
        for j in order:
            _, _, u_te, y_te = tasks[j]
            e = evaluate(cell, learner, u_te, y_te)
            row.append(e)
            if j == k:
                right_after[k] = e
        matrix.append(row)

    final = matrix[-1]
    forgetting = {k: final[idx] - right_after[k]
                  for idx, k in enumerate(order) if idx < len(order) - 1}
    return float(np.mean(list(forgetting.values())))


def sweep(center=1.0, ratios=(1.0, 1.5, 3.0, 6.0, 10.0),
          seeds=(1, 2, 3, 4, 5), n_steps=2500, verbose=True):
    """Varre a largura da faixa de tau em torno de `center`.

    ratio = tau_max/tau_min. ratio=1.0 => tau uniforme (controle).
    """
    results = []
    t0 = time.time()
    for ratio in ratios:
        tmin = center / np.sqrt(ratio)
        tmax = center * np.sqrt(ratio)
        fgt = [run_continual(tmin, tmax, s, n_steps=n_steps) for s in seeds]
        mean_f = float(np.mean(fgt))
        results.append({
            "ratio": ratio, "tau_min": float(tmin), "tau_max": float(tmax),
            "mean_forgetting": mean_f,
            "std_forgetting": float(np.std(fgt)),
            "uniform": bool(ratio == 1.0),
            "per_seed": fgt,
        })
        if verbose:
            tag = "UNIFORME" if ratio == 1.0 else f"larga {ratio:>4.1f}x"
            print(f"  tau [{tmin:.3f}..{tmax:.3f}] ({tag:>9}): "
                  f"forgetting={mean_f:+.4f} ± {np.std(fgt):.4f}  ({time.time()-t0:.0f}s)")
    return results


def analyze(results):
    """Compara heterogêneo (ratio>1) vs uniforme (ratio==1).

    IMPORTANTE: o efeito só conta se estiver FORA do ruído entre seeds. Se
    |delta| < std do uniforme, a heterogeneidade é estatisticamente
    indistinguível de tau uniforme — e o backlog exige dizer isso abertamente,
    não inflar um ganho de 8% que é ruído.
    """
    uni = next((r for r in results if r["uniform"]), None)
    if uni is None:
        return {"error": "sem ponto uniforme de controle"}
    het = [r for r in results if not r["uniform"]]
    best = min(het, key=lambda r: r["mean_forgetting"])
    worst = max(het, key=lambda r: r["mean_forgetting"])
    delta = best["mean_forgetting"] - uni["mean_forgetting"]
    noise = uni["std_forgetting"]
    if abs(delta) < noise:
        verdict = "HETEROGENEIDADE É DECORAÇÃO (delta dentro do ruído ±%.4f)" % noise
    elif delta < 0:
        verdict = "HETEROGENEIDADE AJUDA (fora do ruído)"
    else:
        verdict = "HETEROGENEIDADE PREJUDICA (fora do ruído)"
    return {
        "uniform_forgetting": uni["mean_forgetting"],
        "uniform_std": noise,
        "best_heterog": {"ratio": best["ratio"], "forgetting": best["mean_forgetting"]},
        "worst_heterog": {"ratio": worst["ratio"], "forgetting": worst["mean_forgetting"]},
        "delta_best_vs_uniform": delta,
        "verdict": verdict,
    }


if __name__ == "__main__":
    print("=" * 64)
    print("SWEEP DE TAU — tarefa 1.10")
    print("Heterogeneidade de constante de tempo importa para o esquecimento?")
    print("=" * 64)
    res = sweep()
    ana = analyze(res)
    print("\n-- Análise --")
    print(f"  tau uniforme            : {ana['uniform_forgetting']:+.4f}")
    print(f"  melhor heterogêneo     : ratio {ana['best_heterog']['ratio']:.1f}x "
          f"-> {ana['best_heterog']['forgetting']:+.4f}")
    print(f"  pior heterogêneo       : ratio {ana['worst_heterog']['ratio']:.1f}x "
          f"-> {ana['worst_heterog']['forgetting']:+.4f}")
    print(f"  delta (melhor - uniforme): {ana['delta_best_vs_uniform']:+.4f}")
    print(f"  VEREDITO                : {ana['verdict']}")

    out = Path(__file__).parent / "results_tau_sweep.json"
    out.write_text(json.dumps({"sweep": res, "analysis": ana}, indent=2))
    print(f"\n[ok] -> {out}")
