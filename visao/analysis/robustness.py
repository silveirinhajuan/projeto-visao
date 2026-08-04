"""
robustness.py — Tarefa 1.11 do backlog: robustez a ruído e perda de neurônios.

A Fase 2 é um ENXAME descentralizado: nós CAEM. Se o organismo não tolera a
perda de unidades (ou entrada ruidosa, típica de sensores reais), a Fase 2 é
inviável e o backlog exige saber isso AGORA.

PROTOCOLO:
  - Treina-se o reservatório plástico (LocalLearner completo, C+S+O) no contínuo
    A->B->C exatamente como na Fase 0 / 1.9 / 1.10.
  - DEPOIS do treino, mede-se o erro final das 3 tarefas SOB ESTRESSE NA
    INFERÊNCIA (o modelo não re-treina):
      (a) RUÍDO: entrada contaminada por N(0, sigma) crescente.
      (b) MORTE: k% dos neurônios do estado x são zerados a cada passo
          (falla de hardware / nó off-line no enxame).
  - Degradação = erro_estressado / erro_limpo (1.0 = imune, >>1 = frágil).

MÉTRICA por nível de estresse: mean_degradation sobre 5 seeds.
A tolerância do projeto (critério de viabilidade da Fase 2): a até 20% de morte
de neurônios ou sigma de ruído moderado, a degradação deve ficar < 2x. Se
estourar, a Fase 2 precisa de redundância implícita que ainda não temos.
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


def train_continual(cell, learner, seed, n_steps=2500):
    """Treina A->B->C (protocolo Fase 0). Não retorna nada; cell/learner mudam."""
    for k in "ABC":
        u_tr, y_tr = make_task(k, n_steps, np.random.default_rng(seed * 10 + ord(k)))
        x = np.zeros(cell.n_hidden)
        for i, (ui, yi) in enumerate(zip(u_tr, y_tr)):
            x_prev = x
            x, _ = cell.step(x, ui)
            if i >= 50:
                learner.update(x, yi)
                learner.oja_update(cell, x_prev, x)


def eval_clean(cell, learner, seed):
    """Erro limpo nas 3 tarefas (baseline de degradação)."""
    errs = []
    for k in "ABC":
        u_te, y_te = make_task(k, 1500, np.random.default_rng(seed * 10 + ord(k) + 500))
        x = np.zeros(cell.n_hidden)
        for ui, yi in zip(u_te, y_te):
            x, _ = cell.step(x, ui)
            errs.append(float(((yi - learner.predict(x)) ** 2).mean()))
    return float(np.mean(errs))


def eval_noisy(cell, learner, seed, sigma):
    """Erro com ruído gaussiano na entrada (inferência)."""
    rng = np.random.default_rng(seed * 100 + 7)
    errs = []
    for k in "ABC":
        u_te, y_te = make_task(k, 1500, np.random.default_rng(seed * 10 + ord(k) + 500))
        x = np.zeros(cell.n_hidden)
        for ui, yi in zip(u_te, y_te):
            uu = ui + rng.normal(0, sigma, ui.shape)
            x, _ = cell.step(x, uu)
            errs.append(float(((yi - learner.predict(x)) ** 2).mean()))
    return float(np.mean(errs))


def eval_dropout(cell, learner, seed, frac):
    """Erro com k% dos neurônios MORTOS a cada passo (inferência)."""
    rng = np.random.default_rng(seed * 100 + 13)
    n = cell.n_hidden
    keep = int(round(n * (1.0 - frac)))
    errs = []
    for k in "ABC":
        u_te, y_te = make_task(k, 1500, np.random.default_rng(seed * 10 + ord(k) + 500))
        x = np.zeros(n)
        for ui, yi in zip(u_te, y_te):
            # escolhe keep neurônios sobreviventes (fixos por passo, como nó off)
            idx = rng.choice(n, size=keep, replace=False)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            x_full, _ = cell.step(x, ui)
            x_dead = x_full * mask           # neurônios mortos -> 0
            x = x_dead
            errs.append(float(((yi - learner.predict(x)) ** 2).mean()))
    return float(np.mean(errs))


def run(seed, sigmas=(0.0, 0.2, 0.5, 1.0), drops=(0.0, 0.1, 0.2, 0.4), n_steps=2500):
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=2, n_hidden=96, sparsity=0.6, dt=0.15, rng=rng)
    learner = LocalLearner(96, 1, lr=0.02, oja_lr=0.0015,
                           consolidation=8.0, surprise_gain=4.0, rng=rng)
    train_continual(cell, learner, seed, n_steps=n_steps)
    base = eval_clean(cell, learner, seed)

    noise = {}
    for s in sigmas:
        if s == 0.0:
            noise[str(s)] = base
        else:
            noise[str(s)] = eval_noisy(cell, learner, seed, s)
    drop = {}
    for d in drops:
        if d == 0.0:
            drop[str(d)] = base
        else:
            drop[str(d)] = eval_dropout(cell, learner, seed, d)
    return {
        "seed": seed,
        "clean_error": base,
        "noise_error": noise,
        "drop_error": drop,
        "noise_degradation": {k: (v / base if base > 0 else float("inf"))
                              for k, v in noise.items()},
        "drop_degradation": {k: (v / base if base > 0 else float("inf"))
                             for k, v in drop.items()},
    }


def aggregate(seeds=(1, 2, 3, 4, 5), **kw):
    rows = [run(s, **kw) for s in seeds]
    def mean_dict(key):
        out = {}
        for k in rows[0][key]:
            vals = [r[key][k] for r in rows]
            out[k] = float(np.mean(vals))
        return out
    return {
        "clean_error": float(np.mean([r["clean_error"] for r in rows])),
        "noise_degradation": mean_dict("noise_degradation"),
        "drop_degradation": mean_dict("drop_degradation"),
        "runs": rows,
    }


def verdict(agg, drop_tol=2.0, noise_tol=2.0):
    """Viabilidade da Fase 2: degradação < tol sob estresse relevante."""
    worst_drop = max(agg["drop_degradation"].values())
    worst_noise = max(agg["noise_degradation"].values())
    return {
        "worst_drop_degradation": worst_drop,
        "worst_noise_degradation": worst_noise,
        "phase2_viable": (worst_drop < drop_tol and worst_noise < noise_tol),
        "verdict": (
            "FASE 2 VIÁVEL (tolera perda de nós e ruído)"
            if (worst_drop < drop_tol and worst_noise < noise_tol)
            else "FASE 2 EXIGE REDUNDÂNCIA (degradação acima do tolerável)"
        ),
    }


if __name__ == "__main__":
    print("=" * 64)
    print("ROBUSTEZ — tarefa 1.11 (inferência sob estresse)")
    print("Ruído na entrada + morte de neurônios (nó cai no enxame)")
    print("=" * 64)
    t0 = time.time()
    agg = aggregate()
    v = verdict(agg)
    print(f"\nErro limpo (baseline): {agg['clean_error']:.4f}")
    print("\n-- Degradação por RUÍDO (erro_estressado / limpo) --")
    for k, dd in agg["noise_degradation"].items():
        print(f"  sigma={float(k):.2f}: {dd:.3f}x")
    print("\n-- Degradação por MORTE de neurônios --")
    for k, dd in agg["drop_degradation"].items():
        print(f"  drop={float(k)*100:.0f}%: {dd:.3f}x")
    print(f"\nPior degradação (ruído): {v['worst_noise_degradation']:.3f}x")
    print(f"Pior degradação (morte): {v['worst_drop_degradation']:.3f}x")
    print(f"Tolerância alvo         : 2.0x")
    print(f"\nVEREDITO: {v['verdict']}")
    out = Path(__file__).parent / "results_robustness.json"
    out.write_text(json.dumps({**agg, **v}, indent=2))
    print(f"\n[{time.time()-t0:.0f}s] -> {out}")
