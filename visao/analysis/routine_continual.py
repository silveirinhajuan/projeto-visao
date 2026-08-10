"""
routine_continual.py — Tarefa 5.6: problema REAL contínuo do autor.

Substitui o psMNIST por previsão de série temporal da ROTINA/ESTUDOS:

  R1 "aulas"  : ritmo circadian semanal suave (semestre de aulas)
  R2 "provas" : rajadas curtas e intensas (semana de provas)

Protocolo idêntico à Fase 0 (experiment_continual.py / ablation.py):
reservatório líquido (LiquidCell) + aprendizado LOCAL (sem backprop),
treina R1 e depois R2 SEM replay, e mede o esquecimento de R1.

  forgetting(R1) = erro_teste(R1, pós-R2) - erro_teste(R1, logo-após-R1)

Prova o "nunca-esquecer": o modelo PLÁSTICO (consolidação + surpresa + Oja)
esquece MENOS que o NAIVE (só regra delta).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
from liquid import LiquidCell  # noqa: E402
from plasticity import LocalLearner, NaiveLearner  # noqa: E402


# ----------------------------------------------------------------- rotina real
def make_routine_task(kind: str, n: int, seed: int):
    """Gera (u, y) de um regime de rotina do autor. y é o alvo de previsão.

    kind:
      "aulas"  — semestre de aulas: estudo segue ritmo semanal suave
                 (sobe na semana, cai no fim de semana). Previsão = média
                 móvel semanal do sinal.
      "provas" — semana de provas: rajadas curtas e intensas a cada ~3 dias.
                 Previsão = suavização da rajada.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    noise = rng.normal(0, 1, (n, 2)) * 0.2

    if kind == "aulas":
        weekly = np.sin(2 * np.pi * t / 7.0)            # -1..1, ritmo semanal
        signal = 0.5 + 0.35 * weekly                    # base suave 0.15..0.85
        u = noise.copy()
        u[:, 0] += signal
        u[:, 1] += 0.3 * np.cos(2 * np.pi * t / 30.0)   # ritmo mensal leve
        y = np.convolve(signal, np.ones(7) / 7, mode="same")[:, None]
    elif kind == "provas":
        burst = (np.sin(2 * np.pi * t / 3.0) > 0.4).astype(float)   # rajadas /3d
        signal = burst * (1.0 + 0.2 * np.sin(2 * np.pi * t))        # intensas
        u = noise.copy()
        u[:, 0] += signal
        u[:, 1] += 0.5 * np.roll(burst, 1)              # antecipação de prova
        y = np.convolve(signal, np.ones(3) / 3, mode="same")[:, None]
    else:
        raise ValueError(kind)

    return u, y.astype(np.float64)


# -------------------------------------------------------------- protocolo Fase 0
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


# ----------------------------------------------------------- execução de 1 semente
def run_two_tasks(seed: int, plastic: bool, n_steps: int = 2500,
                 n_hidden: int = 96, n_in: int = 2, warmup: int = 50):
    """Treina R1(aulas) depois R2(provas). Retorna (forgetting_R1, detalhes)."""
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=n_in, n_hidden=n_hidden, sparsity=0.6, dt=0.15, rng=rng)
    if plastic:
        learner = LocalLearner(n_hidden, 1, lr=0.02, oja_lr=0.0015,
                               consolidation=8.0, surprise_gain=4.0, rng=rng)
    else:
        learner = NaiveLearner(n_hidden, 1, lr=0.02, rng=rng)
    use_oja = plastic

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

    # após aprender R2, mede o erro de teste de R1 (aulas)
    _, _, u_te, y_te = tasks["aulas"]
    final_aulas = evaluate(cell, learner, u_te, y_te, warmup)
    forgetting = final_aulas - right_after["aulas"]
    if not np.isfinite(forgetting):
        forgetting = 1e4

    return float(forgetting), {
        "final_aulas": final_aulas,
        "right_after_aulas": right_after["aulas"],
        "right_after_provas": right_after["provas"],
    }


def run_experiment(seeds=(1, 2, 3, 4, 5), n_steps: int = 2500,
                   n_hidden: int = 96, n_in: int = 2, warmup: int = 50) -> dict:
    """Roda plástico e naive em todas as seeds. Retorna esquecimento médio."""
    plas, naiv, detail_p, detail_n = [], [], [], []
    for seed in seeds:
        fp, dp = run_two_tasks(seed, True, n_steps, n_hidden, n_in, warmup)
        fn, dn = run_two_tasks(seed, False, n_steps, n_hidden, n_in, warmup)
        plas.append(fp)
        naiv.append(fn)
        detail_p.append(dp)
        detail_n.append(dn)

    return {
        "plastic_mean_forgetting": float(np.mean(plas)),
        "naive_mean_forgetting": float(np.mean(naiv)),
        "plastic_seeds": plas,
        "naive_seeds": naiv,
        "detail_plastic": detail_p,
        "detail_naive": detail_n,
        "n_seeds": len(seeds),
        "n_steps": n_steps,
        "n_hidden": n_hidden,
    }


def save_results(exp: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # remove detalhes verbosos por seed para manter o JSON enxuto
    slim = {k: v for k, v in exp.items() if not k.startswith("detail_")}
    path.write_text(json.dumps(slim, indent=2))


if __name__ == "__main__":
    print("=" * 64)
    print("TAREFA 5.6 — problema REAL contínuo (rotina/estudos do autor)")
    print("R1=aulas (suave) -> R2=provas (rajadas) | sem replay")
    print("=" * 64)
    exp = run_experiment(seeds=(1, 2, 3, 4, 5), n_steps=2500, n_hidden=96)
    save_results(exp, Path(__file__).parent / "results_routine_continual.json")

    pf = exp["plastic_mean_forgetting"]
    nf = exp["naive_mean_forgetting"]
    print(f"\n  esquecimento R1 (plástico) : {pf:+.4f}")
    print(f"  esquecimento R1 (naive)   : {nf:+.4f}")
    red = (1 - pf / nf) * 100 if nf > 0 else float("nan")
    print(f"  redução de esquecimento   : {red:.1f}%")
    print(f"  nunca-esquecer demonstrado: {'SIM' if pf < nf else 'NÃO'}")
    print(f"\n[ok] -> {Path(__file__).parent / 'results_routine_continual.json'}")
