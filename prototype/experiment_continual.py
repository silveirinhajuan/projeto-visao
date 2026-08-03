"""
experiment_continual.py — O teste que decide se a arquitetura vale alguma coisa.

HIPÓTESE:
  Uma rede líquida (constantes de tempo adaptativas) + plasticidade local
  (consolidação por importância + gate de surpresa) aprende três tarefas
  EM SEQUÊNCIA sem esquecer catastroficamente as anteriores — e o mesmo
  reservatório com aprendizado ingênuo (delta puro) esquece.

PROTOCOLO:
  - 3 tarefas de regressão sobre sinais temporais, apresentadas em sequência.
  - O modelo NUNCA revê dados de tarefas antigas (sem replay, sem buffer).
  - Após cada tarefa, mede-se o erro em TODAS as tarefas vistas até então.
  - Métrica: forgetting = erro_final(tarefa_i) - erro_logo_apos_treinar(tarefa_i)

FALSIFICAÇÃO:
  Se o forgetting do modelo plástico não for menor que o do ingênuo,
  a hipótese está errada e o plano precisa mudar. É isso que faz disto
  ciência e não ficção.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from liquid import LiquidCell            # noqa: E402
from plasticity import LocalLearner, NaiveLearner  # noqa: E402


# ------------------------------------------------------------------ as tarefas
def make_task(kind: str, n: int, rng: np.random.Generator):
    """Sinais temporais com estruturas TEMPORAIS diferentes.

    Todas usam a mesma entrada (ruído + drive senoidal), mas exigem
    computações distintas sobre o histórico. Distribuição de entrada muda
    entre tarefas -> é isso que dispara esquecimento catastrófico.
    """
    t = np.arange(n)
    if kind == "A":  # integração lenta (memória longa)
        u = rng.normal(0, 1, (n, 2)) * 0.3
        u[:, 1] += np.sin(t * 0.05)
        y = np.convolve(u[:, 0], np.ones(30) / 30, mode="same")[:, None]
    elif kind == "B":  # detecção de transiente rápido (memória curta)
        u = rng.normal(0, 1, (n, 2)) * 0.3
        u[:, 1] += np.sin(t * 0.31 + 1.0)
        d = np.diff(u[:, 1], prepend=u[0, 1])
        y = np.tanh(d * 4.0)[:, None]
    elif kind == "C":  # produto cruzado com atraso (memória de médio prazo)
        u = rng.normal(0, 1, (n, 2)) * 0.3
        u[:, 0] += np.sin(t * 0.11 + 2.0)
        lag = 12
        a = np.roll(u[:, 0], lag)
        a[:lag] = 0
        y = (a * u[:, 1])[:, None] * 2.0
    else:
        raise ValueError(kind)
    return u, y


# ----------------------------------------------------------------- treino/eval
def evaluate(cell, learner, u, y, warmup: int = 50) -> float:
    """Erro quadrático médio, SEM atualizar nada."""
    x = np.zeros(cell.n_hidden)
    errs = []
    for i, (ui, yi) in enumerate(zip(u, y)):
        x, _ = cell.step(x, ui)
        if i >= warmup:
            errs.append(float(((yi - learner.predict(x)) ** 2).mean()))
    return float(np.mean(errs))


def train_task(cell, learner, u, y, use_oja: bool, warmup: int = 50):
    x = np.zeros(cell.n_hidden)
    surprises = []
    for i, (ui, yi) in enumerate(zip(u, y)):
        x_prev = x
        x, fx = cell.step(x, ui)
        if i >= warmup:
            _, s = learner.update(x, yi)
            surprises.append(s)
            if use_oja:
                learner.oja_update(cell, x_prev, x)
    return surprises


def run(seed: int, plastic: bool, n_hidden: int = 96, n_steps: int = 4000):
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=2, n_hidden=n_hidden, sparsity=0.6, dt=0.15, rng=rng)

    if plastic:
        learner = LocalLearner(n_hidden, 1, lr=0.02, oja_lr=0.0015,
                               consolidation=8.0, surprise_gain=4.0, rng=rng)
    else:
        learner = NaiveLearner(n_hidden, 1, lr=0.02, rng=rng)

    tasks = {}
    for k in "ABC":
        u_tr, y_tr = make_task(k, n_steps, np.random.default_rng(seed * 10 + ord(k)))
        u_te, y_te = make_task(k, 1500, np.random.default_rng(seed * 10 + ord(k) + 500))
        tasks[k] = (u_tr, y_tr, u_te, y_te)

    right_after = {}
    matrix = []  # matrix[i][j] = erro na tarefa j depois de treinar até a tarefa i
    mean_surprise = {}

    order = list("ABC")
    for i, k in enumerate(order):
        u_tr, y_tr, _, _ = tasks[k]
        sup = train_task(cell, learner, u_tr, y_tr, use_oja=plastic)
        mean_surprise[k] = float(np.mean(sup)) if sup else 0.0

        row = []
        for j in order:
            _, _, u_te, y_te = tasks[j]
            e = evaluate(cell, learner, u_te, y_te)
            row.append(e)
            if j == k:
                right_after[k] = e
        matrix.append(row)

    final = matrix[-1]
    forgetting = {
        k: final[idx] - right_after[k]
        for idx, k in enumerate(order)
        if idx < len(order) - 1  # a última tarefa não pode ter esquecido
    }
    return {
        "final_errors": dict(zip(order, final)),
        "right_after": right_after,
        "forgetting": forgetting,
        "mean_forgetting": float(np.mean(list(forgetting.values()))),
        "mean_final_error": float(np.mean(final)),
        "mean_surprise": mean_surprise,
    }


def main():
    seeds = [1, 2, 3, 4, 5]
    t0 = time.time()
    results = {"plastic": [], "naive": []}

    for s in seeds:
        results["plastic"].append(run(s, plastic=True))
        results["naive"].append(run(s, plastic=False))

    def agg(key, field):
        return np.array([r[field] for r in results[key]])

    summary = {}
    for key in ("plastic", "naive"):
        summary[key] = {
            "mean_forgetting": float(agg(key, "mean_forgetting").mean()),
            "std_forgetting": float(agg(key, "mean_forgetting").std()),
            "mean_final_error": float(agg(key, "mean_final_error").mean()),
        }

    print("=" * 68)
    print("APRENDIZADO CONTÍNUO A -> B -> C  (sem replay, sem rever dados)")
    print(f"{len(seeds)} seeds | reservatório líquido 96 neurônios | numpy puro")
    print("=" * 68)
    for key, label in (("plastic", "LÍQUIDO + PLASTICIDADE LOCAL"), ("naive", "CONTROLE (delta puro)")):
        s = summary[key]
        print(f"\n{label}")
        print(f"  esquecimento médio : {s['mean_forgetting']:+.5f}  (±{s['std_forgetting']:.5f})")
        print(f"  erro final médio   : {s['mean_final_error']:.5f}")

    pf = summary["plastic"]["mean_forgetting"]
    nf = summary["naive"]["mean_forgetting"]
    red = (1 - pf / nf) * 100 if nf > 0 else float("nan")

    print("\n" + "-" * 68)
    print(f"REDUÇÃO DE ESQUECIMENTO: {red:.1f}%")
    verdict = "HIPÓTESE SUSTENTADA" if pf < nf else "HIPÓTESE FALSIFICADA"
    print(f"VEREDITO: {verdict}")
    print("-" * 68)

    print("\nDetalhe (seed 1, modelo plástico):")
    r = results["plastic"][0]
    for k, v in r["forgetting"].items():
        print(f"  esquecimento tarefa {k}: {v:+.5f}   surpresa média: {r['mean_surprise'][k]:.3f}")

    out = Path(__file__).parent / "results_continual.json"
    out.write_text(json.dumps({"summary": summary, "runs": results}, indent=2))
    print(f"\n[{time.time()-t0:.1f}s] resultados -> {out}")


if __name__ == "__main__":
    main()
