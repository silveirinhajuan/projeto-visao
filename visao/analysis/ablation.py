"""
ablation.py — Tarefa 1.9 do backlog: QUAL componente carrega os 97,2%?

A Fase 0 mostrou: LÍQUIDO+PLÁSTICO esquece 97,2% menos que o CONTROLE (delta puro).
Mas o "plástico" na Fase 0 era o pacote fechado LocalLearner = consolidação +
gate de surpresa + Oja. Não sabemos qual dos três é o responsável. Esta ablação
desempacota os três em grade 2x2x2 e mede o esquecimento isolado de cada um.

Reusa o protocolo EXATO da Fase 0 (make_task / evaluate / train_task / run) para
que os números sejam comparáveis. A única diferença: o learner tem 3 chaves
independentes (em vez de "tudo ligado" vs "tudo desligado").

Grade (8 células):
  consolidação (C): on/off  -- importância sináptica trava pesos úteis
  surpresa     (S): on/off  -- gate neuromodulador abre só em mudança de regime
  oja          (O): on/off  -- Hebb normalizado auto-organiza o reservatório

Métrica por configuração (média sobre 5 seeds):
  mean_forgetting  -- esquecimento médio A,B,C em sequência (sem replay)
  mean_final_error -- erro final nas 3 tarefas

A contribuição isolada de um mecanismo = (config com ele) - (config sem ele),
segurando os outros dois. Se um mecanismo for DECORAÇÃO, sua presença não deve
mudar o esquecimento — e diremos isso abertamente.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

sys_mod = __import__("sys")
sys_mod.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
from liquid import LiquidCell  # noqa: E402
from plasticity import LocalLearner  # noqa: E402


# ---------- protocolo idêntico à Fase 0 (experiment_continual.py) ----------
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
            pred = learner.predict(x)
            e = float(np.clip(((yi - pred) ** 2).mean(), 0.0, 1e6))
            if not np.isfinite(e):
                e = 1e6
            errs.append(e)
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


# ---------- learner que isola os 3 mecanismos ----------
class AblationLearner(LocalLearner):
    """Igual ao LocalLearner, mas com chaves INDEPENDENTES.

    A diferença para o NaiveLearner da Fase 0: aqui consolidação e surpresa
    são controláveis SEPARADAMENTE (o NaiveLearner zera os dois juntos).
    """

    def __init__(self, n_hidden, n_out, consolidation_on=True, surprise_on=True,
                 oja_on=True, rng=None):
        super().__init__(
            n_hidden, n_out,
            lr=0.02, oja_lr=0.0015 if oja_on else 0.0,
            consolidation=8.0 if consolidation_on else 0.0,
            surprise_gain=4.0 if surprise_on else 0.0,
            rng=rng,
        )

    def _clip_readout(self):
        # estabilidade numérica: sem freio (consolidação off), o W_out pode divergir
        # quando o portão de surpresa abre sem trava. Clip mantém o experimento
        # mensurável — não altera a semântica das 3 chaves de ablação.
        if not np.all(np.isfinite(self.W_out)):
            self.W_out = np.where(np.isfinite(self.W_out), self.W_out, 0.0)
        np.clip(self.W_out, -1e3, 1e3, out=self.W_out)
        np.clip(self.b_out, -1e3, 1e3, out=self.b_out)

    def update(self, x, target):
        err_sq, s = super().update(x, target)
        self._clip_readout()
        return err_sq, s



# ---------- run de uma configuração ----------
def run_config(consol: bool, surp: bool, oja: bool,
               seeds=(1, 2, 3, 4, 5), n_hidden=96, n_steps=2500):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        cell = LiquidCell(n_in=2, n_hidden=n_hidden, sparsity=0.6, dt=0.15, rng=rng)
        learner = AblationLearner(n_hidden, 1,
                                  consolidation_on=consol, surprise_on=surp,
                                  oja_on=oja, rng=rng)
        tasks = {}
        for k in "ABC":
            u_tr, y_tr = make_task(k, n_steps,
                                   np.random.default_rng(seed * 10 + ord(k)))
            u_te, y_te = make_task(k, 1500,
                                   np.random.default_rng(seed * 10 + ord(k) + 500))
            tasks[k] = (u_tr, y_tr, u_te, y_te)

        order = list("ABC")
        matrix = []
        right_after = {}
        diverged = False
        for k in order:
            u_tr, y_tr, _, _ = tasks[k]
            sup = train_task(cell, learner, u_tr, y_tr, use_oja=oja)
            row = []
            for j in order:
                _, _, u_te, y_te = tasks[j]
                e = evaluate(cell, learner, u_te, y_te)
                if not np.isfinite(e) or e > 1e4:
                    diverged = True
                    e = 1e4
                row.append(e)
                if j == k:
                    right_after[k] = e
            matrix.append(row)

        final = matrix[-1]
        forgetting = {k: final[idx] - right_after[k]
                      for idx, k in enumerate(order) if idx < len(order) - 1}
        mf = float(np.mean(list(forgetting.values())))
        if diverged:
            mf = float("inf")   # colapso numérico, não "redução de esquecimento"
        rows.append({
            "seed": seed,
            "diverged": diverged,
            "final_errors": dict(zip(order, final)),
            "forgetting": forgetting,
            "mean_forgetting": mf,
            "mean_final_error": float(np.mean(final)),
        })
    finite = [r["mean_forgetting"] for r in rows if np.isfinite(r["mean_forgetting"])]
    mean_fgt = float(np.mean(finite)) if finite else float("inf")
    mean_err = float(np.mean([r["mean_final_error"] for r in rows]))
    return {
        "config": {"consolidation": consol, "surprise": surp, "oja": oja},
        "diverged": any(r["diverged"] for r in rows),
        "mean_forgetting": mean_fgt,
        "mean_final_error": mean_err,
        "runs": rows,
    }


# ---------- grade completa ----------
def run_grid(seeds=(1, 2, 3, 4, 5), n_steps=2500, verbose=True):
    configs = [(c, s, o) for c in (False, True)
               for s in (False, True) for o in (False, True)]
    results = []
    t0 = time.time()
    for (c, s, o) in configs:
        r = run_config(c, s, o, seeds=seeds, n_steps=n_steps)
        results.append(r)
        if verbose:
            tag = "".join([("C" if c else "-"), ("S" if s else "-"), ("O" if o else "-")])
            print(f"  [{tag}] forgetting={r['mean_forgetting']:+.4f}  "
                  f"final_err={r['mean_final_error']:.4f}  "
                  f"({time.time()-t0:.0f}s)")
    return results


# ---------- análise de contribuição isolada ----------
def attribution(grid_results):
    """Para cada mecanismo, contribuição = (com) - (sem), segurando os outros dois.

    Ex.: contribuição da consolidação = médio de [(C on,S*,O*) - (C off,S*,O*)],
    feito sobre as 4 combinações dos outros dois mecanismos. Retorna
    dicionário mecanismo -> delta de esquecimento médio (negativo = reduz).
    """
    by_key = {(r["config"]["consolidation"], r["config"]["surprise"], r["config"]["oja"]):
              r["mean_forgetting"] for r in grid_results if np.isfinite(r["mean_forgetting"])}
    mechs = ("consolidation", "surprise", "oja")
    out = {}
    for mech in mechs:
        deltas = []
        others = [m for m in mechs if m != mech]
        for a in (False, True):
            for b in (False, True):
                def key_for(val, oa, ob):
                    d = {"consolidation": None, "surprise": None, "oja": None}
                    d[mech] = val
                    d[others[0]] = oa
                    d[others[1]] = ob
                    return (d["consolidation"], d["surprise"], d["oja"])
                k_on = key_for(True, a, b)
                k_off = key_for(False, a, b)
                if k_on in by_key and k_off in by_key:
                    deltas.append(by_key[k_on] - by_key[k_off])
        out[mech] = float(np.mean(deltas)) if deltas else float("nan")
    return out


if __name__ == "__main__":
    def tag_str(cfg):
        return "".join([("C" if cfg["consolidation"] else "-"),
                        ("S" if cfg["surprise"] else "-"),
                        ("O" if cfg["oja"] else "-")])

    print("=" * 64)
    print("ABLAÇÃO 2x2x2 — Fase 0 desempacotada (tarefa 1.9)")
    print("C=consolidação  S=gate de surpresa  O=Oja")
    print("=" * 64)
    grid = run_grid()
    attr = attribution(grid)
    baseline = next(r for r in grid
                    if not r["config"]["consolidation"]
                    and not r["config"]["surprise"]
                    and not r["config"]["oja"])["mean_forgetting"]
    full = next(r for r in grid
                if r["config"]["consolidation"]
                and r["config"]["surprise"]
                and r["config"]["oja"])["mean_forgetting"]
    diverged_cfgs = [tag_str(r["config"]) for r in grid if r["diverged"]]

    print("\n-- Contribuição isolada (delta de esquecimento vs baseline) --")
    for k, v in attr.items():
        if np.isnan(v):
            print(f"  {k:14s}: INDETERMINÁVEL (todas as células com ele divergiram)")
        else:
            print(f"  {k:14s}: {v:+.4f}  ({'reduz' if v < 0 else 'AUMENTA'} esquecimento)")
    print(f"\n  baseline (---): {baseline:+.4f}")
    print(f"  completo (CSO): {full:+.4f}")
    print(f"  redução total : {(1 - full/baseline)*100:.1f}%")
    if diverged_cfgs:
        print(f"\n  ⚠ CÉLULAS INSTÁVEIS (surpresa ON sem consolidação): {diverged_cfgs}")
        print("    -> reportadas como divergidas; NÃO contam como 'redução'.")

    out = Path(__file__).parent / "results_ablation.json"
    out.write_text(json.dumps({
        "grid": grid, "attribution": attr,
        "baseline_forgetting": baseline, "full_forgetting": full,
        "diverged_configs": diverged_cfgs,
    }, indent=2))
    print(f"\n[ok] -> {out}")
