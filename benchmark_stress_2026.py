#!/usr/bin/env python3
"""benchmark_stress.py — Protocolo de stress: 5 tarefas, mais não-estacionariedade."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from prototype.liquid import LiquidCell, sigmoid
from prototype.plasticity import LocalLearner


def generate_hard_task(name, n_steps=1500, seed=0):
    """Tarefas mais longas e distintas (frequências não-harmônicas)."""
    rng = np.random.default_rng(seed + hash(name) % 200)
    t = np.arange(n_steps)
    h = abs(hash(name)) % 1000
    phase = (h / 1000.0) * 2 * np.pi
    # frequências mais distintas (não múltiplas)
    freq_base = [0.03, 0.07, 0.11, 0.17, 0.23]
    freq = freq_base[hash(name) % len(freq_base)]
    
    noise_level = 0.1 + (h % 5) * 0.05  # ruído varia por tarefa
    
    def _make(gen, ph_off):
        u = gen.normal(0, 1, (n_steps, 2)) * 0.3
        u[:, 1] += np.sin(t * freq + phase + ph_off)
        u[:, 0] += gen.normal(0, noise_level, n_steps)  # ruído no canal alvo
        y = np.convolve(u[:, 0], np.ones(30)/30.0, mode='same')[:, None]
        return u, y

    u_tr, y_tr = _make(rng, 0.0)
    u_te, y_te = _make(np.random.default_rng(h+1), 0.5)
    return {"u_tr": u_tr, "y_tr": y_tr, "u_te": u_te, "y_te": y_te}


HARD_TASKS = ["A", "B", "C", "D", "E"]
SEEDS = list(range(5))


def evaluate(cell, learner, u_te, y_te, get_states=None):
    states, _ = cell.rollout(u_te)
    if get_states:
        states = get_states(states, learner)
    preds = states @ learner.W_out.T + learner.b_out
    return float(((preds[100:] - y_te[100:])**2).mean())


# Variante 1: Baseline
def run_baseline_hard(seed):
    tasks = {n: generate_hard_task(n, seed=seed) for n in HARD_TASKS}
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=2, n_hidden=128, sparsity=0.6, tau_min=0.4, tau_max=4.0, dt=0.1, rng=rng)
    learner = LocalLearner(128, 1, rng=rng)
    
    history = {}
    for name in HARD_TASKS:
        initial = evaluate(cell, learner, tasks[name]["u_te"], tasks[name]["y_te"])
        
        x = np.zeros(128)
        x_prev = np.zeros(128)
        for i in range(len(tasks[name]["u_tr"])):
            x_prev = x.copy()
            x, _ = cell.step(x, tasks[name]["u_tr"][i])
            if i >= 100:
                learner.update(x, tasks[name]["y_tr"][i])
                learner.oja_update(cell, x_prev, x)
        
        final = evaluate(cell, learner, tasks[name]["u_te"], tasks[name]["y_te"])
        history[name] = {"initial": initial, "final": final}
    
    forgetting = {}
    for name in HARD_TASKS[:-1]:
        cur = evaluate(cell, learner, tasks[name]["u_te"], tasks[name]["y_te"])
        forgetting[name] = cur - history[name]["initial"]
    
    return {
        "forgetting": forgetting,
        "mean_forgetting": float(np.mean(list(forgetting.values()))),
        "mean_final_error": float(np.mean([h["final"] for h in history.values()])),
    }


# Variante 2: M-LTC (com mais memória para tarefas difíceis)
class MLTCellHard(LiquidCell):
    def __init__(self, n_in, n_hidden, mem_ratio=0.4, **kwargs):
        super().__init__(n_in, n_hidden, **kwargs)
        rng = self.rng
        n_mem = max(1, int(n_hidden * mem_ratio))
        self.n_mem = n_mem
        self.n_learn = n_hidden - n_mem
        
        self.W_mem = rng.normal(0, 0.2, (n_mem, n_mem)) * 0.3
        self.b_mem = np.zeros(n_mem)
        self.alpha_mem = 0.05
        self.W_ml = rng.normal(0, 0.05, (self.n_learn, n_mem))
        self.W_lm = rng.normal(0, 0.05, (n_mem, self.n_learn))
    
    def step(self, x, u):
        x_mem = x[:self.n_mem]
        x_learn = x[self.n_mem:]
        
        f_learn = sigmoid(self.W_in[self.n_mem:, :] @ u + self.W_rec[self.n_mem:, self.n_mem:] @ x_learn + self.b[self.n_mem:])
        tau_learn = self.dt / self.tau[self.n_mem:]
        dx_learn = (-x_learn + f_learn * self.A[self.n_mem:]) * tau_learn
        
        mem_in = self.W_mem @ x_mem + self.b_mem + self.W_lm @ x_learn
        dx_mem = -self.alpha_mem * x_mem + np.tanh(mem_in) * self.alpha_mem
        dx_learn += (self.W_ml @ x_mem) * tau_learn
        
        return np.concatenate([x_mem + dx_mem, x_learn + dx_learn]), None


def run_mltc_hard(seed):
    tasks = {n: generate_hard_task(n, seed=seed) for n in HARD_TASKS}
    rng = np.random.default_rng(seed)
    cell = MLTCellHard(n_in=2, n_hidden=128, mem_ratio=0.4, sparsity=0.6, tau_min=0.4, tau_max=4.0, dt=0.1, rng=rng)
    learner = LocalLearner(128, 1, rng=rng)
    
    history = {}
    for name in HARD_TASKS:
        initial = evaluate(cell, learner, tasks[name]["u_te"], tasks[name]["y_te"])
        
        x = np.zeros(128)
        x_prev = np.zeros(128)
        for i in range(len(tasks[name]["u_tr"])):
            x_prev = x.copy()
            x, _ = cell.step(x, tasks[name]["u_tr"][i])
            if i >= 100:
                learner.update(x, tasks[name]["y_tr"][i])
                learner.oja_update(cell, x_prev, x)
        
        final = evaluate(cell, learner, tasks[name]["u_te"], tasks[name]["y_te"])
        history[name] = {"initial": initial, "final": final}
    
    forgetting = {}
    for name in HARD_TASKS[:-1]:
        cur = evaluate(cell, learner, tasks[name]["u_te"], tasks[name]["y_te"])
        forgetting[name] = cur - history[name]["initial"]
    
    return {
        "forgetting": forgetting,
        "mean_forgetting": float(np.mean(list(forgetting.values()))),
        "mean_final_error": float(np.mean([h["final"] for h in history.values()])),
    }


# Variante 3: CH-HNN melhorado (inspirado no paper real)
def run_chhnn_hard(seed):
    tasks = {n: generate_hard_task(n, seed=seed) for n in HARD_TASKS}
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=2, n_hidden=128, sparsity=0.6, tau_min=0.4, tau_max=4.0, dt=0.1, rng=rng)
    learner = LocalLearner(128, 1, rng=rng)
    
    # Camada cortical: aprende regularidades ENTRE tarefas (inter-task)
    # Paper original: mPFC-CA1 (regularidades) vs DG-CA3 (memórias específicas)
    W_cort = rng.normal(0, 0.05, (128, 128)) * 0.1
    b_cort = np.zeros(128)
    regularity_buffer = []  # acumula regularidades
    
    def get_cort_states(states, learner):
        return np.tanh(states @ W_cort.T + b_cort)
    
    history = {}
    for name in HARD_TASKS:
        initial = evaluate(cell, learner, tasks[name]["u_te"], tasks[name]["y_te"], get_cort_states)
        
        x = np.zeros(128)
        x_prev = np.zeros(128)
        x_hist = []  # histórico de estados para esta tarefa
        for i in range(len(tasks[name]["u_tr"])):
            x_prev = x.copy()
            x, _ = cell.step(x, tasks[name]["u_tr"][i])
            if i >= 100:
                learner.update(x, tasks[name]["y_tr"][i])
                learner.oja_update(cell, x_prev, x)
                if i % 100 == 0:
                    x_hist.append(x.copy())
        
        # Atualiza regularidades CORTICAIS após cada tarefa
        if x_hist:
            regularity_buffer.append(np.mean(x_hist, axis=0))
            if len(regularity_buffer) >= 2:
                # Aprender mapeamento regularidades -> representação atual
                reg = regularity_buffer[-2]
                cur = regularity_buffer[-1]
                W_cort += 0.001 * np.outer(cur - reg, reg)
        
        final = evaluate(cell, learner, tasks[name]["u_te"], tasks[name]["y_te"], get_cort_states)
        history[name] = {"initial": initial, "final": final}
    
    forgetting = {}
    for name in HARD_TASKS[:-1]:
        cur = evaluate(cell, learner, tasks[name]["u_te"], tasks[name]["y_te"], get_cort_states)
        forgetting[name] = cur - history[name]["initial"]
    
    return {
        "forgetting": forgetting,
        "mean_forgetting": float(np.mean(list(forgetting.values()))),
        "mean_final_error": float(np.mean([h["final"] for h in history.values()])),
    }


def run_experiment():
    results = {"baseline": [], "mltc": [], "ch_hnn": []}
    
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        
        r = run_baseline_hard(seed)
        results["baseline"].append(r)
        print(f"  baseline: fg={r['mean_forgetting']:.4f}, err={r['mean_final_error']:.4f}")
        
        r = run_mltc_hard(seed)
        results["mltc"].append(r)
        print(f"  m-ltc:    fg={r['mean_forgetting']:.4f}, err={r['mean_final_error']:.4f}")
        
        r = run_chhnn_hard(seed)
        results["ch_hnn"].append(r)
        print(f"  ch-hnn:   fg={r['mean_forgetting']:.4f}, err={r['mean_final_error']:.4f}")
    
    return results


def analyze(results):
    print("\n" + "="*70)
    print("STRESS TEST: 5 tarefas, maior não-estacionariedade")
    print("="*70)
    
    print(f"\n{'Variante':<12} {'Forget (±SE)':<22} {'Error (±SE)':<22}")
    print("-"*56)
    
    summary = {}
    for v, runs in results.items():
        fg = [r["mean_forgetting"] for r in runs]
        err = [r["mean_final_error"] for r in runs]
        fg_m, fg_s = np.mean(fg), np.std(fg)/np.sqrt(len(fg))
        err_m, err_s = np.mean(err), np.std(err)/np.sqrt(len(err))
        summary[v] = {"fg": fg_m, "fg_se": fg_s, "err": err_m, "err_se": err_s}
        print(f"{v:<12} {fg_m:+.4f} ± {fg_s:<14.4f} {err_m:.4f} ± {err_s:<14.4f}")
    
    base = summary["baseline"]
    print("\n" + "-"*56)
    print("COMPARAÇÃO vs BASELINE:")
    
    for v in ["mltc", "ch_hnn"]:
        vv = summary[v]
        fg_diff = vv["fg"] - base["fg"]
        err_diff = vv["err"] - base["err"]
        fg_pct = (fg_diff / abs(base["fg"])) * 100 if base["fg"] != 0 else 0
        err_pct = (err_diff / base["err"]) * 100
        
        status = "SUPERA" if (fg_diff < -0.01 and err_diff <= 0.01) or (err_diff < -0.01 and fg_diff <= 0.01) else "PIOR" if fg_diff > 0.01 else "NEUTRO"
        print(f"  {v}: forget {fg_diff:+.4f} ({fg_pct:+.1f}%), error {err_diff:+.4f} ({err_pct:+.1f}%) → {status}")
    
    out = ROOT / "benchmark_stress_results_2026.json"
    with open(out, "w") as f:
        json.dump({"results": results, "summary": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()}}, f, indent=2)
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    t0 = time.time()
    results = run_experiment()
    analyze(results)
    print(f"\nTempo: {time.time()-t0:.1f}s")
