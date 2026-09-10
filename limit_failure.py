#!/usr/bin/env python3
"""limit_failure.py — Encontra o ponto de quebra do baseline com tarefas adversariaIS."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from prototype.liquid import LiquidCell, sigmoid
from prototype.plasticity import LocalLearner


def generate_adversarial_task(name, n_steps=1500, seed=0, difficulty=1.0):
    """Tarefas com frequências próximas e ruído crescente (adversariaIS)."""
    rng = np.random.default_rng(seed + hash(name) % 200)
    t = np.arange(n_steps)
    h = abs(hash(name)) % 1000
    phase = (h / 1000.0) * 2 * np.pi
    
    # Frequências muito próximas (difícil de distinguir)
    base_freq = 0.1
    freq = base_freq + (hash(name) % 5) * 0.005  # 0.10, 0.105, 0.11, 0.115, 0.12
    
    noise_level = 0.1 * difficulty  # ruído proporcional à dificuldade
    
    def _make(gen, ph_off):
        u = gen.normal(0, 1, (n_steps, 2)) * 0.3
        u[:, 1] += np.sin(t * freq + phase + ph_off)
        u[:, 0] += gen.normal(0, noise_level, n_steps)
        y = np.convolve(u[:, 0], np.ones(30)/30.0, mode='same')[:, None]
        return u, y

    u_tr, y_tr = _make(rng, 0.0)
    u_te, y_te = _make(np.random.default_rng(h+1), 0.5)
    return {"u_tr": u_tr, "y_tr": y_tr, "u_te": u_te, "y_te": y_te}


def run_baseline_on_tasks(tasks, n_hidden=128, seed=0):
    """Executa baseline em stream de tarefas."""
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=2, n_hidden=n_hidden, sparsity=0.6, tau_min=0.4, tau_max=4.0, dt=0.1, rng=rng)
    learner = LocalLearner(n_hidden, 1, rng=rng)
    
    history = {}
    for name in tasks:
        states, _ = cell.rollout(tasks[name]["u_te"])
        preds = states @ learner.W_out.T + learner.b_out
        initial = float(((preds[100:] - tasks[name]["y_te"][100:])**2).mean())
        initial = min(initial, 1e6)
        
        x = np.zeros(n_hidden)
        x_prev = np.zeros(n_hidden)
        for i in range(len(tasks[name]["u_tr"])):
            x_prev = x.copy()
            x, _ = cell.step(x, tasks[name]["u_tr"][i])
            if i >= 100:
                learner.update(x, tasks[name]["y_tr"][i])
                learner.oja_update(cell, x_prev, x)
                x = np.clip(x, -10, 10)
        
        states, _ = cell.rollout(tasks[name]["u_te"])
        preds = states @ learner.W_out.T + learner.b_out
        final = float(((preds[100:] - tasks[name]["y_te"][100:])**2).mean())
        final = min(final, 1e6)
        history[name] = {"initial": initial, "final": final}
    
    forgetting = {}
    for name in list(tasks.keys())[:-1]:
        states, _ = cell.rollout(tasks[name]["u_te"])
        preds = states @ learner.W_out.T + learner.b_out
        cur = float(((preds[100:] - tasks[name]["y_te"][100:])**2).mean())
        cur = min(cur, 1e6)
        forgetting[name] = cur - history[name]["initial"]
    
    return {
        "forgetting": forgetting,
        "mean_forgetting": float(np.mean(list(forgetting.values()))),
        "mean_final_error": float(np.mean([h["final"] for h in history.values()])),
    }


def find_failure_point():
    """Aumenta número de tarefas até forgetting > 0."""
    print("="*70)
    print("BUSCA DO LIMITE DE FALHA")
    print("="*70)
    
    results = {}
    
    for n_tasks in [3, 5, 7, 10, 15, 20]:
        print(f"\n--- {n_tasks} tarefas ---")
        task_names = [f"T{i}" for i in range(n_tasks)]
        
        runs = []
        for seed in range(5):
            tasks = {name: generate_adversarial_task(name, seed=seed, difficulty=1.0) 
                     for name in task_names}
            r = run_baseline_on_tasks(tasks, n_hidden=128, seed=seed)
            runs.append(r)
            print(f"  seed {seed}: fg={r['mean_forgetting']:.4f}")
        
        fg_vals = [r["mean_forgetting"] for r in runs]
        mean_fg = np.mean(fg_vals)
        se_fg = np.std(fg_vals) / np.sqrt(len(fg_vals))
        results[n_tasks] = {"fg_mean": float(mean_fg), "fg_se": float(se_fg)}
        print(f"  MÉDIA: fg={mean_fg:.4f}±{se_fg:.4f}")
        
        # Se forgetting positivo, encontramos o limite
        if mean_fg > 0:
            print(f"\n>>> LIMITE ENCONTRADO: {n_tasks} tarefas (forgetting > 0)")
            break
    
    return results


def analyze_failure(results):
    print("\n" + "="*70)
    print("ANÁLISE DO LIMITE DE FALHA")
    print("="*70)
    
    print(f"\n{'N Tarefas':<12} {'Forget (±SE)':<22} {'Status':<15}")
    print("-"*50)
    
    for n, data in results.items():
        fg = data["fg_mean"]
        se = data["fg_se"]
        status = "OK (PBT)" if fg < 0 else "FALHA" if fg > 0.01 else "NEUTRO"
        print(f"{n:<12} {fg:+.4f} ± {se:<14.4f} {status:<15}")
    
    # Encontrar limite
    failure_n = None
    for n, data in results.items():
        if data["fg_mean"] > 0:
            failure_n = n
            break
    
    if failure_n:
        print(f"\n>>> Baseline falha a partir de {failure_n} tarefas adversariaIS")
    else:
        max_n = max(results.keys())
        print(f"\n>>> Baseline aguentou até {max_n} tarefas sem falhar")
    
    out = ROOT / "limit_failure_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    t0 = time.time()
    results = find_failure_point()
    analyze_failure(results)
    print(f"\nTempo: {time.time()-t0:.1f}s")
