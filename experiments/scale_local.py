#!/usr/bin/env python3
"""scale_local.py — Tarefa 7.4: Oja + consolidação em topologia profunda (sem backprop)."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from prototype.liquid import LiquidCell, sigmoid
from prototype.plasticity import LocalLearner


def generate_task(name, n_steps=1000, seed=0):
    rng = np.random.default_rng(seed + hash(name) % 100)
    t = np.arange(n_steps)
    h = abs(hash(name)) % 1000
    phase = (h / 1000.0) * 2 * np.pi
    freq = 0.05 + (h % 7) * 0.04

    def _make(gen, ph_off):
        u = gen.normal(0, 1, (n_steps, 2)) * 0.3
        u[:, 1] += np.sin(t * freq + phase + ph_off)
        y = np.convolve(u[:, 0], np.ones(30)/30.0, mode='same')[:, None]
        return u, y

    u_tr, y_tr = _make(rng, 0.0)
    u_te, y_te = _make(np.random.default_rng(h+1), 0.5)
    return {"u_tr": u_tr, "y_tr": y_tr, "u_te": u_te, "y_te": y_te}


TASKS = ["A", "B", "C", "D", "E"]
SEEDS = list(range(5))


class DeepLiquidStack:
    """2 camadas líquidas com Oja entre camadas (sem backprop)."""
    
    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.n_hidden = n_hidden
        
        # Camada 1: input -> hidden1
        self.cell1 = LiquidCell(n_in, n_hidden, sparsity=0.6, tau_min=0.4, tau_max=4.0, dt=0.1, rng=rng)
        
        # Camada 2: hidden1 -> hidden2
        self.cell2 = LiquidCell(n_hidden, n_hidden, sparsity=0.6, tau_min=0.4, tau_max=4.0, dt=0.1, rng=rng)
        
        # Readout
        self.learner = LocalLearner(n_hidden, n_out, rng=rng)
        
        # Oja entre camadas (auto-organização)
        self.W_inter = rng.normal(0, 0.1, (n_hidden, n_hidden)) * 0.1
        self.oja_lr_inter = 0.001
    
    def forward(self, u):
        """Forward pass: u -> cell1 -> cell2."""
        x1, _ = self.cell1.step(self.x1 if hasattr(self, 'x1') else np.zeros(self.n_hidden), u)
        x2, _ = self.cell2.step(self.x2 if hasattr(self, 'x2') else np.zeros(self.n_hidden), x1)
        return x1, x2
    
    def rollout(self, seq):
        """Rollout completo: seq -> (states1, states2)."""
        states1, states2 = [], []
        x1 = np.zeros(self.n_hidden)
        x2 = np.zeros(self.n_hidden)
        for u in seq:
            x1, _ = self.cell1.step(x1, u)
            x2, _ = self.cell2.step(x2, x1)
            states1.append(x1.copy())
            states2.append(x2.copy())
        return np.array(states1), np.array(states2)
    
    def train_step(self, x1, x2, y):
        """Treina readout + Oja entre camadas."""
        self.learner.update(x2, y)
        self.learner.oja_update(self.cell2, x1, x2)
        
        # Oja entre camadas: W_inter aprende x1 -> x2
        inter_err = x2 - self.W_inter @ x1
        self.W_inter += self.oja_lr_inter * np.outer(inter_err, x1)


def run_deep_stack(seed, n_tasks=5):
    """Executa deep stack em stream de tarefas."""
    task_names = [f"T{i}" for i in range(n_tasks)]
    tasks = {name: generate_task(name, seed=seed) for name in task_names}
    
    stack = DeepLiquidStack(n_in=2, n_hidden=64, n_out=1, seed=seed)
    
    history = {}
    for name in task_names:
        states1, states2 = stack.rollout(tasks[name]["u_te"])
        preds = states2 @ stack.learner.W_out.T + stack.learner.b_out
        initial = float(((preds[50:] - tasks[name]["y_te"][50:])**2).mean())
        initial = min(initial, 1e6)
        
        x1 = np.zeros(64)
        x2 = np.zeros(64)
        for i in range(len(tasks[name]["u_tr"])):
            x1, _ = stack.cell1.step(x1, tasks[name]["u_tr"][i])
            x2, _ = stack.cell2.step(x2, x1)
            if i >= 50:
                stack.train_step(x1, x2, tasks[name]["y_tr"][i])
                x1 = np.clip(x1, -10, 10)
                x2 = np.clip(x2, -10, 10)
        
        states1, states2 = stack.rollout(tasks[name]["u_te"])
        preds = states2 @ stack.learner.W_out.T + stack.learner.b_out
        final = float(((preds[50:] - tasks[name]["y_te"][50:])**2).mean())
        final = min(final, 1e6)
        history[name] = {"initial": initial, "final": final}
    
    forgetting = {}
    for name in task_names[:-1]:
        states1, states2 = stack.rollout(tasks[name]["u_te"])
        preds = states2 @ stack.learner.W_out.T + stack.learner.b_out
        cur = float(((preds[50:] - tasks[name]["y_te"][50:])**2).mean())
        cur = min(cur, 1e6)
        forgetting[name] = cur - history[name]["initial"]
    
    return {
        "forgetting": forgetting,
        "mean_forgetting": float(np.mean(list(forgetting.values()))),
        "mean_final_error": float(np.mean([h["final"] for h in history.values()])),
    }


def run_baseline_flat(seed, n_tasks=5):
    """Baseline de 1 campara para comparação."""
    task_names = [f"T{i}" for i in range(n_tasks)]
    tasks = {name: generate_task(name, seed=seed) for name in task_names}
    
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in=2, n_hidden=128, sparsity=0.6, tau_min=0.4, tau_max=4.0, dt=0.1, rng=rng)
    learner = LocalLearner(128, 1, rng=rng)
    
    history = {}
    for name in task_names:
        states, _ = cell.rollout(tasks[name]["u_te"])
        preds = states @ learner.W_out.T + learner.b_out
        initial = float(((preds[50:] - tasks[name]["y_te"][50:])**2).mean())
        initial = min(initial, 1e6)
        
        x = np.zeros(128)
        x_prev = np.zeros(128)
        for i in range(len(tasks[name]["u_tr"])):
            x_prev = x.copy()
            x, _ = cell.step(x, tasks[name]["u_tr"][i])
            if i >= 50:
                learner.update(x, tasks[name]["y_tr"][i])
                learner.oja_update(cell, x_prev, x)
                x = np.clip(x, -10, 10)
        
        states, _ = cell.rollout(tasks[name]["u_te"])
        preds = states @ learner.W_out.T + learner.b_out
        final = float(((preds[50:] - tasks[name]["y_te"][50:])**2).mean())
        final = min(final, 1e6)
        history[name] = {"initial": initial, "final": final}
    
    forgetting = {}
    for name in task_names[:-1]:
        states, _ = cell.rollout(tasks[name]["u_te"])
        preds = states @ learner.W_out.T + learner.b_out
        cur = float(((preds[50:] - tasks[name]["y_te"][50:])**2).mean())
        cur = min(cur, 1e6)
        forgetting[name] = cur - history[name]["initial"]
    
    return {
        "forgetting": forgetting,
        "mean_forgetting": float(np.mean(list(forgetting.values()))),
        "mean_final_error": float(np.mean([h["final"] for h in history.values()])),
    }


def run_experiment():
    print("="*70)
    print("TAREFA 7.4: Escala com Oja + consolidação em topologia profunda")
    print("="*70)
    
    results = {"deep_stack": [], "baseline_flat": []}
    
    for n_tasks in [5, 10]:
        print(f"\n--- {n_tasks} tarefas ---")
        
        for seed in range(5):
            r = run_deep_stack(seed, n_tasks=n_tasks)
            results["deep_stack"].append(r)
            print(f"  deep seed {seed}: fg={r['mean_forgetting']:.4f}, err={r['mean_final_error']:.4f}")
            
            r = run_baseline_flat(seed, n_tasks=n_tasks)
            results["baseline_flat"].append(r)
            print(f"  flat seed {seed}: fg={r['mean_forgetting']:.4f}, err={r['mean_final_error']:.4f}")
    
    return results


def analyze(results):
    print("\n" + "="*70)
    print("ANÁLISE: Deep Stack vs Baseline Flat")
    print("="*70)
    
    for variant in ["deep_stack", "baseline_flat"]:
        fg = [r["mean_forgetting"] for r in results[variant]]
        err = [r["mean_final_error"] for r in results[variant]]
        print(f"\n{variant}:")
        print(f"  forgetting: {np.mean(fg):.4f} ± {np.std(fg)/np.sqrt(len(fg)):.4f}")
        print(f"  error:      {np.mean(err):.4f} ± {np.std(err)/np.sqrt(len(err)):.4f}")
    
    out = ROOT / "scale_local_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    t0 = time.time()
    results = run_experiment()
    analyze(results)
    print(f"\nTempo: {time.time()-t0:.1f}s")
