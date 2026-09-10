#!/usr/bin/env python3
"""pbt_spectrum.py — Estudo do PBT em nível de espectro de tau."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

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


TASKS = ["A", "B", "C"]
SEEDS = list(range(5))


def analyze_spectrum(cell, task):
    """Analisa o espectro de tau durante o rollout."""
    states, taus = cell.rollout(task["u_tr"])
    
    # Estatísticas de tau
    tau_mean = np.mean(taus, axis=0)  # média temporal por neurônio
    tau_std = np.std(taus, axis=0)
    
    # Distribuição: quantos neurônios rápidos vs lentos
    n_fast = np.sum(tau_mean < 0.5)
    n_slow = np.sum(tau_mean > 2.0)
    n_mid = len(tau_mean) - n_fast - n_slow
    
    return {
        "tau_mean": tau_mean.tolist(),
        "tau_std": tau_std.tolist(),
        "n_fast": int(n_fast),
        "n_slow": int(n_slow),
        "n_mid": int(n_mid),
        "tau_range": float(np.max(tau_mean) - np.min(tau_mean)),
    }


def run_spectrum_study():
    print("="*70)
    print("ESTUDO DO PBT: ESPECTRO DE TAU")
    print("="*70)
    
    results = {}
    
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        tasks = {n: generate_task(n, seed=seed) for n in TASKS}
        
        rng = np.random.default_rng(seed)
        cell = LiquidCell(n_in=2, n_hidden=64, sparsity=0.6, tau_min=0.4, tau_max=4.0, dt=0.1, rng=rng)
        learner = LocalLearner(64, 1, rng=rng)
        
        spectrum_per_task = {}
        for name in TASKS:
            # Analisar espectro ANTES do treino
            spectrum_before = analyze_spectrum(cell, tasks[name])
            
            # Treinar
            x = np.zeros(64)
            x_prev = np.zeros(64)
            for i in range(len(tasks[name]["u_tr"])):
                x_prev = x.copy()
                x, _ = cell.step(x, tasks[name]["u_tr"][i])
                if i >= 50:
                    learner.update(x, tasks[name]["y_tr"][i])
                    learner.oja_update(cell, x_prev, x)
                    x = np.clip(x, -10, 10)
            
            # Analisar espectro DEPOIS do treino
            spectrum_after = analyze_spectrum(cell, tasks[name])
            
            spectrum_per_task[name] = {
                "before": spectrum_before,
                "after": spectrum_after,
                "tau_range_change": spectrum_after["tau_range"] - spectrum_before["tau_range"],
            }
            
            print(f"  {name}: fast={spectrum_after['n_fast']}, slow={spectrum_after['n_slow']}, "
                  f"range={spectrum_after['tau_range']:.3f} (Δ={spectrum_per_task[name]['tau_range_change']:+.3f})")
        
        results[f"seed_{seed}"] = spectrum_per_task
    
    return results


def analyze_spectrum_results(results):
    print("\n" + "="*70)
    print("ANÁLISE DO ESPECTRO DE TAU")
    print("="*70)
    
    # Agregar todos os seeds
    all_changes = []
    for seed_key, tasks in results.items():
        for task_name, data in tasks.items():
            all_changes.append(data["tau_range_change"])
    
    mean_change = np.mean(all_changes)
    se_change = np.std(all_changes) / np.sqrt(len(all_changes))
    
    print(f"\nMudança média no range de tau: {mean_change:+.4f} ± {se_change:.4f}")
    
    if mean_change > 0:
        print("→ Range de tau AUMENTA após treinamento (mais diversidade dinâmica)")
    else:
        print("→ Range de tau DIMINUI após treinamento (menos diversidade)")
    
    # Contar neurônios rápidos vs lentos
    total_fast = sum(data["after"]["n_fast"] for seed in results.values() for data in seed.values())
    total_slow = sum(data["after"]["n_slow"] for seed in results.values() for data in seed.values())
    
    print(f"\nNeurônios rápidos (tau<0.5): {total_fast}")
    print(f"Neurônios lentos (tau>2.0): {total_slow}")
    print(f"Rápidos/Lentos: {total_fast/max(total_slow,1):.2f}")
    
    out = ROOT / "pbt_spectrum_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    t0 = time.time()
    results = run_spectrum_study()
    analyze_spectrum_results(results)
    print(f"\nTempo: {time.time()-t0:.1f}s")
