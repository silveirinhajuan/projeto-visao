#!/usr/bin/env python3
"""
continual_20tasks.py — Tarefa 19: Validação empírica — forgetting em 20+ tarefas adversariais.

Testa o VisaoBrain em 20+ tarefas sequenciais com distribuição shiftada.
Mede: (1) forward transfer, (2) backward transfer, (3) forgetting acumulado.
Compara com baseline naive e com EWC puro.
Meta: provar que Oja+EWC+Surprise supera EWC sozinho.

Uso:
    python -m experiments.continual_20tasks
    python -m experiments.continual_20tasks --n_tasks 20 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Garantir que o projeto seja importável
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visao.brain import VisaoBrain


# ==============================================================
#  GERAÇÃO DE TAREFAS
# ==============================================================

def make_task(kind: str = "sine", n: int = 500, seed: int = 0, noise: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gera tarefa sintética (regressão). Retorna (u_tr, y_tr, u_te, y_te)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    if kind == "sine":
        freq = 0.05
        phase = 0.0
        signal = np.sin(2 * np.pi * freq * t + phase)
    elif kind == "saw":
        signal = 2.0 * (t % 50) / 50.0 - 1.0
    elif kind == "square":
        signal = np.sign(np.sin(2 * np.pi * 0.03 * t))
    elif kind == "mixed":
        signal = np.sin(2 * np.pi * 0.03 * t) + 0.5 * np.cos(2 * np.pi * 0.07 * t)
    elif kind == "amp_mod":
        signal = (1 + 0.5 * np.sin(2 * np.pi * 0.01 * t)) * np.sin(2 * np.pi * 0.05 * t)
    elif kind == "freq_mod":
        signal = np.sin(2 * np.pi * (0.03 + 0.02 * np.sin(2 * np.pi * 0.005 * t)) * t)
    elif kind == "noise":
        signal = rng.normal(0, 0.5, n)
    else:
        signal = np.sin(2 * np.pi * 0.05 * t)

    u = rng.normal(0, noise, (n, 2))
    u[:, 0] += signal
    y = np.convolve(signal, np.ones(5) / 5, mode="same")[:, None]
    
    # Split: 70% train, 30% test
    split = int(0.7 * n)
    u_tr = u[:split].astype(np.float64)
    y_tr = y[:split].astype(np.float64)
    u_te = u[split:].astype(np.float64)
    y_te = y[split:].astype(np.float64)
    return u_tr, y_tr, u_te, y_te


def generate_task_suite(n_tasks: int = 20, seed: int = 0) -> list[dict]:
    """Gera n_tasks tarefas com distribuições distintas."""
    kinds = ["sine", "saw", "square", "mixed", "amp_mod", "freq_mod", "noise"]
    tasks = []
    for i in range(n_tasks):
        kind = kinds[i % len(kinds)]
        # Vary seed and noise for each task
        task_seed = seed + i * 17
        noise = 0.05 + (i % 5) * 0.03
        u_tr, y_tr, u_te, y_te = make_task(kind, n=500, seed=task_seed, noise=noise)
        tasks.append({
            "name": f"task_{i:02d}_{kind}",
            "kind": kind,
            "u_tr": u_tr,
            "y_tr": y_tr,
            "u_te": u_te,
            "y_te": y_te,
        })
    return tasks


# ==============================================================
#  EXPERIMENTO
# ==============================================================

def run_continual_experiment(
    brain: VisaoBrain,
    tasks: list[dict],
    warmup: int = 50,
) -> dict:
    """Executa experimento continual learning em múltiplas tarefas.
    
    Retorna dict com métricas de forgetting, forward/backward transfer.
    """
    n_tasks = len(tasks)
    
    # Registra performance inicial de cada tarefa (antes de treinar nela)
    initial_errors = {}
    # Registra performance após treinar todas as tarefas
    final_errors = {}
    
    # Treina sequencialmente em cada tarefa
    for i, task in enumerate(tasks):
        brain.set_mode("learn")
        
        # Avaliar ANTES de treinar nesta tarefa (forward transfer)
        brain.set_mode("infer")
        eval_before = brain.evaluate_stream(task["u_te"], task["y_te"], warmup=warmup)
        initial_errors[task["name"]] = eval_before["mse"]
        
        # Treinar nesta tarefa
        brain.set_mode("learn")
        for j in range(len(task["u_tr"])):
            brain.learn(task["u_tr"][j], task["y_tr"][j])
    
    # Após treinar todas, avaliar cada tarefa novamente
    for i, task in enumerate(tasks):
        brain.set_mode("infer")
        eval_after = brain.evaluate_stream(task["u_te"], task["y_te"], warmup=warmup)
        final_errors[task["name"]] = eval_after["mse"]
    
    # Calcular métricas
    forgetting = {}
    for name in initial_errors:
        forgetting[name] = final_errors[name] - initial_errors[name]
    
    mean_forgetting = float(np.mean(list(forgetting.values())))
    
    # Forward transfer: performance na primeira tarefa vs performance após treinar tarefas anteriores
    # Medido como: erro na tarefa i após ter treinado tarefas 0..i-1 vs erro inicial (sem treino)
    # Simplificação: comparar initial_errors (que é o erro ANTES de treinar a tarefa)
    # com o erro que teria sem nenhum treino prévio
    
    # Backward transfer: performance em tarefas antigas após treinar novas
    # = -forgetting (positive backward transfer = forgetting negativo)
    backward_transfer = -mean_forgetting
    
    return {
        "n_tasks": n_tasks,
        "mean_forgetting": mean_forgetting,
        "forgetting_per_task": forgetting,
        "initial_errors": initial_errors,
        "final_errors": final_errors,
        "backward_transfer": backward_transfer,
    }


def run_experiment_for_config(
    config_name: str,
    n_tasks: int = 20,
    n_seeds: int = 5,
    seed_start: int = 42,
) -> dict:
    """Roda experimento para uma configuração específica (full, naive, ewc_only)."""
    all_results = []
    
    for seed in range(seed_start, seed_start + n_seeds):
        tasks = generate_task_suite(n_tasks, seed=seed)
        
        if config_name == "full":
            # Oja + EWC + Surprise
            brain = VisaoBrain(
                n_in=2, n_hidden=64, n_out=1,
                consolidation=8.0, surprise_gain=3.0, oja_lr=0.0015,
                seed=seed,
            )
        elif config_name == "naive":
            # Sem nada (baseline)
            brain = VisaoBrain(
                n_in=2, n_hidden=64, n_out=1,
                consolidation=0.0, surprise_gain=0.0, oja_lr=0.0,
                seed=seed,
            )
        elif config_name == "ewc_only":
            # EWC puro (sem Oja, sem Surprise)
            brain = VisaoBrain(
                n_in=2, n_hidden=64, n_out=1,
                consolidation=8.0, surprise_gain=0.0, oja_lr=0.0,
                seed=seed,
            )
        else:
            raise ValueError(f"Config desconhecida: {config_name}")
        
        result = run_continual_experiment(brain, tasks)
        result["seed"] = seed
        all_results.append(result)
    
    # Agregar resultados
    mean_forgetting = float(np.mean([r["mean_forgetting"] for r in all_results]))
    std_forgetting = float(np.std([r["mean_forgetting"] for r in all_results]))
    
    return {
        "config": config_name,
        "n_tasks": n_tasks,
        "n_seeds": n_seeds,
        "mean_forgetting": mean_forgetting,
        "std_forgetting": std_forgetting,
        "per_seed_results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Experimento continual learning 20+ tarefas")
    parser.add_argument("--n_tasks", type=int, default=20, help="Número de tarefas")
    parser.add_argument("--seeds", type=int, default=5, help="Número de seeds")
    parser.add_argument("--output", type=str, default=None, help="Caminho para salvar JSON")
    args = parser.parse_args()
    
    print(f"Experimento: {args.n_tasks} tarefas, {args.seeds} seeds")
    print("=" * 60)
    
    configs = ["full", "ewc_only", "naive"]
    all_results = {}
    
    for config in configs:
        print(f"\nRodando config: {config}...")
        t0 = time.time()
        result = run_experiment_for_config(
            config,
            n_tasks=args.n_tasks,
            n_seeds=args.seeds,
        )
        elapsed = time.time() - t0
        all_results[config] = result
        
        print(f"  Forgetting: {result['mean_forgetting']:.4f} ± {result['std_forgetting']:.4f}")
        print(f"  Tempo: {elapsed:.1f}s")
    
    # Comparação
    print("\n" + "=" * 60)
    print("COMPARAÇÃO:")
    full_fg = all_results["full"]["mean_forgetting"]
    ewc_fg = all_results["ewc_only"]["mean_forgetting"]
    naive_fg = all_results["naive"]["mean_forgetting"]
    
    print(f"  Full (Oja+EWC+Surprise): {full_fg:.4f}")
    print(f"  EWC only:                {ewc_fg:.4f}")
    print(f"  Naive (sem nada):        {naive_fg:.4f}")
    
    if full_fg < ewc_fg:
        print(f"\n  ✅ Full supera EWC-only (delta = {ewc_fg - full_fg:.4f})")
    else:
        print(f"\n  ❌ Full NÃO supera EWC-only (delta = {full_fg - ewc_fg:.4f})")
    
    # Salvar
    output_path = args.output or str(_ROOT / "experiments" / "results_continual_20tasks.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResultados salvos em: {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()
