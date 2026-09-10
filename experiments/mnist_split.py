#!/usr/bin/env python3
"""mnist_split.py — Tarefa 7.4b: MNIST split com Oja + consolidação."""

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


def load_mnist():
    """Load MNIST from sklearn or local cache."""
    try:
        from sklearn.datasets import fetch_openml
        print("Baixando MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data / 255.0, mnist.target.astype(int)
        return X, y
    except Exception as e:
        print(f"Erro ao carregar MNIST: {e}")
        return None, None


def create_split_mnist_tasks(X, y, n_tasks=5):
    """Cria tarefas: task 0 = digits 0-1, task 1 = digits 2-3, etc."""
    tasks = []
    digits_per_task = 10 // n_tasks
    
    for task_id in range(n_tasks):
        start_digit = task_id * digits_per_task
        end_digit = start_digit + digits_per_task
        mask = (y >= start_digit) & (y < end_digit)
        
        X_task = X[mask]
        y_task = y[mask] - start_digit  # labels 0-1 por tarefa
        
        # Split train/test
        n = len(X_task)
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        
        tasks.append({
            "X_tr": X_task[indices[:split]],
            "y_tr": y_task[indices[:split]],
            "X_te": X_task[indices[split:]],
            "y_te": y_task[indices[split:]],
        })
    
    return tasks


def run_mnist_split(seed, n_tasks=5):
    """Executa baseline em MNIST split."""
    X, y = load_mnist()
    if X is None:
        return None
    
    np.random.seed(seed)
    tasks = create_split_mnist_tasks(X, y, n_tasks=n_tasks)
    
    # Codificar MNIST como sequência: 28 passos de 28 dimensões (linhas)
    n_in = 28  # dimensão de cada linha
    n_hidden = 128
    n_out = 2  # 2 classes por tarefa
    
    rng = np.random.default_rng(seed)
    cell = LiquidCell(n_in, n_hidden, sparsity=0.6, tau_min=0.4, tau_max=4.0, dt=0.1, rng=rng)
    learner = LocalLearner(n_hidden, n_out, rng=rng)
    
    history = {}
    for task_id, task in enumerate(tasks):
        # Converter para sequência: (N, 784) -> (N, 28, 28)
        X_tr_seq = task["X_tr"].reshape(-1, 28, 28)
        X_te_seq = task["X_te"].reshape(-1, 28, 28)
        
        # Eval antes
        acc_before = evaluate_mnist(cell, learner, X_te_seq, task["y_te"])
        
        # Treinar
        for epoch in range(3):
            for i in range(len(X_tr_seq)):
                x = np.zeros(n_hidden)
                x_prev = np.zeros(n_hidden)
                for t in range(28):
                    x_prev = x.copy()
                    x, _ = cell.step(x, X_tr_seq[i, t])
                    if t >= 10:
                        target = np.zeros(n_out)
                        target[task["y_tr"][i]] = 1
                        learner.update(x, target)
                        learner.oja_update(cell, x_prev, x)
                        x = np.clip(x, -10, 10)
        
        acc_after = evaluate_mnist(cell, learner, X_te_seq, task["y_te"])
        history[f"task_{task_id}"] = {"before": acc_before, "after": acc_after}
    
    # Forgetting
    forgetting = {}
    for task_id in range(n_tasks - 1):
        X_te_seq = tasks[task_id]["X_te"].reshape(-1, 28, 28)
        cur = evaluate_mnist(cell, learner, X_te_seq, tasks[task_id]["y_te"])
        forgetting[f"task_{task_id}"] = cur - history[f"task_{task_id}"]["after"]
    
    return {
        "forgetting": forgetting,
        "mean_forgetting": float(np.mean(list(forgetting.values()))),
        "mean_final_accuracy": float(np.mean([h["after"] for h in history.values()])),
    }


def evaluate_mnist(cell, learner, X_seq, y):
    """Avalia accuracy em MNIST (sequência)."""
    correct = 0
    total = len(X_seq)
    
    for i in range(total):
        x = np.zeros(cell.n_hidden)
        for t in range(28):
            x, _ = cell.step(x, X_seq[i, t])
        pred = np.argmax(learner.W_out @ x + learner.b_out)
        if pred == y[i]:
            correct += 1
    
    return correct / total


def run_experiment():
    print("="*70)
    print("MNIST SPLIT: Oja + consolidação em domínio visual")
    print("="*70)
    
    results = []
    for seed in range(5):
        print(f"\n=== Seed {seed} ===")
        r = run_mnist_split(seed, n_tasks=5)
        if r:
            results.append(r)
            print(f"  fg={r['mean_forgetting']:.4f}, acc={r['mean_final_accuracy']:.4f}")
    
    if results:
        fg_vals = [r["mean_forgetting"] for r in results]
        acc_vals = [r["mean_final_accuracy"] for r in results]
        
        print("\n" + "="*70)
        print("RESULTADO MNIST SPLIT")
        print("="*70)
        print(f"Forgetting: {np.mean(fg_vals):.4f} ± {np.std(fg_vals)/np.sqrt(len(fg_vals)):.4f}")
        print(f"Accuracy:   {np.mean(acc_vals):.4f} ± {np.std(acc_vals)/np.sqrt(len(acc_vals)):.4f}")
        
        out = ROOT / "mnist_split_results.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSalvo em: {out}")


if __name__ == "__main__":
    t0 = time.time()
    run_experiment()
    print(f"\nTempo: {time.time()-t0:.1f}s")
