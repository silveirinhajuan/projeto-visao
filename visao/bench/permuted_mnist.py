#!/usr/bin/env python3
"""
permuted_mnist.py — Tarefa 73: Permuted-MNIST benchmark padrão.

Permuted-MNIST é o benchmark mais usado em continual learning.
20 tarefas com permutações diferentes dos pixels MNIST.
Cada tarefa compartilha os mesma labels (10 classes) mas tem
distribuição de entrada diferente.

Referências:
  - Kirkpatrick et al. 2017 (EWC paper) — introduziu Permuted-MNIST
  - Zenke et al. 2017 (SI paper)
  - Lopez-Paz & Ranzato 2017 (GEM paper)
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visao.brain import VisaoBrain


# ==============================================================
#  DATA LOADING
# ==============================================================

def load_mnist(images_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST from gzipped files."""
    with gzip.open(images_path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    with gzip.open(labels_path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels


def generate_permutations(n_tasks: int = 20, n_pixels: int = 784,
                          seed: int = 42) -> list[np.ndarray]:
    """Generate deterministic random permutations for each task.
    
    Uses a fixed seed for reproducibility across runs.
    Each permutation is a random shuffling of pixel indices.
    """
    rng = np.random.default_rng(seed)
    perms = []
    for _ in range(n_tasks):
        perms.append(rng.permutation(n_pixels))
    return perms


def load_permuted_mnist(data_dir: str, n_tasks: int = 20,
                        max_per_task: int = 2000,
                        seed: int = 42) -> list[dict]:
    """Load MNIST and create Permuted-MNIST tasks.
    
    Each task applies a different fixed permutation to all images.
    All tasks share the same 10-class labels.
    
    Returns list of dicts, each with:
      - name: task name (e.g., "task_0")
      - permutation: pixel permutation array
      - X_train, y_train: training data (permuted, normalized)
      - X_test, y_test: test data (permuted, normalized)
    """
    data_dir = Path(data_dir)
    train_images, train_labels = load_mnist(
        str(data_dir / "train-images-idx3-ubyte.gz"),
        str(data_dir / "train-labels-idx1-ubyte.gz"),
    )
    test_images, test_labels = load_mnist(
        str(data_dir / "t10k-images-idx3-ubyte.gz"),
        str(data_dir / "t10k-labels-idx1-ubyte.gz"),
    )
    
    # Flatten and normalize
    train_X = train_images.reshape(-1, 784).astype(np.float32) / 255.0
    test_X = test_images.reshape(-1, 784).astype(np.float32) / 255.0
    
    # Generate permutations
    perms = generate_permutations(n_tasks, 784, seed)
    
    tasks = []
    for i, perm in enumerate(perms):
        # Apply permutation to all images
        X_tr = train_X[:max_per_task][:, perm]
        y_tr = train_labels[:max_per_task].astype(np.float32)
        
        X_te = test_X[:max_per_task // 5][:, perm]
        y_te = test_labels[:max_per_task // 5].astype(np.float32)
        
        tasks.append({
            "name": f"task_{i}",
            "task_id": i,
            "permutation": perm,
            "X_train": X_tr,
            "y_train": y_tr,
            "X_test": X_te,
            "y_test": y_te,
        })
    
    return tasks


# ==============================================================
#  RESERVOIR SETUP
# ==============================================================

def create_brain(n_in: int, n_hidden: int, n_out: int, seed: int = 0,
                 spectral_radius: float = 0.95, sparsity: float = 0.3,
                 dt: float = 0.1, lr: float = 0.1,
                 consolidation: float = 8.0,
                 surprise_gain: float = 0.0,
                 oja_lr: float = 0.001) -> VisaoBrain:
    """Create a VisaoBrain with reservoir tuned for classification."""
    brain = VisaoBrain(
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out,
        sparsity=sparsity,
        tau_min=0.4,
        tau_max=4.0,
        dt=dt,
        lr=lr,
        oja_lr=oja_lr,
        consolidation=consolidation,
        surprise_gain=surprise_gain,
        task_type="classification",
        seed=seed,
    )
    
    # Rescale W_rec to target spectral radius
    cell = brain.cell
    eigs = np.linalg.eigvals(cell.W_rec)
    current_sr = np.max(np.abs(eigs))
    if current_sr > 0:
        scale = spectral_radius / current_sr
        cell.W_rec *= scale
    
    return brain


# ==============================================================
#  EXPERIMENT
# ==============================================================

def evaluate_task(brain: VisaoBrain, task: dict, n_steps: int = 10) -> float:
    """Evaluate brain on a single task. Returns accuracy.
    
    For multi-class: uses argmax of 10 output units.
    """
    brain.set_mode("infer")
    
    correct = 0
    total = 0
    for xi, yi in zip(task["X_test"], task["y_test"]):
        brain.reset_state()
        pred = None
        for _ in range(n_steps):
            pred = brain.forward(xi.reshape(-1, 1))
        if pred is None:
            continue
        pred_class = int(np.argmax(pred))
        true_class = int(yi)
        if pred_class == true_class:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def train_on_task(brain: VisaoBrain, task: dict, n_epochs: int = 5,
                     n_steps: int = 10) -> None:
    """Train brain on a single task using settle-then-learn pattern.
    
    For multi-class: one-hot encodes the label for n_out=10.
    """
    X = task["X_train"]
    y = task["y_train"]
    
    for epoch in range(n_epochs):
        rng = np.random.default_rng(epoch)
        idx = rng.permutation(len(X))
        for i in idx:
            brain.reset_state()
            # Settle: run reservoir forward without learning
            for step in range(n_steps):
                brain.settle(X[i].reshape(-1, 1))
            # Learn only on final settled state
            brain.set_mode("learn")
            # One-hot encode label
            label_onehot = np.zeros(brain.n_out)
            label_onehot[int(y[i])] = 1.0
            brain.learn(X[i].reshape(-1, 1), label_onehot, step_reservoir=False)


def run_permuted_mnist_experiment(
    brain: VisaoBrain,
    tasks: list,
    n_epochs: int = 5,
    n_steps: int = 10,
) -> dict:
    """Run Permuted-MNIST continual learning experiment.
    
    Trains on each task sequentially, measuring accuracy on ALL tasks
    after each task. This reveals both learning and forgetting.
    
    Returns dict with:
      - accuracy_matrix: accuracy[i][j] = accuracy on task j after training on task i
      - mean_accuracy: average accuracy on all tasks after last training
      - forgetting: average forgetting across tasks
      - final_accuracies: accuracy on each task after all training
    """
    n_tasks = len(tasks)
    accuracy_matrix = np.zeros((n_tasks, n_tasks))
    
    for task_idx, task in enumerate(tasks):
        train_on_task(brain, task, n_epochs=n_epochs, n_steps=n_steps)
        
        # Evaluate on ALL tasks (including previous)
        for eval_idx, eval_task in enumerate(tasks):
            acc = evaluate_task(brain, eval_task, n_steps=n_steps)
            accuracy_matrix[task_idx][eval_idx] = acc
    
    # Compute metrics
    final_accuracies = accuracy_matrix[-1]
    mean_accuracy = float(np.mean(final_accuracies))
    
    # Forgetting: for each task, how much did we lose from peak?
    forgetting_per_task = []
    for j in range(n_tasks):
        peak = max(accuracy_matrix[i][j] for i in range(j, n_tasks))
        final = accuracy_matrix[-1][j]
        forgetting_per_task.append(peak - final)
    
    forgetting = float(np.mean(forgetting_per_task))
    
    return {
        "accuracy_matrix": accuracy_matrix.tolist(),
        "mean_accuracy": mean_accuracy,
        "forgetting": forgetting,
        "forgetting_per_task": forgetting_per_task,
        "final_accuracies": final_accuracies.tolist(),
    }


def run_experiment_for_config(
    config_name: str,
    n_seeds: int = 3,
    n_tasks: int = 20,
    max_per_task: int = 2000,
    n_epochs: int = 5,
    n_steps: int = 10,
    seed_start: int = 0,
) -> dict:
    """Run experiment for a specific configuration.
    
    Configs:
      - "visao": full VisaoBrain
      - "naive": no consolidation (consolidation=0)
      - "ewc_only": EWC without Oja or surprise
    """
    data_dir = str(Path(__file__).parent / "data")
    tasks = load_permuted_mnist(data_dir, n_tasks=n_tasks,
                                max_per_task=max_per_task)
    
    all_results = []
    
    for seed in range(n_seeds):
        rng_seed = seed_start + seed
        
        if config_name == "visao":
            brain = create_brain(
                n_in=784, n_hidden=256, n_out=10,
                seed=rng_seed, spectral_radius=0.95, lr=0.1,
            )
        elif config_name == "naive":
            brain = create_brain(
                n_in=784, n_hidden=256, n_out=10,
                seed=rng_seed, spectral_radius=0.95, lr=0.1,
            )
            brain.learner.consolidation = 0.0
            brain.learner.surprise_gain = 0.0
            brain.learner.oja_lr = 0.0
        elif config_name == "ewc_only":
            brain = create_brain(
                n_in=784, n_hidden=256, n_out=10,
                seed=rng_seed, spectral_radius=0.95, lr=0.1,
            )
            brain.learner.surprise_gain = 0.0
            brain.learner.oja_lr = 0.0
        else:
            raise ValueError(f"Unknown config: {config_name}")
        
        result = run_permuted_mnist_experiment(brain, tasks, n_epochs=n_epochs,
                                               n_steps=n_steps)
        all_results.append(result)
    
    # Aggregate
    mean_acc = float(np.mean([r["mean_accuracy"] for r in all_results]))
    mean_acc_std = float(np.std([r["mean_accuracy"] for r in all_results]))
    mean_forgetting = float(np.mean([r["forgetting"] for r in all_results]))
    mean_forgetting_std = float(np.std([r["forgetting"] for r in all_results]))
    
    return {
        "config": config_name,
        "n_seeds": n_seeds,
        "n_tasks": n_tasks,
        "mean_accuracy": mean_acc,
        "mean_accuracy_std": mean_acc_std,
        "mean_forgetting": mean_forgetting,
        "mean_forgetting_std": mean_forgetting_std,
        "per_seed_results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Permuted-MNIST Benchmark — Task 73")
    parser.add_argument("--quick", action="store_true", help="Fast mode (fewer samples)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--tasks", type=int, default=20, help="Number of tasks")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task")
    parser.add_argument("--steps", type=int, default=10, help="Reservoir steps per image")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    max_per_task = 500 if args.quick else 2000
    
    print("=" * 60)
    print("PERMUTED-MNIST BENCHMARK — Task 73")
    print("=" * 60)
    
    configs = ["visao", "naive", "ewc_only"]
    all_results = {}
    
    for config in configs:
        print(f"\n--- {config.upper()} ---")
        result = run_experiment_for_config(
            config,
            n_seeds=args.seeds,
            n_tasks=args.tasks,
            max_per_task=max_per_task,
            n_epochs=args.epochs,
            n_steps=args.steps,
        )
        all_results[config] = result
        print(f"  Accuracy: {result['mean_accuracy']:.3f} ± {result['mean_accuracy_std']:.3f}")
        print(f"  Forgetting: {result['mean_forgetting']:.3f} ± {result['mean_forgetting_std']:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<15} {'Accuracy':<20} {'Forgetting':<15}")
    print("-" * 50)
    for config, r in all_results.items():
        print(f"{config:<15} {r['mean_accuracy']:.3f} ± {r['mean_accuracy_std']:.3f}     {r['mean_forgetting']:.3f} ± {r['mean_forgetting_std']:.3f}")
    
    # Save
    if args.output is None:
        args.output = str(Path(__file__).parent / "results_permuted_mnist.json")
    Path(args.output).write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
