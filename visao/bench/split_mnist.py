#!/usr/bin/env python3
"""
split_mnist.py — Tarefa 66: Split-MNIST benchmark.

Split-MNIST is the canonical continual learning benchmark:
  - MNIST digits split into 5 binary classification tasks: (0,1), (2,3), (4,5), (6,7), (8,9)
  - Model learns tasks sequentially
  - Measure: accuracy on each task after learning it, and forgetting of previous tasks

This validates VisaoBrain against the standard benchmark used in the
continual learning literature (Kirkpatrick et al. 2017, Zenke et al. 2017, etc.).

Artefatos:
  - visao/bench/split_mnist.py (this file)
  - visao/tests/test_split_mnist.py (TDD)
  - visao/bench/results_split_mnist.json (output)
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
import time
import tracemalloc
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


def load_split_mnist(data_dir, max_per_task: int = 2000) -> list:
    """Load and split MNIST into 5 binary classification tasks.
    
    Returns list of dicts, each with:
      - name: task name (e.g., "task_0_01")
      - digit_a, digit_b: the two digits
      - X_train, y_train: training data (flattened, normalized)
      - X_test, y_test: test data
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
    
    tasks = []
    splits = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    
    for digit_a, digit_b in splits:
        # Training data
        mask_tr = (train_labels == digit_a) | (train_labels == digit_b)
        X_tr = train_X[mask_tr][:max_per_task]
        y_tr_raw = train_labels[mask_tr][:max_per_task]
        # Binary labels: 0 for digit_a, 1 for digit_b
        y_tr = (y_tr_raw == digit_b).astype(np.float32)
        
        # Test data
        mask_te = (test_labels == digit_a) | (test_labels == digit_b)
        X_te = test_X[mask_te][:max_per_task // 5]
        y_te_raw = test_labels[mask_te][:max_per_task // 5]
        y_te = (y_te_raw == digit_b).astype(np.float32)
        
        tasks.append({
            "name": f"task_{digit_a}{digit_b}",
            "digit_a": digit_a,
            "digit_b": digit_b,
            "X_train": X_tr,
            "y_train": y_tr,
            "X_test": X_te,
            "y_test": y_te,
        })
    
    return tasks


# ==============================================================
#  EXPERIMENT
# ==============================================================

def evaluate_task(brain: VisaoBrain, task: dict, warmup: int = 50) -> float:
    """Evaluate brain on a single task. Returns accuracy."""
    brain.set_mode("infer")
    
    correct = 0
    total = 0
    for xi, yi in zip(task["X_test"], task["y_test"]):
        # Reset state for each new sequence
        brain.reset_state()
        # Feed sequence one timestep at a time (psMNIST style)
        # xi has shape (784,), each pixel is a timestep with shape (1, 1)
        pred = None
        for step in range(xi.shape[0]):
            pred = brain.forward(xi[step].reshape(-1, 1))
        if pred is None:
            continue
        pred_class = int(pred[0] > 0.5)
        true_class = int(yi)
        if pred_class == true_class:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def train_on_task(brain: VisaoBrain, task: dict, n_epochs: int = 1) -> None:
    """Train brain on a single task."""
    brain.set_mode("learn")
    X = task["X_train"]
    y = task["y_train"]
    
    for epoch in range(n_epochs):
        # Shuffle
        rng = np.random.default_rng(epoch)
        idx = rng.permutation(len(X))
        for i in idx:
            # Reset state for each new sequence
            brain.reset_state()
            # Feed each pixel as a timestep (psMNIST)
            for step in range(X.shape[1]):
                brain.learn(X[i, step].reshape(-1, 1), np.array([y[i]]))


def run_split_mnist_experiment(
    brain: VisaoBrain,
    tasks: list,
    n_epochs: int = 1,
) -> dict:
    """Run Split-MNIST continual learning experiment.
    
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
        train_on_task(brain, task, n_epochs=n_epochs)
        
        # Evaluate on ALL tasks (including previous)
        for eval_idx, eval_task in enumerate(tasks):
            acc = evaluate_task(brain, eval_task)
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
    max_per_task: int = 2000,
    n_epochs: int = 1,
    seed_start: int = 0,
) -> dict:
    """Run experiment for a specific configuration.
    
    Configs:
      - "visao": full VisaoBrain
      - "naive": no consolidation (consolidation=0)
      - "ewc_only": EWC without Oja or surprise
    """
    data_dir = str(Path(__file__).parent / "data")
    tasks = load_split_mnist(data_dir, max_per_task=max_per_task)
    
    all_results = []
    
    for seed in range(n_seeds):
        rng_seed = seed_start + seed
        
        if config_name == "visao":
            brain = VisaoBrain(
                n_in=1, n_hidden=128, n_out=1,
                consolidation=8.0, surprise_gain=3.0, oja_lr=0.0015,
                seed=rng_seed,
            )
        elif config_name == "naive":
            brain = VisaoBrain(
                n_in=1, n_hidden=128, n_out=1,
                consolidation=0.0, surprise_gain=0.0, oja_lr=0.0,
                seed=rng_seed,
            )
        elif config_name == "ewc_only":
            brain = VisaoBrain(
                n_in=1, n_hidden=128, n_out=1,
                consolidation=8.0, surprise_gain=0.0, oja_lr=0.0,
                seed=rng_seed,
            )
        else:
            raise ValueError(f"Unknown config: {config_name}")
        
        result = run_split_mnist_experiment(brain, tasks, n_epochs=n_epochs)
        all_results.append(result)
    
    # Aggregate
    mean_acc = float(np.mean([r["mean_accuracy"] for r in all_results]))
    mean_acc_std = float(np.std([r["mean_accuracy"] for r in all_results]))
    mean_forgetting = float(np.mean([r["forgetting"] for r in all_results]))
    mean_forgetting_std = float(np.std([r["forgetting"] for r in all_results]))
    
    return {
        "config": config_name,
        "n_seeds": n_seeds,
        "mean_accuracy": mean_acc,
        "mean_accuracy_std": mean_acc_std,
        "mean_forgetting": mean_forgetting,
        "mean_forgetting_std": mean_forgetting_std,
        "per_seed_results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Split-MNIST Benchmark — Task 66")
    parser.add_argument("--quick", action="store_true", help="Fast mode (fewer samples)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per task")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    max_per_task = 500 if args.quick else 2000
    
    print("=" * 60)
    print("SPLIT-MNIST BENCHMARK — Task 66")
    print("=" * 60)
    
    configs = ["visao", "naive", "ewc_only"]
    all_results = {}
    
    for config in configs:
        print(f"\n--- {config.upper()} ---")
        result = run_experiment_for_config(
            config,
            n_seeds=args.seeds,
            max_per_task=max_per_task,
            n_epochs=args.epochs,
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
        args.output = str(Path(__file__).parent / "results_split_mnist.json")
    Path(args.output).write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
