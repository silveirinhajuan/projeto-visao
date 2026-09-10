"""Split-MNIST benchmark dataset generator.

Downloads MNIST via sklearn.datasets.fetch_openml and splits into 5 binary
continual learning tasks: (0/1), (2/3), (4/5), (6/7), (8/9).

Usage:
    python research/datasets/split_mnist.py
    python research/datasets/split_mnist.py --output data/split_mnist.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# 5 binary tasks: each is a tuple of digit classes
TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def download_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Download MNIST from OpenML via sklearn.

    Returns:
        X: (70000, 784) float64 array, pixel values in [0, 255]
        y: (70000,) int array of digit labels
    """
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    return X.astype(np.float64), y.astype(int)


def split_into_tasks(
    X: np.ndarray,
    Y: np.ndarray,
    tasks: list[tuple[int, int]] | None = None,
    normalize: bool = True,
    binary: bool = True,
) -> dict:
    """Split MNIST into binary classification tasks.

    Args:
        X: (N, 784) pixel data
        Y: (N,) digit labels
        tasks: list of (digit_a, digit_b) pairs
        normalize: scale pixel values to [0, 1]
        binary: convert labels to {0, 1} per task

    Returns:
        dict with keys:
            'tasks': list of (X_task, Y_task) tuples
            'n_tasks': number of tasks
            'task_definitions': list of (d1, d2) tuples
            'input_dim': 784
            'n_classes_per_task': 2
    """
    if tasks is None:
        tasks = TASKS
    task_data = []
    for d1, d2 in tasks:
        mask = (Y == d1) | (Y == d2)
        X_task = X[mask].copy()
        Y_task = Y[mask].copy()
        if binary:
            Y_task = (Y_task == d2).astype(np.float64)
        if normalize:
            X_task /= 255.0
        task_data.append((X_task, Y_task))

    return {
        "tasks": task_data,
        "n_tasks": len(tasks),
        "task_definitions": tasks,
        "input_dim": int(X.shape[1]),
        "n_classes_per_task": 2,
    }


def save_split_mnist(path: str | Path) -> Path:
    """Download, split, and save Split-MNIST to .npz.

    Returns:
        Path to the saved .npz file
    """
    X, Y = download_mnist()
    result = split_into_tasks(X, Y)

    path = Path(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    # Save as .npz
    save_dict = {
        "input_dim": np.array(result["input_dim"]),
        "n_tasks": np.array(result["n_tasks"]),
    }
    for i, (Xt, Yt) in enumerate(result["tasks"]):
        save_dict[f"task_{i}_X"] = Xt
        save_dict[f"task_{i}_Y"] = Yt

    np.savez_compressed(path, **save_dict)
    print(f"Saved Split-MNIST to {path}")
    print(f"  Input dim: {result['input_dim']}")
    print(f"  Tasks: {result['n_tasks']}")
    for i, (Xt, Yt) in enumerate(result["tasks"]):
        d1, d2 = result["task_definitions"][i]
        print(f"  Task {i} (digits {d1}/{d2}): {len(Xt)} samples")

    return path


def load_split_mnist(path: str | Path) -> dict:
    """Load Split-MNIST from .npz file."""
    data = np.load(path)
    n_tasks = int(data["n_tasks"])
    tasks = []
    task_defs = []
    for i in range(n_tasks):
        tasks.append((data[f"task_{i}_X"], data[f"task_{i}_Y"]))
        task_defs.append(TASKS[i])
    return {
        "tasks": tasks,
        "n_tasks": n_tasks,
        "task_definitions": task_defs,
        "input_dim": int(data["input_dim"]),
        "n_classes_per_task": 2,
    }


def get_split_mnist_tasks(
    n_train_per_task: int = 500,
    n_test_per_task: int = 200,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Get Split-MNIST as list of (X_train, Y_train, X_test, Y_test) per task.

    Args:
        n_train_per_task: number of training samples per task
        n_test_per_task: number of test samples per task
        seed: random seed for shuffling

    Returns:
        List of 5 tuples (X_train, Y_train, X_test, Y_test)
    """
    data_path = Path(__file__).parent / "data" / "split_mnist.npz"

    if not data_path.exists():
        save_split_mnist(data_path)

    data = load_split_mnist(data_path)
    rng = np.random.default_rng(seed)

    all_tasks = []
    for Xt, Yt in data["tasks"]:
        n = len(Xt)
        perm = rng.permutation(n)
        Xt, Yt = Xt[perm], Yt[perm]
        X_train = Xt[:n_train_per_task]
        Y_train = Yt[:n_train_per_task].reshape(-1, 1)
        X_test = Xt[n_train_per_task:n_train_per_task + n_test_per_task]
        Y_test = Yt[n_train_per_task:n_train_per_task + n_test_per_task].reshape(-1, 1)
        all_tasks.append((X_train, Y_train, X_test, Y_test))

    return all_tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and split MNIST for CL benchmark")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="research/datasets/data/split_mnist.npz",
        help="Output path for .npz file",
    )
    args = parser.parse_args()
    save_split_mnist(args.output)
