"""Dataset generators for reproducible experiments.

All generators return (X_train, Y_train, X_test, Y_test) as numpy arrays.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class RandomState:
    """Thin wrapper for reproducible RNG."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def standardize(self, X: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        if mean is None:
            mean = X.mean(axis=0, keepdims=True)
        if std is None:
            std = X.std(axis=0, keepdims=True) + 1e-8
        return (X - mean) / std, mean, std


def sine_regression(
    n_train: int = 500,
    n_test: int = 200,
    n_inputs: int = 1,
    noise: float = 0.1,
    freq: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sine wave regression: x(t) -> sin(2*pi*freq*t) + noise."""
    rng = np.random.default_rng(seed)
    t_train = rng.uniform(0, 1, size=(n_train, n_inputs))
    t_test = np.linspace(0, 1, n_test).reshape(-1, n_inputs)

    Y_train = np.sin(2 * np.pi * freq * t_train) + noise * rng.standard_normal((n_train, n_inputs))
    Y_test = np.sin(2 * np.pi * freq * t_test)

    return t_train, Y_train, t_test, Y_test


def mackey_glass(
    n_train: int = 500,
    n_test: int = 200,
    tau: float = 17.0,
    dt: float = 1.0,
    horizon: int = 1,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mackey-Glass chaotic time series prediction."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test + 50  # burn-in

    # Mackey-Glass via Euler
    x = np.zeros(n_total)
    x[:20] = 1.2 + 0.2 * rng.standard_normal(20)

    for t in range(20, n_total - 1):
        dx = (0.2 * x[t - int(tau / dt)]) / (1 + x[t - int(tau / dt)] ** 10) - 0.1 * x[t]
        x[t + 1] = x[t] + dx * dt

    # Build prediction task: x[t] -> x[t+horizon]
    X = np.zeros((n_total - horizon, 1))
    Y = np.zeros((n_total - horizon, 1))
    for i in range(n_total - horizon):
        X[i, 0] = x[i]
        Y[i, 0] = x[i + horizon] + noise * rng.standard_normal()

    # Standardize
    mean, std = X[:n_train].mean(), X[:n_train].std() + 1e-8
    X = (X - mean) / std
    Y = (Y - mean) / std

    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def classification_task(
    n_train: int = 500,
    n_test: int = 200,
    n_features: int = 2,
    n_classes: int = 3,
    n_clusters_per_class: int = 1,
    noise: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Multi-class classification with Gaussian clusters."""
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    # Generate cluster centers
    centers = rng.standard_normal((n_classes, n_features)) * 3.0
    samples_per_class = n_total // n_classes

    X = np.zeros((n_total, n_features))
    Y = np.zeros((n_total, 1))

    idx = 0
    for c in range(n_classes):
        end_idx = idx + samples_per_class
        X[idx:end_idx] = centers[c] + noise * rng.standard_normal((samples_per_class, n_features))
        Y[idx:end_idx] = c
        idx = end_idx

    # Shuffle
    perm = rng.permutation(idx)
    X, Y = X[perm], Y[:idx][perm]

    # Standardize
    mean, std = X[:n_train].mean(axis=0), X[:n_train].std(axis=0) + 1e-8
    X = (X - mean) / std

    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def split_continual_tasks(
    X: np.ndarray,
    Y: np.ndarray,
    n_tasks: int = 5,
    shuffle: bool = True,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a dataset into sequential tasks for continual learning."""
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    task_size = n // n_tasks
    tasks = []
    for t in range(n_tasks):
        start = t * task_size
        end = start + task_size if t < n_tasks - 1 else n
        task_idx = indices[start:end]
        tasks.append((X[task_idx], Y[task_idx]))
    return tasks


def get_generator(name: str) -> Callable:
    """Get a dataset generator by name."""
    generators = {
        "sine_regression": sine_regression,
        "mackey_glass": mackey_glass,
        "classification": classification_task,
    }
    if name not in generators:
        raise ValueError(f"Unknown dataset generator: {name}. Available: {list(generators)}")
    return generators[name]
