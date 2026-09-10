"""Metric computation for research experiments.

Provides standard metrics for regression and classification tasks,
plus continual learning metrics (forgetting, backward/forward transfer).
"""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification accuracy (works for one-hot or integer labels)."""
    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        # One-hot encoded
        true_labels = np.argmax(y_true, axis=-1)
    else:
        true_labels = y_true.ravel().astype(int)

    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        pred_labels = np.argmax(y_pred, axis=-1)
    else:
        pred_labels = y_pred.ravel().astype(int)

    return float(np.mean(true_labels == pred_labels))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_forgetting(performance_matrix: np.ndarray) -> float:
    """Compute forgetting in continual learning.

    performance_matrix[i][j] = performance on task i after training on task j.
    Forgetting = average drop from peak performance on each task.

    Args:
        performance_matrix: (n_tasks, n_tasks) array where entry [i][j] is
            accuracy/MSE on task i after training up to task j.

    Returns:
        Average forgetting across all tasks (lower is better).
    """
    n_tasks = performance_matrix.shape[0]
    if n_tasks <= 1:
        return 0.0

    forgetting = 0.0
    for i in range(n_tasks):
        # Peak performance on task i (after training on task i)
        peak = performance_matrix[i, i]
        # Final performance (after training on all tasks)
        final = performance_matrix[i, -1]
        forgetting += peak - final

    return float(forgetting / n_tasks)


def backward_transfer(performance_matrix: np.ndarray) -> float:
    """Compute Backward Transfer (BWT).

    BWT measures how much learning new tasks affects performance on old tasks.
    Negative BWT = forgetting.

    Args:
        performance_matrix: (n_tasks, n_tasks) array.

    Returns:
        Average backward transfer (negative = forgetting).
    """
    n_tasks = performance_matrix.shape[0]
    if n_tasks <= 1:
        return 0.0

    bwt = 0.0
    for i in range(n_tasks - 1):
        bwt += performance_matrix[i, -1] - performance_matrix[i, i]

    return float(bwt / (n_tasks - 1))


def forward_transfer(
    performance_matrix: np.ndarray,
    baseline_performance: np.ndarray | None = None,
) -> float:
    """Compute Forward Transfer (FWT).

    FWT measures how much prior learning helps with new tasks.

    Args:
        performance_matrix: (n_tasks, n_tasks) array.
        baseline_performance: Performance on each task without any prior learning.
            If None, uses zeros.

    Returns:
        Average forward transfer.
    """
    n_tasks = performance_matrix.shape[0]
    if n_tasks <= 1:
        return 0.0

    if baseline_performance is None:
        baseline_performance = np.zeros(n_tasks)

    fwt = 0.0
    for i in range(1, n_tasks):
        fwt += performance_matrix[i, i - 1] - baseline_performance[i]

    return float(fwt / (n_tasks - 1))


def average_accuracy(performance_matrix: np.ndarray) -> float:
    """Average accuracy after learning all tasks (last column average)."""
    return float(np.mean(performance_matrix[:, -1]))


def learning_curve_area(losses: list[float]) -> float:
    """Area under the learning curve (cumulative sum)."""
    return float(np.sum(losses))


def get_metric(name: str):
    """Get a metric function by name."""
    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy,
        "r2": r2_score,
    }
    if name not in metrics:
        raise ValueError(f"Unknown metric: {name}. Available: {list(metrics)}")
    return metrics[name]


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "regression",
) -> dict[str, float]:
    """Compute all relevant metrics for a task type."""
    if task_type == "regression":
        return {
            "mse": mse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
    else:  # classification
        return {
            "accuracy": accuracy(y_true, y_pred),
            "error_rate": 1.0 - accuracy(y_true, y_pred),
        }
