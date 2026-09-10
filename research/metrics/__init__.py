"""Metrics computation for continual learning benchmarks."""
from __future__ import annotations

import numpy as np


def mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))


def mse_per_task(mse_values: list[float]) -> dict:
    """MSE broken down by task."""
    return {f"mse_task{i}": v for i, v in enumerate(mse_values)}


def forgetting(mse_initial: list, mse_final: list) -> float:
    """Forgetting: how much performance degraded on old tasks.

    Positive = forgot something. Negative = improved (rare).
    """
    mse_initial = np.asarray(mse_initial, dtype=np.float64)
    mse_final = np.asarray(mse_final, dtype=np.float64)
    # Forgetting = max(0, final - initial) averaged over tasks
    return float(np.mean(np.maximum(0, mse_final - mse_initial)))


def backward_transfer(mse_after: list, mse_before: list) -> float:
    """Backward transfer: effect of new learning on old tasks.

    Positive = improved old tasks. Negative = catastrophic forgetting.
    """
    mse_after = np.asarray(mse_after, dtype=np.float64)
    mse_before = np.asarray(mse_before, dtype=np.float64)
    return float(np.mean(mse_after - mse_before))


def forward_transfer(mse_with_prior: float, mse_without_prior: float) -> float:
    """Forward transfer: does learning old tasks help new ones?

    Positive = prior learning helped.
    """
    return float(mse_with_prior - mse_without_prior)


def compute_all_metrics(
    mse_per_task_history: list,
    mse_final_per_task: list,
    wall_time: float,
    n_params: int,
) -> dict:
    """Compute all benchmark metrics from per-task MSE history.

    Args:
        mse_per_task_history: MSE on each task right after training on it.
        mse_final_per_task: MSE on each task at the end of training.
        wall_time: total training time in seconds.
        n_params: number of model parameters.
    """
    mse_total = float(np.mean(mse_final_per_task))
    mse_final = mse_total

    # Forgetting: degradation from best to final
    fgt = forgetting(mse_per_task_history, mse_final_per_task)

    # Backward transfer: average change in old task performance
    bwt = backward_transfer(mse_final_per_task, mse_per_task_history)

    return {
        "mse_total": mse_total,
        "mse_final": mse_final,
        "forgetting": fgt,
        "backward_transfer": bwt,
        "forward_transfer": 0.0,  # requires separate experiment
        "wall_time": wall_time,
        "n_params": n_params,
        "mse_per_task": mse_final_per_task,
        "mse_per_task_history": mse_per_task_history,
    }
