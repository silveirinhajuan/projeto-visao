"""Visualization utilities for research experiments.

Provides plotting functions for learning curves, metric comparisons,
and continual learning performance matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_learning_curve(
    losses: Sequence[float],
    title: str = "Learning Curve",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Plot a single learning curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_comparison_bars(
    metrics: dict[str, dict[str, float]],
    metric_name: str = "mse",
    title: str = "Model Comparison",
    ylabel: str = "MSE",
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Plot bar chart comparing models on a single metric.

    Args:
        metrics: {model_name: {metric_name: value}}
        metric_name: Which metric to compare.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(metrics.keys())
    values = [metrics[m].get(metric_name, 0) for m in models]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))

    bars = ax.bar(models, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_continual_matrix(
    performance_matrix: np.ndarray,
    task_names: Optional[list[str]] = None,
    title: str = "Continual Learning Performance",
    cmap: str = "RdYlGn",
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Plot heatmap of continual learning performance matrix.

    Args:
        performance_matrix: (n_tasks, n_tasks) array.
    """
    n_tasks = performance_matrix.shape[0]
    if task_names is None:
        task_names = [f"Task {i}" for i in range(n_tasks)]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(performance_matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(n_tasks))
    ax.set_yticks(range(n_tasks))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_yticklabels(task_names)
    ax.set_xlabel("After training on")
    ax.set_ylabel("Performance on")
    ax.set_title(title)

    # Add text annotations
    for i in range(n_tasks):
        for j in range(n_tasks):
            val = performance_matrix[i, j]
            color = "white" if val < (performance_matrix.max() + performance_matrix.min()) / 2 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_metric_history(
    histories: dict[str, list[float]],
    title: str = "Metric History",
    xlabel: str = "Step",
    ylabel: str = "Value",
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Plot multiple metric histories on the same axes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, values in histories.items():
        ax.plot(values, label=name, linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_confidence_intervals(
    results: dict[str, dict[str, float]],
    metric_name: str = "mse",
    title: str = "95% Confidence Intervals",
    save_path: Optional[str | Path] = None,
    show: bool = False,
) -> None:
    """Plot confidence intervals for model comparisons.

    Args:
        results: {model_name: {metric_name: {mean, ci_lower, ci_upper}}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results.keys())
    means = [results[m].get(metric_name, {}).get("mean", 0) for m in models]
    ci_lower = [results[m].get(metric_name, {}).get("ci_lower", 0) for m in models]
    ci_upper = [results[m].get(metric_name, {}).get("ci_upper", 0) for m in models]

    errors = [np.array(means) - np.array(ci_lower), np.array(ci_upper) - np.array(means)]

    ax.errorbar(range(len(models)), means, yerr=errors, fmt="o", capsize=5, capthick=2)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric_name.upper())
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
