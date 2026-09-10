"""Visualization utilities for benchmark results."""
from __future__ import annotations

import numpy as np

# Use non-interactive backend so it works without display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_mse_comparison(summary: dict, output_path: str) -> None:
    """Bar plot of MSE for each model with error bars."""
    models = list(summary.keys())
    means = [summary[m]["mse_total"]["mean"] for m in models]
    stds = [summary[m]["mse_total"]["std"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#937860"])
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("MSE (lower is better)", fontsize=12)
    ax.set_title("Benchmark: MSE Comparison (mean ± std over 5 seeds)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.001,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_forgetting_comparison(summary: dict, output_path: str) -> None:
    """Bar plot of forgetting for each model."""
    models = list(summary.keys())
    means = [summary[m]["forgetting"]["mean"] for m in models]
    stds = [summary[m]["forgetting"]["std"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, capsize=5, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#937860"])
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Forgetting (lower is better)", fontsize=12)
    ax.set_title("Benchmark: Catastrophic Forgetting", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_timing_comparison(summary: dict, output_path: str) -> None:
    """Bar plot of wall time for each model."""
    models = list(summary.keys())
    means = [summary[m]["wall_time"]["mean"] for m in models]
    stds = [summary[m]["wall_time"]["std"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, capsize=5, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#937860"])
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Wall Time (seconds)", fontsize=12)
    ax.set_title("Benchmark: Training Time", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_mse_curves(curves: dict, output_path: str) -> None:
    """Plot MSE over training steps for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, mse_series in curves.items():
        arr = np.array(mse_series)  # (n_seeds, n_steps)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        steps = np.arange(len(mean))
        ax.plot(steps, mean, label=model_name)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE (running)")
    ax.set_title("MSE During Training (mean ± std)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
