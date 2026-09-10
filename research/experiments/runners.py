"""Experiment runners for reproducible research.

Provides abstract base class and concrete implementations for:
  - BenchmarkExperiment: compare multiple models on same data
  - ContinualLearningExperiment: sequential task learning
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from research.config_loader import load_config, save_config
from research.datasets.generators import get_generator, split_continual_tasks
from research.metrics.metrics import (
    compute_all_metrics,
    compute_forgetting,
    backward_transfer,
    forward_transfer,
    average_accuracy,
)
from research.models.model_defs import create_model, BaseModel
from research.statistics.stats import aggregate_runs, format_summary
from research.visualization.plotting import (
    plot_learning_curve,
    plot_comparison_bars,
    plot_continual_matrix,
    plot_confidence_intervals,
)


class Experiment(ABC):
    """Abstract base class for all experiments.

    Lifecycle:
      1. __init__(config)
      2. setup()
      3. run() -> results dict
      4. save_results(results)
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name = config.get("name", "experiment")
        self.seed = config.get("seed", 42)
        self.output_dir = Path(config.get("output_dir", "results")) / self.name
        self.save_checkpoints = config.get("save_checkpoints", True)
        self.save_plots = config.get("save_plots", True)
        self.verbose = config.get("verbose", True)
        self._logs: list[str] = []

    def log(self, msg: str) -> None:
        """Log a message."""
        self._logs.append(msg)
        if self.verbose:
            print(msg)

    def setup(self) -> None:
        """Create output directories and initialize experiment."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoint").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

        # Save config
        save_config(self.config, self.output_dir / "config.yaml")
        self.log(f"Experiment: {self.name}")
        self.log(f"Output: {self.output_dir}")
        self.log(f"Seed: {self.seed}")

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Run the experiment and return results."""
        ...

    def save_results(self, results: dict[str, Any]) -> None:
        """Save results to disk (JSON + markdown summary)."""
        # Save metrics JSON
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self.log(f"Metrics saved: {metrics_path}")

        # Save logs
        logs_path = self.output_dir / "logs" / "experiment.log"
        with open(logs_path, "w") as f:
            f.write("\n".join(self._logs))

        # Generate markdown summary
        summary = self._generate_summary(results)
        summary_path = self.output_dir / "summary.md"
        with open(summary_path, "w") as f:
            f.write(summary)
        self.log(f"Summary saved: {summary_path}")

    def _generate_summary(self, results: dict[str, Any]) -> str:
        """Generate markdown summary from results."""
        lines = [
            f"# Experiment: {self.name}",
            "",
            "## Configuration",
            f"- Seed: {self.seed}",
            f"- Type: {self.config.get('type', 'unknown')}",
            "",
            "## Results",
            "",
        ]

        if "model_results" in results:
            lines.append("### Model Results")
            lines.append("")
            lines.append("| Model | Metric | Mean | Std | CI |")
            lines.append("|-------|--------|------|-----|-----|")

            for model_name, metrics in results["model_results"].items():
                for metric_name, stats in metrics.items():
                    if isinstance(stats, dict) and "mean" in stats:
                        lines.append(
                            f"| {model_name} | {metric_name} | "
                            f"{stats['mean']:.4f} | {stats.get('std', 0):.4f} | "
                            f"[{stats.get('ci_lower', 0):.4f}, {stats.get('ci_upper', 0):.4f}] |"
                        )

        if "aggregated" in results:
            lines.append("")
            lines.append("### Aggregated Statistics")
            lines.append("```")
            lines.append(format_summary(results["aggregated"]))
            lines.append("```")

        if "continual_metrics" in results:
            lines.append("")
            lines.append("### Continual Learning Metrics")
            cm = results["continual_metrics"]
            lines.append(f"- Average Accuracy: {cm.get('average_accuracy', 'N/A')}")
            lines.append(f"- Forgetting: {cm.get('forgetting', 'N/A')}")
            lines.append(f"- Backward Transfer: {cm.get('backward_transfer', 'N/A')}")
            lines.append(f"- Forward Transfer: {cm.get('forward_transfer', 'N/A')}")

        lines.append("")
        lines.append(f"---")
        lines.append(f"*Generated by VISÃO Research Harness v1.0*")

        return "\n".join(lines)

    def execute(self) -> dict[str, Any]:
        """Full execution: setup -> run -> save."""
        self.setup()
        start = time.time()
        results = self.run()
        elapsed = time.time() - start
        results["elapsed_seconds"] = elapsed
        self.log(f"Completed in {elapsed:.2f}s")
        self.save_results(results)
        return results


class BenchmarkExperiment(Experiment):
    """Benchmark experiment: compare multiple models on the same dataset."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.dataset_name = config.get("dataset", {}).get("name", "sine_regression")
        self.dataset_params = config.get("dataset", {}).get("params", {})
        self.models_config = config.get("models", [])
        self.metric_names = config.get("metrics", ["mse", "mae"])
        self.n_seeds = config.get("n_seeds", 1)
        self.task_type = config.get("task_type", "regression")

    def run(self) -> dict[str, Any]:
        """Run benchmark across models and seeds."""
        self.log(f"Dataset: {self.dataset_name}")
        self.log(f"Models: {[m['name'] for m in self.models_config]}")
        self.log(f"Seeds: {self.n_seeds}")

        all_results = {}  # {model_name: [per-seed results]}

        for model_cfg in self.models_config:
            model_name = model_cfg["name"]
            all_results[model_name] = []

            for seed_offset in range(self.n_seeds):
                seed = self.seed + seed_offset
                self.log(f"\n--- {model_name} (seed={seed}) ---")

                # Generate dataset
                generator = get_generator(self.dataset_name)
                X_train, Y_train, X_test, Y_test = generator(seed=seed, **self.dataset_params)

                # Determine dimensions
                n_in = X_train.shape[-1]
                n_out = Y_train.shape[-1] if Y_train.ndim > 1 else 1

                # Create and train model
                model = create_model(
                    model_name,
                    n_in=n_in,
                    n_out=n_out,
                    seed=seed,
                    **model_cfg.get("params", {}),
                )

                train_result = model.fit(X_train, Y_train)
                self.log(f"  Training loss: {train_result['final_loss']:.4f}")

                # Evaluate
                model.reset_state()
                predictions = model.predict(X_test)

                # Compute metrics
                metrics = compute_all_metrics(Y_test, predictions, self.task_type)
                self.log(f"  Test metrics: {metrics}")

                # Save checkpoint
                if self.save_checkpoints:
                    ckpt_path = self.output_dir / "checkpoint" / f"{model_name}_seed{seed}.json"
                    model.save(ckpt_path)

                all_results[model_name].append(metrics)

        # Aggregate results
        model_results = {}
        for model_name, results_list in all_results.items():
            model_results[model_name] = aggregate_runs(results_list)

        # Plot comparisons
        if self.save_plots:
            for metric_name in self.metric_names:
                plot_data = {
                    m: model_results[m].get(metric_name, {})
                    for m in all_results
                    if metric_name in model_results.get(m, {})
                }
                if plot_data:
                    plot_comparison_bars(
                        {m: {metric_name: s["mean"]} for m, s in plot_data.items() if isinstance(s, dict)},
                        metric_name=metric_name,
                        title=f"{metric_name.upper()} Comparison - {self.name}",
                        ylabel=metric_name.upper(),
                        save_path=self.output_dir / "plots" / f"comparison_{metric_name}.png",
                    )

            # Confidence interval plot
            if self.n_seeds > 1:
                plot_confidence_intervals(
                    model_results,
                    metric_name=self.metric_names[0],
                    title=f"Model Comparison (95% CI) - {self.name}",
                    save_path=self.output_dir / "plots" / "confidence_intervals.png",
                )

        return {
            "model_results": model_results,
            "dataset": self.dataset_name,
            "n_seeds": self.n_seeds,
            "models": list(all_results.keys()),
        }


class ContinualLearningExperiment(Experiment):
    """Continual learning experiment: sequential task training."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.dataset_name = config.get("dataset", {}).get("name", "sine_regression")
        self.dataset_params = config.get("dataset", {}).get("params", {})
        self.n_tasks = config.get("n_tasks", 5)
        self.models_config = config.get("models", [])
        self.task_type = config.get("task_type", "regression")
        self.n_seeds = config.get("n_seeds", 1)

    def run(self) -> dict[str, Any]:
        """Run continual learning experiment."""
        self.log(f"Continual Learning: {self.n_tasks} tasks")
        self.log(f"Models: {[m['name'] for m in self.models_config]}")

        all_results = {}

        for model_cfg in self.models_config:
            model_name = model_cfg["name"]
            all_results[model_name] = []

            for seed_offset in range(self.n_seeds):
                seed = self.seed + seed_offset
                self.log(f"\n--- {model_name} (seed={seed}) ---")

                # Generate tasks
                generator = get_generator(self.dataset_name)
                X, Y, _, _ = generator(seed=seed, **self.dataset_params)
                tasks = split_continual_tasks(X, Y, n_tasks=self.n_tasks, seed=seed)

                n_in = X.shape[-1]
                n_out = Y.shape[-1] if Y.ndim > 1 else 1

                # Create model
                model = create_model(
                    model_name,
                    n_in=n_in,
                    n_out=n_out,
                    seed=seed,
                    **model_cfg.get("params", {}),
                )

                # Performance matrix: perf[i][j] = perf on task i after training on task j
                perf_matrix = np.zeros((self.n_tasks, self.n_tasks))

                for task_idx, (X_task, Y_task) in enumerate(tasks):
                    self.log(f"  Training on task {task_idx} ({len(X_task)} samples)")
                    model.fit(X_task, Y_task)

                    # Evaluate on all tasks seen so far
                    for eval_idx in range(task_idx + 1):
                        X_eval, Y_eval = tasks[eval_idx]
                        model.reset_state()
                        preds = model.predict(X_eval)
                        metrics = compute_all_metrics(Y_eval, preds, self.task_type)
                        # Use MSE (lower is better) or accuracy (higher is better)
                        if self.task_type == "regression":
                            perf_matrix[eval_idx, task_idx] = metrics["mse"]
                        else:
                            perf_matrix[eval_idx, task_idx] = metrics["accuracy"]

                # Compute continual learning metrics
                if self.task_type == "regression":
                    # For MSE, lower is better, so forgetting = increase in MSE
                    forgetting = compute_forgetting(-perf_matrix)  # negate so higher = better
                    bwt = backward_transfer(-perf_matrix)
                else:
                    forgetting = compute_forgetting(perf_matrix)
                    bwt = backward_transfer(perf_matrix)

                fwt = forward_transfer(perf_matrix)
                avg_acc = average_accuracy(perf_matrix)

                self.log(f"  Forgetting: {forgetting:.4f}")
                self.log(f"  BWT: {bwt:.4f}")
                self.log(f"  Avg performance: {avg_acc:.4f}")

                all_results[model_name].append({
                    "performance_matrix": perf_matrix.tolist(),
                    "forgetting": forgetting,
                    "backward_transfer": bwt,
                    "forward_transfer": fwt,
                    "average_accuracy": avg_acc,
                })

                # Save checkpoint
                if self.save_checkpoints:
                    ckpt_path = self.output_dir / "checkpoint" / f"{model_name}_seed{seed}.json"
                    model.save(ckpt_path)

                # Plot performance matrix
                if self.save_plots:
                    plot_continual_matrix(
                        perf_matrix,
                        task_names=[f"Task {i}" for i in range(self.n_tasks)],
                        title=f"{model_name} - Continual Learning Performance",
                        save_path=self.output_dir / "plots" / f"{model_name}_continual_matrix_seed{seed}.png",
                    )

        # Aggregate
        model_results = {}
        for model_name, results_list in all_results.items():
            model_results[model_name] = aggregate_runs(results_list)

        return {
            "model_results": model_results,
            "continual_metrics": {
                "n_tasks": self.n_tasks,
                "forgetting": {m: np.mean([r["forgetting"] for r in all_results[m]]) for m in all_results},
                "backward_transfer": {m: np.mean([r["backward_transfer"] for r in all_results[m]]) for m in all_results},
                "forward_transfer": {m: np.mean([r["forward_transfer"] for r in all_results[m]]) for m in all_results},
                "average_accuracy": {m: np.mean([r["average_accuracy"] for r in all_results[m]]) for m in all_results},
            },
            "dataset": self.dataset_name,
            "n_seeds": self.n_seeds,
        }


def create_experiment(config: dict[str, Any]) -> Experiment:
    """Factory to create the right experiment type from config."""
    exp_type = config.get("type", "benchmark")
    if exp_type == "benchmark":
        return BenchmarkExperiment(config)
    elif exp_type == "continual":
        return ContinualLearningExperiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")
