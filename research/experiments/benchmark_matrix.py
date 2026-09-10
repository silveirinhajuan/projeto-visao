"""Benchmark Matrix experiment for VISÃO.

Compares MLP, GRU, LSTM, CfC, and VISÃO on a non-stationary stream
(sine → sawtooth → mixed) with identical data, seeds, and computational budget.

Outputs:
    results/benchmark_matrix/metrics.json  — raw per-seed metrics
    results/benchmark_matrix/summary.md    — formatted results table
    results/benchmark_matrix/*.png         — comparison plots
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# Ensure project root is importable
import sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.datasets import NonStationaryStream
from research.models.baselines import MLP, GRU, LSTM, CfC
from research.metrics import forgetting as compute_forgetting, backward_transfer as compute_bwt
from research.statistics import summarize_metrics, compare_models
from research.visualization import plot_mse_comparison, plot_forgetting_comparison, plot_timing_comparison


class BenchmarkMatrix:
    """Standardized benchmark: same data, seeds, budget for all models."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output"]["dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ════════════════════════════════════════════════════════════════════════
    def run(self) -> dict:
        """Run the full benchmark. Returns summary dict."""
        seeds = self.config["seeds"]
        model_configs = self.config["models"]
        dataset_cfg = self.config["dataset"]

        # Accumulator: {model_name: {metric_name: [per-seed values]}}
        all_results: dict[str, dict[str, list]] = {
            name: {
                "mse_total": [], "mse_final": [], "forgetting": [],
                "backward_transfer": [], "forward_transfer": [],
                "wall_time": [], "n_params": [],
            }
            for name in model_configs
        }

        for seed in seeds:
            print(f"\n{'=' * 60}")
            print(f"  SEED {seed}")
            print(f"{'=' * 60}")

            # Generate identical data for all models this seed
            train_data, test_data = self._generate_data(seed, dataset_cfg)

            for model_name, model_cfg in model_configs.items():
                print(f"\n  [{seed}] Training {model_name}...", flush=True)
                model = self._create_model(model_cfg, seed)
                metrics = self._train_and_evaluate(model, model_name, train_data, test_data)

                print(f"  [{seed}] {model_name}: MSE={metrics['mse_total']:.6f}  "
                      f"Fgt={metrics['forgetting']:.6f}  "
                      f"Time={metrics['wall_time']:.2f}s  "
                      f"Params={metrics['n_params']}")

                for k, v in metrics.items():
                    if k in all_results[model_name]:
                        all_results[model_name][k].append(v)

        # Compute statistics
        summary = self._compute_statistics(all_results)

        # Save everything
        self._save_results(all_results, summary)

        return summary

    # ════════════════════════════════════════════════════════════════════════
    #  DATA GENERATION
    # ════════════════════════════════════════════════════════════════════════
    def _generate_data(self, seed: int, dataset_cfg: dict):
        """Generate train/test splits for all tasks.

        Returns:
            train_data: list of (X, Y) per task, each (n_samples, 1)
            test_data:  list of (X, Y) per task, each (n_test, 1)
        """
        tasks_cfg = dataset_cfg["tasks"]
        n_test_per_task = dataset_cfg.get("n_test", 500)

        train_data = []
        test_data = []

        for i, task in enumerate(tasks_cfg):
            # Each task gets a unique seed derived from main seed + task index
            task_seed = seed + i * 1000
            n_samples = task["n_samples"]
            total = n_samples + n_test_per_task

            # Generate signal
            rng = np.random.default_rng(task_seed)
            sig = self._generate_signal(rng, task, total + 1)  # +1 for target

            # Normalize
            mx = np.max(np.abs(sig))
            if mx > 0:
                sig /= mx

            # Split: first n_samples for train, next n_test for test
            X_all = sig[:-1].reshape(-1, 1)
            Y_all = sig[1:].reshape(-1, 1)

            train_data.append((X_all[:n_samples], Y_all[:n_samples]))
            test_data.append((X_all[n_samples: n_samples + n_test_per_task],
                              Y_all[n_samples: n_samples + n_test_per_task]))

        return train_data, test_data

    @staticmethod
    def _generate_signal(rng: np.random.Generator, task: dict, length: int) -> np.ndarray:
        """Generate a 1D signal for a given task."""
        t = np.arange(length)
        if task["type"] == "sine":
            phase = rng.uniform(0, 2 * np.pi)
            return np.sin(2 * np.pi * task["frequency"] * t / 50.0 + phase)
        elif task["type"] == "sawtooth":
            phase = rng.uniform(0, 2 * np.pi)
            return 2.0 * ((t / (50.0 / task["frequency"]) + phase / (2 * np.pi)) % 1.0) - 1.0
        elif task["type"] == "mixed":
            sig = np.zeros(length)
            for f in task["frequencies"]:
                phase = rng.uniform(0, 2 * np.pi)
                sig += np.sin(2 * np.pi * f * t / 50.0 + phase)
            sig /= len(task["frequencies"])
            return sig
        else:
            raise ValueError(f"Unknown task type: {task['type']}")

    # ════════════════════════════════════════════════════════════════════════
    #  MODEL CREATION
    # ════════════════════════════════════════════════════════════════════════
    def _create_model(self, model_cfg: dict, seed: int):
        """Create a model from config dict."""
        mtype = model_cfg["type"]
        n_hidden = model_cfg.get("n_hidden", 64)

        if mtype == "mlp":
            return MLP(n_in=1, n_hidden=n_hidden, n_out=1,
                       lr=model_cfg.get("lr", 0.01), seed=seed)
        elif mtype == "gru":
            return GRU(n_in=1, n_hidden=n_hidden, n_out=1,
                       lr=model_cfg.get("lr", 0.01), seed=seed)
        elif mtype == "lstm":
            return LSTM(n_in=1, n_hidden=n_hidden, n_out=1,
                        lr=model_cfg.get("lr", 0.01), seed=seed)
        elif mtype == "cfc":
            return CfC(n_in=1, n_hidden=n_hidden, n_out=1,
                       lr=model_cfg.get("lr", 0.02),
                       sparsity=model_cfg.get("sparsity", 0.6),
                       tau_min=model_cfg.get("tau_min", 0.4),
                       tau_max=model_cfg.get("tau_max", 4.0),
                       dt=model_cfg.get("dt", 0.15), seed=seed)
        elif mtype == "visao":
            from visao.brain import VisaoBrain
            return VisaoBrain(
                n_in=1, n_hidden=n_hidden, n_out=1,
                lr=model_cfg.get("lr", 0.02),
                oja_lr=model_cfg.get("oja_lr", 0.0015),
                consolidation=model_cfg.get("consolidation", 8.0),
                surprise_gain=model_cfg.get("surprise_gain", 3.0),
                lambda_decay=model_cfg.get("lambda_decay", 0.0005),
                sparsity=model_cfg.get("sparsity", 0.6),
                tau_min=model_cfg.get("tau_min", 0.4),
                tau_max=model_cfg.get("tau_max", 4.0),
                dt=model_cfg.get("dt", 0.15),
                meta_learn=model_cfg.get("meta_learn", False),
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown model type: {mtype}")

    # ════════════════════════════════════════════════════════════════════════
    #  TRAINING & EVALUATION
    # ════════════════════════════════════════════════════════════════════════
    def _train_and_evaluate(self, model, model_name: str,
                            train_data: list, test_data: list) -> dict:
        """Train model on sequential tasks and evaluate.

        Returns dict with: mse_total, mse_final, forgetting, backward_transfer,
                           forward_transfer, wall_time, n_params
        """
        n_tasks = len(train_data)
        mse_per_task_history = []  # MSE right after training on each task
        mse_checkpoints = []  # MSE on each task after each training phase

        start_time = time.time()

        for task_idx in range(n_tasks):
            X_train, Y_train = train_data[task_idx]

            # Train on this task
            self._train_on_task(model, model_name, X_train, Y_train)

            # Evaluate on all tasks seen so far
            task_mses = []
            for eval_idx in range(task_idx + 1):
                X_test, Y_test = test_data[eval_idx]
                mse = self._evaluate(model, model_name, X_test, Y_test)
                task_mses.append(mse)

            mse_checkpoints.append(task_mses)

            # Record MSE on current task right after training
            mse_per_task_history.append(task_mses[-1])

        wall_time = time.time() - start_time

        # Final evaluation on all tasks
        mse_final_per_task = []
        for task_idx in range(n_tasks):
            X_test, Y_test = test_data[task_idx]
            mse = self._evaluate(model, model_name, X_test, Y_test)
            mse_final_per_task.append(mse)

        # Compute forgetting: degradation from best to final
        fgt = compute_forgetting(mse_per_task_history, mse_final_per_task)

        # Backward transfer: average change in old task performance
        bwt = compute_bwt(mse_final_per_task, mse_per_task_history)

        return {
            "mse_total": float(np.mean(mse_final_per_task)),
            "mse_final": float(np.mean(mse_final_per_task)),
            "forgetting": fgt,
            "backward_transfer": bwt,
            "forward_transfer": 0.0,  # requires separate experiment
            "wall_time": wall_time,
            "n_params": model.n_params(),
        }

    def _train_on_task(self, model, model_name: str,
                       X: np.ndarray, Y: np.ndarray):
        """Train model on one task's data."""
        for x, y in zip(X, Y):
            if model_name == "VISÃO":
                result = model.learn(x, y)
            else:
                model.learn(x, y)

    def _evaluate(self, model, model_name: str,
                  X: np.ndarray, Y: np.ndarray) -> float:
        """Evaluate model on test data. Returns MSE."""
        if model_name == "VISÃO":
            return self._evaluate_visao(model, X, Y)
        else:
            model.reset_state()
            mses = []
            for x, y in zip(X, Y):
                pred = model.predict(x)
                mse = float(np.mean((pred - y) ** 2))
                mses.append(mse)
            return float(np.mean(mses))

    @staticmethod
    def _evaluate_visao(model, X: np.ndarray, Y: np.ndarray) -> float:
        """Evaluate VisaoBrain (handles mode switching)."""
        model.set_mode("infer")
        model.reset_state()
        mses = []
        for x, y in zip(X, Y):
            pred = model.forward(x)
            mse = float(np.mean((pred - y) ** 2))
            mses.append(mse)
        model.set_mode("learn")
        return float(np.mean(mses))

    # ════════════════════════════════════════════════════════════════════════
    #  STATISTICS
    # ════════════════════════════════════════════════════════════════════════
    def _compute_statistics(self, all_results: dict) -> dict:
        """Compute mean, std, CI, and Cohen's d for all models."""
        summary = {}

        for model_name, metrics in all_results.items():
            summary[model_name] = {}
            for metric_name, values in metrics.items():
                summary[model_name][metric_name] = summarize_metrics(
                    {metric_name: values}
                )[metric_name]

        # Cohen's d: VISÃO vs each baseline
        visao_mse = all_results["VISÃO"]["mse_total"]
        summary["_comparisons"] = {}
        for model_name in all_results:
            if model_name == "VISÃO":
                continue
            baseline_mse = all_results[model_name]["mse_total"]
            summary["_comparisons"][f"VISÃO_vs_{model_name}"] = compare_models(
                visao_mse, baseline_mse
            )

        return summary

    # ════════════════════════════════════════════════════════════════════════
    #  OUTPUT
    # ════════════════════════════════════════════════════════════════════════
    def _save_results(self, all_results: dict, summary: dict):
        """Save metrics.json, summary.md, and plots."""
        # 1. metrics.json — raw per-seed results
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Saved: {metrics_path}")

        # 2. summary.md — formatted table
        summary_path = self.output_dir / "summary.md"
        with open(summary_path, "w") as f:
            f.write(self._format_summary(all_results, summary))
        print(f"  Saved: {summary_path}")

        # 3. Plots
        try:
            plot_mse_comparison(summary, str(self.output_dir / "mse_comparison.png"))
            plot_forgetting_comparison(summary, str(self.output_dir / "forgetting_comparison.png"))
            plot_timing_comparison(summary, str(self.output_dir / "timing_comparison.png"))
            print(f"  Saved: plots in {self.output_dir}")
        except Exception as e:
            print(f"  [WARN] Plot generation failed: {e}")

    def _format_summary(self, all_results: dict, summary: dict) -> str:
        """Format results as markdown."""
        lines = [
            "# VISÃO Benchmark Matrix Results",
            "",
            "## Configuration",
            f"- Seeds: {self.config['seeds']}",
            f"- Tasks: {', '.join(t['type'] for t in self.config['dataset']['tasks'])}",
            f"- Training steps: {self.config['dataset']['n_train_steps']}",
            f"- Test samples per task: {self.config['dataset'].get('n_test', 500)}",
            "",
            "## Results (mean ± std over seeds)",
            "",
            "| Model | MSE Total | Forgetting | BWT | Wall Time (s) | # Params |",
            "|-------|-----------|------------|-----|---------------|----------|",
        ]

        for model_name in all_results:
            s = summary[model_name]
            mse_str = f"{s['mse_total']['mean']:.6f} ± {s['mse_total']['std']:.6f}"
            fgt_str = f"{s['forgetting']['mean']:.6f} ± {s['forgetting']['std']:.6f}"
            bwt_str = f"{s['backward_transfer']['mean']:.6f} ± {s['backward_transfer']['std']:.6f}"
            time_str = f"{s['wall_time']['mean']:.2f} ± {s['wall_time']['std']:.2f}"
            params_str = str(s['n_params']['mean'])
            lines.append(f"| {model_name} | {mse_str} | {fgt_str} | {bwt_str} | {time_str} | {params_str} |")

        # Cohen's d comparisons
        lines.extend([
            "",
            "## Effect Sizes (Cohen's d: VISÃO vs Baseline)",
            "",
            "| Comparison | Cohen's d | t-stat | p-value | Interpretation |",
            "|------------|-----------|--------|---------|----------------|",
        ])

        for comp_name, comp in summary.get("_comparisons", {}).items():
            d = comp["cohens_d"]
            if abs(d) < 0.2:
                interp = "negligible"
            elif abs(d) < 0.5:
                interp = "small"
            elif abs(d) < 0.8:
                interp = "medium"
            else:
                interp = "large"
            if d < 0:
                interp += " (VISÃO better)"
            else:
                interp += " (baseline better)"
            lines.append(
                f"| {comp_name} | {d:.4f} | {comp['t_stat']:.4f} | {comp['p_value']:.4f} | {interp} |"
            )

        # Per-seed detail
        lines.extend(["", "## Per-Seed Results", ""])
        for model_name, metrics in all_results.items():
            lines.append(f"### {model_name}")
            lines.append(f"- MSE Total: {metrics['mse_total']}")
            lines.append(f"- Forgetting: {metrics['forgetting']}")
            lines.append(f"- Wall Time: {metrics['wall_time']}")
            lines.append("")

        return "\n".join(lines)
