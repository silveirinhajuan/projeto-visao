#!/usr/bin/env python3
"""Run surprise gain sweep for VISÃO project.

Sweeps over surprise_gain values and collects metrics for each.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.config_loader import load_config, save_config
from research.experiments.runners import create_experiment

GAIN_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
SEED = 42
N_SEEDS = 5
N_TASKS = 5

BASE_CONFIG = {
    "name": "sweep_surprise_gain",
    "type": "continual",
    "seed": SEED,
    "n_seeds": N_SEEDS,
    "task_type": "regression",
    "output_dir": "results/sweep_surprise_gain",
    "save_checkpoints": False,
    "save_plots": False,
    "verbose": True,
    "dataset": {
        "name": "sine_regression",
        "params": {
            "n_train": 400,
            "n_test": 150,
            "n_inputs": 1,
            "noise": 0.1,
            "freq": 1.0,
        },
    },
    "n_tasks": N_TASKS,
    "metrics": ["mse", "mae", "forgetting", "backward_transfer", "average_accuracy"],
    "models": [
        {
            "name": "VISÃO",
            "params": {
                "n_hidden": 64,
                "lr": 0.02,
                "sparsity": 0.6,
                "consolidation": 8.0,
                "surprise_gain": 3.0,
                "lambda_decay": 0.0005,
                "meta_learn": True,
            },
        }
    ],
}


def run_experiment(gain_value: float) -> dict:
    """Run a single experiment with the given surprise_gain."""
    config = BASE_CONFIG.copy()
    config["models"][0]["params"]["surprise_gain"] = gain_value
    config["name"] = f"sweep_gain_{gain_value}"

    experiment = create_experiment(config)
    experiment.setup()
    start = time.time()
    results = experiment.run()
    elapsed = time.time() - start
    results["elapsed_seconds"] = elapsed
    experiment.save_results(results)

    # Extract key metrics
    model_results = results.get("model_results", {})
    continual_metrics = results.get("continual_metrics", {})

    visao_results = model_results.get("VISÃO", {})

    return {
        "gain": gain_value,
        "accuracy": visao_results.get("average_accuracy", {}).get("mean", None),
        "accuracy_std": visao_results.get("average_accuracy", {}).get("std", None),
        "forgetting": visao_results.get("forgetting", {}).get("mean", None),
        "forgetting_std": visao_results.get("forgetting", {}).get("std", None),
        "bwt": visao_results.get("backward_transfer", {}).get("mean", None),
        "bwt_std": visao_results.get("backward_transfer", {}).get("std", None),
        "mse": visao_results.get("mse", {}).get("mean", None) if "mse" in visao_results else None,
        "wall_time": elapsed,
    }


def main():
    output_dir = ROOT / "results" / "sweep_surprise_gain"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for gain in GAIN_VALUES:
        print(f"\n{'='*60}")
        print(f"Running gain={gain}")
        print(f"{'='*60}")
        result = run_experiment(gain)
        all_results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f} ± {result['accuracy_std']:.4f}")
        print(f"  Forgetting: {result['forgetting']:.4f} ± {result['forgetting_std']:.4f}")
        print(f"  BWT: {result['bwt']:.4f}")
        print(f"  Wall time: {result['wall_time']:.2f}s")

    # Save all results
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"sweep_results": all_results}, f, indent=2, default=str)

    print(f"\nResults saved to {metrics_path}")
    return all_results


if __name__ == "__main__":
    main()
