"""Main entry point for the VISÃO Research Harness.

Usage:
    python -m research.runner --config configs/benchmark_5models.yaml --seed 42
    python -m research.runner --config configs/continual_5tasks.yaml
    python -m research.runner --config configs/ablation_visao.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from research.config_loader import load_config
from research.experiments.runners import create_experiment


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VISÃO Research Harness — Reproducible Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m research.runner --config configs/benchmark_5models.yaml
  python -m research.runner --config configs/continual_5tasks.yaml --seed 123
  python -m research.runner --config configs/ablation_visao.yaml --n_seeds 5
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML experiment config",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None,
        help="Number of seeds to run (overrides config)",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable plot generation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to research dir
        alt_path = Path(__file__).parent / config_path
        if alt_path.exists():
            config_path = alt_path
        else:
            print(f"ERROR: Config file not found: {args.config}", file=sys.stderr)
            return 1

    print(f"Loading config: {config_path}")
    config = load_config(config_path)

    # Apply CLI overrides
    if args.seed is not None:
        config["seed"] = args.seed
    if args.n_seeds is not None:
        config["n_seeds"] = args.n_seeds
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.no_plots:
        config["save_plots"] = False
    if args.quiet:
        config["verbose"] = False

    # Create and run experiment
    experiment = create_experiment(config)

    print("=" * 60)
    print(f"  VISÃO Research Harness v1.0")
    print(f"  Experiment: {experiment.name}")
    print("=" * 60)

    start = time.time()
    results = experiment.execute()
    elapsed = time.time() - start

    print("=" * 60)
    print(f"  Completed in {elapsed:.2f}s")
    print(f"  Results: {experiment.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
