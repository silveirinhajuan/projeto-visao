"""Statistical aggregation utilities.

Provides functions for computing mean, std, confidence intervals,
effect sizes, and significance tests across multiple runs/seeds.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def mean(values: np.ndarray | list) -> float:
    """Compute mean."""
    return float(np.mean(values))


def std(values: np.ndarray | list, ddof: int = 1) -> float:
    """Compute standard deviation."""
    return float(np.std(values, ddof=ddof))


def sem(values: np.ndarray | list) -> float:
    """Standard error of the mean."""
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def confidence_interval(
    values: np.ndarray | list,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval using normal approximation.

    Args:
        values: Array of values.
        confidence: Confidence level (0.95 = 95%).

    Returns:
        (lower_bound, upper_bound)
    """
    values = np.asarray(values, dtype=float)
    m = np.mean(values)
    s = np.std(values, ddof=1) / np.sqrt(len(values))

    # Z-score for confidence level
    # 0.90 -> 1.645, 0.95 -> 1.96, 0.99 -> 2.576
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    return float(m - z * s), float(m + z * s)


def cohens_d(group1: np.ndarray | list, group2: np.ndarray | list) -> float:
    """Compute Cohen's d effect size between two groups.

    Args:
        group1: First group values.
        group2: Second group values.

    Returns:
        Cohen's d (positive = group1 > group2).
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    n1, n2 = len(g1), len(g2)
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(g1) - np.mean(g2)) / pooled_std)


def paired_ttest(group1: np.ndarray | list, group2: np.ndarray | list) -> dict:
    """Paired t-test (simplified, no scipy dependency).

    Returns dict with t_statistic and approximate p-value.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    if len(g1) != len(g2):
        raise ValueError("Groups must have same length for paired t-test")

    diff = g1 - g2
    n = len(diff)
    if n < 2:
        return {"t_stat": 0.0, "p_value": 1.0}

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    if std_diff == 0:
        return {"t_stat": 0.0, "p_value": 1.0}

    t_stat = mean_diff / (std_diff / np.sqrt(n))

    # Approximate p-value using normal distribution for n > 20
    # For small n, this is approximate
    import math
    z = abs(t_stat)
    p_value = 2 * (1 - _normal_cdf(z))

    return {"t_stat": float(t_stat), "p_value": float(p_value)}


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF using error function."""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def aggregate_runs(
    results: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Aggregate results across multiple runs/seeds.

    Args:
        results: List of metric dicts from each run.

    Returns:
        Dict mapping metric_name -> {mean, std, ci_lower, ci_upper}.
    """
    if not results:
        return {}

    # Collect all metric names
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    aggregated = {}
    for key in all_keys:
        values = [r[key] for r in results if key in r]
        if not values:
            continue

        m = mean(values)
        s = std(values)
        ci_lower, ci_upper = confidence_interval(values)

        aggregated[key] = {
            "mean": m,
            "std": s,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n": len(values),
        }

    return aggregated


def format_summary(aggregated: dict[str, dict[str, float]]) -> str:
    """Format aggregated results as a readable string."""
    lines = []
    for metric, stats in sorted(aggregated.items()):
        lines.append(
            f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
            f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] (n={stats['n']})"
        )
    return "\n".join(lines)
