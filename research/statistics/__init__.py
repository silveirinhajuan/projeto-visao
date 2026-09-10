"""Statistical analysis: mean, std, confidence intervals, effect sizes."""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def mean_std(values) -> tuple:
    """Return (mean, std) of a list of values."""
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def confidence_interval(values, level: float = 0.95) -> tuple:
    """Compute confidence interval using t-distribution."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return (float(arr[0]), float(arr[0]))
    mean = np.mean(arr)
    sem = scipy_stats.sem(arr)
    ci = scipy_stats.t.interval(level, df=n - 1, loc=mean, scale=sem)
    return (float(ci[0]), float(ci[1]))


def cohens_d(group1, group2) -> float:
    """Cohen's d effect size between two groups.

    Positive = group1 has higher mean than group2.
    """
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)


def summarize_metrics(
    results: dict, confidence_level: float = 0.95
) -> dict:
    """Summarize metrics across seeds.

    Args:
        results: {metric_name: [value_seed0, value_seed1, ...]}

    Returns:
        {metric_name: {"mean": ..., "std": ..., "ci_low": ..., "ci_high": ...}}
    """
    summary = {}
    for metric_name, values in results.items():
        mean, std = mean_std(values)
        ci_low, ci_high = confidence_interval(values, confidence_level)
        summary[metric_name] = {
            "mean": mean,
            "std": std,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": len(values),
        }
    return summary


def compare_models(
    model_a_values,
    model_b_values,
) -> dict:
    """Compare two models with Cohen's d and t-test."""
    d = cohens_d(model_a_values, model_b_values)
    # Paired t-test (same seeds)
    if len(model_a_values) == len(model_b_values) and len(model_a_values) >= 2:
        t_stat, p_value = scipy_stats.ttest_rel(model_a_values, model_b_values)
    else:
        t_stat, p_value = scipy_stats.ttest_ind(model_a_values, model_b_values)
    return {
        "cohens_d": d,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }
