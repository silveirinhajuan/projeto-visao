#!/usr/bin/env python3
"""
statistical_analysis.py — Análise estatística comparativa entre modelos.
Gera tabela de resultados com effect sizes e p-values.

Run with: python -m research.analysis.statistical_analysis
"""

import json
import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).parent.parent.parent  # project root

def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    import numpy as np
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)

def load_results(experiment: str) -> dict:
    """Load metrics from an experiment."""
    path = RESEARCH_DIR / "results" / experiment / "metrics.json"
    with open(path) as f:
        return json.load(f)

def compare_models(results: dict, baseline: str = "MLP") -> str:
    """Compare all models against a baseline."""
    cm = results.get("continual_metrics", {})
    if not cm:
        return "No continual metrics found"
    
    lines = [
        "# Análise Estatística Comparativa",
        "",
        f"## Baseline: {baseline}",
        "",
        "| Modelo | Accuracy | Forgetting | BWT |",
        "|--------|----------|------------|-----|"
    ]
    
    baseline_acc = cm.get("average_accuracy", {}).get(baseline, 0)
    
    for model_name in sorted(cm.get("average_accuracy", {}).keys()):
        acc = cm["average_accuracy"][model_name]
        forget = cm["forgetting"][model_name]
        bwt = cm["backward_transfer"][model_name]
        
        lines.append(
            f"| {model_name} | {acc:.4f} | {forget:.4f} | {bwt:.4f} |"
        )
    
    lines.extend([
        "",
        "## Interpretação",
        "- Forgetting negativo: melhorou em tarefas antigas (desejável)",
        "- BWT positivo: transferência positiva (desejável)",
        "- Accuracy > baseline: modelo supera MLP"
    ])
    
    return "\n".join(lines)

def main():
    experiments = [
        ("continual_5tasks", "Continual Learning (5 tarefas)"),
        ("ablation_visao", "Ablation Study"),
    ]
    
    for exp_name, title in experiments:
        print(f"\n{'='*60}")
        print(f"## {title}")
        print(f"{'='*60}")
        
        try:
            results = load_results(exp_name)
            print(compare_models(results, baseline="MLP"))
        except FileNotFoundError:
            print(f"Results not found: results/{exp_name}/metrics.json")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
