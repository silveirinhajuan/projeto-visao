#!/usr/bin/env python3
"""Generate plots and analysis for surprise gain sweep."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load results
results_path = Path(__file__).resolve().parents[1] / "results" / "sweep_surprise_gain" / "metrics.json"
with open(results_path) as f:
    data = json.load(f)

sweep = data["sweep_results"]

gains = [r["gain"] for r in sweep]
acc_means = [r["accuracy"] for r in sweep]
acc_stds = [r["accuracy_std"] for r in sweep]
forget_means = [r["forgetting"] for r in sweep]
forget_stds = [r["forgetting_std"] for r in sweep]
bwt_means = [r["bwt"] for r in sweep]
wall_times = [r["wall_time"] for r in sweep]

output_dir = Path(__file__).resolve().parents[1] / "results" / "sweep_surprise_gain"

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Accuracy vs Gain
ax = axes[0, 0]
ax.errorbar(gains, acc_means, yerr=acc_stds, fmt='o-', capsize=5, capthick=2, color='#2196F3', linewidth=2)
ax.set_xlabel('Surprise Gain', fontsize=12)
ax.set_ylabel('Average Accuracy (↑ better)', fontsize=12)
ax.set_title('Accuracy vs Surprise Gain', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_xticks(gains)
ax.set_xticklabels([str(g) for g in gains])

# 2. Forgetting vs Gain
ax = axes[0, 1]
ax.errorbar(gains, forget_means, yerr=forget_stds, fmt='s-', capsize=5, capthick=2, color='#FF5722', linewidth=2)
ax.set_xlabel('Surprise Gain', fontsize=12)
ax.set_ylabel('Forgetting (↑ worse)', fontsize=12)
ax.set_title('Forgetting vs Surprise Gain', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_xticks(gains)
ax.set_xticklabels([str(g) for g in gains])

# 3. BWT vs Gain
ax = axes[1, 0]
ax.plot(gains, bwt_means, 'D-', color='#4CAF50', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Surprise Gain', fontsize=12)
ax.set_ylabel('Backward Transfer (↑ better)', fontsize=12)
ax.set_title('BWT vs Surprise Gain', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_xticks(gains)
ax.set_xticklabels([str(g) for g in gains])

# 4. Accuracy vs Forgetting tradeoff (Pareto)
ax = axes[1, 1]
scatter = ax.scatter(forget_means, acc_means, c=gains, cmap='viridis', s=100, edgecolors='black', linewidth=0.5)
for i, g in enumerate(gains):
    ax.annotate(f'g={g}', (forget_means[i], acc_means[i]), textcoords="offset points", xytext=(5, 5), fontsize=9)
ax.set_xlabel('Forgetting (↑ worse)', fontsize=12)
ax.set_ylabel('Average Accuracy (↑ better)', fontsize=12)
ax.set_title('Accuracy-Forgetting Tradeoff', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Surprise Gain')

fig.suptitle('VISÃO Surprise Gain Sweep Results', fontsize=16, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(output_dir / "sweep_plot.png", dpi=150, bbox_inches='tight')
print(f"Plot saved to {output_dir / 'sweep_plot.png'}")

# Find optimal gain
# Use utility: accuracy - forgetting - 0.3*std (penalize variance moderately)
utilities = [acc_means[i] - forget_means[i] - 0.3 * acc_stds[i] for i in range(len(gains))]
best_idx = np.argmax(utilities)
optimal_gain = gains[best_idx]

print(f"\nOptimal gain: {optimal_gain}")
print(f"  Accuracy: {acc_means[best_idx]:.4f} ± {acc_stds[best_idx]:.4f}")
print(f"  Forgetting: {forget_means[best_idx]:.4f} ± {forget_stds[best_idx]:.4f}")
print(f"  BWT: {bwt_means[best_idx]:.4f}")
print(f"  Utility: {utilities[best_idx]:.4f}")

# Also compute Pareto front
pareto_gains = []
for i in range(len(gains)):
    is_pareto = True
    for j in range(len(gains)):
        if i != j:
            # j dominates j if it has >= accuracy AND <= forgetting, with at least one strict
            if acc_means[j] >= acc_means[i] and forget_means[j] <= forget_means[i]:
                if acc_means[j] > acc_means[i] or forget_means[j] < forget_means[i]:
                    is_pareto = False
                    break
    if is_pareto:
        pareto_gains.append(gains[i])

print(f"\nPareto-optimal gains: {pareto_gains}")

# Create recommendation summary
summary = f"""# Surprise Gain Sweep — Results & Recommendation

## Configuration
- Dataset: sine_regression (5 tasks, 5 seeds each)
- Model: VISÃO (with meta-learning)
- Gain values tested: {gains}

## Results Summary

| Gain | Accuracy | Std | Forgetting | Std | BWT | Wall Time |
|------|----------|-----|-----------|-----|-----|-----------|
"""

for i in range(len(gains)):
    summary += f"| {gains[i]:.1f} | {acc_means[i]:.4f} | {acc_stds[i]:.4f} | {forget_means[i]:.4f} | {forget_stds[i]:.4f} | {bwt_means[i]:.4f} | {wall_times[i]:.1f}s |\n"

summary += f"""
## Key Findings

1. **Accuracy improves monotonically with gain**: from {acc_means[0]:.4f} (gain=0) to {acc_means[-1]:.4f} (gain=10)
2. **Forgetting peaks around gain=2.5** ({forget_means[5]:.4f}) then slightly decreases
3. **Variance increases substantially at high gains**: std goes from {acc_stds[0]:.4f} (gain=0) to {acc_stds[-1]:.4f} (gain=10)
4. **Diminishing returns**: accuracy gain from 0→3 is +{acc_means[6]-acc_means[0]:.4f}, but from 3→10 is only +{acc_means[-1]-acc_means[6]:.4f}

## Recommendation

**Optimal gain value: {optimal_gain}**

Rationale:
- Accuracy: {acc_means[best_idx]:.4f} ± {acc_stds[best_idx]:.4f} (good balance)
- Forgetting: {forget_means[best_idx]:.4f} ± {forget_stds[best_idx]:.4f} (manageable)
- BWT: {bwt_means[best_idx]:.4f}
- Lower variance than gain=5.0 or 10.0

### Alternative recommendations:
- **For max accuracy** (regardless of variance): gain=10.0 (accuracy={acc_means[-1]:.4f}, but std={acc_stds[-1]:.4f})
- **For best stability**: gain=0.0 (std={acc_stds[0]:.4f}, but accuracy={acc_means[0]:.4f})
- **Pareto-optimal set**: {pareto_gains}

## Context from Ablation Study
- Without surprise (gain=0): accuracy=0.533, forgetting=0.013
- With surprise (gain=3): accuracy=0.607, forgetting=0.053
- **Surprise improves accuracy by ~7.4% at the cost of ~4% more forgetting**

---
*Generated by VISÃO Research Harness — Surprise Gain Sweep*
"""

with open(output_dir / "RECOMMENDATION.md", "w") as f:
    f.write(summary)
print(f"\nRecommendation saved to {output_dir / 'RECOMMENDATION.md'}")

# Update metrics.json with analysis
data["analysis"] = {
    "optimal_gain": optimal_gain,
    "pareto_optimal_gains": pareto_gains,
    "utility_scores": {f"gain_{gains[i]}": utilities[i] for i in range(len(gains))},
    "recommendation": f"Gain {optimal_gain} provides best accuracy-forgetting-variance tradeoff",
}
with open(results_path, "w") as f:
    json.dump(data, f, indent=2, default=str)

print("\nDone!")
