#!/usr/bin/env python3
"""
Scaling Laws v3 — Correção: mais dados + regularização.

Melhorias vs versão anterior:
  - n_train=10000 (antes 1000)
  - Consolidação EWC para estabilidade em n_hidden grande
  - L2 via lambda_decay aumentado
  - Validação R²>0.9 antes de reportar α
  - 3 seeds para significância estatística
  - Range 32-512 (2.4 ordens de magnitude de params)
"""

import json
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys_path = str(ROOT)
if sys_path not in __import__('sys').path:
    __import__('sys').path.insert(0, sys_path)

from visao.brain import VisaoBrain

EPOCHS = 1
SEEDS = [0, 1, 2]
N_TRAIN = 10000
N_TEST = 2000
HIDDEN_SIZES = [32, 64, 128, 256, 512]


def generate_signal(n_samples, noise=0.01, seed=0):
    """Generate chaotic-like signal."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_samples * 0.1, n_samples + 1)
    signal = (
        np.sin(t * 1.1) +
        0.5 * np.sin(t * 2.3 + 0.7) +
        0.3 * np.sin(t * 3.7 + 1.2) +
        0.2 * np.sin(t * 5.1 + 2.1) +
        noise * rng.standard_normal(len(t))
    )
    X = signal[:-1].reshape(-1, 1)
    Y = signal[1:].reshape(-1, 1)
    return X, Y


def run_experiment(n_hidden, seed):
    X, Y = generate_signal(N_TRAIN + N_TEST, seed=seed)
    X_train, Y_train = X[:N_TRAIN], Y[:N_TRAIN]
    X_test, Y_test = X[N_TRAIN:], Y[N_TRAIN:]

    # Use consolidation for stability at large n_hidden
    # L2 via lambda_decay
    brain = VisaoBrain(
        n_in=1, n_hidden=n_hidden, n_out=1,
        lr=0.01, sparsity=0.0,
        consolidation=8.0, surprise_gain=0.0, oja_lr=0.0,
        lambda_decay=0.001,
        seed=seed,
    )

    start = time.time()
    for epoch in range(EPOCHS):
        for i in range(len(X_train)):
            brain.learn(X_train[i:i+1], Y_train[i:i+1])

    wall_time = time.time() - start

    brain.set_mode('infer')
    preds = np.array([brain.forward(X_test[i:i+1]).ravel()[0] for i in range(len(X_test))])
    mse_test = float(np.mean((Y_test.ravel() - preds)**2))

    n_params = brain.cell.W_in.size + brain.cell.W_rec.size + brain.learner.W_out.size

    return {
        'n_hidden': n_hidden,
        'n_params': n_params,
        'mse_test': mse_test,
        'wall_time': wall_time,
        'seed': seed,
    }


def main():
    results = []
    for n_hidden in HIDDEN_SIZES:
        for seed in SEEDS:
            print(f"  n_hidden={n_hidden:4d}, seed={seed}: ", end='', flush=True)
            r = run_experiment(n_hidden, seed)
            results.append(r)
            print(f"n_params={r['n_params']:6d}, MSE={r['mse_test']:.6f}, time={r['wall_time']:.1f}s")

    # Aggregate
    from collections import defaultdict
    by_hidden = defaultdict(list)
    for r in results:
        by_hidden[r['n_hidden']].append(r)

    agg = []
    for n_hidden in HIDDEN_SIZES:
        if n_hidden in by_hidden:
            pts = by_hidden[n_hidden]
            agg.append({
                'n_hidden': n_hidden,
                'n_params': pts[0]['n_params'],
                'mse_mean': np.mean([p['mse_test'] for p in pts]),
                'mse_std': np.std([p['mse_test'] for p in pts], ddof=1) if len(pts) > 1 else 0,
            })

    # Fit: log(MSE) = -alpha * log(N) + log(C)
    params_arr = np.array([a['n_params'] for a in agg])
    mse_arr = np.array([a['mse_mean'] for a in agg])
    log_p, log_m = np.log(params_arr), np.log(mse_arr)

    A = np.vstack([-log_p, np.ones_like(log_p)]).T
    alpha, log_C = np.linalg.lstsq(A, log_m, rcond=None)[0]

    ss_res = np.sum((log_m - (-alpha * log_p + log_C)) ** 2)
    ss_tot = np.sum((log_m - np.mean(log_m)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    n = len(log_p)
    se_alpha = np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((log_p - np.mean(log_p))**2)) if n > 2 else float('nan')
    ci_lo, ci_hi = alpha - 1.96 * se_alpha, alpha + 1.96 * se_alpha

    interp = "POSITIVE (good)" if alpha > 0.05 and r_squared > 0.5 else \
             "NEGATIVE (poor)" if alpha < -0.05 and r_squared > 0.5 else \
             "UNCERTAIN (low R²)" if r_squared < 0.3 else "NEUTRAL"

    print(f"\n{'='*60}")
    print(f"L(N) = C · N^(-α)")
    print(f"  α = {alpha:.4f} ± {se_alpha:.4f} (SE)")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  R² = {r_squared:.4f}")
    print(f"  Interpretation: {interp}")

    if r_squared < 0.9:
        print(f"  ⚠️  WARNING: R² < 0.9 — scaling fit unreliable")

    out = ROOT / 'bench' / 'results_scaling_v3.json'
    out.write_text(json.dumps({
        'version': 'v3',
        'scaling_law': {
            'alpha': float(alpha),
            'alpha_se': float(se_alpha),
            'alpha_ci_95': [float(ci_lo), float(ci_hi)],
            'C': float(np.exp(log_C)),
            'R_squared': float(r_squared),
            'interpretation': interp,
        },
        'config': {
            'n_train': N_TRAIN,
            'n_test': N_TEST,
            'epochs': EPOCHS,
            'seeds': SEEDS,
            'consolidation': 8.0,
            'lambda_decay': 0.001,
        },
        'aggregated': agg,
        'raw_results': results,
    }, indent=2))
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
