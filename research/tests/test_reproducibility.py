#!/usr/bin/env python3
"""
test_reproducibility.py — Verifica que o research harness produz resultados reproduzíveis.
Um dos critérios centrais do Plano de Evolução Profunda.
"""

import json
import subprocess
import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).parent
CONFIG = "configs/continual_5tasks.yaml"
SEED = 42

def run_experiment(seed: int) -> dict:
    """Run the experiment and return metrics."""
    result = subprocess.run(
        [sys.executable, "-m", "research.runner",
         "--config", CONFIG, "--no_plots", "--seed", str(seed)],
        capture_output=True, text=True, cwd=RESEARCH_DIR.parent,
        timeout=120
    )
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Experiment failed: {result.returncode}")
    
    # Read metrics
    metrics_path = RESEARCH_DIR.parent / "results" / "continual_5tasks" / "metrics.json"
    with open(metrics_path) as f:
        return json.load(f)

def test_reproducibility():
    """Same seed must produce same results."""
    print("=== Teste de Reprodutibilidade ===")
    print(f"Config: {CONFIG}, Seed: {SEED}")
    
    # Run twice with same seed
    print("\nRun 1...")
    m1 = run_experiment(SEED)
    print("Run 2...")
    m2 = run_experiment(SEED)
    
    # Compare continual_metrics
    cm1 = m1.get("continual_metrics", {})
    cm2 = m2.get("continual_metrics", {})
    
    all_pass = True
    for key in ["average_accuracy", "forgetting", "backward_transfer"]:
        v1 = cm1.get(key, {})
        v2 = cm2.get(key, {})
        
        if isinstance(v1, dict) and isinstance(v2, dict):
            for model in v1:
                diff = abs(v1[model] - v2[model])
                status = "✓" if diff < 1e-10 else "✗"
                if diff >= 1e-10:
                    all_pass = False
                print(f"  {status} {key}/{model}: {v1[model]:.6f} vs {v2[model]:.6f} (diff={diff:.2e})")
    
    if all_pass:
        print("\n✅ REPRODUTÍVEL: Mesmo seed = mesmos resultados")
    else:
        print("\n❌ NÃO REPRODUTÍVEL")
        sys.exit(1)

def test_seed_variation():
    """Different seeds should produce different results."""
    print("\n=== Teste de Variação entre Seeds ===")
    
    results = {}
    for seed in [42, 43, 44]:
        m = run_experiment(seed)
        cm = m.get("continual_metrics", {})
        results[seed] = cm.get("average_accuracy", {})
    
    # Check that different seeds give different results
    vals = list(results.values())
    all_same = all(v == vals[0] for v in vals)
    
    if not all_same:
        print("✅ Seeds diferentes produzem resultados diferentes")
        for seed, acc in results.items():
            print(f"  seed={seed}: {acc}")
    else:
        print("⚠️  Todos os seeds iguais (pode ser esperado para alguns modelos)")

if __name__ == "__main__":
    test_reproducibility()
    test_seed_variation()
