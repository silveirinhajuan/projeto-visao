#!/usr/bin/env python3
"""infer_mode.py — Tarefa 8.1: Modo inferência eficiente (forward-only)."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from visao.brain import VisaoBrain


def benchmark_infer_vs_learn(n_trials=100):
    """Compara velocidade: modo learn vs modo infer."""
    print("="*60)
    print("BENCHMARK: infer vs learn mode")
    print("="*60)
    
    brain = VisaoBrain(2, 64, 1, seed=42)
    
    # Dados sintéticos
    rng = np.random.default_rng(0)
    u = rng.normal(0, 1, (1000, 2))
    y = rng.normal(0, 1, (1000, 1))
    
    # Modo learn (treinamento completo)
    brain.set_mode('learn')
    t0 = time.time()
    for i in range(len(u)):
        brain.learn(u[i], y[i])
    t_learn = time.time() - t0
    
    # Modo inferência (forward only)
    brain.set_mode('infer')
    t0 = time.time()
    for i in range(len(u)):
        _ = brain.forward(u[i])
    t_infer = time.time() - t0
    
    print(f"\nModo LEARN:  {t_learn:.3f}s ({t_learn/len(u)*1000:.3f}ms/sample)")
    print(f"Modo INFER:  {t_infer:.3f}s ({t_infer/len(u)*1000:.3f}ms/sample)")
    print(f"Speedup:     {t_learn/t_infer:.1f}x")
    
    return {
        't_learn': t_learn,
        't_infer': t_infer,
        'speedup': t_learn / t_infer,
    }


def test_correctness():
    """Verifica que modo infer produz mesmas previsões que learn (sem atualizar)."""
    print("\n" + "="*60)
    print("CORTEÇÃO: infer mode não altera pesos")
    print("="*60)
    
    brain = VisaoBrain(2, 32, 1, seed=42)
    
    # Forward no modo infer
    brain.set_mode('infer')
    u = np.array([0.5, -0.3])
    pred1 = brain.forward(u)
    
    # Forward novamente (mesma predição)
    pred2 = brain.forward(u)
    
    print(f"\nPredição 1: {pred1}")
    print(f"Predição 2: {pred2}")
    print(f"Iguais: {np.allclose(pred1, pred2)}")
    
    # Verificar que pesos não mudaram
    import copy
    brain2 = VisaoBrain(2, 32, 1, seed=42)
    brain2.set_mode('infer')
    state_before = copy.deepcopy(brain2.__dict__)
    
    for _ in range(100):
        brain2.forward(np.random.randn(2))
    
    # Pesos devem ser iguais
    weights_same = np.allclose(brain2.learner.W_out, state_before['learner'].W_out)
    print(f"Pesos inalterados no modo infer: {weights_same}")


if __name__ == "__main__":
    results = benchmark_infer_vs_learn()
    test_correctness()
    
    print("\n" + "="*60)
    print("RESULTADO")
    print("="*60)
    print(f"Speedup do modo infer: {results['speedup']:.1f}x")
