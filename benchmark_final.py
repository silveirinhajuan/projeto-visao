#!/usr/bin/env python3
"""benchmark_final.py — Benchmark final: todas as versões do VisaoBrain."""

import numpy as np
import time
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from visao.brain import VisaoBrain
from visao.memory_profile import MemoryProfiler

print("="*70)
print("BENCHMARK FINAL — VISÃO")
print("="*70)

# Dados
rng = np.random.default_rng(42)
n = 1000
t = np.arange(n)
u = rng.normal(0, 1, (n, 2)) * 0.3
u[:, 1] += np.sin(t * 0.1)
y = np.convolve(u[:, 0], np.ones(30)/30.0, mode='same')[:, None]

results = {}

# 1. Baseline
print("\n[1] Baseline (Oja + consolidação + surpresa)...")
brain = VisaoBrain(2, 64, 1, seed=42)
errors = []
t0 = time.time()
for i in range(n):
    pred = brain.learn(u[i], y[i])
    errors.append(float(np.linalg.norm(y[i] - pred)))
t_learn = time.time() - t0
results['baseline'] = {'error': float(np.mean(errors[50:])), 'time': t_learn}
print(f"    Error: {results['baseline']['error']:.4f}, Time: {t_learn:.3f}s")

# 2. Meta-plasticidade
print("\n[2] Meta-plasticity...")
from visao.metaplasticity import MetaPlasticityLearner
brain_meta = VisaoBrain(2, 64, 1, seed=42, meta_learn=True)
errors_meta = []
t0 = time.time()
for i in range(n):
    pred = brain_meta.learn(u[i], y[i])
    errors_meta.append(float(np.linalg.norm(y[i] - pred)))
t_meta = time.time() - t0
results['meta'] = {'error': float(np.mean(errors_meta[50:])), 'time': t_meta}
print(f"    Error: {results['meta']['error']:.4f}, Time: {t_meta:.3f}s")

# 3. Quantização
print("\n[3] Quantization (int8)...")
from visao.ops.quantize import quantize_brain, measure_memory_savings
brain_q = quantize_brain(brain)
savings = measure_memory_savings(brain, brain_q)
results['quantize'] = {'memory_saving_pct': savings['space_saving_pct']}
print(f"    Memory saving: {savings['space_saving_pct']:.1f}%")

# 4. Modo inferência
print("\n[4] Inference mode...")
brain.set_mode('infer')
t0 = time.time()
for i in range(n):
    _ = brain.forward(u[i])
t_infer = time.time() - t0
results['infer'] = {'time': t_infer, 'speedup': t_learn / t_infer}
print(f"    Time: {t_infer:.3f}s, Speedup: {results['infer']['speedup']:.1f}x")

# 5. Memory profiling
print("\n[5] Memory profiling...")
profiler = MemoryProfiler()
with profiler.track():
    for i in range(n):
        _ = brain.forward(u[i])
results['memory'] = profiler.report()
print(f"    {results['memory']}")

# Resumo
print("\n" + "="*70)
print("RESUMO FINAL")
print("="*70)
print(f"{'Métrica':<25} {'Valor':<20}")
print("-"*45)
print(f"{'Error (baseline)':<25} {results['baseline']['error']:.4f}")
print(f"{'Error (meta)':<25} {results['meta']['error']:.4f}")
print(f"{'Memory saving (int8)':<25} {results['quantize']['memory_saving_pct']:.1f}%")
print(f"{'Inference speedup':<25} {results['infer']['speedup']:.1f}x")

# Salvar
with open('/home/juan/projeto-visao/benchmark_final_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSalvo em: /home/juan/projeto-visao/benchmark_final_results.json")
