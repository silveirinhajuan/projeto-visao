#!/usr/bin/env python3
"""example_full_pipeline.py — Exemplo completo: VisaoBrain + MetaPlasticity + MemoryProfile."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from visao.brain import VisaoBrain
from visao.memory_profile import MemoryProfiler, EcoMode, profile_memory

print("="*70)
print("VISÃO — Full Pipeline Demo")
print("="*70)

# 1. Criar o cérebro
print("\n[1] Criando VisaoBrain...")
brain = VisaoBrain(n_in=2, n_hidden=64, n_out=1, seed=42)
print(f"    {brain}")

# 2. Criar dados sintéticos (3 tarefas)
print("\n[2] Gerando tarefas...")
rng = np.random.default_rng(42)

def make_task(name, n=500):
    t = np.arange(n)
    h = abs(hash(name)) % 1000
    phase = (h / 1000.0) * 2 * np.pi
    freq = 0.05 + (h % 7) * 0.04
    
    u = rng.normal(0, 1, (n, 2)) * 0.3
    u[:, 1] += np.sin(t * freq + phase)
    y = np.convolve(u[:, 0], np.ones(30)/30.0, mode='same')[:, None]
    return u, y

tasks = [make_task(f"task_{i}") for i in range(3)]

# 3. Treinar com profiling
print("\n[3] Treinando com MemoryProfiler...")
profiler = MemoryProfiler()

for task_id, (u, y) in enumerate(tasks):
    brain.set_mode('learn')
    errors = []
    for i in range(len(u)):
        pred = brain.learn(u[i], y[i])
        err = float(np.linalg.norm(y[i] - pred))
        errors.append(err)
    
    print(f"    Tarefa {task_id}: erro médio = {np.mean(errors[50:]):.4f}")
    
    # Detectar forgetting em tarefas anteriores
    if task_id > 0:
        prev_u, prev_y = tasks[task_id - 1]
        prev_errors = []
        brain.set_mode('infer')
        for i in range(len(prev_u)):
            pred = brain.forward(prev_u[i])
            err = float(np.linalg.norm(prev_y[i] - pred))
            prev_errors.append(err)
        print(f"    Tarefa {task_id-1} (forgetting check): erro = {np.mean(prev_errors):.4f}")

print(f"    {profiler.report()}")

# 4. Testar modo Eco
print("\n[4] Testando EcoMode (64 → 16 neurônios)...")
eco = EcoMode(brain, target_neurons=16)
eco.enable()
brain.set_mode('infer')
u_test, y_test = tasks[0]
errors_eco = [float(np.linalg.norm(y_test[i] - brain.forward(u_test[i]))) for i in range(len(u_test))]
print(f"    Erro modo eco: {np.mean(errors_eco):.4f}")
eco.disable()

# 5. Testar metaplasticidade
print("\n[5] Testando MetaPlasticityLearner...")
from visao.metaplasticity import MetaPlasticityLearner

meta_learner = MetaPlasticityLearner(64, 1, rng=rng)
print(f"    lr_per_neuron: {meta_learner.lr_per_neuron}")

# 6. Salvar e carregar
print("\n[6] Save/load test...")
brain.save("/tmp/visao_brain.json")
brain_loaded = VisaoBrain.load("/tmp/visao_brain.json")
print(f"    Original: {brain}")
print(f"    Loaded:   {brain_loaded}")

# 7. Verificar save/load funcionou
brain.set_mode('infer')
brain_loaded.set_mode('infer')
u_test = rng.normal(0, 1, 2)
pred_orig = brain.forward(u_test)
pred_loaded = brain_loaded.forward(u_test)
print(f"    Pred original: {pred_orig}")
print(f"    Pred loaded:   {pred_loaded}")
print(f"    Match: {np.allclose(pred_orig, pred_loaded)}")

print("\n" + "="*70)
print("PIPELINE COMPLETO — OK!")
print("="*70)
