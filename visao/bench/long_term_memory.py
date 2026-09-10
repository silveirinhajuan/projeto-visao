#!/usr/bin/env python3
"""test_long_term_memory.py — Benchmark de memória de longo prazo.

Testa a capacidade do VisaoBrain de:
1. Armazenar informações por longos períodos (1000+ steps)
2. Recuperar informações antigas sem interferência
3. Resistir a interferência de novas informações
4. Consolidar memórias via repetição espaçada
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


def test_long_term_storage(n_steps: int = 2000):
    """Testa armazenamento de longo prazo."""
    brain = VisaoBrain(8, 64, 1, seed=42)
    brain.set_mode('learn')
    
    # Fase 1: Treinar com padrão A por n_steps/2
    pattern_a = np.random.randn(8)
    target_a = np.array([1.0])
    
    errors_a = []
    for i in range(n_steps // 2):
        result = brain.learn(pattern_a, target_a)
        errors_a.append(result['err'])
    
    # Fase 2: Treinar com padrão B (interferente) por n_steps/2
    pattern_b = np.random.randn(8)
    target_b = np.array([-1.0])
    
    errors_b = []
    for i in range(n_steps // 2):
        result = brain.learn(pattern_b, target_b)
        errors_b.append(result['err'])
    
    # Teste: recuperar padrão A (após interferência de B)
    brain.set_mode('infer')
    brain.reset_state()
    output_a = brain.forward(pattern_a)
    error_recovery = float(np.mean((output_a - target_a) ** 2))
    
    return {
        'final_error_a': float(np.mean(errors_a[-10:])),
        'final_error_b': float(np.mean(errors_b[-10:])),
        'error_recovery_a': error_recovery,
        'interference_ratio': error_recovery / max(float(np.mean(errors_a[-10:])), 1e-8),
    }


def test_spaced_repetition(n_facts: int = 10, n_sessions: int = 5):
    """Testa consolidação via repetição espaçada."""
    brain = VisaoBrain(8, 64, 1, seed=42)
    
    # Criar fatos
    facts = [np.random.randn(8) for _ in range(n_facts)]
    targets = [np.array([np.sin(i)]) for i in range(n_facts)]
    
    # Sessões de repetição espaçada
    initial_errors = []
    final_errors = []
    
    for fact, target in zip(facts, targets):
        brain.set_mode('learn')
        brain.reset_state()
        result = brain.learn(fact, target)
        initial_errors.append(result['err'])
    
    # Sessões de repetição
    for session in range(n_sessions):
        for fact, target in zip(facts, targets):
            brain.set_mode('learn')
            brain.reset_state()
            brain.learn(fact, target)
    
    # Medir erros finais
    for fact, target in zip(facts, targets):
        brain.set_mode('infer')
        brain.reset_state()
        output = brain.forward(fact)
        final_errors.append(float(np.mean((output - target) ** 2)))
    
    return {
        'mean_initial_error': float(np.mean(initial_errors)),
        'mean_final_error': float(np.mean(final_errors)),
        'improvement_ratio': float(np.mean(initial_errors)) / max(float(np.mean(final_errors)), 1e-8),
        'n_sessions': n_sessions,
    }


def test_forgetting_curve(n_tasks: int = 5, steps_per_task: int = 200):
    """Testa curva de esquecimento."""
    brain = VisaoBrain(8, 64, 1, seed=42)
    
    tasks = []
    for i in range(n_tasks):
        pattern = np.random.randn(8)
        target = np.array([np.sin(i)])
        tasks.append((pattern, target))
    
    # Treinar cada tarefa sequencialmente
    task_errors = {}
    for task_idx, (pattern, target) in enumerate(tasks):
        brain.set_mode('learn')
        for step in range(steps_per_task):
            result = brain.learn(pattern, target)
        
        # Avaliar todas as tarefas anteriores
        for eval_idx in range(task_idx + 1):
            eval_pattern, eval_target = tasks[eval_idx]
            brain.set_mode('infer')
            brain.reset_state()
            output = brain.forward(eval_pattern)
            error = float(np.mean((output - eval_target) ** 2))
            task_errors[(eval_idx, task_idx)] = error
    
    # Calcular esquecimento
    forgetting = 0.0
    for i in range(n_tasks):
        best = min(task_errors.get((i, j), 1.0) for j in range(i, n_tasks))
        final = task_errors.get((i, n_tasks - 1), 1.0)
        forgetting += max(0, final - best)
    
    return {
        'forgetting': float(forgetting),
        'mean_error': float(np.mean(list(task_errors.values()))),
        'task_errors': {str(k): v for k, v in task_errors.items()},
    }


if __name__ == "__main__":
    print("="*60)
    print("LONG-TERM MEMORY BENCHMARK")
    print("="*60)
    
    print("\n[1] Long-term storage (2000 steps)")
    result = test_long_term_storage()
    print(f"    Final error A: {result['final_error_a']:.4f}")
    print(f"    Final error B: {result['final_error_b']:.4f}")
    print(f"    Recovery error A: {result['error_recovery_a']:.4f}")
    print(f"    Interference ratio: {result['interference_ratio']:.2f}")
    
    print("\n[2] Spaced repetition (5 sessions)")
    result = test_spaced_repetition()
    print(f"    Initial error: {result['mean_initial_error']:.4f}")
    print(f"    Final error: {result['mean_final_error']:.4f}")
    print(f"    Improvement: {result['improvement_ratio']:.2f}x")
    
    print("\n[3] Forgetting curve (5 tasks)")
    result = test_forgetting_curve()
    print(f"    Forgetting: {result['forgetting']:.4f}")
    print(f"    Mean error: {result['mean_error']:.4f}")
    
    print("\n" + "="*60)
    print("LONG-TERM MEMORY BENCHMARK OK!")
    print("="*60)
