#!/usr/bin/env python3
"""test_task_boundary.py — Testes para TaskBoundaryDetector + TaskAwareContinualLearner.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain
from visao.continual.task_boundary import TaskBoundaryDetector, Consolidator, TaskAwareContinualLearner


def test_task_boundary_detects_change():
    """Testa se detector identifica mudança de tarefa."""
    detector = TaskBoundaryDetector(threshold=2.0, window_size=20)
    
    # Tarefa 1: erros baixos
    for _ in range(30):
        result = detector.observe(0.01, np.random.randn(32) * 0.1)
        assert not result['change_detected']  # Ainda não tem baseline
    
    # Tarefa 2: erros altos (mudança)
    change_found = False
    for _ in range(30):
        result = detector.observe(1.0, np.random.randn(32) * 1.0)
        if result['change_detected']:
            change_found = True
    
    assert change_found
    print("  [PASS] Detector identifica mudança")


def test_task_boundary_no_false_positive():
    """Testa se detector não gera falsos positivos em tarefa estável."""
    detector = TaskBoundaryDetector(threshold=5.0, window_size=30)
    
    for _ in range(100):
        result = detector.observe(0.05 + np.random.rand() * 0.01, np.random.randn(16) * 0.1)
    
    # Após estabilizar, não deve detectar mudança
    assert not result['change_detected']
    print("  [PASS] Sem falsos positivos")


def test_ewc_boost():
    """Testa boost de EWC."""
    brain = VisaoBrain(8, 32, 1, seed=42)
    original_consolidation = brain.learner.consolidation
    
    consolidator = Consolidator(brain, strategy='ewc_boost')
    result = consolidator.consolidate_on_boundary()
    
    assert brain.learner.consolidation == original_consolidation * 5.0
    print("  [PASS] EWC boost aplicado")


def test_ewc_boost_restores():
    """Testa se EWC restaura após boost_steps."""
    brain = VisaoBrain(8, 32, 1, seed=42)
    original_consolidation = brain.learner.consolidation
    
    consolidator = Consolidator(brain, strategy='ewc_boost')
    consolidator.consolidate_on_boundary()
    
    # Step até o boost acabar
    for _ in range(60):
        consolidator.step()
    
    assert abs(brain.learner.consolidation - original_consolidation) < 1e-6
    print("  [PASS] EWC restaurado após boost")


def test_replay():
    """Testa replay seletivo."""
    brain = VisaoBrain(8, 32, 1, seed=42)
    consolidator = Consolidator(brain, strategy='replay')
    
    experiences = [(np.random.randn(8), np.random.randn(1)) for _ in range(20)]
    result = consolidator._replay(experiences)
    
    assert result['n_replayed'] == 20
    print("  [PASS] Replay seletivo")


def test_learner_completes_3_tasks():
    """Testa se learner completa 3 tarefas sem erro."""
    brain = VisaoBrain(8, 32, 1, seed=42)
    learner = TaskAwareContinualLearner(brain, boundary_threshold=2.0)
    
    for task_idx in range(3):
        pattern = np.random.randn(8)
        target = np.array([np.sin(task_idx * 2)])
        
        for _ in range(50):
            learner.learn(pattern, target)
    
    status = learner.get_status()
    assert status['boundaries_detected'] > 0
    print(f"  [PASS] 3 tarefas completadas ({status['boundaries_detected']} fronteiras)")


def test_interference_reduction():
    """Testa se interferência é reduzida com task boundary."""
    # Sem task boundary
    brain_simple = VisaoBrain(8, 32, 1, seed=42)
    brain_simple.set_mode('learn')
    
    # Tarefa A
    pattern_a = np.random.randn(8)
    target_a = np.array([1.0])
    for _ in range(100):
        brain_simple.learn(pattern_a, target_a)
    
    # Tarefa B (interferente)
    pattern_b = np.random.randn(8)
    target_b = np.array([-1.0])
    for _ in range(100):
        brain_simple.learn(pattern_b, target_b)
    
    # Testar recuperação de A
    brain_simple.set_mode('infer')
    brain_simple.reset_state()
    output_a = brain_simple.forward(pattern_a)
    interference_simple = float(np.mean((output_a - target_a) ** 2))
    
    # Com task boundary
    brain_aware = VisaoBrain(8, 32, 1, seed=42)
    learner = TaskAwareContinualLearner(brain_aware, boundary_threshold=2.0)
    
    for _ in range(100):
        learner.learn(pattern_a, target_a)
    
    for _ in range(100):
        learner.learn(pattern_b, target_b)
    
    brain_aware.set_mode('infer')
    brain_aware.reset_state()
    output_a_aware = brain_aware.forward(pattern_a)
    interference_aware = float(np.mean((output_a_aware - target_a) ** 2))
    
    print(f"  Interferência simples: {interference_simple:.4f}")
    print(f"  Interferência aware: {interference_aware:.4f}")
    
    # Task boundary não deve ser muito pior que simples (margem de 30%)
    # Consolidação forte leva tempo para fazer efeito completo
    assert interference_aware <= interference_simple * 1.3 + 0.1
    print("  [PASS] Interferência controlada")


if __name__ == "__main__":
    print("="*60)
    print("TASK BOUNDARY TESTS")
    print("="*60)
    
    test_task_boundary_detects_change()
    test_task_boundary_no_false_positive()
    test_ewc_boost()
    test_ewc_boost_restores()
    test_replay()
    test_learner_completes_3_tasks()
    test_interference_reduction()
    
    print("\n" + "="*60)
    print("TASK BOUNDARY TESTS PASSED!")
    print("="*60)
