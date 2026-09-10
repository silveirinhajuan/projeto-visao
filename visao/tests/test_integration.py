#!/usr/bin/env python3
"""test_integration.py — Teste de integração do VisaoCognitiveBrain.

Testa o fluxo completo: percepção → memória → raciocínio → ação.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.integrate import VisaoCognitiveBrain


def test_cognitive_brain_think():
    """Testa o pipeline cognitivo completo."""
    brain = VisaoCognitiveBrain(n_in=8, n_hidden=32, n_out=1, seed=42)
    brain.set_mode('infer')
    
    x = np.random.randn(8)
    result = brain.think(x)
    
    assert 'perception' in result
    assert 'memory_recall' in result
    assert 'reasoning' in result
    assert 'action' in result
    print("    [PASS] think() pipeline completo")


def test_cognitive_brain_plan_and_execute():
    """Testa planejamento e execução."""
    brain = VisaoCognitiveBrain(n_in=8, n_hidden=32, n_out=1, seed=42)
    brain.set_mode('infer')
    
    result = brain.plan_and_execute("test task")
    
    assert 'success' in result
    assert 'subtasks_completed' in result
    assert result['subtasks_completed'] > 0
    print("    [PASS] plan_and_execute()")


def test_cognitive_brain_remember():
    """Testa armazenamento na memória."""
    brain = VisaoCognitiveBrain(n_in=8, n_hidden=32, n_out=1, seed=42)
    brain.set_mode('infer')
    
    event = {
        'input': np.random.randn(8),
        'output': np.random.randn(1),
        'reward': 0.5,
        'task_id': 0
    }
    brain.remember(event)
    print("    [PASS] remember()")


def test_cognitive_brain_save_load():
    """Testa save/load."""
    brain = VisaoCognitiveBrain(n_in=8, n_hidden=32, n_out=1, seed=42)
    brain.set_mode('infer')
    
    path = '/tmp/test_brain_state.json'
    brain.save(path)
    brain.load(path)
    print("    [PASS] save/load")


def test_cognitive_brain_demo():
    """Testa demo."""
    from visao.integrate import demo_cognitive_brain
    demo_cognitive_brain()  # Just check it doesn't crash
    print("    [PASS] demo_cognitive_brain()")


if __name__ == "__main__":
    print("="*60)
    print("INTEGRATION TEST — VisaoCognitiveBrain")
    print("="*60)
    
    test_cognitive_brain_think()
    test_cognitive_brain_plan_and_execute()
    test_cognitive_brain_remember()
    test_cognitive_brain_save_load()
    test_cognitive_brain_demo()
    
    print("\n" + "="*60)
    print("INTEGRATION TESTS PASSED!")
    print("="*60)
