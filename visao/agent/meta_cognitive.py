#!/usr/bin/env python3
"""meta_cognitive.py — Meta-cognição: integra Self-Reflection + Task Boundary + Meta-Learner.

O cérebro monitora seu próprio desempenho e ajusta seus parâmetros automaticamente.
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain
from visao.agent.self_reflection import SelfReflection, PerformanceTracker
from visao.continual.task_boundary import TaskBoundaryDetector, Consolidator
from visao.continual.meta_learner import MetaContinualLearner


class MetaCognitiveBrain:
    """Cérebro meta-cognitivo: monitora, avalia e ajusta seu próprio comportamento.
    
    Integra:
    1. SelfReflection: monitora métricas e sugere ajustes
    2. TaskBoundary: detecta mudanças de tarefa
    3. MetaLearner: adapta hiperparâmetros por tarefa
    4. Auto-ajuste: aplica sugestões automaticamente
    """
    
    def __init__(self, n_in: int = 8, n_hidden: int = 64, n_out: int = 1, 
                 seed: int = 42, auto_adjust: bool = True):
        self.brain = VisaoBrain(n_in, n_hidden, n_out, seed=seed)
        self.reflection = SelfReflection(self.brain)
        self.detector = TaskBoundaryDetector(threshold=2.0, window_size=20)
        self.consolidator = Consolidator(self.brain, strategy='ewc_boost')
        self.meta = MetaContinualLearner(n_in, n_hidden, n_out, seed=seed)
        self.auto_adjust = auto_adjust
        self.history: list[dict] = []
        self.n_adaptations = 0
    
    def learn(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Aprende com monitoramento meta-cognitivo."""
        # Aprender
        self.brain.set_mode('learn')
        result = self.brain.learn(x, y)
        
        # Refletir
        reflection = self.reflection.reflect(result['err'], result['surprise'])
        
        # Detectar fronteira
        boundary = self.detector.observe(result['err'], self.brain.x)
        if boundary['change_detected']:
            self.consolidator.consolidate_on_boundary()
            self.detector.reset()
            result['boundary_detected'] = True
        
        # Auto-ajuste
        if self.auto_adjust and reflection['n_suggestions'] > 0:
            applied = self.reflection.apply_suggestions(reflection)
            result['adaptation'] = applied
            self.n_adaptations += 1
        
        # Step do consolidator
        self.consolidator.step()
        
        result['reflection_stats'] = reflection['stats']
        return result
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.brain.set_mode('infer')
        return self.brain.forward(x)
    
    def set_mode(self, mode: str):
        self.brain.set_mode(mode)
        return self
    
    def get_meta_cognitive_state(self) -> dict:
        """Retorna estado meta-cognitivo completo."""
        return {
            'n_adaptations': self.n_adaptations,
            'reflection_summary': self.reflection.get_reflection_summary(),
            'detector_step': self.detector._step,
            'brain_lr': self.brain.learner.lr,
            'brain_consolidation': self.brain.learner.consolidation,
            'brain_mode': self.brain._mode,
            'tracker_stats': self.reflection.tracker.get_stats(),
        }
    
    def adapt_hyperparams(self, new_lr: float = None, new_consolidation: float = None):
        """Adapta hiperparâmetros do cérebro manualmente."""
        if new_lr is not None and isinstance(new_lr, (int, float)):
            self.brain.learner.lr = float(new_lr)
        if new_consolidation is not None and isinstance(new_consolidation, (int, float)):
            self.brain.learner.consolidation = float(new_consolidation)
        self.n_adaptations += 1


def demo_meta_cognitive():
    """Demonstração do MetaCognitiveBrain."""
    print("="*60)
    print("META-COGNITIVE BRAIN DEMO")
    print("="*60)
    
    brain = MetaCognitiveBrain(n_in=8, n_hidden=32, n_out=1, seed=42)
    
    print("\n[1] Simulando aprendizado com auto-ajuste...")
    
    for i in range(200):
        x = np.random.randn(8)
        y = np.array([np.sin(i * 0.05)])
        result = brain.learn(x, y)
        
        if 'adaptation' in result:
            print(f"  Step {i}: adaptation applied")
    
    print("\n[2] Estado meta-cognitivo:")
    state = brain.get_meta_cognitive_state()
    print(f"  Adaptações: {state['n_adaptations']}")
    print(f"  LR: {state['brain_lr']:.6f}")
    print(f"  Consolidação: {state['brain_consolidation']:.4f}")
    
    print("\n[3] Stats do tracker:")
    stats = state['tracker_stats']
    print(f"  Mean error: {stats.get('mean_error', 0):.6f}")
    print(f"  Mean surprise: {stats.get('mean_surprise', 0):.4f}")
    print(f"  Error trend: {stats.get('error_trend', 0):.4f}")
    
    print("\n" + "="*60)
    print("META-COGNITIVE BRAIN OK!")
    print("="*60)


if __name__ == "__main__":
    demo_meta_cognitive()
