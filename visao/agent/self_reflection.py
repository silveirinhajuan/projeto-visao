#!/usr/bin/env python3
"""self_reflection.py — Auto-reflexão e auto-melhoria do VisaoCognitiveBrain.

O cérebro avalia seu próprio desempenho e sugere melhorias.
Baseado em: 
- "Self-Refine" (Madaan et al., 2023)
- "Reflexion" (Shinn et al., 2023)
- "RISE" (Zhao et al., 2025)
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


class PerformanceTracker:
    """Rastreia métricas de desempenho ao longo do tempo."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.errors: list[float] = []
        self.surprises: list[float] = []
        self.learn_rates: list[float] = []
        self.consolidations: list[float] = []
        self.timestamps: list[float] = []
    
    def record(self, error: float, surprise: float, lr: float, consolidation: float):
        """Registra métricas de um passo."""
        self.errors.append(error)
        self.surprises.append(surprise)
        self.learn_rates.append(lr)
        self.consolidations.append(consolidation)
        self.timestamps.append(time.time())
        
        # Manter janela
        if len(self.errors) > self.window_size:
            self.errors.pop(0)
            self.surprises.pop(0)
            self.learn_rates.pop(0)
            self.consolidations.pop(0)
            self.timestamps.pop(0)
    
    def get_trend(self, metric: str = 'error') -> float:
        """Calcula tendência de uma métrica (-1 = melhorando, 0 = estável, 1 = piorando)."""
        values = getattr(self, metric + 's', [])
        if len(values) < 10:
            return 0.0
        
        # Comparar primeira e segunda metade
        mid = len(values) // 2
        first_half = np.mean(values[:mid])
        second_half = np.mean(values[mid:])
        
        if first_half == 0:
            return 0.0
        
        change = (second_half - first_half) / first_half
        return float(np.clip(change, -1.0, 1.0))
    
    def get_stats(self) -> dict:
        """Retorna estatísticas atuais."""
        if not self.errors:
            return {'n_samples': 0}
        
        return {
            'n_samples': len(self.errors),
            'mean_error': float(np.mean(self.errors)),
            'std_error': float(np.std(self.errors)),
            'mean_surprise': float(np.mean(self.surprises)),
            'mean_lr': float(np.mean(self.learn_rates)),
            'mean_consolidation': float(np.mean(self.consolidations)),
            'error_trend': self.get_trend('error'),
            'surprise_trend': self.get_trend('surprise'),
        }


class SelfReflection:
    """Auto-reflexão: o cérebro avalia e ajusta seu próprio comportamento."""
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
        self.tracker = PerformanceTracker()
        self.reflection_log: list[dict] = []
    
    def reflect(self, error: float, surprise: float) -> dict:
        """Reflete sobre o desempenho atual e sugere ajustes."""
        lr = self.brain.learner.lr
        consolidation = self.brain.learner.consolidation
        
        self.tracker.record(error, surprise, lr, consolidation)
        stats = self.tracker.get_stats()
        
        suggestions = []
        
        # Análise de tendência
        if stats.get('error_trend', 0) > 0.2:
            suggestions.append({
                'issue': 'error_increasing',
                'action': 'increase_lr',
                'magnitude': 1.2,
            })
        elif stats.get('error_trend', 0) < -0.2:
            suggestions.append({
                'issue': 'error_decreasing',
                'action': 'maintain_lr',
                'magnitude': 1.0,
            })
        
        # Análise de surpresa
        if stats.get('mean_surprise', 0) > 2.0:
            suggestions.append({
                'issue': 'high_surprise',
                'action': 'increase_consolidation',
                'magnitude': 1.5,
            })
        
        # Análise de convergência
        if stats.get('std_error', 1.0) < 0.01 and stats.get('mean_error', 1.0) < 0.1:
            suggestions.append({
                'issue': 'converged',
                'action': 'reduce_lr',
                'magnitude': 0.8,
            })
        
        reflection = {
            'timestamp': time.time(),
            'stats': stats,
            'suggestions': suggestions,
            'n_suggestions': len(suggestions),
        }
        
        self.reflection_log.append(reflection)
        return reflection
    
    def apply_suggestions(self, reflection: dict) -> dict:
        """Aplica sugestões de ajuste ao cérebro."""
        applied = []
        
        for suggestion in reflection.get('suggestions', []):
            action = suggestion['action']
            mag = suggestion['magnitude']
            
            if action == 'increase_lr':
                self.brain.learner.lr *= mag
                applied.append(f"lr *= {mag}")
            elif action == 'reduce_lr':
                self.brain.learner.lr *= mag
                applied.append(f"lr *= {mag}")
            elif action == 'increase_consolidation':
                self.brain.learner.consolidation *= mag
                applied.append(f"consolidation *= {mag}")
            elif action == 'maintain_lr':
                applied.append("lr maintained")
        
        return {
            'applied': applied,
            'n_applied': len(applied),
            'new_lr': self.brain.learner.lr,
            'new_consolidation': self.brain.learner.consolidation,
        }
    
    def get_reflection_summary(self) -> dict:
        """Resumo das reflexões."""
        if not self.reflection_log:
            return {'n_reflections': 0}
        
        recent = self.reflection_log[-10:]
        return {
            'n_reflections': len(self.reflection_log),
            'recent_suggestions': sum(r['n_suggestions'] for r in recent),
            'avg_error_trend': np.mean([r['stats'].get('error_trend', 0) for r in recent]),
            'avg_surprise_trend': np.mean([r['stats'].get('surprise_trend', 0) for r in recent]),
        }


if __name__ == "__main__":
    print("="*60)
    print("SELF-REFLECTION DEMO")
    print("="*60)
    
    brain = VisaoBrain(8, 32, 1, seed=42)
    reflector = SelfReflection(brain)
    
    print("\n[1] Simulando aprendizado com reflexão...")
    brain.set_mode('learn')
    
    for i in range(100):
        x = np.random.randn(8)
        y = np.array([np.sin(i * 0.1)])
        result = brain.learn(x, y)
        
        # Refletir a cada 10 passos
        if i % 10 == 0:
            reflection = reflector.reflect(result['err'], result['surprise'])
            if reflection['n_suggestions'] > 0:
                print(f"    Step {i}: {reflection['n_suggestions']} sugestões")
                for s in reflection['suggestions']:
                    print(f"      → {s['issue']}: {s['action']}")
    
    print("\n[2] Aplicando sugestões...")
    if reflector.reflection_log:
        last = reflector.reflection_log[-1]
        applied = reflector.apply_suggestions(last)
        print(f"    Aplicadas: {applied['n_applied']}")
        print(f"    Novo lr: {applied['new_lr']:.4f}")
    
    print("\n[3] Resumo...")
    summary = reflector.get_reflection_summary()
    print(f"    Total de reflexões: {summary['n_reflections']}")
    print(f"    Sugestões recentes: {summary['recent_suggestions']}")
    
    print("\n" + "="*60)
    print("SELF-REFLECTION OK!")
    print("="*60)
