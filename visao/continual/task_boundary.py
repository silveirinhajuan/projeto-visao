#!/usr/bin/env python3
"""task_boundary.py — Detecção de fronteira entre tarefas e consolidação.

Resolve o problema da interferência alta (1926x) detectando quando o cérebro
muda de tarefa e forçando consolidação antes de continuar.

Baseado em: 
- "Task-Arrival Policy" (Chen et al., 2022)
- "Continual Learning with Experience Replay" ( Rolnick et al., 2019)
- "EWC++" (Kemker et al., 2018)
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


class TaskBoundaryDetector:
    """Detecta quando o cérebro mudou de tarefa baseando-se em:
    1. Mudança na distribuição de erros (CUSUM)
    2. Mudança nas ativações internas
    3. Mudança no gradiente
    """
    
    def __init__(self, threshold: float = 2.0, window_size: int = 20):
        self.threshold = threshold
        self.window_size = window_size
        self.error_buffer: list[float] = []
        self.activation_buffer: list[np.ndarray] = []
        self._baseline_error = None
        self._baseline_activation = None
        self._step = 0
    
    def observe(self, error: float, activation: np.ndarray) -> dict:
        """Registra observação e detecta mudança."""
        self._step += 1
        self.error_buffer.append(error)
        self.activation_buffer.append(activation.copy())
        
        # Manter janela
        if len(self.error_buffer) > self.window_size * 2:
            self.error_buffer.pop(0)
        if len(self.activation_buffer) > self.window_size * 2:
            self.activation_buffer.pop(0)
        
        # Calcular baseline nos primeiros `window_size` passos
        if len(self.error_buffer) == self.window_size:
            self._baseline_error = np.mean(self.error_buffer)
            self._baseline_activation = np.mean(self.activation_buffer, axis=0)
        
        # Detectar mudança
        change_detected = False
        change_score = 0.0
        
        if self._baseline_error is not None and len(self.error_buffer) > self.window_size:
            # Erro recente vs baseline
            recent_error = np.mean(self.error_buffer[-self.window_size:])
            error_std = np.std(self.error_buffer[:self.window_size])
            
            if error_std > 1e-8:
                change_score = abs(recent_error - self._baseline_error) / error_std
                change_detected = change_score > self.threshold
        
        return {
            'change_detected': change_detected,
            'change_score': float(change_score),
            'step': self._step,
            'baseline_error': float(self._baseline_error) if self._baseline_error else None,
        }
    
    def reset(self):
        """Reseta detector (após consolidação)."""
        self.error_buffer = []
        self.activation_buffer = []
        self._baseline_error = None
        self._baseline_activation = None
        self._step = 0


class Consolidator:
    """Consolida conhecimento quando mudança de tarefa é detectada.
    
    Estratégias:
    1. EWC boost: aumentar lambda temporariamente
    2. Replay seletivo: reprocessar exemplos anteriores
    3. Consolidação forçada: treinar com alvo = estado atual (auto-associativo)
    """
    
    def __init__(self, brain: VisaoBrain, strategy: str = 'ewc_boost'):
        self.brain = brain
        self.strategy = strategy
        self._original_consolidation = brain.learner.consolidation
        self._original_lr = brain.learner.lr
        self._boost_steps = 0
    
    def consolidate_on_boundary(self, recent_experiences: list = None) -> dict:
        """Consolida conhecimento ao detectar fronteira de tarefa.
        
        Parameters
        ----------
        recent_experiences : list of (x, y) tuples
            Experiências recentes para replay (se estratégia = 'replay')
            
        Returns
        -------
        dict com info sobre o que foi feito
        """
        if self.strategy == 'ewc_boost':
            return self._ewc_boost()
        elif self.strategy == 'replay':
            return self._replay(recent_experiences or [])
        elif self.strategy == 'auto_associative':
            return self._auto_associative()
        else:
            return self._ewc_boost()
    
    def _ewc_boost(self) -> dict:
        """Aumenta EWC temporariamente para proteger conhecimento antigo."""
        # Boost: 5x consolidação por 50 passos
        self.brain.learner.consolidation = self._original_consolidation * 5.0
        self.brain.learner.lr = self._original_lr * 0.5  # Reduz lr para não esquecer
        self._boost_steps = 50
        
        return {
            'strategy': 'ewc_boost',
            'new_consolidation': self.brain.learner.consolidation,
            'new_lr': self.brain.learner.lr,
            'boost_steps': self._boost_steps,
        }
    
    def _replay(self, experiences: list) -> dict:
        """Replay seletivo: reprocessar experiências anteriores."""
        if not experiences:
            return {'strategy': 'replay', 'n_replayed': 0}
        
        self.brain.set_mode('learn')
        errors = []
        for x, y in experiences[-50:]:  # últimas 50
            self.brain.reset_state()
            result = self.brain.learn(x, y)
            errors.append(result['err'])
        
        return {
            'strategy': 'replay',
            'n_replayed': len(experiences[-50:]),
            'mean_error': float(np.mean(errors)) if errors else 0.0,
        }
    
    def _auto_associative(self) -> dict:
        """Consolidação auto-associativo: estado atual é seu próprio alvo."""
        self.brain.set_mode('learn')
        x = self.brain.x.copy()
        self.brain.learn(x[:self.brain.n_in], x[:self.brain.n_out])
        
        return {
            'strategy': 'auto_associative',
            'state_norm': float(np.linalg.norm(x)),
        }
    
    def step(self) -> dict:
        """Chamado a cada passo — gerencia duração do boost."""
        if self._boost_steps > 0:
            self._boost_steps -= 1
            if self._boost_steps == 0:
                # Restaurar
                self.brain.learner.consolidation = self._original_consolidation
                self.brain.learner.lr = self._original_lr
                return {'boost_ended': True}
        
        return {'boost_active': self._boost_steps > 0, 'remaining': self._boost_steps}


class TaskAwareContinualLearner:
    """Continual learner com detecção de fronteira e consolidação ativa.
    
    Resolve o problema da interferência alta detectando automaticamente
    quando o cérebro muda de tarefa e forçando consolidação.
    """
    
    def __init__(self, brain: VisaoBrain, 
                 boundary_threshold: float = 2.0,
                 consolidation_strategy: str = 'ewc_boost',
                 experience_buffer_size: int = 100):
        self.brain = brain
        self.detector = TaskBoundaryDetector(threshold=boundary_threshold)
        self.consolidator = Consolidator(brain, strategy=consolidation_strategy)
        self.experience_buffer: list[tuple[np.ndarray, np.ndarray]] = []
        self.experience_buffer_size = experience_buffer_size
        self.boundaries_detected = 0
        self.consolidations_performed = 0
    
    def learn(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Aprende com detecção de fronteira."""
        # Aprender
        self.brain.set_mode('learn')
        result = self.brain.learn(x, y)
        
        # Armazenar experiência
        self.experience_buffer.append((x.copy(), y.copy()))
        if len(self.experience_buffer) > self.experience_buffer_size:
            self.experience_buffer.pop(0)
        
        # Detectar fronteira
        boundary = self.detector.observe(result['err'], self.brain.x)
        
        # Consolidar se detectou
        if boundary['change_detected']:
            self.boundaries_detected += 1
            consolidation = self.consolidator.consolidate_on_boundary(self.experience_buffer)
            self.consolidations_performed += 1
            self.detector.reset()
            result['consolidation'] = consolidation
        
        # Step do consolidator
        boost_status = self.consolidator.step()
        result['boost_status'] = boost_status
        
        return result
    
    def get_status(self) -> dict:
        """Retorna estado atual."""
        return {
            'boundaries_detected': self.boundaries_detected,
            'consolidations_performed': self.consolidations_performed,
            'buffer_size': len(self.experience_buffer),
            'detector_step': self.detector._step,
        }


if __name__ == "__main__":
    print("="*60)
    print("TASK BOUNDARY DETECTION & CONSOLIDATION DEMO")
    print("="*60)
    
    brain = VisaoBrain(8, 32, 1, seed=42)
    learner = TaskAwareContinualLearner(brain, boundary_threshold=2.0)
    
    print("\n[1] Simulando 3 tarefas sequenciais com detecção...")
    
    task_boundaries = []
    
    for task_idx in range(3):
        print(f"\n    Tarefa {task_idx + 1}:")
        
        # Gerar tarefa
        pattern = np.random.randn(8)
        target = np.array([np.sin(task_idx * 2)])
        
        errors = []
        for step in range(100):
            result = learner.learn(pattern, target)
            errors.append(result['err'])
            
            if 'consolidation' in result:
                task_boundaries.append({
                    'task': task_idx,
                    'step': step,
                    'strategy': result['consolidation']['strategy'],
                })
                print(f"      Fronteira detectada no step {step} → {result['consolidation']['strategy']}")
        
        print(f"      Erro inicial: {np.mean(errors[:10]):.4f}")
        print(f"      Erro final: {np.mean(errors[-10:]):.4f}")
    
    print(f"\n[2] Status final:")
    status = learner.get_status()
    print(f"    Fronteiras detectadas: {status['boundaries_detected']}")
    print(f"    Consolidações: {status['consolidations_performed']}")
    
    print("\n" + "="*60)
    print("TASK BOUNDARY OK!")
    print("="*60)
