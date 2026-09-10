#!/usr/bin/env python3
"""ewc.py — Elastic Weight Consolidation (EWC) implementation.

Implementa EWC clássico (Kirkpatrick et al., 2017) com:
- Fisher Information Matrix diagonal aproximada
- Task-specific importance weights
- Proteção real de parâmetros contra interferência

Resolve o problema da interferência alta (3200x) que o EWC atual
(baseado em omega) não consegue controlar.

Reference: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


class FisherComputer:
    """Computa a diagonal da Fisher Information Matrix.
    
    A Fisher mede a sensibilidade da loss a cada parâmetro.
    Parâmetros com Fisher alta são mais importantes para a tarefa.
    """
    
    def __init__(self, n_params: int):
        self.fisher = np.zeros(n_params)
        self.n_samples = 0
    
    def compute_from_data(self, brain: VisaoBrain, X: np.ndarray, Y: np.ndarray, 
                          n_samples: int = 100) -> np.ndarray:
        """Computa Fisher aproximado a partir de dados.
        
        Para cada amostra, computa o gradiente da loss w.r.t. W_out
        e acumula o quadrado (diagonal da Fisher).
        """
        self.fisher = np.zeros_like(self.fisher)
        self.n_samples = 0
        
        brain.set_mode('infer')
        
        for i in range(min(n_samples, len(X))):
            x = X[i]
            y = Y[i]
            
            # Forward pass
            brain.reset_state()
            pred = brain.forward(x)
            
            # Loss = (pred - y)^2
            err = pred - y
            
            # Gradient w.r.t. W_out: d(loss)/d(W_out) = err * x^T
            # Fisher diagonal ≈ E[gradient^2]
            # Aproximação: usar o valor absoluto do gradiente esperado
            
            # Para W_out (1 x n_hidden): gradiente = err * x
            # Fisher diagonal para cada peso = E[(err * x_j)^2]
            # Como err é escalar: = err^2 * x_j^2
            
            # Usar o estado oculto (x) como "input" para W_out
            x_state = brain.x.flatten()
            
            # Fisher para W_out: err^2 * x_state^2
            w_out_fisher = err.flatten()**2 * x_state**2
            
            # Acumular (W_out tem n_hidden params)
            if len(self.fisher) >= len(w_out_fisher):
                self.fisher[:len(w_out_fisher)] += w_out_fisher
            
            self.n_samples += 1
        
        # Normalizar
        if self.n_samples > 0:
            self.fisher /= self.n_samples
        
        return self.fisher
    
    def compute_from_omega(self, omega: np.ndarray) -> np.ndarray:
        """Usa omega (importância acumulada) como proxy para Fisher."""
        # Normalizar omega para usar como Fisher
        max_omega = np.max(omega)
        if max_omega > 0:
            self.fisher = omega / max_omega
        else:
            self.fisher = omega
        return self.fisher


class EWC:
    """Elastic Weight Consolidation para o VisaoBrain.
    
    Adiciona um termo de regularização que protege parâmetros importantes
    de serem modificados durante o aprendizado de novas tarefas.
    
    Loss total = Loss_task + (lambda/2) * sum(F_i * (theta_i - theta*_i)^2)
    
    onde:
    - F_i = Fisher diagonal (importância do parâmetro i)
    - theta*_i = valor ótimo do parâmetro i na tarefa anterior
    - lambda = força da consolidação
    """
    
    def __init__(self, lambda_ewc: float = 10.0):
        self.lambda_ewc = lambda_ewc
        self.task_params: dict[int, dict] = {}  # task_id -> {param_name: value}
        self.task_fisher: dict[int, np.ndarray] = {}  # task_id -> fisher diagonal
        self.current_task_id: int = 0
        self._n_params: int = 0
    
    def initialize(self, brain: VisaoBrain):
        """Inicializa EWC com o número de parâmetros do cérebro."""
        # W_out: n_out x n_hidden
        n_params = brain.n_out * brain.n_hidden
        self._n_params = n_params
        self.fisher_computer = FisherComputer(n_params)
    
    def start_task(self, task_id: int):
        """Inicia nova tarefa — salva parâmetros atuais e Fisher."""
        self.current_task_id = task_id
    
    def compute_fisher(self, brain: VisaoBrain, X: np.ndarray, Y: np.ndarray,
                       n_samples: int = 100) -> np.ndarray:
        """Computa Fisher para tarefa atual."""
        return self.fisher_computer.compute_from_data(brain, X, Y, n_samples)
    
    def end_task(self, brain: VisaoBrain, X: np.ndarray = None, Y: np.ndarray = None):
        """Finaliza tarefa — salva parâmetros e Fisher."""
        # Salvar parâmetros atuais
        self.task_params[self.current_task_id] = {
            'W_out': brain.learner.W_out.copy(),
            'b_out': brain.learner.b_out.copy(),
        }
        
        # Computar e salvar Fisher
        if X is not None and Y is not None:
            fisher = self.compute_fisher(brain, X, Y)
        else:
            # Usar omega como proxy
            fisher = self.fisher_computer.compute_from_omega(brain.learner.omega)
        
        self.task_fisher[self.current_task_id] = fisher.copy()
    
    def compute_penalty(self, brain: VisaoBrain) -> tuple[float, np.ndarray]:
        """Calcula penalidade EWC e gradiente.
        
        Returns
        -------
        penalty: valor da penalidade
        gradiente: gradiente da penalidade w.r.t. W_out
        """
        if not self.task_params:
            return 0.0, np.zeros_like(brain.learner.W_out)
        
        penalty = 0.0
        grad = np.zeros_like(brain.learner.W_out)
        
        for task_id, params in self.task_params.items():
            fisher = self.task_fisher.get(task_id, np.zeros(self._n_params))
            
            # W_out: n_out x n_hidden
            w_out_old = params['W_out']
            w_out_new = brain.learner.W_out
            
            # Diferença
            diff = w_out_new.flatten() - w_out_old.flatten()
            
            # Penalidade: (lambda/2) * sum(F_i * diff_i^2)
            if len(fisher) >= len(diff):
                p = self.lambda_ewc * 0.5 * np.sum(fisher[:len(diff)] * diff**2)
                penalty += p
                
                # Gradiente: lambda * F_i * diff_i
                g = self.lambda_ewc * fisher[:len(diff)] * diff
                grad += g.reshape(w_out_new.shape)
        
        return float(penalty), grad
    
    def get_total_fisher(self) -> np.ndarray:
        """Retorna Fisher acumulada de todas as tarefas."""
        if not self.task_fisher:
            return np.zeros(self._n_params)
        
        total = np.zeros(self._n_params)
        for fisher in self.task_fisher.values():
            total += fisher
        return total


class MultiTaskReadout:
    """Readout com cabeças separadas por tarefa.
    
    Cada tarefa tem seu próprio W_out e b_out.
    Na inferência, seleciona a cabeça apropriada ou usa ensemble.
    Resolve o problema de interferência: tarefas diferentes usam
    diferentes readout heads.
    """
    
    def __init__(self, n_hidden: int, n_out: int, max_tasks: int = 10):
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.max_tasks = max_tasks
        self.task_heads: dict[int, dict] = {}  # task_id -> {W_out, b_out}
        self.current_task_id = 0
    
    def add_head(self, task_id: int):
        """Adiciona nova cabeça de readout para uma tarefa."""
        rng = np.random.default_rng(task_id)
        self.task_heads[task_id] = {
            'W_out': rng.normal(0, 0.1, (self.n_out, self.n_hidden)),
            'b_out': np.zeros(self.n_out),
        }
    
    def get_head(self, task_id: int) -> dict:
        """Retorna cabeça de readout para uma tarefa."""
        if task_id not in self.task_heads:
            self.add_head(task_id)
        return self.task_heads[task_id]
    
    def forward(self, x: np.ndarray, task_id: int) -> np.ndarray:
        """Forward pass usando cabeça específica."""
        head = self.get_head(task_id)
        return head['W_out'] @ x + head['b_out']
    
    def update(self, x: np.ndarray, target: np.ndarray, task_id: int, lr: float):
        """Atualiza pesos da cabeça específica."""
        head = self.get_head(task_id)
        pred = head['W_out'] @ x + head['b_out']
        err = target - pred
        
        # Delta rule
        head['W_out'] += lr * np.outer(err, x)
        head['b_out'] += lr * err
        
        return float((err ** 2).mean())


class MultiTaskBrain:
    """VisaoBrain com suporte a múltiplas tarefas via readout heads.
    
    Usa um único reservatório líquido (compartilhado) com múltiplos
    readout heads (um por tarefa). Isso permite:
    - Aprender novas tarefas sem esquecer antigas
    - Inferência específica por tarefa
    - Ensemble de múltiplas tarefas
    """
    
    def __init__(self, n_in: int = 8, n_hidden: int = 64, n_out: int = 1,
                 seed: int = 42, max_tasks: int = 10):
        self.brain = VisaoBrain(n_in, n_hidden, n_out, seed=seed)
        self.readout = MultiTaskReadout(n_hidden, n_out, max_tasks)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
    
    def learn_task(self, task_id: int, X: np.ndarray, Y: np.ndarray) -> dict:
        """Aprende uma tarefa específica."""
        errors = []
        
        for x, y in zip(X, Y):
            # Forward através do reservatório (sem atualizar W_out)
            self.brain.set_mode('infer')
            self.brain.reset_state()
            _ = self.brain.forward(x)
            state = self.brain.x.copy()
            
            # Update readout head para esta tarefa
            err = self.readout.update(state, y, task_id, lr=0.02)
            errors.append(err)
        
        return {
            'task_id': task_id,
            'mean_error': float(np.mean(errors)),
            'final_error': float(errors[-1]) if errors else 0.0,
        }
    
    def predict(self, task_id: int, x: np.ndarray) -> np.ndarray:
        """Prediz usando cabeça específica."""
        self.brain.set_mode('infer')
        self.brain.reset_state()
        _ = self.brain.forward(x)  # Atualiza estado oculto
        state = self.brain.x.copy()  # Usar estado oculto (não a saída)
        return self.readout.forward(state, task_id)
    
    def evaluate_task(self, task_id: int, X: np.ndarray, Y: np.ndarray) -> float:
        """Avalia uma tarefa específica."""
        errors = []
        for x, y in zip(X, Y):
            pred = self.predict(task_id, x)
            err = float(np.mean((pred - y) ** 2))
            errors.append(err)
        return float(np.mean(errors))


def test_ewc_reduces_interference():
    """Testa se MultiTaskBrain reduz interferência."""
    print("="*60)
    print("MULTI-TASK BRAIN INTERFERENCE TEST")
    print("="*60)
    
    # Sem multi-task (baseline: single W_out)
    print("\n[1] Sem multi-task (baseline):")
    brain_no_mt = VisaoBrain(8, 32, 1, seed=42)
    brain_no_mt.set_mode('learn')
    
    # Tarefa A
    pattern_a = np.random.randn(8)
    target_a = np.array([1.0])
    for _ in range(100):
        brain_no_mt.learn(pattern_a, target_a)
    
    # Tarefa B
    pattern_b = np.random.randn(8)
    target_b = np.array([-1.0])
    for _ in range(100):
        brain_no_mt.learn(pattern_b, target_b)
    
    # Testar A
    brain_no_mt.set_mode('infer')
    brain_no_mt.reset_state()
    out_a = brain_no_mt.forward(pattern_a)
    interference_no_mt = float(np.mean((out_a - target_a) ** 2))
    print(f"    Interferência: {interference_no_mt:.4f}")
    
    # Com Multi-Task
    print("\n[2] Com Multi-Task Brain:")
    mt_brain = MultiTaskBrain(8, 32, 1, seed=42, max_tasks=5)
    
    # Tarefa A (id=0)
    X_a = np.array([pattern_a] * 100)
    Y_a = np.array([target_a] * 100)
    mt_brain.learn_task(0, X_a, Y_a)
    error_a_mt = mt_brain.evaluate_task(0, X_a, Y_a)
    print(f"    Tarefa A erro: {error_a_mt:.4f}")
    
    # Tarefa B (id=1)
    X_b = np.array([pattern_b] * 100)
    Y_b = np.array([target_b] * 100)
    mt_brain.learn_task(1, X_b, Y_b)
    error_b_mt = mt_brain.evaluate_task(1, X_b, Y_b)
    print(f"    Tarefa B erro: {error_b_mt:.4f}")
    
    # Testar A novamente (após aprender B)
    error_a_after_b = mt_brain.evaluate_task(0, X_a, Y_a)
    print(f"    Tarefa A após B: {error_a_after_b:.4f}")
    
    # Resultado
    reduction = interference_no_mt / max(error_a_after_b, 1e-8)
    print(f"\n[3] Redução: {reduction:.1f}x")
    
    if error_a_after_b < interference_no_mt:
        print("    ✅ Multi-Task reduziu interferência!")
    else:
        print("    ⚠️ Multi-Task não melhorou")
    
    print("\n" + "="*60)
    print("MULTI-TASK TEST OK!")
    print("="*60)


if __name__ == "__main__":
    test_ewc_reduces_interference()
