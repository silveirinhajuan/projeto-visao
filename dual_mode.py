#!/usr/bin/env python3
"""dual_mode.py — Treinamento tradicional + uso contínuo.

O VisaoBrain pode operar em dois modos:
1. 'train' — treinamento supervisionado tradicional (backprop, batch)
2. 'continual' — aprendizado contínuo (Oja + consolidação + surpresa, online)

O modo 'train' é usado para aprendizado inicial rápido.
O modo 'continual' é usado para adaptação contínua sem esquecimento.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from visao.brain import VisaoBrain


class DualModeBrain(VisaoBrain):
    """VisaoBrain com modo treinamento tradicional + contínuo.
    
    Modos:
    - 'train': backpropagation tradicional (batch, epochs)
    - 'continual': aprendizado online (Oja + consolidação + surpresa)
    - 'infer': inferência (forward-only)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gradients = None
        self._batch_X = []
        self._batch_y = []
    
    def set_mode(self, mode):
        """Alterna entre 'train', 'continual', 'infer'."""
        assert mode in ('train', 'continual', 'infer'), f"Modo inválido: {mode}"
        self._mode = mode
        if mode == 'train':
            self._batch_X = []
            self._batch_y = []
    
    def train_step(self, X_batch, y_batch, lr=None):
        """Um passo de treinamento tradicional (backprop simplificado).
        
        Usa gradiente descendente batch sobre o readout.
        O reservatório é fixo (não backprop através do tempo).
        """
        if lr is None:
            lr = self.lr_base
        
        n = len(X_batch)
        errors = []
        
        for i in range(n):
            # Forward
            x = np.zeros(self.n_hidden)
            for t in range(len(X_batch[i])):
                x, _ = self.cell.step(x, X_batch[i][t])
            
            pred = self.learner.W_out @ x + self.learner.b_out
            err = y_batch[i] - pred
            errors.append(float(np.linalg.norm(err)))
            
            # Backprop no readout (linear)
            grad_W = np.outer(err, x)
            self.learner.W_out += lr * grad_W
            self.learner.b_out += lr * err
        
        return float(np.mean(errors))
    
    def train_epoch(self, X, y, epochs=10, lr=None, verbose=False):
        """Treina por múltiplos epochs (modo tradicional).
        
        Parameters
        ----------
        X : array (n_samples, seq_len, n_input)
        y : array (n_samples, n_output)
        epochs : int
        lr : float (opcional)
        """
        if lr is None:
            lr = self.lr_base
        
        history = []
        for epoch in range(epochs):
            errors = []
            for i in range(len(X)):
                err = self.train_step(X[i:i+1], y[i:i+1], lr)
                errors.append(err)
            
            mean_err = float(np.mean(errors))
            history.append(mean_err)
            
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch}: error = {mean_err:.4f}")
        
        return history
    
    def learn(self, x, y):
        """Um passo de aprendizado (modo contínuo ou infer)."""
        if self._mode == 'continual':
            return super().learn(x, y)
        elif self._mode == 'infer':
            return self.forward(x)
        else:
            raise ValueError("Use train_epoch() para modo 'train'")
    
    def forward(self, x):
        """Inferência (forward-only)."""
        self._mode = 'infer'
        return super().forward(x)


def demo_dual_mode():
    """Demonstra os dois modos de treinamento."""
    print("="*60)
    print("DUAL MODE: Treinamento Tradicional + Contínuo")
    print("="*60)
    
    # Dados sintéticos
    rng = np.random.default_rng(42)
    n = 500
    t = np.arange(n)
    u = rng.normal(0, 1, (n, 2)) * 0.3
    u[:, 1] += np.sin(t * 0.1)
    y = np.convolve(u[:, 0], np.ones(30)/30.0, mode='same')[:, None]
    
    # Split: 70% treino, 30% teste
    split = int(0.7 * n)
    X_tr, y_tr = u[:split], y[:split]
    X_te, y_te = u[split:], y[split:]
    
    # 1. Modo tradicional (backprop batch)
    print("\n[1] Modo TRADICIONAL (backprop)...")
    brain = DualModeBrain(2, 64, 1, seed=42)
    brain.set_mode('train')
    
    # Preparar batches (sequências de 1 passo)
    history = brain.train_epoch(
        X_tr.reshape(-1, 1, 2), 
        y_tr, 
        epochs=5, 
        lr=0.01, 
        verbose=True
    )
    print(f"  Final training error: {history[-1]:.4f}")
    
    # Avaliar
    brain.set_mode('infer')
    errors_trad = []
    for i in range(len(X_te)):
        pred = brain.forward(X_te[i])
        errors_trad.append(float(np.linalg.norm(y_te[i] - pred)))
    print(f"  Test error: {np.mean(errors_trad):.4f}")
    
    # 2. Modo contínuo (Oja + consolidação + surpresa)
    print("\n[2] Modo CONTÍNUO (Oja + consolidação + surpresa)...")
    brain_cont = DualModeBrain(2, 64, 1, seed=42)
    brain_cont.set_mode('continual')
    
    errors_cont = []
    for i in range(len(X_tr)):
        brain_cont.learn(X_tr[i], y_tr[i])
    
    # Avaliar
    brain_cont.set_mode('infer')
    for i in range(len(X_te)):
        pred = brain_cont.forward(X_te[i])
        errors_cont.append(float(np.linalg.norm(y_te[i] - pred)))
    print(f"  Test error: {np.mean(errors_cont):.4f}")
    
    # 3. Comparação
    print("\n" + "="*60)
    print("COMPARAÇÃO")
    print("="*60)
    print(f"Modo tradicional:  {np.mean(errors_trad):.4f}")
    print(f"Modo contínuo:     {np.mean(errors_cont):.4f}")
    print(f"Modo tradicional é melhor para batch learning")
    print(f"Modo contínuo é melhor para adaptação online (sem replay)")


if __name__ == "__main__":
    demo_dual_mode()
