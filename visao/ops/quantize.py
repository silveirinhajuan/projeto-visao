#!/usr/bin/env python3
"""quantize.py — Tarefa 11.1: Quantização de pesos para int8 (per-channel)."""

import numpy as np
from pathlib import Path
import sys

# Garantir que o diretório raiz do projeto seja importável
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visao.brain import VisaoBrain


class QuantizedTensor:
    """Tensor quantizado int8 com metadados para dequantização.
    
    Usa quantização per-channel (por neurônio) para melhor precisão.
    """
    
    def __init__(self, w: np.ndarray, axis: int = 0):
        """
        Parameters
        ----------
        w : array
            Pesos a quantizar.
        axis : int
            Eixo para quantização per-channel (0 = por linha).
        """
        self.shape = w.shape
        self.axis = axis
        
        if w.ndim == 1:
            # Vetor: quantização simples
            self.wmin = float(w.min())
            self.wmax = float(w.max())
            scale = (self.wmax - self.wmin) / 255.0
            if scale == 0:
                scale = 1.0
            self.scale = np.array([scale])
            self.q = np.clip(np.round((w - self.wmin) / scale) - 128, -128, 127).astype(np.int8)
        else:
            # Matriz: quantização per-channel
            n_channels = w.shape[axis]
            self.wmin = np.zeros(n_channels)
            self.wmax = np.zeros(n_channels)
            self.scale = np.zeros(n_channels)
            self.q = np.zeros_like(w, dtype=np.int8)
            
            for i in range(n_channels):
                if axis == 0:
                    w_channel = w[i, :]
                else:
                    w_channel = w[:, i]
                
                self.wmin[i] = float(w_channel.min())
                self.wmax[i] = float(w_channel.max())
                scale = (self.wmax[i] - self.wmin[i]) / 255.0
                if scale == 0:
                    scale = 1.0
                self.scale[i] = scale
                
                if axis == 0:
                    self.q[i, :] = np.clip(np.round((w_channel - self.wmin[i]) / scale) - 128, -128, 127)
                else:
                    self.q[:, i] = np.clip(np.round((w_channel - self.wmin[i]) / scale) - 128, -128, 127)
    
    def dequantize(self) -> np.ndarray:
        if self.scale.size == 1:
            return (self.q.astype(np.float64) + 128) * self.scale[0] + self.wmin
        else:
            w = np.zeros_like(self.q, dtype=np.float64)
            for i in range(self.scale.size):
                if self.axis == 0:
                    w[i, :] = (self.q[i, :].astype(np.float64) + 128) * self.scale[i] + self.wmin[i]
                else:
                    w[:, i] = (self.q[:, i].astype(np.float64) + 128) * self.scale[i] + self.wmin[i]
            return w
    
    @property
    def nbytes(self):
        return self.q.nbytes + self.scale.nbytes + self.wmin.nbytes + self.wmax.nbytes


class QuantizedBrain:
    """Versão quantizada do VisaoBrain (int8 per-channel)."""
    
    def __init__(self, brain: VisaoBrain):
        self.n_in = brain.n_in
        self.n_hidden = brain.n_hidden
        self.n_out = brain.n_out
        
        # Quantizar pesos grandes (per-channel)
        self.W_in_q = QuantizedTensor(brain.cell.W_in, axis=0)
        self.W_rec_q = QuantizedTensor(brain.cell.W_rec, axis=0)
        self.W_out_q = QuantizedTensor(brain.learner.W_out, axis=0)
        
        # Manter precisão total para vieses e parâmetros pequenos
        self.b = brain.cell.b.copy()
        self.A = brain.cell.A.copy()
        self.tau = brain.cell.tau.copy()
        self.b_out = brain.learner.b_out.copy()
        
        # Estado
        self.x = np.zeros(self.n_hidden)
        self._mode = 'infer'
    
    def forward(self, u):
        """Inferência com pesos quantizados."""
        W_in = self.W_in_q.dequantize()
        W_rec = self.W_rec_q.dequantize()
        W_out = self.W_out_q.dequantize()
        
        f = 1.0 / (1.0 + np.exp(-(W_in @ u + W_rec @ self.x + self.b)))
        tau_eff = 0.1 / self.tau
        self.x = self.x + (-self.x + f * self.A) * tau_eff
        
        return W_out @ self.x + self.b_out
    
    def reset_state(self):
        self.x = np.zeros(self.n_hidden)
    
    @property
    def memory_bytes(self):
        return (self.W_in_q.nbytes + self.W_rec_q.nbytes + self.W_out_q.nbytes +
                self.b.nbytes + self.A.nbytes + self.tau.nbytes + self.b_out.nbytes)


def measure_memory_savings(brain: VisaoBrain, q_brain: QuantizedBrain) -> dict:
    """Mede economia de memória."""
    original = (brain.cell.W_in.nbytes + brain.cell.W_rec.nbytes + 
                brain.learner.W_out.nbytes + brain.cell.b.nbytes + 
                brain.cell.A.nbytes + brain.cell.tau.nbytes + brain.learner.b_out.nbytes)
    quantized = q_brain.memory_bytes
    return {
        'original_bytes': original,
        'quantized_bytes': quantized,
        'savings_bytes': original - quantized,
        'compression_ratio': original / max(quantized, 1),
        'space_saving_pct': (1 - quantized / max(original, 1)) * 100,
    }


def benchmark_quantization(brain: VisaoBrain, X: np.ndarray, Y: np.ndarray) -> dict:
    """Benchmark: original vs quantizado."""
    q_brain = QuantizedBrain(brain)
    
    brain.set_mode('infer')
    q_brain.reset_state()
    brain.reset_state()
    
    # Original
    preds_orig = [brain.forward(X[i]) for i in range(len(X))]
    mse_orig = float(np.mean((np.array(preds_orig) - Y) ** 2))
    
    # Quantizado
    preds_q = [q_brain.forward(X[i]) for i in range(len(X))]
    mse_q = float(np.mean((np.array(preds_q) - Y) ** 2))
    
    return {
        'mse_original': mse_orig,
        'mse_quantized': mse_q,
        'mse_increase_pct': (mse_q - mse_orig) / max(mse_orig, 1e-10) * 100,
        'memory': measure_memory_savings(brain, q_brain),
    }


if __name__ == "__main__":
    print("="*60)
    print("QUANTIZATION DEMO (per-channel)")
    print("="*60)
    
    rng = np.random.default_rng(42)
    brain = VisaoBrain(2, 64, 1, seed=42)
    
    # Dados
    n = 500
    u = rng.normal(0, 1, (n, 2)) * 0.3
    y = np.convolve(u[:, 0], np.ones(30)/30.0, mode='same')[:, None]
    
    # Treinar
    brain.set_mode('learn')
    for i in range(n):
        brain.learn(u[i], y[i])
    
    # Benchmark
    brain.set_mode('infer')
    results = benchmark_quantization(brain, u, y)
    
    print(f"MSE original:   {results['mse_original']:.4f}")
    print(f"MSE quantized:  {results['mse_quantized']:.4f}")
    print(f"MSE increase:   {results['mse_increase_pct']:.1f}%")
    print(f"Memory saving:  {results['memory']['space_saving_pct']:.1f}%")
    print(f"Compression:    {results['memory']['compression_ratio']:.1f}x")
