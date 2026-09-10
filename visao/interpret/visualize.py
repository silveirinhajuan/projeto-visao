#!/usr/bin/env python3
"""visualize.py — Visualização do estado interno do VisaoBrain.

Gera gráficos e diagramas para entender o comportamento:
1. Estado do reservatório ao longo do tempo
2. Pesos de saída (W_out) — heatmap
3. Métricas de desempenho (erro, surpresa, lr)
4. Topologia da rede líquida
"""

import numpy as np
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


class StateVisualizer:
    """Visualiza estado interno do cérebro líquido."""
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
        self.history: list[dict] = []
    
    def record_step(self, step: int, error: float, surprise: float):
        """Registra estado atual."""
        self.history.append({
            'step': step,
            'error': error,
            'surprise': surprise,
            'lr': self.brain.learner.lr,
            'consolidation': self.brain.learner.consolidation,
            'state_mean': float(np.mean(self.brain.x)),
            'state_std': float(np.std(self.brain.x)),
            'w_out_norm': float(np.linalg.norm(self.brain.learner.W_out)),
        })
    
    def get_history_array(self, metric: str) -> np.ndarray:
        """Retorna histórico de uma métrica como array."""
        return np.array([h[metric] for h in self.history])
    
    def generate_ascii_plot(self, metric: str = 'error', width: int = 60, height: int = 15) -> str:
        """Gera plot ASCII de uma métrica."""
        values = self.get_history_array(metric)
        if len(values) == 0:
            return "No data"
        
        # Normalizar para altura do plot
        vmin, vmax = values.min(), values.max()
        if vmax - vmin < 1e-10:
            normalized = np.full_like(values, height // 2)
        else:
            normalized = ((values - vmin) / (vmax - vmin) * (height - 1)).astype(int)
        
        # Construir grid
        lines = []
        lines.append(f"  {metric.upper()} over time")
        lines.append(f"  {vmax:.4f} |")
        
        for row in range(height - 1, -1, -1):
            line = "  "
            for col in range(min(width, len(normalized))):
                idx = int(col * len(normalized) / width)
                if normalized[idx] == row:
                    line += "█"
                else:
                    line += " "
            lines.append(line)
        
        lines.append(f"  {vmin:.4f} |" + "─" * min(width, len(normalized)))
        lines.append(f"  Steps: 0 to {len(values)}")
        
        return "\n".join(lines)
    
    def generate_summary(self) -> str:
        """Gera resumo textual do estado."""
        if not self.history:
            return "No data recorded"
        
        latest = self.history[-1]
        
        lines = [
            "=" * 50,
            "VISÃO BRAIN STATE SUMMARY",
            "=" * 50,
            f"  Step: {latest['step']}",
            f"  Error: {latest['error']:.6f}",
            f"  Surprise: {latest['surprise']:.4f}",
            f"  Learning rate: {latest['lr']:.6f}",
            f"  Consolidation: {latest['consolidation']:.4f}",
            f"  State mean: {latest['state_mean']:.4f}",
            f"  State std: {latest['state_std']:.4f}",
            f"  W_out norm: {latest['w_out_norm']:.4f}",
            "=" * 50,
        ]
        
        return "\n".join(lines)


class TopologyVisualizer:
    """Visualiza topologia da rede líquida."""
    
    def __init__(self, brain: VisaoBrain):
        self.brain = brain
    
    def describe_topology(self) -> str:
        """Descreve a topologia da rede em texto."""
        lines = [
            "=" * 50,
            "VISÃO BRAIN TOPOLOGY",
            "=" * 50,
            f"  Input dim: {self.brain.n_in}",
            f"  Hidden dim: {self.brain.n_hidden}",
            f"  Output dim: {self.brain.n_out}",
            f"  Total params: {self.brain.n_in * self.brain.n_hidden + self.brain.n_hidden * self.brain.n_hidden + self.brain.n_hidden * self.brain.n_out + self.brain.n_hidden}",
            "",
            "  Weight matrices:",
            f"    W_in:  {self.brain.cell.W_in.shape}",
            f"    W_rec: {self.brain.cell.W_rec.shape}",
            f"    W_out: {self.brain.learner.W_out.shape}",
            f"    b:     {self.brain.cell.b.shape}",
            "",
            "  State:",
            f"    x (hidden state): {self.brain.x.shape}",
            f"    Mode: {self.brain._mode}",
            f"    Step: {self.brain.step}",
            "=" * 50,
        ]
        
        return "\n".join(lines)
    
    def sparsity_report(self) -> str:
        """Relatório de esparsidade dos pesos."""
        w_in = self.brain.cell.W_in
        w_rec = self.brain.cell.W_rec
        w_out = self.brain.learner.W_out
        
        def sparsity(m):
            return float(np.mean(np.abs(m) < 0.01))
        
        lines = [
            "=" * 50,
            "SPARSITY REPORT",
            "=" * 50,
            f"  W_in:  {sparsity(w_in)*100:.1f}% sparse",
            f"  W_rec: {sparsity(w_rec)*100:.1f}% sparse",
            f"  W_out: {sparsity(w_out)*100:.1f}% sparse",
            "=" * 50,
        ]
        
        return "\n".join(lines)


if __name__ == "__main__":
    print("="*60)
    print("VISUALIZATION DEMO")
    print("="*60)
    
    brain = VisaoBrain(8, 32, 1, seed=42)
    visualizer = StateVisualizer(brain)
    topo_vis = TopologyVisualizer(brain)
    
    # Simular aprendizado
    print("\n[1] Simulando 200 passos de aprendizado...")
    brain.set_mode('learn')
    
    for i in range(200):
        x = np.random.randn(8)
        y = np.array([np.sin(i * 0.05)])
        result = brain.learn(x, y)
        visualizer.record_step(i, result['err'], result['surprise'])
    
    # Mostrar topologia
    print("\n[2] Topologia:")
    print(topo_vis.describe_topology())
    
    # Mostrar esparsidade
    print("\n[3] Esparsidade:")
    print(topo_vis.sparsity_report())
    
    # Mostrar resumo
    print("\n[4] Resumo do estado:")
    print(visualizer.generate_summary())
    
    # Mostrar plot ASCII
    print("\n[5] Plot de erro:")
    print(visualizer.generate_ascii_plot('error', width=50, height=10))
    
    print("\n" + "="*60)
    print("VISUALIZATION OK!")
    print("="*60)
