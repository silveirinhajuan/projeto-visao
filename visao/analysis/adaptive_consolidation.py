"""
adaptive_consolidation.py — Tarefa 8.6 (NOVA): Consolidação adaptativa.

Achado do diagnóstico:
  - EWC ajuda em tarefa com fronteira (forgetting ↓ 27%)
  - EWC atrapalha em streaming puro (↑ 14% MSE)

Solução: adaptar força de consolidação baseado em detecção de fronteira.
  - Surpresa persistentemente alta = fronteira de tarefa = aumentar c
  - Surpresa baixa = streaming estável = diminuir c (permitir adaptação)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "visao" / "analysis" / "results_adaptive_consolidation.json"


class AdaptiveConsolidationBrain(VisaoBrain):
    """VisaoBrain com consolidação adaptativa.

    Ajusta a força de consolidação (c) baseado na surpresa acumulada:
    - Surpresa alta por muitos passos = fronteira de tarefa = c alto
    - Surpresa baixa = streaming estável = c baixo
    """

    def __init__(self, *args, c_min: float = 0.0, c_max: float = 16.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_min = c_min
        self.c_max = c_max
        self._surprise_window = []
        self._window_size = 100

    def _local_update(self, x, x_prev, target):
        """Override para adicionar consolidação adaptativa."""
        # Calcular surpresa antes do update
        pred = self.learner.predict(x)
        err = target - pred
        err_mag = float(np.abs(err).mean())
        s = self.learner.surprise(err_mag)

        # Atualizar janela de surpresa
        self._surprise_window.append(s)
        if len(self._surprise_window) > self._window_size:
            self._surprise_window.pop(0)

        # Adaptar consolidação
        if len(self._surprise_window) >= self._window_size:
            mean_surprise = np.mean(self._surprise_window)
            # Surpresa alta -> c alto; surpresa baixa -> c baixo
            # Mapear [1.0, 2.0] -> [c_min, c_max]
            normalized = (mean_surprise - 1.0) / 1.0
            normalized = np.clip(normalized, 0.0, 1.0)
            self.learner.consolidation = self.c_min + (self.c_max - self.c_min) * normalized

        # Chamar update original
        return super()._local_update(x, x_prev, target)


def make_task(kind: str, n: int, seed: int, noise: float = 0.05):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if kind == "sine":
        signal = np.sin(2 * np.pi * 0.05 * t)
    elif kind == "saw":
        signal = 2.0 * (t % 50) / 50.0 - 1.0
    else:
        signal = np.sin(2 * np.pi * 0.03 * t) + 0.5 * np.cos(2 * np.pi * 0.07 * t)
    u = rng.normal(0, noise, (n, 2))
    u[:, 0] += signal
    y = np.convolve(signal, np.ones(5) / 5, mode="same")[:, None]
    return u.astype(np.float64), y.astype(np.float64)


def run_adaptive_vs_fixed(n_steps=2000, seed=42):
    """Compara adaptive vs fixed consolidation em cenário misto."""
    results = {}
    
    for name, brain in [
        ("adaptive", AdaptiveConsolidationBrain(n_in=2, n_hidden=64, n_out=1, seed=seed)),
        ("fixed_c0", VisaoBrain(n_in=2, n_hidden=64, n_out=1, consolidation=0.0, seed=seed)),
        ("fixed_c8", VisaoBrain(n_in=2, n_hidden=64, n_out=1, consolidation=8.0, seed=seed)),
        ("fixed_c32", VisaoBrain(n_in=2, n_hidden=64, n_out=1, consolidation=32.0, seed=seed)),
    ]:
        # Fase 1: tarefa A
        u_a, y_a = make_task("sine", n_steps, seed)
        for ui, yi in zip(u_a, y_a):
            brain.learn(ui, yi)
        
        brain.set_mode("infer")
        eval_a1 = brain.evaluate_stream(u_a[-100:], y_a[-100:])
        
        # Fase 2: tarefa B (fronteira clara)
        brain.set_mode("learn")
        u_b, y_b = make_task("saw", n_steps, seed + 1)
        for ui, yi in zip(u_b, y_b):
            brain.learn(ui, yi)
        
        brain.set_mode("infer")
        eval_a2 = brain.evaluate_stream(u_a[-100:], y_a[-100:])
        eval_b = brain.evaluate_stream(u_b[-100:], y_b[-100:])
        
        results[name] = {
            "forgetting": eval_a2["mse"] - eval_a1["mse"],
            "mse_A_after": eval_a2["mse"],
            "mse_B": eval_b["mse"],
        }
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 8.6 — Consolidação Adaptativa")
    print("=" * 70)
    
    all_results = []
    for seed in (1, 2, 3, 4, 5):
        r = run_adaptive_vs_fixed(seed=seed)
        all_results.append(r)
    
    print("\nResultados (5 seeds):")
    print(f"{'Config':>15} {'Forgetting':>12} {'MSE A (after)':>15} {'MSE B':>10}")
    print("-" * 55)
    
    for name in ["adaptive", "fixed_c0", "fixed_c8", "fixed_c32"]:
        f = np.mean([r[name]["forgetting"] for r in all_results])
        a = np.mean([r[name]["mse_A_after"] for r in all_results])
        b = np.mean([r[name]["mse_B"] for r in all_results])
        print(f"{name:>15} {f:>12.4f} {a:>15.4f} {b:>10.4f}")
    
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(all_results, indent=2))
    print(f"\n[ok] -> {OUT}")
