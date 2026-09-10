"""
task_free.py — Tarefa 8.5: Aprendizado contínuo sem fronteira de tarefa.

O cérebro detecta mudança de distribuição automaticamente (via surpresa)
e ajusta plasticidade. Sem saber "acabou tarefa A, começou tarefa B".

Mecanismo:
  - Surpresa persistentemente alta = mudança de regime
  - Ao detectar: aumentar lr temporariamente, reduzir omega (liberar plasticidade)
  - Após estabilizar: voltar ao normal
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "visao" / "analysis" / "results_task_free.json"


def make_regime_stream(n: int = 4000, seed: int = 0):
    """Gera stream com mudanças de regime não-anunciadas."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    u = np.zeros((n, 2))
    y = np.zeros((n, 1))

    regimes = [
        ("sine_slow", 0, 1000, 0.02),
        ("sine_fast", 1000, 2000, 0.08),
        ("saw", 2000, 3000, 0.0),
        ("mixed", 3000, 4000, 0.0),
    ]

    for kind, start, end, freq in regimes:
        length = end - start
        tt = np.arange(length)
        if kind == "sine_slow":
            signal = np.sin(2 * np.pi * freq * tt)
        elif kind == "sine_fast":
            signal = np.sin(2 * np.pi * freq * tt)
        elif kind == "saw":
            signal = 2.0 * (tt % 30) / 30.0 - 1.0
        else:
            signal = np.sin(2 * np.pi * 0.03 * tt) + 0.5 * np.cos(2 * np.pi * 0.06 * tt)

        u[start:end, 0] += signal
        u[start:end, 1] += 0.3 * np.roll(signal, 1)
        y[start:end, 0] = np.convolve(signal, np.ones(5) / 5, mode="same")[:length]

    u += rng.normal(0, 0.03, u.shape)
    return u.astype(np.float64), y.astype(np.float64), regimes


class TaskFreeDetector:
    """Detecta mudança de regime usando surpresa acumulada."""

    def __init__(self, window: int = 50, threshold: float = 1.5):
        self.window = window
        self.threshold = threshold
        self.surprise_history = []
        self.regime_change_points = []

    def update(self, surprise: float, step: int) -> bool:
        """Retorna True se detectou mudança de regime."""
        self.surprise_history.append(surprise)

        if len(self.surprise_history) < self.window * 2:
            return False

        # Comparar média recente vs histórico
        recent = np.mean(self.surprise_history[-self.window:])
        baseline = np.mean(self.surprise_history[-2*self.window:-self.window])

        if baseline > 0 and recent / baseline > self.threshold:
            self.regime_change_points.append(step)
            return True
        return False


def run_task_free(brain: VisaoBrain, u: np.ndarray, y: np.ndarray) -> dict:
    """Roda o cérebro com detecção de regime."""
    detector = TaskFreeDetector(window=50, threshold=1.5)
    errors = []
    surprises = []
    change_points = []
    lr_boosts = []

    for i in range(len(u)):
        result = brain.learn(u[i], y[i])
        errors.append(result["err"])
        surprises.append(result["surprise"])

        # Detectar mudança de regime
        if detector.update(result["surprise"], i):
            change_points.append(i)
            # Boost temporário de lr para adaptação rápida
            brain.lr = min(brain.lr * 2.0, brain._meta_max)
            brain.learner.lr = brain.lr

        # Decair lr boost gradualmente
        if change_points and i - change_points[-1] > 100:
            brain.lr = max(brain.lr * 0.95, brain.lr_base)
            brain.learner.lr = brain.lr

    # Métricas
    errors = np.array(errors)
    regime_mses = {}
    for name, start, end in [("R1_slow", 0, 1000), ("R2_fast", 1000, 2000), ("R3_saw", 2000, 3000), ("R4_mixed", 3000, 4000)]:
        regime_mses[name] = float(np.mean(errors[start:end] ** 2))

    return {
        "mse_total": float(np.mean(errors ** 2)),
        "mse_final_500": float(np.mean(errors[-500:] ** 2)),
        "regime_mses": regime_mses,
        "n_change_points": len(change_points),
        "mean_surprise": float(np.mean(surprises)),
    }


def run_experiment(seeds: tuple[int, ...] = (1, 2, 3, 4, 5)) -> dict:
    """Roda experimento completo."""
    results = {"with_detection": [], "without_detection": []}

    for seed in seeds:
        u, y, _ = make_regime_stream(n=4000, seed=seed)

        # Com detecção
        brain1 = VisaoBrain(n_in=2, n_hidden=64, n_out=1, consolidation=8.0, meta_learn=True, seed=seed)
        r1 = run_task_free(brain1, u, y)
        results["with_detection"].append(r1)

        # Sem detecção (meta-learn padrão, 1 pass)
        brain2 = VisaoBrain(n_in=2, n_hidden=64, n_out=1, consolidation=8.0, meta_learn=True, seed=seed)
        errs2 = []
        for ui, yi in zip(u, y):
            r = brain2.learn(ui, yi)
            errs2.append(r["err"])
        results["without_detection"].append({"mse_total": float(np.mean(np.array(errs2) ** 2))})

    # Agregar
    agg = {}
    for name, runs in results.items():
        agg[name] = {
            "mse_total": float(np.mean([r["mse_total"] for r in runs])),
            "mse_total_std": float(np.std([r["mse_total"] for r in runs])),
        }
        if "n_change_points" in runs[0]:
            agg[name]["avg_change_points"] = float(np.mean([r["n_change_points"] for r in runs]))

    return agg


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 8.5 — Task-Free Learning")
    print("Stream: 4 regimes não-anunciados")
    print("=" * 70)

    results = run_experiment()

    print("\nResultados (5 seeds):")
    for name, r in results.items():
        print(f"  {name}: MSE = {r['mse_total']:.4f} ± {r.get('mse_total_std', 0):.4f}")
        if "avg_change_points" in r:
            print(f"    mudanças detectadas: {r['avg_change_points']:.1f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] -> {OUT}")
