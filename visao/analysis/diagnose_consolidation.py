"""
diagnose_consolidation.py — Diagnóstico: quando EWC ajuda vs atrapalha?

Achado do benchmark 8.2:
  - naive (sem EWC): MSE=0.19
  - visao_full (com EWC + meta): MSE=0.40
  
Hipótese: EWC é útil quando há tarefas DISTINTAS com fronteiras claras.
Em streaming puro (task-free), a consolidação excessiva impede adaptação.

Teste: variar força de consolidação e medir erro em cenários distintos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "visao" / "analysis" / "results_consolidation_diagnosis.json"


def make_task(kind: str, n: int, seed: int, noise: float = 0.05):
    """Gera tarefa sintética."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if kind == "sine":
        signal = np.sin(2 * np.pi * 0.05 * t)
    elif kind == "saw":
        signal = 2.0 * (t % 50) / 50.0 - 1.0
    elif kind == "mixed":
        signal = np.sin(2 * np.pi * 0.03 * t) + 0.5 * np.cos(2 * np.pi * 0.07 * t)
    else:
        signal = rng.normal(0, 0.5, n)
    u = rng.normal(0, noise, (n, 2))
    u[:, 0] += signal
    y = np.convolve(signal, np.ones(5) / 5, mode="same")[:, None]
    return u.astype(np.float64), y.astype(np.float64)


def run_task_boundary(n_steps=2000, consolidation=8.0, seed=42):
    """Cenário 1: tarefas com fronteira clara (A -> B)."""
    brain = VisaoBrain(n_in=2, n_hidden=64, n_out=1, consolidation=consolidation, seed=seed)
    
    # Tarefa A
    u_a, y_a = make_task("sine", n_steps, seed)
    for ui, yi in zip(u_a, y_a):
        brain.learn(ui, yi)
    
    # Avaliar A
    brain.set_mode("infer")
    eval_a1 = brain.evaluate_stream(u_a[-100:], y_a[-100:])
    
    # Tarefa B
    brain.set_mode("learn")
    u_b, y_b = make_task("saw", n_steps, seed + 1)
    for ui, yi in zip(u_b, y_b):
        brain.learn(ui, yi)
    
    # Avaliar A novamente (esquecimento?)
    brain.set_mode("infer")
    eval_a2 = brain.evaluate_stream(u_a[-100:], y_a[-100:])
    
    # Avaliar B
    eval_b = brain.evaluate_stream(u_b[-100:], y_b[-100:])
    
    forgetting = eval_a2["mse"] - eval_a1["mse"]
    return {
        "forgetting": forgetting,
        "mse_A_before": eval_a1["mse"],
        "mse_A_after": eval_a2["mse"],
        "mse_B": eval_b["mse"],
    }


def run_streaming(n_steps=2000, consolidation=8.0, seed=42):
    """Cenário 2: streaming puro (sem fronteira)."""
    brain = VisaoBrain(n_in=2, n_hidden=64, n_out=1, consolidation=consolidation, seed=seed)
    
    # Stream alternado
    u1, y1 = make_task("sine", n_steps // 2, seed)
    u2, y2 = make_task("saw", n_steps // 2, seed + 1)
    u = np.vstack([u1, u2])
    y = np.vstack([y1, y2])
    
    errs = []
    for ui, yi in zip(u, y):
        result = brain.learn(ui, yi)
        errs.append(result["err"])
    
    # Métricas
    errs = np.array(errs)
    return {
        "mse_total": float(np.mean(errs ** 2)),
        "mse_first_half": float(np.mean(errs[:len(errs) // 2] ** 2)),
        "mse_second_half": float(np.mean(errs[len(errs) // 2:] ** 2)),
    }


def run_diagnosis(consolidation_values=None, seeds=(1, 2, 3, 4, 5)):
    if consolidation_values is None:
        consolidation_values = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    
    results = {}
    for c in consolidation_values:
        print(f"\n--- consolidation={c} ---")
        
        # Cenário 1: tarefa com fronteira
        tb_results = []
        for seed in seeds:
            tb = run_task_boundary(consolidation=c, seed=seed)
            tb_results.append(tb)
        
        # Cenário 2: streaming
        st_results = []
        for seed in seeds:
            st = run_streaming(consolidation=c, seed=seed)
            st_results.append(st)
        
        results[f"c={c}"] = {
            "task_boundary": {
                "forgetting": float(np.mean([r["forgetting"] for r in tb_results])),
                "mse_A": float(np.mean([r["mse_A_after"] for r in tb_results])),
                "mse_B": float(np.mean([r["mse_B"] for r in tb_results])),
            },
            "streaming": {
                "mse_total": float(np.mean([r["mse_total"] for r in st_results])),
                "mse_first": float(np.mean([r["mse_first_half"] for r in st_results])),
                "mse_second": float(np.mean([r["mse_second_half"] for r in st_results])),
            }
        }
        
        tb = results[f"c={c}"]["task_boundary"]
        st = results[f"c={c}"]["streaming"]
        print(f"  Task boundary: forgetting={tb['forgetting']:+.4f}, mse_B={tb['mse_B']:.4f}")
        print(f"  Streaming: total={st['mse_total']:.4f}, second={st['mse_second']:.4f}")
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("DIAGNÓSTICO: quando EWC ajuda vs atrapalha?")
    print("=" * 70)
    
    results = run_diagnosis()
    
    print("\n" + "=" * 70)
    print("RESUMO")
    print(f"{'Consolidation':>15} {'Forgetting':>12} {'Stream MSE':>12} {'Veredicto':>20}")
    print("-" * 70)
    for name, r in results.items():
        f = r["task_boundary"]["forgetting"]
        s = r["streaming"]["mse_total"]
        if f < -0.01 and s < 0.3:
            verdict = "✓ Bom (ajuda nos dois)"
        elif f < -0.01:
            verdict = " Ajuda em tarefa"
        elif s < 0.3:
            verdict = " Ajuda em stream"
        else:
            verdict = "✗ Fraco"
        print(f"{name:>15} {f:>12.4f} {s:>12.4f} {verdict:>20}")
    
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] -> {OUT}")
