"""
profile_memory.py — Tarefa 8.4: Perfil de memória e hardware fraco.

Mede:
  - Pico de RAM para n_hidden = 64, 128, 256, 512
  - Tempo por passo (learn vs infer)
  - Valida que roda em < 2GB RAM e < 10ms/passo no notebook do Juan
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


def get_ram_usage_mb() -> float:
    """Lê /proc/self/status para obter VmRSS em MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024  # kB -> MB
    except:
        pass
    return 0.0


def profile_brain(n_hidden: int, n_steps: int = 500) -> dict:
    """Perfila um brain com n_hidden neurônios."""
    ram_before = get_ram_usage_mb()

    t0 = time.perf_counter()
    brain = VisaoBrain(n_in=2, n_hidden=n_hidden, n_out=1, seed=42)
    init_time = time.perf_counter() - t0

    ram_after_init = get_ram_usage_mb()
    ram_used = ram_after_init - ram_before

    # Modo learn
    rng = np.random.default_rng(42)
    u = rng.normal(0, 0.1, (n_steps, 2))
    y = rng.normal(0, 0.1, (n_steps, 1))

    t0 = time.perf_counter()
    for ui, yi in zip(u, y):
        brain.learn(ui, yi)
    learn_time = (time.perf_counter() - t0) / n_steps * 1000  # ms

    # Modo infer
    brain.set_mode("infer")
    brain.reset_state()

    t0 = time.perf_counter()
    for ui in u:
        brain.forward(ui)
    infer_time = (time.perf_counter() - t0) / n_steps * 1000  # ms

    # Contagem de parâmetros
    n_params = (
        brain.cell.W_in.size
        + brain.cell.W_rec.size
        + brain.cell.b.size
        + brain.cell.A.size
        + brain.learner.W_out.size
        + brain.learner.b_out.size
    )

    return {
        "n_hidden": n_hidden,
        "n_params": n_params,
        "ram_mb": round(ram_used, 2),
        "init_ms": round(init_time * 1000, 2),
        "learn_ms_per_step": round(learn_time, 3),
        "infer_ms_per_step": round(infer_time, 3),
        "ram_ok": ram_used < 2048,  # < 2GB
        "speed_ok": learn_time < 10,  # < 10ms
    }


def run_profile():
    """Roda perfil para vários tamanhos."""
    print("=" * 70)
    print("TAREFA 8.4 — Perfil de Memória e Velocidade")
    print("=" * 70)

    sizes = [64, 128, 256, 512]
    results = []

    for n in sizes:
        print(f"\nProfile n_hidden={n}...")
        r = profile_brain(n)
        results.append(r)
        print(f"  params: {r['n_params']:,}")
        print(f"  RAM: {r['ram_mb']:.1f} MB {'✓' if r['ram_ok'] else '✗ (>2GB)'}")
        print(f"  learn: {r['learn_ms_per_step']:.3f} ms/passo {'✓' if r['speed_ok'] else '✗ (>10ms)'}")
        print(f"  infer: {r['infer_ms_per_step']:.3f} ms/passo")

    print("\n" + "=" * 70)
    print("RESUMO")
    print(f"{'n_hidden':>10} {'params':>10} {'RAM(MB)':>10} {'learn(ms)':>12} {'infer(ms)':>12} {'OK?':>6}")
    print("-" * 70)
    for r in results:
        ok = "✓" if (r["ram_ok"] and r["speed_ok"]) else "✗"
        print(f"{r['n_hidden']:>10} {r['n_params']:>10,} {r['ram_mb']:>10.1f} "
              f"{r['learn_ms_per_step']:>12.3f} {r['infer_ms_per_step']:>12.3f} {ok:>6}")

    return results


if __name__ == "__main__":
    run_profile()
