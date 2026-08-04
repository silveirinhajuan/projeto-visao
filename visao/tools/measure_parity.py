"""Mede a paridade numérica JAX vs numpy e imprime os erros reais."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from prototype.liquid import LiquidCell  # noqa: E402
from visao.core.cfc import CfCCell  # noqa: E402

SEED, N_IN, N_HIDDEN, DT = 123, 2, 48, 0.15

npc = LiquidCell(N_IN, N_HIDDEN, sparsity=0.5, dt=DT, rng=np.random.default_rng(SEED))
jxc = CfCCell(N_IN, N_HIDDEN, sparsity=0.5, dt=DT, rng=np.random.default_rng(SEED))

rng = np.random.default_rng(999)
t = np.arange(600)
u = rng.normal(0, 0.8, (600, N_IN))
u[:, 0] += np.sin(t * 0.07)

s_np, a_np = npc.rollout(u)
s_jx, a_jx = jxc.rollout(u)

err_s = float(np.abs(s_np - np.asarray(s_jx)).max())
err_a = float(np.abs(a_np - np.asarray(a_jx)).max())

calma = np.zeros((300, N_IN))
calma[:, 1] = 0.05 * np.sin(np.arange(300) * 0.02)
agitada = np.random.default_rng(7).normal(0, 1.5, (300, N_IN))


def faixa(cell, seq):
    _, acts = cell.rollout(seq)
    tau = np.asarray(cell.tau_effective(acts[100:]))
    return float(tau.max() - tau.min())


r_np = faixa(npc, agitada) / faixa(npc, calma)
r_jx = faixa(jxc, agitada) / faixa(jxc, calma)

print(f"PARIDADE JAX vs numpy (600 passos, float64, seed {SEED}):")
print(f"  erro maximo estados   : {err_s:.3e}   (criterio: < 1e-5)")
print(f"  erro maximo ativacoes : {err_a:.3e}   (criterio: < 1e-5)")
print(f"\nLIQUIDEZ (razao faixa de tau agitada/calma):")
print(f"  numpy : {r_np:.3f}x")
print(f"  jax   : {r_jx:.3f}x")
print(f"\nDevices JAX: {jax.devices()}")
