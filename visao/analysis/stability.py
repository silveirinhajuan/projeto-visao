"""
stability.py — Diagnóstico de estabilidade do reservatório líquido (Fase 1, tarefa 1.8).

Mede duas coisas que decide se a arquitetura é utilizável como memória contínua:

1. Raio espectral efetivo
   - `spectral_radius_Wrec`: eigenvalue máx. em módulo de W_rec (a matriz recorrente
     congelada). Receita clássica de Echo State Networks: SR < 1 ~ contração.
   - `max_jacobian_spectral_radius`: SR máx. do jacobiano *local* da dinâmica ao
     longo de uma trajetória dirigida (inclui a não-linearidade sigmoid e o vazamento
     tau). É o número que realmente governa a expansão local — o "raio espectral
     efetivo" de fato.

2. Maior expoente de Lyapunov empírico (sob perturbação)
   - `lyapunov_exponent`: lineariza o mapa de passo a cada instante, propaga um
     vetor perturbação δ, renormaliza e acumula log||δ||. Média / dt → taxa
     contínua. LE < 0 ⇒ contração (sub-caótico, memória estável); LE > 0 ⇒ caótico
     (perturbações pequenas explodem — inútil como memória).

Regra do projeto: o reservatório NÃO pode ser caótico (LE < 0) NEM morto
(estados não colapsam a zero sob entrada não-trivial). Esta é a "borda do caos"
medida, não decorada.

Tudo roda em numpy puro sobre os parâmetros da CfCCell; o avanço do estado real
usa cell.step (jax, já compilado). 96 neurônios, poucos segundos.
"""

from __future__ import annotations

import numpy as np

from visao.core.cfc import CfCCell


def spectral_radius_Wrec(cell: CfCCell) -> float:
    """Raio espectral da matriz recorrente congelada W_rec (regra ESN clássica)."""
    W = np.asarray(cell.W_rec)
    eigs = np.linalg.eigvals(W)
    return float(np.max(np.abs(eigs)))


def jacobian_step(cell: CfCCell, x, u) -> np.ndarray:
    """Jacobiano do mapa de um passo y = step(x, u) em relação a x (entrada u fixa).

    y = num/den,  num = x + dt*f*A,  den = 1 + dt*(1/tau + f),  f = sigmoid(W_rec x + c).
    J_ij = (den_i * δ_ij - num_i * dt * f_i*(1-f_i) * W_rec_ij) / den_i^2.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    W_rec = np.asarray(cell.W_rec)
    W_in = np.asarray(cell.W_in)
    b = np.asarray(cell.b)
    A = np.asarray(cell.A)
    tau = np.asarray(cell.tau)
    dt = float(cell.dt)

    c = W_in @ u + b
    g = W_rec @ x + c
    f = 1.0 / (1.0 + np.exp(-g))           # sigmoid
    den = 1.0 + dt * (1.0 / tau + f)
    num = x + dt * f * A
    ff = f * (1.0 - f)

    n = x.size
    I = np.eye(n)
    # outer[i,j] = num_i * dt * ff_i * W_rec_ij
    outer = np.outer(num, dt * ff) * W_rec
    J = (I * den[None, :] - outer) / (den[None, :] ** 2)
    return J


def max_jacobian_spectral_radius(cell: CfCCell, seq, x0=None, n_steps=None) -> float:
    """Maior raio espectral do jacobiano local ao longo da trajetória dirigida."""
    seq = np.asarray(seq, dtype=float)
    if n_steps is None:
        n_steps = len(seq)
    x = np.zeros(cell.n_hidden) if x0 is None else np.asarray(x0, dtype=float)
    srs = []
    for t in range(n_steps):
        J = jacobian_step(cell, x, seq[t])
        srs.append(np.max(np.abs(np.linalg.eigvals(J))))
        x = np.asarray(cell.step(x, seq[t])[0])
    return float(np.max(srs))


def lyapunov_exponent(cell: CfCCell, seq, x0=None, n_steps=None,
                      seed: int = 0, eps: float = 1e-9) -> float:
    """Maior expoente de Lyapunov empírico (taxa contínua = média(log||δ||)/dt).

    LE < 0 ⇒ contração (sub-caótico). LE > 0 ⇒ caótico.
    """
    seq = np.asarray(seq, dtype=float)
    if n_steps is None:
        n_steps = len(seq)
    rng = np.random.default_rng(seed)

    x = np.zeros(cell.n_hidden) if x0 is None else np.asarray(x0, dtype=float)
    delta = rng.normal(0.0, 1.0, cell.n_hidden)
    delta = delta / np.linalg.norm(delta)
    logs = []
    for t in range(n_steps):
        J = jacobian_step(cell, x, seq[t])
        delta = J @ delta
        norm = np.linalg.norm(delta)
        logs.append(np.log(norm + eps))
        delta = delta / norm if norm > 0 else rng.normal(0.0, 1.0, cell.n_hidden)
        x = np.asarray(cell.step(x, seq[t])[0])
    return float(np.mean(logs) / float(cell.dt))


def state_norm_trajectory(cell: CfCCell, seq, x0=None) -> np.ndarray:
    """Norma do estado ao longo da trajetória (para inspecionar colapso/explosão)."""
    seq = np.asarray(seq, dtype=float)
    x = np.zeros(cell.n_hidden) if x0 is None else np.asarray(x0, dtype=float)
    norms = []
    for t in range(len(seq)):
        x = np.asarray(cell.step(x, seq[t])[0])
        norms.append(float(np.linalg.norm(x)))
    return np.asarray(norms)


def is_dead(cell: CfCCell, seq, tol: float = 1e-3) -> bool:
    """Reservatório 'morto' = congelado: não responde à entrada (variância de ||x|| ~ 0).

    Não confundir com o ponto de repouso não-nulo: sob entrada nula o estado
    converge para x* dirigido por A (o potencial de reversão), que é estável e
    correto. 'Morto' é quando, MESMO sob entrada agitada, a trajetória não se
    move — neurônios saturados ou dinâmica colapsada.
    """
    norms = state_norm_trajectory(cell, seq)
    return bool(np.std(norms) < tol)


def report(cell: CfCCell, seq, seed: int = 0) -> dict:
    """Relatório completo de estabilidade para um cell + sequência de entrada."""
    sr_w = spectral_radius_Wrec(cell)
    sr_j = max_jacobian_spectral_radius(cell, seq)
    le = lyapunov_exponent(cell, seq, seed=seed)
    norms = state_norm_trajectory(cell, seq)
    dead = is_dead(cell, seq)
    return {
        "n_hidden": cell.n_hidden,
        "dt": cell.dt,
        "spectral_radius_Wrec": sr_w,
        "max_jacobian_spectral_radius": sr_j,
        "lyapunov_exponent": le,
        "state_norm_final": float(norms[-1]),
        "state_norm_min": float(norms.min()),
        "state_norm_max": float(norms.max()),
        "state_dynamics_std": float(np.std(norms)),
        "dead": dead,
    }


if __name__ == "__main__":
    cell = CfCCell(n_in=2, n_hidden=96, dt=0.1, seed=0)
    T = 600
    rng = np.random.default_rng(1)
    seq = rng.normal(0, 1.5, (T, 2))            # entrada agitada
    seq_null = np.zeros((T, 2))                 # entrada nula (repouso)

    r = report(cell, seq, seed=0)
    rn = report(cell, seq_null, seed=0)

    print("=" * 60)
    print("ESTABILIDADE DO RESERVATÓRIO LÍQUIDO — tarefa 1.8")
    print("=" * 60)
    print(f"\nConfig: n_hidden={r['n_hidden']}, dt={r['dt']}, T={T}")
    print("\n-- Raio espectral (entrada agitada) --")
    print(f"  SR(W_rec) congelado        : {r['spectral_radius_Wrec']:.4f}")
    print(f"  SR(jacobiano local) máx.   : {r['max_jacobian_spectral_radius']:.4f}")
    print("\n-- Expoente de Lyapunov (entrada agitada) --")
    print(f"  LE (taxa contínua)         : {r['lyapunov_exponent']:+.4f}")
    print("  interpretação              :",
          "SUB-CAÓTICO (contrativo)" if r['lyapunov_exponent'] < 0 else "CAÓTICO")
    print("\n-- Estados (entrada agitada) --")
    print(f"  ||x|| final / min / max    : {r['state_norm_final']:.4f} / "
          f"{r['state_norm_min']:.4f} / {r['state_norm_max']:.4f}")
    print("\n-- Entrada nula (repouso) --")
    print(f"  ||x|| final                : {rn['state_norm_final']:.6f} "
          f"(ponto fixo dirigido por A; não nulo, estável)")
    dead = r['dead']
    chaotic = r['lyapunov_exponent'] > 0
    print("\n-- Veredito --")
    print(f"  caótico?  : {'SIM (ruim)' if chaotic else 'não'}")
    print(f"  morto?    : {'SIM (ruim)' if dead else 'não'}")
    print(f"  utilizável: {'SIM' if (not chaotic and not dead) else 'NÃO'}")
