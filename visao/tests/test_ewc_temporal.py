"""
test_ewc_temporal.py — TDD p/ EWCTemporalLearner (Tarefa 7.2).

Cenários:
1. update() reduz erro com o tempo (aprende)
2. omega decai exponencialmente com lambda_decay > 0
3. lambda=0 → equivale a EWC clássico (sem decaimento)
4. time incrementa a cada update
5. EWC-temporal com ótimo lambda supera naive
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquid import LiquidCell
from plasticity import NaiveLearner

from analysis.ewc_temporal import EWCTemporalLearner, make_routine_task, run_two_tasks


def test_learner_aprende():
    """EWCTemporalLearner aprende: erro cai com o treino."""
    rng = np.random.default_rng(42)
    cell = LiquidCell(n_in=2, n_hidden=48, sparsity=0.6, dt=0.15, rng=rng)
    learner = EWCTemporalLearner(
        48, 1, lr=0.02, consolidation=8.0, surprise_gain=0.0,
        lambda_decay=0.001, rng=rng,
    )
    u, y = make_routine_task("aulas", 800, seed=1)

    x = np.zeros(48)
    errs_inicio, errs_fim = [], []
    for i, (ui, yi) in enumerate(zip(u, y)):
        x, _ = cell.step(x, ui)
        if i >= 50:
            pred = learner.predict(x)
            e = float(((yi - pred) ** 2).mean())
            if i < 200:
                errs_inicio.append(e)
            elif i >= 600:
                errs_fim.append(e)
            learner.update(x, yi)

    assert np.mean(errs_fim) < np.mean(errs_inicio), \
        f"Erro não caiu: inicio={np.mean(errs_inicio):.3f} fim={np.mean(errs_fim):.3f}"


def test_omega_decai_com_lambda():
    """Com lambda_decay > 0, omega decai ao longo do tempo."""
    rng = np.random.default_rng(7)
    learner = EWCTemporalLearner(
        24, 1, lr=0.02, consolidation=8.0, surprise_gain=0.0,
        lambda_decay=0.01, rng=rng,
    )
    x = rng.normal(0, 0.3, 24)

    # Treino estável para acumular omega
    for _ in range(200):
        target = np.array([0.3])
        learner.update(x, target)

    omega_antes = np.linalg.norm(learner.omega)
    time_antes = learner.time

    # Mais treino — omega deve decair (lambda_decay > 0)
    for _ in range(200):
        target = np.array([0.3])
        learner.update(x, target)

    omega_depois = np.linalg.norm(learner.omega)
    time_depois = learner.time

    assert time_depois > time_antes, "time não incrementou"
    # Com decaimento forte (0.01), omega deve ter caído significativamente
    assert omega_depois < omega_antes, \
        f"Ω não decaiu: antes={omega_antes:.4f} depois={omega_depois:.4f}"


def test_lambda_zero_equivale_sem_decaimento():
    """lambda=0 ≡ LocalLearner sem surpresa (consolidação only)."""
    rng = np.random.default_rng(123)
    learner_l0 = EWCTemporalLearner(
        24, 1, lr=0.02, consolidation=8.0, surprise_gain=0.0,
        lambda_decay=0.0, rng=rng,
    )
    # LocalLearner com mesmos params
    from plasticity import LocalLearner
    learner_base = LocalLearner(
        24, 1, lr=0.02, consolidation=8.0, surprise_gain=0.0,
        rng=np.random.default_rng(123),
    )

    x = rng.normal(0, 0.3, 24)
    target = np.array([0.5])

    for _ in range(50):
        learner_l0.update(x, target)
        learner_base.update(x, target)

    # ω devem ser praticamente idênticos
    # (pequena diferença pelo fator exp(-0)*omega = omega, mesma coisa)
    assert np.allclose(learner_l0.omega, learner_base.omega, atol=1e-10)


def test_time_incrementa():
    """Contador time incrementa a cada update."""
    rng = np.random.default_rng(55)
    learner = EWCTemporalLearner(
        12, 1, lr=0.02, lambda_decay=0.001, rng=rng,
    )
    x = rng.normal(0, 0.2, 12)

    for i in range(100):
        assert learner.time == i, f"time={learner.time} != {i}"
        learner.update(x, np.array([0.5]))
    assert learner.time == 100


def test_ewc_temporal_supera_naive():
    """Ótimo lambda: EWC-temporal supera naive (5 seeds)."""
    seeds = (1, 2, 3, 4, 5)
    ewc_fgs, naive_fgs = [], []

    for seed in seeds:
        # Ótimo provisório: lambda=0.001
        ewc_fg = run_two_tasks(seed, lambda_decay=0.001, n_steps=2500, n_hidden=96)
        ewc_fgs.append(ewc_fg)

        # Naive
        rng = np.random.default_rng(seed)
        cell = LiquidCell(n_in=2, n_hidden=96, sparsity=0.6, dt=0.15, rng=rng)
        naive = NaiveLearner(96, 1, lr=0.02, rng=rng)

        def evaluate(cell, learner, u, y, warmup=50):
            x = np.zeros(96)
            errs = []
            for i, (ui, yi) in enumerate(zip(u, y)):
                x, _ = cell.step(x, ui)
                if i >= warmup:
                    pred = learner.predict(x)
                    e = float(np.clip(((yi - pred) ** 2).mean(), 0.0, 1e6))
                    errs.append(e)
            return float(np.mean(errs))

        def train(cell, learner, u, y, warmup=50):
            x = np.zeros(96)
            for i, (ui, yi) in enumerate(zip(u, y)):
                x, _ = cell.step(x, ui)
                if i >= warmup:
                    learner.update(x, yi)

        u1, y1 = make_routine_task("aulas", 2500, seed * 10 + 1)
        u2, y2 = make_routine_task("provas", 2500, seed * 10 + 2)
        u1_te, y1_te = make_routine_task("aulas", 1500, seed * 10 + 501)

        train(cell, naive, u1, y1)
        ra1 = evaluate(cell, naive, u1_te, y1_te)
        train(cell, naive, u2, y2)
        final1 = evaluate(cell, naive, u1_te, y1_te)
        naive_fgs.append(final1 - ra1)

    e_mean = float(np.mean(ewc_fgs))
    n_mean = float(np.mean(naive_fgs))
    red = (1 - e_mean / n_mean) * 100 if n_mean > 0 else float("nan")
    print(f"\n  ewc lambda=0.001: {e_mean:+.4f}  naive: {n_mean:+.4f}  red={red:.1f}%")

    assert e_mean < n_mean, f"EWC-temporal ({e_mean:+.4f}) não superou naive ({n_mean:+.4f})"
