"""
test_surprise_decay.py — TDD p/ SurpriseDecayLearner (Tarefa 7.1).

Cenários:
1. update() reduz erro com o tempo (aprende)
2. surpresa alta → omega decai (Ω afrouxa quando erro explode)
3. surpresa baixa → omega cresce normalmente (protege o que é importante)
4. sd=0 → equivale a LocalLearner sem surpresa (controle)
5. sd=0 (ótimo real) supera naive (benefício líquido)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liquid import LiquidCell
from plasticity import LocalLearner, NaiveLearner

from analysis.surprise_decay import SurpriseDecayLearner, make_routine_task, run_two_tasks


def test_learner_aprende_reduzindo_erro():
    """SurpriseDecayLearner aprende: erro cai com o treino."""
    rng = np.random.default_rng(42)
    cell = LiquidCell(n_in=2, n_hidden=48, sparsity=0.6, dt=0.15, rng=rng)
    learner = SurpriseDecayLearner(
        48, 1, lr=0.02, consolidation=8.0, surprise_gain=4.0,
        surprise_decay=0.0, rng=rng,
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


def test_surprise_alta_decai_omega():
    """Quando surpresa explode (mudança de regime), Ω deve decair (sd>0)."""
    rng = np.random.default_rng(7)
    learner = SurpriseDecayLearner(
        24, 1, lr=0.02, consolidation=8.0, surprise_gain=4.0,
        surprise_decay=0.5, rng=rng,
    )
    x = rng.normal(0, 0.3, 24)

    # Primeiro: treino normal para estabilizar err_ema/err_var
    for _ in range(100):
        target = np.array([0.5])
        learner.update(x, target)

    omega_antes = learner.omega.copy()
    omega_norm_antes = np.linalg.norm(omega_antes)

    # Força surpresa alta (erro muito maior que o baseline)
    target_grande = np.array([10.0])
    for _ in range(20):
        learner.update(x, target_grande)

    omega_depois = learner.omega.copy()
    omega_norm_depois = np.linalg.norm(omega_depois)

    assert omega_norm_depois < omega_norm_antes, \
        f"Ω não decaiu: antes={omega_norm_antes:.4f} depois={omega_norm_depois:.4f}"


def test_surprise_baixa_omega_cresce():
    """Quando surpresa é baixa (erro estável), Ω cresce normalmente."""
    rng = np.random.default_rng(99)
    learner = SurpriseDecayLearner(
        24, 1, lr=0.02, consolidation=8.0, surprise_gain=4.0,
        surprise_decay=0.5, rng=rng,
    )
    x = rng.normal(0, 0.3, 24)

    # Treino estável (erro pequeno e consistente)
    for _ in range(200):
        target = np.array([0.3])
        learner.update(x, target)

    # Ω deve ter crescido (foi atualizado por omega += 0.01 * |delta|)
    omega_norm = np.linalg.norm(learner.omega)
    assert omega_norm > 0.0, f"Ω não cresceu: norm={omega_norm:.4f}"


def test_sd_zero_equivale_sem_surpresa():
    """sd=0 equivale a LocalLearner com surprise_gain=0 (consolidação only)."""
    rng = np.random.default_rng(123)
    learner_sd0 = SurpriseDecayLearner(
        24, 1, lr=0.02, consolidation=8.0, surprise_gain=0.0,
        surprise_decay=0.0, rng=rng,
    )
    learner_base = LocalLearner(
        24, 1, lr=0.02, consolidation=8.0, surprise_gain=0.0, rng=np.random.default_rng(123),
    )

    x = rng.normal(0, 0.3, 24)
    target = np.array([0.5])

    for _ in range(50):
        learner_sd0.update(x, target)
        learner_base.update(x, target)

    # ω devem ser idênticos (mesma seed, mesma lógica)
    assert np.allclose(learner_sd0.omega, learner_base.omega, atol=1e-10)


def test_optimo_sd_zero_supera_naive():
    """ACHADO 7.1: sd=0 (consol+Oja sem surpresa) é ótimo e supera naive."""
    seeds = (1, 2, 3, 4, 5)
    plastic_fgs, naive_fgs = [], []

    for seed in seeds:
        # Ótimo: sd=0.0 (surpresa não amplifica lr, não decai omega)
        p_fg = run_two_tasks(seed, surprise_decay=0.0, n_steps=2500, n_hidden=96)
        plastic_fgs.append(p_fg)

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

    p_mean = float(np.mean(plastic_fgs))
    n_mean = float(np.mean(naive_fgs))
    red = (1 - p_mean / n_mean) * 100 if n_mean > 0 else float("nan")
    print(f"\n  plastic sd=0: {p_mean:+.4f}  naive: {n_mean:+.4f}  red={red:.1f}%")

    assert p_mean < n_mean, f"Plástico sd=0 ({p_mean:+.4f}) não superou naive ({n_mean:+.4f})"
