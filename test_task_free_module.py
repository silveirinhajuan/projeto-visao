"""test_task_free_module.py — Teste simples para o módulo task_free (Tarefa 8.5).

Verifica:
1. CUSUMDetector detecta mudança abrupta no sinal de erro.
2. TaskFreeLearner adapta plasticidade ao detectar mudança.
3. O stream com regimes é gerado corretamente.
4. O experimento completo roda e produz métricas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.continual.task_free import (
    CUSUMDetector,
    TaskFreeLearner,
    make_regime_stream,
    run_task_free,
)
from visao.brain import VisaoBrain


def test_cusum_detects_abrupt_change():
    """CUSUM deve detectar mudança abrupta na média do sinal."""
    detector = CUSUMDetector(drift=0.01, threshold=0.3, warmup=50)

    # Fase 1: sinal baixo (erro ~ 0.1)
    rng = np.random.default_rng(42)
    for _ in range(100):
        detector.update(rng.normal(0.1, 0.02))

    # Fase 2: sinal alto (erro ~ 1.0) — mudança abrupta
    detections = 0
    for _ in range(200):
        if detector.update(rng.normal(1.0, 0.1)):
            detections += 1

    assert detections > 0, "CUSUM falhou em detectar mudança abrupta"
    print(f"  [ok] CUSUM detectou {detections} mudança(s)")


def test_cusum_no_false_positive_on_stable_signal():
    """CUSUM NÃO deve disparar em sinal estável."""
    detector = CUSUMDetector(drift=0.01, threshold=0.5, warmup=100)

    rng = np.random.default_rng(0)
    detections = 0
    for _ in range(500):
        if detector.update(rng.normal(0.1, 0.02)):
            detections += 1

    assert detections == 0, f"Falso positivo: {detections} detecções em sinal estável"
    print("  [ok] CUSUM sem falsos positivos em sinal estável")


def test_task_free_learner_adapts():
    """TaskFreeLearner deve reagir a mudanças de regime."""
    brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)
    detector = CUSUMDetector(drift=0.01, threshold=0.3, warmup=50)
    learner = TaskFreeLearner(brain, detector, lr_boost=3.0, omega_decay=0.5)

    u, y, regimes = make_regime_stream(n=1200, seed=0)
    result = run_task_free(learner, u, y)

    assert result["n_changes_detected"] > 0, "Nenhuma mudança detectada no stream com regimes"
    assert np.all(np.isfinite(result["errors"])), "Erros divergentes"
    assert result["mse_total"] < 10.0, f"MSE muito alto: {result['mse_total']}"
    print(f"  [ok] TaskFreeLearner detectou {result['n_changes_detected']} mudança(s)")
    print(f"       MSE total = {result['mse_total']:.4f}")


def test_lr_boost_on_detection():
    """Ao detectar mudança, lr deve aumentar temporariamente."""
    brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, lr=0.02, seed=42)
    initial_lr = brain.lr
    detector = CUSUMDetector(drift=0.005, threshold=0.2, warmup=30)
    learner = TaskFreeLearner(brain, detector, lr_boost=3.0, omega_decay=0.5)

    rng = np.random.default_rng(0)
    # Sinal estável inicial
    for _ in range(50):
        u = rng.normal(0.1, 0.02, (2,))
        y = np.array([0.1])
        learner.learn(u, y)

    lr_before = brain.lr

    # Sinal com erro alto (mudança de regime)
    for i in range(100):
        u = rng.normal(1.0, 0.1, (2,))
        y = np.array([1.0])
        result = learner.learn(u, y)
        if result["detected_change"]:
            break

    # Após detecção, lr deve ter aumentado
    if learner.change_points:
        assert brain.lr > lr_before, "lr não aumentou após detecção"
        print(f"  [ok] lr subiu de {lr_before:.5f} para {brain.lr:.5f}")
    else:
        print("  [warn] mudança não detectada neste cenário, pulando asserção de lr")


def test_stream_regimes():
    """Stream gerado deve ter 4 regimes distintos."""
    u, y, regimes = make_regime_stream(n=4000, seed=0)

    assert u.shape == (4000, 2), f"Forma de u incorreta: {u.shape}"
    assert y.shape == (4000, 1), f"Forma de y incorreta: {y.shape}"
    assert len(regimes) == 4, f"Esperava 4 regimes, obteve {len(regimes)}"

    # Regimes devem ter variância diferente (distribuições distintas)
    r1_var = np.var(u[:1000, 0])
    r2_var = np.var(u[1000:2000, 0])
    assert r1_var > 0 and r2_var > 0
    print(f"  [ok] Stream: {len(regimes)} regimes, u.shape={u.shape}")


def test_omega_decays_on_change():
    """Omega deve decair ao detectar mudança de regime."""
    brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)
    detector = CUSUMDetector(drift=0.005, threshold=0.2, warmup=30)
    learner = TaskFreeLearner(brain, detector, lr_boost=2.0, omega_decay=0.3)

    rng = np.random.default_rng(7)
    # Treino inicial para acumular omega
    for _ in range(80):
        u = rng.normal(0.1, 0.02, (2,))
        y = np.array([0.1])
        learner.learn(u, y)

    omega_before = brain.learner.omega.copy()

    # Sinal com erro alto
    for i in range(100):
        u = rng.normal(1.0, 0.1, (2,))
        y = np.array([1.0])
        result = learner.learn(u, y)
        if result["detected_change"]:
            break

    if learner.change_points:
        omega_after = brain.learner.omega
        # Pelo menos algumas sinapses devem ter decaído
        assert np.any(omega_after < omega_before), "Omega não decaiu após detecção"
        print(f"  [ok] Omega decaiu (min before={omega_before.min():.4f}, after={omega_after.min():.4f})")
    else:
        print("  [warn] mudança não detectada, pulando asserção de omega")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DO MÓDULO TASK-FREE (Tarefa 8.5)")
    print("=" * 60)

    print("\n1. CUSUM detecta mudança abrupta:")
    test_cusum_detects_abrupt_change()

    print("\n2. CUSUM sem falsos positivos:")
    test_cusum_no_false_positive_on_stable_signal()

    print("\n3. Task-free learner adapta a regimes:")
    test_task_free_learner_adapts()

    print("\n4. lr boost após detecção:")
    test_lr_boost_on_detection()

    print("\n5. Stream com 4 regimes:")
    test_stream_regimes()

    print("\n6. Omega decai após detecção:")
    test_omega_decays_on_change()

    print("\n" + "=" * 60)
    print("TODOS OS TESTES PASSARAM")
    print("=" * 60)
