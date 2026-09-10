"""
test_metaplasticity.py — Teste simples do módulo de metaplasticidade (tarefa 8.3).

Verifica:
1. MetaPlasticityLearner treina e reduz erro
2. lr por neurônio diverge (neurônios com erro estável têm lr alto)
3. MetaPlasticityController funciona standalone
4. Estado salva/carrega corretamente
"""

import sys
from pathlib import Path

import numpy as np

# Garantir que o projeto seja importável
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visao.metaplasticity import MetaPlasticityLearner, MetaPlasticityController


def test_basic_training():
    """Teste 1: MetaPlasticityLearner treina e reduz erro."""
    print("=" * 60)
    print("TESTE 1: Treinamento básico com MetaPlasticityLearner")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_hidden, n_out = 32, 3
    learner = MetaPlasticityLearner(
        n_hidden=n_hidden,
        n_out=n_out,
        lr=0.05,
        meta_alpha=2.0,
        meta_beta=0.9,
        rng=rng,
    )

    # Sintético: y = W*x + ruído
    W_true = rng.normal(0, 0.5, (n_out, n_hidden))
    errors = []
    for i in range(200):
        x = rng.normal(0, 1, n_hidden)
        y = W_true @ x + rng.normal(0, 0.05, n_out)
        err_sq, _ = learner.update(x, y)
        errors.append(err_sq)

    # Verificar que erro diminuiu
    first_20 = np.mean(errors[:20])
    last_20 = np.mean(errors[-20:])
    reduction = (first_20 - last_20) / first_20 * 100

    print(f"  Erro inicial (média primeiros 20): {first_20:.4f}")
    print(f"  Erro final (média últimos 20):     {last_20:.4f}")
    print(f"  Redução: {reduction:.1f}%")
    assert last_20 < first_20, "Erro deve diminuir com treinamento"
    print("  ✓ PASS: Erro reduziu com treinamento\n")


def test_lr_divergence():
    """Teste 2: lr por neurônio diverge baseado na variância do erro."""
    print("=" * 60)
    print("TESTE 2: Divergência de lr por neurônio")
    print("=" * 60)

    rng = np.random.default_rng(123)
    n_hidden, n_out = 20, 4
    learner = MetaPlasticityLearner(
        n_hidden=n_hidden,
        n_out=n_out,
        lr=0.05,
        meta_alpha=3.0,
        meta_beta=0.9,
        rng=rng,
    )

    # Criar targets com variâncias diferentes por neurônio
    # Neurônio 0: erro baixo e estável
    # Neurônio 1: erro alto e volátil
    # Neurônio 2: erro médio
    # Neurônio 3: erro muito volátil
    W_true = rng.normal(0, 0.3, (n_out, n_hidden))
    noise_levels = [0.01, 0.5, 0.1, 1.0]  # variância do ruído por neurônio

    for i in range(300):
        x = rng.normal(0, 1, n_hidden)
        y = W_true @ x
        for j in range(n_out):
            y[j] += rng.normal(0, noise_levels[j])
        learner.update(x, y)

    stats = learner.get_lr_stats()
    print(f"  lr médio:  {stats['lr_mean']:.6f}")
    print(f"  lr std:    {stats['lr_std']:.6f}")
    print(f"  lr min:    {stats['lr_min']:.6f}")
    print(f"  lr max:    {stats['lr_max']:.6f}")
    print(f"  lr por neurônio: {stats['lr_per_neuron']}")
    print(f"  σ por neurônio:  {stats['sigma_per_neuron']}")

    # Neurônio com menor ruído (0) deve ter lr MAIOR
    # Neurônio com maior ruído (3) deve have lr MENOR
    lr = stats['lr_per_neuron']
    sigma = stats['sigma_per_neuron']

    print(f"\n  Neurônio 0 (σ={sigma[0]:.4f}): lr={lr[0]:.6f}")
    print(f"  Neurônio 3 (σ={sigma[3]:.4f}): lr={lr[3]:.6f}")

    assert lr[0] > lr[3], f"Neurônio estável (0) deve ter lr > neurônio volátil (3)"
    assert stats['lr_std'] > 0, "lr deve variar entre neurônios"
    print("  ✓ PASS: lr diverge corretamente (estável=alto, volátil=baixo)\n")


def test_controller_standalone():
    """Teste 3: MetaPlasticityController funciona standalone."""
    print("=" * 60)
    print("TESTE 3: MetaPlasticityController standalone")
    print("=" * 60)

    n_out = 3
    ctrl = MetaPlasticityController(
        n_out=n_out,
        lr_base=0.05,
        meta_alpha=2.0,
        meta_beta=0.9,
    )

    # Simular erros: neurônio 0 estável, neurônio 1 volátil
    rng = np.random.default_rng(99)
    for i in range(100):
        err = np.array([
            rng.normal(0, 0.01),   # estável
            rng.normal(0, 0.5),    # volátil
            rng.normal(0, 0.1),    # médio
        ])
        ctrl.update_lr(err)

    lr = ctrl.get_lr()
    print(f"  lr final: {lr}")
    print(f"  Neurônio 0 (estável): lr={lr[0]:.6f}")
    print(f"  Neurônio 1 (volátil): lr={lr[1]:.6f}")

    assert lr[0] > lr[1], "Neurônio estável deve ter lr > volátil"
    print("  ✓ PASS: Controller standalone funciona\n")


def test_state_save_load():
    """Teste 4: Estado salva e carrega corretamente."""
    print("=" * 60)
    print("TESTE 4: Save/Load de estado")
    print("=" * 60)

    rng = np.random.default_rng(7)
    learner = MetaPlasticityLearner(n_hidden=16, n_out=2, lr=0.05, rng=rng)

    # Treinar um pouco
    for i in range(50):
        x = rng.normal(0, 1, 16)
        y = rng.normal(0, 0.5, 2)
        learner.update(x, y)

    st = learner.state()
    lr_before = learner.lr_per_neuron.copy()

    # Criar novo learner e carregar estado
    learner2 = MetaPlasticityLearner(n_hidden=16, n_out=2, lr=0.05)
    learner2.load(st)
    lr_after = learner2.lr_per_neuron

    print(f"  lr antes: {lr_before}")
    print(f"  lr depois: {lr_after}")
    assert np.allclose(lr_before, lr_after), "Estado deve ser preservado"
    print("  ✓ PASS: Estado salvo/carregado corretamente\n")


def test_comparison_with_fixed_lr():
    """Teste 5: Comparação com lr fixo (baseline)."""
    print("=" * 60)
    print("TESTE 5: Comparação MetaPlasticity vs lr fixo")
    print("=" * 60)

    from prototype.plasticity import LocalLearner

    rng = np.random.default_rng(2024)
    n_hidden, n_out = 40, 4

    # MetaPlasticityLearner
    meta_learner = MetaPlasticityLearner(
        n_hidden=n_hidden, n_out=n_out, lr=0.05, meta_alpha=2.0, rng=rng
    )
    # LocalLearner (lr fixo)
    fixed_learner = LocalLearner(
        n_hidden=n_hidden, n_out=n_out, lr=0.05, rng=rng
    )

    # Problema com ruído não-estacionário
    W_true = rng.normal(0, 0.4, (n_out, n_hidden))
    meta_errors = []
    fixed_errors = []

    for i in range(400):
        x = rng.normal(0, 1, n_hidden)
        # Ruído muda na metade do treinamento
        if i < 200:
            noise = rng.normal(0, 0.05, n_out)
        else:
            noise = rng.normal(0, 0.3, n_out)  # ruído aumenta
        y = W_true @ x + noise

        err_meta, _ = meta_learner.update(x, y)
        err_fixed, _ = fixed_learner.update(x, y)
        meta_errors.append(err_meta)
        fixed_errors.append(err_fixed)

    meta_final = np.mean(meta_errors[-50:])
    fixed_final = np.mean(fixed_errors[-50:])

    print(f"  MetaPlasticity erro final: {meta_final:.4f}")
    print(f"  lr fixo erro final:        {fixed_final:.4f}")
    print(f"  Razão (meta/fixed):        {meta_final/fixed_final:.2f}")
    print("  ✓ PASS: Comparação concluída\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTE DO MÓDULO DE METAPLASTICIDADE (Tarefa 8.3)")
    print("=" * 60 + "\n")

    test_basic_training()
    test_lr_divergence()
    test_controller_standalone()
    test_state_save_load()
    test_comparison_with_fixed_lr()

    print("=" * 60)
    print("TODOS OS TESTES PASSARAM ✓")
    print("=" * 60)
