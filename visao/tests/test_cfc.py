"""
test_cfc.py — Contrato da célula líquida em JAX.

Este teste define O QUE a célula CfC precisa ser antes de ela existir.
Escrito ANTES de visao/core/cfc.py (TDD estrito, tarefa 1.2 do backlog).

A asserção central é a que separa uma rede LÍQUIDA de um RNN caro:
a constante de tempo efetiva TEM que variar com a entrada. Se não variar,
portamos um RNN com passos extras e devemos admitir isso.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

jax = pytest.importorskip("jax", reason="tarefa 1.1 instala jax[cpu]")
jnp = pytest.importorskip("jax.numpy")

from visao.core.cfc import CfCCell  # noqa: E402


def entrada_calma(n: int = 300) -> np.ndarray:
    """Sinal quase estático: a rede deveria relaxar em constantes de tempo longas."""
    t = np.arange(n)
    u = np.zeros((n, 2))
    u[:, 1] = 0.05 * np.sin(t * 0.02)
    return u


def entrada_agitada(n: int = 300, seed: int = 7) -> np.ndarray:
    """Ruído forte: a rede deveria acelerar, encurtando tau."""
    return np.random.default_rng(seed).normal(0, 1.5, (n, 2))


class TestLiquidez:
    """A propriedade que justifica a arquitetura inteira."""

    def test_tau_varia_com_a_entrada(self):
        cell = CfCCell(n_in=2, n_hidden=64, dt=0.15, seed=7)

        _, acts_calma = cell.rollout(entrada_calma())
        _, acts_agitada = cell.rollout(entrada_agitada())

        tau_calma = np.asarray(cell.tau_effective(acts_calma[100:]))
        tau_agitada = np.asarray(cell.tau_effective(acts_agitada[100:]))

        faixa_calma = tau_calma.max() - tau_calma.min()
        faixa_agitada = tau_agitada.max() - tau_agitada.min()

        assert faixa_agitada > 2 * faixa_calma, (
            f"a rede não é líquida: faixa de tau agitada={faixa_agitada:.4f} "
            f"vs calma={faixa_calma:.4f}. Esperado ao menos 2x."
        )

    def test_tau_efetivo_e_sempre_positivo(self):
        """tau <= 0 significa dinâmica sem sentido físico."""
        cell = CfCCell(n_in=2, n_hidden=32, dt=0.15, seed=1)
        _, acts = cell.rollout(entrada_agitada(200))
        tau = np.asarray(cell.tau_effective(acts))
        assert np.all(tau > 0)

    def test_tau_efetivo_nunca_excede_tau_base(self):
        """tau_ef = tau/(1 + tau*f) com f >= 0 => tau_ef <= tau. Sanidade da fórmula."""
        cell = CfCCell(n_in=2, n_hidden=32, dt=0.15, seed=2)
        _, acts = cell.rollout(entrada_agitada(200))
        tau = np.asarray(cell.tau_effective(acts))
        assert np.all(tau <= np.asarray(cell.tau)[None, :] + 1e-6)


class TestEstabilidade:
    def test_estados_nao_explodem(self):
        cell = CfCCell(n_in=2, n_hidden=64, dt=0.15, seed=3)
        states, _ = cell.rollout(entrada_agitada(500))
        s = np.asarray(states)
        assert np.all(np.isfinite(s)), "NaN/Inf no rollout"
        assert np.abs(s).max() < 1e3, "estados divergiram"

    def test_entrada_nula_converge(self):
        """Sem entrada, o vazamento deve levar a um estado de repouso estável."""
        cell = CfCCell(n_in=2, n_hidden=32, dt=0.15, seed=4)
        states, _ = cell.rollout(np.zeros((400, 2)))
        s = np.asarray(states)
        movimento_final = np.abs(s[-1] - s[-50]).max()
        assert movimento_final < 1e-2, f"não convergiu: {movimento_final:.5f}"


class TestFormaEContrato:
    def test_formatos_de_saida(self):
        cell = CfCCell(n_in=3, n_hidden=48, dt=0.1, seed=5)
        seq = np.zeros((120, 3))
        states, acts = cell.rollout(seq)
        assert states.shape == (120, 48)
        assert acts.shape == (120, 48)

    def test_step_avanca_um_passo(self):
        cell = CfCCell(n_in=2, n_hidden=16, dt=0.1, seed=6)
        x0 = jnp.zeros(16)
        x1, f1 = cell.step(x0, jnp.array([1.0, -1.0]))
        assert x1.shape == (16,)
        assert f1.shape == (16,)
        assert not np.allclose(np.asarray(x0), np.asarray(x1))

    def test_determinismo_por_seed(self):
        a = CfCCell(n_in=2, n_hidden=32, dt=0.15, seed=42)
        b = CfCCell(n_in=2, n_hidden=32, dt=0.15, seed=42)
        u = entrada_agitada(150)
        assert np.allclose(np.asarray(a.rollout(u)[0]), np.asarray(b.rollout(u)[0]))

    def test_esparsidade_recorrente_e_respeitada(self):
        """A topologia é invariante: onde a máscara é zero, o peso é zero."""
        cell = CfCCell(n_in=2, n_hidden=40, sparsity=0.6, dt=0.15, seed=8)
        W = np.asarray(cell.W_rec)
        mask = np.asarray(cell.mask)
        assert np.all(W[mask == 0] == 0.0)
        assert np.all(np.diag(W) == 0.0), "sem auto-sinapse; o vazamento já faz isso"


class TestBrainPersistencia:
    """O cérebro precisa sair do processo vivo e voltar idêntico.

    save_brain()/load_brain() é a fundação da Fase 6 (agente autocontido)
    e da Fase 3 (arquivo de variantes). Se o round-trip não for bit-faithful
    a nível de 1e-6, todo o resto da Fase 6 é construído sobre areia.
    """

    def test_save_load_roundtrip_pesos(self, tmp_path):
        """Pesos salvos e recarregados numa célula com seed DIFERENTE devem bater."""
        orig = CfCCell(n_in=2, n_hidden=48, dt=0.1, seed=11)
        path = tmp_path / "brain"
        orig.save_brain(path)

        carregada = CfCCell(n_in=2, n_hidden=48, dt=0.1, seed=99)
        carregada.load_brain(path)

        for name in ("W_in", "W_rec", "b", "A", "mask", "tau"):
            a = np.asarray(getattr(orig, name))
            b = np.asarray(getattr(carregada, name))
            assert np.allclose(a, b, atol=1e-6), (
                f"{name} divergiu no round-trip: max err "
                f"{np.abs(a - b).max():.3e}"
            )

    def test_save_load_roundtrip_rollout(self, tmp_path):
        """Rollout da célula recarregada deve ser numericamente idêntico."""
        u = entrada_agitada(150, seed=3)
        orig = CfCCell(n_in=2, n_hidden=48, dt=0.1, seed=11)
        path = tmp_path / "brain"
        orig.save_brain(path)

        carregada = CfCCell(n_in=2, n_hidden=48, dt=0.1, seed=99)
        carregada.load_brain(path)

        so, _ = orig.rollout(u)
        sc, _ = carregada.rollout(u)
        assert np.allclose(np.asarray(so), np.asarray(sc), atol=1e-6), (
            f"rollout divergiu após round-trip: max err "
            f"{np.abs(np.asarray(so) - np.asarray(sc)).max():.3e}"
        )
