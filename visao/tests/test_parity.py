"""
test_parity.py — O porte para JAX é MESMO o mesmo modelo?

Sem este teste, "portar para JAX" é só reescrever o código e torcer.
Mesma seed, mesma topologia, mesma sequência: os estados têm que bater.

Se este arquivo falhar, os resultados da Fase 0 (97,2% menos esquecimento)
NÃO transferem para a implementação JAX, e o número precisa ser remedido
antes de ser citado em qualquer lugar.

Nota sobre precisão: JAX usa float32 por padrão. Habilitamos x64 para
comparar maçãs com maçãs contra o numpy float64.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

jax = pytest.importorskip("jax", reason="tarefa 1.1 instala jax[cpu]")
jax.config.update("jax_enable_x64", True)

from prototype.liquid import LiquidCell  # noqa: E402
from visao.core.cfc import CfCCell  # noqa: E402

SEED = 123
N_IN, N_HIDDEN = 2, 48
DT, SPARSITY = 0.15, 0.5


def par_de_celulas():
    """Duas células construídas com a MESMA sequência de números aleatórios."""
    np_cell = LiquidCell(
        n_in=N_IN, n_hidden=N_HIDDEN, sparsity=SPARSITY, dt=DT,
        rng=np.random.default_rng(SEED),
    )
    jax_cell = CfCCell(
        n_in=N_IN, n_hidden=N_HIDDEN, sparsity=SPARSITY, dt=DT,
        rng=np.random.default_rng(SEED),
    )
    return np_cell, jax_cell


def sequencia(n: int = 300, seed: int = 999) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    u = rng.normal(0, 0.8, (n, N_IN))
    u[:, 0] += np.sin(t * 0.07)
    return u


class TestParidadeDeParametros:
    """Antes de comparar dinâmica, os pesos precisam ser idênticos."""

    def test_pesos_identicos(self):
        npc, jxc = par_de_celulas()
        assert np.allclose(npc.W_in, np.asarray(jxc.W_in))
        assert np.allclose(npc.W_rec, np.asarray(jxc.W_rec))
        assert np.allclose(npc.b, np.asarray(jxc.b))

    def test_tau_e_A_identicos(self):
        npc, jxc = par_de_celulas()
        assert np.allclose(npc.tau, np.asarray(jxc.tau))
        assert np.allclose(npc.A, np.asarray(jxc.A))

    def test_mascara_de_esparsidade_identica(self):
        npc, jxc = par_de_celulas()
        assert np.array_equal(npc.mask, np.asarray(jxc.mask))


class TestParidadeDinamica:
    """A prova real: mesma entrada, mesma trajetória."""

    def test_um_passo_bate(self):
        npc, jxc = par_de_celulas()
        x0 = np.zeros(N_HIDDEN)
        u = np.array([0.7, -0.3])

        x_np, f_np = npc.step(x0, u)
        x_jx, f_jx = jxc.step(x0, u)

        assert np.allclose(x_np, np.asarray(x_jx), atol=1e-10)
        assert np.allclose(f_np, np.asarray(f_jx), atol=1e-10)

    def test_rollout_completo_bate(self):
        """O critério do backlog: atol=1e-5 ao longo de toda a sequência."""
        npc, jxc = par_de_celulas()
        u = sequencia(300)

        s_np, a_np = npc.rollout(u)
        s_jx, a_jx = jxc.rollout(u)

        erro_estados = np.abs(s_np - np.asarray(s_jx)).max()
        erro_ativacoes = np.abs(a_np - np.asarray(a_jx)).max()

        assert erro_estados < 1e-5, f"estados divergem: {erro_estados:.3e}"
        assert erro_ativacoes < 1e-5, f"ativações divergem: {erro_ativacoes:.3e}"

    def test_erro_nao_acumula_ao_longo_do_tempo(self):
        """Divergência crescente indicaria solver diferente, não ruído numérico."""
        npc, jxc = par_de_celulas()
        u = sequencia(600)

        s_np, _ = npc.rollout(u)
        s_jx, _ = jxc.rollout(u)
        erro = np.abs(s_np - np.asarray(s_jx)).max(axis=1)

        assert erro[-1] < 1e-5
        assert erro[-1] < 100 * (erro[:50].max() + 1e-15), "erro cresce sem controle"

    def test_tau_efetivo_bate(self):
        npc, jxc = par_de_celulas()
        u = sequencia(200)

        _, a_np = npc.rollout(u)
        _, a_jx = jxc.rollout(u)

        tau_np = npc.tau_effective(a_np)
        tau_jx = np.asarray(jxc.tau_effective(a_jx))

        assert np.abs(tau_np - tau_jx).max() < 1e-5

    def test_liquidez_medida_e_a_mesma_nas_duas_implementacoes(self):
        """A alegação de '13x de variação de tau' precisa valer nas duas."""
        npc, jxc = par_de_celulas()
        calma = np.zeros((300, N_IN))
        calma[:, 1] = 0.05 * np.sin(np.arange(300) * 0.02)
        agitada = np.random.default_rng(7).normal(0, 1.5, (300, N_IN))

        def faixa(cell, seq, is_jax: bool):
            _, acts = cell.rollout(seq)
            tau = np.asarray(cell.tau_effective(acts[100:]))
            return float(tau.max() - tau.min())

        r_np = faixa(npc, agitada, False) / faixa(npc, calma, False)
        r_jx = faixa(jxc, agitada, True) / faixa(jxc, calma, True)

        assert r_np == pytest.approx(r_jx, rel=1e-4)
        assert r_jx > 1.0, "a implementação JAX precisa ser líquida também"
