"""
test_stability.py — Tarefa 1.8 do backlog: estabilidade do reservatório.

O diagnóstico que decide se a arquitetura líquida é utilizável como memória
contínua. Regra do projeto (BACKLOG 1.8): o reservatório NÃO pode ser caótico
(LE < 0) NEM morto (estados colapsam / não respondem à entrada).

Estes testes travam esse contrato com números medidos, não com opinião.
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
from visao.analysis.stability import (  # noqa: E402
    lyapunov_exponent,
    max_jacobian_spectral_radius,
    spectral_radius_Wrec,
    is_dead,
    report,
)


def _agitada(n=600, seed=1):
    return np.random.default_rng(seed).normal(0, 1.5, (n, 2))


class TestEspectro:
    def test_Wrec_eh_subunitario(self):
        """Receita ESN clássica: SR(W_rec) < 1 evita explosão do recurso cru."""
        cell = CfCCell(n_in=2, n_hidden=96, dt=0.1, seed=0)
        sr = spectral_radius_Wrec(cell)
        assert sr < 1.0, f"SR(W_rec)={sr:.4f} >= 1: reservatório explosivo"

    def test_jacobiano_efetivo_fica_abaixo_da_unidade(self):
        """SR do jacobiano LOCAL (inclui sigmoid + vazamento tau) medido ao longo
        da trajetória. < 1 => contrativo local => 'borda do caos' do lado estável."""
        cell = CfCCell(n_in=2, n_hidden=96, dt=0.1, seed=0)
        sr_j = max_jacobian_spectral_radius(cell, _agitada())
        assert sr_j < 1.0, f"SR(jacobiano)={sr_j:.4f} >= 1: expansão local"


class TestLyapunov:
    def test_expoente_de_lyapunov_e_negativo(self):
        """LE < 0 => sub-caótico: perturbações pequenas contraem. Memória estável."""
        cell = CfCCell(n_in=2, n_hidden=96, dt=0.1, seed=0)
        le = lyapunov_exponent(cell, _agitada(), seed=0)
        assert le < 0, f"LE={le:+.4f} >= 0: reservatório CAÓTICO — inútil como memória"

    def test_nao_e_morto(self):
        """'Morto' = trajetória não se move sob entrada agitada (std de ||x|| ~ 0).
        O ponto de repouso não-nulo dirigido por A é ESTÁVEL e correto, não 'morto'."""
        cell = CfCCell(n_in=2, n_hidden=96, dt=0.1, seed=0)
        r = report(cell, _agitada(), seed=0)
        assert not r["dead"], "reservatório morto: não responde à entrada"
        # e os estados estão em escala finita, nem colapsados nem explodidos
        assert 1e-2 < r["state_norm_final"] < 1e2, \
            f"||x|| final fora de faixa saudável: {r['state_norm_final']:.4f}"


class TestReprodutibilidade:
    def test_report_e_deterministico_por_seed(self):
        """Mesma seed -> mesmo LE (a dinâmica real é jit-determinística dada a seed)."""
        cell_a = CfCCell(n_in=2, n_hidden=64, dt=0.1, seed=11)
        cell_b = CfCCell(n_in=2, n_hidden=64, dt=0.1, seed=11)
        seq = _agitada(400, seed=3)
        le_a = lyapunov_exponent(cell_a, seq, seed=7)
        le_b = lyapunov_exponent(cell_b, seq, seed=7)
        assert np.isclose(le_a, le_b, atol=1e-6), f"LE não reprodutível: {le_a} vs {le_b}"
