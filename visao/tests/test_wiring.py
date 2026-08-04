"""
test_wiring.py — Contrato da topologia Neural Circuit Policy (NCP).

Escrito ANTES de visao/core/wiring.py (TDD estrito, tarefa 1.4 do backlog).

A NCP (Lechner et al., Nature MI 2020) dirige um carro com 19 neurônios.
A esparsidade NÃO é economia — é a topologia que dá generalização. O teste
central exige que, após um update de plasticidade que toca CADA peso (inclusive
os proibidos), a máscara NCP seja reaplicada e nenhuma sinapse proibida sobreviva.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.core.wiring import NCPWiring  # noqa: E402


class TestTopologiaNCP:
    """A estrutura direcionada sensory→inter→command→motor."""

    def test_estrutura_basica_com_sparsity_zero(self):
        """Com sparsity=0 o grafo é exatamente a NCP: blocos permitidos cheios, proibidos vazios."""
        w = NCPWiring(n_sensory=4, n_inter=6, n_command=3, n_motor=2, sparsity=0.0, seed=0)
        mask = np.asarray(w.mask)
        assert mask.shape == (15, 15)

        s0, s1 = w.groups["sensory"]
        i0, i1 = w.groups["inter"]
        c0, c1 = w.groups["command"]
        m0, m1 = w.groups["motor"]

        # sem auto-sinapse
        assert np.all(np.diag(mask) == 0)

        # sensory não recebe recorrência (linhas sensory = 0)
        assert np.all(mask[s0:s1, :] == 0)
        # motor não tem saída (colunas motor = 0)
        assert np.all(mask[:, m0:m1] == 0)
        # command não alimenta inter nem a si mesmo
        assert np.all(mask[i0:i1, c0:c1] == 0)
        assert np.all(mask[c0:c1, c0:c1] == 0)

        # blocos permitidos totalmente presentes (sparsity=0)
        assert np.all(mask[i0:i1, s0:s1] == 1)      # sensory → inter
        assert np.all(mask[c0:c1, s0:s1] == 1)      # sensory → command
        assert np.all(mask[c0:c1, i0:i1] == 1)      # inter → command
        assert np.all(mask[m0:m1, i0:i1] == 1)      # inter → motor
        assert np.all(mask[m0:m1, c0:c1] == 1)      # command → motor
        # inter → inter: fora da diagonal deve estar cheio (sparsity=0)
        inter_block = mask[i0:i1, i0:i1].copy()
        np.fill_diagonal(inter_block, 1)  # diagonal já é zero por design
        assert np.all(inter_block == 1)

    def test_mascara_respeitada_apos_plasticidade(self):
        """Update de plasticidade que altera TODOS os pesos; a topologia deve sobreviver."""
        w = NCPWiring(n_sensory=5, n_inter=8, n_command=4, n_motor=3, sparsity=0.8, seed=1)
        n = w.n_total
        rng = np.random.default_rng(99)
        w_full = rng.normal(0, 1.0, (n, n))
        delta = rng.normal(0, 0.5, (n, n))           # toca cada peso, inclusive proibidos
        w_plastic = w_full + delta

        w_masked = w.enforce(w_plastic)

        forbidden = w.mask == 0
        allowed = w.mask == 1
        # toda posição proibida continua exatamente zero
        assert np.all(w_masked[forbidden] == 0.0)
        # posições permitidas preservam o valor plástico
        assert np.allclose(w_masked[allowed], w_plastic[allowed])
        # sanidade: o plástico REALMENTE violava a topologia
        assert np.any(np.abs(w_plastic[forbidden]) > 0)

    def test_esparsidade_reduz_conexoes(self):
        densa = NCPWiring(4, 6, 3, 2, sparsity=0.0, seed=2)
        esparsa = NCPWiring(4, 6, 3, 2, sparsity=0.9, seed=2)
        assert np.count_nonzero(esparsa.mask) < np.count_nonzero(densa.mask)

    def test_determinismo_por_seed(self):
        a = NCPWiring(4, 6, 3, 2, sparsity=0.5, seed=7)
        b = NCPWiring(4, 6, 3, 2, sparsity=0.5, seed=7)
        assert np.array_equal(a.mask, b.mask)

    def test_enforce_rejeita_formato_errado(self):
        w = NCPWiring(4, 6, 3, 2, sparsity=0.0, seed=3)
        with pytest.raises(ValueError):
            w.enforce(np.zeros((10, 10)))
