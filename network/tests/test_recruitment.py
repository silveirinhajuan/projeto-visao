"""
test_recruitment.py — O modelo de recrutamento precisa estar CERTO.

Se a aritmética que justifica a estratégia de rede estiver errada, o projeto
recruta gente com base em ficção. Estes testes protegem as afirmações que
foram escritas no FASE5_REDE_HUMANA.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from network.recruitment_model import (  # noqa: E402
    DeviceMix,
    FunnelParams,
    cloud_equivalent_usd_year,
    folding_at_home_reality_check,
    funnel,
    retention_curve,
    scenario,
    steady_state,
    throughput_tflops,
)


class TestFunil:
    def test_funil_multiplica_as_taxas(self):
        p = FunnelParams(reached=100_000, learn_rate=0.20, install_rate=0.05)
        assert funnel(p) == 1_000

    def test_taxa_default_respeita_faixa_boinc(self):
        """Anderson: 5-10% dos que CONHECEM participam. Não podemos ser otimistas."""
        assert 0.05 <= FunnelParams().install_rate <= 0.10


class TestRetencao:
    def test_sem_entrada_a_rede_morre(self):
        """A afirmação central do documento: viral sem retenção converge a zero."""
        curva = retention_curve(10_000, monthly_churn=0.30, months=36, monthly_inflow=0)
        assert curva[-1] == 0, "sem entrada contínua a rede DEVE morrer"
        assert curva == sorted(curva, reverse=True), "decaimento deve ser monotônico"

    def test_equilibrio_e_entrada_sobre_churn(self):
        assert steady_state(150, 0.12) == pytest.approx(1250.0)
        assert steady_state(20, 0.03) == pytest.approx(666.67, rel=1e-3)

    def test_curva_converge_para_o_equilibrio(self):
        """A simulação mês a mês tem que bater com a fórmula fechada."""
        inflow, churn = 150, 0.12
        curva = retention_curve(0, churn, months=200, monthly_inflow=inflow)
        assert curva[-1] == pytest.approx(steady_state(inflow, churn), rel=0.01)

    def test_converge_por_cima_tambem(self):
        """Começar acima do equilíbrio também converge — para baixo."""
        inflow, churn = 150, 0.12
        eq = steady_state(inflow, churn)
        curva = retention_curve(int(eq * 5), churn, months=200, monthly_inflow=inflow)
        assert curva[-1] == pytest.approx(eq, rel=0.01)

    def test_churn_zero_nao_divide_por_zero(self):
        assert steady_state(100, 0.0) == float("inf")


class TestVazao:
    def test_disponibilidade_reduz_a_vazao(self):
        """Um nó não computa 24h. Ignorar isso infla o número em ~40%."""
        mix = DeviceMix()
        real = throughput_tflops(1000, mix)
        ideal_mix = DeviceMix(desktop_availability=1.0, mobile_availability=1.0)
        assert real < throughput_tflops(1000, ideal_mix)

    def test_vazao_escala_linear(self):
        mix = DeviceMix()
        assert throughput_tflops(2000, mix) == pytest.approx(
            2 * throughput_tflops(1000, mix))

    def test_custo_nuvem_usa_a_referencia_de_anderson(self):
        """Anderson: 50 GFLOPS por US$0,24/h. 1 PFLOP = 20.000 instâncias."""
        custo = cloud_equivalent_usd_year(1000.0)  # 1 PFLOPS
        esperado = 20_000 * 0.24 * 24 * 365
        assert custo == pytest.approx(esperado)
        assert custo > 40e6, "deve reproduzir a ordem de US$42M/ano do paper"


class TestCenarios:
    def test_viral_perde_para_nicho_no_longo_prazo(self):
        """A conclusão que justifica a estratégia inteira do projeto."""
        mix = DeviceMix()
        viral = scenario("viral", FunnelParams(1_000_000, 0.20, 0.05, 0.30), mix, 0)
        nicho = scenario("nicho", FunnelParams(50_000, 0.30, 0.08, 0.12), mix, 150)

        assert viral["instalaram"] > nicho["instalaram"], "viral atrai mais gente"
        assert viral["equilibrio"] < nicho["equilibrio"], "e entrega menos no fim"
        assert viral["equilibrio"] == 0

    def test_ancoras_institucionais_sao_estaveis(self):
        mix = DeviceMix()
        inst = scenario("inst", FunnelParams(5_000, 0.50, 0.20, 0.03), mix, 20)
        assert inst["ativos_mes_24"] > 0.8 * inst["instalaram"], \
            "churn de 3% deve preservar a maioria dos nós em 2 anos"

    def test_todo_cenario_reporta_equilibrio(self):
        mix = DeviceMix()
        for p, inflow in [(FunnelParams(), 0), (FunnelParams(), 100)]:
            s = scenario("x", p, mix, inflow)
            assert "equilibrio" in s and s["equilibrio"] >= 0


class TestCalibragemFoldingAtHome:
    def test_queda_de_98_porcento_esta_correta(self):
        """Se este número mudar, o documento inteiro precisa ser reescrito."""
        f = folding_at_home_reality_check()
        assert f["queda_pct"] == pytest.approx(98.6, abs=0.1)

    def test_pico_supera_exaflop(self):
        f = folding_at_home_reality_check()
        assert f["pico_pflops_mar_2020"] > 1000, "1,22 exaflops = 1220 PFLOPS"

    def test_crescimento_foi_de_mais_de_dez_vezes(self):
        f = folding_at_home_reality_check()
        ratio = f["voluntarios_mar_2020"] / f["voluntarios_fev_2020"]
        assert ratio > 10, "30k -> 400k em um mês"
