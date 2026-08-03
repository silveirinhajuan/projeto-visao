"""
recruitment_model.py — Modelo quantitativo da rede de voluntários do VISÃO.

Motivo de existir: "convencer pessoas a doar computação" é uma frase de slide.
Este arquivo transforma a frase em aritmética falseável — quantos voluntários
são necessários, quantos ficam, e o que a rede realmente entrega.

PARÂMETROS ANCORADOS EM DADOS PUBLICADOS, não em otimismo:

  Anderson, "BOINC: A Platform for Volunteer Computing" (arXiv:1903.01699):
    - 5% a 10% das pessoas que CONHECEM computação voluntária participam
    - desktops disponíveis ~60% do tempo; móveis ~40%
    - projeto típico: ~US$100k/ano de operação para ~1 PetaFLOPS
    - equivalente em EC2: US$42M/ano  →  420x mais barato
    - 85% Windows, 7% Mac, 7% Linux — heterogeneidade é a regra

  Folding@home (Ars Technica, Wikipedia, Reddit r/Folding 2025):
    - fev/2020:  30.000 voluntários ativos
    - mar/2020: 400.000 voluntários  →  1,22 EXAFLOPS (pico histórico)
    - 2025:     ~17 PFLOPS
    - QUEDA DE 98,6% DO PICO.

A última linha é a lição central deste modelo: uma crise RECRUTA,
mas só estrutura RETÉM. Um plano de recrutamento que não modela churn
está mentindo para si mesmo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


# ════════════════════════════════════════════════════════════ parâmetros

@dataclass
class FunnelParams:
    """Funil de aquisição. Valores default = faixa conservadora do BOINC."""

    reached: int = 100_000          # pessoas que veem a mensagem
    learn_rate: float = 0.20        # leem o suficiente para entender
    install_rate: float = 0.05      # dos que entendem, quantos instalam (BOINC: 5-10%)
    monthly_churn: float = 0.25     # fração que abandona por mês


@dataclass
class DeviceMix:
    """Mix de hardware. Rede voluntária é heterogênea por definição."""

    frac_desktop: float = 0.75
    frac_mobile: float = 0.25
    desktop_availability: float = 0.60   # BOINC: ~60% do tempo
    mobile_availability: float = 0.40    # BOINC: ~40% do tempo
    desktop_gflops: float = 200.0        # CPU+iGPU modesto, sustentado
    mobile_gflops: float = 40.0


# ════════════════════════════════════════════════════════════ dinâmica

def funnel(p: FunnelParams) -> int:
    """Quantos INSTALAM, a partir de quantos foram alcançados."""
    return int(p.reached * p.learn_rate * p.install_rate)


def retention_curve(initial: int, monthly_churn: float, months: int,
                    monthly_inflow: int = 0) -> list[int]:
    """Voluntários ativos mês a mês.

    Modelo: decaimento exponencial + entrada constante. O ponto fixo é
        N* = inflow / churn
    ou seja: sem recrutamento CONTÍNUO, a rede converge para zero.
    Foi exatamente o que aconteceu com o Folding@home pós-pandemia.
    """
    out, n = [], float(initial)
    for _ in range(months):
        n = n * (1 - monthly_churn) + monthly_inflow
        out.append(int(n))
    return out


def steady_state(monthly_inflow: int, monthly_churn: float) -> float:
    """Tamanho de equilíbrio da rede. A única métrica que importa a longo prazo."""
    return monthly_inflow / monthly_churn if monthly_churn > 0 else float("inf")


def throughput_tflops(n_volunteers: int, mix: DeviceMix) -> float:
    """TFLOPS efetivos — já descontada a disponibilidade real dos aparelhos."""
    d = n_volunteers * mix.frac_desktop * mix.desktop_availability * mix.desktop_gflops
    m = n_volunteers * mix.frac_mobile * mix.mobile_availability * mix.mobile_gflops
    return (d + m) / 1000.0


def cloud_equivalent_usd_year(tflops: float) -> float:
    """Custo anual do MESMO poder em nuvem (Anderson: 50 GFLOPS por US$0,24/h)."""
    instances = (tflops * 1000.0) / 50.0
    return instances * 0.24 * 24 * 365


# ════════════════════════════════════════════════════════════ cenários

def scenario(name: str, p: FunnelParams, mix: DeviceMix,
             monthly_inflow: int, months: int = 24) -> dict:
    installed = funnel(p)
    curve = retention_curve(installed, p.monthly_churn, months, monthly_inflow)
    eq = steady_state(monthly_inflow, p.monthly_churn)
    tf_peak = throughput_tflops(installed, mix)
    tf_eq = throughput_tflops(int(eq), mix)
    return {
        "cenario": name,
        "alcancados": p.reached,
        "instalaram": installed,
        "churn_mensal": p.monthly_churn,
        "entrada_mensal": monthly_inflow,
        "ativos_mes_6": curve[5] if len(curve) > 5 else None,
        "ativos_mes_24": curve[-1],
        "equilibrio": int(eq),
        "tflops_pico": round(tf_peak, 1),
        "tflops_equilibrio": round(tf_eq, 1),
        "custo_nuvem_equivalente_usd_ano": round(cloud_equivalent_usd_year(tf_eq)),
    }


def folding_at_home_reality_check() -> dict:
    """O caso real, com os números publicados. Serve de calibragem e de aviso."""
    pico_pflops, hoje_pflops = 1220.0, 17.0
    return {
        "caso": "Folding@home",
        "voluntarios_fev_2020": 30_000,
        "voluntarios_mar_2020": 400_000,
        "pico_pflops_mar_2020": pico_pflops,
        "pflops_2025": hoje_pflops,
        "queda_pct": round((1 - hoje_pflops / pico_pflops) * 100, 1),
        "licao": "Uma crise recruta. Só estrutura retém.",
    }


def main() -> None:
    mix = DeviceMix()

    cenarios = [
        # Otimismo ingênuo: viraliza uma vez e ninguém cuida da retenção.
        scenario(
            "A) Viral sem retenção",
            FunnelParams(reached=1_000_000, learn_rate=0.20,
                         install_rate=0.05, monthly_churn=0.30),
            mix, monthly_inflow=0),

        # Realista: divulgação modesta mas CONTÍNUA, churn tratado como problema.
        scenario(
            "B) Nicho fiel + fluxo contínuo",
            FunnelParams(reached=50_000, learn_rate=0.30,
                         install_rate=0.08, monthly_churn=0.12),
            mix, monthly_inflow=150),

        # Institucional: laboratórios e universidades. Poucos nós, alta retenção.
        scenario(
            "C) Âncoras institucionais",
            FunnelParams(reached=5_000, learn_rate=0.50,
                         install_rate=0.20, monthly_churn=0.03),
            mix, monthly_inflow=20),
    ]

    print("=" * 74)
    print("MODELO DE RECRUTAMENTO — REDE DE VOLUNTÁRIOS DO PROJETO VISÃO")
    print("Parâmetros ancorados em Anderson (arXiv:1903.01699) e Folding@home")
    print("=" * 74)

    fah = folding_at_home_reality_check()
    print(f"\nCALIBRAGEM — {fah['caso']}:")
    print(f"  fev/2020: {fah['voluntarios_fev_2020']:,} voluntários")
    print(f"  mar/2020: {fah['voluntarios_mar_2020']:,} voluntários "
          f"→ {fah['pico_pflops_mar_2020']:.0f} PFLOPS")
    print(f"  2025    : {fah['pflops_2025']:.0f} PFLOPS")
    print(f"  QUEDA DO PICO: {fah['queda_pct']}%")
    print(f"  >>> {fah['licao']}")

    for c in cenarios:
        print(f"\n{'-' * 74}")
        print(f"{c['cenario']}")
        print(f"  alcançados {c['alcancados']:,} → instalaram {c['instalaram']:,}")
        print(f"  churn {c['churn_mensal']:.0%}/mês | entrada {c['entrada_mensal']}/mês")
        print(f"  ativos mês 6: {c['ativos_mes_6']:,} | mês 24: {c['ativos_mes_24']:,}")
        print(f"  EQUILÍBRIO: {c['equilibrio']:,} nós  →  {c['tflops_equilibrio']} TFLOPS")
        print(f"  mesmo poder em nuvem: US$ {c['custo_nuvem_equivalente_usd_ano']:,}/ano")

    print(f"\n{'=' * 74}")
    print("CONCLUSÃO ARITMÉTICA")
    print("=" * 74)
    a, b, c = cenarios
    print(f"O cenário viral (A) atrai {a['instalaram']:,} pessoas e estabiliza em"
          f" {a['equilibrio']:,}.")
    print(f"O cenário de nicho (B) atrai {b['instalaram']:,} — {a['instalaram']//max(b['instalaram'],1)}x menos —"
          f" e estabiliza em {b['equilibrio']:,}.")
    print("\nSem entrada contínua, TODA rede converge para zero. Churn é a física")
    print("do problema; recrutamento em massa sem retenção é encher balde furado.")
    print(f"\nÂncoras institucionais (C): apenas {c['equilibrio']:,} nós, mas churn de")
    print(f"{c['churn_mensal']:.0%}/mês os torna a espinha dorsal — {c['tflops_equilibrio']} TFLOPS estáveis.")

    out = Path(__file__).parent / "results_recruitment.json"
    out.write_text(json.dumps(
        {"folding_at_home": fah, "cenarios": cenarios,
         "device_mix": asdict(mix)}, indent=2, ensure_ascii=False))
    print(f"\nresultados → {out}")


if __name__ == "__main__":
    main()
