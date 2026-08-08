# Problema Zero — Decisão de Recrutamento (tarefa 5.2)

> Decidido autonomamente por ÍRIS em 2026-08-08, com base em pesquisa e nos
> critérios duros da FASE5_REDE_HUMANA.md, após autorização explícita do Juan.
> Esta é a decisão de *qual* problema; o recrutamento em si segue bloqueado
> pelo guardião F5.0 (5.4) até existir um cliente instalável em <5min.

## Veredito

**Problema Zero = Nowcasting de precipitação em bacias urbanas de Fortaleza,
a partir da rede esparsa de pluviômetros FUNCEME.**

Entregar: um modelo CfC (pequeno, executável em laptop) que, dada a série
temporal dos últimos N minutos dos postos vizinhos, nowcasta o volume de
chuva acumulada em janelas de 1h / 3h / 6h para cada posto. Cada nó voluntário
roda um CfC local treinado nos postos da sua região e compartilha deltas de
gradiente (Fase 2) — o organismo melhora onde chove, não num datacenter.

**Fronteira de responsabilidade (escrita e pública):** isto é uma ferramenta de
*nowcasting* e pesquisa, **não** um sistema oficial de alerta de cheia da
Defesa Civil. O projeto não emite ordem de evacuação. Honestidade obrigatória
do Pilar 3 da FASE5.

## Matriz de decisão (4 critérios do projeto)

| Critério | FUNCEME (escolhido) | ECG TinyML | Irrigação semi-árido |
|---|---|---|---|
| **Local** (Caucaia/Fortaleza) | ✅ rede dos seus postos | ❌ dataset global | ✅ Ceará |
| **Mensurável** | ✅ RMSE (mm) 1/3/6h | ✅ F1/sensibilidade | ✅ água poupada |
| **Dados disponíveis** | ✅ 600+ postos 1974+ | ✅ MIT-BIH/PhysioNet | ❌ sem dataset pronto |
| **LTC/CfC bate alternativa** | ✅ não-estacionário favorece aprendizado contínuo | ❌ CNN congelada já faz 92–98% | ⚠️ viável, mas sem dados hoje |

ECG e irrigação foram descartados: ECG falha o critério "local" e já é resolvido
por CNN; irrigação falha "dados disponíveis" (exigiria desdobrar sensores
primeiro). FUNCEME é o único que satisfaz os quatro de imediato.

## Dados reais (verificados em 2026-08-08)

- **FUNCEME**: 600+ estações pluviométricas, série 1974–atual, dados públicos.
  Página oficial de postos: `funceme.br/?page_id=2694`. Repositório de
  pré-processamento da comunidade: `github.com/rubensocj/series-FUNCEME`.
- **INMET BDMEP**: dados históricos nacionais como complemento/cross-check.
- Licença: dados governamentais abertos do Ceará — usáveis para pesquisa.

## Métrica pública (definida ANTES de começar — regra da FASE5)

- **Alvo:** RMSE do volume de chuva acumulada em janelas de 1h, 3h, 6h.
- **Baseline:** persistência/climatologia por posto.
- **Critério de sucesso:** o ensemble CfC (nós federados) reduz o RMSE do
  baseline em **≥15%** na divisão temporal (treino ≤2020, teste 2021–2024) —
  a divisão temporal é o teste de que o modelo *continua* útil sob deriva
  climática, que é where a arquitetura líquida ganha de um modelo congelado.
- **Reuso:** o harness de benchmark da Fase 1 (`visao/bench/timeseries.py`)
  serve de base — troca-se a entrada (série FUNCEME) e a métrica (RMSE).

## Por que isto recruta (Pilar 3 da FASE5)

"Um problema da sua cidade" é o gancho honesto: o voluntário em Fortaleza vê o
modelo nowcastar a chuva do *próprio* bairro e recebe o organismo treinado de
volta (Pilar 2). Começar por casa é verificável e não exige prometer salvar o
mundo.

## Plano de rollout (FASE5_F5.1 → F5.2)

1. **F5.1 (semente, 10–30 nós, meses 1–3):** pessoas que o Juan conhece —
   colegas do cursinho ITA, grupo de estudo. Alta tolerância a bug, alto
   feedback. Meta: churn <10%/mês.
2. **F5.2 (âncoras institucionais, cenário C):** UFC, IFCE, UNIFOR — 666 nós
   estáveis = 62,6 TFLOPS (a espinha dorsal).
3. **F5.3 (nicho fiel, cenário B):** comunidades TinyML/neuromórfica — 1.250
   nós = 117,5 TFLOPS.

## Pré-requisito em aberto (honestidade do guardião 5.4)

O guardião F5.0 ainda deve dizer **PROIBIDO RECRUTAR** porque o item
"cliente instalável em <5min" falta (`network/install_bench.json` inexistente
— não há cliente empacotado ainda). A *decisão* do problema está tomada; o
*recrutamento executável* aguarda o cliente. Não se deve adulterar o benchmark
para forçar "AUTORIZADO".
