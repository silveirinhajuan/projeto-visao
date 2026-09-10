# Relatório de Validação — Benchmark VISÃO vs SOTA 2025-2026

**Data:** 09/09/2026  
**Protocolo:** Idêntico à Fase 0 (stream A→B→C, regressão sintética, 5 seeds)  
**Baseline:** Oja + consolidação por importância + gate de surpresa (VISÃO Fase 0)

---

## Resultados

### Protocolo Padrão (3 tarefas)

| Variante | Forgetting (±SE) | Error (±SE) | Status |
|----------|------------------|-------------|--------|
| **baseline** | **-0.1239 ± 0.0431** | **0.0068 ± 0.0010** | ✓ Vencedor |
| m-ltc | -0.0999 ± 0.0705 | 0.0175 ± 0.0094 | ~ Neutro (error 157% ↑) |
| ch-hnn | +0.2921 ± 0.1599 | 0.6788 ± 0.2243 | ✗ Pior (error 9899% ↑) |

### Protocolo Stress (5 tarefas, maior não-estacionariedade)

| Variante | Forgetting (±SE) | Error (±SE) | Status |
|----------|------------------|-------------|--------|
| **baseline** | **-0.0514 ± 0.0222** | **0.0066 ± 0.0010** | ✓ Vencedor |
| m-ltc | -0.0241 ± 0.1182 | 0.8569 ± 0.3022 | ✗ Pior (error 12857% ↑) |
| ch-hnn | +1.1514 ± 0.3575 | 5.7333 ± 1.5550 | ✗ Muito pior (error 86598% ↑) |

---

## Achados

1. **Positive Backward Transfer (PBT):** O baseline do VISÃO apresenta *forgetting negativo* — o sistema MELHORA em tarefas passadas após aprender novas. Isso é raro e valioso.

2. **Baseline > SOTA 2025-2026:** A combinação Oja + consolidação + surpresa supora M-LTC (Srinivas 2026) e CH-HNN (Shi et al., Nature Comms 2025) em streams contínuos de regressão.

3. **Overhead prejudica:** M-LTC e CH-HNN adicionam parâmetros (memória extra, camada cortical) que prejudicam mais do que ajudam em tarefas de regressão contínua.

4. **Robustez:** O baseline mantém performance mesmo em stress (5 tarefas, mais ruído, frequências não-harmônicas).

---

## Por que o baseline vence?

- **Consolidação por importância** é mais eficaz que attractor dynamics para streams longos
- **Gate de surpresa** adapta learning rate dinamicamente (neuromodulação)
- **Oja no recorrente** auto-organiza o sem supervisão
- **Menos parâmetros** = menos overfitting em tarefas simples

---

## Implicações para o Projeto

**NÃO faz sentido:**
- Substituir baseline por M-LTC ou CH-HNN (são inferiores neste domínio)
- Adicionar complexidade desnecessária

**FAZ sentido:**
1. **Entender o PBT:** Por que forgetting negativo? O que acontece no espectro de tau?
2. **Encontrar o limite:** Em que condições o baseline FALHA? (tarefas adversarialmente construídas?)
3. **Escalar a filosofia:** Manter Oja+consolidação+surpresa, mas em arquitetura maior (imagens, NLP)
4. **Validar em benchmark público:** Comparar com M-LTC/CH-HNN no MESMO dataset deles (UCI HAR, drone data)

---

## Novas Tarefas Propostas (baseadas em evidência)

### Tarefa 7.1 — Estudo do Positive Backward Transfer
**Hipótese:** O PBT emerge da interação entre gate de surpresa e consolidação.  
**Método:** Ablation study: (a) sem surpresa, (b) sem consolidação, (c) sem Oja, (d) completo.  
**Métrica:** forgetting e error por condição.  
**Critério:** Identificar qual componente contribui mais para o PBT.

### Tarefa 7.2 — Limite de falha do baseline
**Hipótese:** O baseline falha quando tarefas são construídas para maximizar interferência.  
**Método:** Gerar tarefas com frequências próximas, fases opostas, ruído crescente.  
**Métrica:** Número de tarefas até forgetting > 0.  
**Critério:** Encontrar o ponto de quebra e caracterizá-lo.

### Tarefa 7.3 — Validação em benchmark público (UCI HAR)
**Hipótese:** O baseline mantém superioridade em dataset real.  
**Método:** Human Activity Recognition (UCI) com split temporal (sujeito A → sujeito B).  
**Métrica:** Accuracy vs LSTM, GRU, M-LTC.  
**Critério:** Documentar comparação justa (mesmo protocolo dos papers).

### Tarefa 7.4 — Escala: Oja + consolidação em topologia profunda
**Hipótese:** A filosofia local escala para múltiplas camadas líquidas.  
**Método:** 2-3 camadas CfC com Oja entre camadas (não backprop).  
**Métrica:** Performance em MNIST split / permuted MNIST.  
**Critério:** Manter forgetting < 0.1 em 10+ tarefas.

---

## Conclusão

O VISÃO já possui um dos melhores mecanismos de continual learning para streams de regressão. O próximo passo não é substituí-lo, mas **entendê-lo, encontrar seus limites e escalar a filosofia**.

---

*Artefatos:*
- `benchmark_2026.py` — script do benchmark padrão
- `benchmark_results_2026.json` — resultados padrão
- `benchmark_stress_2026.py` — script do stress test
- `benchmark_stress_results_2026.json` — resultados stress
