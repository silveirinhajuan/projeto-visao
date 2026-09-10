# Plano de Investigação: Mecanismo de Surpresa

## Contexto

O estudo de ablação revelou que o componente **Surprise** é o mais crítico do VISÃO:
- **Sem surprise**: forgetting = 1.3% (excelente), accuracy = 53.3% (ruim)
- **Com surprise**: forgetting = 5.3%, accuracy = 60.7%
- **Baseline (sem nada)**: forgetting = 8.4%, accuracy = 64.6%

**Hipótese**: O `surprise_gain=3.0` está causando instabilidade porque:
1. Amplifica o learning rate em mudanças de regime → sistema empenado
2. Interage negativamente com a consolidação EWC
3. Não decai ao longo do tempo (surpresa constante = ruído)

## Objetivo

Determinar a configuração ótima de surpresa que:
- Mantém forgetting baixo (< 3%)
- Maximiza accuracy (> 60%)
- É estável ao longo do tempo

## Experimentos

### Exp 1: Sweep de Surprise Gain
- Variar `surprise_gain` em: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]
- Métricas: accuracy, forgetting, BWT, wall time
- 5 seeds cada
- Esperado: curva em U-inverted (muito baixo = subótimo, muito alto = instável)

### Exp 2: Surprise + Consolidação (EWC)
- Testar combinações:
  - surprise=0, EWC=0 (baseline)
  - surprise=ótimo, EWC=0
  - surprise=0, EWC=8.0
  - surprise=ótimo, EWC=8.0
  - surprise=ótimo, EWC=2.0
  - surprise=ótimo, EWC=20.0
- Esperado: EWC alto + surprise alto = conflito (EWC quer consolidar, surpresa quer mudar)

### Exp 3: Surprise com Decaimento Temporal
- Implementar `surprise_gain(t) = gain_0 * exp(-lambda * t)`
- Testar lambda: [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
- Esperado: decaimento melhora estabilidade sem perder benefício inicial

### Exp 4: Análise Qualitativa da Surpresa
- Medir magnitude da surpresa ao longo do treinamento
- Plotar: surprise_t vs task_boundary vs forgetting
- Esperado: surpresa spike na mudança de tarefa → instabilidade

## Critério de Sucesso

Encontrar configuração onde:
- Accuracy > 65%
- Forgetting < 3%
- BWT ≥ 0 (sem transferência negativa)

## Resultado Esperado

Um modelo de surpresa adaptativa que:
1. É alta no início de cada tarefa (exploração)
2. Decai ao longo da tarefa (estabilização)
3. Não conflita com EWC (ou EWC é ajustado)
