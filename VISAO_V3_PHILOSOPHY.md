# VISÃO — Versão 3.0: Arquitetura Cognitiva Agentica

> *"Eu quero que o modelo seja capaz de ser treinado como modelos atuais são, mas também a partir do uso contínuo."*
> — Juan Guerra, 2026

---

## 1. Filosofia do Projeto

VISÃO não é apenas uma rede neural líquida. É uma **arquitetura cognitiva agentica** projetada para operar como um cérebro otimizado — capaz de:

1. **Treinamento tradicional** (como modelos atuais): pre-training supervisionado, fine-tuning, RLHF
2. **Aprendizado contínuo** (a partir do uso): adaptação online sem esquecimento catastrófico
3. **Raciocínio agentico**: planejamento, decomposição de tarefas, uso de ferramentas, memória persistente
4. **Auto-melhoria governada**: evolução autônoma dentro de limites de segurança (R1-R5)

O objetivo é criar uma AGI otimizada que sirva como **cérebro pessoal** — um sistema que aprende com você, adapta-se a você, e opera com você (não por você).

---

## 2. Deepsearch: Estado da Arte (2025-2026)

### 2.1 Liquid Foundation Models (Liquid AI, 2026)

A Liquid AI lançou a família **LFM2.5** (Liquid Foundation Model):

| Modelo | Parâmetros | Uso principal |
|--------|-----------|---------------|
| LFM2.5-1.2B-Thinking | 1.2B | Raciocínio agentico, edge devices |
| LFM2.5-2.6B | 2.6B | Agent workflows, RAG, long-context |
| LFM2.5-24B | 24B | Frontier-level reasoning |

**Arquitetura**: Linear Input-Varying Systems (LIVs) — evolução das redes líquidas (LTC/CfC) para linguagem. Combina:
- Convolutions gated para processamento local
- Grouped query attention para contexto longo
- Mecanismos líquidos para adaptação temporal

**Características-chave**:
- Baixo footprint de memória (roda em mobile)
- Inferência rápida (2x faster que transformers em CPU)
- Raciocínio nativo (thinking tokens)

**Relevância para VISÃO**: A arquitetura LFM prova que redes líquidas podem escalar para linguagem e raciocínio. VISÃO deve adotar princípios similares mas com foco em **aprendizado contínuo** (que LFMs não possuem).

### 2.2 Agentic Neural Networks (𝒜𝒩𝒩) — arXiv 2506.09046 (2025)

**Conceptualização**: Multi-agent systems como redes neurais em camadas.

**Duas fases**:
1. **Forward Phase**: Decomposição de tarefa em subtasks → equipes de agentes especializados por camada
2. **Backward Phase**: "Textual gradients" — feedback backpropagado para refinar prompts, papéis e coordenação

**Resultados**: Supera baselines em MATH, DABench, Creative Writing, HumanEval.

**Auto-evolução**: Agentes criam novas equipes post-training para tarefas nunca vistas.

**Relevância para VISÃO**: O mecanismo de "textual backprop" pode ser adaptado para o VISÃO como **gradiente simbólico** — uma camada neuro-simbólica que ajusta o cérebro líquido baseado em feedback textual/estruturado.

### 2.3 Neuro-Symbolic AI (2025-2026)

**Definição**: Híbrido neural (percepção, pattern recognition) + symbolic (raciocínio, lógica, regras).

**Sistemas de referência**:
- **AlphaGeometry/AlphaProof** (DeepMind): Neural sugere passos → verificador simbólico valida
- **Tool-Use LLMs**: LLM decide → calculadora/executor computa
- **Scallop** (UPenn): Diferenciável — rede neural alimenta fatos para programa lógico, recebe gradientes do raciocínio

**Relevância para VISÃO**: VISÃO precisa de uma **camada simbólica** para raciocínio explícito, planejamento e verificação. A parte neural (líquida) lida com percepção, adaptação e memória implícita.

### 2.4 Cognitive Architectures for AGI (ICML 2026)

**Posições do ICML 2026**:

1. **Hippocampal Explicit Memory**: Funções cognitivas superiores (planejamento estratégico, metacognição, raciocínio simbólico) dependem de memória explícita hipocampal — não surgem apenas de aprendizado estatístico implícito.

2. **Modular Memory for Continual Learning**: Arquiteturas modulares com memória separada permitem:
   - ICL (In-Context Learning): adaptação rápida
   - IWL (In-Weight Learning): atualizações estáveis
   - Mitigação de catastrophic forgetting

**Relevância para VISÃO**: VISÃO precisa de **memória explícita** (episódica, semântica, procedural) além do aprendizado implícito dos pesos.

### 2.5 From AGI to ASI (DeepMind, arXiv 2606.12683, 2026)

**Quatro caminhos para ASI**:
1. **Scaling AGI**: Mais compute, dados, parâmetros
2. **Paradigm shifts**: Novas arquiteturas (ex: liquid networks)
3. **Recursive improvement**: IA automatizando P&D de IA
4. **Multi-agent collectives**: Grupos de IAs alcançando mais que indivíduos

**Fricções**: Tempo de treinamento, energia, diminishing returns, automação parcial.

**Relevância para VISÃO**: O caminho #3 (recursive improvement) é o mais alinhado com a filosofia VISÃO — o sistema melhora a si mesmo continuamente a partir do uso.

---

## 3. Arquitetura Proposta: VISÃO Cognitive Brain

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISÃO COGNITIVE BRAIN                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Perception  │  │   Memory    │  │  Reasoning  │            │
│  │  (LTC/CfC)  │  │  (Hippo.)   │  │ (Symbolic)  │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         │                │                │                    │
│         └────────────────┼────────────────┘                    │
│                          │                                     │
│              ┌───────────┴───────────┐                        │
│              │   Liquid Core (LTC)   │                        │
│              │  Continuous-time ODE  │                        │
│              │  Oja + EWC + Surprise │                        │
│              └───────────┬───────────┘                        │
│                          │                                     │
│         ┌────────────────┼────────────────┐                   │
│         │                │                │                    │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐            │
│  │  Planning   │  │  Execution  │  │  Monitoring │            │
│  │  (Symbolic) │  │  (Agentic)  │  │  (Health)   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Governance Layer (R1-R5)                    │  │
│  │  Quórum k-de-n │ TTL │ Fronteira imutável │ Freio PoW  │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Componentes

#### 3.1.1 Liquid Core (LTC/CfC)
- **Função**: Adaptação contínua, percepção temporal, aprendizado online
- **Mecanismos**: Oja (auto-organização), EWC-temporal (consolidação), Surprise (detecção de mudança)
- **Estado atual**: ✅ Implementado (VisaoBrain, benchmark validado)

#### 3.1.2 Memory System (Hippocampal)
- **Memória Episódica**: Registro de experiências (eventos, interações)
- **Memória Semântica**: Conhecimento factual (grafo de conhecimento)
- **Memória Procedural**: Habilidades aprendidas (políticas de ação)
- **Estado atual**: ❌ Não implementado

#### 3.1.3 Reasoning Layer (Neuro-Symbolic)
- **Neural**: Pattern recognition, intuição, generalização
- **Symbolic**: Lógica, regras, verificação formal, planejamento
- **Interface**: "Textual gradients" (𝒜𝒩𝒩) — feedback estruturado ajusta o liquid core
- **Estado atual**: ❌ Não implementado

#### 3.1.4 Agentic Module
- **Planejamento**: Decomposição de tarefas (forward phase)
- **Execução**: Ação no mundo (tool use, APIs, código)
- **Monitoramento**: Auto-avaliação, detecção de erros, correção
- **Estado atual**: 🟡 Parcial (live_loop.py, task_free.py)

#### 3.1.5 Governance Layer (R1-R5)
- **R1**: Quórum k-de-n para decisões críticas
- **R2**: TTL (time-to-live) para ações irreversíveis
- **R3**: Fronteira imutável selada por manifesto
- **R4**: Freio PoW (proof-of-work) para auto-melhoria
- **R5**: Ledger encadeado de decisões
- **Estado atual**: ✅ Especificado (Fase 4)

---

## 4. Plano de Treinamento

### 4.1 Visão Geral

O treinamento do VISÃO é **dual** — combina pre-training tradicional com aprendizado contínuo:

```
┌──────────────────────────────────────────────────────────────┐
│                    TREINAMENTO VISÃO                         │
├──────────────────────────────────────────────────────────────┤
│  Fase 1: Pre-training (tradicional)                          │
│  ├── Dataset: texto + código + matemática + raciocínio       │
│  ├── Objetivo: linguagem, conhecimento geral,推理            │
│  └── Duração: semanas a meses (depende de compute)           │
│                                                              │
│  Fase 2: Fine-tuning (alinhamento)                           │
│  ├── RLHF / DPO / ORPO                                       │
│  ├── Objetivo: segurança, helpfulness, estilo                │
│  └── Duração: dias a semanas                                 │
│                                                              │
│  Fase 3: Continuous Learning (uso contínuo)                  │
│  ├── Online learning com Oja + EWC + Surprise                │
│  ├── Objetivo: adaptação pessoal, zero forgetting            │
│  └── Duração: contínua (24/7)                                │
│                                                              │
│  Fase 4: Recursive Self-Improvement (governada)              │
│  ├── Auto-melhoria dentro de limites R1-R5                   │
│  ├── Objetivo: otimização contínua de capacidades            │
│  └── Duração: contínua (com freio PoW)                       │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Fase 1: Pre-training

#### 4.2.1 Arquitetura do Modelo

**Opção A: Liquid Transformer Híbrido**
- Backbone: LTC/CfC layers (adaptação temporal) + Attention blocks (contexto longo)
- Tamanho: 1B-7B parâmetros (depende de compute)
- Contexto: 32K-128K tokens
- Treinamento: Next-token prediction + causal LM

**Opção B: Liquid State Space (Mamba-like)**
- Backbone: LTC/CfC + Selective State Spaces
- Vantagem: Inferência O(1) por passo (vs O(n) do attention)
- Treinamento: Next-token prediction

**Recomendação**: Opção A (híbrido) para compatibilidade com ecossistema atual, migrar para B quando estável.

#### 4.2.2 Dataset

| Categoria | Fonte | Proporção |
|-----------|-------|-----------|
| Texto geral | Common Crawl, C4 | 40% |
| Código | GitHub, The Stack | 20% |
| Matemática | ProofWiki, arXiv math | 15% |
| Raciocínio | Chain-of-Thought datasets | 15% |
| Científico | arXiv, PubMed | 10% |

**Tamanho**: 1-5 trillion tokens (depende de compute).

#### 4.2.3 Treinamento

**Infraestrutura**:
- Cloud: Lambda Labs, RunPod, ou Colab Pro+
- Alternativa: Colab CLI (já validado no projeto)
- Custo estimado: $5K-$50K para 1B params

**Hiperparâmetros**:
- Batch size: 1M tokens
- Learning rate: 3e-4 com cosine decay
- Warmup: 1% do total
- Precision: BF16 mixed
- Gradient clipping: 1.0

**Duração**: 2-8 semanas (depende de compute).

### 4.3 Fase 2: Fine-tuning (Alinhamento)

#### 4.3.1 RLHF (Reinforcement Learning from Human Feedback)

**Dataset**: Pares (preferred, rejected) de respostas.
- Fonte: OpenAssistant, Anthropic HH, ou curado manualmente
- Tamanho: 10K-100K pares

**Reward Model**: Treinado para prever preferência humana.
**Policy**: PPO para otimizar o modelo líquido.

#### 4.3.2 DPO (Direct Preference Optimization)

Alternativa mais simples que RLHF — treina diretamente no dataset de preferências sem reward model separado.

**Recomendação**: DPO para iterações rápidas, RLHF para alinhamento final.

### 4.4 Fase 3: Continuous Learning (Uso Contínuo)

#### 4.4.1 Mecanismos (já validados)

- **Oja**: Auto-organização do recorrente
- **EWC-temporal**: Consolidação por importância com decaimento
- **Surprise**: Detecção de mudança de regime (CUSUM)
- **Meta-plasticity**: Taxa de aprendizado adaptativa por neurônio

#### 4.4.2 Interface de Entrada

O sistema recebe:
1. **Texto natural** (conversa, documentos)
2. **Feedback estruturado** (correções, avaliações, recompensas)
3. **Ações e consequências** (resultados de tool use, execução de código)

#### 4.4.3 Política de Aprendizado

- **Sempre ativo**: Cada interação atualiza o liquid core (com EWC para proteger memórias)
- **Batch size**: 1 (online learning)
- **Learning rate**: Adaptativa (meta-plasticity)
- **Proteção**: EWC-temporal impede esquecimento de conhecimento antigo

### 4.5 Fase 4: Recursive Self-Improvement

#### 4.5.1 Mecanismo

O sistema pode:
1. **Gerar hipóteses** de melhoria (novas arquiteturas, hiperparâmetros, dados)
2. **Testar hipóteses** em ambiente sandbox
3. **Validar resultados** contra métricas objetivas
4. **Aplicar melhorias** se aprovadas pelo governance (R1-R5)

#### 4.5.2 Limites (Governance)

- **R1 (Quórum)**: Mudanças críticas requerem k-de-n aprovações
- **R2 (TTL)**: Ações irreversíveis têm timeout
- **R3 (Fronteira)**: Manifesto é imutável
- **R4 (Freio PoW)**: Auto-melhoria requer "proof-of-work" (custo computacional)
- **R5 (Ledger)**: Todas as mudanças são registradas

---

## 5. Próximos Passos Imediatos

### 5.1 Implementar Memory System (Tarefa 13.0)

**Arquivo**: `visao/memory/hippocampus.py`

Componentes:
- **EpisodicBuffer**: Ring buffer de experiências com consolidação
- **SemanticGraph**: Grafo de conhecimento (entidades + relações)
- **ProceduralMemory**: Políticas de ação (skills aprendidas)

### 5.2 Implementar Reasoning Layer (Tarefa 14.0)

**Arquivo**: `visao/reasoning/neuro_symbolic.py`

Componentes:
- **SymbolicSolver**: Planejador lógico (PDDL-like)
- **TextualGradient**: Módulo de feedback estruturado
- **VerificationEngine**: Verificador de provas/códigos

### 5.3 Implementar Agentic Module (Tarefa 15.0)

**Arquivo**: `visao/agent/planner.py`

Componentes:
- **TaskDecomposer**: Divide tarefas complexas em subtasks
- **ToolExecutor**: Executa ferramentas (cálculo, busca, código)
- **SelfMonitor**: Detecta erros e aciona correção

### 5.4 Atualizar BACKLOG

Adicionar tarefas 13.0, 14.0, 15.0 ao BACKLOG.json.

---

## 6. Métricas de Sucesso

| Métrica | Baseline | Meta VISÃO |
|---------|----------|------------|
| Forgetting | >50% (LSTM) | <5% |
| Accuracy (MSE) | 0.38 (atual) | <0.10 |
| Inference speed | 0.1s (atual) | <0.05s |
| Parâmetros | 4288 (atual) | 1B-7B |
| Agentic tasks | 0% | >80% |
| Memory retention | 0% | >90% (30 dias) |

---

## 7. Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|-------|---------------|-----------|
| Catastrophic forgetting | Média | EWC-temporal + memória explícita |
| Instabilidade do liquid core | Média | Gradient clipping + meta-plasticity |
| Alignment failure | Baixa | Governance R1-R5 + RLHF |
| Compute insuficiente | Alta | Colab CLI + cloud spot instances |
| Overfitting ao usuário | Média | Regularização + diversidade de dados |

---

## 8. Conclusão

VISÃO v3.0 evolui de "rede neural líquida com aprendizado contínuo" para **arquitetura cognitiva agentica** — um sistema que combina:

- **Líquido**: Adaptação contínua, percepção temporal
- **Simbólico**: Raciocínio explícito, planejamento, verificação
- **Agentic**: Ação no mundo, tool use, auto-melhoria
- **Governado**: Segurança por design (R1-R5)

O treinamento dual (pre-training + continuous learning) permite que VISÃO seja **treinado como modelos atuais** mas também **aprenda continuamente do uso** — exatamente como o Juan imaginou.

---

*Documento criado: 2026-09-10*
*Autor: ÍRIS (assistente do Juan)*
*Baseado em deepsearch de 20+ fontes (2025-2026)*
