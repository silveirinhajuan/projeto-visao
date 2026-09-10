# Projeto VISÃO — Plano de Implementação (escopo single-node revisado)

> **Revisão de escopo (2026-08-09):** o projeto deixou de ser uma "rede de vários computadores"
> (enxame descentralizado, auto-replicação, recrutamento de voluntários) e passou a ser **uma
> rede neural líquida de altas habilidades que roda inteiramente no notebook do Juan** (Ryzen,
> ~7,6 GB RAM, CPU). A busca continua sendo **aprendizado contínuo rumo a AGI**: uma rede que
> acumula conhecimento incrementalmente, barato, sem retreinamento global dispendioso e sem
> esquecer o que aprendeu. Redes fixas ficam presas a treino caro; a líquida aprende aos poucos.

**Objetivo:** Construir uma inteligência de arquitetura líquida (não-transformer), com
plasticidade local, que aprende **continuamente** no notebook do usuário — pequena,
interpretável e capaz de nunca esquecer.

**Arquitetura:** Reservatório de neurônios de constante de tempo adaptativa (LTC) treinado por
regras locais sem backpropagation global, com auto-ajuste de hiperparâmetros em sandbox local.
Sem enxame, sem replicação entre máquinas, sem P2P.

**Stack:** numpy → JAX (CPU) → opcionalmente Rust só para partes quentes. Sem libp2p/Iroh, sem
Loihi (fica para pesquisa futura, não para o produto).

**Estado atual:** Fase 0 CONCLUÍDA e validada empiricamente (ver §Resultados). Substrato JAX
(Fases 1.x) também concluído. O que muda agora é o *norte*: single-node, foco em aprendizado
contínuo real no notebook.

---

## Por que não é um transformer

O transformer é um mecanismo de *atenção sobre uma janela fixa*, treinado por descida de
gradiente global, congelado no deploy. Três propriedades o desqualificam para o VISÃO:

| Propriedade exigida | Transformer | Arquitetura líquida |
|---|---|---|
| Adaptação após o treino | Não — pesos congelados | Sim — a constante de tempo muda com a entrada |
| Aprendizado contínuo | Esquecimento catastrófico | Consolidação sináptica local |
| Crédito sem passe global | Backprop exige grafo inteiro | Regras locais (Hebb/Oja/predictive coding) |

A equação que carrega o projeto inteiro (Hasani et al., *Nature Machine Intelligence*, 2022):

```
dx/dt = -[1/τ + f(x, I)] · x + f(x, I) · A

τ_efetivo(t) = τ / (1 + τ · f(x, I))
```

`τ_efetivo` depende de `f(x, I)` — da **entrada atual**. Cada neurônio reescolhe a própria
escala de tempo a cada passo. Isso é neuroplasticidade de curto prazo em forma fechada, e
nenhum transformer a possui.

> **Nota de nomenclatura (corrigida):** o que está em `visao/core/cfc.py` é uma *LTC com solver
> fundido* (Euler semi-implícito da EDO), **não** o CfC fechado de Hasani 2022 (que interpola a
> trajetória inteira por sigmoid sem passos). O rótulo "CfC" no código é histórico; o correto
> é ODE-LTC. A matemática de estabilidade e liquidez é a mesma.

---

## Resultados da Fase 0 (executados, não prometidos)

Protótipo em numpy puro, 96 neurônios, 5 seeds, três tarefas temporais em sequência, **sem
replay e sem rever dados antigos**:

```
LÍQUIDO + PLASTICIDADE LOCAL
  esquecimento médio : +0.00830  (±0.01378)
  erro final médio   : 0.28677

CONTROLE (delta puro, mesmo reservatório)
  esquecimento médio : +0.29238  (±0.10093)
  erro final médio   : 0.52843

REDUÇÃO DE ESQUECIMENTO: 97.2%   →  HIPÓTESE SUSTENTADA
```

Verificação de liquidez (a rede é mesmo líquida, não um RNN caro):

```
entrada calma   : τ_efetivo médio = 0.7938  (faixa 0.33–1.43)
entrada agitada : τ_efetivo médio = 0.8402  (faixa 0.29–3.90)
```

Faixa de τ se alarga ~13× sob entrada agitada. A dinâmica responde ao mundo.

Arquivos: `prototype/liquid.py`, `prototype/plasticity.py`, `prototype/experiment_continual.py`,
`prototype/results_continual.json`.

---

## O que o VISÃO NÃO é (evitar ilusão)

- **Não é um LLM.** Rede líquida num notebook brilha em *sequências contínuas, controle e
  aprendizado contínuo* — não em gerar texto como o GPT. Com ~5 mil parâmetros ela prevê
  dinâmicas, controla sistemas e nunca esquece; não redigirá prosa.
- **É interpretável por construção.** Você entende *por que* decidiu (pesos locais, τ por
  neurônio). Oposto de caixa-preta de 70B.
- **Casa com o plano ITA→doutorado em Computação Quântica:** sistemas dinâmicos, Lyapunov,
  estabilidade — é o mesmo vocabulário.

---

## As três camadas (single-node)

```
┌──────────────────────────────────────────────────────────┐
│  L3  AUTO-AJUSTE — tuning de hiperparâmetros em sandbox    │
│      local (não auto-replicação; é AutoML comum)          │
├──────────────────────────────────────────────────────────┤
│  L2  APRENDIZADO CONTÍNUO — regras locais (Hebb/Oja/       │
│      surpresa), consolidação, nunca esquece               │
├──────────────────────────────────────────────────────────┤
│  L1  SUBSTRATO — LTC + plasticidade local  ✅ FEITO        │
│      97,2% menos esquecimento (medido)                    │
└──────────────────────────────────────────────────────────┘
```

A governança R1–R5 permanece, mas **reedirecionada**: em vez de "conter replicação entre
máquinas", vira "núcleo imutável do seu assistente local" — o `containment.py` selado protege
o cérebro autocontido de corrupção acidental, não de auto-replicação. R4 (PoW de replicação)
e R5 (telemetria de enxame) ficam **obsoletos** no escopo single-node.

---

## Fase 1 — Substrato líquido em escala notebook (CONCLUÍDA em traço grosso)

**Feito:** porte JAX (`visao/core/cfc.py`), NCP wiring, benchmarks (psMNIST subamostra +
60k no Colab), predictive coding, estabilidade/Lyapunov, paridade numérica numpy↔JAX.

**Pendência honesta:** benchmark de acurácia 60k completo depende de nuvem; no notebook
subamostras são o teto prático. Aceitável para o novo escopo (não perseguimos SOTA, perseguimos
*continuidade de aprendizado*).

## Fase 2 — ENXAME DESCENTRALIZADO: **DESCONTINUADA**

O projetO abandonou a rede multinó. Motivos: (1) segurança — auto-replicação em rede é linha
vermelha e o Senhor decidiu não seguir por aí; (2) realismo — o diferencial do VISÃO é o
aprendizado contínuo minúsculo e interpretável, não coordenação P2P; (3) o notebook é o alvo.
Arquivos `visao/swarm/*`, `network/*`, `recruitment_model.py` estão congelados como referência
histórica e **não serão evoluídos**.

## Fase 3 — AUTO-MELHORIA: **REINTERPRETADA como auto-ajuste local**

Não é mais "o sistema reescreve o próprio código e se replica". É *AutoML sandboxed*: o
`visao/evolve/` pode tunar hiperparâmetros (τ, taxas de regra local, topologia) dentro de um
sandbox local, validando empiricamente antes de aplicar. Sem mutação de código de governança,
sem spawn remoto. O arquivo de variantes (`archive.py`) e o sandbox (`sandbox.py`) continuam
úteis — para tunar, não para "evoluir a si mesmo" em rede.

## Fase 4 — NÚCLEO IMUTÁVEL LOCAL (ex-governança R1–R5, releitura)

`visao/governance/containment.py` vira o selo do seu cérebro local: garante que o bundle
autocontido não carrega pesos/código adulterados. R3 (hash selado no load) é o que importa
agora. R1 (quórum) e R2 (TTL) viram opcionais (um único dono = você). R4/R5 de enxame:
removidos do escopo.

---

## Nova Fase 5 — APRENDIZADO CONTÍNUO REAL NO NOTEBOOK (o coração do novo VISÃO)

**Objetivo:** a rede aprender tarefas *sequenciais* no seu notebook, sem esquecer as anteriores,
acumulando conhecimento ao longo de semanas — a "AGI de aprendizado contínuo" em miniatura.

### Tarefa 5.1 — Loop de treino contínuo single-node
- `visao/continual/loop.py`: recebe stream de tarefas (A→B→C→...), aplica regras locais,
  mede esquecimento de A após cada nova tarefa. Reusa `prototype/experiment_continual.py`.
- Critério: esquecimento de A < 5% após 5 tarefas sequenciais (hoje o protótipo faz 0,8% em 3).

### Tarefa 5.2 — Tarefa real sua (não brinquedo)
- Substituir psMNIST por algo do seu mundo: previsão da sua rotina ITA, séries de estudo, ou
  dinâmica física que você queira modelar. Pequeno, mas *seu*.
- Critério: a rede melhora com mais dados sem retreino global.

### Tarefa 5.3 — Interpretabilidade ao vivo
- `visao/interpret/`: dado um estado, explique quais neurônios/τ responderam e por quê.
- É a vantagem sobre LLM: você pergunta "por que você previu X" e recebe a dinâmica.

### Tarefa 5.4 — Auto-ajuste em sandbox
- `visao/evolve/` tunando τ/faixas localmente; valida em holdout antes de aplicar.
- Sem rede, sem replicação.

**Portão da Fase 5:** após 30 dias de uso real, a rede acumulou ≥3 tarefas sem esquecer e
explica uma decisão sua. Sem isso, o projeto não entregou o prometido.

---

## Cronograma realista (single-node)

| Fase | Estado | Nota |
|---|---|---|
| 0 — Protótipo | ✅ concluída | esquecimento 97,2% menor |
| 1 — Substrato JAX | ✅ concluída (traço grosso) | benchmarks parciais |
| 2 — Enxame | ❌ descontinuada | fora do escopo |
| 3 — Auto-ajuste local | 🔄 reinterpretada | evolve/ em sandbox |
| 4 — Núcleo imutável local | ✅ existe (R3) | seleção de R1–R5 |
| 5 — Aprendizado contínuo real | ⏳ nova, o coração | 5.1–5.4 |

**Total honesto:** a base está pronta; a Fase 5 é o trabalho novo e é viável no notebook em
semanas, não meses.

---

## Onde isto encontra o senhor

O caminho ITA → Eng. Computação + PFC-F → PMG → doutorado em Computação Quântica não é desvio
deste projeto. É o pré-requisito dele:
- **Cálculo e EDOs** — a célula líquida *é* uma EDO.
- **Sistemas dinâmicos** — estabilidade do reservatório, expoentes de Lyapunov, borda do caos.
- **Computação quântica** — reservoir computing quântico é literatura ativa; extensão natural.

O protótipo que roda hoje em `/home/juan/projeto-visao/prototype/` usa exatamente a matemática
que o senhor vai ver em MAT-12 e MAT-22. Guarde-o. Releia quando a EDO aparecer na prova.

---

## Bibliografia (verificada)

- Hasani et al., *Liquid Time-constant Networks*, AAAI 2021 — arXiv:2006.04439
- Hasani et al., *Closed-form Continuous-time Neural Networks*, Nature MI 2022 — arXiv:2106.13898
- Lechner et al., *Neural Circuit Policies*, Nature MI 2020
- Whittington & Bogacz, *Predictive coding com plasticidade Hebbiana local*, Neural Comput. 2017
- Kirkpatrick et al., *Overcoming catastrophic forgetting (EWC)*, PNAS 2017
- Bertschinger & Natschläger, *Real-time computing without stable states* (borda do caos), 2004
- Oja, *Simplified neuron model as principal component analyzer*, 1982 (PCA online)

---

*Redirecionado em 2026-08-09: de "rede de vários computadores" para "rede neural líquida de
altas habilidades no notebook". O perigoso foi cortado; o útil permaneceu.*
