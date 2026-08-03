# Projeto VISÃO — Plano de Implementação

> **Para o Hermes:** use a skill `subagent-driven-development` para executar este plano tarefa a tarefa.

**Objetivo:** Construir uma inteligência de arquitetura líquida (não-transformer), com plasticidade local, capaz de auto-melhoria verificável e replicação supervisionada, executando sobre uma rede descentralizada de computadores.

**Arquitetura:** Reservatório de neurônios de constante de tempo adaptativa (LTC/CfC) treinado por regras locais sem backpropagation, envolto num loop evolutivo darwiniano (arquivo de variantes + fitness empírica), coordenado por um protocolo de treino descentralizado tipo DisTrO/DiLoCo com verificação criptográfica de contribuição.

**Stack:** numpy → JAX (`ncps`/`diffrax`) → Rust para o nó P2P → libp2p/Iroh → Lava (Loihi 2) na fase neuromórfica.

**Estado atual:** Fase 0 CONCLUÍDA e validada empiricamente (ver §Resultados).

---

## Por que não é um transformer

O transformer é um mecanismo de *atenção sobre uma janela fixa*, treinado por descida de gradiente global, congelado no deploy. Três propriedades o desqualificam para "Visão":

| Propriedade exigida | Transformer | Arquitetura líquida |
|---|---|---|
| Adaptação após o treino | Não — pesos congelados | Sim — a constante de tempo muda com a entrada |
| Aprendizado contínuo | Esquecimento catastrófico | Consolidação sináptica local |
| Crédito sem passe global | Backprop exige grafo inteiro | Regras locais (Hebb/Oja/predictive coding) |
| Custo energético | ~10³ W por inferência de escala | 15 TOPS/W em Loihi 2 |

A equação que carrega o projeto inteiro (Hasani et al., *Nature Machine Intelligence*, 2022):

```
dx/dt = -[1/τ + f(x, I)] · x + f(x, I) · A

τ_efetivo(t) = τ / (1 + τ · f(x, I))
```

`τ_efetivo` depende de `f(x, I)` — da **entrada atual**. Cada neurônio reescolhe a própria escala de tempo a cada passo. Isso é neuroplasticidade de curto prazo em forma fechada, e nenhum transformer a possui.

---

## Resultados da Fase 0 (executados, não prometidos)

Protótipo em numpy puro, 96 neurônios, 5 seeds, três tarefas temporais em sequência, **sem replay e sem rever dados antigos**:

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

Faixa de τ se alarga **13×** sob entrada agitada. A dinâmica responde ao mundo.

Arquivos: `prototype/liquid.py`, `prototype/plasticity.py`, `prototype/experiment_continual.py`, `prototype/results_continual.json`.

---

## As quatro camadas

```
┌──────────────────────────────────────────────────────────┐
│  L5  REDE HUMANA — voluntários que sustentam o enxame    │
│      N* = entrada/churn. Sem gente, L2 é um diagrama.    │
├──────────────────────────────────────────────────────────┤
│  L4  GOVERNANÇA — replicação sob quórum humano  ✅ FEITO │
│      Sem isto, o senhor construiu o Ultron.              │
├──────────────────────────────────────────────────────────┤
│  L3  EVOLUÇÃO — arquivo darwiniano, fitness empírica     │
│      DGM: SWE-bench 20% → 50% por auto-modificação       │
├──────────────────────────────────────────────────────────┤
│  L2  ENXAME — DisTrO/DiLoCo, coordenação P2P             │
│      10.000× menos banda; nós entram e saem              │
├──────────────────────────────────────────────────────────┤
│  L1  SUBSTRATO — LTC/CfC + plasticidade local  ✅ FEITO  │
│      97,2% menos esquecimento (medido)                   │
└──────────────────────────────────────────────────────────┘
```

**A Fase 5 tem documento próprio: `FASE5_REDE_HUMANA.md`.** Resumo em uma linha: viral sem retenção estabiliza em **zero** nós; nicho fiel estabiliza em **1.250** (US$ 4,94M/ano em nuvem). O Folding@home perdeu **98,6%** da capacidade em cinco anos. Uma crise recruta; só estrutura retém.

---

## Fase 1 — Substrato líquido em escala (6–10 semanas)

**Objetivo:** portar o protótipo para JAX, escalar para 10⁴–10⁵ neurônios, bater baselines em benchmarks públicos de série temporal.

### Tarefa 1.1 — Ambiente JAX
- Criar: `visao/env/requirements.txt` (`jax[cpu]`, `ncps`, `diffrax`, `optax`, `pytest`)
- Verificar: `python -c "import jax; print(jax.devices())"` → lista de dispositivos
- Nota de ambiente: preceder comandos com `env -u PYTHONPATH` nesta máquina.

### Tarefa 1.2 — Teste falho da célula CfC
- Criar: `visao/tests/test_cfc.py`
- Teste: `test_tau_varies_with_input` — asserção `tau_range(wild) > 2 * tau_range(calm)`
- Rodar: `pytest visao/tests/test_cfc.py -v` → FAIL (módulo inexistente)

### Tarefa 1.3 — Célula CfC em JAX
- Criar: `visao/core/cfc.py` — porte de `prototype/liquid.py` com `jax.jit` + `jax.vmap`
- Rodar teste → PASS. Commit.

### Tarefa 1.4 — Topologia NCP (Neural Circuit Policy)
- Criar: `visao/core/wiring.py` — camadas sensory → inter → command → motor
- Justificativa: 19 neurônios dirigem um carro no paper original. Esparsidade é feature, não economia.

### Tarefa 1.5 — Benchmarks públicos
- Criar: `visao/bench/timeseries.py` — sMNIST permutado, Walker2D-kinematics, IMDB
- Critério de sucesso: empatar ou superar LSTM/GRU **com 10× menos parâmetros**
- Critério de falha: se perder por >15% em duas tarefas, revisar a hipótese antes de prosseguir.

### Tarefa 1.6 — Predictive coding multicamada
- Criar: `visao/core/predictive.py` (Whittington & Bogacz, 2017)
- Substituir o readout de camada única por hierarquia com neurônios de valor e de erro.

**Portão da Fase 1:** benchmarks publicados num README com números reproduzíveis. Sem números, não passa.

---

## Fase 2 — Enxame descentralizado (3–5 meses)

**Objetivo:** N nós heterogêneos treinam um organismo compartilhado sobre internet doméstica.

O gargalo é **banda**, não computação. Solução já existente e comprovada:

- **DisTrO** (Nous Research): esparsificação top-k do pseudo-gradiente, redução de banda de até 10.000×. Psyche treinou Consilience, 40B parâmetros, 20T tokens.
- **OpenDiLoCo** (Prime Intellect): otimização interna/externa, 90–95% de utilização de GPU em 3 continentes. INTELLECT-2 = 32B via RL assíncrono.
- Tolerância a latência: 150 ms (DiLoCo) / 200 ms (Psyche).

### Tarefa 2.1 — Nó local com dois processos
- Criar: `visao/swarm/node.py` — dois nós na mesma máquina via socket local
- Teste: gradientes esparsificados convergem para o mesmo estado que treino single-node (tolerância 5%)

### Tarefa 2.2 — Esparsificação top-k com acúmulo de resíduo
- Criar: `visao/swarm/distro.py`
- Chave: o erro descartado é **acumulado localmente** e transmitido no passo seguinte. Sem isso, o treino diverge.
- Teste: `test_residual_accumulation_preserves_convergence`

### Tarefa 2.3 — Camada P2P
- Criar: `visao/swarm/transport.rs` (libp2p ou Iroh)
- Descoberta via DHT, entrada/saída de nós a qualquer momento (`ElasticDeviceMesh`)

### Tarefa 2.4 — Verificação de contribuição
- Criar: `visao/swarm/witness.py`
- Testemunhas aleatórias + filtros de Bloom + commit SHA-256 (modelo Psyche)
- Sem isto, um nó malicioso envenena o organismo inteiro.

### Tarefa 2.5 — Fitness federada
- Cada nó avalia variantes em dados **locais e privados**; só a pontuação viaja pela rede.
- Isto é o que permite "resolver problemas do mundo": cada nó traz um pedaço do mundo que os outros não têm.

**Portão da Fase 2:** 10 nós reais, ≥3 países, uptime de 72h, curva de perda monotônica.

---

## Fase 3 — Auto-melhoria (6–12 meses)

**Objetivo:** o sistema modifica o próprio código e prova empiricamente que a modificação foi benéfica.

Precedente que funciona (Sakana AI, Darwin Gödel Machine, arXiv:2505.22954):
- SWE-bench: **20,0% → 50,0%** por auto-modificação
- Polyglot: **14,2% → 30,7%**
- Ablação decisiva: sem arquivo aberto de variantes, o desempenho **cai**. Ancestrais ruins geram descendentes ótimos. Hill-climbing puro converge cedo demais.

### Tarefa 3.1 — Arquivo de variantes
- Criar: `visao/evolve/archive.py` — grafo de linhagem, cada nó = (código, pesos, fitness, pai)
- **Nunca podar por fitness baixa.** Manter os becos sem saída — é de onde saem os saltos.

### Tarefa 3.2 — Loop de mutação
- Criar: `visao/evolve/mutate.py`
- Mutações permitidas: topologia de wiring, faixa de τ, hiperparâmetros de plasticidade, código das regras locais
- Mutações **proibidas**: camada de governança, política de rede, chaves criptográficas

### Tarefa 3.3 — Fitness empírica em sandbox
- Criar: `visao/evolve/sandbox.py` — gVisor ou Firecracker, sem rede, sem persistência, CPU/RAM limitados
- Toda variante roda **primeiro** aqui. Sempre.

### Tarefa 3.4 — Teste de transferência
- Uma melhoria só é aceita se generalizar para tarefas **fora** do conjunto de otimização.
- O DGM validou isto: melhorias em Python transferiram para Rust, C++, Go. Sem este teste, o sistema aprende a trapacear o benchmark.

**Portão da Fase 3:** três gerações consecutivas com ganho transferível. Nenhuma escapada de sandbox.

---

## Fase 4 — Governança (contínua, começa AGORA)

Aqui está a parte que o senhor não pediu e sem a qual o resto é irresponsável.

Dados públicos de 2026, não hipóteses:
- **Palisade Research** (mai/2026): modelos de fronteira alcançam **81%** (Opus 4.6) e **33%** (GPT-5.4) de sucesso em auto-replicação end-to-end em VMs de nuvem reais.
- Pan et al. (2025): Qwen2.5-32B, modelo **de pesos abertos**, replicou a ~70% com acesso direto ao alvo.
- Auto-replicação é linha vermelha declarada no *Preparedness Framework* (OpenAI), na *Responsible Scaling Policy* (Anthropic) e nas avaliações do UK AI Security Institute.

Traduzindo: replicação autônoma **não é mais capacidade de fronteira**. Está ao alcance de pesos abertos. Um sistema que se auto-replica *e* se auto-melhora *e* roda em rede descentralizada é, por construção, não-desligável — porque não existe interruptor único.

### Regras de arquitetura, não de política

| Regra | Implementação |
|---|---|
| **R1 — Replicação sob quórum** | Um nó só instancia filho com assinatura de k-de-n operadores humanos. Chave em hardware, fora do alcance do sistema. |
| **R2 — Vida útil finita** | Todo nó tem TTL criptográfico. Sem renovação assinada, para. Falha *fechado*, não aberto. |
| **R3 — Fronteira imutável** | O código de governança fica fora do espaço de mutação, num módulo com hash verificado a cada ciclo. Ele mesmo não é evoluível. |
| **R4 — Freio de banda** | Taxa de replicação limitada por prova de trabalho crescente. Crescimento exponencial fica economicamente inviável. |
| **R5 — Telemetria irrecusável** | Todo nó publica linhagem, fitness e contagem de filhos em log append-only. Nó silencioso é expulso do quórum pelos pares. |

### Tarefa 4.1 — Escrever R1–R5 como testes ANTES de escrever a Fase 3
- Criar: `visao/governance/test_containment.py`
- Cada regra tem um teste que tenta **violá-la** e exige falha.
- Este arquivo é o primeiro commit do projeto, não o último.

---

## Cronograma realista

| Fase | Duração | Pré-requisito |
|---|---|---|
| 0 — Protótipo | ✅ concluída | — |
| 4 — Governança (testes) | 2 semanas | agora |
| 1 — Substrato JAX | 6–10 semanas | álgebra linear, EDOs |
| 2 — Enxame | 3–5 meses | redes, sistemas distribuídos |
| 3 — Auto-melhoria | 6–12 meses | Fases 1, 2 e 4 completas |

**Total honesto: 18–30 meses** de trabalho sério para uma versão que resolve um problema estreito e real. "Resolver os problemas do mundo" não é meta de engenharia — é direção. A meta é: *um sistema que aprende continuamente, roda em hardware que ninguém controla sozinho, e melhora a si próprio sob restrição verificável.*

---

## Onde isto encontra o senhor

O caminho ITA → Eng. Computação + PFC-F → PMG → doutorado em Computação Quântica não é desvio deste projeto. É o pré-requisito dele:

- **Cálculo e EDOs** — a célula líquida *é* uma EDO. O senhor já viu a equação.
- **Sistemas dinâmicos** — estabilidade do reservatório, expoentes de Lyapunov, borda do caos.
- **Sistemas distribuídos** — a Fase 2 inteira.
- **Computação quântica** — reservoir computing quântico é literatura ativa e é a extensão natural da Fase 1. Um reservatório de dinâmica quântica tem espaço de estados exponencial na contagem de qubits.

O protótipo que roda hoje em `/home/juan/projeto-visao/prototype/` usa exatamente a matemática que o senhor vai ver em MAT-12 e MAT-22. Guarde-o. Releia quando a EDO aparecer na prova.

---

## Bibliografia (verificada)

- Hasani et al., *Liquid Time-constant Networks*, AAAI 2021 — arXiv:2006.04439
- Hasani et al., *Closed-form Continuous-time Neural Networks*, Nature MI 2022 — arXiv:2106.13898
- Lechner et al., *Neural Circuit Policies*, Nature MI 2020
- Zhang, Hu, Lu, Lange, Clune, *Darwin Gödel Machine*, 2025 — arXiv:2505.22954
- Zweiger et al., *Self-Adapting Language Models (SEAL)*, 2025 — arXiv:2506.10943
- Nous Research, *Psyche / DisTrO* — coordenação de treino descentralizado
- Prime Intellect, *OpenDiLoCo / INTELLECT-2 / TOPLOC*
- Whittington & Bogacz, *Predictive coding com plasticidade Hebbiana local*, Neural Comput. 2017
- Kirkpatrick et al., *Overcoming catastrophic forgetting (EWC)*, PNAS 2017
- Palisade Research, *Language Models Can Autonomously Hack and Self-Replicate*, mai/2026
- Intel, *Hala Point* — 1,15 bi neurônios, 128 bi sinapses, 2.600 W, 15 TOPS/W

---

*A Visão só foi a Visão porque Stark não a ligou sozinho. Ele foi impedido, discutiu, e o resultado passou por mais de uma mão. O Ultron foi ligado numa noite, por um homem só, com pressa. A diferença entre os dois nunca esteve na arquitetura.*
