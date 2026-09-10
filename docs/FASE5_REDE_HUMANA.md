# Fase 5 — A Rede Humana

> Camada de recrutamento e retenção do Projeto VISÃO.
> **Anexo ao PLANO.md.** Leia aquele primeiro.

**Objetivo:** construir e sustentar a população de voluntários que fornece o substrato computacional descentralizado — e fazê-lo sem mentir para essas pessoas.

---

## O número que governa esta fase inteira

```
Folding@home
  fev/2020 ····  30.000 voluntários
  mar/2020 ···· 400.000 voluntários  →  1.220 PFLOPS  (recorde mundial)
  2025 ········                          17 PFLOPS

  QUEDA DO PICO: 98,6%
```

O maior sucesso de recrutamento da história da computação voluntária perdeu 98,6% da própria capacidade em cinco anos. Não por fracasso técnico — o software continuou funcionando. Perdeu porque **uma crise recruta, mas só estrutura retém.**

Qualquer plano de persuasão que não comece por este número é publicidade.

---

## Aritmética antes de retórica

Modelo executável em `network/recruitment_model.py`, parâmetros ancorados em Anderson (arXiv:1903.01699) e nos dados públicos do Folding@home. Saída real:

| Cenário | Instalam | Churn/mês | Entrada/mês | **Equilíbrio** | TFLOPS | Equivalente em nuvem |
|---|---:|---:|---:|---:|---:|---:|
| **A** Viral sem retenção | 10.000 | 30% | 0 | **0** | 0 | — |
| **B** Nicho fiel + fluxo contínuo | 1.200 | 12% | 150 | **1.250** | 117,5 | US$ 4,94M/ano |
| **C** Âncoras institucionais | 500 | 3% | 20 | **666** | 62,6 | US$ 2,63M/ano |

O cenário A atrai **8× mais gente** que o B e entrega **nada**. A equação é elementar e implacável:

```
N* = entrada_mensal / churn_mensal
```

Sem entrada contínua, toda rede converge para zero. Churn é a física do problema. Recrutamento em massa sem retenção é encher balde furado com mais pressão.

**Decisão de projeto: perseguimos B+C, nunca A.** Mil nós fiéis valem mais que cem mil curiosos.

---

## Por que alguém doaria um computador

Anderson mediu: entre **5% e 10%** das pessoas que *entendem* computação voluntária participam. A conversão não é o gargalo — a **compreensão** é. Ninguém doa para o que não entende.

Motivações reais, em ordem de força observada:

1. **A ciência importa para mim** — motivação mais forte e mais frágil. Evapora quando o resultado some de vista.
2. **Reconhecimento** (credit) — BOINC descobriu que quase todo participante quer um número que represente sua contribuição. Não vale dinheiro; vale identidade.
3. **Competição por equipes** — o mecanismo viral que de fato funcionou: membros recrutam família, amigos, colegas para subir a equipe no ranking. Crescimento exponencial documentado.
4. **Aprendizado** — Kloetzer et al. (2016): participação em BOINC gera aprendizado real. Voluntários viram semi-especialistas.
5. **Pertencimento** — fórum, identidade de grupo, sensação de projeto compartilhado.

Note o que **não** está na lista: dinheiro. Projetos que pagam atraem mineradores, que otimizam para a recompensa e não para a ciência, e que somem quando o preço cai.

---

## A vantagem estrutural do VISÃO (e o seu risco)

Somos diferentes de um BOINC clássico em três pontos — dois a favor, um contra.

**A favor:**

- **Leveza.** A arquitetura líquida usa ordens de magnitude menos parâmetros que um transformer. Loihi 2 atinge 15 TOPS/W. Um nó VISÃO pode rodar num laptop modesto sem transformá-lo em secador de cabelo. Isso derruba a barreira de entrada que exclui 90% das pessoas do treino de IA.
- **O modelo volta para o voluntário.** Diferente do Folding@home — onde o resultado é um paper que o voluntário nunca lê — aqui cada nó recebe o organismo treinado. Quem doa computação **ganha a inteligência de volta**. Isso muda a proposta de "doação" para "cooperativa".

**Contra — e isto precisa estar em letras grandes:**

> Estamos pedindo às pessoas que rodem uma IA **auto-replicante e auto-modificável** nos computadores delas.

Palisade Research (mai/2026) mediu modelos de fronteira replicando-se autonomamente em VMs reais a **81%** de sucesso. Pesos abertos chegam a 70%. Qualquer pessoa tecnicamente informada vai fazer exatamente a pergunta certa: *"e se isso escapar?"*

**A resposta não pode ser uma promessa. Tem que ser um artefato.** É por isso que a Fase 4 foi construída antes desta, e é por isso que ela é o principal argumento de recrutamento — não um rodapé legal.

---

## A tese de persuasão

**Não vendemos esperança. Mostramos contenção auditável e devolvemos o resultado.**

Três pilares, nesta ordem:

### Pilar 1 — "Você pode auditar os freios em 20 minutos"

`visao/governance/containment.py` é um arquivo único, só stdlib, 39 testes que **tentam violar cada regra e exigem que ela falhe**. Não pedimos confiança; pedimos revisão.

| Regra | O que o voluntário ganha |
|---|---|
| R1 quórum k-de-n | Nenhum nó replica sem assinatura humana em hardware |
| R2 TTL | Sem renovação assinada, o nó **para sozinho**. Falha fechado. |
| R3 fronteira imutável | O código de contenção não é evoluível. Alterar = parada. |
| R4 freio PoW | Replicar 8× custa 10.000× mais que 1× |
| R5 ledger | Toda linhagem é pública e encadeada; nó silencioso é expulso |

Slogan honesto: **"Não confie em nós. Rode os testes."**

### Pilar 2 — "O que você ajudou a treinar volta para você"

O organismo treinado é redistribuído para todo nó participante. Um problema real resolvido — detecção de anomalia em ECG na borda, previsão de enchente em bacia local, controle de irrigação com sensor barato — vira uma ferramenta que o voluntário **usa**, não um agradecimento numa seção de acknowledgments.

### Pilar 3 — "Um problema concreto, escolhido pela rede"

Nada de "resolver os problemas do mundo". Isso não recruta ninguém sério.

O primeiro problema deve ser: **local, mensurável, com dados disponíveis, e que a arquitetura líquida resolva melhor que a alternativa**. Séries temporais em hardware fraco é exatamente o nicho onde LTC/CfC ganha.

Candidatos para Caucaia/Fortaleza — porque começar por casa é honesto e verificável:
- Previsão de cheia em bacia urbana com pluviômetros de baixo custo (série temporal, dados FUNCEME públicos)
- Detecção de anomalia em ECG na borda (literatura TinyML ativa em 2025-2026)
- Otimização de irrigação para agricultura familiar no semiárido

Regra: **um problema por vez, com métrica pública antes de começar.**

---

## Fases de crescimento

### F5.0 — Antes de recrutar qualquer pessoa (pré-requisito absoluto)

Não convide ninguém enquanto isto não existir:

- [x] Fase 4 completa e testada (39 testes verdes) ✅
- [ ] Fase 1 fechada com benchmarks públicos reproduzíveis
- [ ] Cliente que instala em <5 min em Windows/Mac/Linux
- [ ] Um problema real escolhido, com métrica pública definida
- [ ] Documento de risco honesto, incluindo o que pode dar errado

**Recrutar antes disso queima a única primeira impressão que existe.** Voluntário decepcionado não volta, e conta para os outros.

### F5.1 — Semente (10–30 nós, meses 1–3)

Pessoas que o senhor conhece pelo nome. Colegas do ITA, professores, o grupo de estudo. Alta tolerância a bug, alto valor de feedback.

Meta: **churn <10%/mês**. Se a semente evapora, o problema é o produto, não o marketing.

### F5.2 — Âncoras institucionais (cenário C, meses 3–9)

Laboratórios universitários. Churn de 3%/mês porque a máquina fica ligada de todo jeito e o interesse é institucional, não pessoal.

Alvos naturais: UFC, IFCE, UNIFOR (Fortaleza); grupos de computação quântica em USP-SC, UNICAMP, UFRJ, UFMG, UFPE. O ITA é a âncora óbvia quando o senhor entrar.

666 nós institucionais = 62,6 TFLOPS estáveis = US$ 2,63M/ano em nuvem. **Esta é a espinha dorsal.** Não é glamourosa; é o que não some.

### F5.3 — Nicho fiel (cenário B, meses 6–18)

Comunidades que já entendem o problema: r/LocalLLaMA, Nous Research, EleutherAI, entusiastas de neuromórfica, Open Neuromorphic.

Mecanismos de retenção — copiados do que comprovadamente funcionou no BOINC:
- **Credit** por contribuição verificada (Bloom filter + witness da Fase 2 já dá a verificação)
- **Equipes com leaderboard** — o motor viral documentado
- **Relatório mensal** mostrando o que a rede resolveu naquele mês
- **Aprendizado**: cada release explica *o que mudou na arquitetura e por quê*

### F5.4 — Aberto (mês 18+)

Só depois de B e C estáveis. E com um freio explícito: **crescer devagar de propósito**. Uma rede que dobra por mês não é auditável por ninguém.

---

## O que NÃO vamos fazer

Compromissos, não sugestões:

- **Nada de token/criptomoeda.** Atrai mineradores, não colaboradores. Corrompe a fitness da rede: o nó passa a otimizar a recompensa, não o problema. Gridcoin existe; observe-o e não repita.
- **Nada de "resolver os problemas do mundo" em material de divulgação.** É a frase que faz gente séria fechar a aba.
- **Nada de instalação silenciosa, bundle ou opt-out.** Consentimento explícito, desinstalação em um clique.
- **Nada de esconder o consumo.** Mostrar CPU, RAM e energia estimada em tempo real, sempre.
- **Nada de prometer que é seguro.** Prometemos que é **auditável**, que os freios são testados, e que os testes falham quando devem. Diferença enorme.
- **Nada de crescer durante uma crise humanitária para surfar comoção.** Foi o que inflou e depois esvaziou o Folding@home.

---

## Métricas — as únicas que contam

Vaidade vs. verdade:

| ❌ Métrica de vaidade | ✅ Métrica real |
|---|---|
| Total de downloads | Nós ativos nos últimos 7 dias |
| Voluntários cadastrados | **Churn mensal** |
| Pico de FLOPS | **TFLOPS em equilíbrio** (N* = entrada/churn) |
| Menções na imprensa | Problemas resolvidos com métrica pública |
| Estrelas no GitHub | Auditorias externas independentes da Fase 4 |

Dashboard obrigatório desde o primeiro nó. Se não medimos churn desde o dia 1, descobrimos o problema tarde demais — que é precisamente como se perde 98,6%.

---

## Riscos desta fase

| Risco | Mitigação |
|---|---|
| Ninguém confia numa IA auto-replicante | Fase 4 auditável + convite explícito a red-teams externos |
| Sucesso viral seguido de colapso (F@H) | Perseguir B+C, nunca A; medir churn desde o dia 1 |
| Nó malicioso envenena o organismo | Witness + Bloom filter + TOPLOC (Fase 2) |
| Voluntário some porque não vê resultado | Devolver o modelo treinado; relatório mensal |
| Projeto vira "cripto de IA" | Zero token. Compromisso escrito e público. |
| Recrutar antes de estar pronto | Checklist F5.0 é bloqueante, não sugestivo |

---

## Bibliografia desta fase

- Anderson, *BOINC: A Platform for Volunteer Computing* — arXiv:1903.01699
- Anderson & Kristensen, *An Incentive System for Volunteer Computing* — BOINC papers
- Kloetzer et al., *Engagement and learning in Volunteer Computing projects*, 2016
- Voelz et al., *Folding@home: Achievements from over 20 years of citizen science*, 2023
- Ars Technica (abr/2020), *How the pandemic turned Folding@Home into an exaFLOP machine*
- Palisade Research (mai/2026), *Language Models Can Autonomously Hack and Self-Replicate*

---

*O Folding@home provou que quatro milhões de pessoas doam computação quando acreditam que importa. E provou, cinco anos depois, que elas vão embora quando param de ver por quê. A primeira metade é inspiração. A segunda é o requisito de engenharia.*
