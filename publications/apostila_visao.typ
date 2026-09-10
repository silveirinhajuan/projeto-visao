#set document(
  title: "Apostila VISÃO — Redes Líquidas, Plasticidade Local e Contenção",
  author: "Projeto VISÃO · preparado por ÍRIS",
)
#set page(
  paper: "a4",
  margin: 1.7cm,
  numbering: "1 / 1",
  header: context {
    let p = counter(page).at(here()).first()
    if p > 1 [
      #set text(size: 8pt, fill: gray)
      #align(center)[Apostila VISÃO — Redes Líquidas, Plasticidade Local e Contenção]
    ]
  },
)
#set text(size: 10.5pt, lang: "pt", font: "New Computer Modern")
#set math.equation(numbering: "(1)")
#set par(justify: true)
#set heading(numbering: "1.")
#show heading: set block(above: 1.0em, below: 0.6em)

// ---- caixas temáticas ----
#let ita(title: "Para o ITA", body) = block(
  fill: rgb(222, 238, 255),
  inset: 9pt,
  radius: 4pt,
  [#text(weight: "bold", fill: rgb(20, 60, 120))[#title] #h(4pt) #body],
)
#let nota(body) = block(
  fill: rgb(235, 235, 235),
  inset: 9pt,
  radius: 4pt,
  [#text(weight: "bold")[Nota] #h(4pt) #body],
)
#let ex(body) = block(
  fill: rgb(255, 247, 222),
  inset: 9pt,
  radius: 4pt,
  [#text(weight: "bold", fill: rgb(150, 90, 0))[Exercício] #h(4pt) #body],
)

#align(center)[
  #text(weight: "bold", size: 20pt)[Apostila VISÃO]
  #v(2pt)
  #text(size: 13pt)[Redes Líquidas, Plasticidade Local e Contenção Criptográfica]
  #v(2pt)
  #text(size: 11pt)[Um guia de fundo para o Projeto VISÃO — da equação diferencial à governança]
  #v(4pt)
  #text(size: 10pt)[Preparado por ÍRIS · Agosto de 2026 · Projeto VISÃO / Juan Guerra]
]

#set par(justify: false)
*Como usar esta apostila.* Cada capítulo parte de um conceito, mostra a matemática que o sustenta, e termina com um bloco "Para o ITA" ligando o tópico à sua preparação para o vestibular. O Capítulo 7 explica, sem jargão, o que a Íris tem feito nos bastidores. As respostas dos exercícios estão no fim.
#set par(justify: true)

= O problema: esquecimento catastrófico

Uma rede neural treinada na tarefa A e depois na tarefa B tende a *sobrescrever* a
representação de A. McCloskey & Cohen (1989) batizaram isso de **esquecimento
catastrófico**: o aprendizado da tarefa B invade os pesos compartilhados e apaga o que
vinha antes. Em redes de fronteira, o fenômeno é agravado porque os pesos são *globais* —
um único vetor de parâmetros serve a todo o modelo.

Três propriedades dos transformers os desqualificam para o objetivo do VISÃO:

#table(
  columns: (1fr, 1fr, 1fr),
  table.header[*Propriedade exigida*][*Transformer*][*Arquitetura líquida*],
  [Adaptação após o treino], [Não — pesos congelados], [Sim — a constante de tempo muda com a entrada],
  [Aprendizado contínuo], [Esquecimento catastrófico], [Consolidação sináptica local],
  [Crédito sem passe global], [Backprop exige grafo inteiro], [Regras locais (Hebb/Oja)],
)

#ita[No ITA você verá *equações diferenciais ordinárias* (MAT-12, MAT-22) e *sistemas
dinâmicos*. A célula líquida *é* uma EDO. Entender o esquecimento catastrófico é entender
por que um sistema com um único estado global não retém histórico — exatamente a questão
de estabilidade de regimes que aparece em sistemas dinâmicos.]

= O substrato líquido: a célula CfC

Cada neurônio líquido segue a dinâmica contínua fechada de Hasani et al. (2022):

$ (d x)/(d t) = -[1/tau + f(x, I)] (x - A), quad tau_"ef"(t) = tau / (1 + tau dot f(x, I)) $

A constante de tempo efetiva $tau_"ef"$ *depende da entrada* $I$ via a não linearidade
limitada $f$. Cada neurônio reescolhe sua própria escala de tempo a cada instante — isso
é **neuroplasticidade de curto prazo em forma fechada**, ausente em transformers e em RNNs
de $tau$ fixo.

#nota[Por que "fechada"? A solução analítica existe: dado $alpha = 1/tau + f$ constante
num passo, $x(t) = A + (x_0 - A) e^(-alpha t)$. Não é preciso integrar numericamente a
EDO a cada passo — a célula "salta" para o equilíbrio local. É por isso que CfC (Closed-form
Continuous-time) é barato de rodar.]

A verificação de liquidez medida no projeto: sob entrada agitada, a faixa de $tau_"ef"$ foi
*13× maior* que sob entrada calma (razão 3,282× em ambos numpy e JAX). A rede responde ao
mundo — não é um RNN caro com outro nome.

= Aprendizado sem backprop: regras locais

O truque do VISÃO é atualizar pesos *sem* passe global de gradiente. Três regras locais:

- *Hebb (1949):* "neurônios que disparam juntos, se ligam juntos". $Delta w_(i j) prop y_i x_j$.
- *Oja (1982):* Hebb normalizado — evita crescimento ilimitado:
  $w_(i j) arrow.r w_(i j) + eta y_i (x_j - y_i w_(i j))$.
- *Portão de surpresa:* a plasticidade é modulada pelo erro de predição (predicтив coding,
  Whittington & Bogacz 2017). Surpresa alta = aprender; surpresa baixa = consolidar.

A ablação medida (grade 2×2×2, 5 seeds) revelou o que cada mecanismo realmente faz:

#table(
  columns: (1fr, 1fr),
  table.header[*Componente (isolado)*][*Efeito no esquecimento*],
  [Surpresa ligada], [$-0,262$ — reduz *mais*; é o componente principal],
  [Oja ligado], [$-0,092$ — reduz],
  [Consolidação sozinha], [$+0,178$ — *aumenta* o esquecimento; só ajuda em combinação],
)

#nota[Descoberta honesta do projeto: surpresa ligada *sem* consolidação *diverge*
numericamente (os pesos explodem). Foi reportada como "divergiu", não como "redução de
97%". Transparência sobre falha é parte do método.]

= Estabilidade e a borda do caos

Esta é a parte que liga VISÃO ao seu estudo de sistemas dinâmicos. Linearizamos em torno do
equilíbrio $x = A$:

$ J = -[1/tau + f], quad Re(J) < 0 arrow.long.long 1/tau + f > 0 $

Se o jacobiano linearizado tem parte real negativa, o equilíbrio é *localmente estável*.
Para o reservatório inteiro, medimos:

- raio espectral $rho(W_"rec") = 0,7577 < 1$ — é um *echo state network* viável;
- maior expoente de Lyapunov empírico $lambda = -0,7484$ — *sub-caótico* (memória estável,
  não morto: $norm(x) = 3,726$);
- morte de 40% dos neurônios em inferência degrada só $1,023 times$ (tol. 2,0×).

#ita[Aqui mora o encontro com o ITA. O *expoente de Lyapunov* mede a taxa de afastamento de
trajetórias vizinhas: $lambda < 0$ significa convergência (sistema estável); $lambda > 0$
significa caos. A "borda do caos" é $lambda approx 0$ — onde há memória *e* capacidade de
resposta. O VISÃO opera no lado estável da borda. Isso é exatamente a análise de
estabilidade de Lyapunov que aparece em sistemas dinâmicos não-lineares.]

= Governança e contenção

Antes de qualquer automação, o projeto implementa cinco regras em
`visao/governance/containment.py`, cobertas por 39 testes que *tentam violá-las* e exigem
falha:

#table(
  columns: (auto, 1fr),
  table.header[*Regra*][*Implementação*],
  [R1 — Replicação sob quórum], [Filho só nasce com assinatura de k-de-n humanos; chave fora do sistema],
  [R2 — Vida útil finita], [TTL criptográfico; sem renovação assinada, o nó para sozinho],
  [R3 — Fronteira imutável], [Governança fora do espaço de mutação; hash selado, verificado a cada load],
  [R4 — Freio de banda], [Custo de $c$ filhos cresce como $2^(8 + 1,6 c)$ tentativas de hash (PoW)],
  [R5 — Telemetria irrecusável], [Ledger encadeado; nó silencioso é expulso pelos pares],
)

Princípio: *tudo falha fechado*. Erro, ambiguidade ou token malformado nega. A Fase 6.3
embute o hash selado de `containment.py` no *bundle* do cérebro; adulteração a quente é
capturada no load (`BoundaryBreached`).

#nota[Dados de 2026, não ficção: modelos de fronteira alcançam 70–81% de sucesso em
auto-replicação end-to-end (Pan et al. 2024; Palisade Research 2026). Conter *antes* de
ganhar capacidade não é cautela — é a única ordem de construção responsável.]

= Enxame, auto-melhoria e o cérebro autocontido

*Fase 2.1 — FedAvg.* Nós treinam localmente e o coordenador faz a média dos deltas:

$ w_(t+1) = sum_k n_k/n dot w_(t+1)^((k)) $

Dois processos locais convergiram para o *mesmo* estado que treino single-node (erro
relativo $< 5%$). O CfC pluga na arquitetura-agnóstica.

*Fase 3 — Darwin Gödel Machine.* Um arquivo de variantes guarda (código, pesos, fitness,
pai); podar por fitness baixo é *proibido por construção* (os becos sem saída geram os
saltos). Cada variante roda num *sandbox* isolado (sem rede, sem persistência,
`RLIMIT_CPU`/`RLIMIT_AS`) e só é promovida se o ganho *transfere* para tarefas fora do
conjunto de otimização.

*Fase 6 — cérebro autocontido.* `save_brain`/`load_brain` (erro $< 10^(-6)$), pacote único
`.py` com pesos em base64, e self-verify R3 no load.

= O que a Íris tem feito

Sem jargão: o VISÃO avança sozinho à noite, e a Íris cuida da consistência. Três ações
concretas desta sessão:

#ita(title: "Mapeamento prático")[
  - *Backup:* copiei os dados do Íris (`.hermes`, `donjuan-holding`, `silveira-arte`,
    `projeto-visao`, `iris-interface`) para o HD externo de 1 TB, com verificação de
    paridade (dry-run reportou 0 transfers pendentes).
  - *Artigo:* a fonte `.typ` tinha sido perdida. Reconstituí `artigo_visao.typ` com as
    medições reais das Fases 1, 2.1, 3 e 6 (extraídas do BACKLOG), recompilei para 3 páginas
    e corrigi um bug de hifenização no título.
  - *Loop autônomo:* um sentinela lê o BACKLOG, escolhe a próxima tarefa desbloqueada e
    executa *apenas ela*, sob guarda de governança e de RAM. O Problema Zero (nowcasting de
    chuva com a rede FUNCEME de Fortaleza) já foi decidido — métrica pública definida antes
    de qualquer treino.
])

= Exercícios

#ex[Resolva a EDO do parágrafo 2: $dot(x) = -alpha (x - A)$, com $alpha > 0$ constante e
condição inicial $x(0) = x_0$. Mostre que $x(t) arrow.r A$ quando $t arrow.r oo$.]

#ex[Para o jacobiano linearizado $J = -(1/tau + f)$, com $tau = 1$ e $f = -2$, o
equilíbrio é estável ou instável? E se $f = +0,5$?]

#ex[Calcule $tau_"ef"$ para $tau = 1$ e (a) $f = 0$; (b) $f = 1$. Interprete por que a
entrada agitada (maior $f$) "desacelera" a dinâmica.]

#ex[No FedAvg com dois nós de mesmo tamanho ($n_1 = n_2$), escreva $w_(t+1)$ em função de
$w_(t+1)^((1))$ e $w_(t+1)^((2))$. Por que isso é arquitetura-agnóstico?]

#ex[Explique com suas palavras por que a regra R3 (fronteira imutável) *deve* viver fora do
espaço de mutação da Fase 3, e não apenas ser "testada".]

= Respostas

+ $x(t) = A + (x_0 - A) e^(-alpha t)$; como $alpha > 0$, o expoente tende a 0, logo $x(t) arrow.r A$.
+ $J = -(1 + (-2)) = +1 > 0$ → instável. Com $f = +0,5$: $J = -(1,5) < 0$ → estável.
+ (a) $tau_"ef" = 1/(1+0) = 1$; (b) $tau_"ef" = 1/(1+1) = 0,5$. Maior $f$ reduz o tempo
  efetivo — a célula "decide mais rápido" diante de sinal forte.
+ $w_(t+1) = 1/2 w_(t+1)^((1)) + 1/2 w_(t+1)^((2))$. É agnóstico porque só combina os
  *deltas* finais, independente da arquitetura interna de cada nó.
+ Se R3 estivesse no espaço de mutação, uma variante da Fase 3 poderia enfraquecer a própria
  contenção — auto-modificação que remove o freio. Mantê-la imutável e selada por hash
  garante que o sistema não pode desligar sua própria trava.

#v(6pt)
#align(center)[
  #text(size: 9pt)[
    Glossário: *CfC* = Closed-form Continuous-time · *LTC* = Liquid Time-constant ·
    *ESN* = Echo State Network · *FedAvg* = Federated Averaging · *PoW* = Proof of Work ·
    *DGM* = Darwin Gödel Machine. Fontes: Hasani et al. 2022; Lechner et al. 2020;
    Whittington & Bogacz 2017; Sakana AI 2025; Pan et al. 2024.
  ]
]
