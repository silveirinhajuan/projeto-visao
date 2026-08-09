#set document(
  title: "VISÃO — A Nova Realidade (single-node, rumo a AGI contínua)",
  author: "Projeto VISÃO · preparado por ÍRIS",
)
#set page(paper: "a4", margin: 1.5cm, numbering: "1",
  header: context [
    #set text(size: 8pt, fill: gray)
    #align(right)[VISÃO — Nova Realidade · escopo single-node]
  ],
  footer: context [
    #set text(size: 8pt, fill: gray)
    #align(center)[Página #counter(page).display() · material interno do projeto]
  ],
)
#set text(size: 10.5pt, lang: "pt", font: "New Computer Modern")
#set par(justify: false, spacing: 0.55em)
#set heading(numbering: "1.")
#show raw: set text(font: "DejaVu Sans Mono", size: 8.5pt)
#set math.equation(numbering: "(1)")

#let box(fillc, title, body) = block(
  fill: fillc, inset: 9pt, radius: 4pt, width: 100%,
)[
  #text(weight: "bold", fill: rgb(13,17,23))[«#title»] #body
]

#let nota(body) = box(rgb(230,242,255), "Nota", body)
#let atencao(body) = box(rgb(255,235,235), "Atenção", body)

= O que o VISÃO é agora (após a virada de 2026-08-09)

O projeto deixou de ser uma «rede de vários computadores» (enxame, recrutamento de
voluntários, auto-replicação em rede). Passou a ser **uma rede neural líquida de altas
habilidades que roda inteiramente no seu notebook** (Ryzen, ~7,6 GB RAM, CPU) — e a busca
continua sendo **aprendizado contínuo rumo a AGI**: uma rede que acumula conhecimento
incrementalmente, barato, sem retreinamento global dispendioso e sem esquecer o que já
aprendeu.

#atencao[Redes fixas (toda CNN/RNN/transformer treinada uma vez) ficam presas a conhecimento
obtido via treinamento caro e que não continua. A rede líquida do VISÃO *aprende
incrementalmente*: cada nova experiência ajusta pesos locais, sem reprojetar tudo. É por isso
que o projeto existe — e por que a parte de enxame era enfeite (e perigosa), não o núcleo.]

#nota[Corte de escopo (formalizado em `PLANO.md` e `BACKLOG.json`, commit `030f0c3`): removidos
*swarm multi-node*, *peer recruitment*, *self-replication over network*, *PoW spawn gate*,
*remote quorum*, *DGM autonomous self-rewrite*. Mantidos: célula LTC, regras locais,
aprendizado contínuo sem esquecer, estabilidade/Lyapunov, treino single-node, bundle
autocontido, e R3 como «núcleo imutável do assistente local» (não anti-replicação).]

= Por que redes líquidas (e não fixas) são o caminho para AGI contínua

Uma rede neural tradicional guarda *todo* o conhecimento num único vetor de pesos globais.
Treinar a tarefa B sobrescreve a representação da tarefa A → **esquecimento catastrófico**
(McCloskey & Cohen, 1989). Transformer piora: pesos congelados após deploy, backprop exige o
grafo inteiro.

A célula líquida do VISÃO resolve isso por dois lados:

#box(rgb(235,255,235), "Matemática", [
  Cada neurônio segue a EDO (LTC, Hasani 2022):

  $ (d x)/(d t) = -[1/tau + f(x, u)](x - A), $

  onde $f = sigma(W_(i n) u + W_(r e c) x + b) in (0,1)$ e $A$ é potencial de reversão.
  A constante de tempo *efetiva* depende da entrada:

  $ tau_(e f) = tau / (1 + tau dot f). $

  Entrada forte ($f -> 1$): $tau_(e f)$ encolhe → reage rápido. Entrada fraca
  ($f -> 0$): $tau_(e f) -> tau$ → mantém memória de longo prazo (neuroplasticidade de
  curto prazo, em forma fechada). Memória *multiescala emergente* — sem «pastas» no código.
])

#nota[O `cfc.py` do repositório implementa isto como *LTC com solver fundido* (Euler
semi-implícito), **não** o CfC fechado de Hasani (que interpola a trajetória por sigmoid sem
passos). A estabilidade incondicional do solver (denominador sempre $> 1$) é o que permite
usar $d t$ grande sem explosão — técnica clássica de EDO rígida. Corrigido em `b9e8d5a`.]

= O que já é real e medido (não promessa)

Tudo abaixo roda e tem número no repositório:

#box(rgb(245,245,245), "Resultados empíricos", [
  - *Aprendizado contínuo (Fase 0, `prototype/results_continual.json`, 5 seeds):*
    esquecimento **97,2% menor** que controle (média dos seeds; seeds 2–3 até *melhoraram*
    a tarefa A ao treinar B). Ablação 2×2×2 mostra que o componente principal é *surpresa*
    ($-0,262$), não localidade (Oja $-0,092$); consolidação sozinha *piora* ($+0,178$).
  - *Estabilidade (`results_stability.json`):* $rho(W_(r e c)) = 0,7577 < 1$ (ESN viável),
    $max |lambda_J| = 0,9754 < 1$ (jacobiano contrativo), Lyapunov
    $lambda = -0,7484$ (sub-caótico). *Segunda medição independente* via algoritmo de
    Benettin (1980) que eu rodei: $lambda = -0,7876$. Convergência confirma o sinal.
  - *Parâmetros (`results_param_count.json`):* CfC ≈ 5.002 params vs LSTM ≈ 67.850
    (13,56× menor). Gap de acurácia em psMNIST: −12,1 pp (honesto; GRU 60k não medido).
  - *FedAvg local (tarefa 2.1):* 2 processos, convergiu a $< 5%$ do single-node. Agora
    serve só como referência de agregação local — sem rede.
])

#atencao[O que o VISÃO **não** é: não é um LLM. Com ~5 mil parâmetros ele brilha em
*sequências contínuas, controle e aprendizado contínuo* — não em gerar texto como o GPT.
Ele prevê dinâmicas, controla sistemas e nunca esquece; não redigirá prosa. Isso casa com
seu plano de Computação Quântica (sistemas dinâmicos, Lyapunov — mesmo vocabulário).]

= O que muda no roteiro (nova Fase 5)

Em vez de coordenar nós, o esforço vai para **aprendizado contínuo real no notebook**:

#box(rgb(240,255,240), "Nova Fase 5", [
  - *5.1* — Escalar a LTC de $n=96$ para $n approx 512$–$1024$ validando RAM/CPU do
    seu notebook (medir pico de RAM e tempo por passo).
  - *5.2* — Substituir o brinquedo psMNIST por **uma tarefa contínua real sua** (ex.:
    previsão de série temporal da sua rotina/estudos) e demonstrar não-esquecer entre
    $>= 2$ tarefas sequenciais.
  - *5.3* — Auto-ajuste *local* em sandbox (arquitetura/hiperparâmetros) — AutoML comum,
    com R3 protegendo o núcleo. Fora do escopo: replicação em rede.
])

As antigas tarefas 2.1–2.6 (Byzantinos, sharding, quorum de spawn, PoW) estão marcadas
`discontinued_2026_08_09` no BACKLOG.

= Onde isto encosta no seu ITA

- *MAT-22 (sistemas dinâmicos):* a EDO da célula, estabilidade de Lyapunov e a «borda do
  caos» ($rho(W) < 1$ é o *echo state property* de Jaeger 2001, não LTC; «líquido» vem das
  *Liquid State Machines* de Maass 2002) são exatamente o vocabulário que você verá.
- *Álgebra linear / MAT-12:* Oja (1982) é PCA online — o peso converge ao autovetor
  principal da covariância da entrada. É o mesmo motor de PCA em streaming.
- *Estatística experimental:* a ablação 2×2×2 é *design de experimentos* puro.

= Resumo para não perder o fio

VISÃO = uma rede líquida pequena, interpretável e barata que aprende *continuamente* no seu
notebook, nunca esquece, e cujas decisões você entende. A parte de «exército de máquinas»
foi cortada (segurança + realismo). O diferencial real — aprendizado contínuo minúsculo —
está medido e funciona. Próximo passo de engenharia: escalar para o notebook e botar uma
tarefa sua de verdade nela.

#pagebreak()
#set text(size: 8pt, fill: gray)
Fontes vivas: `visao/core/cfc.py`, `visao/governance/containment.py`, `prototype/results_continual.json`,
`visao/analysis/results_stability.json`, `visao/bench/results_param_count.json`, `PLANO.md`, `BACKLOG.json`.
