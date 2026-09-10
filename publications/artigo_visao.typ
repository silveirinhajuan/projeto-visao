#set document(
  title: "Neuroplasticidade Local sem Backpropagation como Substrato de Aprendizado Contínuo em Rede Descentralizada e Auto-Replicante sob Contenção Criptográfica Verificável",
  author: "Juan Guerra · Projeto VISÃO",
)
#set page(numbering: "1", margin: 1.6cm)
#set text(size: 10.5pt, lang: "pt")
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #set par(justify: false)
  #set text(hyphenate: false)
  #text(weight: "bold", size: 14pt)[
    Neuroplasticidade Local sem Backpropagation como Substrato de \
    Aprendizado Contínuo em Rede Descentralizada e Auto-Replicante \
    sob Contenção Criptográfica Verificável
  ]
  #v(4pt)
  #text(size: 11pt)[Juan Guerra · Projeto VISÃO]
  #text(size: 10pt)[Revisado · Agosto de 2026 (Preprint)]
]

= Resumo

Apresentamos um sistema de aprendizado contínuo baseado em redes neurais líquidas de
constante de tempo adaptativa (LTC/CfC), atualizado por regras locais de Hebb/Oja sem
retropropagação do gradiente. Em protótipo de laboratório (96 neurônios, 5 sementes, três
tarefas temporais sequenciais sem replay), o modelo reduziu o esquecimento catastrófico em
97,2% em relação a um controlador de mesma topologia. A contribuição central — a ordem de
construção — foi *executada e medida*: uma camada de governança com quórum humano e
prova-de-trabalho crescente (39 testes de violação) foi implementada e testada *antes* da
auto-melhoria e da escala, invertendo a prática comum de "deploy primeiro, contenção
depois". Desde a submissão inicial (agosto/2026), as Fases 1 (substrato JAX em escala), 2.1
(enxame local FedAvg), 3 (auto-melhoria verificável) e 6 (cérebro autocontido) foram
concluídas com suítes de teste verdes (181 testes em `visao/`). O sistema é posicionado
como subsistema de memória contínua e protocolo de segurança, não como arquitetura de
inteligência geral.

= Introdução

O aprendizado contínuo em redes neurais convencionais esbarra no esquecimento
catastrófico: pesos globais compartilhados fazem o aprendizado da tarefa B sobrescrever a
representação de A#footnote[McCloskey e Cohen, 1989; French, 1999.]. Paralelamente, o
treino de modelos de fronteira exige infraestrutura centralizada proibitiva, e modelos de
fronteira já alcançam 70 a 81% de sucesso em auto-replicação end-to-end#footnote[Pan et
al., 2024; Palisade Research, 2026.], tornando a contenção um problema de engenharia, não
de ficção.

Este trabalho propõe um substrato — arquitetura líquida com plasticidade local — que ataca
o esquecimento por construção, e um protocolo de contenção que enclausura a
auto-replicação antes que ela seja escalada. A seção 2 descreve a célula; a seção 3 a
governança; as seções 4–7 reportam os resultados *medidos* das Fases 1, 2.1, 3 e 6; a
seção 8 discute limitações honestas.

= Substrato: Célula Líquida

Cada neurônio segue a dinâmica contínua fechada de Hasani et al.:

$ (d x)/(d t) = -[1/tau + f(x, I)](x - A), quad tau_(e f)(t) = tau / (1 + tau dot f(x, I)) $

A constante de tempo efetiva $tau_(e f)$ depende da entrada $I$ via a não linearidade $f$.
Cada neurônio reescolhe sua própria escala de tempo a cada instante — neuroplasticidade de
curto prazo em forma fechada, ausente em transformers e em RNNs de $tau$ fixo. A
atualização usa regras locais (Hebb/Oja), sem passe global de gradiente. O critério de
estabilidade linearizado em torno do equilíbrio $x = A$ é:

$ J = -[1/tau + f], quad Re(J) < 0 arrow.long 1/tau + f > 0 $

= Governança por Contenção

Antes de qualquer automação, implementamos cinco regras (R1 a R5) em
`visao/governance/containment.py`, cobertas por 39 testes que tentam violá-las e exigem
falha:

- *R1* — replicação exige quórum de assinaturas humanas;
- *R2* — TTL: sem renovação assinada, o nó para sozinho;
- *R3* — o arquivo de governança é imutável, selado por hash (re-selado pela Fase 6.3);
- *R4* — freio de trabalho: custo de $c$ filhos cresce como $2^(8 + 1.6 c)$ tentativas de hash;
- *R5* — ledger encadeado; nó silencioso é expulso.

Princípio: tudo falha fechado. Erro, ambiguidade ou token malformado nega. O módulo de
governança vive *fora* do espaço de mutação da Fase 3 (invariante R3, guardado por
`mutate.py` e pelo selo de hash verificado em cada load).

= Resultados Medidos

== Fase 0 — Protótipo NumPy (reprodutível)

Protótipo NumPy (96 neurônios, 5 sementes, 3 tarefas sequenciais, sem replay):

#table(
  columns: (1fr, 1fr, 1fr),
  table.header[*Condição*][*Esquecimento*][*Erro final*],
  [Líquido + local], [+0.0083 (±0.0138)], [0.2868],
  [Controle (delta)], [+0.2924 (±0.1009)], [0.5284],
)

Redução de *97,2%* no esquecimento. Verificação de liquidez: a faixa de $tau_(e f)$ foi
13× maior sob entrada agitada que calma. A portabilidade para JAX foi confirmada por
paridade numérica (erro máximo $5.55 times 10^(-16)$; razão de liquidez idêntica 3,282×).

== Fase 1 — Substrato em JAX (escala e benchmarks)

A célula CfC foi portada para JAX (`jax.jit` + `jax.vmap`), com topologia NCP esparsa
(sensory→inter→command→motor) e *predictive coding* multicamada (valor/erro, Whittington &
Bogacz 2017). Resultados medidos:

- *Estabilidade do reservatório* (n=96): raio espectral $rho(W_"rec") = 0,7577 < 1$ (ESN
  viável); maior expoente de Lyapunov empírico $lambda = -0,7484$ (*sub-caótico* — memória
  estável, não morto: $norm(x) = 3,726$).
- *Ablação 2×2×2* (consolidação/surpresa/Oja × 5 seeds): completo reduz 92,6% (forgetting
  +0,013 vs baseline +0,176); o componente *principal* é o portão de surpresa isolado
  ($-0,262$); Oja isolado $-0,092$; consolidação *sozinha* *aumenta* o esquecimento
  (+0,178) — só ajuda em combinação, como freio anti-divergência. Surpresa ligada sem
  consolidação *diverge* numericamente (reportada como tal, não como "redução").
- *Sensibilidade a $tau$*: heterogeneidade de constantes de tempo é *decoração* neste
  protocolo — $tau$ uniforme (+0,0135±0,0103) é indistinguível de $tau$ heterogêneo 10×
  (+0,0125); $Delta = -0,0011$ dentro do ruído. A memória multiescala vem do portão de
  surpresa, não da dispersão de $tau$.
- *Robustez*: ruído de entrada (desvio $= 1,0 times$ amplitude do sinal) degrada apenas
  $1,001 times$; morte de 40% dos neurônios em inferência degrada $1,023 times$ (pior caso
  $1,047 times$, dentro da tolerância 2,0×). O organismo tolera nós caindo — premissa da
  Fase 2 sustentada empiricamente.

*Benchmarks públicos (psMNIST permutado):*

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  table.header[*Modelo*][*Acurácia (2k/60ep)*][*Params*][*Acurácia (60k/Colab T4)*],
  [CfC (h=64)], [0,358], [5.002], [0,6939],
  [LSTM (h=128)], [0,434], [67.850], [0,8149],
  [GRU (h=128)], [0,488], [50.816], [não medido],
)

Razão de parâmetros: LSTM/CfC = *13,56×*, GRU/CfC = *10,16×* (alegação "~10× menos"
sustentada). No complemento 60k, o CfC fica 12,1 pp atrás do LSTM, porém com 13,56× menos
parâmetros. O critério de *sucesso* ("empatar ou superar") não foi atingido; o critério de
*falha* (>15% em duas tarefas) não dispara (1 tarefa completa, gap 12,1% < 15%). *O treino
do GRU em 60k foi interrompido por queda de sessão Colab — acurácia final não medida; não
inventamos o número.* Suíte `visao/` fechou em 181 testes verdes.

== Fase 2.1 — Enxame local (FedAvg)

Nó local de dois processos via socket (`visao/swarm/`), FedAvg por rodadas. Dois nós (200
amostras, shards 100/100, lr=0,01, K=5, 20 rodadas) convergem para o *mesmo estado* que
treino single-node (erro relativo $< 5%$). Arquitetura-agnóstico: o CfC pluga depois.
P2P real (libp2p/Iroh + handshake R3) ficou para a tarefa 2.3.

== Fase 3 — Auto-melhoria verificável

- *Arquivo de variantes* (`archive.py`): grafo pai→filho; podar por fitness baixo é
  *proibido por construção* (`PruningForbidden`) — os becos sem saída são mantidos.
- *Loop de mutação* (`mutate.py`): perturba $tau$/plasticidade/código; guarda R3 detecta
  marcadores proibidos (governança, rede, chaves) e levanta `ForbiddenMutation`.
- *Sandbox de fitness* (`sandbox.py`): subprocesso isolado — sem persistência (dir temporário
  sempre removido), sem rede (sombra de módulos + scan estático), `RLIMIT_CPU`/`RLIMIT_AS`
  matam loop infinito/alocação. Backend trocável sem alterar o contrato.
- *Portão de transferência* (`transfer.py`): variante só é aceita se o ganho no conjunto de
  otimização se sustenta num conjunto *fora* dele; rejeita trapaça de benchmark (teste
  end-to-end com fitness medido no Sandbox).

== Fase 6 — Cérebro autocontido

- *6.1* `save_brain`/`load_brain` round-trip: erro $< 10^(-6)$ nos pesos e no rollout.
- *6.2* pacote único `.py` autocontido: pesos em base64 + import da arquitetura.
- *6.3* self-verify no load (R3): o bundle embute o hash selado de `containment.py`;
  adulteração a quente é capturada no load (`BoundaryBreached`). Ao concluir, o manifesto R3
  foi *re-selado*. Suíte `visao/` (tests+evolve+governance): 124 testes verdes.

= Rede Descentralizada e Retenção

Modelo quantitativo de recrutamento (`network/recruitment_model.py`, ancorado em Anderson
arXiv:1903.01699): viral sem retenção estabiliza em *zero* nós; nicho fiel estabiliza em
1.250 (aprox. US\$ 4,94 M/ano em nuvem); o Folding\@home perdeu 98,6% da capacidade em cinco
anos. Dashboard de churn (`churn_dashboard.py`, SQLite) rastreia nós ativos em 7d, churn
mensal e N\* observado vs previsto.

*Problema zero* (decisão do Juan, 08/ago/2026): *nowcasting* de precipitação em bacias
urbanas de Fortaleza usando a rede FUNCEME (600+ postos desde 1974). Métrica pública
definida *antes*: RMSE(mm) a 1/3/6h vs baseline de persistência; sucesso = ensemble CfC
≥15% melhor em split temporal. O guardião de prontidão (`readiness_check.py`) imprime
*PROIBIDO RECRUTAR* enquanto o cliente instalável ($< 5$ min) e a tarefa 5.2 estiverem
pendentes — honrando seu propósito.

= Discussão e Limitações

O sistema é um *subsistema* — memória contínua eficiente e protocolo de segurança — não uma
arquitetura de inteligência geral. Limitações declaradas e *não maquiadas*:

- n = 5 sementes e 96 neurônios são insuficientes para inferência estatística robusta;
- no benchmark psMNIST 60k, o CfC perde do LSTM em acurácia (12,1 pp) — a vantagem é
  paramétrica (13,56× menos), não de acurácia;
- o GRU em 60k *não foi medido* (queda de sessão); não há número fabricado;
- H3 (contenção populacional) e o recrutamento real (Fase 5) seguem não implantados — o
  guardião bloqueia recrutamento prematuro;
- restam pendentes: 6.4/6.5 (R1+R4 no bundle, 39+ testes), 2.2–2.6 (Byzantinos, prova de
  contribuição, admissão, churn, sharding de 10^5 neurônios) e os portões da Fase 3.

= Conclusão

Redes líquidas com plasticidade local reduzem o esquecimento por construção, e a contenção
deve preceder a capacidade de replicação — *e neste projeto precedeu de fato*. As Fases 1,
2.1, 3 e 6 foram executadas e medidas com suítes verdes, validando a ordem de construção
como disciplina de engenharia, não apenas de intenção. O cérebro autocontido (Fase 6)
carrega arquitetura, pesos e freios num único arquivo verificável, mantendo R3 embutido no
ponto de entrada.

= Referências

- Hasani, R. et al. Liquid Time-constant Networks. *Nature MI* 4, 992–1003 (2022).
- Lechner, M. et al. Neural Circuit Policies (arXiv:1803.08554).
- Douillard, A. et al. DiLoCo (arXiv:2311.08105, 2023).
- Whittington, J. & Bogacz, R. Predictive coding com plasticidade Hebbiana local. *Neural Comp.* 2017.
- Pan, A. et al. Open-weight models and self-replication (2024).
- Palisade Research. AI self-exfiltration (2026).
- RAND. Strengthening Emergency Preparedness for AI (RRA3847-1, 2025).
- Anderson, D. BOINC (arXiv:1903.01699).
- McCloskey, M. & Cohen, N. Catastrophic interference (1989).
