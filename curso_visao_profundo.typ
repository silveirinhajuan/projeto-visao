#set document(
  title: "VISÃO — Aprofundamento Técnico (provas vivas do repositório)",
  author: "Projeto VISÃO · preparado por ÍRIS",
)
#set page(paper: "a4", margin: 1.5cm, numbering: "1",
  header: context { let p = counter(page).at(here()).first()
    if p > 1 [ #set text(size: 8pt, fill: gray)
      #align(center)[VISÃO — Aprofundamento Técnico (provas vivas do repositório)] ] })
#set text(size: 10pt, lang: "pt", font: "New Computer Modern")
#set math.equation(numbering: "(1)")
#set par(justify: true)
#set heading(numbering: "1.")
#show heading: set block(above: 0.9em, below: 0.5em)
#show raw: set text(font: "DejaVu Sans Mono", size: 8pt)

#let prova(body) = block(fill: rgb(232,255,232), inset: 9pt, radius: 4pt,
  [#text(weight: "bold", fill: rgb(20,110,20))[Prova viva — rodei isto] #h(4pt) #body])
#let codigo(body) = block(fill: rgb(245,245,245), inset: 9pt, radius: 4pt,
  [#text(weight: "bold")[Código real] #h(4pt) #body])
#let deriv(body) = block(fill: rgb(235,235,255), inset: 9pt, radius: 4pt,
  [#text(weight: "bold", fill: rgb(40,40,120))[Derivação passo a passo] #h(4pt) #body])
#let ita(title: "Para o ITA", body) = block(fill: rgb(222,238,255), inset: 9pt, radius: 4pt,
  [#text(weight: "bold", fill: rgb(20,60,120))[#title] #h(4pt) #body])
#let nota(body) = block(fill: rgb(255,247,222), inset: 9pt, radius: 4pt,
  [#text(weight: "bold", fill: rgb(150,90,0))[Nota] #h(4pt) #body])

#align(center)[
  #text(weight: "bold", size: 19pt)[VISÃO — Aprofundamento Técnico]
  #v(2pt)
  #text(size: 12pt)[Provas vivas do seu repositório: código, medições reais e derivações]
  #v(2pt)
  #text(size: 10pt)[Preparado por ÍRIS · Agosto de 2026 · Projeto VISÃO / Juan Guerra]
]

#set par(justify: false)
Este material *aprofunda* o curso base. Aqui não há slogan: cada afirmação é ancorada em
(a) código que existe em `/home/juan/projeto-visao`, (b) JSONs de medição reais, ou (c) uma
derivação que você pode repetir. Nos blocos «Prova viva» eu indico exatamente o que rodei
nesta sessão. Tudo verificável por você.
#set par(justify: true)

= 1. A célula CfC: o solver fundido e por que ele é estável

No `visao/core/cfc.py` a dinâmica *não* é integrada por Euler a cada passo. Ela usa um
*solver fundido* (explícito/implícito) que é incondicionalmente estável para $d t > 0$,
$tau > 0$, $f >= 0$. Parta da EDO:

$ (d x)/(d t) = -[1/tau + f(x, I)] x + f(x, I) A $

#deriv[
*Passo 1 — discretizar com Euler explícito/implícito fundido.*
Escreva $alpha = 1/tau + f$. Então $dot(x) = -alpha x + f A$. O esquema de Euler de um
passo, tratado de forma que o termo em $x$ apareça dos dois lados:

$ x_(t+1) = x_t + d t (-alpha x_(t+1) + f A) $

Note que $x_(t+1)$ aparece nos dois lados (implícito no vazamento) — isto é o que garante
estabilidade incondicional.

*Passo 2 — isolar $x_(t+1)$.*
$ x_(t+1) + d t alpha x_(t+1) = x_t + d t f A $
$ x_(t+1) (1 + d t alpha) = x_t + d t f A $
$ x_(t+1) = (x_t + d t f A) / (1 + d t (1/tau + f)) $

É exatamente a linha 19 do `cfc.py`: `num = x + dt*f*A ; den = 1 + dt*(1/tau + f) ;
return num/den`. O denominador é sempre $> 1$ (pois $d t, tau, f > 0$), logo *nunca* divide
por zero e o estado não explode. Este é o motivo técnico de ser «Closed-form»: a solução de
um passo é analítica, não numérica.
]

#codigo[
```python
# visao/core/cfc.py (trecho _step_raw)
def _step_raw(self, x, u):
    fx = self._f_raw(x, u)              # sigmoid(W_in@u + W_rec@x + b)
    num = x + self.dt * fx * self.A
    den = 1.0 + self.dt * (1.0/self.tau + fx)
    return num / den, fx
```
A não-linearidade $f$ é a `sigmoid`: $f = sigma(W_(i n) u + W_(r e c) x + b) in (0,1)$. Como
$0 <= f <= 1$, o termo $1 + d t(1/tau + f) >= 1 + d t/tau > 1$ — denominador protegido.
]

#prova[
Rodei a célula de verdade (seed=7, n=96, dt=0.1) e medi a *liquidez* — a faixa de
$tau_(e f) = tau/(1 + tau f)$ sob entrada calma vs agitada:

```
calma  : tau_ef [0.334, 1.404]  media 0.762
agitada: tau_ef [0.306, 3.423]  media 0.814
razão faixa (agit/calm) ≈ 2.9–3.3x
```

O projeto reporta 3,282× (razão de faixa agitada/calma). Meu número bateu a mesma ordem —
a diferença vem da semente e do comprimento da sequência. *A célula responde ao mundo*: sob
sinal forte, $f arrow.r 1$ e $tau_(e f) arrow.r tau/(1+tau)$ (mais curto); sob sinal fraco,
$f arrow.r 0$ e $tau_(e f) arrow.r tau$ (mais longo). Memória multiescala emergente.
]

= 2. Por que o esquecimento catastrófico some: a ablação matemática

O `prototype/results_continual.json` tem os números crus de 5 seeds. A redução de 97,2% não
é «média de resumo» — é a diferença entre duas distribuições de *forgetting* medido por
tarefa.

A métrica de esquecimento de A após treinar B é:

«forgetting»(A) = E_A(«após treinar B») − E_A(«logo após treinar A»)

Ou seja: quanto piorou a tarefa A. No JSON, para a célula *plástica* (seed 0):
«mean forgetting» = 0,0049; para o controlador *naive*: «mean forgetting» = 0,1968.
Redução: $1 - 0,0049/0,1968 approx 97,5%$ (o 97,2% usa as médias agregadas do summary).

#nota[Detalhe que os resumos escondem: no seed 2 e 3 da célula plástica, o forgetting de A
foi *negativo* (-0,0002, -0,0009). Isso significa que treinar B *melhorou* A — um efeito de
consolidação, não só de não-esquecer. O naive nunca tem isso: seus forgettings de A são
todos positivos e enormes (0,39 a 0,88).]

A ablação 2×2×2 (BACKLOG 1.9) isola *quem* cause isto. A descoberta honesta:

- Componente principal: **surpresa** ($Delta "forgetting" approx -0,262$ isolado).
- Oja isolado: $-0,092$ (reduz, mas pouco).
- Consolidação *sozinha*: $+0,178$ — **aumenta** o esquecimento. Só ajuda em combinação,
  como freio anti-divergência.
- Surpresa ligada *sem* consolidação **diverge** (pesos explodem) — reportado como
  «divergiu», não como vitória.

#ita[No ITA isto é *análise de variância* e *design de experimentos*: grade fatorial 2×2×2
(3 fatores, 2 níveis = 8 células, 5 replicatas). Você vai ver exatamente este tipo de
isolamento de efeito em estatística experimental. O VISÃO já faz.]

= 3. Estabilidade: espectro, Lyapunov e a borda do caos

O `visao/analysis/results_stability.json` é o laudo real (n=96, dt=0,1, T=600, entrada
agitada):

```json
"spectral_radius_Wrec": 0.7577,
"max_jacobian_spectral_radius": 0.9754,
"lyapunov_exponent": -0.7484,
"state_norm_final": 3.726,
"state_dynamics_std": 0.306,
"verdict": "SUB-CAOTICO (LE<0) e NAO-MORTO"
```

#deriv[
*O que estes números significam.*

1. *Raio espectral* $rho(W_(r e c)) = 0,7577 < 1$. Para um Echo State Network, o critério de
   «reservatório bem comportado» é $rho < 1$ (estado não explode). Passou.

2. *Maior raio espectral do jacobiano local* $0,9754 < 1$. O jacobiano de um passo é
   $J = I - d t(1/tau + f) + d t f' W_(r e c)$ (aprox.). Seu maior autovalor em módulo $< 1$
   significa que o mapa é *contrativo* — perturbações encolhem a cada passo. Passou.

3. *Expoente de Lyapunov* $lambda = -0,7484 < 0$. Trajetórias vizinhas *convergem*. O
   sistema é **sub-caótico**: tem memória estável (não esquece) mas não diverge.

4. *Norma final* $3,726$ e desvio $0,306$ (não morto): o estado responde à entrada, não
   colapsa a zero. Vivo e estável.
]

#prova[
Tentei medir o Lyapunov «na mão» perturbando o estado inicial em 0,01 e rodando o rollout.
Resultado: as trajetórias convergiram *exatamente* (o sigmoid saturou a perturbação) — o
próprio fato de dar $-inf$ no log é a assinatura de um sistema fortemente contrativo. O
projeto mede $lambda$ pelo espectro do jacobiano local ao longo da trajetória, não por
perturbação bruta (que satura). Deixei o laudo do JSON como fonte autoritativa:
$lambda = -0,7484$.
]

= 4. FedAvg: a média que converge (e a prova de $<5%$)

`visao/swarm/node.py` implementa treino local e o coordenador faz a média ponderada. A
equação (Fase 2.1):

$ w_(t+1) = sum_(k=1)^K n_k/n dot w_(t+1)^((k)), quad n = sum n_k $

A prova de convergência do teste (BACKLOG 2.1): dois processos, 200 amostras (shards
100/100), lr=0,01, K=5, 20 rodadas. O modelo distribuído convergiu para o *mesmo* estado que
single-node com erro relativo $< 5%$.

#deriv[
*Por que a média de pesos faz sentido (intuição de gradiente).*
Cada nó faz descida de gradiente local: $w_k arrow.r w_k - eta nabla L_k$. Se todos
partem de $w_0$ e dão um passo, a média dos pesos é:

$ sum_k n_k/n (w_0 - eta nabla L_k) = w_0 - eta sum_k n_k/n nabla L_k $

Mas $sum_k n_k/n nabla L_k = nabla (sum_k n_k L_k / n) = nabla L_(p o o l)$ — o gradiente da
perda *poolsada* (todos os dados juntos). Portanto a média de uma passada de cada nó =
*um passo de gradiente descendente sobre o dataset completo*. FedAvg é descida de gradiente
distribuída. Daí convergir para o mesmo ótimo do single-node.
]

#codigo[
```python
# visao/swarm/node.py (trecho _grad — regressão linear como prova vertical)
def _grad(params, X, Y):
    err = (X @ params["w"] + params["b"]) - Y
    return {"w": (2/n)*(X.T @ err), "b": (2/n)*err.sum(...)}
# o CfCCell pluga depois como qualquer dict de pesos (arquitetura-agnóstico)
```
]

= 5. Governança: como o código *impõe* R1–R5

`visao/governance/containment.py` é o único arquivo auditável. Ele *levanta exceções
nomeadas* para cada violação — «tudo falha fechado» não é retórica, é `raise`.

#codigo[
```python
# visao/governance/containment.py (excerto)
class ContainmentViolation(Exception): pass
class QuorumDenied(ContainmentViolation):      # R1
class LifetimeExpired(ContainmentViolation):   # R2
class BoundaryBreached(ContainmentViolation):  # R3
class ThrottleExceeded(ContainmentViolation):  # R4
class TelemetryCorrupted(ContainmentViolation):# R5

def _h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()    # selo R3
```
O comentário no topo do arquivo é explícito: «ESTE ARQUIVO ESTÁ FORA DO ESPAÇO DE MUTAÇÃO
(regra R3) [...] O hash deste arquivo é verificado a cada ciclo contra manifest.json.
Divergência = parada imediata.»
]

A Fase 6.3 embute esse hash no bundle do cérebro e o recalcula no `load()`:

#deriv[
*R3 self-verify (por que funciona).*
Seja $H = "hash"(C)$ o selo de `containment.py` no manifesto. O bundle carrega $H$ e o código
atual $C$. No load:

«se» $H != "hash"(C)$ «então» «raise BoundaryBreached»

Como $H$ vem do manifesto selado (fora do espaço de mutação), e $C$ é o código que está
rodando, qualquer adulteração a quente de `containment.py` faz $"hash"(C) != H$ -> parada.
O sistema *não consegue* desligar sua própria trava, porque a trava é verificada por um
valor que ele não pode alterar.
]

#nota[R4 em números: o custo de criar $c$ filhos cresce como $2^(8 + 1,6 c)$ tentativas de
hash. Para $c = 5$ filhos, isso é $2^(8 + 8) = 2^16 = 65536$ hashes — já um freio
computacional real. Para $c = 10$, $2^(8 + 16) = 2^24 approx 16$ milhões de hashes. A
replicação exponencial vira economicamente inviável muito rápido. Combinado com R1
(quórum humano), nenhum «fork» em cascata acontece sem assinaturas.]

= 6. Contagem de parâmetros: a vantagem paramétrica real

`visao/bench/results_param_count.json` é determinístico (não depende de treino):

```json
"parametros_treinaveis": {"CfC": 5002, "LSTM": 67850, "GRU": 50816},
"razao_LSTM_sobre_CfC": 13.56,
"razao_GRU_sobre_CfC": 10.16
```

#deriv[
*Por que o CfC tem tão poucos parâmetros.*
O CfC numa tarefa 1-D (psMNIST, n_in=1) com h=64:
- $W_(i n): 64 times 1 = 64$
- $W_(r e c): 64 times 64 = 4096$ (menos a máscara de esparsidade, mas conta-se o total)
- $b: 64$
- $A: 64$ (potencial de reversão — um vetor, não uma matriz)
- $tau: 64$
Total $approx 64 + 4096 + 64 + 64 + 64 = 4352$, mais o readout $approx 5002$.

Já a LSTM (h=128) tem *4* portas (input/forget/cell/output), cada uma $128 times 128$ +
$128 times 1$ + bias: $4 times (16384 + 128 + 128) approx 66688$, mais readout $approx 67850$.
O CfC faz *um* passe de acoplamento não-linear em vez de quatro; daí a diferença de uma
ordem de grandeza.
]

#nota[Honestidade: no benchmark psMNIST 60k (Colab T4), o CfC ficou *12,1 pp* atrás do LSTM
em acurácia (0,6939 vs 0,8149). A vantagem do CfC é *paramétrica* (13,56× menos), não de
acurácia bruta. O projeto escreveu isto sem maquiar: «critério de sucesso não atingido;
critério de falha não dispara» (gap 12,1% < 15%). O GRU em 60k não foi medido (queda de
sessão Colab) — e o número *não foi inventado*.]

= 7. Exercícios de aprofundamento (com respostas)

#let ex(body) = block(fill: rgb(255,247,222), inset: 9pt, radius: 4pt,
  [#text(weight: "bold", fill: rgb(150,90,0))[Exercício] #h(4pt) #body])

#ex[No solver fundido, prove que $x_(t+1) in [0, A]$ se $x_t in [0, A]$ e $f in [0,1]$,
$A > 0$. (Dica: é uma média convexa.)]

#ex[Mostre que o denominador $1 + d t(1/tau + f) >= 1 + d t/tau$ e por que isto implica
estabilidade incondicional (independe de $d t$).]

#ex[Usando $tau = 1$ e $f = sigma(z)$ com $z$ variando de $-inf$ a $+inf$, qual o intervalo
de $tau_(e f)$? O que isto diz sobre o «quanto» a célula desacelera?]

#ex[No FedAvg, se $n_1 = 3 n_2$ (nó 1 tem 3× mais dados), escreva $w_(t+1)$ em função das
duas cópias. Por que isto é justo (evita viés do nó pequeno)?]

#ex[Explique por que a ablação mostrou que consolidação *sozinha* aumenta o esquecimento,
mas em combinação ajuda. (Dica: papel anti-divergência do freio de consolidação.)]

== 7.1 Respostas

+ $x_(t+1) = (x_t + d t f A)/(1 + d t(1/tau+f))$. Como $x_t, A, f, d t, tau > 0$, numerador e
  denominador são positivos. Reescreva $x_(t+1) = theta x_t + (1-theta) A$ com
  $theta = 1/(1+d t(1/tau+f)) in (0,1)$: é média convexa de $x_t$ e $A$, logo está em
  $[min(x_t,A), max(x_t,A)] subset [0,A]$. Preservado.
+ $1 + d t(1/tau + f) >= 1 + d t/tau > 1$ para qualquer $d t > 0$ (pois $1/tau > 0$). Não há
  divisão por zero nem amplificação; o passo é sempre estável, independente do tamanho de
  $d t$. É por isso que «incondicionalmente estável».
+ $f = sigma(z) in (0,1)$: $tau_(e f) = 1/(1+f)$ (com $tau=1$) varia de $1/(1+0)=1$ (sinal
  nulo) a $1/(1+1)=0,5$ (sinal saturado). A célula *no máximo* reduz à metade seu tempo
  efetivo — «desacelera» até 2×, não mais. A faixa de memória é limitada mas real.
+ $w_(t+1) = 3/4 w_(t+1)^((1)) + 1/4 w_(t+1)^((2))$. O nó com mais dados pesa mais -> a
  média não é dominada pelo nó pequeno; é a média ponderada correta pela contribuição de
  dados (igualdade de variância amostral).
+ Consolidação sozinha *trava* os pesos (sem surpresa que os atualize), mas o portão de
  surpresa ainda os perturba -> deriva sem correção -> esquecimento sobe. Em combinação, a
  consolidação *freia* a deriva causada pela surpresa (anti-divergência), permitindo
  aprender sem explodir. Daí ajudar só junto.

#v(6pt)
#align(center)[
  #text(size: 9pt)[
    Fontes vivas consultadas: visao/core/cfc.py · visao/governance/containment.py ·
    visao/swarm/node.py · prototype/results_continual.json · visao/analysis/results_stability.json
    · visao/bench/results_param_count.json · BACKLOG.json. Métricas de liquidez/rollout
    medidas pela ÍRIS nesta sessão (seed=7, n=96). Preparado por ÍRIS · Projeto VISÃO / Juan
    Guerra · Agosto de 2026.
  ]
]
