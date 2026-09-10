# Redes Neurais Líquidas e Modelos de Linguagem

## Da Liquid Time-Constant Network aos Liquid Foundation Models

Relatório técnico — literatura acadêmica e estado da arte até setembro de 2026

---

## 1. Resumo executivo

As Liquid Neural Networks (LNNs) surgiram como uma família de redes neurais recorrentes de tempo contínuo nas quais a dinâmica interna do modelo é descrita por sistemas diferenciais e pode se adaptar ao estado e à entrada.

A linhagem central pode ser representada aproximadamente por:

```
Neural ODE → LTC → CfC → Liquid-S4 → Liquid Foundation Models
```

Entretanto, essa cadeia não significa que um LFM moderno seja simplesmente uma LTC gigantesca.

A evolução é mais sutil:

- **LTCs** introduzem constantes de tempo adaptativas em redes recorrentes contínuas.
- **CfCs** procuram obter uma dinâmica equivalente ou aproximada em forma computacional fechada, eliminando a necessidade de resolver numericamente uma EDO a cada atualização.
- **Liquid-S4** combina ideias de redes líquidas com State Space Models (SSMs), aproximando a dinâmica líquida de uma forma compatível com os mecanismos eficientes de S4.
- **Liquid Foundation Models** levam princípios de sistemas dinâmicos, processamento de sinais e álgebra linear numérica para modelos generativos de larga escala.
- **LFM2**, publicado em 2025, abandona a ideia de que uma arquitetura puramente líquida seja necessariamente a solução ótima e utiliza uma arquitetura híbrida, combinando convoluções causais curtas com blocos de Grouped Query Attention.
- **LFM2.5**, lançado em 2026, mantém essa filosofia de arquitetura eficiente e a leva para modelos de linguagem, visão-linguagem, áudio e agentes on-device.

Portanto, a resposta à pergunta:

> "Existem LNNs aplicadas a LLMs?"

é:

**Sim, mas a aplicação mais importante não é uma simples substituição de cada Transformer block por uma LTC.** A pesquisa evoluiu para arquiteturas híbridas de modelos de estado, convoluções, atenção e operadores inspirados em sistemas dinâmicos.

A própria Liquid AI descreve os LFMs como modelos fundamentados em teoria de sistemas dinâmicos, processamento de sinais e álgebra linear numérica.

---

## 2. O problema fundamental

A arquitetura Transformer revolucionou o processamento de linguagem porque o mecanismo de self-attention permite que cada token interaja diretamente com outros tokens do contexto.

Para uma sequência `X = (x_1, x_2, ..., x_T)`, a atenção calcula aproximadamente:

```
Q = XW_Q,  K = XW_K,  V = XW_V
Attention(Q,K,V) = softmax(QK^T / √d_k) V
```

O problema fundamental é que `QK^T ∈ R^{T×T}`. Assim, o custo de atenção cresce aproximadamente como `O(T²)` em relação ao comprimento da sequência.

Além disso, durante geração autoregressiva, o modelo normalmente mantém uma KV cache, cujo tamanho cresce com o contexto.

Isso cria uma tensão:

```
mais contexto ⇒ mais memória
mais tokens   ⇒ mais computação
```

Essa é uma das razões pelas quais arquiteturas alternativas ao Transformer são tão importantes.

---

## 3. A ideia das redes neurais líquidas

Uma RNN convencional pode ser representada por:

```
h_t = f(h_{t-1}, x_t; θ)
```

Existe uma atualização discreta: `h_{t-1} → h_t`.

Uma rede contínua no tempo, por outro lado, pode ser descrita por:

```
dh(t)/dt = f(h(t), x(t), t; θ)
```

Aqui o estado não precisa ser conceitualmente interpretado como uma sequência de passos discretos. Ele é uma **variável dinâmica**.

Essa mudança é muito mais profunda do que simplesmente trocar uma função de ativação. A rede passa a ser tratada como um **sistema dinâmico aprendido**.

---

## 4. Liquid Time-Constant Networks

O trabalho fundamental é: **Hasani et al., "Liquid Time-constant Networks", AAAI 2021**.

O artigo introduz uma classe de redes neurais recorrentes contínuas nas quais os neurônios possuem time constants que variam com o estado e a entrada.

A equação simplificada apresentada no trabalho pode ser entendida como:

```
dx(t)/dt = -[w_τ + f(x(t), I(t))] x(t) + A f(x(t), I(t))
```

O detalhe crucial é: **`τ = τ(x, I)`**.

Ou seja, a escala temporal efetiva do neurônio pode mudar. É daí que vem a palavra **liquid**.

Uma rede tradicional pode possuir uma dinâmica relativamente rígida. Uma rede líquida pode alterar sua dinâmica dependendo do estímulo.

---

## 5. Intuição física

Considere dois sinais `x_1(t)` e `x_2(t)`. Imagine que um representa uma mudança extremamente rápida e outro uma tendência lenta.

Um sistema com uma única escala temporal fixa precisa encontrar uma maneira de representar ambos.

Uma unidade líquida pode, conceitualmente, modificar sua escala de resposta:

```
τ_rápida ≠ τ_lenta
```

Portanto:

> O modelo não aprende apenas "o que responder" mas também **"em que escala temporal responder"**.

Essa propriedade é particularmente natural para problemas físicos, controle, robótica, sensores e séries temporais.

---

## 6. Por que isso não virou imediatamente uma alternativa aos Transformers?

O próprio trabalho posterior sobre Closed-form Continuous-time Neural Networks é bastante explícito:

> "Para language modeling em larga escala, Transformers e variantes eram considerados escolhas fortes."

O motivo é **computacional**. Resolver uma EDO numericamente durante cada atualização pode ser caro.

Uma dinâmica como `dh/dt = f(h, x, t)` não significa automaticamente que podemos calcular `h(t + Δt)` em uma única operação barata.

É necessário algum método numérico: `ODEsolver(f, h, t, t + Δt)`. Isso pode envolver várias avaliações da função `f`.

Portanto:

```
dinâmica elegante ≠ arquitetura eficiente
```

Essa distinção é central para entender toda a evolução posterior.

---

## 7. Closed-form Continuous-time Networks

Em 2022, Hasani et al. publicaram: **"Closed-form continuous-time neural networks"**, Nature Machine Intelligence.

O objetivo era atacar justamente o problema computacional das LTCs.

Em vez de depender diretamente de um solucionador numérico de EDO para cada atualização, os autores derivam uma aproximação em forma fechada da dinâmica.

A ideia é obter algo parecido com:

```
h(t + Δt) = F(h(t), x(t), Δt; θ)
```

sem executar todo o processo de integração numérica.

Isso produz os **CfCs** — Closed-form Continuous-time Neural Networks.

---

## 8. Por que CfC é importante para LLMs?

Porque ele começa a transformar:

```
modelo baseado em EDO → operador computacional eficiente
```

Essa diferença é fundamental. Um modelo contínuo só se torna competitivo em hardware moderno quando sua dinâmica pode ser transformada em operações que GPUs, CPUs e NPUs executam eficientemente.

O paper mostra que CfCs preservam características de modelos contínuos enquanto podem ser significativamente mais rápidos e estáveis.

Mas existe uma ressalva histórica muito importante: CfC não foi concebido como um substituto direto de Transformer para LLMs.

O próprio artigo posiciona CfC principalmente em:

- séries temporais
- robótica
- sistemas embarcados
- controle
- dados irregulares
- aplicações nas quais eficiência e interpretabilidade são importantes

Para grandes modelos de linguagem, os autores ainda consideravam Transformers uma escolha forte.

---

## 9. O salto para State Space Models

O próximo passo conceitual é extremamente importante.

Considere um sistema linear:

```
dx(t)/dt = Ax(t) + Bu(t)
y(t) = Cx(t) + Du(t)
```

Isso é um **State Space Model**. A ideia fundamental é:

```
entrada → estado → saída
```

O sistema não precisa guardar todas as entradas individualmente. Ele mantém uma representação comprimida: `x_t`.

Isso é extremamente atraente para processamento de sequências.

---

## 10. S4

A família **Structured State Space Models**, especialmente S4, mostrou que sistemas de estado estruturados poderiam ser treinados eficientemente e funcionar muito bem em sequências longas.

O ponto conceitual é:

```
x_t = A x_{t-1} + B u_t
```

Enquanto a atenção tenta relacionar explicitamente elementos da sequência, um SSM mantém um estado comprimido.

Isso produz uma dicotomia fundamental:

- **Transformer**: memória explícita
- **SSM**: memória comprimida no estado

Essa segunda abordagem é uma das raízes conceituais da atual corrida por arquiteturas pós-Transformer.

---

## 11. Liquid-S4

Em 2022/2023, Hasani et al. publicaram: **"Liquid Structural State-Space Models"**.

Esse trabalho conecta diretamente **LTC** e **S4**. O modelo resultante é chamado **Liquid-S4**.

O paper propõe uma versão linearizada de uma dinâmica LTC dentro da estrutura de um SSM.

A ideia pode ser visualizada como: `LTC + S4 → Liquid-S4`.

---

## 12. O que há de "líquido" no Liquid-S4?

Em um SSM tradicional: `ẋ = Ax + Bu`, a matriz `A` é essencialmente fixa durante a operação.

No conceito líquido, a dinâmica pode depender da entrada: **`A = A(u_t)`**.

Portanto: **dinâmica adaptativa** em vez de **dinâmica fixa**.

Essa é uma diferença conceitual extremamente importante. O Liquid-S4 procura preservar a eficiência estrutural dos SSMs enquanto introduz uma dinâmica dependente da entrada.

---

## 13. Resultados do Liquid-S4

O paper relata resultados no Long Range Arena (LRA) e em reconhecimento de fala.

O Liquid-S4 alcançou média reportada de aproximadamente **87,32%** no benchmark LRA e obteve **96,78%** no dataset completo de Speech Commands, com redução de aproximadamente 30% no número de parâmetros em comparação ao S4.

Mais importante que os números individuais é o princípio:

> "A dinâmica do sistema pode considerar as características do sinal de entrada durante a computação."

---

## 14. Liquid-S4 e linguagem

O código do projeto Liquid-S4 inclui experimentos de **language modeling no WikiText-103**.

Portanto: `Liquid → SSM → Language Modeling` não é apenas uma hipótese. Já houve experimentação explícita com modelagem autoregressiva de linguagem.

Entretanto, isso ainda é muito diferente de um LLM moderno de bilhões de parâmetros.

---

## 15. O nascimento dos Liquid Foundation Models

Em setembro de 2024, a **Liquid AI** anunciou sua primeira geração de Liquid Foundation Models.

A empresa descreveu os LFMs como modelos construídos a partir de unidades computacionais baseadas profundamente em:

- sistemas dinâmicos
- processamento de sinais
- álgebra linear numérica

A proposta era construir modelos gerais capazes de processar: `text + audio + video + time series + signals`.

A primeira geração incluía: **1.3B, 3.1B, 40.3B (MoE)**.

Isso representa uma mudança de escala histórica. A pesquisa deixa de perguntar "Será que uma rede líquida consegue aprender uma série temporal?" e passa a perguntar "Podemos construir um foundation model competitivo usando princípios de sistemas dinâmicos?".

---

## 16. A arquitetura não deve ser confundida com uma LTC gigante

Essa é provavelmente a maior fonte de confusão sobre o assunto.

Não é correto imaginar: `LFM = LTC × 10^9`.

A evolução arquitetural é muito mais complexa. Os LFMs utilizam operadores derivados de diferentes áreas matemáticas. A filosofia é: **selecionar estruturas matematicamente eficientes** em vez de simplesmente aumentar o tamanho de uma arquitetura conhecida.

---

## 17. LFM2: a arquitetura híbrida

O avanço mais importante documentado academicamente atualmente é o **LFM2 Technical Report**, publicado em 2025.

O LFM2 foi desenvolvido especificamente para eficiência em dispositivos. A família inclui modelos de: **350M, 700M, 1.2B, 2.6B** e um modelo MoE de **8.3B** parâmetros totais (aproximadamente **1.5B** parâmetros ativos). Todos os modelos utilizam contexto de até **32K** tokens.

---

## 18. A arquitetura do LFM2

O LFM2 utiliza um backbone híbrido constituído principalmente por:

- **Gated Short Convolutions**
- **Grouped Query Attention** (em menor quantidade)

Isso é extremamente importante. O resultado **não é uma LNN pura**. É: `convolução + estado + atenção`.

A arquitetura procura explorar diferentes mecanismos para diferentes tipos de dependência.

---

## 19. Por que manter atenção?

Porque atenção continua sendo extremamente boa para determinadas relações de longo alcance.

Considere: "João colocou o livro na mesa porque ele precisava sair." Resolver a referência de "ele" pode exigir relacionar elementos distantes da sequência. Um mecanismo de atenção é naturalmente adequado para encontrar essas relações.

Portanto, a pergunta moderna não é necessariamente `Attention OR SSM`. É: **`Attention AND efficient state dynamics`**.

---

## 20. A arquitetura híbrida

Podemos representar conceitualmente o LFM2 como:

```
x_t → [local/state processing] → [occasional global attention] → h_t
```

A maior parte do processamento pode utilizar operações mais eficientes, enquanto a atenção aparece em menor quantidade. Isso pode reduzir custo e memória sem eliminar completamente a capacidade de interações globais.

---

## 21. Hardware-in-the-loop Architecture Search

Outro aspecto extraordinariamente interessante do LFM2 é que os autores não fizeram apenas Neural Architecture Search abstrato. Eles utilizaram **Hardware-in-the-loop architecture search**, ou seja, a arquitetura é avaliada considerando diretamente as características do hardware-alvo.

Isso muda a função objetivo. Em vez de simplesmente `min L(θ)`, pode-se pensar em algo como:

```
min [L(θ) + λ₁C_latency + λ₂C_memory + λ₃C_energy]
```

A arquitetura deixa de ser otimizada apenas para "ser inteligente". Ela passa a ser otimizada para: **inteligência por unidade de recurso**.

---

## 22. Resultado de eficiência

O LFM2 reporta até aproximadamente **2× mais velocidade** de prefill e decode em CPUs, em determinadas comparações com modelos de tamanho semelhante.

Isso é importante porque a proposta da Liquid AI não é somente vencer benchmarks. É fazer o modelo funcionar em CPU, GPU, NPU, smartphones, sistemas embarcados, edge devices.

---

## 23. LFM2.5

Em janeiro de 2026, a Liquid AI lançou a família **LFM2.5**. A empresa informa que o pré-treinamento passou de **10T** para **28T** tokens. Além disso, a etapa de pós-treinamento foi ampliada com reinforcement learning.

Isso demonstra que a abordagem não ficou restrita à arquitetura. Ela passou a envolver: `architecture + pretraining + distillation + RL + hardware optimization`.

---

## 24. Raciocínio on-device

Em janeiro de 2026, a Liquid AI também anunciou o **LFM2.5-1.2B-Thinking**. O modelo foi projetado para raciocínio local e a empresa afirma que ele pode operar em menos de aproximadamente **900 MB** de memória em um telefone.

Isso demonstra uma consequência prática da filosofia: `modelo pequeno + arquitetura eficiente + pós-treinamento → capacidade significativa`.

---

## 25. LFM2.5-2.6B e agentes

Em agosto de 2026, a Liquid AI lançou o **LFM2.5-2.6B**, direcionado a workloads agentic. A empresa o descreve como capaz de executar localmente tarefas envolvendo:

- planejamento
- chamadas de ferramentas
- workflows de múltiplas etapas

Isso representa outra mudança importante: `Liquid architecture → LLM → agent`. O objetivo deixa de ser simplesmente gerar texto. É executar processos.

---

## 26. Linha temporal da pesquisa

| Ano | Trabalho | Principal contribuição |
|-----|----------|------------------------|
| 2018 | Liquid Time-Constant RNNs | Fundamentação inicial da capacidade de aproximação |
| 2021 | LTC | Redes recorrentes contínuas com constantes de tempo adaptativas |
| 2022 | CfC | Dinâmica contínua em forma fechada |
| 2022/2023 | Liquid-S4 | Integração entre liquid networks e SSMs |
| 2024 | LFM primeira geração | Fundação de modelos generativos baseada em sistemas dinâmicos |
| 2025 | LFM2 | Arquitetura híbrida otimizada para edge |
| 2026 | LFM2.5 | Escalonamento, RL, multimodalidade e agentes on-device |

A linhagem é: `ODE → LTC → CfC → SSM → Liquid-S4 → LFM → LFM2 → LFM2.5`.

---

## 27. Comparação conceitual

| Característica | Transformer | LTC | CfC | S4 | Liquid-S4 | LFM2 |
|---------------|-------------|-----|-----|-----|-----------|------|
| Tempo contínuo | Não | Sim | Sim | Não diretamente | Sim/inspirado | Indiretamente |
| Estado recorrente | Não tradicional | Sim | Sim | Sim | Sim | Sim |
| Atenção | Sim | Não | Não | Não | Não | Sim, limitada |
| Dinâmica dependente da entrada | Indiretamente | Sim | Sim | Limitada | Sim | Parcial/híbrida |
| Longas sequências | Boa | Possível | Boa | Excelente | Excelente | Boa |
| KV cache | Sim | Não | Não tradicional | Não | Não | Reduzido/próprio |
| Hardware edge | Difícil em grandes | Bom potencial | Muito bom | Bom | Bom | Objetivo central |
| Foundation model | Sim | Não originalmente | Não originalmente | Não originalmente | Não | Sim |

---

## 28. O verdadeiro problema: memória

Uma forma mais profunda de entender toda essa pesquisa é pensar em memória.

Um Transformer mantém informação contextual através de representações explícitas. Em um modelo autoregresso: `KV cache ≈ {K_1, V_1, ..., K_T, V_T}`. Isso significa que a memória cresce com T.

Em um sistema de estado: `x_t = f(x_{t-1}, u_t)`. A memória pode ser representada por um estado comprimido: **`x_t`**.

Então temos:

- **Transformer** → memória = conjunto de representações anteriores
- **SSM** → memória = estado comprimido
- **Liquid/Adaptive SSM** → memória = estado comprimido + dinâmica adaptativa

Esse é provavelmente o conceito mais importante de toda essa área.

---

## 29. O trade-off fundamental

Existe, porém, um problema. Se comprimimos toda a história `(x_1, ..., x_T)` em um único estado `h_T`, pode haver perda de informação. Matematicamente, essa transformação pode ser muitos-para-um.

Portanto: `h_T` não necessariamente contém toda a informação necessária para reconstruir cada detalhe do passado. Esse é um dos motivos pelos quais atenção continua sendo poderosa.

---

## 30. A hipótese híbrida

A solução mais interessante é:

```
compressão recorrente + acesso seletivo ao passado
```

ou:

```
estado eficiente + atenção ocasional
```

Essa é precisamente a direção arquitetural observada no LFM2. O modelo não tenta jogar fora tudo que o Transformer faz. Ele tenta descobrir: **Qual é a quantidade mínima de atenção necessária para preservar capacidade de modelagem enquanto o restante da computação utiliza mecanismos mais eficientes?**

---

## 31. Relação com Mamba

Uma comparação com Mamba é inevitável. Mamba também parte de State Space Models e procura construir um mecanismo eficiente para sequências longas.

Uma abstração simplificada é: `h_t = Ā_t h_{t-1} + B_t x_t`. O diferencial é que os parâmetros efetivos da dinâmica podem depender da entrada.

Isso cria uma família de modelos que também explora: **`state-space dynamics + input-dependent computation`**.

Portanto, **Liquid-S4 e Mamba pertencem a uma mesma grande revolução conceitual**: modelos de sequência baseados em estado, embora sejam arquiteturalmente diferentes.

---

## 32. Liquid versus Mamba

Uma distinção conceitual interessante:

- **Liquid**: motivação original nasce de dynamical systems + continuous-time neural networks
- **Mamba**: linhagem nasce de structured SSMs + selective state spaces

Ambos chegam a algo semelhante: **estado adaptativo**. Mas a matemática e a implementação não são idênticas.

Isso é importante para evitar o erro: "Liquid Neural Network é simplesmente Mamba." **Não é**.

---

## 33. O problema da escala

Existe uma pergunta ainda não completamente resolvida:

> Uma arquitetura líquida pode escalar tão bem quanto um Transformer?

Esse problema é diferente de provar que uma LNN funciona. Uma rede de 10^6 parâmetros e uma de 10^10 vivem em regimes completamente diferentes. É preciso estudar: scaling laws, estabilidade, otimização, paralelização, largura, profundidade, comprimento de contexto, qualidade dos dados, treinamento distribuído, eficiência de hardware.

---

## 34. O problema da paralelização

Transformers possuem uma vantagem enorme durante treinamento: os tokens de uma sequência podem ser processados paralelamente.

Uma RNN tradicional possui: `h_t = f(h_{t-1}, x_t)`. Portanto `h_t` depende de `h_{t-1}`. Isso cria uma dependência sequencial.

SSMs modernos desenvolveram técnicas para contornar parte desse problema, permitindo computação paralela durante treinamento. Esse é um dos motivos pelos quais a conexão **Liquid → SSM** é tão importante.

---

## 35. O papel da álgebra linear

Para escalar um modelo, a elegância da EDO não é suficiente. Precisamos transformar a matemática em operações eficientes de álgebra linear: `XW`, `AX`, `FFT(X)`, convoluções, kernels especializados, etc.

É por isso que a descrição da Liquid AI enfatiza não somente sistemas dinâmicos, mas também: **numerical linear algebra** e **signal processing**.

---

## 36. O papel do processamento de sinais

Uma sequência de tokens pode ser tratada como um sinal discreto: `x_1, x_2, ..., x_T`. Um operador de convolução causal pode ser escrito:

```
y_t = Σ_{k=0}^{K-1} w_k x_{t-k}
```

Isso fornece uma forma eficiente de capturar dependências locais. Portanto, a arquitetura pode separar **dependência local** de **dependência global**. É exatamente essa divisão que aparece de forma muito clara na arquitetura híbrida do LFM2.

---

## 37. LFM2 como sistema híbrido

Uma abstração útil seria:

```
x_t → C(x_t) [processamento local] → S(h_{t-1}, x_t) [estado] → A(h_t) [atenção seletiva] → h_t
```

O princípio é: **não usar a operação mais cara para tudo**, mas **usar cada mecanismo onde ele é mais eficiente**.

---

## 38. O que realmente caracteriza um "Liquid Foundation Model"?

Não existe uma definição acadêmica universal segundo a qual todo LFM precisa possuir uma LTC explícita. "Liquid Foundation Model" é uma família arquitetural da Liquid AI. A conexão com LNNs está na linhagem científica e nos princípios utilizados no projeto dos operadores.

Portanto: **`LNN ≠ LFM`**, mas **`LNN → ideias de dinâmica → Liquid architectures → LFM`** é uma descrição historicamente muito mais correta.

---

## 39. Evidência acadêmica versus marketing

É importante separar três níveis de evidência.

**Nível 1 — fundamentação acadêmica**: LTC, CfC, Liquid-S4. Esses trabalhos possuem artigos acadêmicos e formulações matemáticas explícitas.

**Nível 2 — technical reports**: LFM2 Technical Report. É uma publicação técnica extremamente relevante, mas não deve ser confundida automaticamente com um paper revisado por pares em conferência.

**Nível 3 — anúncios de produto**: LFM2.5, LFM2.5-Thinking, LFM2.5-2.6B. São informações importantes sobre o estado da tecnologia, mas são divulgadas pela própria empresa e devem ser tratadas como claims do fabricante quando não acompanhadas de validação independente.

---

## 40. Papers fundamentais

1. **Hasani et al. (2021)**. Liquid Time-constant Networks. AAAI. DOI: 10.1609/AAAI.V35I9.16936
2. **Hasani et al. (2022)**. Closed-form continuous-time neural networks. Nature Machine Intelligence 4, 992–1003. DOI: 10.1038/s42256-022-00556-7
3. **Hasani et al. (2022/2023)**. Liquid Structural State-Space Models. ICLR 2023 / arXiv:2209.12951
4. **Amini et al. (2025)**. LFM2 Technical Report. arXiv:2511.23404
5. **Liquid AI (2024)**. Liquid Foundation Models: Our First Series of Generative AI Models
6. **Liquid AI (2026)**. Introducing LFM2.5: The Next Generation of On-Device AI
7. **Liquid AI (2026)**. LFM2.5-1.2B-Thinking: On-Device Reasoning Under 1GB
8. **Liquid AI (2026)**. LFM2.5-2.6B: Deploy Agents Everywhere

---

## 41. Um paper que NÃO deve ser confundido

Existe um paper chamado **"Liquid: Language Models are Scalable Multi-modal Generators"** de Wu et al., 2024. Apesar do nome, ele não é sobre Liquid Neural Networks. O trabalho usa "Liquid" como nome de uma arquitetura multimodal e parte de LLMs como LLaMA/Gemma.

Portanto: **`Liquid Neural Networks ≠ "Liquid" multimodal model de Wu al.`**

---

## 42. Repositórios relevantes

Os autores das LTCs disponibilizaram código oficial para experimentação. O repositório inclui implementações de LTC, Neural ODEs e outros modelos contínuos. Também existe o código oficial de CfC, incluindo variantes que permitem comparar CfC com LTC e arquiteturas híbridas. O Liquid-S4 também possui implementação pública.

---

## 43. Estado da arte em setembro de 2026

O panorama atual pode ser resumido assim:

- **Transformer** continua sendo dominante em LLMs gerais
- Existe uma segunda grande linha: **SSM / recurrent / state-space architectures**
- E uma terceira direção: **hybrid architectures** que combinam attention + convolution + state dynamics

Os LFMs estão claramente nessa terceira categoria.

---

## 44. A hipótese científica central

A grande questão não é "LNNs conseguem gerar texto?" — isso já é uma pergunta pequena demais.

A questão realmente interessante é:

> Podemos construir modelos de linguagem cuja memória seja um sistema dinâmico aprendido?

Em outras palavras: `h_{t+1} = F(h_t, x_t)` em vez de depender predominantemente de `Attention(Q, K, V)`.

E mais profundamente:

> Podemos aprender a dinâmica da memória juntamente com o conhecimento?

---

## 45. Uma interpretação baseada em sistemas dinâmicos

Um Transformer pode ser visto, em primeira aproximação, como um sistema que recalcula relações entre elementos do contexto. Um modelo recorrente/SSM pode ser visto como: `input → state transition → state → output`.

O estado funciona como uma variável latente dinâmica: `x(t)`. Então um LLM poderia ser interpretado como um sistema dinâmico estocástico aprendido:

```
dh/dt = F_θ(h, x, t)
```

ou, na forma discreta: `h_{t+1} = F_θ(h_t, x_t)`.

Essa perspectiva conecta diretamente: machine learning, teoria de controle, sistemas dinâmicos, processamento de sinais, teoria da informação, neurociência computacional.

---

## 46. Uma hipótese ainda mais ousada

Considere a memória de um modelo como `M_t`. Em um Transformer: `M_t ≈ (K_1, V_1, ..., K_t, V_t)`. Em um modelo de estado: `M_t = h_t`.

Mas podemos imaginar: `M_t = (h_t, θ_t)`, onde até a dinâmica do sistema muda. Então:

```
dh/dt = F(h, x, θ_t)
dθ/dt = G(h, x)
```

Isso produziria uma espécie de **meta-dinâmica da memória** na qual não apenas o estado muda, mas também a forma como o sistema armazena informação. Isso ainda é uma questão de pesquisa, não uma característica estabelecida dos LFMs.

---

## 47. Onde as LNNs podem ser especialmente fortes

A vantagem potencial não é necessariamente superar Transformers em todos os benchmarks. Pode estar em:

1. **Edge AI**: pouca memória + baixa latência
2. **Sistemas contínuos**: robótica, sensores e controle
3. **Dados multimodais temporais**: áudio, sensores, vídeo e séries temporais
4. **Agentes persistentes**: sistemas que precisam manter estado continuamente
5. **Long-running AI**: agentes que permanecem ativos durante horas ou dias

Aqui a ideia de **estado persistente** fica muito mais natural do que uma KV cache gigantesca.

---

## 48. Um cenário particularmente interessante: agentes

Imagine um agente funcionando durante 24 horas. Um Transformer tradicional poderia acumular 10^6, 10^7, 10^8 tokens. Guardar todo esse contexto explicitamente torna-se caro. Um sistema de estado poderia manter `h_t` como representação comprimida da experiência.

Isso produz uma arquitetura potencialmente: **`agent = LLM + dynamical memory`**. Essa pode ser uma das aplicações mais interessantes das ideias liquid.

---

## 49. Mas existe um problema fundamental

Compressão implica perda. Se `(x_1, ..., x_T) → h_T`, então informações diferentes podem produzir estados semelhantes: `x^{(1)}_{1:T} ≠ x^{(2)}_{1:T}` mas `h_T^{(1)} ≈ h_T^{(2)}`.

Nesse caso, o sistema esqueceu uma distinção relevante. Portanto: **memória eficiente ≠ memória perfeita**. O desafio é encontrar a representação mínima que preserve aquilo que importa. Isso é essencialmente um problema de compressão de informação.

---

## 50. Conexão com teoria da informação

Podemos imaginar: `X_{1:T} → H_T`. Queremos maximizar `I(H_T; Y)` onde I é informação mútua e Y representa o que será necessário no futuro. Ao mesmo tempo, queremos minimizar `I(H_T; X_{1:T})` ou, mais precisamente, a quantidade de informação/memória necessária para preservar a tarefa.

Isso sugere uma interpretação extremamente interessante:

> Um bom mecanismo de memória não precisa lembrar tudo. Precisa lembrar aquilo que será útil.

Esse princípio aparece de diferentes formas em SSMs, memória neural, compressão e agentes.

---

## 51. A grande questão de pesquisa

Uma formulação possível seria: **Como aprender dinamicamente o que deve ser esquecido?**

Um sistema líquido já possui uma característica natural: `τ = τ(x, h)`. Isso significa que diferentes componentes podem responder em escalas temporais diferentes. Então podemos ter: `τ₁ << τ₂ << τ₃`.

Alguns estados mudam rapidamente. Outros permanecem por muito tempo. Isso produz naturalmente uma hierarquia temporal: **memória curta + memória média + memória longa**.

---

## 52. Uma arquitetura hipotética

Uma arquitetura futura poderia ter: `x_t → E(x_t)` seguido por três escalas:

```
h_t^{(short)},  h_t^{(medium)},  h_t^{(long)}
```

Com: `τ_s < τ_m < τ_l`.

O estado poderia evoluir como:

```
dh_s/dt = F_s(x, h_s)
dh_m/dt = F_m(x, h_m, h_s)
dh_l/dt = F_l(h_m, h_l)
```

Uma pequena quantidade de atenção poderia funcionar como mecanismo de recuperação: `r_t = Attention(q_t, H_memory)`.

Teríamos: **Liquid Memory + Selective Attention** em vez de **Full Attention Everywhere**.

---

## 53. Por que isso pode ser relevante para LLMs pequenos

Modelos pequenos são particularmente sensíveis a eficiência. Se o modelo possui 1B parâmetros, reduzir a memória e o custo de inferência pode ser mais importante do que adicionar centenas de milhões de parâmetros. Isso explica a estratégia da Liquid AI: **maximize capability per parameter** e **maximize capability per watt** em vez de simplesmente **maximize parameter count**.

---

## 54. Evidências atuais da direção

A família LFM2 já cobre modelos de centenas de milhões a bilhões de parâmetros e foi explicitamente desenvolvida visando deployment on-device. A família atual da Liquid AI também inclui modelos de texto, visão-linguagem e áudio. Em 2026, a empresa continuou ampliando a linha com modelos voltados a raciocínio e agentes.

---

## 55. Mas não devemos concluir que "Transformers morreram"

Isso seria uma conclusão errada. Transformers continuam extremamente fortes porque possuem: excelente paralelização, ecossistema gigantesco, hardware altamente otimizado, enorme quantidade de pesquisa, escalabilidade demonstrada, capacidade de modelar relações complexas.

A questão científica é outra: **Existe uma arquitetura melhor para cada regime de computação?** Provavelmente sim.

---

## 56. O futuro provavelmente será híbrido

A tendência observada aponta para: **Attention + SSM + Convolution + Recurrence** em diferentes proporções.

Não necessariamente haverá um único vencedor. É possível que tenhamos arquiteturas especializadas para: cloud, desktop, smartphone, wearable, robótica, veículos, agentes persistentes, multimodalidade.

---

## 57. Mapa conceitual completo

```
                    NEURAL ODEs
                        │
                        ▼
             CONTINUOUS-TIME RNNs
                        │
                        ▼
              LIQUID TIME-CONSTANT
                   NETWORKS
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
            CfC                  LTC theory
             │
             ▼
      EFFICIENT CONTINUOUS
          DYNAMICS
             │
             ▼
       STATE SPACE MODELS
             │
       ┌─────┴─────┐
       │           │
       ▼           ▼
      S4         Liquid-S4
       │           │
       └─────┬─────┘
             │
             ▼
     LONG-SEQUENCE MODELING
             │
             ▼
   LIQUID FOUNDATION MODELS
             │
             ▼
          LFM1
             │
             ▼
          LFM2
             │
             ▼
         LFM2.5
             │
       ┌─────┼─────┐
       │     │     │
       ▼     ▼     ▼
     TEXT   VL    AUDIO
       │
       ▼
    REASONING
       │
       ▼
     AGENTS
```

---

## 58. Conclusão

A história das Liquid Neural Networks não é simplesmente `LNN → LLM`. A história real é mais interessante:

```
Sistemas dinâmicos → Redes neurais contínuas → LTC → CfC → State Space Models → Liquid-S4 → Liquid Foundation Models → modelos de linguagem eficientes
```

O ponto central não é substituir o Transformer por uma "LNN gigante". O ponto é descobrir como representar memória e processamento sequencial de maneira mais eficiente.

A ideia mais poderosa da abordagem líquida é que o estado não precisa possuir uma única escala temporal. Podemos ter `τ₁, τ₂, ..., τ_n` adaptando-se à dinâmica da entrada.

Em uma interpretação de sistemas dinâmicos: **inteligência ≈ dinâmica adaptativa da representação**.

---

## 59. Questões de pesquisa em aberto

1. **Escalabilidade**: Como as leis de escala de LNN/SSM se comparam às dos Transformers?
2. **Memória**: Qual é a capacidade máxima de informação de um estado recorrente comprimido?
3. **Forgetting**: Como aprender automaticamente o que deve ser esquecido?
4. **Multi-timescale memory**: Podemos aprender uma distribuição ótima de constantes de tempo?
5. **Attention budget**: Qual é a quantidade mínima de atenção necessária para preservar desempenho?
6. **Continual learning**: Uma dinâmica líquida permite aprendizado contínuo mais natural?
7. **Agent memory**: Um estado dinâmico pode substituir parte da memória externa dos agentes?
8. **Hardware**: Qual arquitetura é ótima para cada combinação CPU/NPU/GPU?
9. **Interpretabilidade**: Podemos interpretar os estados como variáveis dinâmicas semanticamente significativas?
10. **Física**: Podemos incorporar invariantes, conservação e estabilidade diretamente na arquitetura?

---

## 60. Bibliografia essencial

1. Hasani, R.; Lechner, M.; Amini, A.; Rus, D.; Grosu, R. Liquid Time-constant Networks. AAAI, 2021.
2. Hasani, R.; Lechner, M.; Amini, A.; Liebenwein, L.; Ray, A.; Tschaikowski, M.; Teschl, G.; Rus, D. Closed-form continuous-time neural networks. Nature Machine Intelligence, 2022.
3. Hasani et al. Liquid Structural State-Space Models. ICLR 2023 / arXiv:2209.12951.
4. Amini et al. LFM2 Technical Report. arXiv:2511.23404, 2025.
5. Liquid AI (2024). Liquid Foundation Models: Our First Series of Generative AI Models.
6. Liquid AI (2026). Introducing LFM2.5: The Next Generation of On-Device AI.
7. Liquid AI (2026). LFM2.5-1.2B-Thinking: On-Device Reasoning Under 1GB.
8. Liquid AI (2026). LFM2.5-2.6B: Deploy Agents Everywhere.

---

## 61. Ordem recomendada de estudo

1. RNNs → 2. Sistemas dinâmicos → 3. Neural ODEs → 4. LTC → 5. CfC → 6. State Space Models → 7. S4 → 8. Liquid-S4 → 9. Mamba / selective SSMs → 10. Transformers → 11. Hybrid architectures → 12. LFM2 → 13. LFM2.5

---

## 62. Síntese final

A pergunta inicial era: "Existem aplicações de LNNs para LLMs?"

A resposta científica mais precisa é: **Sim. Mas a aplicação mais significativa evoluiu de LNNs puras para uma família mais ampla de arquiteturas de processamento sequencial baseadas em dinâmica, estados e operadores eficientes.**

A contribuição fundamental das LNNs foi introduzir a ideia de que a dinâmica interna da rede pode ser adaptativa no tempo. A contribuição dos CfCs foi tornar essa dinâmica mais computacionalmente prática. A contribuição dos Liquid-S4 foi aproximar essa ideia dos modernos State Space Models. E a contribuição dos Liquid Foundation Models é tentar transformar esses princípios em foundation models reais.

A direção mais promissora, portanto, parece menos `LNN vs. Transformer` e mais:

```
Dynamic State + Selective Attention + Efficient Operators
```

Essa formulação abre uma questão de pesquisa muito mais profunda:

> Como construir uma memória neural que aprenda não apenas o que lembrar, mas também por quanto tempo cada informação deve permanecer relevante?

Essa é precisamente a região conceitual em que sistemas dinâmicos, LNNs, SSMs e LLMs começam a convergir.

---

*Documento recebido e integrado ao Projeto VISÃO em 10/09/2026*
