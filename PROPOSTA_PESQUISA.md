# Projeto VISÃO — Proposta de Pesquisa

**Título de trabalho:** *Neuroplasticidade local sem backpropagation como substrato
para aprendizado contínuo em rede descentralizada e auto-replicante, sob contenção
criptográfica verificável.*

**Status:** protótipo validado em escala de laboratório (Fase 0). Proposta em
refinamento para submissão em workshop de *AI Alignment & Safety* ou *NeurIPS
Workshop on Machine Learning for Systems*.

**Autoria:** Juan Guerra (investigador principal) · assistente: ÍRIS (Hermes).

**Palavras-chave:** redes neurais líquidas, aprendizado contínuo, plasticidade
local, computação descentralizada, governança de IA auto-replicante.

---

## 1. Motivação e Lacuna

Três limitações dos paradigmas dominantes motivam este trabalho:

1. **Esquecimento catastrófico.** Redes treinadas por backpropagation armazenam
   conhecimento em pesos globais compartilhados; aprender a tarefa $B$ sobrescreve
   a representação de $A$ (McCloskey & Cohen, 1989; French, 1999).
2. **Dependência de infraestrutura centralizada.** O treino de modelos de fronteira
   exige clusters com orçamento proibitivo, criando um gargalo de acesso e um ponto
   único de controle — e de falha.
3. **Risco de auto-replicação sem freio.** Modelos de fronteira já alcançam 70–81%
   de sucesso em auto-replicação end-to-end em ambientes reais (Pan et al., 2024;
   Palisade Research, 2026). Capacidade de auto-replicação deixou de ser fronteira.

Nenhuma dessas três questões é abordada por um único corpo de trabalho. Este
projeto propõe um substrato — **arquitetura líquida com plasticidade local** —
que ataca (1) por construção, e um protocolo de **contenção** que enclausura (3)
antes que (2) seja escalado.

---

## 2. Hipótese Central

> **H1.** Uma rede de constantes de tempo adaptativas (LTC/CfC) atualizada por
> regras locais de Hebb/Oja — sem retropropagação do gradiente — apresenta
> esquecimento médio em aprendizado contínuo **estatisticamente inferior** ao de
> um controlador de mesmo tamanho treinado por backpropagation, sob a mesma
> sequência de tarefas e sem replay de dados antigos.

A hipótese foi **parcialmente sustentada em escala reduzida** (Fase 0, $n=5$
seeds, 96 neurônios): redução de 97,2% no esquecimento. O refinamento científico
exigido é estender a $10^4$–$10^5$ neurônios, múltiplos baselines públicos, e
controle de variáveis.

**Hipóteses secundárias:**

> **H2.** A plasticidade da constante de tempo efetiva ($\tau_{\text{ef}}$) correlaciona
> positivamente com a retenção em tarefas sequenciais — i.e., a "liquidez" não é
> ornamento, é mecanismo.

> **H3.** Um protocolo de replicação com quórum humano + prova-de-trabalho (PoW)
> crescente mantém a taxa de crescimento da rede abaixo de um teto seguro sem
> intervenção humana contínua, mesmo sob incentivo de expansão.

---

## 3. Estado da Arte e Posicionamento

| Corpo de trabalho | Contribuição | O que este projeto adiciona |
|---|---|---|
| LTC/CfC (Hasani et al., *Nature MI* 2022; Lechner et al., arXiv:2006.04439) | Dinâmica contínua estável e expressiva | Plasticidade **local** sem backprop; não apenas inferência |
| `ncps` (Lechner) | Fiação esparsa tipo NCP | Topologia evoluída por arquivo darwiniano |
| DisTrO/DiLoCo (Prime Intellect; Douillard et al., 2023) | Treino descentralizado com 10⁴× menos banda | Acoplamento a contenção criptográfica |
| DGM / SEAL (OpenAI, 2024–25) | Auto-modificação com melhoria empírica | Freio de governança *antes* da auto-modificação |
| Containment de IA (RAND, 2025; Pan et al., 2024) | Caso para contenção | Implementação *auditável e testada* (39 testes) |

O diferencial é a **ordem**: contenção (L4) construída e testada antes da
auto-melhoria (L3) e da escala (L1/L2) — invertendo a prática comum de "deploy
primeiro, governança depois".

---

## 4. Desenho Experimental

### 4.1 Substrato (H1, H2)

- **Modelo.** Célula CfC em JAX (`visao/core/cfc.py`), `jit` + `lax.scan`.
  Projeção verificada contra a implementação NumPy com erro máximo
  $5{,}55\times10^{-16}$ (paridade numérica, `visao/tests/test_parity.py`).
- **Tarefas.** *Permuted MNIST* e *Split CIFAR-100* (padrão de continual
  learning) + série temporal de *physionet* (domínio natural da arquitetura
  líquida). Avaliação também em *Continual-Learning-Benchmark* (GT-RIPL).
- **Baselines.**
  - (a) CfC idêntica treinada por backprop (BPTT).
  - (b) LSTM / GRU (controle recorrente padrão).
  - (c) Reservatório LTC **sem** plasticidade local (delta puro) — o controle
    interno da Fase 0.
  - (d) EWC / GEM (métodos de continual learning por regularização).
- **Métricas.** Esquecimento médio ($\Delta$), acurácia final, área sob a
  curva de retenção; variância sobre $\ge 10$ seeds.
- **Controle de variáveis.** mesma topologia NCP (`AutoNCP`), mesmo orçamento
  de passos, mesma semente de dados; única diferença = regra de atualização.

### 4.2 Estabilidade e liquidez (H2)

- `network/diagnose_liquid.py` (tool do projeto) mede: faixa de $\tau_{\text{ef}}$
  calmo vs. agitado (razão de liquidez), raio espectral de $W_{\text{rec}}$,
  expoente de Lyapunov empírico, saúde do reservatório. Critério de rejeição:
  Lyapunov $> 0$ (caótico) ou raio espectral $> 10$ (fora de escala).

### 4.3 Governança (H3)

- Simulação de Monte Carlo do `ReplicationThrottle` (`visao/governance`).
  Variáveis: geração, número de filhos, poder de processamento. Métrica:
  tempo até o teto de dificuldade (32 bits = $2^{32}$ hashes) sob incentivo
  maximizante de replicação. Hipótese: nenhuma trajetória atinge expansão
  descontrolada em janela de 1 ano de tempo de máquina.

---

## 5. Resultados Preliminares (Fase 0)

Protótipo NumPy, 96 neurônios, 5 seeds, 3 tarefas temporais sequenciais, sem
replay:

| Condição | Esquecimento | Erro final |
|---|---:|---:|
| Líquido + plasticidade local | $+0{,}0083$ (±0,0138) | 0,2868 |
| Controle (delta puro) | $+0{,}2924$ (±0,1009) | 0,5284 |

Redução de **97,2%** no esquecimento. Liquidez: faixa de $\tau_{\text{ef}}$
**13×** maior sob entrada agitada que calma.

**Limitação declarada:** $n=5$ seeds e 96 neurônios são insuficientes para
inferência estatística robusta; a Fase 1 eleva para $n\ge 10$ e $10^4$–$10^5$
neurônios antes de qualquer afirmação de significância.

---

## 6. Contribuições Esperadas

1. Evidência empírica de que plasticidade local *sem backprop* reduz
   esquecimento em regime de continual learning em escala.
2. Um protocolo de contenção para IA auto-replicante **implementado e testado**
   (não apenas proposto), com 39 testes de violação.
3. Um modelo de recrutamento voluntário baseado em dados (não em retórica),
   com equilibro previsto $N^* = \text{entrada}/\text{churn}$.

---

## 7. Limitações e Riscos

- **Escala.** A Fase 0 é de laboratório; generalização para $10^5$ neurônios
  e dados reais é incerta e é o foco da Fase 1.
- **Ausência de baseline público direto.** Nenhum trabalho combina LTC + plasticidade
  local + conteção; a comparação é contra ablações internas e métodos canônicos.
- **Hipótese H3 não validada empiricamente** — só simulada. Requer implantação
  real para confirmação.
- **Risco de contenção ser contornada** por evolução do próprio agente (L3) —
  mitigado por R3 (arquivo de governança imutável, fora do espaço de mutação).

---

## 8. Cronograma e Marcos

| Fase | Escopo | Marco de validação |
|---|---|---|
| 1 | CfC em JAX, 10⁴–10⁵ neurônios, baselines | Paper interno + reprodução de H1 em escala |
| 2 | Enxame DisTrO/DiLoCo + verificação cripto | Treino distribuído sem queda de fitness |
| 3 | Arquivo darwiniano + fitness empírica | Auto-melhoria sob R1–R5, sem violação |
| 4 | Replicação supervisionada (R4) | Expansão respeita teto em simulação |
| 5 | Rede humana (recrutamento) | $N^*$ estável $> 10^3$ nós |

Cada marco tem **critério de parada explícito** e é auditável por testes, não
por narrativa.

---

## 9. Reprodutibilidade

- Código: `/home/juan/projeto-visao` (git, 73 testes automatizados).
- Ambiente: `visao/env/requirements.txt` (JAX, ncps, diffrax, optax, pytest).
- Sementes fixas; resultados em JSON (`prototype/results_continual.json`).
- Governança: `visao/governance/` — 39 testes de violação, arquivo selado
  por hash (`seal.py`).

---

## 10. Declaração de Segurança

Este projeto trata de capacidade de auto-replicação **como objeto de estudo de
contenção**, não como objetivo de desbloqueio. A camada de governança (R1–R5)
foi deliberada e primariamente construída para **limitar** a própria capacidade
do sistema. Nenhuma expansão de rede ocorre sem quórum humano assinado
(R1) e prova de trabalho crescente (R4). O manifesto de contenção é imutável
(R3) e verificado a cada ciclo.

---

## Referências Selecionadas

- Hasani, R. et al. *Liquid Time-constant Networks*. Nature MI 4, 992–1003 (2022).
- Lechner, M. et al. *Neural Circuit Policies* (arXiv:1803.08554); `ncps` (GitHub).
- Douillard, A. et al. *DiLoCo* (arXiv:2311.08105, 2023).
- Pan, A. et al. *Open-weight models and self-replication* (2024).
- Palisade Research. *AI self-exfiltration* (2026).
- RAND. *Strengthening Emergency Preparedness… for AI* (RRA3847-1, 2025).
- McCloskey, M. & Cohen, N. *Catastrophic interference* (1989).
- GT-RIPL. *Continual-Learning-Benchmark* (GitHub).
