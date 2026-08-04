# Fase 3 — Auto-Melhoria Darwiniana

> Camada de evolução do próprio código do Projeto VISÃO. **Anexo ao PLANO.md.**
> Estado: desenho (não implementado). Depende da contenção (L4) já testada e
> do agente autocontido (L6) para carregar variantes.

---

## Objetivo

O sistema modifica o próprio código e **provou empiricamente** que a mudança
foi benéfica — antes de promovê-la. Não é "LLM reescreve a si mesma e
esperamos o melhor"; é seleção com fitness medida em sandbox.

---

## Precedente que funciona (Sakana AI, Darwin Gödel Machine, arXiv:2505.22954)

- SWE-bench: **20,0% → 50,0%** por auto-modificação.
- Polyglot: **14,2% → 30,7%**.
- Ablação decisiva: sem arquivo aberto de variantes, o desempenho **cai**.
  Ancestrais "ruins" geram descendentes ótimos. Hill-climbing puro converte
  cedo demais.

---

## 1. Arquivo de variantes (3.1)

`visao/evolve/archive.py` — grafo de linhagem; cada nó =
`(código, pesos, fitness, pai)`.

**Regra invariável:** *nunca podar por fitness baixa.* Manter os becos sem
saída — é de onde saem os saltos. O arquivo é o "DNA" evolutivo da VISÃO.

---

## 2. Loop de mutação (3.2)

`visao/evolve/mutate.py`.

- **Permitidas:** topologia de wiring, faixa de $\tau$, hiperparâmetros de
  plasticidade, código das regras locais.
- **Proibidas (R3):** camada de governança, política de rede, chaves
  criptográficas. O `containment.py` é imutável e **fora do espaço de mutação**
  — verificado a cada ciclo (R3, já implementado e testado com 39 testes).

---

## 3. Fitness empírica em sandbox (3.3)

`visao/evolve/sandbox.py` — gVisor ou Firecracker, sem rede, sem
persistência, CPU/RAM limitados. **Toda variante roda primeiro aqui. Sempre.**
Sem escapada de sandbox = critério de portão.

---

## 4. Teste de transferência (3.4)

Uma melhoria só é aceita se generaliza para tarefas **fora** do conjunto de
otimização. O DGM validou: melhorias em Python transferiram para Rust/C++/Go.
Sem este teste, o sistema aprende a trapacear o benchmark.

---

## Portão da Fase 3

- Três gerações consecutivas com ganho **transferível**.
- **Nenhuma** escapada de sandbox.
- Nenhuma mutação em `containment.py` (R3 intacto).

---

## Por que isto é seguro (e não um Gõdel descontrolado)

A auto-melhoria opera **dentro** da contenção, não acima dela:

- O arquivo de governança é imutável (R3) e verificado a cada ciclo.
- A variante só promovida após sandbox + transferência — sem efeito em
  produção até provar benefício.
- Replicação de variantes ainda exige R1 + R4 (Fase 2.4).

O Darwin Gödel Machine melhora código; a VISÃO melhora *células líquidas*, sob
freio. Diferente de "AGI que se auto-reescreve".

---

## Sub-tarefas (Fase 3)

- **3.1** Arquivo de variantes — `visao/evolve/archive.py` (grafo de linhagem).
- **3.2** Loop de mutação — `visao/evolve/mutate.py` (permitidas/proibidas).
- **3.3** Sandbox de fitness — `visao/evolve/sandbox.py` (gVisor/Firecracker).
- **3.4** Teste de transferência — generalização fora do conjunto de otimização.
