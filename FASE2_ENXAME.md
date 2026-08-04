# Fase 2 — O Enxame Descentralizado

> Camada de coordenação P2P do Projeto VISÃO. **Anexo ao PLANO.md.**
> Estado: desenho (não implementado). A Fase 6 (agente autocontido) é
> pré-requisito de execução — sem o bundle, não há o que rodar no voluntário.

---

## Objetivo

Permitir que máquinas voluntárias entrem na rede, contribuam com computação
para o treino do modelo líquido, e saiam — **sem servidor central, sem
confiança cega, e sob as regras R1–R5 de contenção**.

---

## Princípios (herdados)

- **Admissão ≠ replicação.** Entrar como worker é livre (após auto-verificação
  R3). Spawnar filho (crescer a rede) exige R1 + R4.
- **Banda é o gargalo.** Usar DisTrO/DiLoCo: sincronia a cada $K$ passos, não
  a cada passo → ~$10^4\times$ menos tráfego que all-reduce centralizado.
- **Nós são hostis por padrão.** Agregação robusta a Byzantinos; prova de
  contribuição para afastar free-rider e poison.

---

## 1. Onboarding (a máquina entra)

1. Voluntário baixa o bundle autocontido (Fase 6).
2. No `launch`, o bundle **se auto-verifica** (R3): hash de
   `visao/governance/containment.py` vs. manifesto selado. Diverge → aborta.
3. Gera par `ed25519` → `node_id` estável.
4. Contata nó bootstrap (lista fixa ou DHT) apresentando **atestado de
   capacidade** assinado (CPU/RAM/janela de disponibilidade).
5. Bootstrap registra no **lineage ledger** (R5) e insere o nó na *ring*.

---

## 2. Transporte (como falam)

`libp2p` ou `Iroh` (Rust). Cada mensagem assinada por `node_id`:
- descoberta (mDNS/DHT),
- NAT traversal (relay),
- autenticação ponta-a-ponta sem servidor central.

---

## 3. Treino descentralizado (como aprendem juntos)

- Modelo global $\theta$ fragmentado (sharding) ou replicado por conjunto.
- Cada nó: $K$ passos locais nos **seus** dados → $\delta_i = \theta_i - \theta$.
- Compressão: top-k + quantização; *gossip* só do diff.
- Agregação: **trimmed mean / Krum** (robusta a $\le 1/3$ nós maliciosos).
- Sincronia a cada $K$ passos → ~$10^4\times$ menos banda (DisTrO).

---

## 4. Prova de contribuição (sem free-rider, sem poison)

- Nó compromete *commitment* ao gradiente antes do passo local.
- Após o passo: envia gradiente + prova de que executou $K$ passos
  (spot-check ou prova leve).
- Sem prova → fora da agregação; falha registrada no ledger (R5).

---

## 5. Crescimento (como a rede se expande)

A rede **não** se replica sozinha:

```
N* = entrada / churn          (recruitment_model.py)
se N < alvo:  recrutamento (Fase 5) atrai voluntários
se um nó quer spawnar filho:  exige R1 (quórum humano) + R4 (PoW 2^(8+1.6c))
```

Humanos aprovam expansão. É o freio que impede botnet.

---

## 6. Churn (voluntário desliga)

Nó sai → rede encolhe → recrutamento contínuo repõe ($N^*$ estável). Ledger
(R5) expulsa nó silencioso; shard reconstruído por redundância dos pares.

---

## Riscos de engenharia (declarados)

- **Byzantine:** Krum resiste a $\le 1/3$ maliciosos; acima, modelo corrompe.
- **Prova de computo leve e à Prova:** pesquisa aberta — equilíbrio entre
  barato p/ honesto e caro p/ mentiroso.
- **Sharding de modelo líquido** em $10^5$ neurônios: não feito (Fase 1 nem lá).

---

## Sub-tarefas (Fase 2)

- **2.1** Transporte P2P — bootstrap/libp2p + handshake assinado (R3 no join).
- **2.2** Agregação robusta — trimmed mean/Krum sobre deltas comprimidos.
- **2.3** Prova de contribuição — commitment + spot-check de $K$ passos.
- **2.4** Admissão de filho — orquestra R1 (quórum) + R4 (PoW) no spawn.
- **2.5** Churn + ledger — expulsão de nó silencioso, reconstrução de shard.
- **2.6** Sharding do modelo líquido — fragmentar `CfCCell` em $10^5$ neurônios.
