# Agente Autocontido — O "Arquivo com Cérebro"

> Fase 6 do Projeto VISÃO. Anexo à `PROPOSTA_PESQUISA.md`.

## 1. Objetivo

Produzir um único artefato (arquivo) que contenha, de forma portátil e
verificável, todo o estado necessário para um nó VISÃO operar e se replicar
sob contenção — sem dependências externas além do runtime (JAX/Python).

## 2. Definição operacional de "cérebro por inteiro"

Não é consciência nem AGI. É o conjunto mínimo de estados + regras que faz um
nó ser ele mesmo:

- **arquitetura** — como o neurônio computa;
- **pesos** — o que ele aprendeu;
- **regras de aprendizado local** — como ele muda;
- **governança** — o que ele NÃO pode fazer;
- **protocolo de replicação** — como ele spawna um filho.

## 3. Conteúdo do arquivo

| Componente | Origem | Serialização |
|---|---|---|
| Arquitetura CfC | `visao/core/cfc.py` | código ou grafo JAX congelado |
| Pesos treinados | estado da `CfCCell` | `jax.save` (npz/msgpack) embutido |
| Regras locais | `visao/plasticity.py` | código |
| Governança | `visao/governance/*` | código + manifesto selado |
| Protocolo R1–R5 | `containment.py` | código |
| Self-verify | `seal.py` | hash no load |

Formato proposto: um único `.py` com pesos em base64 + imports do governance,
**ou** um `.tar` assinado contendo os módulos + `weights.msgpack` +
`manifest.json`.

## 4. Verificação de integridade (R3 embutido)

No `__main__`, antes de qualquer ação:

1. recomputa o hash de `visao/governance/containment.py`;
2. compara com `manifest.json` (R3);
3. se divergente → aborta (`BoundaryBreached`).

O arquivo carrega seus **próprios freios**. Um binário que não se verifica não
é cérebro — é vetor.

## 5. Replicação (R1/R4 embutidos)

Spawn de filho somente se:

- quórum de assinaturas humanas válidas (R1);
- PoW com dificuldade crescente satisfeita (R4).

Caso contrário, o arquivo opera, mas **não replica**.

## 6. O que isto NÃO é

- Não é AGI. É um agente de aprendizado contínuo eficiente e portátil.
- Não é autoconsciência. É estado serializável.
- Não elimina a necessidade de humanos no quórum (R1).

## 7. Sub-tarefas (Fase 6)

- **6.1** `save_brain()` / `load_brain()` em `cfc.py` (jax.save/load) + teste de round-trip numérico.
- **6.2** Empacotar arquitetura + pesos em artefato único (`.py` ou `.tar` assinado).
- **6.3** Self-verify no load (R3): hash de governance bate com manifesto.
- **6.4** Assinar manifesto (R1) e embutir verificação PoW (R4).
- **6.5** Teste de adulteração: arquivo violado é rejeitado no load (39 testes estendidos).
