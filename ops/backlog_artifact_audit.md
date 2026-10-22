# Auditoria de artefatos declarados x disco — 21/10/2026

**Por que existe:** o `BACKLOG.json` é declarado "fonte da verdade para o loop autônomo".
Esta auditoria verifica se os arquivos que as tarefas `done` declaram como entrega
realmente existem em disco. **Não é heartbeat — é reconciliação de integridade**.

## Método (reproduzível)

```bash
cd ~/projeto-visao
python3 ops/audit_backlog_artifacts.py
```

## Resultado CORRETO (21/10/2026)

|| métrica | valor |
|---|---|---|
| arquivos declarados em `files` | **90** |
| **PRESENT** (existe no path declarado) | **32** |
| **RELOCATED** (movido, existe em outro lugar) | **6** |
| **MISSING** (ausência real) | **52** |

## O que mudou em relação ao relatório anterior (01:12)

O relatório antigo reportava **58 de 89** artefatos como ausentes. Esse numero
era **falso**. A causa raiz:

    git log --all --diff-filter=D -- <path>

Sem `-M` e sem `-c` (combined diff) o git nao entende RENAMES. Um arquivo que foi
MOVIDO (renomeado) aparece como `D` no path antigo, e o script concluia "este
artefato foi apagado por <commit>". Na verdade o conteudo so mudou de lugar.

Prova definitiva: os 4 docs que o relatorio antigo culpava o commit 6af0c10 —
AGENTE_AUTOCONTIDO.md, FASE2_ENXAME.md, FASE3_EVOLUCAO.md, PROPOSTA_PESQUISA.md —
existem HOJE em `docs/` (git ls-files confirma: TRACKED). O commit 6af0c10 os
renomeou (similaridade 100%):

    $ git show -M --name-status 6af0c10 | grep R100
    R100  PROPOSTA_PESQUISA.md  docs/PROPOSTA_PESQUISA.md

Esta versao classifica cada path declarado em TRES categorias:
  - PRESENT   : existe no path declarado
  - RELOCATED : nao existe no path declarado, MAS o conteudo existe em outro lugar
    do repo (rename detectado com -M, ou basename encontrada)
  - MISSING   : nao existe em lugar nenhum (ausencia real)

O que interessa ao loop autonomo e' MISSING — evidencia genuinamente perdida.

### RELOCATED (6 paths) — o conteudo mudou de lugar

  - AGENTE_AUTOCONTIDO.md  ->  docs/AGENTE_AUTOCONTIDO.md  (via 6af0c1017)
  - FASE2_ENXAME.md        ->  docs/FASE2_ENXAME.md        (via 6af0c1017)
  - FASE3_EVOLUCAO.md      ->  docs/FASE3_EVOLUCAO.md      (via 6af0c1017)
  - PROPOSTA_PESQUISA.md   ->  docs/PROPOSTA_PESQUISA.md   (via 6af0c1017)
  - visao/bench/baselines.py ->  research/models/baselines.py (basename)
  - visao/env/requirements.txt ->  requirements.txt (basename)

### MISSING (52) — evidencia genuinamente perdida

24 removidas por `300f53f chore: remove obsolete tests and add missing __init__.py`
24 removidas por `dbba6b6 chore: limpeza de dead code e diretórios não utilizados`
2 removidas por `f06156a chore: remove duplicate analysis files and old results`
2 removidas por `6602ecd fix: integração VisaoCognitiveBrain com APIs reais`
1 arquivo nunca versionado (sem rastro no git)

### Referências de import a modulos ausentes

(nenhuma — varredura de imports em visao/ nao encontra referencias aos modulos
ausentes)

## Decisao pendente (humana — Artigo 01 do Código Humanitas)

Não se rebaixa nenhuma tarefa `done` sem decisão explícita do Juan. Opções:

- **(a)** Restaurar os artefatos a partir do git (`git checkout <commit> -- <path>`) e
  re-selar o que for necessario;
- **(b)** Emendar o BACKLOG marcando as tarefas afetadas como `done_artifact_removed`
  (honesto: o trabalho existiu, a evidência não), mantendo o historico;
- **(c)** Aceitar que o BACKLOG é um registro histórico e criar um BACKLOG novo so com
  o que existe em disco.

Registrado como tarefa bloqueada `79` no `BACKLOG.json`. O script de auditoria
corrigido fica em `ops/audit_backlog_artifacts.py`.