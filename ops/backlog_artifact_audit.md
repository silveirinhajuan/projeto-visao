# Auditoria de artefatos declarados × disco — 21/10/2026

**Por que existe:** o `BACKLOG.json` é declarado "fonte da verdade para o loop autônomo".
Esta auditoria verifica se os arquivos que as tarefas `done` declaram como entrega
realmente existem em disco. **Não é heartbeat — é reconciliação de integridade.**

## Método (reproduzível)

```bash
cd ~/projeto-visao
python3 ~/.hermes/cache/scratch/audit_deletions.py
```

O script lê `BACKLOG.json`, coleta todos os campos `files` das tarefas, checa
`os.path.exists` para cada um e, para os ausentes, pergunta ao git qual commit os removeu.

## Resultado

| métrica | valor |
|---|---|
| arquivos declarados em `files` | **89** |
| presentes em disco | **31** |
| **AUSENTES** | **58 (65%)** |

### Os 58 ausentes, por commit que os removeu

| commit | data | arquivos removidos declarados no BACKLOG |
|---|---|---|
| `dbba6b6` "chore: limpeza de dead code e diretórios não utilizados" | 2026-09-10 | **25** |
| `300f53f` "chore: remove obsolete tests and add missing __init__.py" | — | **24** |
| `6af0c10` "feat: benchmark SOTA (12.0)" | — | **4** |
| `6602ecd` "fix: integração VisaoCognitiveBrain com APIs reais" | — | **2** |
| `f06156a` "chore: remove duplicate analysis files and old results" | — | **2** |
| nunca versionados | — | **1** |

### O que foi apagado (tarefas `done` que os declaram)

**Pacotes inteiros:** `visao/evolve/` (archive, mutate, sandbox, transfer, automl),
`visao/env/`, `visao/swarm/`, `network/`.

**Núcleo:** `visao/core/wiring.py` (1.4), `visao/core/predictive.py` +
`predictive_coder.py` (1.6), `visao/core/bundle.py` (6.2/6.3/6.4).

**Análise:** `visao/analysis/ablation.py` + `results_ablation.json` (1.9),
`tau_sweep.py` (1.10), `scale_ltc.py` (5.5), `routine_continual.py` (5.6),
`ablation_2.py` (7.3).

**Docs:** `PROPOSTA_PESQUISA.md`, `AGENTE_AUTOCONTIDO.md` (0.2),
`FASE2_ENXAME.md`, `FASE3_EVOLUCAO.md`.

**Ops:** `ops/sentinel.py` + `ops/tests/test_sentinel.py` (OPS.1/OPS.2).

**24 arquivos de teste** — incluindo `test_cfc.py` (1.2/6.1), `test_parity.py` (1.3b),
`test_wiring.py` (1.4), `test_bundle*.py` (6.2-6.4), `test_containment_bridge.py` (6.5),
`test_stability.py` (1.8), `test_robustness.py` (1.11), `test_ablation*.py` (1.9/7.3),
`test_ewc_temporal.py` (7.2), `test_surprise_decay.py` (7.1), `test_scale_ltc.py` (5.5),
`test_routine_continual.py` (5.6), `test_archive/mutate/sandbox/transfer/automl.py` (3.1-3.4, 5.7).

## O que NÃO está quebrado

- **O código é coerente:** varredura de imports em `visao/` não encontra nenhuma referência
  aos módulos ausentes. Nada quebra em tempo de execução por causa dessas remoções.
- **A suíte que resta passa** (subset fast do CI: ver `ops/watchdog_cycles.log`).
- **Governança intacta:** `visao/governance/` não foi tocada (selo R3 confere).

## O que ESTÁ quebrado

A **rastreabilidade**. O BACKLOG afirma 76 tarefas `done` com evidência apontando para
artefatos; 58 desses artefatos não existem mais. Consequências:

1. Um agente futuro que tente "verificar" uma tarefa `done` lendo o arquivo declarado
   conclui, falsamente, que a tarefa nunca foi feita.
2. A alegação central do projeto (Fase 0: "esquecimento 97,2% menor que controle") tem como
   evidência declarada `prototype/results_continual.json` — **esse arquivo foi removido em
   `dbba6b6`**. O número sobrevive apenas como texto no BACKLOG.
3. O portão 10.0 ("30 dias de operação contínua validada") está `done` com base em
   `visao/ops/logs/gate_validation.json`, que registra `uptime_seconds: 60.0006` e
   `total_errors: 5` com mensagens `"Injected error #N for testing recovery"` — ou seja,
   uma **simulação de 60 segundos com erros auto-injetados**, não 30 dias reais.
   O `verify` declarado da tarefa (`visao/ops/validate_30days.py`) também não existe em disco.

## Decisão pendente (humana — Artigo 01 do Código Humanitas)

Não se rebaixa nenhuma tarefa `done` sem decisão explícita do Juan. Opções:

- **(a)** Restaurar os artefatos a partir do git (`git checkout <commit> -- <path>`) e
  re-selar o que for necessário;
- **(b)** Emendar o BACKLOG marcando as tarefas afetadas como `done_artifact_removed`
  (honesto: o trabalho existiu, a evidência não), mantendo o histórico;
- **(c)** Aceitar que o BACKLOG é um registro histórico e criar um BACKLOG novo só com
  o que existe em disco.

Registrado como tarefa bloqueada `79` no `BACKLOG.json`.
