# VISÃO — Plano de Melhoria e Limpeza

## Problemas Identificados

### 1. Código Duplicado
- `experiments/ablation_pbt.py` e `experiments/ablation_pbt_v2.py` — versões antigas
- `visao/analysis/ablation.py` e `visao/analysis/ablation_2.py` — duplicado
- `visao/analysis/diagnose_omega.py`, `v2`, `v3` — múltiplas versões

### 2. Módulos Não Relacionados ao VISÃO
- `network/` — churn dashboard, recruitment model (projeto separado)
- `ops/` — sentinel, nightly_brief (infraestrutura, não core)

### 3. Legacy Code
- `prototype/` — código antigo (liquid.py, plasticity.py)
- `experiments/` — benchmarks iniciais (devem ficar só resultados finais)

### 4. Arquivos Espalhados
- 106 JSON files de resultados
- 44 test files (alguns obsoletos)
- Sem requirements.txt / setup.py

### 5. Documentação Incompleta
- Sem documentação de API
- Sem guia de contribuição
- Sem changelog

### 6. Sem CI/CD
- Sem GitHub Actions
- Sem automação de testes

### 7. Sem __init__.py
- Diretórios sem package marker

### 8. Código Morto
- Módulos não importados por ninguém
- Testes de código removido

## Plano de Ação

1. **Limpeza**: remover duplicados e legacy
2. **Organização**: mover módulos não-core para subdirs
3. **Consolidação**: juntar testes e resultados
4. **Documentação**: requirements.txt, API docs, contrib
5. **CI/CD**: GitHub Actions para testes
6. **Verificação**: rodar todos os testes após limpeza
