# Projeto VISÃO

Rede neural líquida (CfC + NCP + predictive coding) com plasticidade local,
treinada em CPU. Este README é o **portão da Fase 1**: só abre a Fase 2 por
conter números **medidos e reprodutíveis** (nada de alegações sem evidência).

## Estado

- Fase 0 (fundação): **concluída** — hipótese de redução de esquecimento sustentada.
- Fase 1 (substrato): **concluída** — 15/15 tarefas fechadas, suíte verde.
- Fase 2 (enxame): **bloqueada** até este portão — agora com números no README.

## Números medidos (reprodutíveis)

### Fase 0 — esquecimento contínuo (prototype/results_continual.json)
- Mean forgetting — plástico: `0.00830` | naive: `0.29238`
- **Redução de esquecimento: 97.2%** (plastic vs naive, 5 seeds)
- Erro final — plástico: `0.2868` | naive: `0.5284`

### Eficiência de parâmetros (visao/bench/results_param_count.json, psMNIST)
- CfC (h=64): **5002** parâmetros treináveis
- LSTM (h=128): 67850  |  GRU (h=128): 50816
- Razão **LSTM/CfC = 13.56x**  |  **GRU/CfC = 10.16x**
- Claim do projeto "~10x menos parâmetros": **sustentada**

### Estabilidade do reservatório (visao/analysis/results_stability.json, n=96, dt=0.1)
- Raio espectral efetivo W_rec: `0.758` (<1, ESN ok)
- Maior SR do jacobiano local: `0.975` (<1, contrativo)
- Expoente de Lyapunov empírico: `-0.748` (sub-caótico, memória estável)
- `||x||` final `3.726`, std dinâmico `0.306` (vivo, responde à entrada)

### Ablação (visao/analysis/results_ablation.json, grade 2x2x2 x 5 seeds)
- Completo (C+S+O): forgetting `+0.013` | baseline (`---`): `+0.176`
- Contribuição isolada (delta vs baseline, células estáveis):
  - Surpresa: `-0.262` (componente principal, reduz)
  - Oja: `-0.092` (reduz)
  - Consolidação: `+0.178` (sozinha aumenta — só ajuda em combinação)

### Heterogeneidade de tau (visao/analysis/results_tau_sweep.json)
- tau uniforme: `+0.0135 ± 0.0103` | melhor heterogêneo (10x): `+0.0125`
- Delta `-0.0011` **dentro do ruído** → heterogeneidade de tau é **DECORAÇÃO**
  neste protocolo; memória multiescala vem do gate de surpresa.

### Robustez (visao/analysis/results_robustness.json, estresse na inferência)
- Erro limpo: `0.2885`
- Ruído (sigma 0.2/0.5/1.0): imune (degradação ≤ `1.001x`)
- Morte de neurônios (10%/20%/40%): `1.047x` / `1.041x` / `1.023x`
- Veredito: **Fase 2 viável** (pior degradação 1.047x, tolerância 2.0x)

### Paridade JAX vs numpy (visao/tests/test_parity.py)
- Erro máximo de estado `5.55e-16`, de ativação `2.22e-16` (paridade de máquina)
- Razão de liquidez idêntica: `3.282x` em ambos

## Como reproduzir

```bash
env -u PYTHONPATH python3 -m pytest -q      # suíte completa (verde)
env -u PYTHONPATH python3 -m pytest visao/tests/test_readme_gate.py -v  # este portão
```

## Próximo passo

Com os números acima no README, a Fase 2 (transporte P2P + enxame com churn)
está desbloqueada — aguardando aprovação do Juan.
