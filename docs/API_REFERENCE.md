# VISÃO — API Reference

## Core Modules

### `visao.brain.VisaoBrain`

Classe principal do cérebro VISÃO. Combina LiquidCell (LTC/CfC) com Oja + EWC + Surprise.

```python
from visao.brain import VisaoBrain

brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)
```

#### Parâmetros

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `n_in` | int | — | Dimensão da entrada |
| `n_hidden` | int | — | Neurônios no reservatório líquido |
| `n_out` | int | — | Dimensão da saída |
| `tau` | float | 1.0 | Time constant do líquido |
| `lr` | float | 0.01 | Taxa de aprendizado base |
| `ewc_lambda` | float | 100.0 | Força da consolidação EWC |
| `oja_alpha` | float | 0.001 | Taxa de auto-organização de Oja |
| `surprise_gain` | float | 1.0 | Ganho da surpresa |
| `seed` | int | None | Seed para reprodutibilidade |

#### Métodos

##### `forward(x: np.ndarray) → np.ndarray`
Forward pass (inferência). Retorna saída do líquido.

##### `learn(x: np.ndarray, y: np.ndarray) → float`
Aprendizado online. Atualiza pesos via Oja + EWC + Surprise. Retorna loss.

##### `set_mode(mode: str) → VisaoBrain`
Modos: `'train'`, `'continual'`, `'infer'`, `'agentic'`.

##### `save(path: str)`
Salva estado completo (pesos + config).

##### `load(path: str) → VisaoBrain`
Carrega estado completo.

---

### `visao.memory.hippocampus.HippocampusMemory`

Sistema de memória inspirado no hipocampo.

```python
from visao.memory.hippocampus import HippocampusMemory

memory = HippocampusMemory(n_hidden=32, buffer_capacity=1000, semantic_capacity=500)
```

#### Componentes

| Classe | Descrição |
|--------|-----------|
| `EpisodicBuffer` | Buffer circular de experiências com consolidação |
| `SemanticGraph` | Grafo de conhecimento (entidades + relações) |
| `ProceduralMemory` | Armazenamento de políticas/ações aprendidas |

---

### `visao.reasoning.neuro_symbolic.NeuroSymbolicReasoner`

Camada de raciocínio neuro-simbólico.

```python
from visao.reasoning.neuro_symbolic import NeuroSymbolicReasoner

reasoner = NeuroSymbolicReasoner(n_hidden=32, n_in=2, n_out=1)
```

#### Componentes

| Classe | Descrição |
|--------|-----------|
| `SymbolicSolver` | Planejador lógico (PDDL-like) |
| `TextualGradient` | Feedback estruturado para o liquid core |
| `VerificationEngine` | Verificador de provas/códigos/planos |

---

### `visao.agent.planner.AgenticOrchestrator`

Módulo agentico: planejamento + execução + monitoramento.

```python
from visao.agent.planner import AgenticOrchestrator

agent = AgenticOrchestrator(brain)
result = agent.run_task("Calcular a raiz quadrada de 144")
```

#### Componentes

| Classe | Descrição |
|--------|-----------|
| `TaskDecomposer` | Decompõe tarefas em subtasks |
| `ToolExecutor` | Executa ferramentas (calc, search, code) |
| `SelfMonitor` | Monitora saúde do sistema |

---

### `visao.integrate.VisaoCognitiveBrain`

Classe que integra todos os módulos.

```python
from visao.integrate import VisaoCognitiveBrain

brain = VisaoCognitiveBrain(2, 32, 1, seed=42)
brain.set_mode('continual')

# Pipeline cognitivo completo
result = brain.think(np.array([0.5, -0.3]))

# Capacidades agenticas
result = brain.plan_and_execute("Minha tarefa")

# Memória
brain.remember({'input': x, 'output': y, 'context': {}})
memories = brain.recall(query, top_k=5)

# Save/load
brain.save("/tmp/brain.json")
brain_loaded = VisaoCognitiveBrain.load("/tmp/brain.json")
```

---

## Utility Modules

### `visao.ops.jit_opt`

Compilação JIT com Numba para aceleração.

```python
from visao.ops.jit_opt import JitBrain

jit_brain = JitBrain.from_brain(brain)
# speedup 4.5x-10.85x
```

### `visao.ops.quantize`

Quantização int8 para economia de memória.

```python
from visao.ops.quantize import Quantizer

quantizer = Quantizer(bits=8, scheme='per_channel')
```

### `visao.ops.prune`

Poda de sinapses inúteis.

```python
from visao.ops.prune import SynapsePruner

pruner = SynapsePruner(threshold=0.01, target_sparsity=0.8)
```

### `visao.ops.live_loop`

Loop de operação contínua (30 dias).

```python
from visao.ops.live_loop import LiveLoop

loop = LiveLoop(brain, checkpoint_interval=3600)
```

---

## Benchmarks

### `visao.bench.benchmark_sota`

Comparação com LSTM, GRU, Transformer.

```bash
python3 visao/bench/benchmark_sota.py --quick
```

### `visao.bench.streaming_bench`

Benchmark de streaming task-free.

```bash
python3 visao/bench/streaming_bench.py
```

---

## Testing

```bash
# Run all tests
pytest

# Run specific module tests
pytest visao/tests/test_brain.py
pytest visao/memory/test_hippocampus.py
pytest visao/reasoning/test_neuro_symbolic.py
```
