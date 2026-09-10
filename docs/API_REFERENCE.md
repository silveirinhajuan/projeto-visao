# VISÃO — API Reference

> **Version:** 3.0.0 · **Last updated:** 2026-09-10

## Core

### `visao.brain.VisaoBrain`

Main brain class. Combines LiquidCell (LTC/CfC) with Oja + EWC + Surprise + Meta-learning.

```python
from visao.brain import VisaoBrain

brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42, meta_learn=True)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_in` | int | — | Input dimension |
| `n_hidden` | int | — | Liquid reservoir neurons |
| `n_out` | int | — | Output dimension |
| `tau` | float | 1.0 | Base time constant |
| `lr` | float | 0.01 | Base learning rate |
| `ewc_lambda` | float | 100.0 | EWC consolidation strength |
| `oja_lr` | float | 0.002 | Oja self-organization rate |
| `surprise_gain` | float | 3.0 | Surprise gate gain |
| `meta_learn` | bool | False | Enable meta-learning |
| `seed` | int | None | RNG seed |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `(x: np.ndarray) → np.ndarray` | Inference pass |
| `learn` | `(x: np.ndarray, y: np.ndarray) → dict` | Online learning step. Returns `{err, surprise, lr}` |
| `set_mode` | `(mode: str) → VisaoBrain` | Modes: `'learn'`, `'infer'` |
| `reset_state` | `() → None` | Reset hidden state |
| `save` | `(path: str) → None` | Save full state (weights + config) |
| `load` | `(path: str) → VisaoBrain` | Load full state (classmethod) |
| `evaluate` | `(stream) → dict` | Evaluate on stream, returns `{mse, mae}` |

---

### `visao.integrate.VisaoCognitiveBrain`

Unified cognitive architecture: VisaoBrain + Memory + Reasoning + Agentic.

```python
from visao.integrate import VisaoCognitiveBrain

cog = VisaoCognitiveBrain(n_in=2, n_hidden=32, n_out=1, seed=42)
```

**Additional methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `think` | `(x: np.ndarray) → dict` | Full cognitive cycle |
| `plan_and_execute` | `(task: str) → dict` | Plan + execute via agent |
| `remember` | `(experience: dict) → None` | Store in episodic memory |

---

### `visao.core.cfc.CfCCell`

Closed-form Continuous-time cell (JAX). The liquid dynamics core.

```python
from visao.core.cfc import CfCCell

cell = CfCCell(n_in=2, n_hidden=32, sparsity=0.5, tau_min=0.4, tau_max=4.0)
```

---

## Memory

### `visao.memory.hippocampus.Hippocampus`

Hippocampus-inspired memory system.

```python
from visao.memory.hippocampus import Hippocampus

mem = Hippocampus(n_hidden=32, buffer_capacity=1000, semantic_capacity=500)
```

**Components:**

| Class | Description |
|-------|-------------|
| `EpisodicBuffer` | Circular experience buffer with consolidation |
| `SemanticGraph` | Knowledge graph (entities + relations) |
| `ProceduralMemory` | Learned policies/actions |

---

### `visao.memory.multi_timescale.MultiTimescaleMemory`

Three-timescale memory hierarchy (τ=0.1, 1.0, 10.0).

```python
from visao.memory.multi_timescale import MultiTimescaleMemory

mtm = MultiTimescaleMemory(n_hidden=32, capacity=1000)
```

---

## Reasoning

### `visao.reasoning.neuro_symbolic.ReasoningLayer`

Neuro-symbolic reasoning layer.

```python
from visao.reasoning.neuro_symbolic import ReasoningLayer

rl = ReasoningLayer(n_hidden=32, n_in=2, n_out=1)
```

**Components:**

| Class | Description |
|-------|-------------|
| `SymbolicSolver` | PDDL-like logical planner (forward chaining) |
| `TextualGradient` | Feedback that modulates liquid core |
| `VerificationEngine` | Logical proof verifier |

---

### `visao.reasoning.inference.InferenceEngine`

Forward/backward chaining inference engine.

```python
from visao.reasoning.inference import InferenceEngine, KnowledgeBase

kb = KnowledgeBase()
engine = InferenceEngine(kb=kb)
```

---

## Continual Learning

### `visao.continual.meta_learner.MetaContinualLearner`

Meta-learner for continual learning (MAML-style).

```python
from visao.continual.meta_learner import MetaContinualLearner

mcl = MetaContinualLearner(n_in=2, n_hidden=32, n_out=1, n_tasks=5)
```

---

### `visao.continual.task_boundary.TaskBoundaryDetector`

Detects task boundaries and triggers consolidation.

```python
from visao.continual.task_boundary import TaskBoundaryDetector, Consolidator

tbd = TaskBoundaryDetector(window=50, threshold=2.0)
cons = Consolidator(ewc_lambda=100.0)
```

---

### `visao.continual.task_free.TaskFreeLearner`

Task-free continual learning with CUSUM change detection.

```python
from visao.continual.task_free import TaskFreeLearner, CUSUMDetector

tfl = TaskFreeLearner(n_in=2, n_hidden=32, n_out=1)
```

---

### `visao.continual.ewc.MultiTaskBrain`

Multi-task brain with separate readout heads per task.

```python
from visao.continual.ewc import MultiTaskBrain, MultiTaskReadout

mtb = MultiTaskBrain(n_in=2, n_hidden=32, n_out=1, n_tasks=5)
```

---

## Agent

### `visao.agent.planner.AgenticOrchestrator`

Planning and execution orchestrator.

```python
from visao.agent.planner import AgenticOrchestrator

agent = AgenticOrchestrator(n_hidden=32, n_in=2, n_out=1)
```

**Components:**

| Class | Description |
|-------|-------------|
| `TaskDecomposer` | Breaks tasks into subtasks |
| `ToolExecutor` | Executes actions |
| `SelfMonitor` | Monitors own performance |

---

### `visao.agent.self_reflection.SelfReflection`

Performance tracking and self-adjustment.

```python
from visao.agent.self_reflection import PerformanceTracker, SelfReflection

tracker = PerformanceTracker(window=100)
sr = SelfReflection(tracker=tracker)
```

---

### `visao.agent.meta_cognitive.MetaCognitiveBrain`

Meta-cognitive brain with self-reflection + task boundary + meta-learner.

```python
from visao.agent.meta_cognitive import MetaCognitiveBrain

mcb = MetaCognitiveBrain(n_in=2, n_hidden=32, n_out=1, n_tasks=5)
```

---

## Interpretability

### `visao.interpret.causal.CausalAnalyzer`

Causal analysis and OOD stress testing.

```python
from visao.interpret.causal import CausalAnalyzer, CausalGeneralizationTest

ca = CausalAnalyzer(brain)
```

---

### `visao.interpret.saliency.VisualBackProp`

Visual backpropagation for saliency maps.

```python
from visao.interpret.saliency import VisualBackProp, OODStressTest, RangeTest

vbp = VisualBackProp(brain)
```

---

### `visao.interpret.visualize.StateVisualizer`

ASCII state and topology visualization.

```python
from visao.interpret.visualize import StateVisualizer, TopologyVisualizer

sv = StateVisualizer(brain)
```

---

## Operations

### `visao.ops.live_loop.LiveLoop`

Continuous 24/7 operation loop with checkpoint/recovery.

```python
from visao.ops.live_loop import LiveLoop

loop = LiveLoop(brain=brain, data_source=data_source)
metrics = loop.start(duration_seconds=86400)
```

**Components:**

| Class | Description |
|-------|-------------|
| `CircuitBreaker` | Prevents cascade failures |
| `CheckpointManager` | Automatic checkpointing |
| `BrainHealthMonitor` | Health monitoring with trends |

---

### `visao.ops.watchdog.VisaoWatchdog`

Process watchdog for autonomous operation.

```python
from visao.ops.watchdog import VisaoWatchdog, AgentHeartbeat

watchdog = VisaoWatchdog(check_interval=60, heartbeat_timeout=600)
watchdog.start()
```

---

### `visao.ops.jit_opt.JitBrain`

JIT-compiled brain optimization.

```python
from visao.ops.jit_opt import JitBrain

jit_brain = JitBrain(brain)
```

---

### `visao.ops.quantize.QuantizedBrain`

Quantized brain for efficient inference.

```python
from visao.ops.quantize import QuantizedBrain, QuantizedTensor

qbrain = QuantizedBrain(brain, bits=8)
```

---

### `visao.ops.prune.SynapsePruner`

Synaptic pruning for efficiency.

```python
from visao.ops.prune import SynapsePruner, PersistentSynapsePruner

pruner = SynapsePruner(threshold=0.01)
```

---

## Governance

### `visao.governance.containment.ContainedNode`

Governance containment with R1-R5 rules.

```python
from visao.governance.containment import ContainedNode, QuorumGate, LifetimeGuard
```

**Rules:**

| Rule | Class | Description |
|------|-------|-------------|
| R1 | `QuorumGate` | k-of-n quorum for replication |
| R2 | `LifetimeGuard` | Token lifetime limit |
| R3 | `BoundaryMonitor` | Code integrity verification |
| R4 | `ReplicationThrottle` | PoW throttle on spawn |
| R5 | `LineageLedger` | Append-only lineage ledger |

---

## Analysis

### `visao.analysis.ewc_temporal.EWCTemporalLearner`

EWC with temporal decay of importance.

```python
from visao.analysis.ewc_temporal import EWCTemporalLearner

ewc_tl = EWCTemporalLearner(n_hidden=32, n_out=1, decay_rate=0.01)
```

---

### `visao.analysis.surprise_decay.SurpriseDecayLearner`

Surprise as omega decay (not lr amplification).

```python
from visao.analysis.surprise_decay import SurpriseDecayLearner

sdl = SurpriseDecayLearner(n_hidden=32, n_out=1, surprise_decay=0.5)
```

---

### `visao.analysis.adaptive_consolidation.AdaptiveConsolidationBrain`

Brain with adaptive consolidation strength.

```python
from visao.analysis.adaptive_consolidation import AdaptiveConsolidationBrain

acb = AdaptiveConsolidationBrain(n_in=2, n_hidden=32, n_out=1)
```

---

## Benchmarks

### `visao.bench.benchmark_sota`

Benchmark: VISÃO vs LSTM/GRU/Transformer.

```bash
python -m visao.bench.benchmark_sota --quick
```

### `visao.bench.scaling_laws`

Scaling laws experiments.

```bash
python -m visao.bench.scaling_laws
```

### `visao.bench.streaming_bench`

Streaming learning benchmark.

```bash
python -m visao.bench.streaming_bench
```
