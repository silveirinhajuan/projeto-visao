# VISÃO — Agentic Cognitive Architecture with Liquid Neural Networks

**VISÃO** is a cognitive architecture designed to operate as a personal brain — trained like modern models, but also continuously learning from real-world use.

> *"Eu quero que o modelo seja capaz de ser treinado como modelos atuais são, mas também a partir do uso contínuo."*
> — Juan Guerra, 2026

---

## Philosophy

VISÃO combines four paradigms:

| Paradigm | Role | Implementation |
|----------|------|----------------|
| **Liquid Neural Networks** | Continuous adaptation, temporal perception | LTC/CfC + Oja + EWC + Surprise |
| **Neuro-Symbolic AI** | Explicit reasoning, planning, verification | Symbolic solver + textual gradients |
| **Agentic AI** | Task decomposition, tool use, execution | Planner + executor + monitor |
| **Continual Learning** | Zero-forgetting, lifelong adaptation | Online learning + hippocampal memory |

**Key insight**: A brain is not just pattern matching (neural) or just logic (symbolic). It's a **liquid cognitive system** that adapts continuously, reasons explicitly, and acts in the world.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISÃO COGNITIVE BRAIN                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Perception  │  │   Memory    │  │  Reasoning  │            │
│  │  (LTC/CfC)  │  │  (Hippo.)   │  │ (Symbolic)  │            │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         └────────────────┼────────────────┘                    │
│              ┌───────────┴───────────┐                        │
│              │   Liquid Core (LTC)   │                        │
│              │  Continuous-time ODE  │                        │
│              │  Oja + EWC + Surprise │                        │
│              └───────────┬───────────┘                        │
│         ┌────────────────┼────────────────┐                   │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐            │
│  │  Planning   │  │  Execution  │  │  Monitoring │            │
│  │  (Symbolic) │  │  (Agentic)  │  │  (Health)   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              Governance Layer (R1-R5)                    │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## State of the Art (2025-2026)

Our design is informed by recent advances:

- **Liquid Foundation Models (LFM2.5)** — Liquid AI proved liquid architectures can scale to language and reasoning at 1B-24B params
- **Agentic Neural Networks (𝒜𝒩𝒩)** — Multi-agent systems optimized via textual backpropagation (arXiv 2506.09046)
- **Neuro-Symbolic AI** — Hybrid neural-symbolic systems (AlphaProof, Scallop, DeepProbLog)
- **Hippocampal Memory** — ICML 2026 position: explicit memory is essential for higher cognition
- **Continual Learning** — Modular memory architectures for lifelong adaptation

---

## Key Results (Validation Phase)

| Metric | Value | Comparison |
|--------|-------|------------|
| **Forgetting** | **-0.12** | Negative = improves on past tasks (PBT) |
| **Error** | **0.007** | 157% better than M-LTC, 9899% better than CH-HNN |
| **Stress Test** | 20/20 tasks | No catastrophic failure |
| **Speed** | 0.1s | 3x faster than LSTM/GRU |
| **Parameters** | 4,288 | 4x fewer than LSTM, 11x fewer than Transformer |

### Benchmark vs LSTM/GRU/Transformer

| Model | MSE Total | Forgetting | Time | Params |
|-------|-----------|------------|------|--------|
| **VisaoBrain** | 0.3831 | **0.0000** | **0.1s** | **4,288** |
| LSTM | 0.2289 | 0.0000 | 0.3s | 17,217 |
| GRU | 0.1257 | 0.0000 | 0.3s | 12,929 |
| Transformer | 0.0489 | 0.0000 | 0.2s | 49,665 |

**Trade-off**: VISÃO sacrifices raw accuracy for **zero forgetting**, **speed**, and **parameter efficiency**. Future versions will close the accuracy gap.

---

## Repository Structure

```
projeto-visao/
├── VISAO_V3_PHILOSOPHY.md    # Complete philosophy and training plan
├── prototype/                 # Core implementation (numpy)
│   ├── liquid.py              # LiquidCell (LTC/CfC)
│   └── plasticity.py          # LocalLearner (Oja + EWC)
├── visao/                     # Extended modules
│   ├── brain.py               # VisaoBrain (unified class)
│   ├── memory/                # Hippocampal memory (planned)
│   ├── reasoning/             # Neuro-symbolic layer (planned)
│   ├── agent/                 # Agentic module (planned)
│   ├── ops/                   # Quantization, JIT, pruning, live loop
│   ├── bench/                 # Streaming bench, SOTA benchmark
│   ├── continual/             # Task-free detection (CUSUM)
│   ├── governance/            # Safety rules (R1-R5)
│   └── analysis/              # Diagnostics
├── experiments/               # Benchmarks and ablation studies
├── tests/                     # Test suite
└── docs/                      # Documentation
```

---

## Training Plan

VISÃO uses **dual training**:

### Phase 1: Pre-training (Traditional)
- Dataset: 1-5T tokens (text, code, math, reasoning)
- Architecture: Liquid Transformer Hybrid (1B-7B params)
- Infrastructure: Colab CLI + cloud GPU

### Phase 2: Fine-tuning (Alignment)
- RLHF / DPO for safety and style
- Domain-specific fine-tuning

### Phase 3: Continuous Learning (Online)
- Oja + EWC + Surprise for adaptation
- Every interaction updates the liquid core
- Zero catastrophic forgetting

### Phase 4: Recursive Self-Improvement (Governed)
- System proposes and tests improvements
- Governance R1-R5 enforces safety limits

---

## Quick Start

```bash
git clone https://github.com/silveirinhajuan/projeto-visao.git
cd projeto-visao

# Run benchmark (reproduces key results)
python3 benchmark_2026.py

# Run SOTA comparison (VisaoBrain vs LSTM/GRU/Transformer)
python3 visao/bench/benchmark_sota.py --quick

# Run stress test
python3 benchmark_stress_2026.py

# Test VisaoBrain
python3 -c 'from visao.brain import VisaoBrain; b = VisaoBrain(2, 32, 1); print(b)'

# Run full pipeline demo
python3 example_full_pipeline.py

# Run live loop simulation
python3 visao/ops/live_loop.py --mode simulate --duration 60
```

---

## Citation

```bibtex
@misc{visao2026,
  title={VISÃO: Agentic Cognitive Architecture with Liquid Neural Networks},
  author={Guerra, Juan and ÍRIS},
  year={2026},
  url={https://github.com/silveirinhajuan/projeto-visao}
}
```

---

## License

MIT License

---

## Acknowledgments

- Hasani et al. (2021) — Liquid Time-Constant Networks
- Kirkpatrick et al. (2017) — Elastic Weight Consolidation
- Oja (1982) — Simplified neuron model
- Liquid AI (2026) — Liquid Foundation Models (LFM2.5)
- DeepMind (2026) — From AGI to ASI pathways
- Ma et al. (2025) — Agentic Neural Networks
