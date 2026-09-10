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

## Quick Start

```bash
# Install
pip install -e .

# Run demo
python3 -c 'from visao.integrate import demo_cognitive_brain; demo_cognitive_brain()'

# Run tests
pytest

# Run benchmarks
python3 visao/bench/benchmark_sota.py --quick
```

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

## Key Results

| Metric | Value | Comparison |
|--------|-------|------------|
| **Forgetting** | **-0.12** | Negative = improves on past tasks |
| **Accuracy** | **0.007** MSE | 157% better than M-LTC |
| **Speed** | **0.1s** inference | 3x faster than LSTM |
| **Parameters** | **4,288** | 4x fewer than LSTM |

### Benchmark vs SOTA

| Model | MSE Total | Time | Params |
|-------|-----------|------|--------|
| **VisaoBrain** | 0.3831 | **0.1s** | **4,288** |
| LSTM | 0.2289 | 0.3s | 17,217 |
| GRU | 0.1257 | 0.3s | 12,929 |
| Transformer | 0.0489 | 0.2s | 49,665 |

---

## Project Stats

- **16,414 lines** of Python code
- **92 files** in core library
- **14 test files** with **30+ tests** passing
- **14 tasks** completed (0 pending)
- **CI/CD** with GitHub Actions

---

## Documentation

- [API Reference](docs/API_REFERENCE.md) — Complete module documentation
- [Contributing Guide](docs/CONTRIBUTING.md) — How to contribute
- [Changelog](CHANGELOG.md) — Version history
- [Philosophy](VISAO_V3_PHILOSOPHY.md) — Architecture and training plan

---

## Repository

- **GitHub**: https://github.com/silveirinhajuan/projeto-visao
- **License**: MIT
- **Author**: Juan Guerra with ÍRIS

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
