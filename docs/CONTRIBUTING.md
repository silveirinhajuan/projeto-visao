# Contributing to VISÃO

Thank you for your interest in contributing to VISÃO! This document outlines the guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/projeto-visao.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest visao/tests/ tests/ -v`

## Project Structure

```
projeto-visao/
├── visao/                    # Core library
│   ├── brain.py              # VisaoBrain (main class)
│   ├── integrate.py          # VisaoCognitiveBrain (integrated)
│   ├── metaplasticity.py     # Meta-plasticity learner
│   ├── memory/               # Hippocampal memory system
│   │   ├── hippocampus.py    # Hippocampus (episodic + semantic + procedural)
│   │   └── multi_timescale.py # Three-timescale memory
│   ├── reasoning/            # Neuro-symbolic reasoning
│   │   ├── neuro_symbolic.py # Symbolic solver + textual gradient
│   │   └── inference.py      # Forward/backward chaining
│   ├── agent/                # Agentic module
│   │   ├── planner.py        # Planner + executor + monitor
│   │   ├── self_reflection.py # Performance tracking
│   │   └── meta_cognitive.py # Meta-cognitive brain
│   ├── ops/                  # Optimization & operations
│   │   ├── live_loop.py      # 24/7 continuous loop
│   │   ├── watchdog.py       # Process watchdog
│   │   ├── jit_opt.py        # JIT compilation
│   │   ├── quantize.py       # Quantization
│   │   ├── prune.py          # Synaptic pruning
│   │   ├── monitor.py        # Brain health monitor
│   │   └── data_source.py    # Data source for live loop
│   ├── bench/                # Benchmarks
│   ├── continual/            # Continual learning
│   │   ├── meta_learner.py   # Meta-learner (MAML-style)
│   │   ├── task_boundary.py  # Task boundary detection
│   │   ├── task_free.py      # Task-free learning (CUSUM)
│   │   ├── ewc.py            # EWC + MultiTaskBrain
│   │   └── loop.py           # Continual loop
│   ├── governance/           # Safety rules (R1-R5)
│   ├── analysis/             # Analysis scripts
│   ├── interpret/            # Interpretability
│   ├── core/                 # Core components (CfC)
│   └── tests/                # Test suites
├── tests/                    # Top-level tests
├── docs/                     # Documentation
├── prototype/                # Reference implementation (numpy)
└── experiments/              # Experimental benchmarks
```

## Code Style

- Follow PEP 8
- Use type hints for all functions
- Document all public classes and methods
- Keep functions focused and small

## Testing

- All new features must have tests
- Run full test suite before submitting PR
- Tests should be fast (< 30s total)

```bash
pytest visao/tests/ tests/ -v --tb=short
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with tests
3. Run tests: `pytest`
4. Commit with descriptive message: `git commit -m "feat: add new feature"`
5. Push to your fork: `git push origin feature/your-feature`
6. Open a Pull Request

## Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation
- `chore:` — Maintenance
- `test:` — Tests
- `refactor:` — Code refactoring
- `perf:` — Performance improvement

## Governance

VISÃO is governed by rules R1-R5 defined in `visao/governance/`. All contributions must respect these safety rules.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
