# Contributing to VISÃO

Thank you for your interest in contributing to VISÃO! This document outlines the guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/projeto-visao.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest`

## Project Structure

```
projeto-visao/
├── visao/                    # Core library
│   ├── brain.py              # VisaoBrain (main class)
│   ├── integrate.py          # VisaoCognitiveBrain (integrated)
│   ├── memory/               # Hippocampal memory system
│   ├── reasoning/            # Neuro-symbolic reasoning
│   ├── agent/                # Agentic module
│   ├── ops/                  # Optimization (JIT, quantize, prune)
│   ├── bench/                # Benchmarks
│   ├── continual/            # Continual learning
│   ├── governance/           # Safety rules (R1-R5)
│   ├── analysis/             # Analysis scripts
│   ├── evolve/               # AutoML and evolution
│   ├── interpret/            # Interpretability
│   ├── swarm/                # Swarm intelligence
│   └── core/                 # Core components (CfC, wiring)
├── tests/                    # Test suite
├── docs/                     # Documentation
├── experiments/              # Experimental benchmarks
└── prototype/                # Reference implementation
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
pytest -v
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
