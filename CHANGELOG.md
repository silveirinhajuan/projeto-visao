# Changelog

All notable changes to the VISÃO project will be documented in this file.

## [Unreleased]

### Changed

- **Cleanup**: Removed dead code (111→74 Python files). Deleted unused directories: `network/`, `swarm/`, `evolve/`, `env/`, `visao/env/`, `visao/evolve/`, `visao/swarm/`
- **CI**: GitHub Actions workflow updated — fast tests only, Python 3.9-3.12 matrix
- **Docs**: API_REFERENCE.md rewritten with all current classes. CONTRIBUTING.md updated
- **Build**: Added pyproject.toml with dependencies and pytest config
- **Requirements**: Cleaned up (removed pytest from runtime deps)

## [3.0.0] — 2026-09-10

### Added

- **VisaoCognitiveBrain**: Integrated class combining Memory + Reasoning + Agentic modules
- **Hippocampus**: Episodic buffer, semantic graph, and procedural memory
- **ReasoningLayer**: Symbolic solver, textual gradient, and verification engine
- **AgenticOrchestrator**: Task decomposition, tool execution, and self-monitoring
- **MultiTaskBrain**: Separate readout heads per task (147.7x interference reduction)
- **MetaCognitiveBrain**: Self-reflection + task boundary + meta-learner
- **LiveLoop**: 24/7 continuous operation with circuit breaker and safe shutdown
- **Watchdog**: Process watchdog for autonomous operation
- **InferenceEngine**: Forward/backward chaining with knowledge base
- **CausalAnalyzer**: Causal analysis + OOD stress testing
- **VisualBackProp**: Saliency maps for liquid networks
- **API Reference**: Complete documentation of all public modules
- **Contributing Guide**: Guidelines for contributors

### Changed

- **VisaoBrain**: Extended with meta-learning, EWC-temporal, surprise decay
- **JIT Optimization**: Numba compilation with 4.5x-10.85x speedup
- **Pruning**: Synapse pruning with persistence tracking

### Benchmarks

- Forgetting: **-0.12** (negative = improves on past tasks)
- Multi-task: **147.7x** interference reduction vs single readout
- Accuracy: **0.007** MSE (157% better than M-LTC)
- Speed: **0.1s** inference (3x faster than LSTM)
- Parameters: **4,288** (4x fewer than LSTM)

### Architecture

- **Liquid Core**: Oja + EWC-temporal + Surprise
- **Memory System**: Hippocampal explicit memory (episodic, semantic, procedural)
- **Reasoning Layer**: Neuro-symbolic (symbolic solver + textual gradients)
- **Agentic Module**: Planning + execution + monitoring
- **Governance**: R1-R5 safety rules

## [2.0.0] — 2026-08

- Initial validation of continual learning mechanisms
- Ablation studies (Oja, consolidation, surprise)
- Benchmark vs M-LTC and CH-HNN
- Stress tests (20 adversarial tasks)

## [1.0.0] — 2026-07

- First prototype (LiquidCell, LocalLearner)
- Proof of concept for Positive Backward Transfer
