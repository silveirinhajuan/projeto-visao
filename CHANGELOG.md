# Changelog

All notable changes to the VISÃO project will be documented in this file.

## [3.0.0] — 2026-09-10

### Added

- **VisaoCognitiveBrain**: Integrated class combining Memory + Reasoning + Agentic modules
- **HippocampusMemory**: Episodic buffer, semantic graph, and procedural memory
- **NeuroSymbolicReasoner**: Symbolic solver, textual gradient, and verification engine
- **AgenticOrchestrator**: Task decomposition, tool execution, and self-monitoring
- **API Reference**: Complete documentation of all public modules
- **Contributing Guide**: Guidelines for contributors
- **Changelog**: This file

### Changed

- **DualModeBrain**: Extended to support 4 modes (train, continual, infer, agentic)
- **JIT Optimization**: Numba compilation with 4.5x-10.85x speedup
- **Pruning**: Synapse pruning with persistence tracking
- **Live Loop**: Continuous operation with circuit breaker and safe shutdown

### Benchmarks

- Forgetting: **-0.12** (negative = improves on past tasks)
- Accuracy: **0.007** MSE (157% better than M-LTC)
- Speed: **0.1s** inference (3x faster than LSTM)
- Parameters: **4,288** (4x fewer than LSTM)

### Architecture

- **Liquid Core**: Oja + EWC-temporal + Surprise
- **Memory System**: Hippocampal explicit memory (episodic, semantic, procedural)
- **Reasoning Layer**: Neuro-symbolic (symbolic solver + textual gradients)
- **Agentic Module**: Planning + execution + monitoring
- **Governance**: R1-R5 safety rules

## [2.0.0] — 2026-08 (implied)

- Initial validation of continual learning mechanisms
- Ablation studies (Oja, consolidation, surprise)
- Benchmark vs M-LTC and CH-HNN
- Stress tests (20 adversarial tasks)

## [1.0.0] — 2026-07 (implied)

- First prototype (LiquidCell, LocalLearner)
- Proof of concept for Positive Backward Transfer
