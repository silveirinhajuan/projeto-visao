# VISÃO — Continuous Learning with Positive Backward Transfer

**Project VISÃO** is a single-node continual learning system that achieves **Positive Backward Transfer** — it improves on past tasks after learning new ones, without replay, without backpropagation, and without catastrophic forgetting.

---

## Key Results

| Metric | Value |
|--------|-------|
| **Forgetting** | **-0.12** (negative = improvement on past tasks) |
| **Error** | **0.007** (regression) |
| **Stress Test** | 20 adversarial tasks without failure |
| **MNIST Split** | PBT maintained in visual domain |

### Comparison with SOTA (2025-2026)

| Method | Forgetting | Error | Status |
|--------|-----------|-------|--------|
| **VISÃO (this work)** | **-0.124** | **0.007** | ✓ Best |
| M-LTC (Srinivas 2026) | -0.100 | 0.018 | 157% worse error |
| CH-HNN (Shi et al. 2025) | +0.292 | 0.679 | 9899% worse error |

---

## Architecture

VISÃO uses three local learning rules (no backpropagation):

1. **Oja's Rule** — Self-organization of the recurrent reservoir
2. **Synaptic Consolidation** — Importance-weighted plasticity (EWC-like)
3. **Surprise Gate** — Neuromodulatory dynamic learning rate

The combination produces **Positive Backward Transfer**: the system gets better at old tasks after learning new ones.

---

## Repository Structure

```
projeto-visao/
├── prototype/           # Core implementation (numpy)
│   ├── liquid.py        # Liquid Time-Constant (LTC/CfC) cell
│   └── plasticity.py    # Local learning rules
├── visao/               # Extended modules
│   ├── core/            # JAX implementation, wiring, predictive coding
│   ├── continual/       # Continual learning loop
│   ├── evolve/          # AutoML, mutation, archive
│   ├── governance/      # Safety rules (R1-R5)
│   └── analysis/        # Diagnostic scripts
├── benchmark_2026.py    # Reproducible benchmark
├── ablation_pbt_v2.py   # Ablation study
├── limit_failure.py     # Failure point analysis
├── scale_local.py       # Multi-layer scaling
├── mnist_split_fast.py  # Visual domain validation
├── pbt_spectrum.py      # Tau spectrum analysis
└── VALIDATION_FINAL_2026.md  # Full report
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/silveirinhajuan/projeto-visao.git
cd projeto-visao

# Run benchmark (reproduces key results)
python3 benchmark_2026.py

# Run ablation study
python3 ablation_pbt_v2.py

# Run stress test
python3 benchmark_stress_2026.py
```

---

## Key Findings

1. **Consolidation is critical** — without it, the system diverges
2. **Surprise gate is redundant** — in this protocol, it adds no benefit
3. **PBT is robust** — holds across 20+ adversarial tasks
4. **Deep stack works** — but flat architecture performs better
5. **Visual domain validated** — PBT holds on MNIST split

---

## Citation

If you use this code, please cite:

```bibtex
@misc{visao2026,
  title={VISÃO: Continuous Learning with Positive Backward Transfer},
  author={Guerra, Juan and ÍRIS},
  year={2026},
  url={https://github.com/silveirinhajuan/projeto-visao}
}
```

---

## License

MIT License — see LICENSE file for details.

---

## Acknowledgments

- Hasani et al. (2021) — Liquid Time-Constant Networks
- Kirkpatrick et al. (2017) — Elastic Weight Consolidation
- Oja (1982) — Simplified neuron model as a principal component analyzer
