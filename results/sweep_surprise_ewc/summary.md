# Experiment 2: Surprise × EWC Interaction

## Design
- **Factors**: surprise_gain × ewc_lambda
- **Levels**: [0.0, 1.0, 2.0, 3.0] × [0.0, 2.0, 8.0, 20.0]
- **Combinations**: 16
- **Tasks**: 5 sequential sine regression tasks
- **Seeds**: 5 (42-46)
- **Total runs**: 80

## Results

### Forgetting (lower is better)

| Surprise ↓ / EWC → | 0.0 | 2.0 | 8.0 | 20.0 |
|---------------------|-----|-----|-----|------|
| 0.0 | 0.0894 | 0.0240 | 0.0129 | **0.0107** |
| 1.0 | 0.0894 | 0.0867 | 0.0786 | 0.0661 |
| 2.0 | 0.0894 | 0.0883 | 0.0849 | 0.0782 |
| 3.0 | 0.0894 | 0.0889 | 0.0871 | 0.0832 |

### Average MSE (lower is better)

| Surprise ↓ / EWC → | 0.0 | 2.0 | 8.0 | 20.0 |
|---------------------|-----|-----|-----|------|
| 0.0 | 0.6539 | 0.5551 | 0.5328 | **0.5275** |
| 1.0 | 0.6539 | 0.6476 | 0.6320 | 0.6115 |
| 2.0 | 0.6539 | 0.6511 | 0.6434 | 0.6311 |
| 3.0 | 0.6539 | 0.6524 | 0.6478 | 0.6397 |

### Backward Transfer (BWT, closer to 0 is better)

| Surprise ↓ / EWC → | 0.0 | 2.0 | 8.0 | 20.0 |
|---------------------|-----|-----|-----|------|
| 0.0 | -0.1117 | -0.0300 | -0.0161 | **-0.0133** |
| 1.0 | -0.1117 | -0.1084 | -0.0982 | -0.0826 |
| 2.0 | -0.1117 | -0.1104 | -0.1061 | -0.0978 |
| 3.0 | -0.1117 | -0.1111 | -0.1089 | -0.1039 |

## Key Findings

1. **Surprise has NO effect when EWC=0**: All rows with EWC=0 are identical regardless of surprise_gain. The surprise mechanism only interacts with consolidation.

2. **When surprise=0, increasing EWC improves BOTH metrics**:
   - EWC=0: forgetting=0.089, MSE=0.654
   - EWC=20: forgetting=0.011, MSE=0.528

3. **Surprise and EWC interact antagonistically**: When surprise>0, the benefit of EWC is partially counteracted. Surprise decays omega (importance weights), weakening consolidation.

4. **Without meta-learning, surprise is detrimental**: The original VISÃO-full config (surprise=3, EWC=8) showed forgetting=0.053, MSE=0.607. With surprise=0 and EWC=8, we get forgetting=0.013, MSE=0.533 — much better.

## Optimal Combination

| Metric | Value |
|--------|-------|
| surprise_gain | **0.0** |
| ewc_lambda | **20.0** (or 8.0 for practical purposes) |
| Forgetting | 0.0107 |
| Avg MSE | 0.5275 |
| BWT | -0.0133 |

## Recommendation

**Use surprise_gain=0.0 with ewc_lambda=8.0** for the best balance:
- EWC=8 gives near-optimal forgetting (0.013 vs 0.011 at EWC=20)
- Diminishing returns beyond EWC=8
- Without meta-learning, surprise is detrimental, not helpful
- If meta-learning is enabled, re-evaluate as surprise may provide benefits in that context

## Comparison with Ablation Results

| Config | Forgetting | Avg MSE |
|--------|------------|---------|
| VISÃO-full (sg=3, ewc=8, meta=1) | 0.053 | 0.607 |
| VISÃO-no_surprise (sg=0, ewc=8, meta=0) | 0.013 | 0.533 |
| VISÃO-no_consolidation (sg=3, ewc=0, meta=0) | 0.089 | 0.654 |
| **Optimal (sg=0, ewc=20, meta=0)** | **0.011** | **0.528** |

The optimal configuration from this sweep outperforms all ablation variants.
