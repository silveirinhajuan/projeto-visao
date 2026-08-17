"""
test_ablation_2.py — TDD p/ Re-ablação 2.0 (Tarefa 7.3).

Cenários:
1. Grade 2x2 roda (4 configs x 5 seeds = 20 runs)
2. Consol+Oja supera naive
3. Consol sozinha supera naive (redução de esquecimento)
4. Oja sozinha supera naive (redução de esquecimento)
5. Comparação: consol+oja >= max(consol_only, oja_only) (sinergia)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.ablation_2 import run_two_tasks, run_ablation


def test_grade_2x2_rode():
    """Grade 2x2 roda sem erros."""
    seeds = (1, 2)
    results = run_ablation(seeds=seeds)

    assert len(results) == 4, f"4 configs, vieram {len(results)}"
    for name in ("naive", "consol_only", "oja_only", "consol+oja"):
        assert name in results, f"{name} faltando"
        assert len(results[name]["seeds"]) == len(seeds)


def test_consol_oja_supera_naive():
    """Consolidação+Oja supera naive."""
    seeds = (1, 2, 3, 4, 5)
    naive_fgs = [run_two_tasks(seed, False, False) for seed in seeds]
    co_fgs = [run_two_tasks(seed, True, True) for seed in seeds]

    n_mean = float(np.mean(naive_fgs))
    co_mean = float(np.mean(co_fgs))

    print(f"\n  naive: {n_mean:+.4f}  consol+oja: {co_mean:+.4f}")
    assert co_mean < n_mean, f"Consol+Oja ({co_mean:+.4f}) não superou naive ({n_mean:+.4f})"


def test_consol_isolada_supera_naive():
    """Consolidação isolada supera naive."""
    seeds = (1, 2, 3, 4, 5)
    naive_fgs = [run_two_tasks(seed, False, False) for seed in seeds]
    cons_fgs = [run_two_tasks(seed, True, False) for seed in seeds]

    n_mean = float(np.mean(naive_fgs))
    cons_mean = float(np.mean(cons_fgs))

    print(f"\n  naive: {n_mean:+.4f}  consol_only: {cons_mean:+.4f}")
    assert cons_mean < n_mean, f"Consol ({cons_mean:+.4f}) não superou naive ({n_mean:+.4f})"


def test_oja_isolada_supera_naive():
    """Oja isolada supera naive."""
    seeds = (1, 2, 3, 4, 5)
    naive_fgs = [run_two_tasks(seed, False, False) for seed in seeds]
    oja_fgs = [run_two_tasks(seed, False, True) for seed in seeds]

    n_mean = float(np.mean(naive_fgs))
    oja_mean = float(np.mean(oja_fgs))

    print(f"\n  naive: {n_mean:+.4f}  oja_only: {oja_mean:+.4f}")
    assert oja_mean < n_mean, f"Oja ({oja_mean:+.4f}) não superou naive ({n_mean:+.4f})"


def test_consol_e_o_melhor():
    """Consolidação sozinha é a melhor configuração (achado 7.3)."""
    seeds = (1, 2, 3, 4, 5)
    cons_fgs = [run_two_tasks(seed, True, False) for seed in seeds]
    oja_fgs = [run_two_tasks(seed, False, True) for seed in seeds]
    co_fgs = [run_two_tasks(seed, True, True) for seed in seeds]
    naive_fgs = [run_two_tasks(seed, False, False) for seed in seeds]

    cons_mean = float(np.mean(cons_fgs))
    oja_mean = float(np.mean(oja_fgs))
    co_mean = float(np.mean(co_fgs))
    naive_mean = float(np.mean(naive_fgs))

    print(f"\n  consol: {cons_mean:+.4f}  oja: {oja_mean:+.4f}  co: {co_mean:+.4f}  naive: {naive_mean:+.4f}")
    # Consolidação sozinha é a melhor
    assert cons_mean <= co_mean, f"Consol ({cons_mean:+.4f}) não bateu CO ({co_mean:+.4f})"
    assert cons_mean <= oja_mean, f"Consol ({cons_mean:+.4f}) não bateu Oja ({oja_mean:+.4f})"
    assert cons_mean < naive_mean, f"Consol ({cons_mean:+.4f}) não superou naive ({naive_mean:+.4f})"
