"""
scale_ltc.py — tarefa 5.1: escala da célula líquida para o notebook.

Mede o pico de RAM de processo e o tempo por passo de integração para
n_hidden em {512, 1024}, validando que a arquitetura cabe no envelope do
notebook do autor (7.6GB RAM, CPU Ryzen, single-node). Não treina — só
exercita a dinâmica (rollout), que é o piso de memória/CPU de qualquer uso
contínuo no aprendizado incremental.

Métrica de RAM: VmRSS corrente lida de /proc/self/status (residente real do
processo, inclui o runtime XLA). Não usamos ru_maxrss: ele é o pico
MONOTÔNICO de toda a vida do processo e zeraria o delta de uma célula medida
depois de outra já ter tocado o teto. O envelope "caber em 7.6GB" é sobre o
processo inteiro rodando no notebook — por isso reportamos o RSS observado.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from visao.core.cfc import CfCCell

ROOT = Path(__file__).resolve().parents[2]
SCALE_RESULT_PATH = str(ROOT / "visao" / "analysis" / "results_scale_ltc.json")


def _rss_mb() -> float:
    """RSS residente corrente do processo em MB (VmRSS de /proc/self/status)."""
    status = Path("/proc/self/status").read_text()
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0  # unidade: kB
    raise RuntimeError("VmRSS indisponível em /proc/self/status")


def measure_cell(
    n_hidden: int,
    n_in: int = 2,
    T: int = 500,
    dt: float = 0.1,
    seed: int = 0,
) -> dict:
    """Instancia CfCCell(n_hidden) e mede pico de RAM + tempo por passo.

    Retorna: n_hidden, n_in, T, dt, rss_before_mb, rss_after_mb,
    peak_ram_mb (RSS máximo observado), cell_ram_delta_mb (custo marginal
    da célula) e time_per_step_s.
    """
    cell = CfCCell(n_in=n_in, n_hidden=n_hidden, dt=dt, seed=seed)
    rng = np.random.default_rng(seed + 1)
    seq = rng.normal(0.0, 1.0, (T, n_in)).astype(np.float32)

    rss_before = _rss_mb()

    # warmup: dispara o jit compile fora da medição de tempo
    _ = cell.rollout(seq)
    rss_mid = _rss_mb()

    t0 = __import__("time").perf_counter()
    states, _ = cell.rollout(seq)
    t1 = __import__("time").perf_counter()
    rss_after = _rss_mb()

    # força materialização para que o XLA não adie liberação de buffers
    _ = float(np.asarray(states).sum())

    peak = max(rss_before, rss_mid, rss_after)
    return {
        "n_hidden": int(n_hidden),
        "n_in": int(n_in),
        "T": int(T),
        "dt": float(dt),
        "rss_before_mb": round(rss_before, 2),
        "rss_mid_mb": round(rss_mid, 2),
        "rss_after_mb": round(rss_after, 2),
        "peak_ram_mb": round(peak, 2),
        "cell_ram_delta_mb": round(rss_after - rss_before, 2),
        "time_per_step_s": round((t1 - t0) / T, 8),
        "rollout_total_s": round(t1 - t0, 6),
    }


def run_scale(out_path: str = SCALE_RESULT_PATH) -> dict:
    """Mede n=512 e n=1024, emite JSON de resultado real, retorna o dict."""
    result = {
        "budget_mb": round(7.6 * 1024, 1),
        "n_512": measure_cell(n_hidden=512, T=500, seed=0),
        "n_1024": measure_cell(n_hidden=1024, T=500, seed=0),
    }
    Path(out_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    d = run_scale()
    for key in ("n_512", "n_1024"):
        r = d[key]
        print(
            f"{key}: peak={r['peak_ram_mb']:.1f}MB  "
            f"delta_celula={r['cell_ram_delta_mb']:.1f}MB  "
            f"tempo/passo={r['time_per_step_s'] * 1e3:.3f}ms  T={r['T']}"
        )
    fits = (
        d["n_512"]["peak_ram_mb"] < d["budget_mb"]
        and d["n_1024"]["peak_ram_mb"] < d["budget_mb"]
    )
    print(f"orçamento: {d['budget_mb']:.0f}MB  cabem: {fits}")
