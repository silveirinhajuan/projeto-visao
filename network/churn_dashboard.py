"""
churn_dashboard.py — A métrica que mata projetos voluntários.

O Folding@home atraiu 400k voluntários em março de 2020 (1,22 exaflops) e
caiu para 17 PFLOPS em 2025: 98,6% de perda. A causa não foi recrutamento,
foi churn descoberto tarde. Este módulo mede, desde o nó zero:

  - nós ativos em janela de 7 dias (rolling 7d active)
  - churn mensal (fração dos ativos no mês que some no seguinte)
  - N* observado vs previsto  (N* calculado = entrada / churn)

Sem dados reais ainda: o esquema é SQLite e uma série sintética (que segue
exatamente a ODE de retenção) valida a matemática antes de qualquer nó vivo.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS heartbeat (
    node_id TEXT NOT NULL,
    day     TEXT NOT NULL,
    PRIMARY KEY (node_id, day)
);
"""


def _iso(d) -> str:
    return d.isoformat() if isinstance(d, date) else str(d)


def _month_bounds(year: int, month: int):
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start.isoformat(), end.isoformat()


def _iter_months(year: int, month: int, n: int):
    for i in range(n):
        m = month + i
        y = year + (m - 1) // 12
        mm = (m - 1) % 12 + 1
        yield y, mm


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def record_heartbeat(conn: sqlite3.Connection, node_id: str, day) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO heartbeat(node_id, day) VALUES (?, ?)",
        (node_id, _iso(day)),
    )
    conn.commit()


def active_in_window(conn: sqlite3.Connection, ref_date, window_days: int = 7) -> int:
    """Conta nós distintos com heartbeat na janela de `window_days` até ref_date."""
    ref = ref_date if isinstance(ref_date, date) else date.fromisoformat(str(ref_date))
    start = (ref - timedelta(days=window_days - 1)).isoformat()
    end = _iso(ref)
    cur = conn.execute(
        "SELECT COUNT(DISTINCT node_id) FROM heartbeat WHERE day BETWEEN ? AND ?",
        (start, end),
    )
    return cur.fetchone()[0]


def monthly_active(conn: sqlite3.Connection, year: int, month: int) -> int:
    s, e = _month_bounds(year, month)
    cur = conn.execute(
        "SELECT COUNT(DISTINCT node_id) FROM heartbeat WHERE day >= ? AND day < ?",
        (s, e),
    )
    return cur.fetchone()[0]


def _overlap(conn: sqlite3.Connection, y1: int, m1: int, y2: int, m2: int) -> int:
    s1, e1 = _month_bounds(y1, m1)
    s2, e2 = _month_bounds(y2, m2)
    cur = conn.execute(
        """
        SELECT COUNT(DISTINCT b1.node_id)
        FROM heartbeat b1
        WHERE b1.day >= ? AND b1.day < ?
          AND EXISTS (
            SELECT 1 FROM heartbeat b2
            WHERE b2.node_id = b1.node_id AND b2.day >= ? AND b2.day < ?
          )
        """,
        (s1, e1, s2, e2),
    )
    return cur.fetchone()[0]


def monthly_churn_rate(conn: sqlite3.Connection, year: int, month: int) -> float:
    """Fração dos nós ativos no mês que NÃO aparecem no mês seguinte."""
    a = monthly_active(conn, year, month)
    if a == 0:
        return 0.0
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    overlap = _overlap(conn, year, month, ny, nm)
    return (a - overlap) / a


def monthly_series(conn: sqlite3.Connection, year: int, month: int, n_months: int):
    return [monthly_active(conn, y, m) for y, m in _iter_months(year, month, n_months)]


def steady_state_predicted(monthly_inflow: float, monthly_churn: float) -> float:
    return monthly_inflow / monthly_churn if monthly_churn > 0 else float("inf")


def estimate_params(conn: sqlite3.Connection, year: int, month: int, n_months: int = 12):
    series = monthly_series(conn, year, month, n_months)
    churns = []
    for y, m in _iter_months(year, month, n_months - 1):
        churns.append(monthly_churn_rate(conn, y, m))
    churn = sum(churns) / len(churns) if churns else 0.0
    inflows = []
    for i in range(len(series) - 1):
        n_t, n_t1 = series[i], series[i + 1]
        inflows.append(n_t1 - n_t * (1 - churn))
    inflow = sum(inflows) / len(inflows) if inflows else 0.0
    return {"monthly_churn": churn, "monthly_inflow": inflow}


def steady_state_observed(conn: sqlite3.Connection, year: int, month: int, n_months: int = 12):
    series = monthly_series(conn, year, month, n_months)
    half = max(1, len(series) // 2)
    tail = series[-half:]
    return sum(tail) / len(tail)


def compare_nstar(conn: sqlite3.Connection, year: int, month: int, n_months: int = 12):
    params = estimate_params(conn, year, month, n_months)
    pred = steady_state_predicted(params["monthly_inflow"], params["monthly_churn"])
    obs = steady_state_observed(conn, year, month, n_months)
    return {
        "observed": obs,
        "predicted": pred,
        "ratio": obs / pred if pred and pred != float("inf") else None,
        "monthly_churn": params["monthly_churn"],
        "monthly_inflow": params["monthly_inflow"],
    }


def seed_synthetic(conn: sqlite3.Connection, *, initial: int, monthly_inflow: int,
                   monthly_churn: float, months: int, start_year: int = 2025,
                   start_month: int = 1, day_of_month: int = 15) -> list[int]:
    """Popula o DB com heartbeats cujos totais mensais seguem a ODE de retenção
    exatamente: active_{t+1} = active_t*(1-churn) + inflow. Assim o N* observado
    tem de bater com o previsto (inflow/churn)."""
    init_db(conn)
    counts = []
    n = float(initial)
    for _ in range(months):
        counts.append(int(round(n)))
        n = n * (1 - monthly_churn) + monthly_inflow

    alive: list[str] = [f"node_{i}" for i in range(counts[0])]
    counter = counts[0]
    for t, (y, m) in enumerate(_iter_months(start_year, start_month, months)):
        day = date(y, m, day_of_month)
        for node in alive:
            record_heartbeat(conn, node, day)
        if t < months - 1:
            survivors = alive[: int(round(counts[t] * (1 - monthly_churn)))]
            new_count = counts[t + 1] - len(survivors)
            new_nodes = [f"node_{counter + i}" for i in range(max(0, new_count))]
            counter += max(0, new_count)
            alive = survivors + new_nodes
    return counts


def main() -> None:
    conn = sqlite3.connect(":memory:")
    counts = seed_synthetic(conn, initial=2000, monthly_inflow=200,
                            monthly_churn=0.2, months=60)
    res = compare_nstar(conn, 2025, 1, n_months=60)
    active_7d = active_in_window(conn, date(2025, 12, 15), window_days=7)
    report = {
        "serie_sintetica_meses": len(counts),
        "ativos_7d_exemplo": active_7d,
        "churn_mensal_medio": round(res["monthly_churn"], 4),
        "entrada_mensal_media": round(res["monthly_inflow"], 1),
        "nstar_observado": round(res["observed"], 1),
        "nstar_previsto": round(res["predicted"], 1),
        "razao_observado_sobre_previsto": round(res["ratio"], 4),
    }
    out = Path(__file__).parent / "results_churn.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("=" * 70)
    print("DASHBOARD DE CHURN — VISÃO (série sintética de validação)")
    print("=" * 70)
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nresultados -> {out}")


if __name__ == "__main__":
    main()
