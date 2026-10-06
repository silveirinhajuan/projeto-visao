#!/usr/bin/env python3
"""VISÃO watchdog heartbeat (canonical, repo-versioned).

Records one watchdog cycle. Since 06/10/2026 a heartbeat writes ONLY to the
append-only, NON-versioned log `ops/watchdog_cycles.log` (covered by the repo's
`*.log` ignore rule). It no longer touches `BACKLOG.json`.

Why: the previous version updated `BACKLOG.json` (`_last_watchdog_cycle` plus a
rolling 2-cycle `evidence` window) on every cycle. With the watchdog running
every ~20-30 min that produced ~25 commits/DAY that changed nothing but that one
file, drowning real work in `git log` and rewriting the very file the autonomous
loop re-reads each cycle. Health history belongs in an append-only log, not in
git history. Liveness is independently recorded by the scheduler in
`~/.hermes/cron/executions.db`.

Usage: heartbeat_update.py "<cycle text>" "<dd/mm HH:MM>"
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = Path(__file__).resolve().parent / "watchdog_cycles.log"

# Fortaleza (GMT-3), where the project owner and the cron schedule live.
TZ = timezone(timedelta(hours=-3))


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: heartbeat_update.py '<cycle text>' '<dd/mm HH:MM>'")
        return 2
    cycle_text, stamp = sys.argv[1], sys.argv[2]
    entry = cycle_text.strip()
    if not entry.startswith("CICLO"):
        entry = "CICLO " + entry.lstrip()
    if not entry.endswith(stamp):
        entry = f"{entry} [{stamp}]"

    LOG.parent.mkdir(parents=True, exist_ok=True)
    before = LOG.stat().st_size if LOG.exists() else 0
    with LOG.open("a", encoding="utf-8") as f:
        f.write(entry + "\n\n")

    # Verify the append actually landed (same guard the old version had).
    text = LOG.read_text(encoding="utf-8")
    marker = entry.strip().split("(")[0].strip()
    if marker not in text:
        print("FAILED: cycle not persisted in the log")
        return 1

    after = LOG.stat().st_size
    print(f"OK: {after - before} bytes appended to {LOG.name} ({after} total)")
    print(f"OK: BACKLOG.json untouched (heartbeats are not versioned)")
    print(f"at {datetime.now(TZ).strftime('%d/%m/%Y %H:%M')} GMT-3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
