#!/usr/bin/env python3
"""VISÃO watchdog heartbeat (canonical, repo-versioned).

Records one watchdog cycle:
  1. appends the full entry to ops/watchdog_cycles.log (append-only history);
  2. keeps only the 2 most recent cycles in BACKLOG.json task 76 `evidence`;
  3. updates `_last_watchdog_cycle`.

Earlier versions hardcoded a single cycle string and let heartbeats pile up in
task 76 `blocked_reason` (5841 chars / 14 cycles by 06/10/2026), bloating the
file that the autonomous loop reads. Consolidation happened 06/10/2026.

Usage: heartbeat_update.py "<cycle text>" "<dd/mm HH:MM>"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKLOG = ROOT / "BACKLOG.json"
LOG = Path(__file__).resolve().parent / "watchdog_cycles.log"

KEEP_IN_BACKLOG = 2


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: heartbeat_update.py '<cycle text>' '<dd/mm HH:MM>'")
        return 2
    cycle_text, stamp = sys.argv[1], sys.argv[2]
    entry = cycle_text.strip()
    if not entry.startswith("CICLO"):
        entry = "CICLO " + entry.lstrip()

    with BACKLOG.open("r", encoding="utf-8") as f:
        data = json.load(f)

    task = next((t for t in data["tasks"] if str(t.get("id")) == "76"), None)
    if task is None:
        print("ERROR: task 76 not found")
        return 1

    old = task.get("evidence", "")
    cycles = [c for c in old.split("CICLO") if c.strip()]
    kept = cycles[-KEEP_IN_BACKLOG:] if len(cycles) > KEEP_IN_BACKLOG else cycles
    task["evidence"] = "CICLO".join(kept) + entry
    data["_last_watchdog_cycle"] = stamp

    with BACKLOG.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with LOG.open("a", encoding="utf-8") as f:
        f.write(entry + "\n\n")

    with BACKLOG.open("r", encoding="utf-8") as f:
        check = f.read()
    marker = entry.strip().split("(")[0].strip()
    if marker not in check:
        print("FAILED: cycle not persisted in BACKLOG.json")
        return 1

    print(f"OK: evidence {len(old)} -> {len(task['evidence'])} chars")
    print(f"OK: cycle appended to {LOG.name}")
    print("tail:", task["evidence"][-160:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
