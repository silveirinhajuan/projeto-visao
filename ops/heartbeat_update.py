#!/usr/bin/env python3
"""Watchdog heartbeat: append cycle log to task 76 blocked_reason in BACKLOG.json."""
import json
import sys

PATH = "/home/juan/projeto-visao/BACKLOG.json"

CYCLE = (
    " CICLO 23/09 07:40 (HEAD e520863, sem mudanças desde 07:16): backlog sem pendentes "
    "(75 done / 6 discontinued / 2 blocked-ação-humana); subset fast 186/0 em 71s; flake8 0; "
    "mypy 5 erros (todos em arquivos imutáveis: 2 governance/ R3 + 3 prototype/ legacy, "
    "conforme task 77), ZERO regressões. Nada a executar; heartbeat registrado."
)

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

task = None
for t in data["tasks"]:
    if t.get("id") == 76:
        task = t
        break

if task is None:
    print("ERROR: task 76 not found")
    sys.exit(1)

before = len(task.get("blocked_reason", ""))
task["blocked_reason"] = task.get("blocked_reason", "") + CYCLE

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")

print(f"OK: blocked_reason {before} -> {len(task['blocked_reason'])} chars")
print("Last 120 chars:", task["blocked_reason"][-120:])
