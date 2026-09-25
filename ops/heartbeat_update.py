#!/usr/bin/env python3
"""Watchdog heartbeat: append cycle log to task 76 evidence in BACKLOG.json."""
import json
import sys

PATH = "/home/juan/projeto-visao/BACKLOG.json"

CYCLE = (
    " CICLO 24/09 20:59 (HEAD 01089ed, sem mudanças desde 16:20): backlog sem pendentes "
    "(75 done / 6 discontinued / 2 blocked-ação-humana); fast 186/0 em 71.8s; flake8 0; "
    "mypy 5 erros (todos em arquivos imutáveis: 2 governance/ R3 + 3 prototype/ legacy, "
    "zero regressões); disco estável em 88% (28G livres, purga anterior sustentada); "
    "RAM ok (4,8G available). Heartbeat registrado."
)

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

task = None
for t in data["tasks"]:
    if t.get("id") == 76 or t.get("id") == "76":
        task = t
        break

if task is None:
    print("ERROR: task 76 not found")
    sys.exit(1)

# Anexa ao evidence (onde vivem os ciclos recentes), mantendo só os 2 anteriores
old = task.get("evidence", "")
cycles = [c for c in old.split("CICLO") if c.strip()]
kept = cycles[-2:] if len(cycles) > 2 else cycles
task["evidence"] = "CICLO".join(kept) + CYCLE

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")

# Verificação: reler do disco e confirmar persistência
with open(PATH, "r", encoding="utf-8") as f:
    check = f.read()
assert "CICLO 24/09 20:59" in check, "FALHOU: ciclo não persistiu no disco"

print(f"OK: evidence {len(old)} -> {len(task['evidence'])} chars")
print("Last 120 chars:", task["evidence"][-120:])
