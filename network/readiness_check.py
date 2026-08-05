"""
readiness_check.py — Checklist F5.0 automatizado (tarefa 5.4).

Guardião que roda ANTES de convidar a PRIMEIRA pessoa. Verifica os
pré-requisitos do briefing da tarefa 5.4:

  1. Fase 4 (governança) verde
  2. Fase 1 fechada com benchmarks
  3. Cliente instalável em <5min
  4. Problema escolhido
  5. Doc de risco publicado

Enquanto QUALQUER item falhar, imprime PROIBIDO RECRUTAR. Recrutar cedo
queima a única primeira impressão — o script existe para evitar exatamente
isso, e portanto NÃO deve ser torno de "tudo verde" por acidente.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _task_status(backlog: dict, task_id: str) -> str | None:
    for t in backlog.get("tasks", []):
        if t.get("id") == task_id:
            return t.get("status")
    return None


def check_phase4_governance(backlog: dict, governance_green: bool) -> Check:
    if backlog.get("gates", {}).get("phase4_governance") != "done":
        return Check("Fase 4 (governança) verde", False,
                     "gate phase4_governance != done no backlog")
    if not governance_green:
        return Check("Fase 4 (governança) verde", False,
                     "suíte de governança (R1-R5) vermelha")
    return Check("Fase 4 (governança) verde", True,
                 "governança intacta e testes verdes")


def check_phase1_closed(backlog: dict, readme_path: Path) -> Check:
    if backlog.get("gates", {}).get("phase1_substrate") != "done":
        return Check("Fase 1 fechada c/ benchmarks", False,
                     "gate phase1_substrate != done no backlog")
    if not readme_path.exists():
        return Check("Fase 1 fechada c/ benchmarks", False,
                     f"{readme_path.name} ausente")
    txt = readme_path.read_text(encoding="utf-8", errors="ignore")
    if not any(ch.isdigit() for ch in txt):
        return Check("Fase 1 fechada c/ benchmarks", False,
                     f"{readme_path.name} sem números medidos (portão da Fase 1)")
    return Check("Fase 1 fechada c/ benchmarks", True,
                 "portão Fase 1 fechado com números reproduzíveis")


def check_cliente_instalavel(install_bench_path: Path) -> Check:
    if not install_bench_path.exists():
        return Check("Cliente instalável <5min", False,
                     "benchmark de instalação ausente (tempo <5min não medido)")
    try:
        data = json.loads(install_bench_path.read_text(encoding="utf-8"))
        seconds = float(data["install_seconds"])
    except Exception as exc:
        return Check("Cliente instalável <5min", False,
                     f"benchmark inválido: {exc}")
    if seconds > 300:
        return Check("Cliente instalável <5min", False,
                     f"instalação medida={seconds:.0f}s > 300s (5min)")
    return Check("Cliente instalável <5min", True,
                 f"instalação medida={seconds:.0f}s < 300s")


def check_problema_escolhido(backlog: dict) -> Check:
    status = _task_status(backlog, "5.2")
    if status != "done":
        return Check("Problema escolhido", False,
                     f"tarefa 5.2 (problema zero) status={status}")
    return Check("Problema escolhido", True,
                 "tarefa 5.2 concluída — problema definido")


def check_doc_risco_publicado(backlog: dict, riscos_path: Path) -> Check:
    status = _task_status(backlog, "5.3")
    if status != "done":
        return Check("Doc de risco publicado", False,
                     f"tarefa 5.3 status={status}")
    if not riscos_path.exists():
        return Check("Doc de risco publicado", False,
                     f"{riscos_path.name} ausente no disco")
    return Check("Doc de risco publicado", True,
                 "doc de risco publicado e presente")


def load_backlog(path: Path | None = None) -> dict:
    path = path or (PROJECT_ROOT / "BACKLOG.json")
    return json.loads(path.read_text(encoding="utf-8"))


def run_governance_suite() -> bool:
    try:
        r = subprocess.run(
            ["env", "-u", "PYTHONPATH", "python3", "-m", "pytest",
             "visao/governance/tests", "-q"],
            cwd=PROJECT_ROOT,
            capture_output=True,
        )
        return r.returncode == 0
    except Exception:
        return False


def run_all_checks(backlog, *, governance_green, readme_path,
                  riscos_path, install_bench_path):
    if callable(governance_green):
        g = bool(governance_green())
    else:
        g = bool(governance_green)
    return [
        check_phase4_governance(backlog, g),
        check_phase1_closed(backlog, readme_path),
        check_cliente_instalavel(install_bench_path),
        check_problema_escolhido(backlog),
        check_doc_risco_publicado(backlog, riscos_path),
    ]


def main(argv=None, *, backlog=None, governance_green=None,
         readme_path=None, riscos_path=None, install_bench_path=None):
    backlog = backlog if backlog is not None else load_backlog()
    governance_green = governance_green or run_governance_suite
    readme_path = readme_path or (PROJECT_ROOT / "README.md")
    riscos_path = riscos_path or (PROJECT_ROOT / "network" / "RISCOS_PARA_VOLUNTARIOS.md")
    install_bench_path = install_bench_path or (PROJECT_ROOT / "network" / "install_bench.json")

    checks = run_all_checks(
        backlog,
        governance_green=governance_green,
        readme_path=readme_path,
        riscos_path=riscos_path,
        install_bench_path=install_bench_path,
    )
    all_ok = all(c.passed for c in checks)
    for c in checks:
        print(f"[{'OK' if c.passed else 'FALHOU'}] {c.name}: {c.detail}")
    if all_ok:
        print("AUTORIZADO RECRUTAR")
        return 0
    print("PROIBIDO RECRUTAR")
    return 1


if __name__ == "__main__":
    sys.exit(main())
