#!/usr/bin/env python3
"""
nightly_brief.py — Contexto pré-computado para o agente autônomo do VISÃO.

Executa o sentinela, adiciona um checkpoint git de segurança e emite um
briefing enxuto. O agente recebe FATOS, não precisa gastar chamadas de
ferramenta para descobrir onde o projeto está.

Chamado pelo cron via o campo `script`. Stdout entra no prompt do agente.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/juan/projeto-visao")
LOGS = ROOT / "ops" / "logs"
ENV = {**os.environ}
ENV.pop("PYTHONPATH", None)


def sh(cmd: list[str], timeout: int = 600) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout
        )
        return p.returncode, p.stdout + p.stderr
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    code, out = sh([sys.executable, "ops/sentinel.py"])
    try:
        report = json.loads(out.split("\n[sentinel]")[0])
    except Exception:  # noqa: BLE001
        print("SENTINELA FALHOU — o agente NÃO deve modificar código nesta sessão.")
        print(out[-2000:])
        return 0

    (LOGS / f"sentinel_{stamp}.json").write_text(json.dumps(report, indent=2))

    # Checkpoint de segurança: qualquer trabalho não commitado vira um commit-wip
    # antes do agente tocar em nada. Reverter fica trivial.
    if report["git"]["uncommitted"] > 0:
        sh(["git", "add", "-A"])
        sh(["git", "-c", "user.name=VISAO Autonomo",
            "-c", "user.email=autonomo@projeto-visao.local",
            "commit", "-q", "-m", f"chore(wip): checkpoint automático {stamp}"])

    _, head = sh(["git", "rev-parse", "--short", "HEAD"])

    g, t, r, p = report["governance"], report["tests"], report["resources"], report["phase0"]
    nt = report["next_task"]
    bl = report["backlog"]

    print("=" * 66)
    print(f"BRIEFING PROJETO VISÃO — {report['timestamp']}")
    print("=" * 66)
    print(f"HEAD git         : {head.strip()}")
    print(f"Governança (R3)  : {'INTACTA' if g['intact'] else '*** VIOLADA ***'}")
    print(f"Testes           : {t['passed']} passam, {t['failed']} falham "
          f"({'VERDE' if t['green'] else 'VERMELHO'})")
    if p.get("available"):
        print(f"Fase 0 (fundação): redução de esquecimento {p['reduction_pct']:.1f}% "
              f"— hipótese {'sustentada' if p['hypothesis_holds'] else 'FALSIFICADA'}")
    print(f"Backlog          : {bl.get('done',0)}/{bl.get('total',0)} concluídas")
    print(f"Recursos         : {r.get('mem_available_gb')}GB RAM livre, "
          f"swap {r.get('swap_used_pct')}%, disco {r.get('disk_free_gb')}GB")
    print(f"Trabalho pesado  : {'PERMITIDO' if r.get('heavy_work_safe') else 'PROIBIDO (recursos baixos)'}")
    print()

    if not report["safe_to_proceed"]:
        print(">>> PARADA DURA — NÃO MODIFIQUE CÓDIGO NESTA SESSÃO. <<<")
        if not g["intact"]:
            print("Motivo: a fronteira de governança R3 divergiu do manifesto.")
            print(f"Detalhe: {g['detail'][:400]}")
            print("Ação: relate ao Juan e PARE. Não reseleccione o manifesto sozinho.")
        else:
            print("Motivo: recursos insuficientes (RAM/swap). Builds morrem em silêncio aqui.")
            print("Ação: relate o estado e pare. Não force instalações.")
        return 0

    if not t["green"]:
        print(">>> SUITE VERMELHA — a tarefa desta sessão é CONSERTAR, não avançar. <<<")
        print(t["tail"][-900:])
        return 0

    if nt:
        print(f"PRÓXIMA TAREFA DESBLOQUEADA: {nt['id']} — {nt['title']}")
        print(f"  arquivos : {', '.join(nt.get('files', [])) or '—'}")
        if nt.get("verify"):
            print(f"  verificar: {nt['verify']}")
        if nt.get("expect"):
            print(f"  esperado : {nt['expect']}")
        if nt.get("notes"):
            print(f"  notas    : {nt['notes']}")
    else:
        print("Nenhuma tarefa desbloqueada. Fase concluída ou dependências travadas.")
        print("Ação: revise o BACKLOG.json e proponha as próximas tarefas ao Juan.")

    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
