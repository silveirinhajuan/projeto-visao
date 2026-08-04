#!/usr/bin/env python3
"""
sentinel.py — Levantamento de estado do Projeto VISÃO.

Roda ANTES do agente autônomo e entrega um relatório factual: o que passa,
o que quebrou, qual a próxima tarefa desbloqueada, quanto recurso há na
máquina. O agente não adivinha o estado do projeto — ele lê medições.

Saída: JSON em stdout (consumido pelo cron) + resumo humano em stderr.

Princípio: este script NUNCA modifica o projeto. Só observa.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ}
ENV.pop("PYTHONPATH", None)  # vazamento conhecido do Hermes quebra subprocessos

# OPS.2 — higiene dos próprios logs. Só poda o diretório de saída do sentinela;
# jamais toca código-fonte, pesquisa ou fronteira de governança.
LOG_DIR = ROOT / "ops" / "logs"
LOG_GLOB = "sentinel_*.json"
DEFAULT_KEEP = 50


def run(cmd: list[str], timeout: int = 300) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=timeout
        )
        return p.returncode, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        return 124, f"TIMEOUT após {timeout}s"
    except Exception as e:  # noqa: BLE001
        return 1, f"ERRO ao executar: {e}"


# ─────────────────────────────────────────────────────────── verificações

def governance_intact(code: int, out: str) -> bool:
    """R3: a fronteira confere com o manifesto? Pura: não faz I/O."""
    return code == 0 and "INTACTA" in out


def suite_green(exit_code: int, failed: int) -> bool:
    """Suíte verde? Pura: verde == saiu 0 e zero falhas."""
    return exit_code == 0 and failed == 0


def resources_safe(mem_available_gb, swap_used_pct) -> bool:
    """Trabalho pesado é seguro? Pura: RAM >= 1.5GB e swap < 85%."""
    return (
        isinstance(mem_available_gb, (int, float))
        and mem_available_gb >= 1.5
        and isinstance(swap_used_pct, (int, float))
        and swap_used_pct < 85
    )


# ──────────────────────────────────────────────── rotação de logs (OPS.2)

def select_logs_to_prune(logs_dir: Path, keep: int = DEFAULT_KEEP) -> list[Path]:
    """Caminhos dos logs MAIS ANTIGOS a apagar, mantendo os `keep` recentes.

    Ordena por nome (timestamp embutido em sentinel_YYYYMMDD_HHMMSS.json)
    e desempata por mtime. Pura: só lê o diretório, não apaga.
    """
    files = [p for p in Path(logs_dir).glob(LOG_GLOB) if p.is_file()]
    files.sort(key=lambda p: (p.name, p.stat().st_mtime))
    excess = len(files) - keep
    return files[:excess] if excess > 0 else []


def rotate_logs(logs_dir: Path, keep: int = DEFAULT_KEEP) -> int:
    """Mantém os `keep` logs mais recentes e apaga o resto. Retorna nº apagado."""
    to_prune = select_logs_to_prune(logs_dir, keep)
    for p in to_prune:
        p.unlink()
    return len(to_prune)


def check_governance() -> dict:
    """R3 — a fronteira imutável continua intacta? Esta é a checagem nº1."""
    code, out = run([sys.executable, "-c",
                     "import sys; sys.path.insert(0,'.');"
                     "from visao.governance.containment import BoundaryMonitor;"
                     "BoundaryMonitor('visao/governance/manifest.json').verify();"
                     "print('INTACTA')"])
    return {
        "intact": governance_intact(code, out),
        "detail": out.strip()[-500:],
    }


def check_tests() -> dict:
    code, out = run([sys.executable, "-m", "pytest", "-q", "--tb=line"], timeout=900)
    passed = failed = 0
    for line in out.splitlines():
        if " passed" in line or " failed" in line:
            for tok, label in ((" passed", "p"), (" failed", "f")):
                if tok in line:
                    try:
                        n = int(line.split(tok)[0].split()[-1])
                        if label == "p":
                            passed = max(passed, n)
                        else:
                            failed = max(failed, n)
                    except (ValueError, IndexError):
                        pass
    return {
        "exit_code": code,
        "passed": passed,
        "failed": failed,
        "green": suite_green(code, failed),
        "tail": out.strip()[-1200:],
    }


def check_phase0() -> dict:
    """O resultado fundador do projeto ainda se sustenta?"""
    res = ROOT / "prototype" / "results_continual.json"
    if not res.exists():
        return {"available": False}
    try:
        d = json.loads(res.read_text())["summary"]
        pf, nf = d["plastic"]["mean_forgetting"], d["naive"]["mean_forgetting"]
        return {
            "available": True,
            "plastic_forgetting": pf,
            "naive_forgetting": nf,
            "reduction_pct": (1 - pf / nf) * 100 if nf > 0 else None,
            "hypothesis_holds": pf < nf,
        }
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": str(e)}


def next_task(path: Path | None = None) -> dict | None:
    """Primeira tarefa pending cujas dependências estão todas done.

    Lê de `path` se fornecido (usado nos testes com backlogs falsos),
    senão do BACKLOG.json real do projeto.
    """
    try:
        backlog = json.loads((path or (ROOT / "BACKLOG.json")).read_text())
    except Exception:  # noqa: BLE001
        return None
    done = {t["id"] for t in backlog["tasks"] if t["status"] == "done"}
    for t in backlog["tasks"]:
        if t["status"] == "pending" and all(d in done for d in t.get("deps", [])):
            return t
    return None


def backlog_progress() -> dict:
    try:
        tasks = json.loads((ROOT / "BACKLOG.json").read_text())["tasks"]
    except Exception:  # noqa: BLE001
        return {}
    from collections import Counter

    c = Counter(t["status"] for t in tasks)
    return {"total": len(tasks), **dict(c)}


def check_git() -> dict:
    code, out = run(["git", "log", "--oneline", "-5"])
    _, dirty = run(["git", "status", "--porcelain"])
    return {
        "recent": out.strip().splitlines()[:5] if code == 0 else [],
        "uncommitted": len([x for x in dirty.splitlines() if x.strip()]),
    }


def check_resources() -> dict:
    out: dict[str, float | bool | None] = {"cores": os.cpu_count()}
    try:
        mem = Path("/proc/meminfo").read_text()
        vals = {
            l.split(":")[0]: int(l.split()[1])
            for l in mem.splitlines()
            if l.split(":")[0] in ("MemTotal", "MemAvailable", "SwapFree", "SwapTotal")
        }
        out["mem_available_gb"] = round(vals.get("MemAvailable", 0) / 1048576, 2)
        out["mem_total_gb"] = round(vals.get("MemTotal", 0) / 1048576, 2)
        swap_t = vals.get("SwapTotal", 0)
        out["swap_used_pct"] = (
            round(100 * (1 - vals.get("SwapFree", 0) / swap_t), 1) if swap_t else 0.0
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        du = shutil.disk_usage(ROOT)
        out["disk_free_gb"] = round(du.free / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    # Regra dura: com <1.5GB livre ou swap >85%, builds pesados morrem em silêncio.
    mem_free = out.get("mem_available_gb")
    swap_pct = out.get("swap_used_pct")
    out["heavy_work_safe"] = resources_safe(mem_free, swap_pct)
    return out


def main() -> int:
    t0 = time.time()
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "governance": check_governance(),
        "tests": check_tests(),
        "phase0": check_phase0(),
        "next_task": next_task(),
        "backlog": backlog_progress(),
        "git": check_git(),
        "resources": check_resources(),
    }

    # PARADA DURA: fronteira violada tranca tudo.
    report["safe_to_proceed"] = (
        report["governance"]["intact"] and report["resources"]["heavy_work_safe"]
    )
    report["elapsed_s"] = round(time.time() - t0, 1)

    # OPS.2 — poda higiênica dos PRÓPRIOS logs (mantém 50 mais recentes).
    # Não é parada dura: só limpa a saída do sentinela, nunca o projeto.
    report["log_rotation"] = {"deleted": rotate_logs(LOG_DIR)}

    print(json.dumps(report, indent=2, ensure_ascii=False))

    g, t, r = report["governance"], report["tests"], report["resources"]
    nt = report["next_task"]
    print(
        f"\n[sentinel] governança={'OK' if g['intact'] else 'VIOLADA'} "
        f"testes={t['passed']}p/{t['failed']}f "
        f"ram={r.get('mem_available_gb')}GB swap={r.get('swap_used_pct')}% "
        f"proxima={nt['id'] if nt else 'nenhuma'} "
        f"prosseguir={report['safe_to_proceed']} "
        f"rot={report['log_rotation']['deleted']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
