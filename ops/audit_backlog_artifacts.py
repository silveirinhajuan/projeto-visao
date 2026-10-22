#!/usr/bin/env python3
"""Auditoria de integridade: artefatos declarados no BACKLOG x disco.

Correcao da versao anterior (21/10/2026 01:12), que reportava **58 de 89**
artefatos como ausentes. Esse numero era falso. A causa raiz:

    git log --all --diff-filter=D -- <path>

Sem `-M` e sem `-c` o git nao entende RENAMES. Um arquivo que foi MOVIDO
(renomeado) aparece como `D` no path antigo, e o script concluia "este artefato
foi apagado por <commit>". Na verdade o conteudo so mudou de lugar.

Prova: os 4 docs que o relatorio culpava o commit 6af0c10 —
AGENTE_AUTOCONTIDO.md, FASE2_ENXAME.md, FASE3_EVOLUCAO.md, PROPOSTA_PESQUISA.md —
existem HOJE em `docs/` (git ls-files confirma: TRACKED). O commit 6af0c10 os
renomeou com similaridade 100%:

    $ git show -M --name-status 6af0c10 | grep R100
    R100  PROPOSTA_PESQUISA.md  docs/PROPOSTA_PESQUISA.md

Esta versao classifica cada path declarado em TRES categorias:
  - PRESENT   : existe no path declarado
  - RELOCATED : nao existe no path declarado, mas o conteudo existe em outro
                lugar do repo (rename com -M, ou basename encontrada)
  - MISSING   : nao existe em lugar nenhum (ausencia real)

O que interessa ao loop autonomo e' MISSING — evidencia genuinamente perdida.

Uso de `--raw` em vez de `--name-status`: o raw usa espacos como separador e
nao emite linha em branco entre commits, o que torna o parsing inambiguo.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict

ROOT = "/home/juan/projeto-visao"


def run(args: list[str]) -> str:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout


def declared_paths() -> dict[str, list[str]]:
    """path -> [task ids] a partir do BACKLOG.json."""
    with open(os.path.join(ROOT, "BACKLOG.json"), encoding="utf-8") as fh:
        backlog = json.load(fh)
    tasks = backlog["tasks"] if isinstance(backlog, dict) and "tasks" in backlog else backlog
    decl: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        for path in task.get("files", []) or []:
            decl[path].append(str(task.get("id")))
    return dict(decl)


def deleting_commits(path: str) -> list[tuple[str, str]]:
    """Commits (hash, assunto) que removeram `path` — obtidos com filtro de path.

    ATENCAO: `git log --diff-filter=D -- <path>` so consegue ver UM lado do par
    de rename. Um arquivo MOVIDO aparece aqui como `D` no path antigo. E'
    exatamente por isso que a auditoria original mentiu: ela tratava esse `D`
    como remocao real. A confirmacao de rename exige o diff SEM filtro (veja
    `find_move`).
    """
    out = run(["git", "log", "--all", "--format=%H\t%s", "--diff-filter=D", "--", path])
    commits: list[tuple[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        sha, _, subject = line.partition("\t")
        commits.append((sha, subject))
    return commits


def find_move(path: str) -> tuple[str, str] | None:
    """Se `path` foi MOVIDO por um rename, devolve (commit, destino).

    Estrategia em dois passos, porque `git log -- <path>` filtra um lado so' do
    par e por isso o `-M` nunca dispara com filtro de path:

      1. descobre QUAL commit removeu o path (filtro de path, `--diff-filter=D`);
      2. pede o diff DESSE commit SEM filtro (`git show -M --raw`), onde o git
         consegue emparelhar origem e destino e emite `R100\\0<origem>\\0<destino>`.

    Sem o passo 2, todo rename e' reportado como remocao — o bug original.
    """
    for sha, subject in deleting_commits(path):
        # `-r` e' obrigatorio (sem ele o diff sai agregado por diretorio).
        # Layout com -z: tokens[0]=sha do commit; depois, para cada mudanca,
        # um token `:modo modo sha sha STATUS` seguido de 1 path (A/D/M) ou
        # 2 paths (R/C). Sem `-r` o rename nao aparece.
        tokens = run(["git", "diff-tree", "-r", "-M", "--raw", "-z", sha]).split("\0")
        label = f"{sha[:9]} {subject}"
        i = 1
        while i < len(tokens):
            tok = tokens[i]
            if not tok.startswith(":"):
                i += 1
                continue
            parts = tok.split()
            status = parts[-1] if parts else ""
            if status.startswith("R") and i + 2 < len(tokens):
                old_path, new_path = tokens[i + 1], tokens[i + 2]
                if old_path == path:
                    return label, new_path
                i += 3
            else:
                i += 2
    return None


def basename_index() -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for fn in filenames:
            index[fn].append(os.path.relpath(os.path.join(dirpath, fn), ROOT))
    return index


def main() -> int:
    os.chdir(ROOT)
    decl = declared_paths()
    index = basename_index()

    present: list[str] = []
    relocated: list[tuple[str, str, str]] = []
    missing: list[tuple[str, str]] = []

    for path in sorted(decl):
        tasks = decl[path]
        if os.path.exists(os.path.join(ROOT, path)):
            present.append(path)
            continue

        move = find_move(path)
        base_hits = [h for h in index.get(os.path.basename(path), []) if h != path]

        if move:
            relocated.append((path, move[1], f"{move[0]}"))
        elif base_hits:
            relocated.append((path, base_hits[0], "basename"))
        else:
            deleting = deleting_commits(path)
            missing.append((path, deleting[0][1][:60] if deleting else "(nunca versionado)"))

    print("=" * 78)
    print("AUDITORIA DE INTEGRIDADE — BACKLOG x DISCO")
    print("=" * 78)
    print(f"caminhos declarados em `files` : {len(decl)}")
    print(f"  PRESENT   (existe)            : {len(present)}")
    print(f"  RELOCATED (movido, existe)    : {len(relocated)}")
    print(f"  MISSING   (ausencia real)     : {len(missing)}")
    print()

    if relocated:
        print("-" * 78)
        print("RELOCATED — o path declarado nao existe, MAS o conteudo esta no repo")
        print("-" * 78)
        for path, dest, via in relocated:
            print(f"  {path}")
            print(f"     -> {dest}   (via {via})   <- tasks {','.join(decl[path])}")
        print()

    if missing:
        print("-" * 78)
        print("MISSING — evidencia genuinamente perdida (sem destino no repo)")
        print("-" * 78)
        for path, deleting in missing:
            print(f"  {path}   <- tasks {','.join(decl[path])}   [{deleting}]")
        print()

    print("-" * 78)
    print("Referencias de import a modulos ausentes")
    print("-" * 78)
    import re

    mods = {p[:-3].replace("/", ".") for p, _ in missing if p.endswith(".py")}
    hits = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, "visao")):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            src_path = os.path.join(dirpath, fn)
            try:
                with open(src_path, encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
            except OSError:
                continue
            for mod in mods:
                if re.search(r"\b(import|from)\s+" + re.escape(mod) + r"\b", src):
                    hits.append((os.path.relpath(src_path, ROOT), mod))
    for src_path, mod in sorted(hits):
        print(f"  {src_path}  ->  {mod}")
    if not hits:
        print("  (nenhuma)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
