"""Regression tests for ops/audit_backlog_artifacts.py.

Contexto (21/10/2026): a auditoria original reportava **58 de 89** artefatos
declarados no BACKLOG como ausentes em disco. O numero era falso. A causa era
`git log --diff-filter=D` sem deteccao de rename: um arquivo MOVIDO aparece como
deletado no path antigo, e o script o contava como evidencia perdida.

Estes testes fixam a distincao entre:
  - RELOCATED : o path declarado sumiu, mas o conteudo existe em outro lugar
  - MISSING   : o path declarado sumiu e nao ha destino no repo

Sem eles, o bug volta silenciosamente.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIT = os.path.join(ROOT, "ops", "audit_backlog_artifacts.py")

spec = importlib.util.spec_from_file_location("audit_backlog_artifacts", AUDIT)
if spec is None or spec.loader is None:
    raise ImportError(f"nao foi possivel carregar {AUDIT}")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def git(*args: str, cwd: str = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    ).stdout


def set_root(path: str) -> None:
    """O modulo guarda ROOT como constante de modulo; reatribuir via setattr
    evita que o type checker reclame de atributo desconhecido."""
    setattr(audit, "ROOT", path)


@pytest.fixture()
def relocated_repo(tmp_path):
    """Repo git com um rename R100: docs/A.md -> A.md (o path antigo 'some' some')."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git("init", "-q", cwd=str(repo))
    git("config", "user.email", "t@t.t", cwd=str(repo))
    git("config", "user.name", "t", cwd=str(repo))

    (repo / "docs").mkdir()
    (repo / "docs" / "A.md").write_text("conteudo original\n", encoding="utf-8")
    git("add", "-A", cwd=str(repo))
    git("commit", "-q", "-m", "add", cwd=str(repo))

    # move para a raiz — o path declarado deixa de existir
    (repo / "A.md").write_text("conteudo original\n", encoding="utf-8")
    (repo / "docs" / "A.md").unlink()
    git("add", "-A", cwd=str(repo))
    git("commit", "-q", "-m", "move", cwd=str(repo))
    return repo


@pytest.fixture()
def deleted_repo(tmp_path):
    """Repo git onde o arquivo e' genuinamente removido (sem destino)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git("init", "-q", cwd=str(repo))
    git("config", "user.email", "t@t.t", cwd=str(repo))
    git("config", "user.name", "t", cwd=str(repo))

    (repo / "gone.py").write_text("x = 1\n", encoding="utf-8")
    git("add", "-A", cwd=str(repo))
    git("commit", "-q", "-m", "add", cwd=str(repo))

    (repo / "gone.py").unlink()
    git("add", "-A", cwd=str(repo))
    git("commit", "-q", "-m", "delete", cwd=str(repo))
    return repo


def test_rename_is_detected_as_move(relocated_repo):
    """Um rename R100 deve ser devolvido por find_move, nao tratado como delete."""
    set_root(str(relocated_repo))
    move = audit.find_move("docs/A.md")
    assert move is not None, "rename R100 nao detectado — o bug original"
    commit, dest = move
    assert dest == "A.md", f"destino inesperado: {dest!r}"
    assert commit.startswith("move") or "move" in commit


def test_delete_is_not_a_move(deleted_repo):
    """Uma remocao real nao deve ser confundida com rename."""
    set_root(str(deleted_repo))
    assert audit.find_move("gone.py") is None


def test_missing_path_is_labelled_never_tracked(tmp_path):
    """Path que nunca existiu nao deve produzir destino nem commit de delete."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git("init", "-q", cwd=str(repo))
    set_root(str(repo))
    assert audit.find_move("nunca/existiu.py") is None


def test_classify_on_real_backlog():
    """No repo real: os 4 docs 'perdidos' do relatorio antigo devem ser RELOCATED.

    E' a assercao que captura o erro original: AGENTE_AUTOCONTIDO.md,
    FASE2_ENXAME.md, FASE3_EVOLUCAO.md e PROPOSTA_PESQUISA.md foram movidos
    para docs/ pelo commit 6af0c10 — nao apagados.
    """
    set_root(ROOT)
    decl = audit.declared_paths()
    index = audit.basename_index()

    for path in (
        "AGENTE_AUTOCONTIDO.md",
        "FASE2_ENXAME.md",
        "FASE3_EVOLUCAO.md",
        "PROPOSTA_PESQUISA.md",
    ):
        assert path in decl, f"{path} nao esta declarado no BACKLOG"
        assert not os.path.exists(os.path.join(ROOT, path))
        assert audit.find_move(path) is not None, (
            f"{path}: o relatorio antigo dizia 'apagado por 6af0c10', "
            "mas o conteudo existe em docs/ — deve ser RELOCATED"
        )
        assert os.path.exists(os.path.join(ROOT, "docs", path))


def test_real_missing_are_genuinely_gone():
    """Os artefatos realmente ausentes nao podem ter destino no repo."""
    set_root(ROOT)
    decl = audit.declared_paths()
    index = audit.basename_index()

    genuinely_missing = [
        p
        for p in decl
        if not os.path.exists(os.path.join(ROOT, p))
        and audit.find_move(p) is None
        and not [h for h in index.get(os.path.basename(p), []) if h != p]
    ]
    assert genuinely_missing, "a auditoria nao encontrou nenhuma ausencia real"

    for path in genuinely_missing:
        # nenhum commit de rename aponta para um destino existente
        assert audit.find_move(path) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
