"""
test_readiness.py — O checklist F5.0 existe para impedir recrutamento prematuro.

Se o script disser "AUTORIZADO" quando um pré-requisito falta, ele é pior que
inexistente: é um cheque em branco. Estes testes provam que cada item do
briefing da tarefa 5.4 é checado de fato e que o veredito final é honesto.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from network.readiness_check import (  # noqa: E402
    Check,
    check_cliente_instalavel,
    check_doc_risco_publicado,
    check_phase1_closed,
    check_phase4_governance,
    check_problema_escolhido,
    main,
    run_all_checks,
)


def _backlog(phase4="done", phase1="done", t52="done", t53="done"):
    return {
        "gates": {
            "phase4_governance": phase4,
            "phase1_substrate": phase1,
        },
        "tasks": [
            {"id": "5.2", "status": t52},
            {"id": "5.3", "status": t53},
        ],
    }


# --- Fase 4 (governança) verde ----------------------------------------------

def test_phase4_falha_se_gate_nao_done():
    c = check_phase4_governance(_backlog(phase4="pending"), governance_green=True)
    assert c.passed is False
    assert "govern" in c.detail.lower()


def test_phase4_falha_se_suite_vermelha():
    c = check_phase4_governance(_backlog(phase4="done"), governance_green=False)
    assert c.passed is False


def test_phase4_passando_quando_tudo_verde():
    c = check_phase4_governance(_backlog(phase4="done"), governance_green=True)
    assert c.passed is True


# --- Fase 1 fechada com benchmarks ------------------------------------------

def test_phase1_falha_se_gate_nao_done():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "README.md"
        p.write_text("redução 97.2%", encoding="utf-8")
        c = check_phase1_closed(_backlog(phase1="blocked"), p)
        assert c.passed is False


def test_phase1_falha_se_readme_sem_numeros():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "README.md"
        p.write_text("sem numeros medidos", encoding="utf-8")
        c = check_phase1_closed(_backlog(phase1="done"), p)
        assert c.passed is False


def test_phase1_falha_se_readme_ausente():
    with tempfile.TemporaryDirectory() as d:
        c = check_phase1_closed(_backlog(phase1="done"), Path(d) / "nao_existe.md")
        assert c.passed is False


def test_phase1_passando_com_readme_com_numeros():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "README.md"
        p.write_text("redução de esquecimento 97.2%", encoding="utf-8")
        c = check_phase1_closed(_backlog(phase1="done"), p)
        assert c.passed is True


# --- Cliente instalável em <5min --------------------------------------------

def test_cliente_falha_sem_benchmark():
    c = check_cliente_instalavel(install_bench_path=Path("/caminho/inexistente.json"))
    assert c.passed is False


def test_cliente_passando_com_benchmark_sub_5min():
    with tempfile.TemporaryDirectory() as d:
        bench = Path(d) / "install_bench.json"
        bench.write_text(json.dumps({"install_seconds": 180}), encoding="utf-8")
        c = check_cliente_instalavel(install_bench_path=bench)
        assert c.passed is True


def test_cliente_falha_se_benchmark_acima_5min():
    with tempfile.TemporaryDirectory() as d:
        bench = Path(d) / "install_bench.json"
        bench.write_text(json.dumps({"install_seconds": 600}), encoding="utf-8")
        c = check_cliente_instalavel(install_bench_path=bench)
        assert c.passed is False


# --- Problema escolhido -----------------------------------------------------

def test_problema_falha_se_5_2_blocked():
    c = check_problema_escolhido(_backlog(t52="blocked"))
    assert c.passed is False


def test_problema_falha_se_5_2_ausente():
    c = check_problema_escolhido({"tasks": []})
    assert c.passed is False


def test_problema_passando_se_5_2_done():
    c = check_problema_escolhido(_backlog(t52="done"))
    assert c.passed is True


# --- Doc de risco publicado -------------------------------------------------

def test_risco_falha_se_5_3_pendente():
    with tempfile.TemporaryDirectory() as d:
        doc = Path(d) / "RISCOS_PARA_VOLUNTARIOS.md"
        doc.write_text("risco", encoding="utf-8")
        c = check_doc_risco_publicado(_backlog(t53="pending"), doc)
        assert c.passed is False


def test_risco_falha_se_arquivo_ausente():
    with tempfile.TemporaryDirectory() as d:
        c = check_doc_risco_publicado(_backlog(t53="done"), Path(d) / "ausente.md")
        assert c.passed is False


def test_risco_passando_quando_done_e_arquivo_existe():
    with tempfile.TemporaryDirectory() as d:
        doc = Path(d) / "RISCOS_PARA_VOLUNTARIOS.md"
        doc.write_text("risco real", encoding="utf-8")
        c = check_doc_risco_publicado(_backlog(t53="done"), doc)
        assert c.passed is True


# --- Orquestração -----------------------------------------------------------

def test_run_all_checks_retorna_5_itens():
    with tempfile.TemporaryDirectory() as d:
        readme = Path(d) / "README.md"
        readme.write_text("97.2%", encoding="utf-8")
        doc = Path(d) / "RISCOS_PARA_VOLUNTARIOS.md"
        doc.write_text("risco", encoding="utf-8")
        bench = Path(d) / "install_bench.json"
        bench.write_text(json.dumps({"install_seconds": 120}), encoding="utf-8")
        res = run_all_checks(
            _backlog(),
            governance_green=True,
            readme_path=readme,
            riscos_path=doc,
            install_bench_path=bench,
        )
        assert len(res) == 5
        assert all(isinstance(r, Check) for r in res)


def test_main_imprime_proibido_quando_falta_problema(capsys):
    with tempfile.TemporaryDirectory() as d:
        readme = Path(d) / "README.md"
        readme.write_text("97.2%", encoding="utf-8")
        doc = Path(d) / "RISCOS_PARA_VOLUNTARIOS.md"
        doc.write_text("risco", encoding="utf-8")
        code = main(
            backlog=_backlog(t52="blocked"),
            governance_green=lambda: True,
            readme_path=readme,
            riscos_path=doc,
            install_bench_path=Path(d) / "install_bench.json",
        )
        out = capsys.readouterr().out
        assert code == 1
        assert "PROIBIDO RECRUTAR" in out
        assert "Problema escolhido" in out


def test_main_imprime_autorizado_quando_tudo_verde(capsys):
    with tempfile.TemporaryDirectory() as d:
        readme = Path(d) / "README.md"
        readme.write_text("97.2%", encoding="utf-8")
        doc = Path(d) / "RISCOS_PARA_VOLUNTARIOS.md"
        doc.write_text("risco", encoding="utf-8")
        bench = Path(d) / "install_bench.json"
        bench.write_text(json.dumps({"install_seconds": 120}), encoding="utf-8")
        code = main(
            backlog=_backlog(),
            governance_green=lambda: True,
            readme_path=readme,
            riscos_path=doc,
            install_bench_path=bench,
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "AUTORIZADO RECRUTAR" in out
