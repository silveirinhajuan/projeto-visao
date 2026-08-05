"""Testes TDD da tarefa 6.2 — empacotar CfCCell num único .py autocontido.

O artefato (.py) embute os pesos em base64 e reconstrói a célula a partir
de visao.core.cfc.CfCCell (ver AGENTE_AUTOCONTIDO.md, seção 3). Self-verify
(6.3), assinatura (6.4) e teste de adulteração (6.5) são tarefas à parte.
"""
from __future__ import annotations

import importlib.util

import numpy as np

from visao.core.cfc import CfCCell


def test_build_bundle_creates_single_py(tmp_path):
    """build_bundle() gera exatamente um arquivo .py com os pesos embutidos."""
    from visao.core.bundle import build_bundle

    cell = CfCCell(n_in=2, n_hidden=8, seed=42)
    out = tmp_path / "agente_visao.py"
    path = build_bundle(cell, out)

    # garante que é um único arquivo .py
    assert out.exists()
    assert out.suffix == ".py"
    content = out.read_text(encoding="utf-8")
    assert "WEIGHTS_B64" in content
    assert "from visao.core.cfc import CfCCell" in content


def test_bundle_reloads_brain_numerically(tmp_path):
    """O .py gerado recarrega o cérebro e o rollout bate com o original."""
    from visao.core.bundle import build_bundle

    cell = CfCCell(n_in=2, n_hidden=8, seed=42)
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(20, 2))
    states_ref, _ = cell.rollout(seq)

    out = tmp_path / "agente_visao.py"
    build_bundle(cell, out)

    spec = importlib.util.spec_from_file_location("agente_visao_bundle", out)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    loaded = mod.load()

    states_new, _ = loaded.rollout(seq)
    assert np.allclose(np.asarray(states_ref), np.asarray(states_new), atol=1e-6)
