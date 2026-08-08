"""
test_seal.py — Self-verify no load (R3), tarefa 6.3.

Testa que seal.verify_boundary() confere o código de governança contra o
manifesto e ABORTA com BoundaryBreached se houver divergência. É o portão
que impede re-selar código adulterado.

Rodar:  cd ~/projeto-visao && python3 -m pytest visao/governance/tests/test_seal.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from visao.governance.containment import BoundaryBreached, BoundaryMonitor  # noqa: E402
import visao.governance.seal as seal  # noqa: E402


def _make_manifest(tmp_path: Path):
    gov = tmp_path / "containment.py"
    gov.write_text("x = 1\n")
    manifest = tmp_path / "manifest.json"
    mon = BoundaryMonitor(manifest)
    manifest.write_text(json.dumps(mon.build_manifest([gov]), indent=2))
    return manifest, gov


def test_verify_boundary_passa_quando_intacto(tmp_path):
    manifest, _ = _make_manifest(tmp_path)
    # Não deve levantar: arquivo bate com o manifesto.
    seal.verify_boundary(manifest_path=manifest)


def test_verify_boundary_aborta_quando_adulterado(tmp_path):
    manifest, gov = _make_manifest(tmp_path)
    gov.write_text("x = 2  # adulterado\n")  # diverge do hash selado
    with pytest.raises(BoundaryBreached):
        seal.verify_boundary(manifest_path=manifest)
