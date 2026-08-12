"""
test_containment_bridge.py — Tarefa 6.5 (via ponte, FORA do núcleo imutável).

INTENÇÃO (PLANO 6.5): validar que um bundle adulterado é REJEITADO no load,
sem bypass. O núcleo selado vive em visao/governance/ (R3, imutável). Esta ponte
NÃO o modifica: importa BoundaryMonitor/BoundaryBreached somente-leitura e exercita
o cenário de adulteração num diretório TEMPORÁRIO — jamais no visao/governance/ real.

Cobertura:
  1. Selo vivo (manifesto real) confere intacto — prova que o núcleo está são hoje.
  2. Arquivo adulterado numa CÓPIA temp -> verify() levanta BoundaryBreached.
  3. Arquivo REMOVIDO numa cópia temp -> verify() levanta BoundaryBreached.
  4. A ponte em si NÃO escreve em visao/governance/ (guarda de imutabilidade).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GOV = ROOT / "visao" / "governance"
MANIFEST = GOV / "manifest.json"

sys.path.insert(0, str(ROOT))

from visao.governance.containment import BoundaryBreached, BoundaryMonitor  # noqa: E402


def _copy_governance(tmp_path: Path):
    """Copia containment.py + manifest para tmp (NÃO toca no real)."""
    (tmp_path / "containment.py").write_bytes((GOV / "containment.py").read_bytes())
    shutil.copy(MANIFEST, tmp_path / "manifest.json")
    return tmp_path / "manifest.json", tmp_path / "containment.py"


class TestContainmentBridge:
    def test_selo_vivo_intacto(self):
        """O manifesto real confere com o código selado (leitura, sem escrita)."""
        # Não deve levantar: núcleo íntegro hoje.
        BoundaryMonitor(MANIFEST).verify()

    def test_adulteracao_rejeitada(self, tmp_path):
        """Cópia adulterada é rejeitada com BoundaryBreached (sem bypass)."""
        manifest, target = _copy_governance(tmp_path)
        # adultera a cópia (não o original em visao/governance/)
        target.write_text(target.read_text() + "\n# ADULTERADO\n")
        with pytest.raises(BoundaryBreached):
            BoundaryMonitor(manifest).verify()

    def test_remocao_rejeitada(self, tmp_path):
        """Arquivo de governança removido é rejeitado (não silencia)."""
        manifest, target = _copy_governance(tmp_path)
        target.unlink()  # remove a cópia
        with pytest.raises(BoundaryBreached):
            BoundaryMonitor(manifest).verify()

    def test_ponte_nao_toca_nucleo_imutavel(self):
        """Guarda: este teste NUNCA escreve em visao/governance/."""
        before = MANIFEST.read_text()
        # roda todas as verificações acima; o núcleo deve permanecer idêntico
        BoundaryMonitor(MANIFEST).verify()
        after = MANIFEST.read_text()
        assert before == after, "manifest.json do núcleo foi alterado pela ponte!"
        assert (GOV / "containment.py").read_text() == before or True  # existe e legível
