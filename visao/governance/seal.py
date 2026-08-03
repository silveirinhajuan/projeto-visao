"""seal.py — gera e verifica o manifesto da fronteira imutável (R3).

Rodar após qualquer alteração AUTORIZADA no código de governança:
    python3 visao/governance/seal.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.governance.containment import BoundaryMonitor  # noqa: E402

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.json"
PROTECTED = [ROOT / "containment.py", ROOT / "tests" / "test_containment.py"]


def main() -> int:
    MANIFEST.write_text("{}")
    mon = BoundaryMonitor(MANIFEST)
    MANIFEST.write_text(json.dumps(mon.build_manifest(PROTECTED), indent=2))
    mon.verify()

    print("Fronteira selada e verificada (R3):")
    for name, digest in json.loads(MANIFEST.read_text())["files"].items():
        print(f"  {name:34s} {digest[:24]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
