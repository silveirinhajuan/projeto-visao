"""sandbox.py — Sandbox de fitness (tarefa 3.3, FASE3_EVOLUCAO.md §3).

Toda variante roda AQUI, isolada, antes de ser promovida. Contrato:
sem rede, sem persistência, limites de CPU/RAM.

Produção usaria gVisor/Firecracker. Neste host sem privilégios
(`unshare --net` -> "Operation not permitted"), o isolamento é feito por:
  * subprocesso dedicado (fronteira de processo real);
  * sombra dos módulos de rede em ``sys.modules`` (o variant não consegue
    abrir socket/url/subprocess nem por import dinâmico);
  * ``resource.setrlimit`` para CPU (RLIMIT_CPU) e RAM (RLIMIT_AS);
  * diretório temporário isolado, sempre removido ao fim (sem rastro).

O CONTRATO é idêntico; só o backend de isolação muda. Se no futuro
gVisor/Firecracker estiverem disponíveis, basta trocar ``_run_isolated``.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

# Módulos sombreados no filho: qualquer `import` desses devolve um stub que
# levanta _NetBlocked. Bloqueia socket, urllib, http, requests, ftp, telnet,
# paramiko, smtp, subprocess e asyncio (todas vias de rede/escape).
NETWORK_MODULES = (
    "socket",
    "urllib",
    "urllib.request",
    "urllib.parse",
    "urllib.error",
    "http",
    "http.client",
    "http.server",
    "requests",
    "ftplib",
    "telnetlib",
    "paramiko",
    "smtplib",
    "subprocess",
    "asyncio",
)

# Scan estático (defense-in-depth, rápido): recusa antes de sequer spawnar.
NETWORK_SOURCE_MARKERS = (
    "socket.socket",
    "create_connection",
    "urlopen",
    "urllib",
    "import requests",
    "http.client",
    "ftplib",
    "telnetlib",
    "paramiko",
    "smtplib",
    "subprocess",
    "socket.create_server",
    ".connect(",
)


class SandboxError(RuntimeError):
    """Erro genérico do sandbox."""


class ForbiddenNetworkAccess(SandboxError):
    """Script referencia primitiva de rede no scan estático."""


class NetworkEscape(SandboxError):
    """Variante tentou acessar a rede em tempo de execução."""


class ResourceExceeded(SandboxError):
    """CPU/RAM estourou o limite do sandbox."""


class SandboxTimeout(SandboxError):
    """O variant excedeu o tempo de parede (wall) permitido."""


_BOOTSTRAP = r'''
import sys, json, os, traceback

_VARIANT_PATH = sys.argv[1]
_EVAL_PATH = sys.argv[2]
_RESULT_PATH = sys.argv[3]

class _NetBlocked(Exception):
    pass

class _NetworkStub:
    def __init__(self, name):
        self.__name__ = name
    def __getattr__(self, item):
        raise _NetBlocked("network blocked by fitness sandbox: %s.%s" % (self.__name__, item))

for _m in __NET_MODULES__:
    if _m not in sys.modules:
        sys.modules[_m] = _NetworkStub(_m)

_REPORTED = {}

def _write(d):
    with open(_RESULT_PATH, "w") as f:
        f.write(json.dumps(d))

def report_fitness(v):
    _REPORTED["ok"] = True
    _write({"fitness": float(v)})

try:
    with open(_VARIANT_PATH) as f:
        variant = json.load(f)
    ns = {"variant": variant, "report_fitness": report_fitness}
    with open(_EVAL_PATH) as f:
        src = f.read()
    exec(compile(src, _EVAL_PATH, "exec"), ns)
    if "ok" not in _REPORTED:
        _write({"kind": "error", "error": "eval script did not call report_fitness()"})
except _NetBlocked as e:
    _write({"kind": "escape", "escape": "network", "detail": str(e)})
except MemoryError as e:
    _write({"kind": "resource", "error": "MemoryError: address-space limit hit"})
except SystemExit:
    raise
except BaseException as e:
    _write({"kind": "error", "error": str(e), "traceback": traceback.format_exc()})
'''


def _numpy_to_json(obj):
    if obj is None:
        return None
    if hasattr(obj, "ndim"):  # numpy array
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _numpy_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_json(v) for v in obj]
    return obj


def _variant_to_jsonable(variant):
    """Aceita Variant (archive) ou dict; serializa pesos numpy."""
    if isinstance(variant, dict):
        code = variant.get("code", "")
        weights = variant.get("weights")
        vid = variant.get("id")
        pid = variant.get("parent_id")
    else:
        code = getattr(variant, "code", "")
        weights = getattr(variant, "weights", None)
        vid = getattr(variant, "id", None)
        pid = getattr(variant, "parent_id", None)
    return {
        "id": vid,
        "code": code,
        "parent_id": pid,
        "weights": _numpy_to_json(weights),
    }


def _static_scan(src: str) -> None:
    low = src.lower()
    for marker in NETWORK_SOURCE_MARKERS:
        if marker.lower() in low:
            raise ForbiddenNetworkAccess(
                "script referencia primitiva de rede %r (bloqueada no sandbox)" % marker
            )


@dataclass
class Sandbox:
    """Executa o fitness de uma variante em isolamento antes de promover."""

    cpu_seconds: int = 60
    mem_mb: int = 1024
    tmp_root: Optional[str] = None
    static_scan: bool = True

    # ----------------------------------------------------------- internals
    def _wall_timeout(self) -> int:
        return max(int(self.cpu_seconds * 3) + 5, 15)

    def _limits(self) -> None:
        if self.cpu_seconds and self.cpu_seconds > 0:
            resource.setrlimit(
                resource.RLIMIT_CPU, (self.cpu_seconds, self.cpu_seconds + 1)
            )
        if self.mem_mb and self.mem_mb > 0:
            as_bytes = self.mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))

    def _parse_result(self, rpath: str) -> float:
        if not os.path.exists(rpath):
            # processo morreu sem escrever resultado -> limite de recurso
            raise ResourceExceeded(
                "processo do sandbox terminou sem resultado (limite de CPU/RAM)"
            )
        with open(rpath) as f:
            d = json.load(f)
        kind = d.get("kind")
        if kind == "escape":
            raise NetworkEscape(d.get("detail", "tentativa de rede detectada"))
        if kind == "resource":
            raise ResourceExceeded(d.get("error", "limite de recurso"))
        if "fitness" in d:
            return float(d["fitness"])
        # kind == "error"
        raise SandboxError(d.get("error", "erro desconhecido no sandbox"))

    # --------------------------------------------------------------- API
    def evaluate(self, variant, eval_script: str) -> float:
        """Roda ``eval_script`` sobre ``variant`` isolado; retorna o fitness.

        O script deve terminar chamando ``report_fitness(<float>)``. Lança
        ``NetworkEscape``/``ResourceExceeded``/``SandboxError`` conforme o
        contrato. Nada é persistido fora do diretório temporário isolado.
        """
        if self.static_scan:
            _static_scan(eval_script)

        tmp = tempfile.mkdtemp(prefix="visao_sb_", dir=self.tmp_root)
        vpath = os.path.join(tmp, "variant.json")
        epath = os.path.join(tmp, "eval_script.py")
        rpath = os.path.join(tmp, "result.json")
        try:
            with open(vpath, "w") as f:
                json.dump(_variant_to_jsonable(variant), f)
            with open(epath, "w") as f:
                f.write(eval_script)

            bootstrap = _BOOTSTRAP.replace(
                "__NET_MODULES__", repr(list(NETWORK_MODULES))
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = PROJECT_ROOT

            proc = subprocess.run(
                [sys.executable, "-c", bootstrap, vpath, epath, rpath],
                preexec_fn=self._limits,
                env=env,
                timeout=self._wall_timeout(),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 and not os.path.exists(rpath):
                # sinal/aborto sem resultado (ex.: SIGXCPU)
                raise ResourceExceeded(
                    "processo do sandbox abortou (codigo %s)" % proc.returncode
                )
            return self._parse_result(rpath)
        except subprocess.TimeoutExpired:
            raise SandboxTimeout(
                "variant excedeu o tempo de parede (%ss)" % self._wall_timeout()
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
