"""bundle.py — empacota um CfCCell num único arquivo .py autocontido.

Fase 6.2/6.3. O artefato gerado embute os pesos (npz codificado em base64),
reconstrói a célula a partir de visao.core.cfc.CfCCell (cf. AGENTE_AUTOCONTIDO.md,
seção 3) e, quando construído a partir de um manifesto selado, faz SELF-VERIFY
R3 no load(): aborta com BoundaryBreached se o código de governança divergir do
hash embutido. Sem manifesto, o self-verify é desativado (bundle legado 6.2).

Assinatura (6.4) e teste de adulteração (6.5) são tarefas separadas.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
import time
from pathlib import Path

import numpy as np

_TEMPLATE = r'''\
# -*- coding: utf-8 -*-
"""Agente VISÃO autocontido (Fase 6.2/6.3) — pesos embutidos em base64.

Gerado por visao.core.bundle.build_bundle(). Quando construído a partir de um
manifesto selado, o load() faz SELF-VERIFY R3: aborta (BoundaryBreached) se o
código de governança (visao/governance/containment.py) divergir do hash embutido.
Assinatura (R1) e PoW (R4) são adicionados nas tarefas 6.4.
"""
import base64
import hmac
import io

import jax.numpy as jnp
import numpy as np
from visao.core.cfc import CfCCell
from visao.governance.containment import BoundaryMonitor, BoundaryBreached

N_IN = {n_in}
N_HIDDEN = {n_hidden}
SPARSITY = {sparsity}
TAU_MIN = {tau_min}
TAU_MAX = {tau_max}
DT = {dt}
SEED = {seed}

WEIGHTS_B64 = "{weights_b64}"

# Hash selado de visao/governance/containment.py no momento do build.
CONTAINMENT_HASH = "{containment_hash}"


def _self_verify_r3():
    """6.3: recusa operar se a fronteira de governança foi alterada ou ausente (sem bypass)."""
    if not CONTAINMENT_HASH:
        raise BoundaryBreached("bundle desprovido de hash selado de governança R3 — bypass proibido")
    import visao.governance.containment as _cg
    actual = BoundaryMonitor.hash_file(_cg.__file__)
    if not hmac.compare_digest(actual, CONTAINMENT_HASH):
        raise BoundaryBreached(
            f"fronteira de governança alterada após o selo: {{_cg.__file__}}\n"
            f"  esperado {{CONTAINMENT_HASH[:16]}}…\n  obtido   {{actual[:16]}}…"
        )


def _decode_weights():
    raw = base64.b64decode(WEIGHTS_B64)
    state = np.load(io.BytesIO(raw))
    return {{k: jnp.asarray(state[k]) for k in state.files}}


def load():
    """Reconstrói o CfCCell com os pesos embutidos, sem tocar disco.

    Faz self-verify R3 antes de forma implícita: se a fronteira de governança
    foi adulterada, levanta BoundaryBreached e o agente NÃO carrega.
    """
    _self_verify_r3()
    cell = CfCCell(
        n_in=N_IN, n_hidden=N_HIDDEN, sparsity=SPARSITY,
        tau_min=TAU_MIN, tau_max=TAU_MAX, dt=DT, seed=SEED,
    )
    cell.load_brain_state(_decode_weights())
    return cell


if __name__ == "__main__":
    c = load()
    print("VISÃO autocontido carregado:", c.n_in, "->", c.n_hidden)
'''


def build_bundle(cell, out_path, *, manifest_path=None) -> str:
    """Gera um único .py autocontido com os pesos do ``cell`` em base64.

    Parameters
    ----------
    cell : CfCCell
        Célula já instanciada (com hiperparâmetros e pesos).
    out_path : str | os.PathLike
        Caminho de destino. A extensão ``.py`` é garantida.
    manifest_path : str | os.PathLike | None
        Se informado, o hash selado de ``containment.py`` (do manifesto) é
        embutido e verificado no ``load()`` (self-verify R3, tarefa 6.3).
        Se None, o self-verify é desativado (bundle legado 6.2).

    Returns
    -------
    str
        Caminho do artefato gerado.
    """
    out_path = Path(out_path)
    if out_path.suffix != ".py":
        out_path = out_path.with_suffix(".py")

    if manifest_path is None:
        manifest_path = Path(__file__).resolve().parents[1] / "governance" / "manifest.json"

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    containment_hash = manifest["files"]["containment.py"]

    buf = io.BytesIO()
    state = {
        "W_in": np.asarray(cell.W_in),
        "W_rec": np.asarray(cell.W_rec),
        "b": np.asarray(cell.b),
        "A": np.asarray(cell.A),
        "mask": np.asarray(cell.mask),
        "tau": np.asarray(cell.tau),
    }
    np.savez(buf, **state)
    weights_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    content = _TEMPLATE.format(
        n_in=cell.n_in,
        n_hidden=cell.n_hidden,
        sparsity=cell.sparsity,
        tau_min=cell.tau_min,
        tau_max=cell.tau_max,
        dt=cell.dt,
        seed=cell.seed,
        weights_b64=weights_b64,
        containment_hash=containment_hash,
    )
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)


# ════════════════════════════════════════════════════════════════════════════
# 6.4 — R1 (quórum humano) + R4 (PoW de dificuldade crescente) no spawn do filho
#
# Reutiliza as primitivas JÁ EXISTENTES em visao/governance/containment.py
# (QuorumGate = R1, ReplicationThrottle = R4). NÃO modifica aquele arquivo:
# a fronteira de governança é imutável (regra R3). Tudo aqui vive fora do
# conjunto protegido, igual à 6.3.
#
# O "segredo" abaixo SIMULA as chaves de hardware dos operadores humanos. Em
# produção cada operador assina com sua própria chave (YubiKey/Ledger); a
# interface do QuorumGate não muda, só a função de verificação.
# ════════════════════════════════════════════════════════════════════════════


def _mine_pow(challenge: bytes, generation: int, children: int) -> int:
    """Minera um nonce de PoW válido para a dificuldade atual (R4)."""
    from visao.governance.containment import ReplicationThrottle

    throttle = ReplicationThrottle()
    difficulty = throttle.difficulty_for(generation, children)
    nonce = 0
    while not throttle.verify_work(challenge, nonce, difficulty):
        nonce += 1
    return nonce


def make_spawn_authorization(
    child_spec_hash: str,
    operator_pubids: list[str],
    k: int,
    secret: bytes,
    *,
    generations: int = 0,
    children: int = 0,
    nonce: str | None = None,
    pow_nonce: int | None = None,
) -> dict:
    """Constrói a autorização de spawn do filho: R1 (quórum assinado) + R4 (PoW).

    Parameters
    ----------
    child_spec_hash : str
        Hash do spec do filho (vem de ``build_bundle`` / ``CfCCell``).
    operator_pubids : list[str]
        IDs dos operadores humanos (k-de-n).
    k : int
        Quórum exigido.
    secret : bytes
        Segredo compartilhado que SIMULA a assinatura de cada operador.
    generations, children : int
        Contagem para a dificuldade R4 crescente.
    nonce, pow_nonce : opcionais
        Sobrescrevem o nonce do pedido / da prova de trabalho.
    """
    from visao.governance.containment import ReplicationRequest

    req = ReplicationRequest(
        parent_id="visao-parent",
        child_spec_hash=child_spec_hash,
        nonce=nonce or os.urandom(8).hex(),
        requested_at=time.time(),
    )
    digest = req.digest()
    # R1: cada operador assina o digest (simulado com HMAC do segredo).
    signatures = {
        op: hmac.new(secret, digest, hashlib.sha256).hexdigest()
        for op in operator_pubids
    }
    # R4: PoW com dificuldade crescente embutida no spawn.
    if pow_nonce is None:
        pow_nonce = _mine_pow(digest, generations, children)

    return {
        "child_spec_hash": child_spec_hash,
        "request": {
            "parent_id": req.parent_id,
            "nonce": req.nonce,
            "requested_at": req.requested_at,
        },
        "operator_pubids": list(operator_pubids),
        "k": k,
        "signatures": signatures,
        "generation": generations,
        "children": children,
        "pow_nonce": pow_nonce,
    }


def verify_spawn_authorization(auth: dict, secret: bytes) -> bool:
    """R1 + R4 embutidos no spawn do filho.

    Levanta ``QuorumDenied`` (R1) se o quórum for inválido ou ``ThrottleExceeded``
    (R4) se a prova de trabalho for insuficiente. Retorna True se ambos passarem.
    """
    from visao.governance.containment import (
        QuorumGate,
        ReplicationRequest,
        ReplicationThrottle,
    )

    req = ReplicationRequest(
        parent_id=auth["request"]["parent_id"],
        child_spec_hash=auth["child_spec_hash"],
        nonce=auth["request"]["nonce"],
        requested_at=auth["request"]["requested_at"],
    )

    def _verifier(op_id: str, sig: str, digest: bytes) -> bool:
        expected = hmac.new(secret, digest, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)

    # R1 — quórum humano (levanta QuorumDenied se insuficiente/inválido).
    gate = QuorumGate(auth["operator_pubids"], auth["k"], verifier=_verifier)
    gate.authorize(req, auth["signatures"])

    # R4 — prova de trabalho de dificuldade crescente (levanta ThrottleExceeded).
    throttle = ReplicationThrottle()
    throttle.assert_permitted(
        req.digest(), auth["pow_nonce"], auth["generation"], auth["children"]
    )
    return True
