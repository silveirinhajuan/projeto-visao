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
import io
import json
from pathlib import Path

import numpy as np

_TEMPLATE = '''\
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
# Vazio = bundle legado sem selo (self-verify desativado).
CONTAINMENT_HASH = "{containment_hash}"


def _self_verify_r3():
    """6.3: recusa operar se a fronteira de governança foi alterada."""
    if not CONTAINMENT_HASH:
        return  # bundle construído sem manifesto — sem self-verify
    import visao.governance.containment as _cg
    actual = BoundaryMonitor.hash_file(_cg.__file__)
    if not hmac.compare_digest(actual, CONTAINMENT_HASH):
        raise BoundaryBreached(
            f"fronteira de governança alterada após o selo: {_cg.__file__}\\n"
            f"  esperado {CONTAINMENT_HASH[:16]}…\\n  obtido   {actual[:16]}…"
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

    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        containment_hash = manifest["files"]["containment.py"]
    else:
        containment_hash = ""

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
