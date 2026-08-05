"""bundle.py — empacota um CfCCell num único arquivo .py autocontido.

Fase 6.2. O artefato gerado embute os pesos (npz codificado em base64) e
reconstrói a célula a partir de visao.core.cfc.CfCCell (cf. AGENTE_AUTOCONTIDO.md,
seção 3: ".py com pesos em base64 + imports do governance"). Self-verify (6.3),
assinatura (6.4) e teste de adulteração (6.5) são tarefas separadas.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

_TEMPLATE = '''\
# -*- coding: utf-8 -*-
"""Agente VISÃO autocontido (Fase 6.2) — pesos embutidos em base64.

Gerado por visao.core.bundle.build_bundle(). Self-verify (R3), assinatura
(R1) e PoW (R4) são adicionados nas tarefas 6.3 / 6.4.
"""
import base64
import io

import jax.numpy as jnp
import numpy as np
from visao.core.cfc import CfCCell

N_IN = {n_in}
N_HIDDEN = {n_hidden}
SPARSITY = {sparsity}
TAU_MIN = {tau_min}
TAU_MAX = {tau_max}
DT = {dt}
SEED = {seed}

WEIGHTS_B64 = "{weights_b64}"


def _decode_weights():
    raw = base64.b64decode(WEIGHTS_B64)
    state = np.load(io.BytesIO(raw))
    return {{k: jnp.asarray(state[k]) for k in state.files}}


def load():
    """Reconstrói o CfCCell com os pesos embutidos, sem tocar disco."""
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


def build_bundle(cell, out_path) -> str:
    """Gera um único .py autocontido com os pesos do ``cell`` em base64.

    Parameters
    ----------
    cell : CfCCell
        Célula já instanciada (com hiperparâmetros e pesos).
    out_path : str | os.PathLike
        Caminho de destino. A extensão ``.py`` é garantida.

    Returns
    -------
    str
        Caminho do artefato gerado.
    """
    out_path = Path(out_path)
    if out_path.suffix != ".py":
        out_path = out_path.with_suffix(".py")

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
    )
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)
