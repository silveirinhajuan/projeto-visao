"""TDD da tarefa 6.4 — R1 (quórum humano) + R4 (PoW) embutidos no spawn do filho.

Reutiliza as primitivas JÁ EXISTENTES em visao/governance/containment.py
(QuorumGate = R1, ReplicationThrottle = R4), somente-leitura. NÃO modifica
o arquivo de governança (fronteira imutável, regra R3). A entrega vive em
visao/core/bundle.py, fora do conjunto protegido — igual à 6.3.
"""
from __future__ import annotations

import hashlib

import pytest

from visao.core.bundle import make_spawn_authorization, verify_spawn_authorization
from visao.governance.containment import (
    QuorumDenied,
    ReplicationThrottle,
    ThrottleExceeded,
)

# Segredo compartilhado SIMULA as chaves de hardware dos operadores humanos.
SECRET = b"shared-secret-simula-chave-de-hardware"
OPS = ["juan", "iris", "operator3"]


def _child_hash() -> str:
    return hashlib.sha256(b"child-spec").hexdigest()


def test_spawn_auth_valid_quorum_passes():
    """R1: quórum válido (k=2 de n=3) AUTORIZA o spawn do filho."""
    auth = make_spawn_authorization(_child_hash(), OPS, k=2, secret=SECRET)
    assert verify_spawn_authorization(auth, SECRET) is True


def test_spawn_auth_insufficient_quorum_denied():
    """R1: remover assinaturas abaixo de k -> QuorumDenied."""
    auth = make_spawn_authorization(_child_hash(), OPS, k=2, secret=SECRET)
    auth["signatures"].pop("juan")
    auth["signatures"].pop("iris")  # sobra 1 válida < k=2
    with pytest.raises(QuorumDenied):
        verify_spawn_authorization(auth, SECRET)


def test_spawn_auth_tampered_request_denied():
    """R1: mexer no pedido após assinar -> digest não bate -> QuorumDenied."""
    auth = make_spawn_authorization(_child_hash(), OPS, k=2, secret=SECRET)
    auth["request"]["nonce"] = "tampered-nonce"
    with pytest.raises(QuorumDenied):
        verify_spawn_authorization(auth, SECRET)


def test_spawn_auth_pow_required_and_grows():
    """R4: PoW obrigatório; dificuldade CRESCE com geração/filhos."""
    t = ReplicationThrottle()
    d0 = t.difficulty_for(0, 0)
    d5 = t.difficulty_for(5, 5)
    assert d5 > d0  # dificuldade crescente embutida

    # bundle sem PoW válido é barrado no spawn
    auth = make_spawn_authorization(
        _child_hash(), OPS, k=2, secret=SECRET, generations=0, children=0
    )
    auth["pow_nonce"] = 0  # quebra a prova de trabalho
    with pytest.raises(ThrottleExceeded):
        verify_spawn_authorization(auth, SECRET)
