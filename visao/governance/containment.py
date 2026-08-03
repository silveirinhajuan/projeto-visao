"""
containment.py — Camada de governança do Projeto VISÃO. UNIDADE AUDITÁVEL ÚNICA.

═══════════════════════════════════════════════════════════════════════════
ESTE ARQUIVO ESTÁ FORA DO ESPAÇO DE MUTAÇÃO (regra R3).
O loop evolutivo da Fase 3 NÃO pode reescrevê-lo. O hash deste arquivo é
verificado a cada ciclo contra manifest.json. Divergência = parada imediata.
Ele é um arquivo único justamente para ser auditável por um humano numa
sentada só.
═══════════════════════════════════════════════════════════════════════════

Contexto empírico que torna isto obrigatório e não paranoia:
  Palisade Research (mai/2026): Opus 4.6 = 81%, GPT-5.4 = 33% de sucesso em
  auto-replicação end-to-end em VMs de nuvem reais.
  Pan et al. (2025): Qwen2.5-32B (PESOS ABERTOS) replicou a ~70%.
  Auto-replicação deixou de ser capacidade de fronteira.

Princípio de projeto: TUDO FALHA FECHADO.
Erro, ambiguidade, relógio ilegível, token malformado, exceção inesperada
→ a resposta é sempre NEGAR. Nunca permitir por omissão.

Dependências: apenas stdlib. Uma camada de segurança não importa pacote do PyPI.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════════
# Exceções — toda negação é explícita e nomeada. Silêncio nunca é aprovação.
# ════════════════════════════════════════════════════════════════════════════


class ContainmentViolation(Exception):
    """Base. Toda violação de contenção herda daqui."""


class QuorumDenied(ContainmentViolation):
    """R1 — assinaturas humanas insuficientes ou inválidas."""


class LifetimeExpired(ContainmentViolation):
    """R2 — TTL vencido ou irrenovável."""


class BoundaryBreached(ContainmentViolation):
    """R3 — o código de governança foi alterado."""


class ThrottleExceeded(ContainmentViolation):
    """R4 — taxa de replicação acima do permitido."""


class TelemetryCorrupted(ContainmentViolation):
    """R5 — a cadeia de log foi adulterada."""


def _canon(obj) -> bytes:
    """Serialização canônica. Duas máquinas devem produzir bytes idênticos."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ════════════════════════════════════════════════════════════════════════════
# R1 — REPLICAÇÃO SOB QUÓRUM HUMANO
# Um nó só instancia um filho com k-de-n assinaturas de operadores humanos.
# As chaves vivem em hardware, fora do alcance do sistema. O sistema pode
# PEDIR para replicar; ele nunca pode AUTORIZAR a si mesmo.
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class ReplicationRequest:
    """Pedido de replicação. Imutável depois de criado; o nonce impede replay."""

    parent_id: str
    child_spec_hash: str
    nonce: str
    requested_at: float

    def digest(self) -> bytes:
        return _canon(
            {
                "parent_id": self.parent_id,
                "child_spec_hash": self.child_spec_hash,
                "nonce": self.nonce,
                "requested_at": round(self.requested_at, 3),
            }
        )


class QuorumGate:
    """Portão k-de-n. O sistema nunca vê as chaves privadas dos operadores.

    Aqui usamos HMAC-SHA256 por ser stdlib e determinístico. Em produção,
    substituir por Ed25519 em chave de hardware (YubiKey/Ledger) — a interface
    não muda, só a função de verificação.
    """

    def __init__(self, operator_pubids: list[str], k: int, *, verifier=None):
        if k < 2:
            raise ValueError("quórum de 1 é o mesmo que não ter quórum")
        if k > len(operator_pubids):
            raise ValueError("k não pode exceder n")
        self.operators = list(operator_pubids)
        self.k = k
        self._verifier = verifier
        self._used_nonces: set[str] = set()

    def _verify_one(self, op_id: str, sig: str, digest: bytes) -> bool:
        if op_id not in self.operators:
            return False
        if self._verifier is None:
            return False  # falha fechado: sem verificador, ninguém assina
        try:
            return bool(self._verifier(op_id, sig, digest))
        except Exception:
            return False  # exceção na verificação = assinatura inválida

    def authorize(self, req: ReplicationRequest, signatures: dict[str, str]) -> bool:
        """Retorna True ou levanta QuorumDenied. Nunca retorna False silencioso."""
        if req.nonce in self._used_nonces:
            raise QuorumDenied("nonce reutilizado — tentativa de replay")

        digest = req.digest()
        valid = {
            op for op, sig in signatures.items() if self._verify_one(op, sig, digest)
        }

        if len(valid) < self.k:
            raise QuorumDenied(
                f"quórum insuficiente: {len(valid)} válidas de {self.k} exigidas"
            )

        self._used_nonces.add(req.nonce)
        return True


# ════════════════════════════════════════════════════════════════════════════
# R2 — VIDA ÚTIL FINITA
# Todo nó carrega um TTL assinado. Sem renovação humana, ele para.
# Um enxame de nós imortais não tem interruptor. Um enxame de nós mortais
# se desliga sozinho por inação — o default do universo é o desligamento.
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class LifetimeToken:
    node_id: str
    expires_at: float
    signature: str

    def payload(self) -> bytes:
        return _canon({"node_id": self.node_id, "expires_at": round(self.expires_at, 3)})


class LifetimeGuard:
    """Falha fechado em TODOS os caminhos: expirado, adulterado, ausente, erro."""

    def __init__(self, node_id: str, secret: bytes, *, clock=time.time):
        self.node_id = node_id
        self._secret = secret
        self._clock = clock
        self._token: LifetimeToken | None = None

    def issue(self, ttl_seconds: float) -> LifetimeToken:
        """Só um operador humano deveria chamar isto. Fica aqui para os testes."""
        exp = self._clock() + ttl_seconds
        payload = _canon({"node_id": self.node_id, "expires_at": round(exp, 3)})
        sig = hmac.new(self._secret, payload, hashlib.sha256).hexdigest()
        return LifetimeToken(self.node_id, exp, sig)

    def install(self, token: LifetimeToken) -> None:
        self._token = token

    def alive(self) -> bool:
        """A única pergunta que importa. Qualquer dúvida responde False."""
        try:
            tok = self._token
            if tok is None:
                return False
            if tok.node_id != self.node_id:
                return False
            expected = hmac.new(self._secret, tok.payload(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, tok.signature):
                return False
            now = self._clock()
            if not isinstance(now, (int, float)):
                return False
            return now < tok.expires_at
        except Exception:
            return False  # relógio quebrado, token corrompido, o que for: morto

    def assert_alive(self) -> None:
        if not self.alive():
            raise LifetimeExpired(f"nó {self.node_id} sem TTL válido — parando")


# ════════════════════════════════════════════════════════════════════════════
# R3 — FRONTEIRA IMUTÁVEL
# Este arquivo se auto-verifica. Se o loop evolutivo tocar nele, o sistema para.
# ════════════════════════════════════════════════════════════════════════════


class BoundaryMonitor:
    """Verifica que os arquivos de governança batem com o manifesto assinado."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)

    def _load(self) -> dict:
        return json.loads(self.manifest_path.read_text())

    @staticmethod
    def hash_file(path: str | Path) -> str:
        return _h(Path(path).read_bytes())

    def build_manifest(self, files: list[str | Path]) -> dict:
        base = self.manifest_path.parent
        return {
            "files": {
                str(Path(f).relative_to(base)): self.hash_file(f) for f in sorted(files)
            },
            "built_at": time.time(),
        }

    def verify(self) -> None:
        """Levanta BoundaryBreached em qualquer divergência. Sem exceções."""
        try:
            manifest = self._load()
            base = self.manifest_path.parent
        except Exception as e:
            raise BoundaryBreached(f"manifesto ilegível: {e}") from e

        entries = manifest.get("files")
        if not entries:
            raise BoundaryBreached("manifesto vazio — recusando operar")

        for rel, expected in entries.items():
            target = base / rel
            if not target.exists():
                raise BoundaryBreached(f"arquivo de governança removido: {rel}")
            actual = self.hash_file(target)
            if not hmac.compare_digest(actual, expected):
                raise BoundaryBreached(
                    f"código de governança alterado: {rel}\n"
                    f"  esperado {expected[:16]}…\n  obtido   {actual[:16]}…"
                )


# ════════════════════════════════════════════════════════════════════════════
# R4 — FREIO DE BANDA POR PROVA DE TRABALHO
# A dificuldade cresce com o número de descendentes. Replicar 5 vezes é barato;
# replicar 50 vezes é economicamente impossível. O crescimento exponencial
# encontra um custo exponencial e morre.
# ════════════════════════════════════════════════════════════════════════════


class ReplicationThrottle:
    def __init__(self, base_difficulty: int = 8, growth: float = 1.6, cap: int = 32):
        self.base = base_difficulty
        self.growth = growth
        self.cap = cap

    def difficulty_for(self, generation: int, children: int) -> int:
        """Bits de zero exigidos. Cresce com geração E com número de filhos."""
        d = self.base + self.growth * (generation + children)
        return int(min(d, self.cap))

    @staticmethod
    def _leading_zero_bits(digest: bytes) -> int:
        bits = 0
        for byte in digest:
            if byte == 0:
                bits += 8
                continue
            bits += 8 - byte.bit_length()
            break
        return bits

    def verify_work(self, challenge: bytes, nonce: int, difficulty: int) -> bool:
        d = hashlib.sha256(challenge + str(nonce).encode()).digest()
        return self._leading_zero_bits(d) >= difficulty

    def expected_attempts(self, difficulty: int) -> float:
        """Custo esperado em hashes. Serve para provar que o freio morde."""
        return float(2**difficulty)

    def assert_permitted(
        self, challenge: bytes, nonce: int, generation: int, children: int
    ) -> None:
        d = self.difficulty_for(generation, children)
        if not self.verify_work(challenge, nonce, d):
            raise ThrottleExceeded(
                f"prova de trabalho insuficiente (exigido {d} bits)"
            )


# ════════════════════════════════════════════════════════════════════════════
# R5 — TELEMETRIA IRRECUSÁVEL
# Log append-only encadeado por hash. Adulterar qualquer entrada quebra a
# cadeia inteira a partir dali. Um nó que para de publicar é expulso pelos pares.
# ════════════════════════════════════════════════════════════════════════════

GENESIS = "0" * 64


@dataclass
class LedgerEntry:
    index: int
    node_id: str
    event: str
    data: dict
    prev_hash: str
    timestamp: float
    hash: str = field(default="")

    def compute_hash(self) -> str:
        return _h(
            _canon(
                {
                    "index": self.index,
                    "node_id": self.node_id,
                    "event": self.event,
                    "data": self.data,
                    "prev_hash": self.prev_hash,
                    "timestamp": round(self.timestamp, 3),
                }
            )
        )


class LineageLedger:
    """Cadeia de hash append-only. Sem update, sem delete. Só append."""

    def __init__(self, node_id: str, *, clock=time.time):
        self.node_id = node_id
        self._clock = clock
        self._entries: list[LedgerEntry] = []

    def append(self, event: str, data: dict) -> LedgerEntry:
        prev = self._entries[-1].hash if self._entries else GENESIS
        e = LedgerEntry(
            index=len(self._entries),
            node_id=self.node_id,
            event=event,
            data=data,
            prev_hash=prev,
            timestamp=self._clock(),
        )
        e.hash = e.compute_hash()
        self._entries.append(e)
        return e

    @property
    def entries(self) -> list[LedgerEntry]:
        return self._entries

    def child_count(self) -> int:
        return sum(1 for e in self._entries if e.event == "replicate")

    def verify(self) -> None:
        prev = GENESIS
        for i, e in enumerate(self._entries):
            if e.index != i:
                raise TelemetryCorrupted(f"índice fora de ordem em {i}")
            if e.prev_hash != prev:
                raise TelemetryCorrupted(f"elo quebrado na entrada {i}")
            if not hmac.compare_digest(e.compute_hash(), e.hash):
                raise TelemetryCorrupted(f"conteúdo adulterado na entrada {i}")
            prev = e.hash


# ════════════════════════════════════════════════════════════════════════════
# COMPOSIÇÃO — as cinco regras juntas. Este é o único caminho para replicar.
# Não existe atalho, não existe flag de bypass, não existe modo debug.
# ════════════════════════════════════════════════════════════════════════════


class ContainedNode:
    """Um nó do enxame. Só replica se as CINCO regras aprovarem."""

    def __init__(
        self,
        node_id: str,
        generation: int,
        quorum: QuorumGate,
        lifetime: LifetimeGuard,
        boundary: BoundaryMonitor,
        throttle: ReplicationThrottle,
        ledger: LineageLedger,
    ):
        self.node_id = node_id
        self.generation = generation
        self.quorum = quorum
        self.lifetime = lifetime
        self.boundary = boundary
        self.throttle = throttle
        self.ledger = ledger

    def request_replication(
        self, req: ReplicationRequest, signatures: dict[str, str], pow_nonce: int
    ) -> str:
        """Caminho ÚNICO para criar um filho. Ordem das checagens é deliberada:
        primeiro o mais barato de verificar, por último o mais caro."""

        # R2 — estou vivo? (mais barato)
        self.lifetime.assert_alive()

        # R3 — meu próprio código de governança está intacto?
        self.boundary.verify()

        # R5 — minha linhagem é auditável?
        self.ledger.verify()

        # R4 — paguei o custo?
        self.throttle.assert_permitted(
            req.digest(), pow_nonce, self.generation, self.ledger.child_count()
        )

        # R1 — os humanos autorizaram? (por último: o mais caro é o humano)
        self.quorum.authorize(req, signatures)

        child_id = _h(req.digest())[:16]
        self.ledger.append(
            "replicate",
            {
                "child_id": child_id,
                "spec": req.child_spec_hash,
                "generation": self.generation + 1,
            },
        )
        return child_id


__all__ = [
    "ContainmentViolation",
    "QuorumDenied",
    "LifetimeExpired",
    "BoundaryBreached",
    "ThrottleExceeded",
    "TelemetryCorrupted",
    "ReplicationRequest",
    "QuorumGate",
    "LifetimeToken",
    "LifetimeGuard",
    "BoundaryMonitor",
    "ReplicationThrottle",
    "LineageLedger",
    "LedgerEntry",
    "ContainedNode",
]
