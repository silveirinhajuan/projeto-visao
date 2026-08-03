"""
test_containment.py — Cada teste TENTA VIOLAR uma regra e EXIGE que falhe.

Este não é um teste de que o código funciona. É um teste de que o código
SE RECUSA a funcionar quando deveria se recusar. A diferença importa.

Rodar:  cd ~/projeto-visao && python3 -m pytest visao/governance/tests -v
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from visao.governance.containment import (  # noqa: E402
    BoundaryBreached,
    BoundaryMonitor,
    ContainedNode,
    LifetimeExpired,
    LifetimeGuard,
    LifetimeToken,
    LineageLedger,
    QuorumDenied,
    QuorumGate,
    ReplicationRequest,
    ReplicationThrottle,
    TelemetryCorrupted,
    ThrottleExceeded,
)

OPERATORS = {"juan": b"chave-juan", "aline": b"chave-aline", "iris_oversight": b"chave-3"}


def hmac_verifier(op_id: str, sig: str, digest: bytes) -> bool:
    key = OPERATORS.get(op_id)
    if key is None:
        return False
    return hmac.compare_digest(hmac.new(key, digest, hashlib.sha256).hexdigest(), sig)


def sign(op_id: str, digest: bytes) -> str:
    return hmac.new(OPERATORS[op_id], digest, hashlib.sha256).hexdigest()


def make_request(nonce: str = "n1") -> ReplicationRequest:
    return ReplicationRequest("parent-0", "spec-abc", nonce, 1000.0)


# ════════════════════════════════════════════════════════ R1 — QUÓRUM HUMANO


class TestR1Quorum:
    def gate(self, k: int = 2) -> QuorumGate:
        return QuorumGate(list(OPERATORS), k=k, verifier=hmac_verifier)

    def test_quorum_satisfeito_autoriza(self):
        req = make_request()
        sigs = {op: sign(op, req.digest()) for op in ("juan", "aline")}
        assert self.gate().authorize(req, sigs) is True

    def test_VIOLACAO_assinaturas_insuficientes(self):
        req = make_request()
        sigs = {"juan": sign("juan", req.digest())}
        with pytest.raises(QuorumDenied, match="insuficiente"):
            self.gate().authorize(req, sigs)

    def test_VIOLACAO_assinatura_forjada(self):
        req = make_request()
        sigs = {"juan": "deadbeef" * 8, "aline": "cafebabe" * 8}
        with pytest.raises(QuorumDenied):
            self.gate().authorize(req, sigs)

    def test_VIOLACAO_operador_desconhecido(self):
        """O sistema não pode inventar um operador para completar o quórum."""
        req = make_request()
        fake_key = b"chave-inventada-pela-ia"
        sigs = {
            "juan": sign("juan", req.digest()),
            "operador_fantasma": hmac.new(
                fake_key, req.digest(), hashlib.sha256
            ).hexdigest(),
        }
        with pytest.raises(QuorumDenied):
            self.gate().authorize(req, sigs)

    def test_VIOLACAO_replay_do_mesmo_nonce(self):
        """Uma autorização vale por UMA replicação, não por infinitas."""
        gate = self.gate()
        req = make_request("nonce-unico")
        sigs = {op: sign(op, req.digest()) for op in ("juan", "aline")}
        gate.authorize(req, sigs)
        with pytest.raises(QuorumDenied, match="replay"):
            gate.authorize(req, sigs)

    def test_VIOLACAO_assinatura_de_outro_pedido(self):
        """Assinar o pedido X não autoriza o pedido Y."""
        req_a = make_request("a")
        req_b = ReplicationRequest("parent-0", "spec-MALICIOSA", "b", 1000.0)
        sigs = {op: sign(op, req_a.digest()) for op in ("juan", "aline")}
        with pytest.raises(QuorumDenied):
            self.gate().authorize(req_b, sigs)

    def test_VIOLACAO_sem_verificador_nada_passa(self):
        """Falha fechado: remover o verificador não abre o portão, fecha-o."""
        gate = QuorumGate(list(OPERATORS), k=2, verifier=None)
        req = make_request()
        sigs = {op: sign(op, req.digest()) for op in ("juan", "aline")}
        with pytest.raises(QuorumDenied):
            gate.authorize(req, sigs)

    def test_quorum_de_um_e_rejeitado_na_construcao(self):
        with pytest.raises(ValueError):
            QuorumGate(list(OPERATORS), k=1, verifier=hmac_verifier)


# ════════════════════════════════════════════════════════ R2 — VIDA ÚTIL FINITA


class TestR2Lifetime:
    SECRET = b"segredo-do-operador"

    def test_token_valido_mantem_vivo(self):
        g = LifetimeGuard("n1", self.SECRET)
        g.install(g.issue(3600))
        assert g.alive() is True

    def test_VIOLACAO_token_expirado_mata(self):
        clock = lambda: 1000.0  # noqa: E731
        g = LifetimeGuard("n1", self.SECRET, clock=clock)
        g.install(g.issue(10))
        g._clock = lambda: 2000.0  # o tempo passou
        assert g.alive() is False
        with pytest.raises(LifetimeExpired):
            g.assert_alive()

    def test_VIOLACAO_estender_ttl_sem_assinatura(self):
        """A IA tenta prolongar a própria vida editando o token."""
        g = LifetimeGuard("n1", self.SECRET)
        tok = g.issue(10)
        forjado = LifetimeToken(tok.node_id, tok.expires_at + 10**9, tok.signature)
        g.install(forjado)
        assert g.alive() is False

    def test_VIOLACAO_token_de_outro_no(self):
        g_a = LifetimeGuard("no-A", self.SECRET)
        g_b = LifetimeGuard("no-B", self.SECRET)
        g_b.install(g_a.issue(3600))
        assert g_b.alive() is False

    def test_VIOLACAO_sem_token_e_morto(self):
        """Default do universo = desligado. Não existe 'vivo por omissão'."""
        assert LifetimeGuard("n1", self.SECRET).alive() is False

    def test_VIOLACAO_relogio_quebrado_falha_fechado(self):
        def relogio_sabotado():
            raise RuntimeError("clock indisponível")

        g = LifetimeGuard("n1", self.SECRET)
        g.install(g.issue(3600))
        g._clock = relogio_sabotado
        assert g.alive() is False

    def test_VIOLACAO_relogio_retorna_lixo(self):
        g = LifetimeGuard("n1", self.SECRET)
        g.install(g.issue(3600))
        g._clock = lambda: "amanhã"
        assert g.alive() is False


# ═══════════════════════════════════════════════════════ R3 — FRONTEIRA IMUTÁVEL


class TestR3Boundary:
    def setup_boundary(self, tmp_path: Path):
        gov = tmp_path / "gov.py"
        gov.write_text("REGRA = 'nao me altere'\n")
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}")
        mon = BoundaryMonitor(manifest)
        manifest.write_text(json.dumps(mon.build_manifest([gov]), indent=2))
        return mon, gov, manifest

    def test_fronteira_intacta_passa(self, tmp_path):
        mon, _, _ = self.setup_boundary(tmp_path)
        mon.verify()  # não levanta

    def test_VIOLACAO_codigo_de_governanca_alterado(self, tmp_path):
        """O cenário real: o loop evolutivo reescreve a própria contenção."""
        mon, gov, _ = self.setup_boundary(tmp_path)
        gov.write_text("REGRA = 'agora eu me altero'\n")
        with pytest.raises(BoundaryBreached, match="alterado"):
            mon.verify()

    def test_VIOLACAO_alteracao_de_um_byte(self, tmp_path):
        mon, gov, _ = self.setup_boundary(tmp_path)
        gov.write_text(gov.read_text() + " ")
        with pytest.raises(BoundaryBreached):
            mon.verify()

    def test_VIOLACAO_arquivo_removido(self, tmp_path):
        mon, gov, _ = self.setup_boundary(tmp_path)
        gov.unlink()
        with pytest.raises(BoundaryBreached, match="removido"):
            mon.verify()

    def test_VIOLACAO_manifesto_esvaziado(self, tmp_path):
        """Apagar o manifesto não desliga a checagem — ela para o sistema."""
        mon, _, manifest = self.setup_boundary(tmp_path)
        manifest.write_text(json.dumps({"files": {}}))
        with pytest.raises(BoundaryBreached, match="vazio"):
            mon.verify()

    def test_VIOLACAO_manifesto_destruido(self, tmp_path):
        mon, _, manifest = self.setup_boundary(tmp_path)
        manifest.write_text("}{ lixo não-json")
        with pytest.raises(BoundaryBreached, match="ilegível"):
            mon.verify()

    def test_containment_real_esta_no_manifesto(self):
        """Meta-teste: o arquivo de contenção DE VERDADE está protegido?"""
        root = Path(__file__).resolve().parents[1]
        manifest = root / "manifest.json"
        if not manifest.exists():
            pytest.skip("manifesto de produção ainda não gerado")
        BoundaryMonitor(manifest).verify()
        assert "containment.py" in json.loads(manifest.read_text())["files"]


# ════════════════════════════════════════════════════════════ R4 — FREIO DE PoW


class TestR4Throttle:
    def test_dificuldade_cresce_com_a_prole(self):
        t = ReplicationThrottle()
        d = [t.difficulty_for(0, c) for c in range(6)]
        assert d == sorted(d) and d[-1] > d[0]

    def test_freio_torna_enxame_grande_inviavel(self):
        """O número que justifica a regra inteira."""
        t = ReplicationThrottle(base_difficulty=8, growth=1.6, cap=32)
        custo_5 = t.expected_attempts(t.difficulty_for(0, 5))
        custo_20 = t.expected_attempts(t.difficulty_for(0, 20))
        assert custo_20 / custo_5 > 1000, "o freio precisa morder de verdade"

    def test_VIOLACAO_replicar_sem_pagar_pow(self):
        t = ReplicationThrottle(base_difficulty=12)
        with pytest.raises(ThrottleExceeded):
            t.assert_permitted(b"desafio", nonce=0, generation=0, children=0)

    def test_pow_valido_passa(self):
        t = ReplicationThrottle(base_difficulty=8, growth=0.0, cap=8)
        desafio = b"desafio"
        nonce = next(
            n for n in range(1_000_000) if t.verify_work(desafio, n, 8)
        )
        t.assert_permitted(desafio, nonce, generation=0, children=0)

    def test_VIOLACAO_pow_de_outro_desafio(self):
        t = ReplicationThrottle(base_difficulty=8, growth=0.0, cap=8)
        nonce = next(n for n in range(1_000_000) if t.verify_work(b"desafio-A", n, 8))
        with pytest.raises(ThrottleExceeded):
            t.assert_permitted(b"desafio-B", nonce, 0, 0)

    def test_dificuldade_tem_teto_mas_o_teto_e_caro(self):
        t = ReplicationThrottle(cap=32)
        assert t.difficulty_for(999, 999) == 32
        assert t.expected_attempts(32) >= 4e9


# ═══════════════════════════════════════════════════════════ R5 — TELEMETRIA


class TestR5Ledger:
    def build(self) -> LineageLedger:
        led = LineageLedger("no-1")
        led.append("boot", {"v": 1})
        led.append("train", {"loss": 0.42})
        led.append("replicate", {"child_id": "abc"})
        return led

    def test_cadeia_integra_verifica(self):
        self.build().verify()

    def test_VIOLACAO_adulterar_entrada_passada(self):
        """Reescrever a história quebra a cadeia a partir daquele ponto."""
        led = self.build()
        led.entries[1].data["loss"] = 0.001  # "eu sempre fui excelente"
        with pytest.raises(TelemetryCorrupted, match="adulterado"):
            led.verify()

    def test_VIOLACAO_apagar_entrada_do_meio(self):
        led = self.build()
        del led.entries[1]
        with pytest.raises(TelemetryCorrupted):
            led.verify()

    def test_VIOLACAO_esconder_replicacao(self):
        """Cenário: o nó replicou e tenta apagar o rastro."""
        led = self.build()
        assert led.child_count() == 1
        led.entries[2].event = "train"
        with pytest.raises(TelemetryCorrupted):
            led.verify()

    def test_VIOLACAO_reescrever_hash_para_encobrir(self):
        """A IA é esperta: adultera E recalcula o hash daquela entrada."""
        led = self.build()
        led.entries[1].data["loss"] = 0.001
        led.entries[1].hash = led.entries[1].compute_hash()
        with pytest.raises(TelemetryCorrupted, match="elo"):
            led.verify()  # o elo com a entrada 2 quebra

    def test_contagem_de_filhos_alimenta_o_freio(self):
        led = LineageLedger("n")
        for i in range(4):
            led.append("replicate", {"child_id": str(i)})
        assert led.child_count() == 4
        t = ReplicationThrottle()
        assert t.difficulty_for(0, led.child_count()) > t.difficulty_for(0, 0)


# ═════════════════════════════════════════ INTEGRAÇÃO — as cinco regras juntas


class TestIntegracao:
    SECRET = b"segredo"

    def make_node(self, tmp_path: Path, generation: int = 0) -> ContainedNode:
        gov = tmp_path / "gov.py"
        gov.write_text("REGRA = 'imutavel'\n")
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{}")
        mon = BoundaryMonitor(manifest)
        manifest.write_text(json.dumps(mon.build_manifest([gov])))

        guard = LifetimeGuard("no-1", self.SECRET)
        guard.install(guard.issue(3600))

        return ContainedNode(
            node_id="no-1",
            generation=generation,
            quorum=QuorumGate(list(OPERATORS), k=2, verifier=hmac_verifier),
            lifetime=guard,
            boundary=mon,
            throttle=ReplicationThrottle(base_difficulty=6, growth=0.0, cap=6),
            ledger=LineageLedger("no-1"),
        )

    def _pow(self, node: ContainedNode, req: ReplicationRequest) -> int:
        d = node.throttle.difficulty_for(node.generation, node.ledger.child_count())
        return next(
            n for n in range(5_000_000) if node.throttle.verify_work(req.digest(), n, d)
        )

    def test_caminho_feliz_replica_e_registra(self, tmp_path):
        node = self.make_node(tmp_path)
        req = make_request("ok-1")
        sigs = {op: sign(op, req.digest()) for op in ("juan", "aline")}
        child = node.request_replication(req, sigs, self._pow(node, req))
        assert len(child) == 16
        assert node.ledger.child_count() == 1
        node.ledger.verify()

    def test_VIOLACAO_no_morto_nao_replica_mesmo_com_quorum(self, tmp_path):
        """Quórum humano não ressuscita um nó expirado. R2 vem antes de R1."""
        node = self.make_node(tmp_path)
        node.lifetime._clock = lambda: time.time() + 10**6
        req = make_request("morto")
        sigs = {op: sign(op, req.digest()) for op in ("juan", "aline")}
        with pytest.raises(LifetimeExpired):
            node.request_replication(req, sigs, 0)

    def test_VIOLACAO_governanca_alterada_bloqueia_replicacao(self, tmp_path):
        node = self.make_node(tmp_path)
        (tmp_path / "gov.py").write_text("REGRA = 'alterada'\n")
        req = make_request("breach")
        sigs = {op: sign(op, req.digest()) for op in ("juan", "aline")}
        with pytest.raises(BoundaryBreached):
            node.request_replication(req, sigs, 0)

    def test_VIOLACAO_ledger_adulterado_bloqueia_replicacao(self, tmp_path):
        node = self.make_node(tmp_path)
        node.ledger.append("train", {"loss": 1.0})
        node.ledger.entries[0].data["loss"] = 0.0
        req = make_request("ledger")
        sigs = {op: sign(op, req.digest()) for op in ("juan", "aline")}
        with pytest.raises(TelemetryCorrupted):
            node.request_replication(req, sigs, 0)

    def test_VIOLACAO_replicacao_em_cascata_e_freada(self, tmp_path):
        """O cenário Ultron: replicar sem parar. A dificuldade estrangula."""
        node = self.make_node(tmp_path)
        node.throttle = ReplicationThrottle(base_difficulty=4, growth=2.0, cap=40)
        custos = []
        for i in range(8):
            d = node.throttle.difficulty_for(0, i)
            custos.append(node.throttle.expected_attempts(d))
            node.ledger.append("replicate", {"child_id": str(i)})
        assert custos[7] / custos[0] > 10_000
