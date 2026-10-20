"""
test_live_loop.py — Testes para live_loop e monitor.
"""

from __future__ import annotations

import sys
import tempfile
import time
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain
from visao.ops.live_loop import (
    LiveLoop,
    CircuitBreaker,
    CheckpointManager,
    BrainHealthMonitor,
    HealthStatus,
    simulate_24h,
    validate_30day_gate,
    rotate_log_if_needed,
    count_jsonl_lines,
    ROTATE_MAX_LINES,
    ROTATE_KEEP_LINES,
    HEALTH_LOG,
)
from visao.ops.monitor import BrainMonitor


class TestLiveLoop:
    def test_create_default(self):
        loop = LiveLoop()
        assert loop.brain is not None
        assert loop.checkpoint_interval == 5000

    def test_create_custom_brain(self):
        brain = VisaoBrain(n_in=2, n_hidden=16, n_out=1, seed=42)
        loop = LiveLoop(brain=brain)
        assert loop.brain is brain

    def test_state_initialization(self):
        loop = LiveLoop()
        assert loop.state.name == "INITIALIZING"
        assert loop.is_running is False

    def test_stop(self):
        loop = LiveLoop()
        loop._running = True
        loop.stop()
        assert loop.is_running is False
        assert loop.state.name == "STOPPING"

    def test_short_run(self):
        """Testa execução curta do loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            brain = VisaoBrain(n_in=2, n_hidden=16, n_out=1, seed=42)
            loop = LiveLoop(
                brain=brain,
                checkpoint_dir=tmpdir,
                log_dir=tmpdir,
                checkpoint_interval=100,
                health_check_interval=50,
                log_interval=50,
            )
            loop._current_sleep = 0.0001
            metrics = loop.start(duration_seconds=1)
            assert metrics.total_steps > 0
            assert loop.state.name == "STOPPED"

    def test_checkpoint_creation(self):
        """Testa criação de checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            brain = VisaoBrain(n_in=2, n_hidden=16, n_out=1, seed=42)
            loop = LiveLoop(
                brain=brain,
                checkpoint_dir=tmpdir,
                log_dir=tmpdir,
                checkpoint_interval=50,
                health_check_interval=1000,
                log_interval=1000,
            )
            loop._current_sleep = 0.0001
            loop.start(duration_seconds=2)
            # Deve ter criado pelo menos 1 checkpoint
            checkpoints = list(Path(tmpdir).glob("brain_step_*.json"))
            assert len(checkpoints) >= 1

    def test_get_summary(self):
        loop = LiveLoop()
        summary = loop.get_summary()
        assert "state" in summary
        assert "total_steps" in summary


class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker(threshold=3)
        assert cb.can_execute() is True

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is True  # ainda não atingiu threshold
        cb.record_failure()
        assert cb.can_execute() is False  # abriu

    def test_reset(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is False
        cb.reset()
        assert cb.can_execute() is True

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(threshold=2, reset_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute() is False
        time.sleep(0.15)
        assert cb.can_execute() is True  # half-open


class TestCheckpointManager:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            brain = VisaoBrain(n_in=2, n_hidden=16, n_out=1, seed=42)
            mgr = CheckpointManager(tmpdir, max_checkpoints=5)
            path = mgr.save(brain, step=100)
            assert path.exists()

            result = mgr.load_latest()
            assert result is not None
            loaded_path, step = result
            assert step == 100

    def test_prune_old_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            brain = VisaoBrain(n_in=2, n_hidden=16, n_out=1, seed=42)
            mgr = CheckpointManager(tmpdir, max_checkpoints=3)
            for i in range(5):
                mgr.save(brain, step=i * 100)

            checkpoints = list(Path(tmpdir).glob("brain_step_*.json"))
            assert len(checkpoints) == 3

    def test_load_specific(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            brain = VisaoBrain(n_in=2, n_hidden=16, n_out=1, seed=42)
            mgr = CheckpointManager(tmpdir)
            mgr.save(brain, step=42)
            step = mgr.load_specific(Path(tmpdir) / "brain_step_42.json")
            assert step == 42


class TestBrainHealthMonitor:
    def test_initial_status(self):
        monitor = BrainHealthMonitor()
        assert monitor.get_health_status() == HealthStatus.HEALTHY

    def test_healthy_after_good_results(self):
        monitor = BrainHealthMonitor()
        for _ in range(50):
            monitor.update({"err": 0.01, "surprise": 1.0}, lr=0.02)
        assert monitor.get_health_status() == HealthStatus.HEALTHY

    def test_critical_after_high_error(self):
        monitor = BrainHealthMonitor()
        # Erro alto e variável → CRITICAL (não DEAD, pois DEAD exige std≈0)
        for i in range(50):
            monitor.update({"err": 5.0 + (i % 3), "surprise": 3.0}, lr=0.02)
        assert monitor.get_health_status() == HealthStatus.CRITICAL

    def test_trends(self):
        monitor = BrainHealthMonitor()
        for i in range(50):
            monitor.update({"err": 0.01 * i, "surprise": 1.0 + 0.1 * i}, lr=0.02)
        trends = monitor.get_trends()
        assert "error_trend" in trends
        assert "surprise_trend" in trends

    def test_stats(self):
        monitor = BrainHealthMonitor()
        for i in range(10):
            monitor.update({"err": 0.1 * i, "surprise": 1.0 + 0.1 * i}, lr=0.02)
        stats = monitor.get_stats()
        assert stats["status"] == "running"
        assert stats["mse"] >= 0


class TestMonitor:
    def test_create(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = BrainMonitor(window=50, log_dir=tmpdir)
            assert m.window == 50

    def test_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = BrainMonitor(window=50, log_dir=tmpdir)
            m.update({"err": 0.5, "surprise": 1.2}, lr=0.02)
            assert len(m.errors) == 1

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = BrainMonitor(window=50, log_dir=tmpdir)
            for i in range(10):
                m.update({"err": 0.1 * i, "surprise": 1.0 + 0.1 * i}, lr=0.02)

            stats = m.get_stats()
            assert stats["status"] == "running"
            assert stats["mse"] >= 0
            assert stats["mae"] >= 0

    def test_save_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = BrainMonitor(window=50, log_dir=tmpdir)
            m.update({"err": 0.5, "surprise": 1.2}, lr=0.02)
            m.save_log(step=1)

            log_file = Path(tmpdir) / "monitor.jsonl"
            assert log_file.exists()


class TestSimulation:
    def test_simulate_24h_runs(self):
        """Testa que a simulação roda sem erros."""
        summary = simulate_24h(inject_errors=False, verbose=False)
        assert summary["total_steps"] > 0
        assert summary["state"] == "STOPPED"

    def test_simulate_with_errors(self):
        """Testa simulação com injeção de erros."""
        summary = simulate_24h(inject_errors=True, verbose=False)
        assert summary["total_steps"] > 0

    def test_validate_gate(self):
        """Testa validação do portão."""
        summary = simulate_24h(inject_errors=True, verbose=False)
        result = validate_30day_gate(summary)
        assert "gate_passed" in result
        assert "criteria" in result


def _write_jsonl(path: Path, n: int) -> None:
    """Helper: escreve n linhas JSON válidas em path."""
    with open(path, "a") as f:
        for i in range(n):
            f.write(json.dumps({"i": i}) + "\n")


class TestLogRotation:
    """Tarefa 78 — rotação de logs do LiveLoop.

    O CheckpointManager poda checkpoints antigos, mas health.jsonl e
    events.jsonl cresciam SEM LIMITE (41MB / 125k linhas observados em
    20/10/2026, com o disco do projeto a 90%). Log de operação contínua
    que nunca rotaciona é dívida disfarçada de telemetria.
    """

    def test_count_jsonl_lines(self, tmp_path):
        p = tmp_path / "x.jsonl"
        assert count_jsonl_lines(p) == 0  # inexistente -> 0
        _write_jsonl(p, 7)
        assert count_jsonl_lines(p) == 7

    def test_rotate_does_nothing_under_limit(self, tmp_path):
        p = tmp_path / "health.jsonl"
        _write_jsonl(p, 5)
        rotate_log_if_needed(p, max_lines=100)
        assert p.exists()
        assert count_jsonl_lines(p) == 5  # intacto

    def test_rotate_truncates_over_limit(self, tmp_path):
        p = tmp_path / "health.jsonl"
        _write_jsonl(p, 250)
        rotate_log_if_needed(p, max_lines=100, keep_lines=40)
        assert p.exists()  # nunca apaga o arquivo, só trunca
        assert count_jsonl_lines(p) == 40  # mantém a cauda (keep_lines)

    def test_rotate_keeps_tail_not_head(self, tmp_path):
        """As linhas mais recentes (cauda) sobrevivem — o histórico antigo já
        foi consolidado nos checkpoints."""
        p = tmp_path / "health.jsonl"
        _write_jsonl(p, 100)
        rotate_log_if_needed(p, max_lines=100, keep_lines=10)
        last = json.loads(open(p).readlines()[-1])
        assert last["i"] == 99  # a última linha é a mais recente

    def test_rotate_boundary_at_limit(self, tmp_path):
        p = tmp_path / "health.jsonl"
        _write_jsonl(p, 100)
        rotate_log_if_needed(p, max_lines=100, keep_lines=10)
        assert count_jsonl_lines(p) == 100  # exatamente no limite -> não rotaciona

    def test_rotate_default_limit_is_finite(self):
        # Regressão: o limite default NÃO pode ser ilimitado/None, e o
        # keep precisa ser menor que o max (senão nunca trunca).
        assert isinstance(ROTATE_MAX_LINES, int)
        assert ROTATE_MAX_LINES > 0
        assert ROTATE_MAX_LINES <= 100000
        assert 0 < ROTATE_KEEP_LINES < ROTATE_MAX_LINES

    def test_rotate_missing_file_is_noop(self, tmp_path):
        p = tmp_path / "nao_existe.jsonl"
        rotate_log_if_needed(p, max_lines=10)  # não deve levantar
        assert not p.exists()

    def test_liveloop_uses_custom_log_dir(self, tmp_path):
        """Regressão: log_dir customizado era ignorado (constantes de módulo
        capturadas na importação) e o log caía no diretório do repo."""
        log_dir = tmp_path / "meu_log"
        loop = LiveLoop(checkpoint_dir=tmp_path / "ck", log_dir=log_dir)
        assert loop.log_dir == log_dir
        assert loop.health_log == log_dir / "health.jsonl"
        assert loop.event_log == log_dir / "events.jsonl"
        assert loop.health_log != HEALTH_LOG  # não é mais o default global

    def test_liveloop_respects_rotation_limit(self, tmp_path):
        """Com log_dir isolado, verificar que a rotação é invocada e limita
        o arquivo — sem tocar o health.jsonl real do repositório."""
        log_dir = tmp_path / "lg"
        loop = LiveLoop(
            checkpoint_dir=tmp_path / "ck",
            log_dir=log_dir,
            max_log_lines=20,
            keep_log_lines=8,
        )
        assert loop.max_log_lines == 20
        _write_jsonl(log_dir / "health.jsonl", 50)
        _write_jsonl(log_dir / "events.jsonl", 50)
        loop._rotate_logs_if_needed()
        assert count_jsonl_lines(log_dir / "health.jsonl") == 8
        assert count_jsonl_lines(log_dir / "events.jsonl") == 8

    def test_rotation_leaves_valid_jsonl(self, tmp_path):
        """Após rotacionar, o que sobra precisa ser JSONL válido (não corrompido)."""
        p = tmp_path / "health.jsonl"
        _write_jsonl(p, 500)
        rotate_log_if_needed(p, max_lines=100)
        for line in open(p):
            json.loads(line)  # levanta se estiver corrompido


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
