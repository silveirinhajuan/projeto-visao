"""
test_live_loop.py — Testes para live_loop e monitor.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain
from visao.ops.live_loop import LiveLoop, run_live_loop
from visao.ops.monitor import BrainMonitor


class TestLiveLoop:
    def test_create_default(self):
        loop = LiveLoop()
        assert loop.brain is not None
        assert loop.checkpoint_interval == 1000

    def test_create_custom_brain(self):
        brain = VisaoBrain(n_in=2, n_hidden=16, n_out=1, seed=42)
        loop = LiveLoop(brain=brain)
        assert loop.brain is brain

    def test_fallback_data(self):
        loop = LiveLoop()
        x, y = loop._fallback_data()
        assert x.shape == (2,)
        assert y.shape == (1,)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))

    def test_recover_without_checkpoint(self):
        loop = LiveLoop()
        loop._recover()  # deve reiniciar sem crash
        assert loop._step == 0

    def test_recover_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loop = LiveLoop(checkpoint_dir=tmpdir, log_dir=tmpdir)
            # Treinar um pouco
            for i in range(10):
                x, y = loop._fallback_data()
                loop.brain.learn(x, y)
                loop._step += 1

            # Checkpoint
            loop._checkpoint()

            # Recuperar
            loop._recover()
            assert loop._step == 10

    def test_run_live_loop_short(self):
        """Testa execução curta do loop."""
        brain = run_live_loop(duration_seconds=2)
        assert brain.step > 0


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
