"""
test_brain.py — Testes para VisaoBrain (Tarefa 8.0).

Cobre:
  - Instanciação e parametrizacão
  - Aprendizado (modo learn)
  - Inferência (modo infer)
  - Persistência (save/load round-trip)
  - Continuidade (não esquece após tarefa B)
  - Meta-learning (ajuste adaptativo de lr)
  - Eficiência do modo infer vs learn
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


# ==============================================================
#  FIXTURES
# ==============================================================

@pytest.fixture
def simple_brain():
    """Cérebro pequeno para testes rápidos."""
    return VisaoBrain(n_in=2, n_hidden=16, n_out=1, seed=42)


@pytest.fixture
def meta_brain():
    """Cérebro com meta-learning."""
    return VisaoBrain(n_in=2, n_hidden=16, n_out=1, meta_learn=True, seed=42)


def make_task(kind: str = "sine", n: int = 500, seed: int = 0, noise: float = 0.1):
    """Gera tarefa sintética (regressão)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if kind == "sine":
        freq = 0.05
        phase = 0.0
        signal = np.sin(2 * np.pi * freq * t + phase)
    elif kind == "saw":
        signal = 2.0 * (t % 50) / 50.0 - 1.0
    elif kind == "mixed":
        signal = np.sin(2 * np.pi * 0.03 * t) + 0.5 * np.cos(2 * np.pi * 0.07 * t)
    else:
        signal = rng.normal(0, 0.5, n)

    u = rng.normal(0, noise, (n, 2))
    u[:, 0] += signal
    y = np.convolve(signal, np.ones(5) / 5, mode="same")[:, None]
    return u.astype(np.float64), y.astype(np.float64)


# ==============================================================
#  TESTES: INSTANCIAÇÃO E API
# ==============================================================

class TestInstantiation:
    def test_create_default(self):
        brain = VisaoBrain()
        assert brain.n_in == 2
        assert brain.n_hidden == 64
        assert brain.n_out == 1

    def test_create_custom(self):
        brain = VisaoBrain(n_in=3, n_hidden=32, n_out=2)
        assert brain.n_in == 3
        assert brain.n_hidden == 32
        assert brain.n_out == 2

    def test_repr(self, simple_brain):
        s = repr(simple_brain)
        assert "VisaoBrain" in s
        assert "n_hidden=16" in s

    def test_set_mode(self, simple_brain):
        simple_brain.set_mode("infer")
        assert simple_brain.mode == "infer"
        simple_brain.set_mode("learn")
        assert simple_brain.mode == "learn"

    def test_set_mode_invalid(self, simple_brain):
        with pytest.raises(ValueError):
            simple_brain.set_mode("invalid")

    def test_reset_state(self, simple_brain):
        simple_brain.set_mode("infer")
        simple_brain.forward([0.5, 0.3])
        simple_brain.reset_state()
        assert np.allclose(simple_brain.x, 0.0)


# ==============================================================
#  TESTES: APRENDIZADO
# ==============================================================

class TestLearning:
    def test_learn_returns_dict(self, simple_brain):
        result = simple_brain.learn([0.5, 0.3], [0.7])
        assert "pred" in result
        assert "err" in result
        assert "surprise" in result

    def test_learn_step_increment(self, simple_brain):
        assert simple_brain.step == 0
        simple_brain.learn([0.5, 0.3], [0.7])
        assert simple_brain.step == 1

    def test_learn_reduces_error(self, simple_brain):
        """Após treinar na mesma tarefa, erro deve diminuir."""
        u, y = make_task("sine", n=200, seed=0)
        errs = []
        for i in range(min(100, len(u))):
            result = simple_brain.learn(u[i], y[i])
            errs.append(result["err"])

        # Erro final < erro inicial (média últimos 20 < média primeiros 20)
        assert np.mean(errs[-20:]) < np.mean(errs[:20])

    def test_forward_without_learning(self, simple_brain):
        """Modo infer: forward não atualiza pesos."""
        w_before = simple_brain.learner.W_out.copy()
        simple_brain.set_mode("infer")
        simple_brain.forward([0.5, 0.3])
        w_after = simple_brain.learner.W_out
        assert np.allclose(w_before, w_after)

    def test_surprise_is_positive(self, simple_brain):
        result = simple_brain.learn([0.5, 0.3], [0.7])
        assert result["surprise"] > 0

    def test_batch_learn(self, simple_brain):
        """Aprende em batch sem erros."""
        u, y = make_task("sine", n=50, seed=0)
        for ui, yi in zip(u, y):
            simple_brain.learn(ui, yi)
        assert simple_brain.step == 50


# ==============================================================
#  TESTES: PERSISTÊNCIA
# ==============================================================

class TestPersistence:
    def test_save_load_roundtrip(self, simple_brain):
        """Save/load deve preservar pesos."""
        u, y = make_task("sine", n=50, seed=0)
        for ui, yi in zip(u, y):
            simple_brain.learn(ui, yi)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            simple_brain.save(path)
            brain2 = VisaoBrain.load(path)
            assert np.allclose(simple_brain.learner.W_out, brain2.learner.W_out)
            assert np.allclose(simple_brain.learner.omega, brain2.learner.omega)
            assert brain2.step == simple_brain.step
        finally:
            Path(path).unlink()

    def test_load_config_matches(self):
        brain = VisaoBrain(n_in=2, n_hidden=20, n_out=1, lr=0.03, seed=123)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            brain.save(path)
            brain2 = VisaoBrain.load(path)
            assert brain2.n_hidden == 20
            assert brain2.lr_base == 0.03
        finally:
            Path(path).unlink()


# ==============================================================
#  TESTES: CONTINUAL LEARNING (anti-esquecimento)
# ==============================================================

class TestContinual:
    def test_two_tasks_no_catastrophic_forgetting(self):
        """Aprender tarefa B não deve degradar catastroficamente tarefa A."""
        brain = VisaoBrain(n_in=2, n_hidden=32, n_out=1, seed=42)

        # Treinar tarefa A
        u_a, y_a = make_task("sine", n=300, seed=0)
        for ui, yi in zip(u_a, y_a):
            brain.learn(ui, yi)

        # Avaliar A (em modo infer, com estado limpo)
        brain.set_mode("infer")
        eval_a_before = brain.evaluate_stream(u_a[-100:], y_a[-100:])

        # Treinar tarefa B
        brain.set_mode("learn")
        u_b, y_b = make_task("saw", n=300, seed=1)
        for ui, yi in zip(u_b, y_b):
            brain.learn(ui, yi)

        # Avaliar A novamente
        brain.set_mode("infer")
        eval_a_after = brain.evaluate_stream(u_a[-100:], y_a[-100:])

        # Esquecimento deve ser pequeno (< 100% de aumento no MSE)
        # Usar valor absoluto para evitar divisão por zero
        mse_before = max(eval_a_before["mse"], 1e-6)
        mse_after = eval_a_after["mse"]
        degradation = (mse_after - mse_before) / mse_before
        assert degradation < 1.0, f"Esquecimento catastrófico: {degradation:.1%}"

    def test_ewc_reduces_forgetting_vs_naive(self):
        """EWC-temporal deve esquecer menos que baseline sem consolidacao."""
        # Com EWC
        brain_ewc = VisaoBrain(n_in=2, n_hidden=32, n_out=1, consolidation=8.0, seed=42)
        # Sem EWC (consolidation=0)
        brain_naive = VisaoBrain(n_in=2, n_hidden=32, n_out=1, consolidation=0.0, seed=42)

        u_a, y_a = make_task("sine", n=200, seed=0)
        u_b, y_b = make_task("saw", n=200, seed=1)

        for brain in [brain_ewc, brain_naive]:
            for ui, yi in zip(u_a, y_a):
                brain.learn(ui, yi)

        eval_before_ewc = brain_ewc.evaluate_stream(u_a[-100:], y_a[-100:])
        eval_before_naive = brain_naive.evaluate_stream(u_a[-100:], y_a[-100:])

        for brain in [brain_ewc, brain_naive]:
            for ui, yi in zip(u_b, y_b):
                brain.learn(ui, yi)

        eval_after_ewc = brain_ewc.evaluate_stream(u_a[-100:], y_a[-100:])
        eval_after_naive = brain_naive.evaluate_stream(u_a[-100:], y_a[-100:])

        # EWC deve ter menos esquecimento (diferença absoluta)
        forget_ewc = eval_after_ewc["mse"] - eval_before_ewc["mse"]
        forget_naive = eval_after_naive["mse"] - eval_before_naive["mse"]

        # EWC <= naive (com margem para variação estocástica)
        assert forget_ewc <= forget_naive + 0.05


# ==============================================================
#  TESTES: META-LEARNING
# ==============================================================

class TestMetaLearning:
    def test_meta_adjusts_lr(self, meta_brain):
        """Meta-learning deve ajustar lr."""
        lr_before = meta_brain.lr
        u, y = make_task("sine", n=100, seed=0)
        for ui, yi in zip(u, y):
            meta_brain.learn(ui, yi)
        # lr deve ter mudado
        assert meta_brain.lr != lr_before or meta_brain._err_trend != 0

    def test_meta_lr_bounded(self, meta_brain):
        """lr deve ficar dentro dos limites."""
        u, y = make_task("sine", n=200, seed=0)
        for ui, yi in zip(u, y):
            meta_brain.learn(ui, yi)
        assert meta_brain.lr >= meta_brain._meta_min
        assert meta_brain.lr <= meta_brain._meta_max


# ==============================================================
#  TESTES: AVALIAÇÃO
# ==============================================================

class TestEvaluation:
    def test_evaluate_stream(self, simple_brain):
        u, y = make_task("sine", n=100, seed=0)
        result = simple_brain.evaluate_stream(u, y, warmup=20)
        assert "mse" in result
        assert "mae" in result
        assert result["mse"] >= 0
        assert result["mae"] >= 0

    def test_evaluate_better_after_learning(self, simple_brain):
        """Após aprender, avaliação deve melhorar (em held-out data)."""
        # Dados de treino
        u_tr, y_tr = make_task("sine", n=300, seed=0, noise=0.01)
        # Dados de teste (mesma distribuição, diferente realização)
        u_te, y_te = make_task("sine", n=100, seed=99, noise=0.01)

        eval_before = simple_brain.evaluate_stream(u_te, y_te)

        for ui, yi in zip(u_tr, y_tr):
            simple_brain.learn(ui, yi)

        eval_after = simple_brain.evaluate_stream(u_te, y_te)
        assert eval_after["mse"] < eval_before["mse"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
