"""
brain.py — Tarefa 8.0: VISÃO Brain — classe unificada.

Combina TODOS os mecanismos validados do Projeto VISÃO em uma única classe:
  - LiquidCell (LTC/CfC): reservatório com tau dependente da entrada
  - EWC-temporal: consolidação com decaimento exponencial de importância
  - Surprise decay: surpresa decai omega (não amplifica lr)
  - Oja: auto-organização do recorrente
  - Meta-learning: ajuste adaptativo de lr

Referências dos mecanismos:
  - Fase 0 / 0.1: prototype/liquid.py + plasticity.py (esquecimento 97.2% menor)
  - 7.1: surprise_decay.py (surpresa como decaimento de omega)
  - 7.2: ewc_temporal.py (decaimento temporal de importancia)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Garantir que o prototype/ seja importável
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prototype.liquid import LiquidCell
from prototype.plasticity import LocalLearner


class VisaoBrain:
    """Cérebro VISÃO: aprendizado contínuo com plasticidade local.

    Combina todos os mecanismos validados:
      - Reservatório líquido (LTC/CfC) com tau variável
      - Readout local com delta rule
      - Consolidação por importância (EWC) com decaimento temporal
      - Surpresa como decaimento de omega (não amplificação de lr)
      - Oja no recorrente (auto-organização)
      - Meta-learning (ajuste adaptativo de lr)
      - Modo inferência eficiente

    API
    ---
    brain = VisaoBrain(n_in=2, n_hidden=64, n_out=1)
    brain.learn(x, y)   # um passo de aprendizado
    brain.forward(x)    # inferência (mais rápido, sem overhead)
    brain.set_mode("learn" | "infer")
    brain.save(path) / brain.load(path)

    Parâmetros
    ----------
    n_in : dimensão da entrada
    n_hidden : número de neurônios líquidos (reservatório)
    n_out : dimensão da saída
    sparsity : fração de sinapses recorrentes zeradas
    tau_min, tau_max : faixa das constantes de tempo base
    lr : taxa de aprendizado base
    oja_lr : taxa de aprendizado do Oja
    consolidation : força da consolidação (EWC)
    surprise_gain : sensibilidade do gate de surpresa
    lambda_decay : decaimento temporal de importância
    meta_learn : se True, ajusta lr baseado na tendência de erro
    dt : passo de integração do reservatório
    seed : semente para reprodutibilidade
    """

    def __init__(
        self,
        n_in: int = 2,
        n_hidden: int = 64,
        n_out: int = 1,
        sparsity: float = 0.6,
        tau_min: float = 0.4,
        tau_max: float = 4.0,
        lr: float = 0.02,
        oja_lr: float = 0.0015,
        consolidation: float = 8.0,
        surprise_gain: float = 3.0,
        lambda_decay: float = 0.0005,
        meta_learn: bool = False,
        adaptive_consolidation: bool = False,
        c_min: float = 0.0,
        c_max: float = 16.0,
        dt: float = 0.15,
        seed: int = 0,
    ):
        # --- Hiperparâmetros ---
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr_base = lr
        self.lambda_decay = lambda_decay
        self.meta_learn = meta_learn
        self._mode = "learn"

        # Estado interno
        self._rng = np.random.default_rng(seed)
        self._step = 0

        # --- Sub-módulos ---
        self.cell = LiquidCell(
            n_in=n_in,
            n_hidden=n_hidden,
            sparsity=sparsity,
            tau_min=tau_min,
            tau_max=tau_max,
            dt=dt,
            rng=self._rng,
        )

        # Learner (readout) — usamos a classe base LocalLearner
        # e aplicamos EWC-temporal + surprise decay manualmente
        self.learner = LocalLearner(
            n_hidden=n_hidden,
            n_out=n_out,
            lr=lr,
            oja_lr=oja_lr,
            consolidation=consolidation,
            surprise_gain=surprise_gain,
            rng=self._rng,
        )

        # Estado do reservatório
        self.x = np.zeros(n_hidden)

        # --- Meta-learning ---
        self.lr = lr
        self._err_trend = 0.0
        self._prev_err = 1.0
        self._meta_alpha = 0.01  # taxa de ajuste do meta-learner
        self._meta_min = lr * 0.1
        self._meta_max = lr * 5.0

        # --- Consolidação adaptativa ---
        self.adaptive_consolidation = adaptive_consolidation
        self.c_min = c_min
        self.c_max = c_max
        self._surprise_window = []
        self._window_size = 100

    # ==============================================================
    #  API PÚBLICA
    # ==============================================================

    def learn(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Um passo completo: forward do reservatório + aprendizado local.

        Raises
        ------
        ValueError
            Se chamado em modo "infer" — este modo só permite forward().
        """
        if self._mode == "infer":
            raise ValueError(
                "learn() não disponível no modo 'infer'. "
                "Use forward() para inferência ou set_mode('learn')."
            )
        # ... rest unchanged
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        # 1. Forward do reservatório líquido
        x_prev = self.x.copy()
        self.x, _ = self.cell.step(self.x, x)

        # 2. Predição do readout
        pred = self.learner.predict(self.x)

        # 3. Aprendizado local (com todos os mecanismos)
        err_sq, surprise = self._local_update(self.x, x_prev, y)

        # 4. Meta-learning
        if self.meta_learn:
            self._meta_update(float(np.abs(y - pred).mean()))

        self._step += 1

        return {
            "pred": pred,
            "err": float(np.sqrt(max(err_sq, 0.0))),
            "surprise": surprise,
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
            """Inferência rápida: só forward do reservatório + readout.

            Em modo "infer", é a única operação permitida.
            Otimizado: inline do step() para evitar overhead de chamada
            e cópia de estado (x_prev) que só é necessário para Oja.
            Raises
            ------
            ValueError
                Se chamado em modo "learn" — modo aprendizado requer
                atualização de pesos via learn().
            """
            x = np.asarray(x, dtype=np.float64).ravel()
            if self._mode == "learn":
                raise ValueError(
                    "forward() não disponível no modo 'learn'. "
                    "Use learn() para treinamento ou set_mode('infer')."
                )
            # Inline de cell.step() — evita chamada de função + cópia de x_prev
            fx = self.cell.f(self.x, x)
            num = self.x + self.cell.dt * fx * self.cell.A
            den = 1.0 + self.cell.dt * (1.0 / self.cell.tau + fx)
            self.x = num / den
            # Readout
            return self.learner.W_out @ self.x + self.learner.b_out

    def set_mode(self, mode: str) -> "VisaoBrain":
        """Alterna entre 'learn' (padrão) e 'infer' (só forward)."""
        if mode not in ("learn", "infer"):
            raise ValueError(f"mode deve ser 'learn' ou 'infer', recebido {mode!r}")
        self._mode = mode
        return self

    def reset_state(self) -> None:
        """Reseta o estado interno do reservatório (não os pesos)."""
        self.x = np.zeros(self.n_hidden)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def step(self) -> int:
        return self._step

    # ==============================================================
    #  PERSISTÊNCIA
    # ==============================================================

    def save(self, path: str | Path) -> None:
        """Salva o cérebro em JSON compacto (pesos + estado + hiperparams)."""
        data = {
            "config": {
                "n_in": self.n_in,
                "n_hidden": self.n_hidden,
                "n_out": self.n_out,
                "sparsity": float(np.mean(self.cell.mask == 0)),
                "tau_min": float(np.min(self.cell.tau)),
                "tau_max": float(np.max(self.cell.tau)),
                "lr_base": self.lr_base,
                "lambda_decay": self.lambda_decay,
                "meta_learn": self.meta_learn,
                "dt": self.cell.dt,
                "seed": int(self._rng.integers(0, 2**31)),
            },
            "weights": {
                "W_in": self.cell.W_in.tolist(),
                "W_rec": self.cell.W_rec.tolist(),
                "b": self.cell.b.tolist(),
                "A": self.cell.A.tolist(),
                "tau": self.cell.tau.tolist(),
                "W_out": self.learner.W_out.tolist(),
                "b_out": self.learner.b_out.tolist(),
                "omega": self.learner.omega.tolist(),
            },
            "state": {
                "x": self.x.tolist(),
                "err_ema": self.learner.err_ema,
                "err_var": self.learner.err_var,
                "lr": self.lr,
                "step": self._step,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))

    @classmethod
    def load(cls, path: str | Path) -> "VisaoBrain":
        """Carrega cérebro de JSON."""
        with open(path) as f:
            data = json.load(f)

        cfg = data["config"]
        brain = cls(
            n_in=cfg["n_in"],
            n_hidden=cfg["n_hidden"],
            n_out=cfg["n_out"],
            sparsity=cfg["sparsity"],
            tau_min=cfg["tau_min"],
            tau_max=cfg["tau_max"],
            lr=cfg["lr_base"],
            lambda_decay=cfg["lambda_decay"],
            meta_learn=cfg["meta_learn"],
            dt=cfg["dt"],
            seed=cfg["seed"],
        )

        w = data["weights"]
        brain.cell.W_in = np.array(w["W_in"], dtype=np.float64)
        brain.cell.W_rec = np.array(w["W_rec"], dtype=np.float64)
        brain.cell.b = np.array(w["b"], dtype=np.float64)
        brain.cell.A = np.array(w["A"], dtype=np.float64)
        brain.cell.tau = np.array(w["tau"], dtype=np.float64)
        brain.learner.W_out = np.array(w["W_out"], dtype=np.float64)
        brain.learner.b_out = np.array(w["b_out"], dtype=np.float64)
        brain.learner.omega = np.array(w["omega"], dtype=np.float64)

        st = data["state"]
        brain.x = np.array(st["x"], dtype=np.float64)
        brain.learner.err_ema = st["err_ema"]
        brain.learner.err_var = st["err_var"]
        brain.lr = st["lr"]
        brain._step = st["step"]

        return brain

    # ==============================================================
    #  INTERNOS
    # ==============================================================

    def _local_update(
        self, x: np.ndarray, x_prev: np.ndarray, target: np.ndarray
    ) -> tuple[float, float]:
        """Um passo de aprendizado local com todos os mecanismos.

        Aplica na ordem:
          1. Predição
          2. Surpresa (gate neuromodulatório)
          3. Consolidação adaptativa (se ativada)
          4. lr efetivo: base / (1 + consolidation * omega)
          5. Atualização delta rule
          6. Crescimento de importancia (EWC)
          7. Decaimento temporal de importancia (EWC-temporal)
          8. Surpresa como decaimento de omega (surprise_decay 7.1)
          9. Oja no recorrente
          10. Atualização dos baselines de surpresa
        """
        learner = self.learner

        # 1. Predição
        pred = learner.predict(x)
        err = target - pred
        err_mag = float(np.abs(err).mean())

        # 2. Surpresa
        s = learner.surprise(err_mag)

        # 3. Consolidação adaptativa
        if self.adaptive_consolidation:
            self._surprise_window.append(s)
            if len(self._surprise_window) > self._window_size:
                self._surprise_window.pop(0)
            if len(self._surprise_window) >= self._window_size:
                mean_s = np.mean(self._surprise_window)
                normalized = np.clip((mean_s - 1.0) / 1.0, 0.0, 1.0)
                learner.consolidation = self.c_min + (self.c_max - self.c_min) * normalized

        # 4. lr efetivo: EWC fecha pela importancia; surpresa NÃO amplifica
        #    (conforme 7.1: surpresa decai omega, não amplifica lr)
        eff = learner.lr / (1.0 + learner.consolidation * learner.omega)

        # 5. Delta rule
        delta = np.outer(err, x)
        learner.W_out += eff * delta
        learner.b_out += learner.lr * err

        # 6. Crescimento de importancia (EWC)
        learner.omega += 0.01 * np.abs(delta)

        # 7. Decaimento temporal de importancia (EWC-temporal 7.2)
        if self.lambda_decay > 0:
            learner.omega *= np.exp(-self.lambda_decay)

        # 8. Surpresa como decaimento de omega (surprise_decay 7.1)
        #    Surpresa alta -> decai omega (afrouxa consolidacao, permite aprender)
        if s > 1.0:
            decay = np.exp(-(s - 1.0) * 0.5)  # factor ~0.6 for s=2
            learner.omega *= decay

        # 9. Oja no recorrente (auto-organizacao)
        if learner.oja_lr > 0:
            learner.oja_update(self.cell, x_prev, x)

        # 10. Atualizacao dos baselines de surpresa (Welford)
        d = err_mag - learner.err_ema
        learner.err_ema += 0.02 * d
        learner.err_var += 0.02 * (d * d - learner.err_var)

        return float((err ** 2).mean()), s

    def _meta_update(self, err_mag: float) -> None:
        """Meta-learning: ajusta lr baseado na tendência de erro.

        - Erro descendo -> lr pode subir (aprender mais rapido)
        - Erro subindo -> lr deve descer (estabilizar)
        """
        trend = err_mag - self._prev_err
        self._err_trend = 0.9 * self._err_trend + 0.1 * trend
        self._prev_err = err_mag

        if self._err_trend > 0:
            # Erro subindo: reduzir lr
            self.lr *= (1.0 - self._meta_alpha)
        else:
            # Erro descendo: aumentar lr (cautelosamente)
            self.lr *= (1.0 + self._meta_alpha * 0.5)

        # Clamp
        self.lr = float(np.clip(self.lr, self._meta_min, self._meta_max))
        self.learner.lr = self.lr

    # ==============================================================
    #  UTILITÁRIOS
    # ==============================================================

    def evaluate_stream(self, X: np.ndarray, Y: np.ndarray, warmup: int = 50) -> dict:
        """Avalia o cérebro em um stream de dados (sem aprender).

        Retorna {"mse": ..., "mae": ..., "n": ...}
        """
        self.reset_state()
        preds = []
        for i, (xi, yi) in enumerate(zip(X, Y)):
            pred = self.forward(xi)
            if i >= warmup:
                preds.append((pred, yi))

        if not preds:
            return {"mse": float("inf"), "mae": float("inf"), "n": 0}

        preds_arr = np.array([p for p, _ in preds])
        targets_arr = np.array([t for _, t in preds])
        mse = float(np.mean((preds_arr - targets_arr) ** 2))
        mae = float(np.mean(np.abs(preds_arr - targets_arr)))
        return {"mse": mse, "mae": mae, "n": len(preds)}

    def __repr__(self) -> str:
        n_params = (
            self.cell.W_in.size
            + self.cell.W_rec.size
            + self.cell.b.size
            + self.cell.A.size
            + self.learner.W_out.size
            + self.learner.b_out.size
        )
        return (
            f"VisaoBrain(n_in={self.n_in}, n_hidden={self.n_hidden}, "
            f"n_out={self.n_out}, params={n_params}, step={self._step}, "
            f"mode={self._mode!r})"
        )
