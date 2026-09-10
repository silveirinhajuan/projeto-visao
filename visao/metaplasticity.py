"""
metaplasticity.py — Tarefa 8.3: Metaplasticidade — aprende a taxa de aprendizado.

A metaplasticidade é o mecanismo que ajusta a taxa de aprendizado (lr) por
neurônio baseado na variância do erro observado para cada neurônio.

Conceito neurocientífico:
    Metaplasticidade = "plasticidade da plasticidade". A sinapse não apenas
    muda seu peso, mas muda QUÃO RÁPIDO ela muda seu peso baseado no histórico
    de atividade (Abraham & Bear, 1996 — regra BCM).

Regra de ajuste:
    - Neurônios com erro estável (baixa variância) → lr ALTO (aprendem rápido)
    - Neurônios com erro volátil (alta variância) → lr BAIXO (estabilizam)

Fórmula (inspirada em Adam, mas puramente local):
    σ_i = sqrt(EMA((err_i - mean_i)²))
    lr_i = lr_base / (1 + α · σ_i)

    Onde α controla a sensibilidade à variância.

Diferença para Adam:
    Adam normaliza o gradiente (direção). A metaplasticidade modula a taxa
    de aprendizado (velocidade) baseada na estabilidade local do erro.
    Não requer backprop — é puramente local.

Referências:
    - Abraham & Bear (1996) — Metaplasticity: the plasticity of synaptic plasticity
    - Kingma & Ba (2015) — Adam (inspiração para momentos)
    - Riedel et al. (2023) — Local meta-learning for liquid networks
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Garantir que o prototype/ seja importável
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prototype.plasticity import LocalLearner


class MetaPlasticityLearner(LocalLearner):
    """Learner com metaplasticidade: lr adaptativo por neurônio baseado em variância do erro.

    Herda de LocalLearner e adiciona:
    - Per-neuron learning rate (array de shape (n_out,))
    - Variance tracking por neurônio (EMA do erro)
    - Regra de adaptação: lr_i = lr_base / (1 + α · σ_i)

    Neurônios com erro estável (σ baixo) aprendem mais rápido.
    Neurônios com erro volátil (σ alto) aprendem mais devagar.

    Parâmetros adicionais vs LocalLearner:
    ----------
    meta_alpha : float (default 2.0)
        Sensibilidade à variância. Alto α → lr cai mais rápido com σ.
    meta_beta : float (default 0.9)
        Decay da EMA para tracking de variância. Próximo de 1 = memória longa.
    lr_min : float (default 0.001)
        Limite inferior do lr por neurônio (evita congelamento).
    lr_max : float (default 0.5)
        Limite superior do lr por neurônio (evita explosão).
    """

    def __init__(
        self,
        n_hidden: int,
        n_out: int,
        lr: float = 0.05,
        oja_lr: float = 0.002,
        consolidation: float = 1.0,
        surprise_gain: float = 3.0,
        meta_alpha: float = 2.0,
        meta_beta: float = 0.9,
        lr_min: float = 0.001,
        lr_max: float = 0.5,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            n_hidden=n_hidden,
            n_out=n_out,
            lr=lr,
            oja_lr=oja_lr,
            consolidation=consolidation,
            surprise_gain=surprise_gain,
            rng=rng,
        )

        self.lr_base = lr
        self.meta_alpha = meta_alpha
        self.meta_beta = meta_beta
        self.lr_min = lr_min
        self.lr_max = lr_max

        # Estado por neurônio de saída
        self.lr_per_neuron = np.full(n_out, lr)
        self._err_mean = np.zeros(n_out)   # EMA do erro (1º momento)
        self._err_var = np.ones(n_out)     # EMA da variância do erro (2º momento)
        self._post_omega_hook = None       # hook p/ EWC-temporal + surprise decay

    def update(self, x: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        """Um passo de aprendizado com metaplasticidade.

        Returns
        -------
        err_sq : float
            Erro quadrático médio do passo.
        surprise : float
            Fator de surpresa do passo.
        """
        pred = self.predict(x)
        err = target - pred  # shape (n_out,)
        err_mag = float(np.abs(err).mean())

        s = self.surprise(err_mag)

        # --- Metaplasticidade: atualiza variância do erro por neurônio ---
        d = err - self._err_mean
        self._err_mean = self.meta_beta * self._err_mean + (1.0 - self.meta_beta) * err
        # Variância = E[(err - mean)²] via EMA
        self._err_var = (
            self.meta_beta * self._err_var
            + (1.0 - self.meta_beta) * (d * d)
        )

        # Desvio padrão do erro por neurônio
        sigma = np.sqrt(np.abs(self._err_var)) + 1e-8

        # Regra de metaplasticidade: lr inversamente proporcional à variância
        # σ alto → denominador grande → lr baixo (estabiliza)
        # σ baixo → denominador próximo de 1 → lr alto (acelera)
        self.lr_per_neuron = self.lr_base / (1.0 + self.meta_alpha * sigma)
        self.lr_per_neuron = np.clip(self.lr_per_neuron, self.lr_min, self.lr_max)

        # lr efetivo: metaplasticidade × surpresa × consolidação
        # Shape: (n_out, 1) * scalar / (n_out, n_hidden) → (n_out, n_hidden)
        eff = (self.lr_per_neuron[:, None] * s) / (
            1.0 + self.consolidation * self.omega
        )

        # Delta rule com lr adaptativo
        delta = np.outer(err, x)
        self.W_out += eff * delta
        self.b_out += self.lr_per_neuron * s * err

        # Importância cresce onde a sinapse fez trabalho útil
        self.omega += 0.01 * np.abs(delta)

        # --- Hooks para mecanismos externos (EWC-temporal, surprise decay) ---
        if self._post_omega_hook is not None:
            self.omega = self._post_omega_hook(self.omega, s)

        # Baselines de surpresa (Welford simplificado)
        d_surp = err_mag - self.err_ema
        self.err_ema += 0.02 * d_surp
        self.err_var += 0.02 * (d_surp * d_surp - self.err_var)

        return float((err ** 2).mean()), s

    def set_post_omega_hook(self, hook):
        """Define callback para modificação pós-update de omega.

        Assinatura: hook(omega, surprise) -> omega_modificado
        Usado por VisaoBrain para injetar EWC-temporal e surprise decay.
        """
        self._post_omega_hook = hook

    def get_lr_stats(self) -> dict:
        """Retorna estatísticas do lr por neurônio para diagnóstico."""
        return {
            "lr_mean": float(np.mean(self.lr_per_neuron)),
            "lr_std": float(np.std(self.lr_per_neuron)),
            "lr_min": float(np.min(self.lr_per_neuron)),
            "lr_max": float(np.max(self.lr_per_neuron)),
            "lr_per_neuron": self.lr_per_neuron.copy(),
            "sigma_per_neuron": np.sqrt(np.abs(self._err_var)).copy(),
        }

    # ------------------------------------------------------------------ estado
    def state(self) -> dict:
        st = super().state()
        st.update({
            "lr_per_neuron": self.lr_per_neuron.copy(),
            "err_mean": self._err_mean.copy(),
            "err_var": self._err_var.copy(),
        })
        return st

    def load(self, st: dict) -> None:
        super().load(st)
        self.lr_per_neuron = st["lr_per_neuron"].copy()
        self._err_mean = st["err_mean"].copy()
        self._err_var = st["err_var"].copy()


class MetaPlasticityController:
    """Controlador standalone de metaplasticidade (pode ser usado com qualquer learner).

    Mantém o estado de metaplasticidade separado do learner, permitindo
    composição modular sem herança.

    Uso:
        ctrl = MetaPlasticityController(n_out=3, lr_base=0.05)
        for x, y in stream:
            lr = ctrl.get_lr()
            # ... treinar com lr ...
            ctrl.update_lr(err_por_neuron)
    """

    def __init__(
        self,
        n_out: int,
        lr_base: float = 0.05,
        meta_alpha: float = 2.0,
        meta_beta: float = 0.9,
        lr_min: float = 0.001,
        lr_max: float = 0.5,
    ):
        self.n_out = n_out
        self.lr_base = lr_base
        self.meta_alpha = meta_alpha
        self.meta_beta = meta_beta
        self.lr_min = lr_min
        self.lr_max = lr_max

        self.lr = np.full(n_out, lr_base)
        self._err_mean = np.zeros(n_out)
        self._err_var = np.ones(n_out)

    def get_lr(self) -> np.ndarray:
        """Retorna o lr atual por neurônio."""
        return self.lr.copy()

    def update_lr(self, err: np.ndarray) -> np.ndarray:
        """Atualiza lr baseado no erro observado.

        Parameters
        ----------
        err : array shape (n_out,) — erro por neurônio.

        Returns
        -------
        lr : array shape (n_out,) — novo lr por neurônio.
        """
        err = np.asarray(err, dtype=np.float64).ravel()
        d = err - self._err_mean
        self._err_mean = self.meta_beta * self._err_mean + (1.0 - self.meta_beta) * err
        self._err_var = (
            self.meta_beta * self._err_var + (1.0 - self.meta_beta) * (d * d)
        )

        sigma = np.sqrt(np.abs(self._err_var)) + 1e-8
        self.lr = self.lr_base / (1.0 + self.meta_alpha * sigma)
        self.lr = np.clip(self.lr, self.lr_min, self.lr_max)
        return self.lr.copy()

    def reset(self) -> None:
        """Reseta o estado de metaplasticidade."""
        self.lr = np.full(self.n_out, self.lr_base)
        self._err_mean = np.zeros(self.n_out)
        self._err_var = np.ones(self.n_out)
