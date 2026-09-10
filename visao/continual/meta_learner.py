"""meta_learner.py — MetaContinualLearner: meta-learning para continual learning.

Implementa meta-learning inspirado em MAML/RL² para:
  1. Adaptar a taxa de aprendizado por tarefa (per-task lr)
  2. Detectar a tarefa dominante atual via assinatura de erro/ativação
  3. Ajustar automaticamente EWC (consolidation) e Surprise (surprise_gain)

Mecanismos:
  - TaskSignature: buffer circular de (err_ema, activation_mean, activation_var)
    que identifica a tarefa corrente por similaridade.
  - TaskBank: dicionário de tarefas conhecidas com assinatura + hiperparâmetros
    ótimos aprendidos (lr, consolidation, surprise_gain).
  - Meta-update (estilo MAML): após detectar transição, ajusta hiperparâmetros
    baseado no histórico de transições anteriores (meta-learning de hiperparâmetros).
  - Adaptive EWC/Surprise: quando tarefa conhecida é detectada, carrega hiperparâmetros
    ótimos; quando tarefa nova explora com consolidação reduzida e surpresa amplificada.

Referências:
  - Finn et al. (2017) — MAML
  - Duan et al. (2016) — RL²
  - Kirkpatrick et al. (2017) — EWC
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Garantir imports do projeto
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visao.brain import VisaoBrain


class TaskSignature:
    """Assinatura compacta de uma tarefa baseada em estatísticas de erro e ativação.

    Usada para detectar a tarefa dominante atual e recuperar hiperparâmetros
    ótimos do banco de tarefas.
    """

    def __init__(self, n_hidden: int, buffer_size: int = 50):
        self.n_hidden = n_hidden
        self.buffer_size = buffer_size
        self._err_buffer: list[float] = []
        self._act_means: list[np.ndarray] = []
        self._act_vars: list[np.ndarray] = []

    def update(self, err_mag: float, activation: np.ndarray) -> None:
        """Atualiza buffer circular com observação atual."""
        self._err_buffer.append(err_mag)
        if len(self._err_buffer) > self.buffer_size:
            self._err_buffer.pop(0)

        self._act_means.append(activation.copy())
        if len(self._act_means) > self.buffer_size:
            self._act_means.pop(0)

        self._act_vars.append((activation ** 2).copy())
        if len(self._act_vars) > self.buffer_size:
            self._act_vars.pop(0)

    def is_ready(self) -> bool:
        return len(self._err_buffer) >= self.buffer_size // 2

    def vector(self) -> np.ndarray:
        """Retorna vetor de assinatura (err_stats + act_stats)."""
        if not self._err_buffer:
            return np.zeros(7)

        err_arr = np.array(self._err_buffer)
        act_mean_arr = np.array([a.mean() for a in self._act_means])
        act_var_arr = np.array([a.mean() for a in self._act_vars])

        sig = np.array([
            err_arr.mean(), err_arr.std(), err_arr[-1],
            act_mean_arr.mean(), act_mean_arr.std(),
            act_var_arr.mean(), act_var_arr.std(),
        ])
        return sig

    def centroid(self) -> np.ndarray:
        """Centroide das ativações (para matching)."""
        if not self._act_means:
            return np.zeros(self.n_hidden)
        return np.mean(self._act_means[-self.buffer_size // 2:], axis=0)

    def reset(self) -> None:
        self._err_buffer.clear()
        self._act_means.clear()
        self._act_vars.clear()


class TaskEntry:
    """Entrada no banco de tarefas: assinatura + hiperparâmetros ótimos."""

    def __init__(self, task_id: int, signature: TaskSignature):
        self.task_id = task_id
        self.signature = signature
        self.lr_opt: float = 0.02
        self.consolidation_opt: float = 8.0
        self.surprise_gain_opt: float = 3.0
        self.n_observations: int = 0
        self.total_error: float = 0.0

    @property
    def mean_error(self) -> float:
        if self.n_observations == 0:
            return float("inf")
        return self.total_error / self.n_observations

    def similarity(self, sig_vec: np.ndarray) -> float:
        """Similaridade cosseno com a assinatura armazenada."""
        ref = self.signature.vector()
        norm_ref = np.linalg.norm(ref)
        norm_sig = np.linalg.norm(sig_vec)
        if norm_ref < 1e-10 or norm_sig < 1e-10:
            return 0.0
        return float(np.dot(ref, sig_vec) / (norm_ref * norm_sig))


class MetaContinualLearner:
    """Meta-learner para continual learning com adaptação automática.

    Envolve um VisaoBrain e adiciona:
    1. Adaptação de lr por tarefa (per-task lr via TaskBank)
    2. Detecção de tarefa dominante (TaskSignature + similarity matching)
    3. Ajuste automático de EWC (consolidation) e Surprise (surprise_gain)

    Uso:
        mcl = MetaContinualLearner(n_in=2, n_hidden=64, n_out=1)
        for task_stream in tasks:
            for x, y in task_stream:
                mcl.learn(x, y)
            mcl.end_task()  # consolida metaplasticidade da tarefa
    """

    def __init__(
        self,
        n_in: int = 2,
        n_hidden: int = 64,
        n_out: int = 1,
        lr_base: float = 0.02,
        consolidation: float = 8.0,
        surprise_gain: float = 3.0,
        lambda_decay: float = 0.0005,
        signature_buffer: int = 50,
        similarity_threshold: float = 0.85,
        exploration_factor: float = 0.5,
        adapt_ewc: bool = True,
        adapt_surprise: bool = True,
        adapt_lr: bool = True,
        seed: int = 0,
    ):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr_base = lr_base
        self.consolidation_base = consolidation
        self.surprise_gain_base = surprise_gain
        self.lambda_decay = lambda_decay
        self.similarity_threshold = similarity_threshold
        self.exploration_factor = exploration_factor
        self.adapt_ewc = adapt_ewc
        self.adapt_surprise = adapt_surprise
        self.adapt_lr = adapt_lr

        # Cérebro base
        self.brain = VisaoBrain(
            n_in=n_in,
            n_hidden=n_hidden,
            n_out=n_out,
            lr=lr_base,
            consolidation=consolidation,
            surprise_gain=surprise_gain,
            lambda_decay=lambda_decay,
            meta_learn=False,  # controlamos meta-learning aqui
            seed=seed,
        )

        # Estado meta
        self.task_bank: dict[int, TaskEntry] = {}
        self.current_task_id: int = 0
        self._next_task_id: int = 0
        self.signature = TaskSignature(n_hidden, buffer_size=signature_buffer)
        self._task_start_step: int = 0
        self._task_errors: list[float] = []
        self._task_steps: int = 0

        # Histórico de transições para meta-learning
        self._transition_history: list[dict] = []

    @property
    def current_task(self) -> Optional[TaskEntry]:
        return self.task_bank.get(self.current_task_id)

    def _detect_task(self) -> tuple[int, float]:
        """Detecta tarefa dominante baseada em similaridade de assinatura.

        Returns
        -------
        task_id : int
            ID da tarefa detectada (ou nova tarefa se similaridade baixa).
        similarity : float
            Similaridade com a tarefa detectada (0 se tarefa nova).
        """
        if not self.signature.is_ready():
            return self.current_task_id, 0.0

        sig_vec = self.signature.vector()

        best_id = -1
        best_sim = -1.0
        for tid, entry in self.task_bank.items():
            sim = entry.similarity(sig_vec)
            if sim > best_sim:
                best_sim = sim
                best_id = tid

        if best_sim >= self.similarity_threshold and best_id >= 0:
            return best_id, best_sim

        return -1, best_sim  # sinaliza nova tarefa

    def _create_task_entry(self) -> int:
        """Cria nova entrada no banco de tarefas."""
        tid = self._next_task_id
        self._next_task_id += 1
        entry = TaskEntry(tid, TaskSignature(self.n_hidden))
        entry.lr_opt = self.lr_base
        entry.consolidation_opt = self.consolidation_base
        entry.surprise_gain_opt = self.surprise_gain_base
        self.task_bank[tid] = entry
        return tid

    def _apply_task_hyperparams(self, task_id: int, is_new: bool = False) -> None:
        """Aplica hiperparâmetros da tarefa detectada ao cérebro."""
        if task_id < 0 or task_id not in self.task_bank:
            # Tarefa nova: exploração (reduz EWC, amplifica surpresa)
            if self.adapt_ewc:
                self.brain.learner.consolidation = self.consolidation_base * self.exploration_factor
            if self.adapt_surprise:
                self.brain.learner.surprise_gain = self.surprise_gain_base * (2.0 - self.exploration_factor)
            if self.adapt_lr:
                self.brain.learner.lr = self.lr_base * 1.5
        else:
            entry = self.task_bank[task_id]
            if self.adapt_ewc:
                self.brain.learner.consolidation = entry.consolidation_opt
            if self.adapt_surprise:
                self.brain.learner.surprise_gain = entry.surprise_gain_opt
            if self.adapt_lr:
                self.brain.learner.lr = entry.lr_opt

        self.current_task_id = max(task_id, 0)

    def learn(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Um passo de aprendizado com meta-adaptação.

        A cada chamada:
        1. Executa learn() no cérebro
        2. Atualiza assinatura da tarefa
        3. A cada N passos, detecta tarefa e ajusta hiperparâmetros
        """
        result = self.brain.learn(x, y)

        # Atualiza assinatura
        self.signature.update(result["err"], self.brain.x)
        self._task_errors.append(result["err"])
        self._task_steps += 1

        # Detecção periódica (a cada 25 passos para não sobrecarregar)
        if self._task_steps % 25 == 0 and self.signature.is_ready():
            task_id, sim = self._detect_task()
            if task_id != self.current_task_id:
                # Transição detectada
                if task_id == -1:
                    task_id = self._create_task_entry()
                    self._apply_task_hyperparams(task_id, is_new=True)
                else:
                    self._apply_task_hyperparams(task_id, is_new=False)

        return result

    def end_task(self) -> dict:
        """Finaliza tarefa atual: consolida hiperparâmetros ótimos no banco.

        Chamado entre tarefas para consolidar metaplasticidade.
        """
        if self.current_task_id < 0 or self.current_task_id not in self.task_bank:
            return {}

        entry = self.task_bank[self.current_task_id]

        # Atualiza média de erro da tarefa
        if self._task_errors:
            entry.total_error = float(np.mean(self._task_errors[-100:]))
            entry.n_observations += 1

        # Meta-learning: ajusta hiperparâmetros ótimos baseado no desempenho
        # (estilo MAML outer loop: se erro alto, aumenta lr; se erro baixo, consolida mais)
        if len(self._task_errors) > 20:
            recent_err = np.mean(self._task_errors[-20:])
            old_err = np.mean(self._task_errors[:20]) if len(self._task_errors) > 40 else recent_err

            if recent_err < old_err:
                # Melhorou: hiperparâmetros atuais são bons
                entry.lr_opt = self.brain.learner.lr
                entry.consolidation_opt = self.brain.learner.consolidation
                entry.surprise_gain_opt = self.brain.learner.surprise_gain
            else:
                # Piorou: ajusta lr para cima (precisa aprender mais rápido)
                entry.lr_opt = min(self.brain.learner.lr * 1.2, self.lr_base * 3.0)

        # Consolida assinatura
        entry.signature = self.signature

        # Reseta para próxima tarefa
        self.signature = TaskSignature(self.n_hidden)
        self._task_errors = []
        self._task_steps = 0
        self._task_start_step = self.brain.step

        return {
            "task_id": entry.task_id,
            "lr_opt": entry.lr_opt,
            "consolidation_opt": entry.consolidation_opt,
            "surprise_gain_opt": entry.surprise_gain_opt,
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.brain.set_mode("infer").forward(x)

    def set_mode(self, mode: str) -> "MetaContinualLearner":
        self.brain.set_mode(mode)
        return self

    def reset_state(self) -> None:
        self.brain.reset_state()

    def get_diagnostics(self) -> dict:
        """Retorna estado interno para diagnóstico."""
        return {
            "current_task_id": self.current_task_id,
            "n_known_tasks": len(self.task_bank),
            "task_bank": {
                tid: {
                    "lr_opt": e.lr_opt,
                    "consolidation_opt": e.consolidation_opt,
                    "surprise_gain_opt": e.surprise_gain_opt,
                    "mean_error": e.mean_error,
                    "n_obs": e.n_observations,
                }
                for tid, e in self.task_bank.items()
            },
            "brain_lr": self.brain.learner.lr,
            "brain_consolidation": self.brain.learner.consolidation,
            "brain_surprise_gain": self.brain.learner.surprise_gain,
            "signature_ready": self.signature.is_ready(),
        }


# ==============================================================
#  FUNÇÕES DE EXPERIMENTO
# ==============================================================

def generate_task(kind: str, n: int = 500, seed: int = 0, noise: float = 0.1):
    """Gera tarefa sintética de regressão.

    Tipos:
      - "sine": senoide de baixa frequência
      - "saw": dente de serra
      - "mixed": combinação de senoides
      - "high_freq": senoide de alta frequência
      - "amortecida": senoide com decaimento exponencial
      - "quadrada": onda quadrada
      - "noise": ruído gaussiano
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    if kind == "sine":
        signal = np.sin(2 * np.pi * 0.05 * t)
    elif kind == "saw":
        signal = 2.0 * (t % 50) / 50.0 - 1.0
    elif kind == "mixed":
        signal = np.sin(2 * np.pi * 0.03 * t) + 0.5 * np.cos(2 * np.pi * 0.07 * t)
    elif kind == "high_freq":
        signal = np.sin(2 * np.pi * 0.15 * t)
    elif kind == "amortecida":
        signal = np.sin(2 * np.pi * 0.05 * t) * np.exp(-0.003 * t)
    elif kind == "quadrada":
        signal = np.sign(np.sin(2 * np.pi * 0.04 * t))
    elif kind == "noise":
        signal = rng.normal(0, 0.5, n)
    else:
        signal = np.sin(2 * np.pi * 0.05 * t)

    u = rng.normal(0, noise, (n, 2))
    u[:, 0] += signal
    y = np.convolve(signal, np.ones(5) / 5, mode="same")[:, None]
    return u.astype(np.float64), y.astype(np.float64)


def _get_brain(learner):
    """Retorna o objeto VisaoBrain, seja de MetaContinualLearner ou direto."""
    if isinstance(learner, MetaContinualLearner):
        return learner.brain
    return learner


def run_continual_experiment(
    learner,
    tasks: list[tuple[np.ndarray, np.ndarray]],
    eval_window: int = 100,
) -> dict:
    """Executa experimento continual learning.

    Parameters
    ----------
    learner : MetaContinualLearner ou VisaoBrain
    tasks : lista de (X, Y) para cada tarefa
    eval_window : tamanho da janela de avaliação

    Returns
    -------
    dict com métricas por tarefa e esquecimento acumulado.
    """
    n_tasks = len(tasks)
    task_errors = {i: [] for i in range(n_tasks)}
    task_final_mse = {}
    brain = _get_brain(learner)

    for task_idx, (X, Y) in enumerate(tasks):
        learner.set_mode("learn")
        learner.reset_state()

        # Treina na tarefa
        for xi, yi in zip(X, Y):
            learner.learn(xi, yi)

        if isinstance(learner, MetaContinualLearner):
            learner.end_task()

        # Avalia em todas as tarefas vistas
        for eval_idx in range(task_idx + 1):
            X_eval, Y_eval = tasks[eval_idx]
            learner.set_mode("infer")
            learner.reset_state()
            eval_result = brain.evaluate_stream(
                X_eval[-eval_window:], Y_eval[-eval_window:]
            )
            task_errors[eval_idx].append(eval_result["mse"])
            if eval_idx == task_idx:
                task_final_mse[eval_idx] = eval_result["mse"]

    # Calcula esquecimento (backward transfer)
    forgetting = 0.0
    for i in range(n_tasks):
        if len(task_errors[i]) > 1:
            best = min(task_errors[i])
            final = task_errors[i][-1]
            forgetting += max(0, final - best)

    return {
        "task_errors": task_errors,
        "task_final_mse": task_final_mse,
        "forgetting": forgetting,
        "mean_final_mse": np.mean(list(task_final_mse.values())),
    }


def compare_baseline_vs_meta(
    tasks: list[tuple[np.ndarray, np.ndarray]],
    n_hidden: int = 32,
    seed: int = 42,
) -> dict:
    """Compara baseline (VisaoBrain sem meta-learning) vs MetaContinualLearner.

    Retorna métricas de comparação.
    """
    # Baseline: VisaoBrain com parâmetros padrão
    baseline = VisaoBrain(
        n_in=2, n_hidden=n_hidden, n_out=1,
        consolidation=8.0, surprise_gain=3.0,
        meta_learn=False, seed=seed,
    )

    # MetaContinualLearner
    meta = MetaContinualLearner(
        n_in=2, n_hidden=n_hidden, n_out=1,
        consolidation=8.0, surprise_gain=3.0,
        seed=seed,
    )

    # Executa experimentos
    result_baseline = run_continual_experiment(baseline, tasks)
    result_meta = run_continual_experiment(meta, tasks)

    improvement = (
        (result_baseline["forgetting"] - result_meta["forgetting"])
        / max(result_baseline["forgetting"], 1e-8)
    ) * 100

    return {
        "baseline": result_baseline,
        "meta": result_meta,
        "improvement_pct": improvement,
        "forgetting_reduction": result_baseline["forgetting"] - result_meta["forgetting"],
    }
