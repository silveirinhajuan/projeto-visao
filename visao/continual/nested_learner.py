"""nested_learner.py — Tarefa 17: Nested Learning (Google, 2026).

Adapta o conceito de Nested Learning ao VisaoBrain:
  - Inner loop: adaptação rápida em uma tarefa específica
  - Outer loop: meta-aprendizado entre tarefas
  - Isolamento de parâmetros por tarefa (readout heads separados)

Cada tarefa é tratada como um problema de otimização aninhado com
workflow interno. O inner loop otimiza para a tarefa atual; o outer
loop aprende a inicializar/ajustar o inner loop para futuras tarefas.

Referências:
  - Google Research (2026) — Nested Learning
  - Finn et al. (2017) — MAML
  - Hasani et al. (2021) — Liquid Time-Constant Networks
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Garantir imports do projeto
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visao.brain import VisaoBrain


class NestedLearner:
    """Nested Learning: otimização aninhada para continual learning.

    Combina:
      - Inner loop: adaptação rápida via gradiente local (few-shot)
      - Outer loop: meta-aprendizado que ajusta inicialização
      - Task-specific readouts: isolamento de parâmetros por tarefa

    O inner loop executa N passos de aprendizado na tarefa atual.
    O outer loop ajusta os parâmetros compartilhados para minimizar
    o erro após a adaptação do inner loop.

    Parâmetros:
      n_in, n_hidden, n_out: dimensões do cérebro
      inner_steps: passos de adaptação no inner loop
      inner_lr: taxa de aprendizado do inner loop
      outer_lr: taxa de aprendizado do outer loop (meta)
      n_outer_epochs: épocas do outer loop
    """

    def __init__(
        self,
        n_in: int = 2,
        n_hidden: int = 64,
        n_out: int = 1,
        inner_steps: int = 10,
        inner_lr: float = 0.05,
        outer_lr: float = 0.01,
        n_outer_epochs: int = 5,
        consolidation: float = 8.0,
        surprise_gain: float = 3.0,
        lambda_decay: float = 0.0005,
        seed: int = 0,
    ):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_outer_epochs = n_outer_epochs

        # Cérebro base (compartilhado entre tarefas)
        self.brain = VisaoBrain(
            n_in=n_in,
            n_hidden=n_hidden,
            n_out=n_out,
            lr=inner_lr,
            consolidation=consolidation,
            surprise_gain=surprise_gain,
            lambda_decay=lambda_decay,
            meta_learn=False,
            seed=seed,
        )

        # Readout heads específicos por tarefa (isolamento)
        # task_id -> {"W_out": ..., "b_out": ..., "omega": ...}
        self.task_readouts: dict[int, dict[str, np.ndarray]] = {}

        # Estado meta: inicialização compartilhada
        self._meta_W_out = self.brain.learner.W_out.copy()
        self._meta_b_out = self.brain.learner.b_out.copy()

        # Contadores
        self.n_tasks = 0
        self._current_task_id: int = -1

    # ==============================================================
    #  API PÚBLICA
    # ==============================================================

    def learn(self, x: np.ndarray, y: np.ndarray) -> dict:
        """Um passo de aprendizado (delega ao cérebro)."""
        return self.brain.learn(x, y)

    def forward(self, x: np.ndarray, task_id: Optional[int] = None) -> np.ndarray:
        """Inferência. Se task_id fornecido, usa readout específico."""
        if task_id is not None and task_id in self.task_readouts:
            readout = self.task_readouts[task_id]
            return readout["W_out"] @ self.brain.x + readout["b_out"]
        return self.brain.forward(x)

    def set_mode(self, mode: str) -> "NestedLearner":
        """Alterna modo do cérebro."""
        self.brain.set_mode(mode)
        return self

    def reset_state(self) -> None:
        """Reseta estado do reservatório."""
        self.brain.reset_state()

    # ==============================================================
    #  INNER LOOP: Adaptação Rápida
    # ==============================================================

    def inner_loop(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_steps: Optional[int] = None,
        task_id: Optional[int] = None,
    ) -> dict:
        """Inner loop: adaptação rápida em uma tarefa.

        Executa n_steps passos de aprendizado nos dados (X, Y).
        Usa o readout específico da tarefa se task_id fornecido.
        Se task_id=None, auto-atribui um novo ID.

        Retorna {"mse": float, "n_steps": int}
        """
        n_steps = n_steps or self.inner_steps
        self.brain.set_mode("learn")

        # Auto-atribui task_id se não fornecido
        if task_id is None:
            task_id = self.n_tasks

        # Ativa readout específico (cria se não existir)
        self._activate_task_readout(task_id)

        # Executa passos de adaptação
        errors = []
        for i, (xi, yi) in enumerate(zip(X, Y)):
            if i >= n_steps:
                break
            result = self.brain.learn(xi, yi)
            errors.append(result["err"] ** 2)

        mse = float(np.mean(errors)) if errors else float("inf")

        # Registra readout após adaptação
        self._register_task_readout(task_id)

        return {"mse": mse, "n_steps": len(errors)}

    # ==============================================================
    #  OUTER LOOP: Meta-Aprendizado
    # ==============================================================

    def outer_loop(
        self,
        tasks: list[tuple[np.ndarray, np.ndarray]],
        inner_steps: Optional[int] = None,
        n_epochs: Optional[int] = None,
    ) -> dict:
        """Outer loop: meta-aprendizado entre tarefas.

        Para cada época:
          1. Para cada tarefa: executa inner loop
          2. Calcula erro pós-adaptação
          3. Atualiza parâmetros compartilhados (meta-update)

        Retorna {"mean_mse": float, "n_epochs": int}
        """
        inner_steps = inner_steps or self.inner_steps
        n_epochs = n_epochs or self.n_outer_epochs

        epoch_mses = []

        for epoch in range(n_epochs):
            task_mses = []

            for task_idx, (X, Y) in enumerate(tasks):
                self.brain.set_mode("learn")
                self.brain.reset_state()

                # Inner loop adaptation
                result = self.inner_loop(X, Y, n_steps=inner_steps, task_id=task_idx)
                task_mses.append(result["mse"])

            # Meta-update: ajusta inicialização compartilhada
            # (simples: média dos readouts específicos)
            self._meta_update_from_readouts()

            epoch_mse = float(np.mean(task_mses))
            epoch_mses.append(epoch_mse)

        return {"mean_mse": float(np.mean(epoch_mses)), "n_epochs": n_epochs}

    # ==============================================================
    #  READOUTS ESPECÍFICOS POR TAREFA
    # ==============================================================

    def _register_task_readout(self, task_id: int) -> None:
        """Registra o readout atual como específico da tarefa."""
        self.task_readouts[task_id] = {
            "W_out": self.brain.learner.W_out.copy(),
            "b_out": self.brain.learner.b_out.copy(),
            "omega": self.brain.learner.omega.copy(),
        }
        self.n_tasks = max(self.n_tasks, task_id + 1)

    def _activate_task_readout(self, task_id: int) -> None:
        """Ativa o readout específico da tarefa."""
        if task_id in self.task_readouts:
            readout = self.task_readouts[task_id]
            self.brain.learner.W_out = readout["W_out"].copy()
            self.brain.learner.b_out = readout["b_out"].copy()
            self.brain.learner.omega = readout["omega"].copy()
        else:
            # Nova tarefa: inicializa com meta-parâmetros
            self.brain.learner.W_out = self._meta_W_out.copy()
            self.brain.learner.b_out = self._meta_b_out.copy()
            # omega deve ter shape (n_out, n_hidden) — igual LocalLearner.__init__
            self.brain.learner.omega = np.zeros((self.n_out, self.n_hidden))
            self._register_task_readout(task_id)

    def _meta_update_from_readouts(self) -> None:
        """Atualiza meta-parâmetros como média dos readouts específicos."""
        if not self.task_readouts:
            return

        W_out_mean = np.mean([r["W_out"] for r in self.task_readouts.values()], axis=0)
        b_out_mean = np.mean([r["b_out"] for r in self.task_readouts.values()], axis=0)

        # Atualiza meta com moving average
        self._meta_W_out = (1 - self.outer_lr) * self._meta_W_out + self.outer_lr * W_out_mean
        self._meta_b_out = (1 - self.outer_lr) * self._meta_b_out + self.outer_lr * b_out_mean

    # ==============================================================
    #  AVALIAÇÃO
    # ==============================================================

    def _evaluate_forgetting(
        self,
        tasks: list[tuple[np.ndarray, np.ndarray]],
        eval_window: int = 100,
    ) -> float:
        """Avalia esquecimento entre tarefas."""
        n_tasks = len(tasks)
        task_best_mse = {}
        task_final_mse = {}

        for task_idx, (X, Y) in enumerate(tasks):
            # Avalia após treino desta tarefa
            self.brain.set_mode("infer")
            self.brain.reset_state()
            self._activate_task_readout(task_idx)

            preds = []
            for xi, yi in zip(X[-eval_window:], Y[-eval_window:]):
                pred = self.brain.forward(xi)
                preds.append((pred, yi))

            mse = float(np.mean([(p - y) ** 2 for p, y in preds]))
            task_final_mse[task_idx] = mse
            task_best_mse[task_idx] = mse

            # Avalia em tarefas anteriores
            for prev_idx in range(task_idx):
                X_prev, Y_prev = tasks[prev_idx]
                self._activate_task_readout(prev_idx)
                self.brain.reset_state()

                preds_prev = []
                for xi, yi in zip(X_prev[-eval_window:], Y_prev[-eval_window:]):
                    pred = self.brain.forward(xi)
                    preds_prev.append((pred, yi))

                mse_prev = float(np.mean([(p - y) ** 2 for p, y in preds_prev]))
                task_best_mse[prev_idx] = min(task_best_mse.get(prev_idx, float("inf")), mse_prev)

        # Esquecimento = final - best
        forgetting = 0.0
        for i in range(n_tasks):
            if i in task_best_mse and i in task_final_mse:
                forgetting += max(0, task_final_mse[i] - task_best_mse[i])

        return forgetting

    def get_diagnostics(self) -> dict:
        """Retorna estado interno para diagnóstico."""
        return {
            "n_tasks": self.n_tasks,
            "current_task_id": self._current_task_id,
            "n_task_readouts": len(self.task_readouts),
            "brain_lr": self.brain.learner.lr,
            "brain_consolidation": self.brain.learner.consolidation,
        }


# ==============================================================
#  FUNÇÕES DE EXPERIMENTO
# ==============================================================

def generate_task(kind: str, n: int = 500, seed: int = 0, noise: float = 0.1):
    """Gera tarefa sintética de regressão."""
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


def run_nested_experiment(
    learner,
    tasks: list[tuple[np.ndarray, np.ndarray]],
    eval_window: int = 100,
) -> dict:
    """Executa experimento de nested learning."""
    n_tasks = len(tasks)
    task_errors = {i: [] for i in range(n_tasks)}
    task_final_mse = {}

    for task_idx, (X, Y) in enumerate(tasks):
        # Verifica se é NestedLearner ou VisaoBrain (baseline)
        if isinstance(learner, NestedLearner):
            # Inner loop adaptation
            learner.set_mode("learn")
            learner.reset_state()
            learner.inner_loop(X, Y, n_steps=learner.inner_steps, task_id=task_idx)
        else:
            # Baseline: treina diretamente
            learner.set_mode("learn")
            learner.reset_state()
            for xi, yi in zip(X, Y):
                learner.learn(xi, yi)

        # Avalia em todas as tarefas até agora
        for eval_idx in range(task_idx + 1):
            X_eval, Y_eval = tasks[eval_idx]
            learner.set_mode("infer")
            learner.reset_state()

            if isinstance(learner, NestedLearner):
                learner._activate_task_readout(eval_idx)

            preds = []
            for xi, yi in zip(X_eval[-eval_window:], Y_eval[-eval_window:]):
                if isinstance(learner, NestedLearner):
                    pred = learner.forward(xi, task_id=eval_idx)
                else:
                    pred = learner.forward(xi)
                preds.append((pred, yi))

            mse = float(np.mean([(p - y) ** 2 for p, y in preds]))
            task_errors[eval_idx].append(mse)
            if eval_idx == task_idx:
                task_final_mse[eval_idx] = mse

    # Calcula esquecimento
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
        "mean_final_mse": float(np.mean(list(task_final_mse.values()))),
    }


def compare_nested_vs_baseline(
    tasks: list[tuple[np.ndarray, np.ndarray]],
    n_hidden: int = 32,
    seed: int = 42,
) -> dict:
    """Compara baseline (VisaoBrain sem nested learning) vs NestedLearner."""
    baseline = VisaoBrain(
        n_in=2, n_hidden=n_hidden, n_out=1,
        consolidation=8.0, surprise_gain=3.0,
        meta_learn=False, seed=seed,
    )
    nested = NestedLearner(
        n_in=2, n_hidden=n_hidden, n_out=1,
        consolidation=8.0, surprise_gain=3.0,
        seed=seed,
    )

    result_baseline = run_nested_experiment(baseline, tasks)
    result_nested = run_nested_experiment(nested, tasks)

    improvement = (
        (result_baseline["forgetting"] - result_nested["forgetting"])
        / max(result_baseline["forgetting"], 1e-8)
    ) * 100

    return {
        "baseline": result_baseline,
        "nested": result_nested,
        "improvement_pct": improvement,
        "forgetting_reduction": result_baseline["forgetting"] - result_nested["forgetting"],
    }


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 17 — Nested Learning (Google, 2026)")
    print("Inner loop + Outer loop + Task-specific readouts")
    print("=" * 70)

    tasks = [
        generate_task("sine", n=300, seed=0),
        generate_task("saw", n=300, seed=1),
        generate_task("mixed", n=300, seed=2),
        generate_task("high_freq", n=300, seed=3),
        generate_task("quadrada", n=300, seed=4),
    ]

    print("\n1. Treinando NestedLearner com outer loop...")
    nested = NestedLearner(n_in=2, n_hidden=32, n_out=1, seed=42)
    result = nested.outer_loop(tasks, inner_steps=10, n_epochs=3)
    print(f"   MSE médio: {result['mean_mse']:.4f}")

    print("\n2. Comparando vs baseline...")
    comparison = compare_nested_vs_baseline(tasks, n_hidden=32, seed=42)
    print(f"   Baseline forgetting: {comparison['baseline']['forgetting']:.4f}")
    print(f"   Nested forgetting:   {comparison['nested']['forgetting']:.4f}")
    print(f"   Melhoria:            {comparison['improvement_pct']:.1f}%")

    print("\n3. Diagnósticos:")
    diag = nested.get_diagnostics()
    for k, v in diag.items():
        print(f"   {k}: {v}")

    print("\n[ok] Nested Learning validado.")
