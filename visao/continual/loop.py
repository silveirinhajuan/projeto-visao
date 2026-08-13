"""loop.py — Loop contínuo single-node do Projeto VISÃO (Tarefa 5.8).

Stream A->B->C->... no notebook: aplica regras de aprendizado LOCAIS (sem
backprop global, sem replay de dados antigos) e mede o esquecimento de tarefas
passadas após cada nova tarefa.

Reusa os mecanismos validados na Fase 0 / tarefa 1.9 (prototype/):
  - LiquidCell  : reservatório líquido LTC/CfC em numpy.
  - LocalLearner: readout local + consolidação por importância + gate de
                  surpresa + Oja no recorrente (o pacote que carrega os 97,2%).

Este módulo é o coração do VISÃO revisitado: o organismo que aprende de forma
incremental sem esquecer o que já viu.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# O prototype/ vive na raiz do projeto; garanta que ele esteja importável
# independentemente de como este módulo foi importado.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from prototype.liquid import LiquidCell  # noqa: E402
from prototype.plasticity import LocalLearner  # noqa: E402


def generate_synthetic_task(name: str, n_steps: int = 1000,
                            rng: np.random.Generator | None = None) -> dict:
    """Gera uma tarefa de regressão sintética (média móvel do canal 0).

    Tarefas diferentes (nomes diferentes) recebem frequência/fase distintas,
    criando distribuições distintas — o que torna o esquecimento mensurável.

    Retorna dict com u_tr, y_tr, u_te, y_te (arrays numpy).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    t = np.arange(n_steps)

    h = abs(hash(str(name))) % 1000
    phase = (h / 1000.0) * (2.0 * np.pi)
    freq = 0.05 + (h % 7) * 0.04

    def _make(gen: np.random.Generator, ph_offset: float) -> tuple[np.ndarray, np.ndarray]:
        u = gen.normal(0, 1, (n_steps, 2)).astype(np.float64) * 0.3
        u[:, 1] += np.sin(t * freq + phase + ph_offset)
        y = np.convolve(u[:, 0], np.ones(30) / 30.0, mode="same")[:, None]
        return u, y

    u_tr, y_tr = _make(rng, 0.0)
    rng2 = np.random.default_rng(h + 1)
    u_te, y_te = _make(rng2, 0.5)
    return {"u_tr": u_tr, "y_tr": y_tr, "u_te": u_te, "y_te": y_te}


class ContinualLoop:
    """Loop de aprendizado contínuo single-node com plasticidade local."""

    def __init__(self, n_in: int = 2, n_hidden: int = 64, seed: int = 0,
                 dt: float = 0.1, sparsity: float = 0.6,
                 tau_min: float = 0.4, tau_max: float = 4.0,
                 lr: float = 0.02, oja_lr: float = 0.0015,
                 consolidation: float = 8.0, surprise_gain: float = 4.0,
                 warmup: int = 50):
        if n_in <= 0:
            raise ValueError(f"n_in deve ser > 0, obtido {n_in}")
        if n_hidden <= 0:
            raise ValueError(f"n_hidden deve ser > 0, obtido {n_hidden}")

        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.seed = int(seed)
        self.dt = float(dt)
        self.sparsity = float(sparsity)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.lr = float(lr)
        self.oja_lr = float(oja_lr)
        self.consolidation = float(consolidation)
        self.surprise_gain = float(surprise_gain)
        self.warmup = int(warmup)

        rng = np.random.default_rng(seed)
        self.cell = LiquidCell(
            n_in=self.n_in, n_hidden=self.n_hidden, sparsity=self.sparsity,
            tau_min=self.tau_min, tau_max=self.tau_max, dt=self.dt, rng=rng,
        )
        # Learner criado sob demanda (n_out só se conhece no 1º add).
        self.learner: LocalLearner | None = None
        self.history: dict[str, dict] = {}
        self.order: list[str] = []

    # ----------------------------------------------------------- treino
    def add_and_train_task(self, name: str, u_tr, y_tr, u_te, y_te) -> dict:
        """Treina uma tarefa e registra erro inicial/final e esquecimento.

        Retorna dict com 'eval_error' (MSE do teste pós-treino) e
        'initial_error' (MSE do teste pré-treino).
        """
        u_tr = np.asarray(u_tr, dtype=np.float64)
        y_tr = np.asarray(y_tr, dtype=np.float64)
        u_te = np.asarray(u_te, dtype=np.float64)
        y_te = np.asarray(y_te, dtype=np.float64)
        n_out = y_tr.shape[1] if y_tr.ndim > 1 else 1

        if self.learner is None:
            self.learner = LocalLearner(
                self.n_hidden, n_out,
                lr=self.lr, oja_lr=self.oja_lr,
                consolidation=self.consolidation,
                surprise_gain=self.surprise_gain,
                rng=np.random.default_rng(self.seed + len(self.order) + 1),
            )

        initial_error = self._evaluate(u_te, y_te)

        x = np.zeros(self.n_hidden)
        x_prev = np.zeros(self.n_hidden)
        for i in range(len(u_tr)):
            x_prev = x
            x, _ = self.cell.step(x, u_tr[i])
            if i >= self.warmup:
                self.learner.update(x, y_tr[i])
                self.learner.oja_update(self.cell, x_prev, x)

        eval_error = self._evaluate(u_te, y_te)

        self.order.append(name)
        self.history[name] = {
            "initial_error": float(initial_error),
            "eval_error": float(eval_error),
            "u_te": u_te.tolist(),
            "y_te": y_te.tolist(),
        }
        return {"eval_error": float(eval_error), "initial_error": float(initial_error)}

    # ---------------------------------------------------------- avaliação
    def _evaluate(self, u, y) -> float:
        states, _ = self.cell.rollout(u)
        preds = states @ self.learner.W_out.T + self.learner.b_out
        warm = min(self.warmup, len(u) - 1)
        err = float(((preds[warm:] - y[warm:]) ** 2).mean())
        if not np.isfinite(err):
            err = 1e6
        return err

    # -------------------------------------------------------- esquecimento
    def get_forgetting(self) -> dict:
        """Esquecimento de cada tarefa ANTES da mais recente.

        forgetting[t] = erro_atual(t) - erro_inicial(t). A tarefa mais recente
        não é medida (ela mesma é o 'agora').
        """
        fg: dict[str, float] = {}
        for name in self.order[:-1]:
            rec = self.history[name]
            u = np.asarray(rec["u_te"], dtype=np.float64)
            y = np.asarray(rec["y_te"], dtype=np.float64)
            cur = self._evaluate(u, y)
            fg[name] = float(cur - rec["initial_error"])
        return fg

    def get_mean_forgetting(self) -> float:
        fg = self.get_forgetting()
        if not fg:
            return 0.0
        return float(np.mean(list(fg.values())))

    def check_gate_status(self, threshold_pct: float = 0.05,
                          min_tasks: int = 3) -> dict:
        """Portão de não-esquecimento: média de esquecimento < threshold?"""
        fg = self.get_forgetting()
        mean_fg = self.get_mean_forgetting()
        passed = (len(self.order) >= min_tasks) and (mean_fg < threshold_pct)
        return {
            "passed": bool(passed),
            "task_count": len(self.order),
            "mean_forgetting": float(mean_fg),
            "threshold_pct": float(threshold_pct),
            "forgetting": {k: float(v) for k, v in fg.items()},
        }

    # ---------------------------------------------------------- persistência
    def save_state(self, path) -> None:
        path = str(path)
        cell = self.cell
        learner = self.learner
        data = {
            "n_in": self.n_in, "n_hidden": self.n_hidden, "seed": self.seed,
            "dt": self.dt, "sparsity": self.sparsity,
            "tau_min": self.tau_min, "tau_max": self.tau_max,
            "lr": self.lr, "oja_lr": self.oja_lr,
            "consolidation": self.consolidation, "surprise_gain": self.surprise_gain,
            "warmup": self.warmup,
            "cell": {
                "W_in": cell.W_in.tolist(), "W_rec": cell.W_rec.tolist(),
                "b": cell.b.tolist(), "A": cell.A.tolist(),
                "mask": cell.mask.tolist(), "tau": cell.tau.tolist(),
            },
            "learner": {
                "n_out": learner.W_out.shape[0],
                "W_out": learner.W_out.tolist(), "b_out": learner.b_out.tolist(),
                "omega": learner.omega.tolist(), "lr": learner.lr,
                "oja_lr": learner.oja_lr, "consolidation": learner.consolidation,
                "surprise_gain": learner.surprise_gain,
                "err_ema": learner.err_ema, "err_var": learner.err_var,
            },
            "order": self.order,
            "history": self.history,
        }
        with open(path, "w") as fh:
            import json
            json.dump(data, fh)

    @classmethod
    def load_state(cls, path) -> "ContinualLoop":
        import json
        with open(str(path)) as fh:
            data = json.load(fh)

        loop = cls(
            n_in=data["n_in"], n_hidden=data["n_hidden"], seed=data["seed"],
            dt=data["dt"], sparsity=data["sparsity"],
            tau_min=data["tau_min"], tau_max=data["tau_max"],
            lr=data["lr"], oja_lr=data["oja_lr"],
            consolidation=data["consolidation"], surprise_gain=data["surprise_gain"],
            warmup=data["warmup"],
        )
        cd = data["cell"]
        loop.cell.W_in = np.array(cd["W_in"])
        loop.cell.W_rec = np.array(cd["W_rec"])
        loop.cell.b = np.array(cd["b"])
        loop.cell.A = np.array(cd["A"])
        loop.cell.mask = np.array(cd["mask"])
        loop.cell.tau = np.array(cd["tau"])
        loop.cell.W_rec *= loop.cell.mask  # topologia é invariante

        ld = data["learner"]
        loop.learner = LocalLearner(
            data["n_hidden"], ld["n_out"],
            lr=ld["lr"], oja_lr=ld["oja_lr"],
            consolidation=ld["consolidation"], surprise_gain=ld["surprise_gain"],
            rng=np.random.default_rng(0),
        )
        loop.learner.W_out = np.array(ld["W_out"])
        loop.learner.b_out = np.array(ld["b_out"])
        loop.learner.omega = np.array(ld["omega"])
        loop.learner.err_ema = ld["err_ema"]
        loop.learner.err_var = ld["err_var"]

        loop.order = list(data["order"])
        loop.history = {k: dict(v) for k, v in data["history"].items()}
        return loop
