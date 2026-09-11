"""Model definitions for research experiments.

Provides simple numpy-based models that implement a common interface:
  - fit(X, Y) -> train on data
  - predict(X) -> inference
  - reset_state() -> reset internal state
  - save(path) / load(path) -> persistence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class BaseModel:
    """Abstract base model interface."""

    def __init__(self, name: str = "model"):
        self.name = name
        self._is_trained = False

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> dict:
        """Train the model. Returns training metrics."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference."""
        raise NotImplementedError

    def reset_state(self) -> None:
        """Reset internal state (for recurrent models)."""
        pass

    def save(self, path: str | Path) -> None:
        """Save model weights to JSON."""
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        """Load model weights from JSON."""
        raise NotImplementedError

    @property
    def is_trained(self) -> bool:
        return self._is_trained


class MLPModel(BaseModel):
    """Simple 2-layer MLP with tanh activation."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 64,
        lr: float = 0.01,
        epochs: int = 50,
        seed: int = 0,
        task_type: str = "regression",
    ):
        super().__init__(name="MLP")
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs
        self.task_type = task_type
        self._rng = np.random.default_rng(seed)

        # Xavier init
        scale1 = np.sqrt(2.0 / (n_in + n_hidden))
        scale2 = np.sqrt(2.0 / (n_hidden + n_out))
        self.W1 = self._rng.standard_normal((n_in, n_hidden)) * scale1
        self.b1 = np.zeros(n_hidden)
        self.W2 = self._rng.standard_normal((n_hidden, n_out)) * scale2
        self.b2 = np.zeros(n_out)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(X @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return h, out

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> dict:
        n = len(X)
        losses = []
        
        # Adam optimizer parameters
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        
        for epoch in range(self.epochs):
            # Batch gradient descent (much faster for classification)
            if self.task_type == "classification":
                # Forward: output logits (no sigmoid) for stable BCE gradient
                h = np.tanh(X @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                pred = self._sigmoid(logits)
                
                # BCE gradient w.r.t. logits: (pred - target)
                err = pred - Y
                dW2 = h.T @ err / n
                db2 = err.sum(axis=0) / n
                dh = err @ self.W2.T
                dh *= (1 - h ** 2)
                dW1 = X.T @ dh / n
                db1 = dh.sum(axis=0) / n

                # Adam update
                t = epoch + 1
                for param, grad, m, v in [
                    (self.W1, dW1, mW1, vW1),
                    (self.W2, dW2, mW2, vW2),
                    (self.b1, db1, mb1, vb1),
                    (self.b2, db2, mb2, vb2),
                ]:
                    m[:] = beta1 * m + (1 - beta1) * grad
                    v[:] = beta2 * v + (1 - beta2) * grad ** 2
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)
                    param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
                
                losses.append(float((err ** 2).mean()))
            else:
                # Mini-batch SGD for regression
                perm = self._rng.permutation(n)
                epoch_loss = 0.0
                for i in perm:
                    xi, yi = X[i:i+1], Y[i:i+1]
                    h, pred = self._forward(xi)
                    err = pred - yi

                    dW2 = h.T @ err
                    db2 = err.sum(axis=0)
                    dh = err @ self.W2.T
                    dh *= (1 - h ** 2)
                    dW1 = xi.T @ dh
                    db1 = dh.sum(axis=0)

                    self.W2 -= self.lr * dW2
                    self.b2 -= self.lr * db2
                    self.W1 -= self.lr * dW1
                    self.b1 -= self.lr * db1
                    epoch_loss += float((err ** 2).mean())

                losses.append(epoch_loss / n)

        self._is_trained = True
        return {"losses": losses, "final_loss": losses[-1]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, out = self._forward(X)
        if self.task_type == "classification":
            out = self._sigmoid(out)
        return out

    def save(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "n_in": self.n_in, "n_out": self.n_out, "n_hidden": self.n_hidden,
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            data = json.load(f)
        self.W1 = np.array(data["W1"])
        self.b1 = np.array(data["b1"])
        self.W2 = np.array(data["W2"])
        self.b2 = np.array(data["b2"])
        self._is_trained = True


class GRUModel(BaseModel):
    """Simple GRU recurrent model."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 64,
        lr: float = 0.01,
        epochs: int = 50,
        seed: int = 0,
        task_type: str = "regression",
    ):
        super().__init__(name="GRU")
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs
        self.task_type = task_type
        self._rng = np.random.default_rng(seed)

        # GRU weights
        scale = np.sqrt(1.0 / n_hidden)
        self.Wz = self._rng.standard_normal((n_in, n_hidden)) * scale
        self.Uz = self._rng.standard_normal((n_hidden, n_hidden)) * scale
        self.Wr = self._rng.standard_normal((n_in, n_hidden)) * scale
        self.Ur = self._rng.standard_normal((n_hidden, n_hidden)) * scale
        self.Wh = self._rng.standard_normal((n_in, n_hidden)) * scale
        self.Uh = self._rng.standard_normal((n_hidden, n_hidden)) * scale
        self.W_out = self._rng.standard_normal((n_hidden, n_out)) * scale
        self.b_out = np.zeros(n_out)

        self.h = np.zeros(n_hidden)

    def reset_state(self) -> None:
        self.h = np.zeros(self.n_hidden)

    def _gru_step(self, x: np.ndarray) -> np.ndarray:
        # x is 1D array of shape (n_in,)
        x = x.ravel()
        self.h = self.h.ravel()
        z = 1.0 / (1.0 + np.exp(-(x @ self.Wz + self.h @ self.Uz)))  # sigmoid
        r = 1.0 / (1.0 + np.exp(-(x @ self.Wr + self.h @ self.Ur)))
        h_tilde = np.tanh(x @ self.Wh + (r * self.h) @ self.Uh)
        self.h = (1 - z) * self.h + z * h_tilde
        return self.h

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> dict:
        n = len(X)
        losses = []
        for epoch in range(self.epochs):
            self.reset_state()
            epoch_loss = 0.0
            for i in range(n):
                self._gru_step(X[i])
                pred = self.h @ self.W_out + self.b_out
                err = pred - Y[i]
                # Simple gradient update on readout
                self.W_out -= self.lr * np.outer(self.h, err)
                self.b_out -= self.lr * err.ravel()
                epoch_loss += float((err ** 2).mean())
            losses.append(epoch_loss / n)

        self._is_trained = True
        return {"losses": losses, "final_loss": losses[-1]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.reset_state()
        preds = []
        for i in range(len(X)):
            self._gru_step(X[i])
            out = self.h @ self.W_out + self.b_out
            if self.task_type == "classification":
                out = 1.0 / (1.0 + np.exp(-np.clip(out, -500, 500)))
            preds.append(out.copy())
        return np.array(preds)

    def save(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "Wz": self.Wz.tolist(), "Uz": self.Uz.tolist(),
            "Wr": self.Wr.tolist(), "Ur": self.Ur.tolist(),
            "Wh": self.Wh.tolist(), "Uh": self.Uh.tolist(),
            "W_out": self.W_out.tolist(), "b_out": self.b_out.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            data = json.load(f)
        self.Wz = np.array(data["Wz"])
        self.Uz = np.array(data["Uz"])
        self.Wr = np.array(data["Wr"])
        self.Ur = np.array(data["Ur"])
        self.Wh = np.array(data["Wh"])
        self.Uh = np.array(data["Uh"])
        self.W_out = np.array(data["W_out"])
        self.b_out = np.array(data["b_out"])
        self._is_trained = True


class LSTMModel(BaseModel):
    """Simple LSTM recurrent model."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 64,
        lr: float = 0.01,
        epochs: int = 50,
        seed: int = 0,
        task_type: str = "regression",
    ):
        super().__init__(name="LSTM")
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs
        self.task_type = task_type
        self._rng = np.random.default_rng(seed)

        scale = np.sqrt(1.0 / n_hidden)
        # Input, forget, output gates + candidate
        self.Wi = self._rng.standard_normal((n_in, n_hidden)) * scale
        self.Ui = self._rng.standard_normal((n_hidden, n_hidden)) * scale
        self.Wf = self._rng.standard_normal((n_in, n_hidden)) * scale
        self.Uf = self._rng.standard_normal((n_hidden, n_hidden)) * scale
        self.Wo = self._rng.standard_normal((n_in, n_hidden)) * scale
        self.Uo = self._rng.standard_normal((n_hidden, n_hidden)) * scale
        self.Wc = self._rng.standard_normal((n_in, n_hidden)) * scale
        self.Uc = self._rng.standard_normal((n_hidden, n_hidden)) * scale
        self.W_out = self._rng.standard_normal((n_hidden, n_out)) * scale
        self.b_out = np.zeros(n_out)

        self.h = np.zeros(n_hidden)
        self.c = np.zeros(n_hidden)

    def reset_state(self) -> None:
        self.h = np.zeros(self.n_hidden)
        self.c = np.zeros(self.n_hidden)

    def _lstm_step(self, x: np.ndarray) -> np.ndarray:
        x = x.ravel()
        self.h = self.h.ravel()
        self.c = self.c.ravel()
        i = 1.0 / (1.0 + np.exp(-(x @ self.Wi + self.h @ self.Ui)))
        f = 1.0 / (1.0 + np.exp(-(x @ self.Wf + self.h @ self.Uf)))
        o = 1.0 / (1.0 + np.exp(-(x @ self.Wo + self.h @ self.Uo)))
        c_tilde = np.tanh(x @ self.Wc + self.h @ self.Uc)
        self.c = f * self.c + i * c_tilde
        self.h = o * np.tanh(self.c)
        return self.h

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> dict:
        n = len(X)
        losses = []
        for epoch in range(self.epochs):
            self.reset_state()
            epoch_loss = 0.0
            for i in range(n):
                self._lstm_step(X[i])
                pred = self.h @ self.W_out + self.b_out
                err = pred - Y[i]
                self.W_out -= self.lr * np.outer(self.h, err)
                self.b_out -= self.lr * err.ravel()
                epoch_loss += float((err ** 2).mean())
            losses.append(epoch_loss / n)

        self._is_trained = True
        return {"losses": losses, "final_loss": losses[-1]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.reset_state()
        preds = []
        for i in range(len(X)):
            self._lstm_step(X[i])
            out = self.h @ self.W_out + self.b_out
            if self.task_type == "classification":
                out = 1.0 / (1.0 + np.exp(-np.clip(out, -500, 500)))
            preds.append(out.copy())
        return np.array(preds)

    def save(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "Wi": self.Wi.tolist(), "Ui": self.Ui.tolist(),
            "Wf": self.Wf.tolist(), "Uf": self.Uf.tolist(),
            "Wo": self.Wo.tolist(), "Uo": self.Uo.tolist(),
            "Wc": self.Wc.tolist(), "Uc": self.Uc.tolist(),
            "W_out": self.W_out.tolist(), "b_out": self.b_out.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            data = json.load(f)
        self.Wi = np.array(data["Wi"]); self.Ui = np.array(data["Ui"])
        self.Wf = np.array(data["Wf"]); self.Uf = np.array(data["Uf"])
        self.Wo = np.array(data["Wo"]); self.Uo = np.array(data["Uo"])
        self.Wc = np.array(data["Wc"]); self.Uc = np.array(data["Uc"])
        self.W_out = np.array(data["W_out"])
        self.b_out = np.array(data["b_out"])
        self._is_trained = True


class CfCModel(BaseModel):
    """Closed-form Continuous-time model (simplified CfC)."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 64,
        lr: float = 0.01,
        epochs: int = 50,
        tau: float = 1.0,
        seed: int = 0,
        task_type: str = "regression",
    ):
        super().__init__(name="CfC")
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs
        self.tau = tau
        self.task_type = task_type
        self._rng = np.random.default_rng(seed)

        scale = np.sqrt(1.0 / n_hidden)
        self.W_in = self._rng.standard_normal((n_in, n_hidden)) * scale
        self.W_rec = self._rng.standard_normal((n_hidden, n_hidden)) * scale * 0.1
        self.b = np.zeros(n_hidden)
        self.W_out = self._rng.standard_normal((n_hidden, n_out)) * scale
        self.b_out = np.zeros(n_out)

        self.x_state = np.zeros(n_hidden)

    def reset_state(self) -> None:
        self.x_state = np.zeros(self.n_hidden)

    def _cfc_step(self, u: np.ndarray) -> np.ndarray:
        u = u.ravel()
        self.x_state = self.x_state.ravel()
        # dx/dt = -x/tau + f(W_in @ u + W_rec @ x + b)
        dt = 0.1
        act = np.tanh(u @ self.W_in + self.x_state @ self.W_rec + self.b)
        dx = (-self.x_state / self.tau + act) * dt
        self.x_state = self.x_state + dx
        return self.x_state

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> dict:
        n = len(X)
        losses = []
        for epoch in range(self.epochs):
            self.reset_state()
            epoch_loss = 0.0
            for i in range(n):
                self._cfc_step(X[i])
                pred = self.x_state @ self.W_out + self.b_out
                err = pred - Y[i]
                self.W_out -= self.lr * np.outer(self.x_state, err)
                self.b_out -= self.lr * err.ravel()
                epoch_loss += float((err ** 2).mean())
            losses.append(epoch_loss / n)

        self._is_trained = True
        return {"losses": losses, "final_loss": losses[-1]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.reset_state()
        preds = []
        for i in range(len(X)):
            self._cfc_step(X[i])
            out = self.x_state @ self.W_out + self.b_out
            if self.task_type == "classification":
                out = 1.0 / (1.0 + np.exp(-np.clip(out, -500, 500)))
            preds.append(out.copy())
        return np.array(preds)

    def save(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "W_in": self.W_in.tolist(), "W_rec": self.W_rec.tolist(),
            "b": self.b.tolist(),
            "W_out": self.W_out.tolist(), "b_out": self.b_out.tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str | Path) -> None:
        with open(path) as f:
            data = json.load(f)
        self.W_in = np.array(data["W_in"])
        self.W_rec = np.array(data["W_rec"])
        self.b = np.array(data["b"])
        self.W_out = np.array(data["W_out"])
        self.b_out = np.array(data["b_out"])
        self._is_trained = True


class VisaoModel(BaseModel):
    """Wrapper around VisaoBrain for the research harness."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 64,
        lr: float = 0.02,
        epochs: int = 1,  # VisaoBrain learns online
        seed: int = 0,
        task_type: str = "regression",
        **kwargs,
    ):
        super().__init__(name="VISÃO")
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.task_type = task_type
        self.extra_kwargs = kwargs

        # Import here to avoid hard dependency
        try:
            from visao.brain import VisaoBrain
            self.model = VisaoBrain(
                n_in=n_in,
                n_hidden=n_hidden,
                n_out=n_out,
                lr=lr,
                seed=seed,
                task_type=task_type,
                **kwargs,
            )
        except ImportError:
            # Fallback: use CfC as a stand-in
            self.model = CfCModel(n_in, n_out, n_hidden, lr, 50, seed=seed, task_type=task_type)
            self.name = "VISÃO(fallback)"

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> dict:
        n = len(X)
        losses = []
        for epoch in range(self.epochs):
            self.model.reset_state()
            epoch_loss = 0.0
            for i in range(n):
                result = self.model.learn(X[i:i+1], Y[i:i+1])
                epoch_loss += result["err"] ** 2
            losses.append(epoch_loss / n)

        self._is_trained = True
        return {"losses": losses, "final_loss": losses[-1]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.set_mode("infer")
        self.model.reset_state()
        preds = []
        for i in range(len(X)):
            pred = self.model.forward(X[i:i+1])
            preds.append(pred)
        self.model.set_mode("learn")
        return np.array(preds)

    def reset_state(self) -> None:
        self.model.reset_state()

    def save(self, path: str | Path) -> None:
        self.model.save(path)

    def load(self, path: str | Path) -> None:
        self.model.load(path)
        self._is_trained = True


def create_model(name: str, n_in: int, n_out: int, **kwargs) -> BaseModel:
    """Factory function to create models by name.
    
    Supports ablation variants like "VISÃO-full", "VISÃO-no_meta", etc.
    by recognizing the VISÃO prefix.
    """
    name_stripped = name.strip()
    
    # Direct matches
    models = {
        "MLP": MLPModel,
        "GRU": GRUModel,
        "LSTM": LSTMModel,
        "CfC": CfCModel,
        "VISÃO": VisaoModel,
        "Visao": VisaoModel,
        "visao": VisaoModel,
    }
    
    if name_stripped in models:
        return models[name_stripped](n_in=n_in, n_out=n_out, **kwargs)
    
    # Ablation variants: "VISÃO-full", "VISÃO-no_meta", etc.
    if name_stripped.startswith(("VISÃO-", "Visao-", "visao-")):
        return VisaoModel(n_in=n_in, n_out=n_out, **kwargs)
    
    raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
