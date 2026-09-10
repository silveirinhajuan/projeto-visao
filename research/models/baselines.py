"""Baselines for the VISÃO benchmark matrix.

Simple-but-correct numpy implementations of MLP, GRU, LSTM, and CfC
(reservoir-only, no plasticity). All models share a common interface:

    model.reset_state()
    pred = model.predict(x)    # x is a single timestep vector
    model.learn(x, y)          # online update with target y

Parameter budgets are kept comparable (n_hidden=64 by default).
"""
from __future__ import annotations

import numpy as np

# Reuse VISÃO's LiquidCell for the CfC baseline — same reservoir, no plasticity.
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prototype.liquid import LiquidCell


# ═══════════════════════════════════════════════════════════════════════════════
#  MLP — simple feedforward baseline (no recurrence)
# ═══════════════════════════════════════════════════════════════════════════════
class MLP:
    """2-layer feedforward net with tanh hidden. Processes one timestep at a time.

    Since it has no memory, this is the "worst case for temporal tasks" baseline
    that shows why recurrence matters.
    """

    def __init__(self, n_in: int = 1, n_hidden: int = 64, n_out: int = 1,
                 lr: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.lr = lr

        # Xavier init
        self.W1 = rng.normal(0, np.sqrt(1.0 / n_in), (n_hidden, n_in))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(1.0 / n_hidden), (n_out, n_hidden))
        self.b2 = np.zeros(n_out)

        # Adam moments
        self.m = {k: np.zeros_like(v) for k, v in
                  [("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2)]}
        self.v = {k: np.zeros_like(v) for k, v in
                  [("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2)]}
        self.t = 0

    def _adam_update(self, grads: dict):
        """Apply Adam update. grads: {param_name: gradient_array}."""
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for key, g in grads.items():
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * g ** 2
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)
            param = getattr(self, key)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        h = np.tanh(self.W1 @ x + self.b1)  # (n_hidden,)
        return self.W2 @ h + self.b2  # (n_out,)

    def learn(self, x: np.ndarray, y: np.ndarray) -> float:
        """One SGD step with Adam. Returns MSE."""
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        # Forward (manual, for gradient)
        z1 = self.W1 @ x + self.b1
        h = np.tanh(z1)
        pred = self.W2 @ h + self.b2
        err = pred - y
        mse = float((err ** 2).mean())

        # Backward
        dW2 = np.outer(err, h)
        db2 = err.copy()
        dh = self.W2.T @ err
        dz1 = dh * (1 - h ** 2)
        dW1 = np.outer(dz1, x)
        db1 = dz1.copy()

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        self._adam_update(grads)

        return mse

    def reset_state(self):
        """Stateless — nothing to reset."""
        pass

    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


# ═══════════════════════════════════════════════════════════════════════════════
#  GRU — Gated Recurrent Unit (standard formulation)
# ═══════════════════════════════════════════════════════════════════════════════
class GRU:
    """Minimal GRU with Adam. Single layer, single output."""

    def __init__(self, n_in: int = 1, n_hidden: int = 64, n_out: int = 1,
                 lr: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_hidden = n_hidden
        self.lr = lr

        # Gate weights: z (update), r (reset)
        scale = 1.0 / np.sqrt(n_hidden)
        self.Wz = rng.normal(0, scale, (n_hidden, n_in))
        self.Uz = rng.normal(0, scale, (n_hidden, n_hidden))
        self.bz = np.zeros(n_hidden)
        self.Wr = rng.normal(0, scale, (n_hidden, n_in))
        self.Ur = rng.normal(0, scale, (n_hidden, n_hidden))
        self.br = np.zeros(n_hidden)
        self.Wh = rng.normal(0, scale, (n_hidden, n_in))
        self.Uh = rng.normal(0, scale, (n_hidden, n_hidden))
        self.bh = np.zeros(n_hidden)

        # Readout
        self.W_out = rng.normal(0, scale, (n_out, n_hidden))
        self.b_out = np.zeros(n_out)

        self.h = np.zeros(n_hidden)
        self.x_prev = np.zeros(n_in)

        # Adam moments
        self._init_adam()

    def _init_adam(self):
        self.m = {}
        self.v = {}
        self.t = 0
        for name in ["Wz", "Uz", "bz", "Wr", "Ur", "br", "Wh", "Uh", "bh", "W_out", "b_out"]:
            p = getattr(self, name)
            self.m[name] = np.zeros_like(p)
            self.v[name] = np.zeros_like(p)

    def _adam_step(self, grads: dict):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for key, g in grads.items():
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * g ** 2
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)
            param = getattr(self, key)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward one step, update hidden, return output. Does NOT learn."""
        x = np.asarray(x, dtype=np.float64).ravel()
        z = sigmoid(self.Wz @ x + self.Uz @ self.h + self.bz)
        r = sigmoid(self.Wr @ x + self.Ur @ self.h + self.br)
        h_tanh = np.tanh(self.Wh @ x + self.Uh @ (r * self.h) + self.bh)
        self.h = (1 - z) * self.h + z * h_tanh
        self.x_prev = x.copy()
        return self.W_out @ self.h + self.b_out

    def learn(self, x: np.ndarray, y: np.ndarray) -> float:
        """BPTT truncated to 1 step. Returns MSE."""
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        # Forward
        h_prev = self.h.copy()
        z = sigmoid(self.Wz @ x + self.Uz @ h_prev + self.bz)
        r = sigmoid(self.Wr @ x + self.Ur @ h_prev + self.br)
        h_tilde_in = self.Wh @ x + self.Uh @ (r * h_prev) + self.bh
        h_tilde = np.tanh(h_tilde_in)
        h = (1 - z) * h_prev + z * h_tilde
        pred = self.W_out @ h + self.b_out
        err = pred - y
        mse = float((err ** 2).mean())

        # Backward (truncated BPTT, 1 step)
        # dL/dW_out
        dW_out = np.outer(err, h)
        db_out = err.copy()
        dh = self.W_out.T @ err  # (n_hidden,)

        # Through h = (1-z)*h_prev + z*h_tilde
        dz = dh * (h_tilde - h_prev)  # (n_hidden,)
        dh_prev = dh * (1 - z)
        dh_tilde = dh * z

        # Through tanh
        dht_in = dh_tilde * (1 - h_tilde ** 2)

        # Through reset gate
        dr = (self.Uh.T @ dht_in) * h_prev
        dh_prev += (self.Uh.T @ dht_in) * r

        # Update gate gradients
        dz_s = dz * sigmoid_prime(self.Wz @ x + self.Uz @ h_prev + self.bz)
        dWz = np.outer(dz_s, x)
        dUz = np.outer(dz_s, h_prev)
        dbz = dz_s

        # Reset gate gradients
        dr_s = dr * sigmoid_prime(self.Wr @ x + self.Ur @ h_prev + self.br)
        dWr = np.outer(dr_s, x)
        dUr = np.outer(dr_s, h_prev)
        dbr = dr_s

        # Candidate hidden gradients
        dWh = np.outer(dht_in, x)
        dUh = np.outer(dht_in, r * h_prev)
        dbh = dht_in

        # Contribution from reset to dh_prev via r*h_prev
        dh_prev += (self.Ur.T @ dr_s) * r  # only the part through reset

        # Through update gate to dh_prev via Uz
        dh_prev += (self.Uz.T @ dz_s)

        # Candidate hidden via Uh
        dh_prev += (self.Uh.T @ dht_in) * r

        grads = {
            "Wz": dWz, "Uz": dUz, "bz": dbz,
            "Wr": dWr, "Ur": dUr, "br": dbr,
            "Wh": dWh, "Uh": dUh, "bh": dbh,
            "W_out": dW_out, "b_out": db_out,
        }

        self._adam_step(grads)
        self.h = h
        self.x_prev = x.copy()
        return mse

    def reset_state(self):
        self.h = np.zeros(self.n_hidden)
        self.x_prev = np.zeros(self.Wz.shape[1])

    def n_params(self) -> int:
        return (
            self.Wz.size + self.Uz.size + self.bz.size +
            self.Wr.size + self.Ur.size + self.br.size +
            self.Wh.size + self.Uh.size + self.bh.size +
            self.W_out.size + self.b_out.size
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  LSTM — Long Short-Term Memory
# ═══════════════════════════════════════════════════════════════════════════════
class LSTM:
    """Minimal LSTM with Adam."""

    def __init__(self, n_in: int = 1, n_hidden: int = 64, n_out: int = 1,
                 lr: float = 0.01, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_hidden = n_hidden
        self.lr = lr
        scale = 1.0 / np.sqrt(n_hidden)

        # Input gate
        self.Wi = rng.normal(0, scale, (n_hidden, n_in))
        self.Ui = rng.normal(0, scale, (n_hidden, n_hidden))
        self.bi = np.zeros(n_hidden)
        # Forget gate
        self.Wf = rng.normal(0, scale, (n_hidden, n_in))
        self.Uf = rng.normal(0, scale, (n_hidden, n_hidden))
        self.bf = np.zeros(n_hidden)
        # Output gate
        self.Wo = rng.normal(0, scale, (n_hidden, n_in))
        self.Uo = rng.normal(0, scale, (n_hidden, n_hidden))
        self.bo = np.zeros(n_hidden)
        # Cell input
        self.Wc = rng.normal(0, scale, (n_hidden, n_in))
        self.Uc = rng.normal(0, scale, (n_hidden, n_hidden))
        self.bc = np.zeros(n_hidden)

        # Readout
        self.W_out = rng.normal(0, scale, (n_out, n_hidden))
        self.b_out = np.zeros(n_out)

        self.h = np.zeros(n_hidden)
        self.c = np.zeros(n_hidden)

        self._init_adam()

    def _init_adam(self):
        self.m = {}
        self.v = {}
        self.t = 0
        for name in ["Wi", "Ui", "bi", "Wf", "Uf", "bf",
                     "Wo", "Uo", "bo", "Wc", "Uc", "bc", "W_out", "b_out"]:
            p = getattr(self, name)
            self.m[name] = np.zeros_like(p)
            self.v[name] = np.zeros_like(p)

    def _adam_step(self, grads: dict):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for key, g in grads.items():
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * g
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * g ** 2
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)
            param = getattr(self, key)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        i = sigmoid(self.Wi @ x + self.Ui @ self.h + self.bi)
        f = sigmoid(self.Wf @ x + self.Uf @ self.h + self.bf)
        o = sigmoid(self.Wo @ x + self.Uo @ self.h + self.bo)
        c_tilde = np.tanh(self.Wc @ x + self.Uc @ self.h + self.bc)
        self.c = f * self.c + i * c_tilde
        self.h = o * np.tanh(self.c)
        return self.W_out @ self.h + self.b_out

    def learn(self, x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        h_prev = self.h.copy()
        c_prev = self.c.copy()

        i = sigmoid(self.Wi @ x + self.Ui @ h_prev + self.bi)
        f = sigmoid(self.Wf @ x + self.Uf @ h_prev + self.bf)
        o = sigmoid(self.Wo @ x + self.Uo @ h_prev + self.bo)
        c_tilde = np.tanh(self.Wc @ x + self.Uc @ h_prev + self.bc)
        c = f * c_prev + i * c_tilde
        h = o * np.tanh(c)
        pred = self.W_out @ h + self.b_out
        err = pred - y
        mse = float((err ** 2).mean())

        # Backward
        dW_out = np.outer(err, h)
        db_out = err.copy()
        dh = self.W_out.T @ err

        # Through h = o * tanh(c)
        do = dh * np.tanh(c)
        dc = dh * o * (1 - np.tanh(c) ** 2)

        # Through c = f*c_prev + i*c_tilde
        df = dc * c_prev
        di = dc * c_tilde
        dc_tilde = dc * i
        dc_prev = dc * f

        # Through tanp
        dc_in = dc_tilde * (1 - c_tilde ** 2)

        # Gate gradients
        di_s = di * sigmoid_prime(self.Wi @ x + self.Ui @ h_prev + self.bi)
        dWi = np.outer(di_s, x)
        dUi = np.outer(di_s, h_prev)
        dbi = di_s

        df_s = df * sigmoid_prime(self.Wf @ x + self.Uf @ h_prev + self.bf)
        dWf = np.outer(df_s, x)
        dUf = np.outer(df_s, h_prev)
        dbf = df_s

        do_s = do * sigmoid_prime(self.Wo @ x + self.Uo @ h_prev + self.bo)
        dWo = np.outer(do_s, x)
        dUo = np.outer(do_s, h_prev)
        dbo = do_s

        dWc = np.outer(dc_in, x)
        dUc = np.outer(dc_in, h_prev)
        dbc = dc_in

        grads = {
            "Wi": dWi, "Ui": dUi, "bi": dbi,
            "Wf": dWf, "Uf": dUf, "bf": dbf,
            "Wo": dWo, "Uo": dUo, "bo": dbo,
            "Wc": dWc, "Uc": dUc, "bc": dbc,
            "W_out": dW_out, "b_out": db_out,
        }

        self._adam_step(grads)
        self.h = h
        self.c = c
        return mse

    def reset_state(self):
        self.h = np.zeros(self.n_hidden)
        self.c = np.zeros(self.n_hidden)

    def n_params(self) -> int:
        n = 0
        for name in ["Wi", "Ui", "bi", "Wf", "Uf", "bf",
                     "Wo", "Uo", "bo", "Wc", "Uc", "bc", "W_out", "b_out"]:
            n += getattr(self, name).size
        return n


# ═══════════════════════════════════════════════════════════════════════════════
#  CfC — Liquid reservoir + simple delta-rule readout (NO plasticity)
# ═══════════════════════════════════════════════════════════════════════════════
class CfC:
    """Liquid reservoir (LTC/CfC) with a plain delta-rule readout.

    This is VISÃO WITHOUT the plasticity mechanisms (no Oja, no EWC, no surprise).
    Uses the exact same LiquidCell as VisaoBrain — only the learner is simpler.
    """

    def __init__(self, n_in: int = 1, n_hidden: int = 64, n_out: int = 1,
                 lr: float = 0.02, sparsity: float = 0.6,
                 tau_min: float = 0.4, tau_max: float = 4.0,
                 dt: float = 0.15, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.cell = LiquidCell(
            n_in=n_in, n_hidden=n_hidden, sparsity=sparsity,
            tau_min=tau_min, tau_max=tau_max, dt=dt, rng=rng,
        )
        self.n_out = n_out
        self.lr = lr
        scale_out = 1.0 / np.sqrt(n_hidden)
        self.W_out = rng.normal(0, scale_out, (n_out, n_hidden))
        self.b_out = np.zeros(n_out)
        self.x = np.zeros(n_hidden)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        self.x, _ = self.cell.step(self.x, x)
        return self.W_out @ self.x + self.b_out

    def learn(self, x: np.ndarray, y: np.ndarray) -> float:
        """Plain delta rule. Returns MSE."""
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        self.x, _ = self.cell.step(self.x, x)
        pred = self.W_out @ self.x + self.b_out
        err = pred - y
        mse = float((err ** 2).mean())
        # Delta rule (no consolidation, no surprise, no Oja)
        self.W_out -= self.lr * np.outer(err, self.x)
        self.b_out -= self.lr * err
        return mse

    def reset_state(self):
        self.x = np.zeros(self.cell.n_hidden)

    def n_params(self) -> int:
        return (
            self.cell.W_in.size + self.cell.W_rec.size +
            self.cell.b.size + self.cell.A.size +
            self.W_out.size + self.b_out.size
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)
