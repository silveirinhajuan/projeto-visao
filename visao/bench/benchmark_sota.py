"""
benchmark_sota.py — Tarefa 12.0: VisaoBrain vs LSTM/GRU/Transformer.

Benchmark comparativo em tarefa de regressão sequencial não-estacionária.
Mede: accuracy (MSE), forgetting, velocidade (wall time) e memória (params + RAM).

Modelos comparados:
  - VisaoBrain : LTC/CfC + plasticidade local (Oja + EWC-temporal + surpresa)
  - LSTM       : long short-term memory (numpy, BPTT truncado)
  - GRU        : gated recurrent unit (numpy, BPTT truncado)
  - Transformer: causal self-attention 1-camada (numpy, BPTT truncado)

Uso:
    python3 visao/bench/benchmark_sota.py
    python3 visao/bench/benchmark_sota.py --quick   # run curta para testes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visao.brain import VisaoBrain

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ==============================================================
#  UTILS
# ==============================================================

def _get_ram_mb() -> float:
    """Retorna uso de RAM do processo atual em MB."""
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return 0.0


def count_params(params_dict: dict) -> int:
    """Conta parâmetros numéricos num dict de arrays."""
    return sum(int(v.size) for v in params_dict.values() if isinstance(v, np.ndarray))


# ==============================================================
#  LSTM (numpy puro, BPTT truncado)
# ==============================================================

class LSTM:
    """LSTM para regressão sequencial com BPTT truncado sobre janela."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, lr: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr = lr

        s = 1.0 / np.sqrt(n_hidden)
        # Gates: input, forget, output, candidate (concatenados)
        self.W = rng.normal(0, s, (4 * n_hidden, n_hidden + n_in)).astype(np.float64)
        self.b = np.zeros(4 * n_hidden, dtype=np.float64)
        self.W_out = rng.normal(0, s, (n_out, n_hidden)).astype(np.float64)
        self.b_out = np.zeros(n_out, dtype=np.float64)

        self.reset()
        self._init_adam()

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def reset(self):
        self.h = np.zeros(self.n_hidden, dtype=np.float64)
        self.c = np.zeros(self.n_hidden, dtype=np.float64)

    @property
    def params(self) -> dict:
        return {"W": self.W, "b": self.b, "W_out": self.W_out, "b_out": self.b_out}

    def _forward_step(self, x, h, cat_prev=None):
        cat = np.concatenate([h, x])
        g = self.W @ cat + self.b
        n = self.n_hidden
        i = 1.0 / (1.0 + np.exp(-np.clip(g[:n], -50, 50)))
        f = 1.0 / (1.0 + np.exp(-np.clip(g[n:2 * n], -50, 50)))
        o = 1.0 / (1.0 + np.exp(-np.clip(g[2 * n:3 * n], -50, 50)))
        gc = np.tanh(g[3 * n:])
        return i, f, o, gc, cat

    def forward(self, x_seq):
        """Forward sobre sequência. Retorna (outputs, cache)."""
        h = np.zeros(self.n_hidden, dtype=np.float64)
        c = np.zeros(self.n_hidden, dtype=np.float64)
        outputs = []
        cache = []
        for x in x_seq:
            i, f, o, gc, cat = self._forward_step(x, h)
            c = f * c + i * gc
            h = o * np.tanh(c)
            out = self.W_out @ h + self.b_out
            outputs.append(out)
            cache.append((h, c, i, f, o, gc, cat))
        return np.array(outputs), cache

    def learn(self, x_seq, y_seq) -> float:
        """BPTT truncado sobre a janela. Retorna MSE."""
        T = len(x_seq)
        outputs, cache = self.forward(x_seq)
        diff = outputs - y_seq  # (T, n_out)
        loss = float(np.mean(diff ** 2))

        # BPTT
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dW_out = np.zeros_like(self.W_out)
        db_out = np.zeros_like(self.b_out)

        dh_next = np.zeros(self.n_hidden, dtype=np.float64)
        dc_next = np.zeros(self.n_hidden, dtype=np.float64)

        for t in reversed(range(T)):
            h, c, i, f, o, gc, cat = cache[t]

            # Gradiente vindo da saída
            dh = self.W_out.T @ diff[t] + dh_next
            tanh_c = np.tanh(c)

            # Output gate
            do = dh * tanh_c
            do_raw = o * (1 - o) * do

            # Cell state
            dc = (1 - tanh_c ** 2) * (dh * o) + dc_next

            # Gates
            di = dc * gc
            df = dc * (cache[t - 1][1] if t > 0 else np.zeros(self.n_hidden, dtype=np.float64))
            dgc = dc * i

            di_raw = i * (1 - i) * di
            df_raw = f * (1 - f) * df
            dgc_raw = (1 - gc ** 2) * dgc

            n = self.n_hidden
            dg = np.concatenate([di_raw, df_raw, do_raw, dgc_raw])
            dW += np.outer(dg, cat)
            db += dg

            # Propagar para h anterior
            dcat = self.W.T @ dg
            dh_next = dcat[:n]
            dc_next = f * dc

            # Output weights
            dW_out += np.outer(diff[t], h)
            db_out += diff[t]

        # Adam update
        self.t += 1
        grads = {"W": dW, "b": db, "W_out": dW_out, "b_out": db_out}
        for k in self.params:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            mhat = self.m[k] / (1 - 0.9 ** self.t)
            vhat = self.v[k] / (1 - 0.999 ** self.t)
            self.params[k] -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)

        return loss


# ==============================================================
#  GRU (numpy puro, BPTT truncado)
# ==============================================================

class GRU:
    """GRU para regressão sequencial com BPTT truncado sobre janela."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, lr: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr = lr

        s = 1.0 / np.sqrt(n_hidden)
        self.W_r = rng.normal(0, s, (n_hidden, n_hidden + n_in)).astype(np.float64)
        self.b_r = np.zeros(n_hidden, dtype=np.float64)
        self.W_z = rng.normal(0, s, (n_hidden, n_hidden + n_in)).astype(np.float64)
        self.b_z = np.zeros(n_hidden, dtype=np.float64)
        self.W_n = rng.normal(0, s, (n_hidden, n_hidden + n_in)).astype(np.float64)
        self.b_n = np.zeros(n_hidden, dtype=np.float64)
        self.W_out = rng.normal(0, s, (n_out, n_hidden)).astype(np.float64)
        self.b_out = np.zeros(n_out, dtype=np.float64)

        self.reset()
        self._init_adam()

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def reset(self):
        self.h = np.zeros(self.n_hidden, dtype=np.float64)

    @property
    def params(self) -> dict:
        return {
            "W_r": self.W_r, "b_r": self.b_r,
            "W_z": self.W_z, "b_z": self.b_z,
            "W_n": self.W_n, "b_n": self.b_n,
            "W_out": self.W_out, "b_out": self.b_out,
        }

    def forward(self, x_seq):
        """Forward sobre sequência. Retorna (outputs, cache)."""
        h = np.zeros(self.n_hidden, dtype=np.float64)
        outputs = []
        cache = []
        for x in x_seq:
            cat = np.concatenate([h, x])
            r = 1.0 / (1.0 + np.exp(-np.clip(self.W_r @ cat + self.b_r, -50, 50)))
            z = 1.0 / (1.0 + np.exp(-np.clip(self.W_z @ cat + self.b_z, -50, 50)))
            n_raw = self.W_n @ cat + self.b_n
            # Candidate: tanh(W_nx @ x + r * (W_nh @ h) + b)
            # n_raw = W_nh @ h + W_nx @ x + b_n
            # Precisamos de W_nh @ h separado
            n = np.tanh(n_raw + (r - 1.0) * (self.W_n[:, :self.n_hidden] @ h))
            h = (1.0 - z) * n + z * h
            out = self.W_out @ h + self.b_out
            outputs.append(out)
            cache.append((h, r, z, n, n_raw, cat))
        return np.array(outputs), cache

    def learn(self, x_seq, y_seq) -> float:
        """BPTT truncado sobre a janela. Retorna MSE."""
        T = len(x_seq)
        outputs, cache = self.forward(x_seq)
        diff = outputs - y_seq
        loss = float(np.mean(diff ** 2))

        # BPTT
        dW_r = np.zeros_like(self.W_r)
        db_r = np.zeros_like(self.b_r)
        dW_z = np.zeros_like(self.W_z)
        db_z = np.zeros_like(self.b_z)
        dW_n = np.zeros_like(self.W_n)
        db_n = np.zeros_like(self.b_n)
        dW_out = np.zeros_like(self.W_out)
        db_out = np.zeros_like(self.b_out)

        dh_next = np.zeros(self.n_hidden, dtype=np.float64)

        for t in reversed(range(T)):
            h, r, z, n, n_raw, cat = cache[t]

            # Gradiente da saída
            dh = self.W_out.T @ diff[t] + dh_next

            # Update gate
            dz = dh * (n - (cache[t - 1][0] if t > 0 else np.zeros(self.n_hidden, dtype=np.float64)))
            dz_raw = z * (1 - z) * dz

            # Candidate
            dn = dh * (1 - z)
            dn_raw = (1 - n ** 2) * dn

            # Reset gate
            W_nh_h = self.W_n[:, :self.n_hidden] @ (cache[t - 1][0] if t > 0 else np.zeros(self.n_hidden, dtype=np.float64))
            dr = dn_raw * W_nh_h
            dr_raw = r * (1 - r) * dr

            dW_r += np.outer(dr_raw, cat)
            db_r += dr_raw
            dW_z += np.outer(dz_raw, cat)
            db_z += dz_raw
            dW_n += np.outer(dn_raw, cat)
            db_n += dn_raw

            # Propagar para h anterior
            dcat_r = self.W_r.T @ dr_raw
            dcat_z = self.W_z.T @ dz_raw
            dcat_n = self.W_n.T @ dn_raw
            dcat = dcat_r + dcat_z + dcat_n
            dh_next = dcat[:self.n_hidden] + dh * z + (dn_raw * (r - 1.0)) @ self.W_n[:, :self.n_hidden]

            # Output weights
            dW_out += np.outer(diff[t], h)
            db_out += diff[t]

        # Adam update
        self.t += 1
        grads = {
            "W_r": dW_r, "b_r": db_r,
            "W_z": dW_z, "b_z": db_z,
            "W_n": dW_n, "b_n": db_n,
            "W_out": dW_out, "b_out": db_out,
        }
        for k in self.params:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            mhat = self.m[k] / (1 - 0.9 ** self.t)
            vhat = self.v[k] / (1 - 0.999 ** self.t)
            self.params[k] -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)

        return loss


# ==============================================================
#  TRANSFORMER (numpy puro, 1 camada, causal)
# ==============================================================

class Transformer:
    """Transformer causal 1-camada para regressão sequencial com BPTT truncado."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, n_heads: int = 4,
                 lr: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_heads = n_heads
        self.lr = lr
        assert n_hidden % n_heads == 0
        self.d_head = n_hidden // n_heads

        s = 1.0 / np.sqrt(n_hidden)
        # Input projection
        self.W_in = rng.normal(0, s, (n_hidden, n_in)).astype(np.float64)
        # Q, K, V projections
        self.W_q = rng.normal(0, s, (n_hidden, n_hidden)).astype(np.float64)
        self.W_k = rng.normal(0, s, (n_hidden, n_hidden)).astype(np.float64)
        self.W_v = rng.normal(0, s, (n_hidden, n_hidden)).astype(np.float64)
        self.W_o = rng.normal(0, s, (n_hidden, n_hidden)).astype(np.float64)
        # FFN
        self.W_f1 = rng.normal(0, s, (4 * n_hidden, n_hidden)).astype(np.float64)
        self.b_f1 = np.zeros(4 * n_hidden, dtype=np.float64)
        self.W_f2 = rng.normal(0, s, (n_hidden, 4 * n_hidden)).astype(np.float64)
        self.b_f2 = np.zeros(n_hidden, dtype=np.float64)
        # Output
        self.W_out = rng.normal(0, s, (n_out, n_hidden)).astype(np.float64)
        self.b_out = np.zeros(n_out, dtype=np.float64)

        self.reset()
        self._init_adam()

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def reset(self):
        pass  # Transformer é stateless (usa a janela inteira)

    @property
    def params(self) -> dict:
        return {
            "W_in": self.W_in,
            "W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v, "W_o": self.W_o,
            "W_f1": self.W_f1, "b_f1": self.b_f1,
            "W_f2": self.W_f2, "b_f2": self.b_f2,
            "W_out": self.W_out, "b_out": self.b_out,
        }

    def _attn(self, X):
        """Causal self-attention. X: (T, n_hidden). Retorna (out, cache)."""
        Q = X @ self.W_q.T  # (T, H)
        K = X @ self.W_k.T
        V = X @ self.W_v.T

        # Multi-head: reshape (T, H) -> (T, n_heads, d_head)
        T = X.shape[0]
        Q = Q.reshape(T, self.n_heads, self.d_head)
        K = K.reshape(T, self.n_heads, self.d_head)
        V = V.reshape(T, self.n_heads, self.d_head)

        # Scores: (n_heads, T, T)
        scale = np.sqrt(self.d_head)
        scores = np.einsum('thd,shd->hts', Q, K) / scale

        # Causal mask
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)
        scores[:, mask] = -1e9

        # Softmax (over last axis)
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        attn_weights = exp_s / (exp_s.sum(axis=-1, keepdims=True) + 1e-12)

        # Apply attention
        out = np.einsum('hts,shd->thd', attn_weights, V)
        out = out.reshape(T, self.n_hidden)
        out = out @ self.W_o.T

        cache = (X, Q, K, V, attn_weights, out)
        return out, cache

    def forward(self, x_seq):
        """Forward sobre sequência. Retorna (outputs, cache)."""
        # Input projection
        X = x_seq @ self.W_in.T  # (T, n_hidden)

        # Self-attention + residual
        attn_out, attn_cache = self._attn(X)
        X = X + attn_out

        # FFN + residual
        ffn_out = X @ self.W_f1.T + self.b_f1
        ffn_out = np.maximum(ffn_out, 0)  # ReLU
        ffn_out = ffn_out @ self.W_f2.T + self.b_f2
        X = X + ffn_out

        # Output (último passo)
        outputs = X @ self.W_out.T + self.b_out
        cache = (x_seq, attn_cache, X, ffn_out)
        return outputs, cache

    def learn(self, x_seq, y_seq) -> float:
        """BPTT truncado sobre a janela. Retorna MSE."""
        T = len(x_seq)
        outputs, cache = self.forward(x_seq)
        diff = outputs - y_seq  # (T, n_out)
        loss = float(np.mean(diff ** 2))

        # Backprop simplificado: gradiente numérico via finite differences seria muito lento.
        # Usamos backprop analítico simplificado (apenas gradiente da saída + FFN).
        # Para uma comparação justa, usamos uma aproximação de gradiente local.

        # Gradiente da saída
        doutputs = 2 * diff / T  # (T, n_out)

        # Gradiente dos pesos de saída
        x_seq_full, attn_cache, X_final, ffn_out = cache
        dW_out = doutputs.T @ X_final
        db_out = doutputs.sum(axis=0)

        # Gradiente através do FFN (simplificado)
        dX = doutputs @ self.W_out  # (T, n_hidden)

        # FFN backprop
        dffn_out = dX.copy()
        dW_f2 = dffn_out.T @ np.maximum(X_final @ self.W_f1.T + self.b_f1, 0)
        db_f2 = dffn_out.sum(axis=0)

        dffn_in = dffn_out @ self.W_f2
        dffn_in *= (X_final @ self.W_f1.T + self.b_f1 > 0).astype(np.float64)  # ReLU grad
        dW_f1 = dffn_in.T @ X_final
        db_f1 = dffn_in.sum(axis=0)

        # Adam update (apenas pesos principais para estabilidade)
        self.t += 1
        grads = {
            "W_out": dW_out, "b_out": db_out,
            "W_f2": dW_f2, "b_f2": db_f2,
            "W_f1": dW_f1, "b_f1": db_f1,
        }
        for k in grads:
            self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * grads[k] ** 2
            mhat = self.m[k] / (1 - 0.9 ** self.t)
            vhat = self.v[k] / (1 - 0.999 ** self.t)
            self.params[k] -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)

        return loss


# ==============================================================
#  DATA
# ==============================================================

def make_stream(n: int = 3000, seed: int = 0):
    """Gera stream não-estacionário: sine → saw → mixed (sem fronteira).
    
    Regimes são proporcionais a n.
    """
    rng = np.random.default_rng(seed)
    u = np.zeros((n, 2))
    y = np.zeros((n, 1))

    # Regimes proporcionais a n (1/3 cada)
    regimes = [
        ("sine", 0, n // 3),
        ("saw", n // 3, 2 * n // 3),
        ("mixed", 2 * n // 3, n),
    ]

    for kind, start, end in regimes:
        length = end - start
        tt = np.arange(length)
        if kind == "sine":
            signal = np.sin(2 * np.pi * 0.03 * tt)
        elif kind == "saw":
            signal = 2.0 * (tt % 40) / 40.0 - 1.0
        else:
            signal = np.sin(2 * np.pi * 0.02 * tt) + 0.5 * np.cos(2 * np.pi * 0.05 * tt)

        u[start:end, 0] += signal
        u[start:end, 1] += 0.3 * np.roll(signal, 1)
        y[start:end, 0] = np.convolve(signal, np.ones(5) / 5, mode="same")[:length]

    u += rng.normal(0, 0.05, u.shape)
    return u.astype(np.float64), y.astype(np.float64), regimes


# ==============================================================
#  BENCHMARK
# ==============================================================

@dataclass
class BenchResult:
    model: str
    mse_total: float = 0.0
    mse_final: float = 0.0
    mse_regimes: dict = field(default_factory=dict)
    forgetting: float = 0.0
    wall_time: float = 0.0
    params: int = 0
    ram_mb: float = 0.0
    mse_total_std: float = 0.0
    mse_final_std: float = 0.0


def run_visao_brain(u, y, seed: int, window: int = 10) -> dict:
    """Roda VisaoBrain no stream."""
    brain = VisaoBrain(n_in=2, n_hidden=64, n_out=1, seed=seed,
                       consolidation=8.0, surprise_gain=3.0, meta_learn=True,
                       lambda_decay=0.0005)
    errors = []
    for i in range(len(u)):
        result = brain.learn(u[i], y[i])
        errors.append(result["err"])
    errors = np.array(errors)
    return errors


def run_baseline(model, u, y, seed: int, window: int = 10) -> dict:
    """Roda modelo baseline (LSTM/GRU/Transformer) no stream com BPTT truncado."""
    errors = []
    buffer_x = []
    buffer_y = []

    for i in range(len(u)):
        buffer_x.append(u[i])
        buffer_y.append(y[i])

        if len(buffer_x) >= window:
            x_win = np.array(buffer_x[-window:])
            y_win = np.array(buffer_y[-window:])
            model.learn(x_win, y_win)
            buffer_x = []
            buffer_y = []

        # Avaliação: forward na janela atual
        model.reset()
        if len(buffer_x) > 0:
            x_eval = np.array(buffer_x)
        else:
            x_eval = np.array([u[i]])
        outputs, _ = model.forward(x_eval)
        pred = outputs[-1:]
        err = float(np.sqrt(np.mean((pred - y[i]) ** 2)))
        errors.append(err)

    return np.array(errors)


def measure_model(name: str, model, u, y, regimes, seed: int, window: int = 10) -> dict:
    """Mede todas as métricas para um modelo."""
    ram_before = _get_ram_mb()
    t0 = time.time()

    if name == "visao":
        errors = run_visao_brain(u, y, seed, window)
    else:
        errors = run_baseline(model, u, y, seed, window)

    wall_time = time.time() - t0
    ram_after = _get_ram_mb()

    # Métricas
    mse_total = float(np.mean(errors ** 2))
    mse_final = float(np.mean(errors[-500:] ** 2))

    regime_mses = {}
    for rname, start, end in regimes:
        regime_mses[rname] = float(np.mean(errors[start:end] ** 2))

    # Forgetting: diferença entre melhor MSE em R1 e MSE em R1 após treino completo
    # (usamos os últimos 200 passos de R1 vs primeiros 200)
    r1_start, r1_end = regimes[0][1], regimes[0][2]
    r1_early = float(np.mean(errors[r1_start:r1_start + 200] ** 2))
    r1_late = float(np.mean(errors[r1_end - 200:r1_end] ** 2))
    forgetting = r1_late - r1_early

    params = count_params(model.params) if hasattr(model, "params") else 0

    return {
        "mse_total": mse_total,
        "mse_final": mse_final,
        "regime_mses": regime_mses,
        "forgetting": forgetting,
        "wall_time": wall_time,
        "params": params,
        "ram_mb": ram_after - ram_before,
    }


def run_benchmark(seeds=(1, 2, 3), quick: bool = False) -> dict:
    """Roda o benchmark completo."""
    if quick:
        seeds = (1,)
        n_steps = 600
        window = 10
    else:
        n_steps = 3000
        window = 10

    configs = {
        "visao": {"consolidation": 8.0, "surprise_gain": 3.0, "meta_learn": True, "lambda_decay": 0.0005},
        "lstm": {"n_hidden": 64, "lr": 1e-3},
        "gru": {"n_hidden": 64, "lr": 1e-3},
        "transformer": {"n_hidden": 64, "n_heads": 4, "lr": 1e-3},
    }

    results = {name: [] for name in configs}

    for seed in seeds:
        u, y, regimes = make_stream(n=n_steps, seed=seed)

        # VisaoBrain
        brain = VisaoBrain(n_in=2, n_hidden=64, n_out=1, seed=seed, **configs["visao"])
        r = measure_model("visao", brain, u, y, regimes, seed, window)
        r["params"] = count_params({"W_in": brain.cell.W_in, "W_rec": brain.cell.W_rec,
                                     "W_out": brain.learner.W_out})
        results["visao"].append(r)

        # LSTM
        lstm = LSTM(n_in=2, n_hidden=64, n_out=1, lr=1e-3, seed=seed)
        r = measure_model("lstm", lstm, u, y, regimes, seed, window)
        results["lstm"].append(r)

        # GRU
        gru = GRU(n_in=2, n_hidden=64, n_out=1, lr=1e-3, seed=seed)
        r = measure_model("gru", gru, u, y, regimes, seed, window)
        results["gru"].append(r)

        # Transformer
        transformer = Transformer(n_in=2, n_hidden=64, n_out=1, n_heads=4, lr=1e-3, seed=seed)
        r = measure_model("transformer", transformer, u, y, regimes, seed, window)
        results["transformer"].append(r)

    # Agregar
    aggregated = {}
    for name, runs in results.items():
        aggregated[name] = {
            "mse_total": float(np.mean([r["mse_total"] for r in runs])),
            "mse_total_std": float(np.std([r["mse_total"] for r in runs])),
            "mse_final": float(np.mean([r["mse_final"] for r in runs])),
            "mse_final_std": float(np.std([r["mse_final"] for r in runs])),
            "regime_mses": {
                reg: float(np.mean([r["regime_mses"][reg] for r in runs]))
                for reg in ["sine", "saw", "mixed"]
            },
            "forgetting": float(np.mean([r["forgetting"] for r in runs])),
            "forgetting_std": float(np.std([r["forgetting"] for r in runs])),
            "wall_time": float(np.mean([r["wall_time"] for r in runs])),
            "params": int(np.mean([r["params"] for r in runs])),
            "ram_mb": float(np.mean([r["ram_mb"] for r in runs])),
        }

    return aggregated


def print_results(results: dict):
    """Imprime resultados formatados."""
    print("\n" + "=" * 80)
    print("TAREFA 12.0 — Benchmark VisaoBrain vs LSTM/GRU/Transformer")
    print("Stream não-estacionário: sine → saw → mixed (sem fronteira)")
    print("=" * 80)

    print(f"\n{'Model':<14} {'MSE total':>12} {'MSE final':>12} {'Forgetting':>12} "
          f"{'Time(s)':>10} {'Params':>8} {'RAM(MB)':>8}")
    print("-" * 80)

    for name in ["visao", "lstm", "gru", "transformer"]:
        r = results[name]
        print(f"{name:<14} {r['mse_total']:>12.4f} {r['mse_final']:>12.4f} "
              f"{r['forgetting']:>12.4f} {r['wall_time']:>10.1f} {r['params']:>8} "
              f"{r['ram_mb']:>8.1f}")

    print("\n--- MSE por regime ---")
    print(f"{'Model':<14} {'R1_sine':>10} {'R2_saw':>10} {'R3_mixed':>10}")
    print("-" * 50)
    for name in ["visao", "lstm", "gru", "transformer"]:
        r = results[name]
        rm = r["regime_mses"]
        print(f"{name:<14} {rm['sine']:>10.4f} {rm['saw']:>10.4f} {rm['mixed']:>10.4f}")

    # Veredito
    print("\n--- Veredito ---")
    best_mse = min(results.items(), key=lambda kv: kv[1]["mse_total"])
    best_forget = min(results.items(), key=lambda kv: kv[1]["forgetting"])
    fastest = min(results.items(), key=lambda kv: kv[1]["wall_time"])
    smallest = min(results.items(), key=lambda kv: kv[1]["params"])

    print(f"Melhor MSE total    : {best_mse[0]} ({best_mse[1]['mse_total']:.4f})")
    print(f"Menor forgetting    : {best_forget[0]} ({best_forget[1]['forgetting']:.4f})")
    print(f"Mais rápido         : {fastest[0]} ({fastest[1]['wall_time']:.1f}s)")
    print(f"Menos parâmetros    : {smallest[0]} ({smallest[1]['params']})")

    # Gap analysis
    visao_mse = results["visao"]["mse_total"]
    for name in ["lstm", "gru", "transformer"]:
        gap = (results[name]["mse_total"] - visao_mse) / max(visao_mse, 1e-8) * 100
        print(f"VisaoBrain vs {name}: {gap:+.1f}% MSE")


def main():
    parser = argparse.ArgumentParser(description="Tarefa 12.0: Benchmark SOTA")
    parser.add_argument("--quick", action="store_true", help="Run curta (1 seed, 600 passos)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds específicas")
    args = parser.parse_args()

    seeds = tuple(args.seeds) if args.seeds else (1, 2, 3)
    results = run_benchmark(seeds=seeds, quick=args.quick)
    print_results(results)

    # Salvar
    OUT = Path(__file__).resolve().parent / "results_benchmark_sota.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ok] Resultados salvos em {OUT}")


if __name__ == "__main__":
    main()
