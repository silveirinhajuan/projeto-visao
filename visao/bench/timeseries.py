"""
timeseries.py — Benchmarks públicos de série temporal (tarefa 1.5).

Primeiro alvo: MNIST sequencial permutado (psMNIST / "sMNIST permutado").
Compara a célula líquida CfC contra um baseline LSTM, reportando
acurácia e contagem de parâmetros treináveis — a alegação do projeto é
"empatar ou superar LSTM/GRU com ~10x menos parâmetros".

Adam implementado em JAX puro (sem optax) para manter o ambiente enxuto.
"""

from __future__ import annotations

import gzip
import os

import jax
import jax.numpy as jnp
import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_PERM = {}


# ------------------------------------------------------------------ dados
def _idx_reader(path: str, n_items: int, rows: int, cols: int, header: int) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(header)  # cabeçalho IDX (imagens=16, labels=8)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_items, rows, cols)


def load_smnist_permuted(
    n_train=None, n_test=None, perm_seed: int = 42, data_dir: str = _DATA_DIR
):
    """Carrega MNIST como sequência (T=784, 1) sob uma permutação fixa de pixels.

    Retorna (x_train, y_train, x_test, y_test); x em float32 [0,1], y int32.
    """
    if perm_seed not in _PERM:
        rng = np.random.default_rng(perm_seed)
        _PERM[perm_seed] = rng.permutation(784)
    perm = _PERM[perm_seed]

    x_tr = _idx_reader(os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
                       60000, 28, 28, 16).reshape(60000, 784) / 255.0
    y_tr = _idx_reader(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
                       60000, 1, 1, 8).reshape(60000).astype(np.int32)
    x_te = _idx_reader(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
                       10000, 28, 28, 16).reshape(10000, 784) / 255.0
    y_te = _idx_reader(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"),
                       10000, 1, 1, 8).reshape(10000).astype(np.int32)

    x_tr = x_tr[:, perm].reshape(-1, 784, 1).astype(np.float32)
    x_te = x_te[:, perm].reshape(-1, 784, 1).astype(np.float32)
    if n_train is not None:
        x_tr, y_tr = x_tr[:n_train], y_tr[:n_train]
    if n_test is not None:
        x_te, y_te = x_te[:n_test], y_te[:n_test]
    return x_tr, y_tr, x_te, y_te


# ------------------------------------------------------------------ Adam
def adam_init(params):
    return {
        "mu": jax.tree_util.tree_map(jnp.zeros_like, params),
        "nu": jax.tree_util.tree_map(jnp.zeros_like, params),
        "t": 0,
    }


def adam_update(params, grads, opt, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = opt["t"] + 1
    mu = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, opt["mu"], grads)
    nu = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * (g * g), opt["nu"], grads)
    mhat = jax.tree_util.tree_map(lambda m: m / (1 - b1**t), mu)
    nhat = jax.tree_util.tree_map(lambda v: v / (1 - b2**t), nu)
    new = jax.tree_util.tree_map(
        lambda p, mh, nh: p - lr * mh / (jnp.sqrt(nh) + eps), params, mhat, nhat
    )
    return new, {"mu": mu, "nu": nu, "t": t}


# ------------------------------------------------------------------ modelos
def cfc_params(cfc_hidden: int = 64, sparsity: float = 0.5, seed: int = 0,
               tau_min: float = 0.5, tau_max: float = 50.0):
    rng = np.random.default_rng(seed)
    h = cfc_hidden
    # n_in = 1 (um pixel por passo): a escala de entrada segue 1/sqrt(n_in).
    scale_in = 1.0 / np.sqrt(1)
    scale_rec = 1.0 / np.sqrt(h)
    W_in = rng.normal(0, scale_in, (h, 1)).astype(np.float32)
    W_rec = rng.normal(0, scale_rec, (h, h)).astype(np.float32)
    mask = (rng.random((h, h)) > sparsity).astype(np.float32)
    np.fill_diagonal(mask, 0.0)
    W_rec = W_rec * mask
    b = np.zeros(h, dtype=np.float32)
    A = rng.normal(0, 1.0, h).astype(np.float32)
    # tau heterogêneo em escala log: neurônios lentos guardam contexto.
    tau = np.exp(rng.uniform(np.log(tau_min), np.log(tau_max), h))
    tau_raw = np.log(np.exp(tau) - 1.0).astype(np.float32)  # softplus^{-1}
    W_out = rng.normal(0, 1.0 / np.sqrt(h), (10, h)).astype(np.float32)
    b_out = np.zeros(10, dtype=np.float32)
    return {
        "W_in": jnp.asarray(W_in), "W_rec": jnp.asarray(W_rec),
        "b": jnp.asarray(b), "A": jnp.asarray(A),
        "tau_raw": jnp.asarray(tau_raw),
        "W_out": jnp.asarray(W_out), "b_out": jnp.asarray(b_out),
        "mask": jnp.asarray(mask),
    }


def make_cfc_forward(dt: float = 0.1):
    def forward(params, seq):  # seq (T,1)
        h = jnp.zeros(params["W_in"].shape[0])
        tau = jax.nn.softplus(params["tau_raw"]) + 1e-3
        mask = params["mask"]

        def body(h, x):
            f = jax.nn.sigmoid(params["W_in"] @ x + (params["W_rec"] * mask) @ h + params["b"])
            h = (h + dt * (f * params["A"])) / (1.0 + dt * (1.0 / tau + f))
            return h, h

        # agrega memória de TODA a sequência (média temporal), não só o fim
        _, states = jax.lax.scan(body, h, seq)
        h_mean = jnp.mean(states, axis=0)
        return params["W_out"] @ h_mean + params["b_out"]
    return forward


def lstm_params(lstm_hidden: int = 128, seed: int = 1):
    rng = np.random.default_rng(seed)
    h = lstm_hidden
    s = 1.0 / np.sqrt(h)
    W = rng.normal(0, s, (4 * h, h + 1)).astype(np.float32)
    b = np.zeros(4 * h, dtype=np.float32)
    W_out = rng.normal(0, 1.0 / np.sqrt(h), (10, h)).astype(np.float32)
    b_out = np.zeros(10, dtype=np.float32)
    return {"W": jnp.asarray(W), "b": jnp.asarray(b),
            "W_out": jnp.asarray(W_out), "b_out": jnp.asarray(b_out)}


def lstm_forward(params, seq):  # seq (T,1)
    h = jnp.zeros(params["W"].shape[0] // 4)
    c = jnp.zeros_like(h)
    W, b = params["W"], params["b"]

    def body(carry, x):
        h, c = carry
        cat = jnp.concatenate([h, x])
        g = W @ cat + b
        i = jax.nn.sigmoid(g[:h.shape[0]])
        f = jax.nn.sigmoid(g[h.shape[0]:2 * h.shape[0]])
        o = jax.nn.sigmoid(g[2 * h.shape[0]:3 * h.shape[0]])
        gc = jnp.tanh(g[3 * h.shape[0]:])
        c = f * c + i * gc
        h = o * jnp.tanh(c)
        return (h, c), None

    (h, _), _ = jax.lax.scan(body, (h, c), seq)
    return params["W_out"] @ h + params["b_out"]


# ------------------------------------------------------------------ GRU
def gru_params(gru_hidden: int = 128, seed: int = 2):
    rng = np.random.default_rng(seed)
    h = gru_hidden
    s = 1.0 / np.sqrt(h)
    W_r = rng.normal(0, s, (h, h + 1)).astype(np.float32)
    b_r = np.zeros(h, dtype=np.float32)
    W_z = rng.normal(0, s, (h, h + 1)).astype(np.float32)
    b_z = np.zeros(h, dtype=np.float32)
    W_n = rng.normal(0, s, (h, h + 1)).astype(np.float32)
    b_n = np.zeros(h, dtype=np.float32)
    W_out = rng.normal(0, 1.0 / np.sqrt(h), (10, h)).astype(np.float32)
    b_out = np.zeros(10, dtype=np.float32)
    return {
        "W_r": jnp.asarray(W_r), "b_r": jnp.asarray(b_r),
        "W_z": jnp.asarray(W_z), "b_z": jnp.asarray(b_z),
        "W_n": jnp.asarray(W_n), "b_n": jnp.asarray(b_n),
        "W_out": jnp.asarray(W_out), "b_out": jnp.asarray(b_out),
    }


def gru_forward(params, seq):  # seq (T,1)
    h = jnp.zeros(params["W_r"].shape[0])
    W_r, b_r = params["W_r"], params["b_r"]
    W_z, b_z = params["W_z"], params["b_z"]
    W_n, b_n = params["W_n"], params["b_n"]

    def body(h, x):
        cat = jnp.concatenate([h, x])
        r = jax.nn.sigmoid(W_r @ cat + b_r)
        z = jax.nn.sigmoid(W_z @ cat + b_z)
        g_n = W_n @ cat + b_n
        # candidato GRU: tanh(W_nx@x + r*(W_nh@h) + b) ; g_n ja tem W_nh@h,
        # entao subtrai e re-adiciona ponderado por r.
        r_h = W_n[:, : h.shape[0]] @ h
        n = jnp.tanh(g_n + (r - 1.0) * r_h)
        h = (1.0 - z) * n + z * h
        return h, h

    _, states = jax.lax.scan(body, h, seq)
    return params["W_out"] @ states[-1] + params["b_out"]


# ------------------------------------------------------------------ treino
def count_params(params) -> int:
    return int(sum(int(p.size) for k, p in params.items() if k != "mask"))


def _xent(logits, y):
    onehot = jax.nn.one_hot(y, 10)
    logp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.mean(logp * onehot)


def _train(forward, params_init, x, y, n_epochs, batch, lr):
    params = params_init
    opt = adam_init(params)
    n = x.shape[0]
    rng = np.random.default_rng(123)
    grad_fn = jax.jit(jax.grad(lambda p, xb, yb: _xent(
        jax.vmap(forward, in_axes=(None, 0))(p, xb), yb)))
    fwd_jit = jax.jit(lambda p, xb: jax.vmap(forward, in_axes=(None, 0))(p, xb))
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            g = grad_fn(params, x[idx], y[idx])
            params, opt = adam_update(params, g, opt, lr=lr)
    return params, fwd_jit


def _accuracy(forward_jit, params, x, y, batch=128):
    correct = 0
    n = x.shape[0]
    for i in range(0, n, batch):
        logits = forward_jit(params, x[i:i + batch])
        correct += int(jnp.sum(jnp.argmax(logits, axis=-1) == y[i:i + batch]))
    return correct / n


def benchmark_smnist(
    n_train=2000, n_test=500, n_epochs=30,
    cfc_hidden=64, lstm_hidden=128, batch=128, lr=1e-3, seed=0,
    data_dir=_DATA_DIR,
):
    x_tr, y_tr, x_te, y_te = load_smnist_permuted(
        n_train=n_train, n_test=n_test, perm_seed=42, data_dir=data_dir
    )
    y_tr = jnp.asarray(y_tr)
    y_te = jnp.asarray(y_te)

    cfc_p = cfc_params(cfc_hidden=cfc_hidden, seed=seed)
    cfc_fwd = make_cfc_forward()
    cfc_p, cfc_jit = _train(cfc_fwd, cfc_p, x_tr, y_tr, n_epochs, batch, lr)
    cfc_acc = _accuracy(cfc_jit, cfc_p, x_te, y_te)

    lstm_p = lstm_params(lstm_hidden=lstm_hidden, seed=seed + 1)
    lstm_p, lstm_jit = _train(lstm_forward, lstm_p, x_tr, y_tr, n_epochs, batch, lr)
    lstm_acc = _accuracy(lstm_jit, lstm_p, x_te, y_te)

    return {
        "cfc": {"accuracy": float(cfc_acc), "params": count_params(cfc_p)},
        "lstm": {"accuracy": float(lstm_acc), "params": count_params(lstm_p)},
    }
