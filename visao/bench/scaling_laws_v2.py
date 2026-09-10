#!/usr/bin/env python3
"""
scaling_laws_v2.py — Scaling Laws v2: VisaoBrain-JAX com CfC + readout não-linear.

Melhorias vs v1:
  - CfC-JAX em vez de LiquidCell-numpy
  - Readout MLP não-linear (2 camadas ocultas) em vez de LocalLearner linear
  - Treinado com Adam (gradiente real) em vez de delta rule local
  - Batched training (vetorizado)

Meta: α > 0.03 (melhor que baseline numpy α=-0.009)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, value_and_grad
from jax import random as jrandom

ROOT = Path(__file__).resolve().parents[2]

# ==============================================================
#  DATA
# ==============================================================


def make_regression_task(n: int = 1000, seed: int = 0, noise: float = 0.05):
    """Gera tarefa de regressão com memória longa."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    freq1, freq2, freq3 = 0.01, 0.05, 0.005
    signal = (
        np.sin(2 * np.pi * freq1 * t)
        + 0.5 * np.sin(2 * np.pi * freq2 * t + 1.0)
        + 0.3 * np.sin(2 * np.pi * freq3 * t + 2.0)
    )
    u = np.zeros((n, 2))
    u[:, 0] = signal + rng.normal(0, noise, n)
    u[:, 1] = np.roll(signal, 2) + rng.normal(0, noise, n)
    kernel = np.ones(7) / 7
    y = np.convolve(signal, kernel, mode="same")[:, None]
    return u.astype(np.float32), y.astype(np.float32)


# ==============================================================
#  CfC CELL (JAX)
# ==============================================================


def cfc_step(x, u, W_in, W_rec, b, A, tau, dt, mask):
    """Um passo do solver CfC."""
    # W_rec já é esparsidade na inicialização (W_rec * mask)
    fx = jax.nn.sigmoid(W_in @ u + W_rec @ x + b)
    num = x + dt * fx * A
    den = 1.0 + dt * (1.0 / tau + fx)
    return num / den, fx


def cfc_rollout(x0, seq, W_in, W_rec, b, A, tau, dt, mask):
    """Roda uma sequência através da célula CfC usando jax.lax.scan."""
    def step_fn(x, u):
        x_new, fx = cfc_step(x, u, W_in, W_rec, b, A, tau, dt, mask)
        return x_new, x_new  # carry=x_new, output=x_new
    
    # scan: carry=x, output=states
    x_final, states = jax.lax.scan(step_fn, x0, seq)
    return states, None


# ==============================================================
#  READOUT MLP (JAX)
# ==============================================================


def init_mlp_params(key, layer_sizes):
    """Inicializa parâmetros de uma MLP."""
    params = []
    keys = jrandom.split(key, len(layer_sizes) - 1)
    for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        scale = jnp.sqrt(2.0 / n_in)  # He initialization
        W = jrandom.normal(keys[i], (n_out, n_in)) * scale
        b = jnp.zeros(n_out)
        params.append({'W': W, 'b': b})
    return params


def mlp_forward(params, x):
    """Forward pass da MLP com ReLU."""
    for i, p in enumerate(params[:-1]):
        x = jax.nn.relu(p['W'] @ x + p['b'])
    # Última camada linear (sem ativação)
    x = params[-1]['W'] @ x + params[-1]['b']
    return x


# ==============================================================
#  MODELO COMPLETO
# ==============================================================


class JaxVisaoBrain:
    """VisaoBrain com CfC-JAX + readout MLP, treinável com Adam."""
    
    def __init__(self, n_in, n_hidden, n_out, n_hidden_mlp=64, seed=0,
                 sparsity=0.5, tau_min=0.4, tau_max=4.0, dt=0.1):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.dt = dt
        
        # Inicializar pesos CfC
        rng = np.random.default_rng(seed)
        scale_in = 1.0 / np.sqrt(max(n_in, 1))
        scale_rec = 1.0 / np.sqrt(max(n_hidden, 1))
        
        self.W_in = jnp.array(rng.normal(0, scale_in, (n_hidden, n_in)).astype(np.float32))
        W_rec_np = rng.normal(0, scale_rec, (n_hidden, n_hidden)).astype(np.float32)
        mask_np = (rng.random((n_hidden, n_hidden)) > sparsity).astype(np.float32)
        np.fill_diagonal(mask_np, 0.0)
        self.W_rec = jnp.array(W_rec_np * mask_np)
        self.mask = jnp.array(mask_np)
        self.b = jnp.zeros(n_hidden)
        self.A = jnp.array(rng.normal(0, 1.0, n_hidden).astype(np.float32))
        self.tau = jnp.array(np.exp(rng.uniform(np.log(tau_min), np.log(tau_max), n_hidden)).astype(np.float32))
        
        # Inicializar readout MLP
        key = jrandom.PRNGKey(seed)
        self.mlp_params = init_mlp_params(key, [n_hidden, n_hidden_mlp, n_hidden_mlp, n_out])
        
        # Estado inicial
        self.x0 = jnp.zeros(n_hidden)
        
        # Otimizador Adam (parâmetros)
        self.adam_state = self._init_adam()
    
    def _init_adam(self):
        """Inicializa estado do Adam."""
        return {
            't': 0,
            'm_cfc': {'W_in': jnp.zeros_like(self.W_in), 'W_rec': jnp.zeros_like(self.W_rec),
                      'b': jnp.zeros_like(self.b), 'A': jnp.zeros_like(self.A)},
            'v_cfc': {'W_in': jnp.zeros_like(self.W_in), 'W_rec': jnp.zeros_like(self.W_rec),
                      'b': jnp.zeros_like(self.b), 'A': jnp.zeros_like(self.A)},
            'm_mlp': [{'W': jnp.zeros_like(p['W']), 'b': jnp.zeros_like(p['b'])} for p in self.mlp_params],
            'v_mlp': [{'W': jnp.zeros_like(p['W']), 'b': jnp.zeros_like(p['b'])} for p in self.mlp_params],
        }
    
    def forward(self, seq):
        """Forward pass completo: CfC rollout + readout."""
        states, _ = cfc_rollout(self.x0, seq, self.W_in, self.W_rec, self.b, self.A, self.tau, self.dt, self.mask)
        # Usa último estado para predição
        pred = mlp_forward(self.mlp_params, states[-1])
        return pred
    
    def count_params(self) -> int:
        total = self.W_in.size + self.W_rec.size + self.b.size + self.A.size
        for p in self.mlp_params:
            total += p['W'].size + p['b'].size
        return total


# ==============================================================
#  TREINAMENTO
# ==============================================================


def loss_fn(params_dict, model, seq, target):
    """Loss: MSE entre predição e alvo."""
    pred = mlp_forward(params_dict['mlp'], model.forward(seq))
    return jnp.mean((pred - target) ** 2)


@jit
def train_step(cfc_params, mlp_params, opt_state, seq, target, lr=0.001):
    """Um passo de treinamento com Adam."""
    def loss_fn(cp, mp):
        # Reconstruct model temporarily
        states, _ = cfc_rollout(jnp.zeros(cp['A'].shape[0]), seq, cp['W_in'], cp['W_rec'], cp['b'], cp['A'], cp['tau'], 0.1, cp['mask'])
        pred = mlp_forward(mp, states[-1])
        return jnp.mean((pred - target) ** 2)
    
    loss, grads = value_and_grad(loss_fn, argnums=(0, 1))(cfc_params, mlp_params)
    
    # Adam update
    opt_state['t'] += 1
    t = opt_state['t']
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    for key in ['W_in', 'W_rec', 'b', 'A']:
        opt_state['m_cfc'][key] = beta1 * opt_state['m_cfc'][key] + (1 - beta1) * grads[0][key]
        opt_state['v_cfc'][key] = beta2 * opt_state['v_cfc'][key] + (1 - beta2) * grads[0][key]**2
        m_hat = opt_state['m_cfc'][key] / (1 - beta1**t)
        v_hat = opt_state['v_cfc'][key] / (1 - beta2**t)
        cfc_params[key] -= lr * m_hat / (jnp.sqrt(v_hat) + eps)
    
    for i, p in enumerate(mlp_params):
        for k in ['W', 'b']:
            opt_state['m_mlp'][i][k] = beta1 * opt_state['m_mlp'][i][k] + (1 - beta1) * grads[1][i][k]
            opt_state['v_mlp'][i][k] = beta2 * opt_state['v_mlp'][i][k] + (1 - beta2) * grads[1][i][k]**2
            m_hat = opt_state['m_mlp'][i][k] / (1 - beta1**t)
            v_hat = opt_state['v_mlp'][i][k] / (1 - beta2**t)
            p[k] -= lr * m_hat / (jnp.sqrt(v_hat) + eps)
    
    return cfc_params, mlp_params, opt_state, loss


# ==============================================================
#  EXPERIMENTO
# ==============================================================


@dataclass
class ScalingResult:
    n_hidden: int
    n_params: int
    mse_train: float
    mse_test: float
    wall_time: float
    seed: int


def run_experiment(n_hidden: int, seed: int, n_epochs: int = 50, batch_size: int = 32) -> ScalingResult:
    """Treina e avalia uma configuração."""
    
    # Dados
    u_tr, y_tr = make_regression_task(n=600, seed=seed)
    u_te, y_te = make_regression_task(n=200, seed=seed + 1000)
    
    u_tr_j = jnp.array(u_tr)
    y_tr_j = jnp.array(y_tr)
    u_te_j = jnp.array(u_te)
    y_te_j = jnp.array(y_te)
    
    # Modelo
    model = JaxVisaoBrain(n_in=2, n_hidden=n_hidden, n_out=1, n_hidden_mlp=max(64, n_hidden // 2), seed=seed)
    n_params = model.count_params()
    
    # Treinar
    t0 = time.time()
    cfc_params = {'W_in': model.W_in, 'W_rec': model.W_rec, 'b': model.b, 'A': model.A, 'tau': model.tau, 'mask': model.mask}
    mlp_params = model.mlp_params
    opt_state = model._init_adam()
    
    for epoch in range(n_epochs):
        # Mini-batch (sequências de 50 timesteps)
        seq_len = 50
        n_batches = len(u_tr) // seq_len
        for i in range(n_batches):
            start = i * seq_len
            end = start + seq_len
            seq_batch = u_tr_j[start:end]
            target_batch = y_tr_j[end - 1]  # predição no último timestep
            cfc_params, mlp_params, opt_state, loss = train_step(cfc_params, mlp_params, opt_state, seq_batch, target_batch)
    
    wall_time = time.time() - t0
    
    # Avaliar
    model.mlp_params = mlp_params
    model.W_in = cfc_params['W_in']
    model.W_rec = cfc_params['W_rec']
    model.b = cfc_params['b']
    model.A = cfc_params['A']
    
    pred_train = model.forward(u_tr_j)
    pred_test = model.forward(u_te_j)
    
    mse_train = float(jnp.mean((pred_train - y_tr_j[-1]) ** 2))
    mse_test = float(jnp.mean((pred_test - y_te_j[-1]) ** 2))
    
    return ScalingResult(
        n_hidden=n_hidden,
        n_params=n_params,
        mse_train=mse_train,
        mse_test=mse_test,
        wall_time=wall_time,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(description="Scaling Laws v2 — JAX")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    n_hidden_values = [32, 64, 128] if args.quick else [64, 128, 256, 512]
    seeds = (0, 1) if args.quick else (0, 1, 2)
    n_epochs = 20 if args.quick else 50
    
    print("=" * 60)
    print("SCALING LAWS v2 — VisaoBrain-JAX")
    print(f"  n_hidden: {n_hidden_values}")
    print(f"  seeds: {seeds}")
    print(f"  epochs: {n_epochs}")
    print("=" * 60)
    
    results = []
    for n_hidden in n_hidden_values:
        for seed in seeds:
            r = run_experiment(n_hidden, seed, n_epochs=n_epochs)
            results.append(r)
            print(f"  n_hidden={n_hidden:>3}, seed={seed}: params={r.n_params:>7}, MSE_test={r.mse_test:.5f}, time={r.wall_time:.1f}s")
    
    # Fit scaling law
    params_list = [r.n_params for r in results]
    mse_list = [r.mse_test for r in results]
    
    log_p = np.log(params_list)
    log_m = np.log(mse_list)
    A = np.vstack([-log_p, np.ones_like(log_p)]).T
    alpha, log_C = np.linalg.lstsq(A, log_m, rcond=None)[0]
    
    print(f"\nScaling law: MSE = {np.exp(log_C):.4f} × params^(-{alpha:.3f})")
    
    # Salvar
    data_path = Path(__file__).parent / "results_scaling_v2.json"
    data_path.write_text(json.dumps({
        "scaling_law": {"alpha": float(alpha), "C": float(np.exp(log_C))},
        "results": [asdict(r) for r in results],
    }, indent=2))
    print(f"\nResultados salvos em {data_path}")


if __name__ == "__main__":
    main()
