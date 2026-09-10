#!/usr/bin/env python3
"""mnist_fast.py — MNIST split rápido (subset para execução local)."""

import json
import time
import numpy as np
from sklearn.datasets import fetch_openml

print("Carregando MNIST (subset)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(int)

# Subset: 500 amostras por tarefa (treino rápido)
tasks = []
for task_id in range(5):
    mask = (y >= task_id*2) & (y < task_id*2 + 2)
    X_task, y_task = X[mask], (y[mask] - task_id*2)
    idx = np.random.permutation(len(X_task))[:1000]  # max 1000
    split = 800
    tasks.append({
        'X_tr': X_task[idx[:split]].reshape(-1, 28, 28),
        'y_tr': y_task[idx[:split]],
        'X_te': X_task[idx[split:]].reshape(-1, 28, 28),
        'y_te': y_task[idx[split:]],
    })

class LiquidCell:
    def __init__(self, n_in, n_hidden, sparsity=0.6, tau_min=0.4, tau_max=4.0, dt=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.n_hidden = n_hidden
        self.dt = dt
        self.W_in = rng.normal(0, 1/np.sqrt(n_in), (n_hidden, n_in))
        self.W_rec = rng.normal(0, 1/np.sqrt(n_hidden), (n_hidden, n_hidden))
        self.b = np.zeros(n_hidden)
        mask = rng.random((n_hidden, n_hidden)) > sparsity
        np.fill_diagonal(mask, False)
        self.mask = mask.astype(float)
        self.W_rec *= self.mask
        self.tau = np.exp(rng.uniform(np.log(tau_min), np.log(tau_max), n_hidden))
        self.A = rng.normal(0, 1, n_hidden)
    
    def step(self, x, u):
        f = 1/(1+np.exp(-(self.W_in @ u + self.W_rec @ x + self.b)))
        tau_eff = self.dt / self.tau
        dx = (-x + f * self.A) * tau_eff
        return x + dx, None

class LocalLearner:
    def __init__(self, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W_out = rng.normal(0, 0.1, (n_out, n_hidden))
        self.b_out = np.zeros(n_out)
        self.omega = np.zeros((n_out, n_hidden))
        self.lr, self.oja_lr = 0.05, 0.002
        self.consolidation, self.surprise_gain = 1.0, 3.0
        self.err_ema, self.err_var = 1.0, 1.0
    
    def update(self, x, target):
        pred = self.W_out @ x + self.b_out
        err = target - pred
        err_norm = float(np.linalg.norm(err))
        surprise = 1.0 + self.surprise_gain * max(0, (err_norm - self.err_ema) / (self.err_var**0.5 + 1e-8))
        grad = np.outer(err, x) / (self.omega * self.consolidation + 1.0)
        self.W_out += self.lr * surprise * grad
        self.b_out += self.lr * surprise * err
        self.omega += np.abs(err)[:, None] * x[None, :]
        self.err_ema = 0.95*self.err_ema + 0.05*err_norm
        self.err_var = 0.95*self.err_var + 0.05*(err_norm-self.err_ema)**2
    
    def oja_update(self, cell, x_prev, x):
        dW = self.oja_lr * (np.outer(x, x_prev) - np.diag(x**2) @ cell.W_rec)
        cell.W_rec += dW * cell.mask

def evaluate(cell, learner, X, y):
    correct = 0
    for i in range(len(X)):
        x = np.zeros(cell.n_hidden)
        for t in range(28):
            x, _ = cell.step(x, X[i, t])
        if np.argmax(learner.W_out @ x + learner.b_out) == y[i]:
            correct += 1
    return correct / len(y)

results = []
for seed in range(3):
    np.random.seed(seed)
    cell = LiquidCell(28, 64, seed=seed)
    learner = LocalLearner(64, 2, seed=seed)
    
    history = {}
    for task_id, task in enumerate(tasks):
        acc_before = evaluate(cell, learner, task['X_te'], task['y_te'])
        
        for epoch in range(3):
            for i in range(len(task['X_tr'])):
                x = np.zeros(64)
                for t in range(28):
                    x, _ = cell.step(x, task['X_tr'][i, t])
                    if t >= 10:
                        target = np.zeros(2)
                        target[task['y_tr'][i]] = 1
                        learner.update(x, target)
                        x = np.clip(x, -10, 10)
        
        acc_after = evaluate(cell, learner, task['X_te'], task['y_te'])
        history[task_id] = {'before': acc_before, 'after': acc_after}
        print(f"  seed {seed} task {task_id}: {acc_before:.3f} -> {acc_after:.3f}")
    
    forgetting = []
    for task_id in range(4):
        cur = evaluate(cell, learner, tasks[task_id]['X_te'], tasks[task_id]['y_te'])
        forgetting.append(cur - history[task_id]['after'])
    
    results.append({
        'forgetting': float(np.mean(forgetting)),
        'accuracy': float(np.mean([h['after'] for h in history.values()]))
    })
    print(f"  -> fg={results[-1]['forgetting']:.4f}, acc={results[-1]['accuracy']:.4f}")

print(f"\nFINAL: fg={np.mean([r['forgetting'] for r in results]):.4f} +/- {np.std([r['forgetting'] for r in results]):.4f}")
print(f"       acc={np.mean([r['accuracy'] for r in results]):.4f} +/- {np.std([r['accuracy'] for r in results]):.4f}")

with open('/home/juan/projeto-visao/mnist_split_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Salvo em: /home/juan/projeto-visao/mnist_split_results.json")
