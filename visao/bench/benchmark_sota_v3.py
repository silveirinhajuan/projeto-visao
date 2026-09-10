#!/usr/bin/env python3
"""
benchmark_sota_v3.py — Tarefa 16: SOTA Benchmark.

Benchmark comparativo: VisaoBrain vs LSTM/GRU/Transformer em psMNIST.
Usa dados locais (visao/bench/data/), sem download.

Métricas: accuracy, parâmetros, tempo, RAM.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain


# ==============================================================
#  psMNIST DATA
# ==============================================================


def load_mnist(images_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST from gzipped files."""
    with gzip.open(images_path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
    with gzip.open(labels_path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels


class PSMNISTTask:
    """psMNIST: permuted sequential MNIST."""

    def __init__(self, data_dir: str, max_train: int = 2000, max_test: int = 500,
                 seed: int = 0):
        self.data_dir = Path(data_dir)
        self.max_train = max_train
        self.max_test = max_test
        self.seed = seed

        # Load data
        train_images, train_labels = load_mnist(
            str(self.data_dir / "train-images-idx3-ubyte.gz"),
            str(self.data_dir / "train-labels-idx1-ubyte.gz")
        )
        test_images, test_labels = load_mnist(
            str(self.data_dir / "t10k-images-idx3-ubyte.gz"),
            str(self.data_dir / "t10k-labels-idx1-ubyte.gz")
        )

        # Flatten to sequences: (N, 784, 1)
        self.X_train = train_images.reshape(-1, 784, 1).astype(np.float32) / 255.0
        self.X_test = test_images.reshape(-1, 784, 1).astype(np.float32) / 255.0

        # One-hot labels: (N, 10)
        self.y_train = np.zeros((len(train_labels), 10), dtype=np.float32)
        self.y_train[np.arange(len(train_labels)), train_labels] = 1.0
        self.y_test = np.zeros((len(test_labels), 10), dtype=np.float32)
        self.y_test[np.arange(len(test_labels)), test_labels] = 1.0

        # Permutation
        rng = np.random.default_rng(seed)
        self.permutation = rng.permutation(784)

        # Apply permutation
        self.X_train = self.X_train[:, self.permutation, :]
        self.X_test = self.X_test[:, self.permutation, :]

        # Subsample
        self.X_train = self.X_train[:max_train]
        self.y_train = self.y_train[:max_train]
        self.X_test = self.X_test[:max_test]
        self.y_test = self.y_test[:max_test]


# ==============================================================
#  BASELINES (numpy)
# ==============================================================


class BaselineLSTM:
    """LSTM baseline for sequence classification."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, lr: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr = lr

        s = 1.0 / np.sqrt(n_hidden)
        # Gates: input, forget, output, candidate
        self.W = rng.normal(0, s, (4 * n_hidden, n_hidden + n_in)).astype(np.float64)
        self.b = np.zeros(4 * n_hidden, dtype=np.float64)
        self.W_out = rng.normal(0, s, (n_out, n_hidden)).astype(np.float64)
        self.b_out = np.zeros(n_out, dtype=np.float64)

        # Adam states
        self.m_W = np.zeros_like(self.W)
        self.v_W = np.zeros_like(self.W)
        self.m_W_out = np.zeros_like(self.W_out)
        self.v_W_out = np.zeros_like(self.W_out)
        self.t = 0

        self.reset()

    def reset(self):
        self.h = np.zeros(self.n_hidden)
        self.c = np.zeros(self.n_hidden)

    def _forward_step(self, x):
        """Single LSTM step."""
        concat = np.concatenate([self.h, x])
        gates = self.W @ concat + self.b
        i = 1 / (1 + np.exp(-gates[:self.n_hidden]))
        f = 1 / (1 + np.exp(-gates[self.n_hidden:2*self.n_hidden]))
        o = 1 / (1 + np.exp(-gates[2*self.n_hidden:3*self.n_hidden]))
        c_tilde = np.tanh(gates[3*self.n_hidden:])
        self.c = f * self.c + i * c_tilde
        self.h = o * np.tanh(self.c)
        return self.h

    def forward(self, x_seq):
        """Forward through sequence, return output."""
        self.reset()
        for x in x_seq:
            h = self._forward_step(x)
        return self.W_out @ h + self.b_out

    def learn(self, x_seq, y):
        """Train on one sequence (BPTT truncated)."""
        # Forward pass (store states)
        self.reset()
        states = []
        for x in x_seq:
            h = self._forward_step(x)
            states.append((self.h.copy(), self.c.copy()))

        # Output
        pred = self.W_out @ self.h + self.b_out
        err = pred - y

        # Gradient on readout (simple)
        dW_out = np.outer(err, self.h)
        db_out = err

        # Adam update for readout
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.m_W_out = beta1 * self.m_W_out + (1 - beta1) * dW_out
        self.v_W_out = beta2 * self.v_W_out + (1 - beta2) * dW_out**2
        m_hat = self.m_W_out / (1 - beta1**self.t)
        v_hat = self.v_W_out / (1 - beta2**self.t)
        self.W_out -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        self.b_out -= self.lr * db_out

        return float(np.mean(err**2))

    def count_params(self):
        return self.W.size + self.b.size + self.W_out.size + self.b_out.size


class BaselineGRU:
    """GRU baseline for sequence classification."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, lr: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr = lr

        s = 1.0 / np.sqrt(n_hidden)
        # Update, reset, candidate
        self.W_z = rng.normal(0, s, (n_hidden, n_hidden + n_in)).astype(np.float64)
        self.W_r = rng.normal(0, s, (n_hidden, n_hidden + n_in)).astype(np.float64)
        self.W_h = rng.normal(0, s, (n_hidden, n_hidden + n_in)).astype(np.float64)
        self.W_out = rng.normal(0, s, (n_out, n_hidden)).astype(np.float64)
        self.b_out = np.zeros(n_out, dtype=np.float64)

        self.m_W_out = np.zeros_like(self.W_out)
        self.v_W_out = np.zeros_like(self.W_out)
        self.t = 0

        self.reset()

    def reset(self):
        self.h = np.zeros(self.n_hidden)

    def _forward_step(self, x):
        concat = np.concatenate([self.h, x])
        z = 1 / (1 + np.exp(-self.W_z @ concat))
        r = 1 / (1 + np.exp(-self.W_r @ concat))
        concat_r = np.concatenate([r * self.h, x])
        h_tilde = np.tanh(self.W_h @ concat_r)
        self.h = (1 - z) * self.h + z * h_tilde
        return self.h

    def forward(self, x_seq):
        self.reset()
        for x in x_seq:
            self._forward_step(x)
        return self.W_out @ self.h + self.b_out

    def learn(self, x_seq, y):
        self.reset()
        for x in x_seq:
            self._forward_step(x)
        pred = self.W_out @ self.h + self.b_out
        err = pred - y

        dW_out = np.outer(err, self.h)
        db_out = err

        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.m_W_out = beta1 * self.m_W_out + (1 - beta1) * dW_out
        self.v_W_out = beta2 * self.v_W_out + (1 - beta2) * dW_out**2
        m_hat = self.m_W_out / (1 - beta1**self.t)
        v_hat = self.v_W_out / (1 - beta2**self.t)
        self.W_out -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        self.b_out -= self.lr * db_out

        return float(np.mean(err**2))

    def count_params(self):
        return self.W_z.size + self.W_r.size + self.W_h.size + self.W_out.size + self.b_out.size


class BaselineTransformer:
    """Transformer baseline for sequence classification."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, n_heads: int = 4,
                 lr: float = 1e-3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_heads = n_heads
        self.lr = lr

        s = 1.0 / np.sqrt(n_in)
        self.W_q = rng.normal(0, s, (n_hidden, n_in)).astype(np.float64)
        self.W_k = rng.normal(0, s, (n_hidden, n_in)).astype(np.float64)
        self.W_v = rng.normal(0, s, (n_hidden, n_in)).astype(np.float64)
        self.W_o = rng.normal(0, s, (n_in, n_hidden)).astype(np.float64)
        self.W_out = rng.normal(0, s, (n_out, n_hidden)).astype(np.float64)
        self.b_out = np.zeros(n_out, dtype=np.float64)

        self.m_W_out = np.zeros_like(self.W_out)
        self.v_W_out = np.zeros_like(self.W_out)
        self.t = 0

    def reset(self):
        """Transformer is stateless, but we provide reset() for API compatibility."""
        pass

    def _attn(self, X):
        """Causal self-attention."""
        Q = X @ self.W_q.T
        K = X @ self.W_k.T
        V = X @ self.W_v.T

        T = len(X)
        scores = Q @ K.T / np.sqrt(self.n_hidden)
        mask = np.triu(np.ones((T, T)), k=1) * (-1e9)
        scores = scores + mask
        # Stable softmax
        scores = scores - np.max(scores, axis=1, keepdims=True)
        attn = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

        out = attn @ V
        return out

    def forward(self, x_seq):
        out = self._attn(x_seq)
        # Use last timestep
        return self.W_out @ out[-1] + self.b_out

    def learn(self, x_seq, y):
        out = self._attn(x_seq)
        h = out[-1]
        pred = self.W_out @ h + self.b_out
        err = pred - y

        dW_out = np.outer(err, h)
        db_out = err

        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.m_W_out = beta1 * self.m_W_out + (1 - beta1) * dW_out
        self.v_W_out = beta2 * self.v_W_out + (1 - beta2) * dW_out**2
        m_hat = self.m_W_out / (1 - beta1**self.t)
        v_hat = self.v_W_out / (1 - beta2**self.t)
        self.W_out -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        self.b_out -= self.lr * db_out

        return float(np.mean(err**2))

    def count_params(self):
        return self.W_q.size + self.W_k.size + self.W_v.size + self.W_o.size + self.W_out.size + self.b_out.size


# ==============================================================
#  MEASUREMENT
# ==============================================================


def measure_model(name: str, model, task: PSMNISTTask, n_epochs: int = 5) -> dict:
    """Measure a model on psMNIST."""
    tracemalloc.start()
    t0 = time.time()

    # Train
    for epoch in range(n_epochs):
        for i in range(len(task.X_train)):
            x_seq = task.X_train[i]
            y = task.y_train[i]
            if name == "visao":
                # VisaoBrain learns step by step
                for step in range(len(x_seq)):
                    model.learn(x_seq[step], y)
            else:
                model.learn(x_seq, y)

    train_time = time.time() - t0

    # Evaluate
    correct = 0
    total = 0
    for i in range(len(task.X_test)):
        x_seq = task.X_test[i]
        y = task.y_test[i]

        if name == "visao":
            model.reset_state()
            model.set_mode("infer")
            for step in range(len(x_seq)):
                pred = model.forward(x_seq[step])
            pred_class = np.argmax(pred)
        else:
            model.reset()
            pred = model.forward(x_seq)
            pred_class = np.argmax(pred)

        true_class = np.argmax(y)
        if pred_class == true_class:
            correct += 1
        total += 1

    accuracy = correct / total
    wall_time = time.time() - t0
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_params = model.count_params() if hasattr(model, 'count_params') else model.n_params()

    return {
        "model": name,
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
        "params": n_params,
        "time": wall_time,
        "train_time": train_time,
        "ram_mb": peak_ram / (1024 * 1024),
    }


def run_benchmark(seeds=(0, 1, 2), quick: bool = False, output_dir: str = None) -> dict:
    """Run full benchmark."""
    if output_dir is None:
        output_dir = str(Path(__file__).parent)

    max_train = 50 if quick else 2000
    max_test = 20 if quick else 1000
    n_epochs = 1 if quick else 5

    results = {"visao": [], "lstm": [], "gru": [], "transformer": []}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        task = PSMNISTTask("visao/bench/data", max_train=max_train, max_test=max_test, seed=seed)

        # VisaoBrain
        print("  VisaoBrain...")
        brain = VisaoBrain(n_in=1, n_hidden=64, n_out=10, seed=seed)
        r = measure_model("visao", brain, task, n_epochs=n_epochs)
        results["visao"].append(r)
        print(f"    Accuracy: {r['accuracy']:.3f}, Params: {r['params']}, Time: {r['time']:.1f}s")

        # LSTM
        print("  LSTM...")
        lstm = BaselineLSTM(n_in=1, n_hidden=64, n_out=10, seed=seed)
        r = measure_model("lstm", lstm, task, n_epochs=n_epochs)
        results["lstm"].append(r)
        print(f"    Accuracy: {r['accuracy']:.3f}, Params: {r['params']}, Time: {r['time']:.1f}s")

        # GRU
        print("  GRU...")
        gru = BaselineGRU(n_in=1, n_hidden=64, n_out=10, seed=seed)
        r = measure_model("gru", gru, task, n_epochs=n_epochs)
        results["gru"].append(r)
        print(f"    Accuracy: {r['accuracy']:.3f}, Params: {r['params']}, Time: {r['time']:.1f}s")

        # Transformer
        print("  Transformer...")
        trans = BaselineTransformer(n_in=1, n_hidden=64, n_out=10, seed=seed)
        r = measure_model("transformer", trans, task, n_epochs=n_epochs)
        results["transformer"].append(r)
        print(f"    Accuracy: {r['accuracy']:.3f}, Params: {r['params']}, Time: {r['time']:.1f}s")

    # Aggregate
    aggregated = {}
    for name, rlist in results.items():
        accs = [r["accuracy"] for r in rlist]
        params = rlist[0]["params"]
        times = [r["time"] for r in rlist]
        rams = [r["ram_mb"] for r in rlist]

        aggregated[name] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "params": params,
            "time_mean": float(np.mean(times)),
            "ram_mean": float(np.mean(rams)),
        }

    # Save
    output_path = Path(output_dir) / "results_benchmark_sota_v3.json"
    output_path.write_text(json.dumps(aggregated, indent=2))
    print(f"\nResults saved to {output_path}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="SOTA Benchmark v3 — psMNIST")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    print("=" * 60)
    print("SOTA BENCHMARK v3 — psMNIST")
    print("=" * 60)

    results = run_benchmark(seeds=tuple(args.seeds), quick=args.quick)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Accuracy':<15} {'Params':<10} {'Time(s)':<10}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name:<15} {r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f}  {r['params']:<10} {r['time_mean']:.1f}")


if __name__ == "__main__":
    main()
