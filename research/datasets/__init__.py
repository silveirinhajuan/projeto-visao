"""Synthetic non-stationary dataset: sine → sawtooth → mixed.

Each task generates a 1D time-series prediction problem (predict next value).
The stream is concatenated: first all sine, then sawtooth, then mixed.
This creates a continual-learning scenario where the distribution shifts.
"""
from __future__ import annotations

import numpy as np


class NonStationaryStream:
    """Generates a non-stationary time-series stream.

    Task 1: sine wave (smooth, periodic)
    Task 2: sawtooth (sharp transitions, periodic)
    Task 3: mixed (sum of two sines — multi-frequency)

    Each sample is (x_t, y_t) where y_t = x_{t+1} (one-step-ahead prediction).
    """

    def __init__(self, seq_length: int = 200, seed: int = 0,
                 tasks: list[dict] | None = None):
        self.seq_length = seq_length
        self.rng = np.random.default_rng(seed)
        self.tasks = tasks or self._default_tasks()

    @staticmethod
    def _default_tasks() -> list[dict]:
        return [
            {"type": "sine", "frequency": 1.0, "n_samples": 1000},
            {"type": "sawtooth", "frequency": 2.0, "n_samples": 1000},
            {"type": "mixed", "frequencies": [1.0, 3.0], "n_samples": 1000},
        ]

    def _gen_sine(self, n: int, freq: float) -> np.ndarray:
        t = np.arange(n + self.seq_length + 1)
        phase = self.rng.uniform(0, 2 * np.pi)
        signal = np.sin(2 * np.pi * freq * t / 50.0 + phase)
        return signal

    def _gen_sawtooth(self, n: int, freq: float) -> np.ndarray:
        t = np.arange(n + self.seq_length + 1)
        phase = self.rng.uniform(0, 2 * np.pi)
        signal = 2.0 * ((t / (50.0 / freq) + phase / (2 * np.pi)) % 1.0) - 1.0
        return signal

    def _gen_mixed(self, n: int, frequencies: list[float]) -> np.ndarray:
        t = np.arange(n + self.seq_length + 1)
        signal = np.zeros(n + self.seq_length + 1)
        for f in frequencies:
            phase = self.rng.uniform(0, 2 * np.pi)
            signal += np.sin(2 * np.pi * f * t / 50.0 + phase)
        signal /= len(frequencies)
        return signal

    def generate(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Generate the full stream.

        Returns:
            X: (N, seq_length, 1) input sequences
            Y: (N, seq_length, 1) target sequences
            info: dict with task boundaries
        """
        all_signals = []
        task_boundaries = []
        cursor = 0

        for task in self.tasks:
            cursor += task.get("n_samples", 1000)
            task_boundaries.append(cursor)

            if task["type"] == "sine":
                sig = self._gen_sine(task["n_samples"], task["frequency"])
            elif task["type"] == "sawtooth":
                sig = self._gen_sawtooth(task["n_samples"], task["frequency"])
            elif task["type"] == "mixed":
                sig = self._gen_mixed(task["n_samples"], task["frequencies"])
            else:
                raise ValueError(f"Unknown task type: {task['type']}")
            all_signals.append(sig)

        signal = np.concatenate(all_signals)
        N = len(signal) - self.seq_length

        # Normalize to [-1, 1]
        mx = np.max(np.abs(signal))
        if mx > 0:
            signal /= mx

        # Build sequences: x_t = signal[t], y_t = signal[t+1]
        X = np.empty((N, self.seq_length, 1), dtype=np.float64)
        Y = np.empty((N, self.seq_length, 1), dtype=np.float64)
        for i in range(N):
            X[i, :, 0] = signal[i: i + self.seq_length]
            Y[i, :, 0] = signal[i + 1: i + 1 + self.seq_length]

        info = {
            "task_boundaries": task_boundaries,
            "total_samples": N,
            "seq_length": self.seq_length,
        }
        return X, Y, info

    def generate_flat(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Generate as flat (N, 1) samples for online learning."""
        X, Y, info = self.generate()
        N = X.shape[0]
        # Flatten: each sample is one timestep
        X_flat = X.reshape(N * self.seq_length, 1)
        Y_flat = Y.reshape(N * self.seq_length, 1)
        info["total_samples"] = N * self.seq_length
        return X_flat, Y_flat, info
