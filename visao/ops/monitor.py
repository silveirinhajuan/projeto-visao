"""
monitor.py — Tarefa 9.2: Monitoramento de saúde do cérebro.

Coleta métricas do cérebro em tempo real e salva para análise:
  - Erro (janela deslizante)
  - Surpresa
  - lr (se meta-learn)
  - Uso de RAM
  - Passos por segundo
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np


class BrainMonitor:
    """Monitora saúde do cérebro VISÃO."""

    def __init__(self, window: int = 100, log_dir: str | Path | None = None):
        from visao.ops.live_loop import LOG_DIR
        self.log_dir = Path(log_dir) if log_dir else LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.window = window
        self.errors = deque(maxlen=window)
        self.surprises = deque(maxlen=window)
        self.lrs = deque(maxlen=window)
        self.timestamps = deque(maxlen=window)

    def update(self, result: dict, lr: float = 0.0):
        """Atiliza métricas com novo resultado."""
        self.errors.append(result["err"])
        self.surprises.append(result["surprise"])
        self.lrs.append(lr)
        self.timestamps.append(time.time())

    def get_stats(self) -> dict:
        """Retorna estatísticas atuais."""
        if not self.errors:
            return {"status": "starting"}

        errs = np.array(self.errors)
        surps = np.array(self.surprises)

        # Passos por segundo
        if len(self.timestamps) > 1:
            dt = self.timestamps[-1] - self.timestamps[0]
            pps = len(self.timestamps) / dt if dt > 0 else 0
        else:
            pps = 0

        return {
            "status": "running",
            "mse": float(np.mean(errs ** 2)),
            "mae": float(np.mean(errs)),
            "surprise_mean": float(np.mean(surps)),
            "surprise_std": float(np.std(surps)),
            "lr": float(self.lrs[-1]) if self.lrs else 0,
            "steps_per_sec": round(pps, 1),
            "window_size": len(self.errors),
        }

    def save_log(self, step: int):
        """Salva snapshot no log."""
        stats = self.get_stats()
        stats["step"] = step
        stats["time"] = time.time()

        log_file = self.log_dir / "monitor.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(stats) + "\n")

    def print_status(self, step: int):
        """Imprime status no console."""
        stats = self.get_stats()
        if stats["status"] == "running":
            print(f"[Monitor] step={step} mse={stats['mse']:.4f} "
                  f"surprise={stats['surprise_mean']:.3f} lr={stats['lr']:.4f} "
                  f"pps={stats['steps_per_sec']}")
