"""
data_source.py — Tarefa 9.1: Fonte de dados real.

Integra com dados reais do Juan para alimentar o live loop.
Se não houver dados reais disponíveis, usa fallback sintético.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class RoutineDataSource:
    """Fonte de dados baseada na rotina do Juan.

    Usa dados de:
    - Horários de estudo (Super Productivity / rotina)
    - Sensores (se disponíveis)
    - Dados sintéticos como fallback
    """

    def __init__(self, use_real: bool = False):
        self.use_real = use_real
        self._step = 0
        self._fallback = SyntheticFallback()

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (x, y) para o cérebro aprender."""
        if self.use_real:
            data = self._try_real_data()
            if data is not None:
                return data

        return self._fallback.get(self._step)

    def _try_real_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Tenta obter dados reais (ex: rotina, sensores)."""
        # TODO: integrar com Super Productivity API, sensores, etc.
        return None


class SyntheticFallback:
    """Fallback sintético: gera dados não-estacionários."""

    def __init__(self):
        self._regimes = [
            ("sine_slow", 0.02),
            ("sine_fast", 0.08),
            ("saw", 0.0),
            ("mixed", 0.0),
        ]

    def get(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Gera um passo de dados."""
        # Alternar regimes a cada 1000 passos
        regime_idx = (step // 1000) % len(self._regimes)
        kind, freq = self._regimes[regime_idx]

        t = step
        if kind == "sine_slow":
            signal = np.sin(2 * np.pi * freq * t)
        elif kind == "sine_fast":
            signal = np.sin(2 * np.pi * freq * t)
        elif kind == "saw":
            signal = 2.0 * (t % 50) / 50.0 - 1.0
        else:
            signal = np.sin(2 * np.pi * 0.03 * t) + 0.5 * np.cos(2 * np.pi * 0.07 * t)

        x = np.array([signal, 0.3 * np.cos(2 * np.pi * 0.03 * t)])
        y = np.array([signal])
        return x, y


if __name__ == "__main__":
    source = RoutineDataSource()
    for i in range(10):
        x, y = source.get()
        print(f"step={i}: x={x.round(3)}, y={y.round(3)}")
