"""
live_loop.py — Tarefa 9.0: Loop vivo de operação contínua.

O cérebro VISÃO aprende continuamente, 24/7, sem intervenção humana.
Se não há dados reais disponíveis, usa dados sintéticos como fallback
(gerador de regimes variados).

Regras:
  - Nunca para (loop infinito com sleep adaptativo)
  - Auto-recuperação em caso de erro
  - Checkpoint periódico do cérebro
  - Log de métricas para monitoramento
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visao.brain import VisaoBrain

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "visao" / "ops" / "checkpoints"
LOG_DIR = ROOT / "visao" / "ops" / "logs"


class LiveLoop:
    """Loop vivo: aprendizado contínuo 24/7."""

    def __init__(
        self,
        brain: VisaoBrain | None = None,
        checkpoint_dir: str | Path = CHECKPOINT_DIR,
        log_dir: str | Path = LOG_DIR,
        checkpoint_interval: int = 1000,  # salvar a cada N passos
        sleep_on_idle: float = 0.1,  # segundos sem dados
    ):
        self.brain = brain or VisaoBrain(n_in=2, n_hidden=64, n_out=1, seed=42)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_interval = checkpoint_interval
        self.sleep_on_idle = sleep_on_idle

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._step = 0
        self._running = False
        self._error_count = 0
        self._last_checkpoint = 0

    def start(self, data_source=None):
        """Inicia o loop vivo (infinito)."""
        self._running = True
        print(f"[LiveLoop] Iniciando (brain={self.brain})")

        while self._running:
            try:
                # 1. Obter dados (real ou sintético)
                if data_source is not None:
                    x, y = data_source.get()
                else:
                    x, y = self._fallback_data()

                # 2. Aprender
                result = self.brain.learn(x, y)
                self._step += 1

                # 3. Checkpoint periódico
                if self._step - self._last_checkpoint >= self.checkpoint_interval:
                    self._checkpoint()
                    self._last_checkpoint = self._step

                # 4. Log
                if self._step % 100 == 0:
                    self._log(result)

            except KeyboardInterrupt:
                print("[LiveLoop] Interrompido pelo usuário")
                self._checkpoint()
                break
            except Exception as e:
                self._error_count += 1
                print(f"[LiveLoop] Erro no passo {self._step}: {e}")
                self._recover()

    def stop(self):
        """Para o loop."""
        self._running = False

    def _fallback_data(self):
        """Dados sintéticos quando não há fonte real."""
        t = self._step
        freq = 0.03 + 0.02 * np.sin(2 * np.pi * t / 10000)  # freq variável
        signal = np.sin(2 * np.pi * freq * t)
        x = np.array([signal, 0.3 * np.cos(2 * np.pi * freq * t)])
        y = np.array([signal])
        return x, y

    def _checkpoint(self):
        """Salva estado do cérebro."""
        path = self.checkpoint_dir / f"brain_step_{self._step}.json"
        self.brain.save(path)
        print(f"[LiveLoop] Checkpoint: {path}")

    def _log(self, result: dict):
        """Log de métricas."""
        log_file = self.log_dir / "live_loop.jsonl"
        entry = {
            "step": self._step,
            "err": result["err"],
            "surprise": result["surprise"],
            "lr": self.brain.lr,
            "time": time.time(),
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _recover(self):
        """Auto-recuperação: tenta restaurar do último checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("brain_step_*.json"))
        if checkpoints:
            last = checkpoints[-1]
            print(f"[LiveLoop] Recuperando de {last}")
            self.brain = VisaoBrain.load(last)
            self._step = int(last.stem.split("_")[-1])
        else:
            print("[LiveLoop] Sem checkpoint, reiniciando brain")
            self.brain = VisaoBrain(n_in=2, n_hidden=64, n_out=1, seed=42)
            self._step = 0


def run_live_loop(duration_seconds: int = 3600):
    """Roda o loop vivo por uma duração finita (para testes)."""
    loop = LiveLoop(checkpoint_interval=500)
    brain = loop.brain

    print(f"[Test] Rodando loop por {duration_seconds}s...")
    start = time.time()
    step = 0

    while time.time() - start < duration_seconds:
        # Dados sintéticos
        t = step
        freq = 0.03 + 0.02 * np.sin(2 * np.pi * t / 10000)
        signal = np.sin(2 * np.pi * freq * t)
        x = np.array([signal, 0.3 * np.cos(2 * np.pi * freq * t)])
        y = np.array([signal])

        result = brain.learn(x, y)
        step += 1

        if step % 500 == 0:
            print(f"  step={step}, err={result['err']:.4f}, surprise={result['surprise']:.3f}")

    print(f"[Test] Finalizado: {step} passos em {time.time() - start:.1f}s")
    return brain


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VISÃO Live Loop")
    parser.add_argument("--duration", type=int, default=60, help="Duração em segundos (0=infinito)")
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    args = parser.parse_args()

    if args.duration > 0:
        run_live_loop(args.duration)
    else:
        loop = LiveLoop(checkpoint_dir=args.checkpoint_dir)
        loop.start()
