"""
memory_profile.py — Tarefa 8.4: Perfil de memória e otimização para hardware fraco.

Fornece:
  - @profile_memory: decorator para medir RAM/CPU durante inferência
  - MemoryProfiler: classe para medição detalhada de recursos
  - Precisão reduzida: float16/float8 para economizar memória
  - Modo eco: reduz número de neurônios para hardware fraco (celular, Raspberry Pi)

Uso:
    from visao.memory_profile import profile_memory, MemoryProfiler, EcoMode

    @profile_memory
    def minha_inferencia(x):
        return brain.forward(x)

    profiler = MemoryProfiler()
    with profiler.track():
        resultado = brain.forward(x)
    print(profiler.report())

    eco = EcoMode(brain, target_neurons=32)
    eco.enable()  # reduz neurônios
"""

from __future__ import annotations

import functools
import time
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ==============================================================
#  MEDIÇÃO DE RECURSOS
# ==============================================================

def _get_process() -> "psutil.Process":
    """Retorna o processo atual via psutil."""
    return psutil.Process(os.getpid())


def get_ram_usage_mb() -> float:
    """
    Retorna uso de RAM do processo atual em MB.

    Usa psutil se disponível, senão lê /proc/self/status (Linux).
    """
    if HAS_PSUTIL:
        return _get_process().memory_info().rss / (1024 * 1024)
    # Fallback: /proc/self/status (Linux)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024  # kB -> MB
    except (OSError, ValueError):
        pass
    return 0.0


def get_cpu_percent() -> float:
    """
    Retorna percentual de CPU do processo atual.

    Usa psutil se disponível, senão retorna 0.0.
    """
    if HAS_PSUTIL:
        return _get_process().cpu_percent(interval=None)
    return 0.0


# ==============================================================
#  DECORATOR @profile_memory
# ==============================================================

@dataclass
class MemoryProfile:
    """Resultado de uma medição de memória."""
    function_name: str
    ram_before_mb: float
    ram_after_mb: float
    ram_peak_mb: float
    cpu_percent: float
    duration_s: float
    extra: dict = field(default_factory=dict)

    @property
    def ram_delta_mb(self) -> float:
        return self.ram_after_mb - self.ram_before_mb

    def __repr__(self) -> str:
        return (
            f"MemoryProfile({self.function_name}: "
            f"ΔRAM={self.ram_delta_mb:+.2f}MB, "
            f"peak={self.ram_peak_mb:.2f}MB, "
            f"CPU={self.cpu_percent:.1f}%, "
            f"t={self.duration_s*1000:.2f}ms)"
        )


def profile_memory(func: Optional[Callable] = None, *, repeats: int = 1) -> Callable:
    """
    Decorator que mede RAM/CPU durante a execução de uma função.

    Usage:
        @profile_memory
        def f(x): ...

        @profile_memory(repeats=5)
        def g(x): ...

    Salva o resultado em `f.__memory_profile__` (última execução).
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            proc = _get_process() if HAS_PSUTIL else None

            ram_before = get_ram_usage_mb()
            cpu_before = get_cpu_percent()

            # Peak tracking via callback
            peak_ram = ram_before

            def update_peak():
                nonlocal peak_ram
                current = get_ram_usage_mb()
                if current > peak_ram:
                    peak_ram = current

            t0 = time.perf_counter()
            results = []
            for _ in range(repeats):
                result = fn(*args, **kwargs)
                results.append(result)
                update_peak()
            duration = time.perf_counter() - t0

            ram_after = get_ram_usage_mb()
            cpu_after = get_cpu_percent()

            profile = MemoryProfile(
                function_name=fn.__name__,
                ram_before_mb=ram_before,
                ram_after_mb=ram_after,
                ram_peak_mb=peak_ram,
                cpu_percent=cpu_after,
                duration_s=duration / repeats,
                extra={"repeats": repeats, "n_args": len(args)},
            )
            wrapper.__memory_profile__ = profile

            # Print summary
            print(f"📊 {profile}")

            return results[-1] if repeats == 1 else results
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ==============================================================
#  MemoryProfiler — gerenciador de contexto
# ==============================================================

class MemoryProfiler:
    """
    Gerenciador de contexto para tracking de memória.

    Usage:
        profiler = MemoryProfiler()
        with profiler.track():
            brain.forward(x)
        print(profiler.report())
    """

    def __init__(self, label: str = "block"):
        self.label = label
        self.records: list[MemoryProfile] = []

    def track(self, label: Optional[str] = None) -> "MemoryTracker":
        return MemoryTracker(self, label or self.label)

    def add(self, profile: MemoryProfile) -> None:
        self.records.append(profile)

    def report(self) -> str:
        if not self.records:
            return "MemoryProfiler: nenhum registro."
        total_ram = sum(r.ram_delta_mb for r in self.records)
        total_time = sum(r.duration_s for r in self.records)
        peak = max(r.ram_peak_mb for r in self.records)
        lines = [
            f"{'='*60}",
            f"MemoryProfiler: {self.label}",
            f"{'='*60}",
            f"  blocos:     {len(self.records)}",
            f"  ΔRAM total: {total_ram:+.2f} MB",
            f"  RAM pico:   {peak:.2f} MB",
            f"  tempo:      {total_time*1000:.2f} ms",
        ]
        for i, r in enumerate(self.records):
            lines.append(f"  [{i}] {r}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


class MemoryTracker:
    """Context manager para MemoryProfiler."""

    def __init__(self, profiler: MemoryProfiler, label: str):
        self.profiler = profiler
        self.label = label
        self.ram_before = 0.0
        self.t0 = 0.0
        self.peak = 0.0

    def __enter__(self) -> "MemoryTracker":
        self.ram_before = get_ram_usage_mb()
        self.peak = self.ram_before
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        duration = time.perf_counter() - self.t0
        ram_after = get_ram_usage_mb()
        self.peak = max(self.peak, ram_after)
        profile = MemoryProfile(
            function_name=self.label,
            ram_before_mb=self.ram_before,
            ram_after_mb=ram_after,
            ram_peak_mb=self.peak,
            cpu_percent=get_cpu_percent(),
            duration_s=duration,
        )
        self.profiler.add(profile)


# ==============================================================
#  PRECISÃO REDUZIDA (float16/float8)
# ==============================================================

# Mapeamento de dtype suportados
DTYPE_MAP = {
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
}

# float8 não é nativo em numpy; emular via scaling
try:
    np.float8_  # numpy 2.0+
    DTYPE_MAP["float8"] = np.float8_
except AttributeError:
    # Emular float8: uint8 com escala (quantização)
    pass


def reduce_precision(arr: np.ndarray, target_dtype: str = "float16") -> np.ndarray:
    """
    Reduz a precisão de um array para economizar memória.

    Para float8 em numpy < 2.0, usa quantização via uint8 com escala.

    Parameters
    ----------
    arr : array de entrada
    target_dtype: "float16" ou "float8"

    Returns
    -------
    array com precisão reduzida
    """
    if target_dtype == "float16":
        return arr.astype(np.float16)
    elif target_dtype == "float8":
        # Quantização: mapeia range para uint8 [0, 255]
        if arr.size == 0:
            return arr.astype(np.float16)  # fallback
        amin, amax = arr.min(), arr.max()
        scale = (amax - amin) / 255.0 if amax > amin else 1.0
        quantized = np.round((arr - amin) / scale).astype(np.uint8)
        # Retorna como float16 com metadados (mais portável que uint8 para cálculo)
        restored = quantized.astype(np.float16) * np.float16(scale) + np.float16(amin)
        return restored
    else:
        raise ValueError(f"dtype não suportado: {target_dtype}")


def memory_savings(original_dtype: str, target_dtype: str) -> float:
    """
    Retorna fator de economia de memória (ex: 2.0 = 50% menos).

    float64 -> float16 = 4x menos memória
    float64 -> float8  = 8x menos memória
    """
    bits = {
        "float64": 64,
        "float32": 32,
        "float16": 16,
        "float8": 8,
    }
    return bits.get(original_dtype, 64) / bits.get(target_dtype, 64)


# ==============================================================
#  MODO ECO — menos neurônios
# ==============================================================

class EcoMode:
    """
    Modo eco: reduz o número de neurônios para hardware fraco.

    Em vez de criar um brain novo, este wrapper projeta o estado
    e pesos para um subespaço menor via seleção de neurônios
    por importância (maior |A| = mais importante).

    Usage:
        eco = EcoMode(brain, target_neurons=32)
        eco.enable()
        # brain agora opera com menos neurônios
        eco.disable()
        # restaura original
    """

    def __init__(self, brain: Any, target_neurios: int = 32):
        """
        Parameters
        ----------
        brain: instância de VisaoBrain
        target_neurios: número alvo de neurônios (deve ser < n_hidden)
        """
        self.brain = brain
        self.target = min(target_neurios, brain.n_hidden)
        self._original = {}
        self._enabled = False
        self._active_indices: Optional[np.ndarray] = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Ativa modo eco: reduz neurônios."""
        if self._enabled:
            return
        brain = self.brain
        n = brain.n_hidden
        k = self.target

        # Selecionar neurônios mais importantes (por |A|)
        importance = np.abs(brain.cell.A)
        self._active_indices = np.argsort(importance)[-k:]

        # Salvar estado original
        self._original = {
            "n_hidden": n,
            "W_in": brain.cell.W_in.copy(),
            "W_rec": brain.cell.W_rec.copy(),
            "b": brain.cell.b.copy(),
            "A": brain.cell.A.copy(),
            "tau": brain.cell.tau.copy(),
            "mask": brain.cell.mask.copy(),
            "W_out": brain.learner.W_out.copy(),
            "x": brain.x.copy(),
            "omega": brain.learner.omega.copy(),
        }

        # Reduzir pesos
        idx = self._active_indices
        brain.cell.W_in = brain.cell.W_in[idx, :]
        brain.cell.W_rec = brain.cell.W_rec[idx, :][:, idx]
        brain.cell.b = brain.cell.b[idx]
        brain.cell.A = brain.cell.A[idx]
        brain.cell.tau = brain.cell.tau[idx]
        brain.cell.mask = brain.cell.mask[idx, :][:, idx]

        # Ajustar learner
        brain.learner.W_out = brain.learner.W_out[:, idx]
        brain.learner.omega = brain.learner.omega[:, idx]

        # Ajustar estado
        brain.x = brain.x[idx]
        brain.n_hidden = k

        self._enabled = True
        print(f"🌿 Modo ECO ativado: {n} → {k} neurônios "
              f"({(1-k/n)*100:.0f}% menos memória)")

    def disable(self) -> None:
        """Desativa modo eco: restaura neurônios originais."""
        if not self._enabled:
            return
        brain = self.brain
        orig = self._original

        brain.cell.W_in = orig["W_in"]
        brain.cell.W_rec = orig["W_rec"]
        brain.cell.b = orig["b"]
        brain.cell.A = orig["A"]
        brain.cell.tau = orig["tau"]
        brain.cell.mask = orig["mask"]
        brain.learner.W_out = orig["W_out"]
        brain.learner.omega = orig["omega"]
        brain.x = orig["x"]
        brain.n_hidden = orig["n_hidden"]

        self._enabled = False
        print(f"🌿 Modo ECO desativado: restaurado {orig['n_hidden']} neurônios")

    def memory_saved_mb(self) -> float:
        """Estimativa de memória economizada em MB."""
        if not self._enabled:
            return 0.0
        orig_n = self._original["n_hidden"]
        target_n = self.target
        # Pesos recorrentes: n_hidden^2
        saved_params = (orig_n ** 2 - target_n ** 2)
        # float64 = 8 bytes
        return saved_params * 8 / (1024 * 1024)


# ==============================================================
#  FUNÇÕES DE ALTO NÍVEL
# ==============================================================

def measure_inference(brain: Any, x: np.ndarray, n_repeats: int = 100) -> MemoryProfile:
    """
    Mede memória e tempo de uma inferência repetida.

    Parameters
    ----------
    brain: instância de VisaoBrain
    x: entrada (n_in,)
    n_repeats: número de repetições para média

    Returns
    -------
    MemoryProfile com resultados
    """
    ram_before = get_ram_usage_mb()
    t0 = time.perf_counter()

    for _ in range(n_repeats):
        brain.forward(x)

    duration = (time.perf_counter() - t0) / n_repeats
    ram_after = get_ram_usage_mb()

    return MemoryProfile(
        function_name="inference",
        ram_before_mb=ram_before,
        ram_after_mb=ram_after,
        ram_peak_mb=ram_after,
        cpu_percent=get_cpu_percent(),
        duration_s=duration,
        extra={"n_repeats": n_repeats},
    )


def estimate_memory_usage(n_in: int, n_hidden: int, n_out: int,
                          dtype: str = "float64") -> dict:
    """
    Estimativa teórica de memória para um brain.

    Returns
    -------
    dict com contagem de parâmetros e memória estimada.
    """
    bytes_per_param = {
        "float64": 8, "float32": 4, "float16": 2, "float8": 1,
    }.get(dtype, 8)

    # Pesos do reservatório
    w_in = n_hidden * n_in
    w_rec = n_hidden * n_hidden
    b = n_hidden
    a = n_hidden
    tau = n_hidden

    # Pesos do learner
    w_out = n_out * n_hidden
    b_out = n_out
    omega = n_out * n_hidden

    total_params = w_in + w_rec + b + a + tau + w_out + b_out + omega
    total_bytes = total_params * bytes_per_param

    return {
        "n_params": total_params,
        "bytes_per_param": bytes_per_param,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "breakdown": {
            "W_in": w_in, "W_rec": w_rec, "b": b, "A": a, "tau": tau,
            "W_out": w_out, "b_out": b_out, "omega": omega,
        },
    }


# ==============================================================
#  TESTE / DEMO
# ==============================================================

def run_demo():
    """Demonstração do módulo de memória."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from visao.brain import VisaoBrain

    print("=" * 60)
    print("DEMO: memory_profile.py — Tarefa 8.4")
    print("=" * 60)

    # 1. Criar brain
    brain = VisaoBrain(n_in=2, n_hidden=64, n_out=1, seed=42)
    x = np.array([0.1, -0.2])

    # 2. Medir inferência
    print("\n[1] Medindo inferência...")
    profile = measure_inference(brain, x, n_repeats=100)
    print(f"    {profile}")

    # 3. Estimativa de memória
    print("\n[2] Estimativa de memória (float64):")
    est = estimate_memory_usage(2, 64, 1, "float64")
    print(f"    params: {est['n_params']:,}")
    print(f"    memória: {est['total_mb']:.4f} MB")

    print("\n[3] Estimativa de memória (float16):")
    est16 = estimate_memory_usage(2, 64, 1, "float16")
    print(f"    memória: {est16['total_mb']:.4f} MB "
          f"({memory_savings('float64','float16'):.0f}x menos)")

    # 4. Modo eco
    print("\n[4] Modo ECO:")
    eco = EcoMode(brain, target_neurios=32)
    eco.enable()
    print(f"    memória economizada: {eco.memory_saved_mb():.4f} MB")
    eco.disable()

    # 5. Decorator
    print("\n[5] Decorator @profile_memory:")

    @profile_memory(repeats=10)
    def test_inference():
        return brain.forward(x)

    test_inference()

    print("\n" + "=" * 60)
    print("DEMO COMPLETO")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
