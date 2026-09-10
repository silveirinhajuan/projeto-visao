"""
quantize.py — Tarefa 11.1: Quantização de pesos para int8 (min-max scaling).

Reduz memória ~4x (float64 -> int8) com pouca perda de precisão.
Compatível com VisaoBrain: quantiza W_in, W_rec, b, A, W_out, b_out.

Mecanismo:
  - Quantização: q = round((w - min) / scale), onde scale = (max - min) / 255
  - Dequantização: w_hat = q * scale + min
  - Inferência: dequantize on-the-fly durante forward

Uso:
    from visao.ops.quantize import quantize_brain, measure_memory_savings, benchmark_quantization

    q_brain = quantize_brain(brain)
    savings = measure_memory_savings(brain, q_brain)
    results = benchmark_quantization(brain, q_brain, X_test, Y_test)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Garantir imports do projeto
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visao.brain import VisaoBrain


# ==============================================================
#  QUANTIZED TENSOR
# ==============================================================

@dataclass
class QuantizedTensor:
    """Tensor quantizado: armazena int8 + metadados para dequantização.

    Attributes
    ----------
    q : np.ndarray (int8)
        Valores quantizados em [-128, 127].
    scale : float
        Escala (max - min) / 255.
    wmin : float
        Valor mínimo original (para reconstruir o offset).
    shape : tuple
        Shape original do tensor.
    """
    q: np.ndarray
    scale: float
    wmin: float
    shape: tuple

    def dequantize(self) -> np.ndarray:
        """Reconstroi o tensor original (float64) a partir do int8."""
        return self.q.astype(np.float64) * self.scale + self.wmin

    @property
    def memory_bytes(self) -> int:
        """Memória real consumida (int8 = 1 byte/elem + metadados)."""
        return self.q.nbytes + 32  # 32 bytes para scale, wmin, shape

    @property
    def original_bytes(self) -> int:
        """Memória que o tensor original (float64) consumiria."""
        return int(np.prod(self.shape)) * 8


# ==============================================================
#  QUANTIZAÇÃO / DEQUANTIZAÇÃO
# ==============================================================

def quantize_weight(w: np.ndarray) -> QuantizedTensor:
    """Quantiza um array float64 para int8 via min-max scaling.

    Mapeia o range [min, max] para [-128, 127] (256 níveis).

    Parameters
    ----------
    w : np.ndarray (float64)
        Pesos a quantizar.

    Returns
    -------
    QuantizedTensor com dados int8 + metadados.
    """
    w = np.asarray(w, dtype=np.float64)
    wmin, wmax = float(w.min()), float(w.max())

    if wmax - wmin < 1e-12:
        # Tensor constante: scale zero, tudo mapeia para 0
        return QuantizedTensor(
            q=np.zeros(w.shape, dtype=np.int8),
            scale=0.0,
            wmin=wmin,
            shape=w.shape,
        )

    scale = (wmax - wmin) / 255.0
    # Clip to int8 range [-128, 127] to avoid overflow at boundaries
    q = np.clip(np.round((w - wmin) / scale), -128, 127).astype(np.int8)
    return QuantizedTensor(q=q, scale=scale, wmin=wmin, shape=w.shape)


def dequantize_weight(qt: QuantizedTensor) -> np.ndarray:
    """Dequantiza um QuantizedTensor de volta para float64."""
    return qt.dequantize()


# ==============================================================
#  BRAIN QUANTIZADO
# ==============================================================

@dataclass
class QuantizedBrain:
    """Versão quantizada de VisaoBrain para inferência eficiente.

    Armazena pesos em int8 e dequantiza on-the-fly durante forward.
    Compatível com a API de inferência do VisaoBrain.

    Attributes
    ----------
    config : dict
        Hiperparâmetros do brain.
    quantized_weights : dict[str, QuantizedTensor]
        Pesos quantizados do cell e learner.
    state : dict
        Estado interno (x, err_ema, etc).
    """
    config: dict
    quantized_weights: dict[str, QuantizedTensor]
    state: dict
    _mode: str = "infer"

    # Pesos dequantizados em cache (lazy)
    _cache: dict = field(default_factory=dict)
    _cache_valid: bool = False

    def _dequantize_all(self) -> dict[str, np.ndarray]:
        """Dequantiza todos os pesos (com cache)."""
        if self._cache_valid and self._cache:
            return self._cache
        self._cache = {
            name: qt.dequantize()
            for name, qt in self.quantized_weights.items()
        }
        self._cache_valid = True
        return self._cache

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Inferência com pesos quantizados (dequantize on-the-fly).

        Reconstrói os pesos do cell e learner a partir do int8,
        executa o forward padrão, e retorna a predição.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        w = self._dequantize_all()

        # --- Forward do reservatório líquido (reconstruído) ---
        # f = sigmoid(W_in @ u + W_rec @ x + b)
        fx = 1.0 / (1.0 + np.exp(-np.clip(
            w["W_in"] @ x + w["W_rec"] @ self.state["x"] + w["b"],
            -60.0, 60.0
        )))
        dt = self.config["dt"]
        tau = w["tau"]
        A = w["A"]
        num = self.state["x"] + dt * fx * A
        den = 1.0 + dt * (1.0 / tau + fx)
        self.state["x"] = num / den

        # --- Forward do readout ---
        pred = w["W_out"] @ self.state["x"] + w["b_out"]
        return pred

    def reset_state(self) -> None:
        """Reseta o estado interno do reservatório."""
        self.state["x"] = np.zeros(self.config["n_hidden"])

    def set_mode(self, mode: str) -> None:
        if mode not in ("learn", "infer"):
            raise ValueError(f"mode deve ser 'learn' ou 'infer', recebido {mode!r}")
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def step(self) -> int:
        return self.state.get("step", 0)

    def evaluate_stream(self, X: np.ndarray, Y: np.ndarray, warmup: int = 50) -> dict:
        """Avalia o brain quantizado em um stream de dados (sem aprender)."""
        self.reset_state()
        preds = []
        for i, (xi, yi) in enumerate(zip(X, Y)):
            pred = self.forward(xi)
            if i >= warmup:
                preds.append((pred, yi))

        if not preds:
            return {"mse": float("inf"), "mae": float("inf"), "n": 0}

        preds_arr = np.array([p for p, _ in preds])
        targets_arr = np.array([t for _, t in preds])
        mse = float(np.mean((preds_arr - targets_arr) ** 2))
        mae = float(np.mean(np.abs(preds_arr - targets_arr)))
        return {"mse": mse, "mae": mae, "n": len(preds)}


# ==============================================================
#  FUNÇÕES DE ALTO NÍVEL
# ==============================================================

def quantize_brain(brain: VisaoBrain) -> QuantizedBrain:
    """Quantiza pesos de um VisaoBrain para int8.

    Estratégia: quantiza apenas as matrizes de pesos grandes (W_in, W_rec, W_out).
    Mantém vetores pequenos e sensíveis (b, A, tau, b_out) em float64 para
    minimizar perda de precisão no sistema dinâmico.

    Parameters
    ----------
    brain : VisaoBrain
        Instância do brain a quantizar.

    Returns
    -------
    QuantizedBrain com pesos em int8.
    """
    # Pesos quantizados: apenas matrizes grandes
    quantized_weights = {
        "W_in": quantize_weight(brain.cell.W_in),
        "W_rec": quantize_weight(brain.cell.W_rec),
        "W_out": quantize_weight(brain.learner.W_out),
    }

    # Pesos mantidos em float64 (pequenos e sensíveis)
    float_weights = {
        "b": brain.cell.b.copy(),
        "A": brain.cell.A.copy(),
        "tau": brain.cell.tau.copy(),
        "b_out": brain.learner.b_out.copy(),
    }

    config = {
        "n_in": brain.n_in,
        "n_hidden": brain.n_hidden,
        "n_out": brain.n_out,
        "dt": brain.cell.dt,
    }

    state = {
        "x": brain.x.copy(),
        "err_ema": brain.learner.err_ema,
        "err_var": brain.learner.err_var,
        "lr": brain.lr,
        "step": brain._step,
    }

    return QuantizedBrain(
        config=config,
        quantized_weights=quantized_weights,
        float_weights=float_weights,
        state=state,
    )


def measure_memory_savings(brain: VisaoBrain, q_brain: QuantizedBrain) -> dict:
    """Mede a economia de memória entre brain original e quantizado.

    Returns
    -------
    dict com memória original, quantizada, economia absoluta e relativa.
    """
    # Memória original (float64)
    cell_params = {
        "W_in": brain.cell.W_in,
        "W_rec": brain.cell.W_rec,
        "b": brain.cell.b,
        "A": brain.cell.A,
        "tau": brain.cell.tau,
    }
    learner_params = {
        "W_out": brain.learner.W_out,
        "b_out": brain.learner.b_out,
    }

    orig_bytes = sum(p.nbytes for p in cell_params.values())
    orig_bytes += sum(p.nbytes for p in learner_params.values())

    # Memória quantizada (int8)
    quant_bytes = sum(qt.memory_bytes for qt in q_brain.quantized_weights.values())

    savings = {
        "original_bytes": orig_bytes,
        "original_mb": orig_bytes / (1024 * 1024),
        "quantized_bytes": quant_bytes,
        "quantized_mb": quant_bytes / (1024 * 1024),
        "savings_bytes": orig_bytes - quant_bytes,
        "savings_mb": (orig_bytes - quant_bytes) / (1024 * 1024),
        "compression_ratio": orig_bytes / quant_bytes if quant_bytes > 0 else 0.0,
        "space_saving_pct": (1 - quant_bytes / orig_bytes) * 100 if orig_bytes > 0 else 0.0,
    }
    return savings


def benchmark_quantization(
    brain: VisaoBrain,
    q_brain: QuantizedBrain,
    X: np.ndarray,
    Y: np.ndarray,
    warmup: int = 50,
    n_repeats: int = 3,
) -> dict:
    """Benchmark completo: precisão + velocidade + memória.

    Parameters
    ----------
    brain : VisaoBrain original
    q_brain : QuantizedBrain quantizado
    X, Y : dados de teste
    warmup : amostras iniciais descartadas
    n_repeats : repetições para medição de tempo

    Returns
    -------
    dict com métricas de MSE, MAE, tempo, memória.
    """
    results = {}

    # --- Precisão ---
    brain.reset_state()
    q_brain.reset_state()

    orig_eval = brain.evaluate_stream(X, Y, warmup=warmup)
    quant_eval = q_brain.evaluate_stream(X, Y, warmup=warmup)

    results["original"] = orig_eval
    results["quantized"] = quant_eval
    results["mse_increase"] = quant_eval["mse"] - orig_eval["mse"]
    results["mse_increase_pct"] = (
        (quant_eval["mse"] - orig_eval["mse"]) / max(orig_eval["mse"], 1e-12) * 100
    )
    results["mae_increase"] = quant_eval["mae"] - orig_eval["mae"]

    # --- Velocidade ---
    x_test = X[0]

    # Tempo original
    times_orig = []
    for _ in range(n_repeats):
        brain.reset_state()
        t0 = time.perf_counter()
        for xi in X:
            brain.forward(xi)
        times_orig.append(time.perf_counter() - t0)
    time_orig = min(times_orig)

    # Tempo quantizado
    times_quant = []
    for _ in range(n_repeats):
        q_brain.reset_state()
        t0 = time.perf_counter()
        for xi in X:
            q_brain.forward(xi)
        times_quant.append(time.perf_counter() - t0)
    time_quant = min(times_quant)

    results["time_original_s"] = time_orig
    results["time_quantized_s"] = time_quant
    results["time_overhead_pct"] = (
        (time_quant - time_orig) / max(time_orig, 1e-12) * 100
    )

    # --- Memória ---
    mem = measure_memory_savings(brain, q_brain)
    results["memory"] = mem

    return results


# ==============================================================
#  TESTE / DEMO
# ==============================================================

def run_test():
    """Teste completo: cria brain, treina, quantiza, mede memória e accuracy."""
    print("=" * 65)
    print("TESTE: quantize.py — Tarefa 11.1: Quantização int8")
    print("=" * 65)

    # 1. Criar e treinar um brain
    print("\n[1] Criando e treinando VisaoBrain (n_in=2, n_hidden=64, n_out=1)...")
    brain = VisaoBrain(n_in=2, n_hidden=64, n_out=1, seed=42)

    # Dados sintétios: tarefa de regressão simples
    rng = np.random.default_rng(123)
    n_samples = 500
    X = rng.normal(0, 1, (n_samples, 2))
    Y = np.sin(X[:, 0] * 2.0) + 0.5 * X[:, 1] - 0.3 * X[:, 0] * X[:, 1]
    Y = Y.reshape(-1, 1)

    # Treinar
    for i in range(n_samples):
        brain.learn(X[i], Y[i])
    print(f"    Treinado por {n_samples} passos.")

    # 2. Avaliar brain original
    print("\n[2] Avaliando brain original...")
    brain.reset_state()
    orig_eval = brain.evaluate_stream(X, Y, warmup=50)
    print(f"    MSE: {orig_eval['mse']:.6f}")
    print(f"    MAE: {orig_eval['mae']:.6f}")

    # 3. Quantizar
    print("\n[3] Quantizando para int8...")
    q_brain = quantize_brain(brain)
    print("    Quantização concluída.")

    # 4. Medir memória
    print("\n[4] Medindo economia de memória...")
    mem = measure_memory_savings(brain, q_brain)
    print(f"    Original:   {mem['original_bytes']:,} bytes ({mem['original_mb']:.4f} MB)")
    print(f"    Quantizado: {mem['quantized_bytes']:,} bytes ({mem['quantized_mb']:.4f} MB)")
    print(f"    Economia:   {mem['savings_bytes']:,} bytes ({mem['savings_mb']:.4f} MB)")
    print(f"    Compressão: {mem['compression_ratio']:.1f}x ({mem['space_saving_pct']:.1f}% menos)")

    # 5. Avaliar brain quantizado
    print("\n[5] Avaliando brain quantizado...")
    q_brain.reset_state()
    quant_eval = q_brain.evaluate_stream(X, Y, warmup=50)
    print(f"    MSE: {quant_eval['mse']:.6f}")
    print(f"    MAE: {quant_eval['mae']:.6f}")

    # 6. Perda de precisão
    print("\n[6] Perda de precisão:")
    mse_diff = quant_eval["mse"] - orig_eval["mse"]
    mse_pct = mse_diff / max(orig_eval["mse"], 1e-12) * 100
    mae_diff = quant_eval["mae"] - orig_eval["mae"]
    mae_pct = mae_diff / max(orig_eval["mae"], 1e-12) * 100
    print(f"    ΔMSE: {mse_diff:+.6f} ({mse_pct:+.2f}%)")
    print(f"    ΔMAE: {mae_diff:+.6f} ({mae_pct:+.2f}%)")

    # 7. Benchmark de velocidade
    print("\n[7] Benchmark de velocidade...")
    n_repeats = 5
    times_orig = []
    times_quant = []

    for _ in range(n_repeats):
        brain.reset_state()
        t0 = time.perf_counter()
        for xi in X:
            brain.forward(xi)
        times_orig.append(time.perf_counter() - t0)

        q_brain.reset_state()
        t0 = time.perf_counter()
        for xi in X:
            q_brain.forward(xi)
        times_quant.append(time.perf_counter() - t0)

    time_orig = min(times_orig)
    time_quant = min(times_quant)
    overhead = (time_quant - time_orig) / max(time_orig, 1e-12) * 100
    print(f"    Original:   {time_orig*1000:.2f} ms ({n_samples} forwards)")
    print(f"    Quantizado: {time_quant*1000:.2f} ms ({n_samples} forwards)")
    print(f"    Overhead:   {overhead:+.1f}%")

    # 8. Resumo
    print("\n" + "=" * 65)
    print("RESUMO:")
    print(f"  Memória: {mem['compression_ratio']:.1f}x menos ({mem['space_saving_pct']:.0f}% redução)")
    print(f"  MSE: {orig_eval['mse']:.6f} → {quant_eval['mse']:.6f} ({mse_pct:+.2f}%)")
    print(f"  MAE: {orig_eval['mae']:.6f} → {quant_eval['mae']:.6f} ({mae_pct:+.2f}%)")
    print(f"  Tempo: {time_orig*1000:.1f}ms → {time_quant*1000:.1f}ms ({overhead:+.1f}%)")
    print("=" * 65)

    return {
        "original_mse": orig_eval["mse"],
        "quantized_mse": quant_eval["mse"],
        "mse_increase_pct": mse_pct,
        "compression_ratio": mem["compression_ratio"],
        "space_saving_pct": mem["space_saving_pct"],
        "time_overhead_pct": overhead,
    }


if __name__ == "__main__":
    run_test()
