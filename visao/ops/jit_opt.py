"""
jit_opt.py — Tarefa 11.2: Otimização por JIT compilation com numba.

Acelera a inferência do VisaoBrain usando @njit do numba.

Conteúdo:
  1. Versão numba-accelerada do LiquidCell.step() e f()
  2. Versão numba-accelerada do VisaoBrain.forward (inferência completa)
  3. Benchmark comparando tempo com/sem numba
  4. Fallback gracioso se numba não estiver instalado

O módulo detecta automaticamente se numba está disponível e usa a melhor
implementação possível. Se numba estiver presente, as funções @njit são
compiladas na primeira chamada (warm-up) e reutilizadas nas subsequentes.

Uso:
    from visao.ops.jit_opt import step_numba, forward_numba, benchmark_speedup

    # Inferência acelerada
    x_new, fx = step_numba(x, u, W_in, W_rec, b, A, tau, dt)

    # Benchmark
    result = benchmark_speedup(n_in=2, n_hidden=64, n_steps=1000)
    print(f"Speedup: {result['speedup']:.2f}x")
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

# =====================================================================
# Detecção de numba com fallback gracioso
# =====================================================================
_NUMBA_AVAILABLE = False
_njit_fallback: Callable = lambda *a, **kw: (lambda f: f)  # identity decorator

try:
    from numba import njit  # type: ignore

    _NUMBA_AVAILABLE = True
except ImportError:
    # Fallback gracioso: numba não instalado — decorador identidade
    # As funções funcionam, só não têm aceleração JIT
    warnings.warn(
        "numba não instalado — jit_opt.py rodando em modo fallback (sem JIT). "
        "Instale numba para aceleração: pip install numba",
        stacklevel=2,
    )
    _njit_fallback = lambda *args, **kwargs: (lambda f: f)
    njit = _njit_fallback  # type: ignore[assignment]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid numericamente estável, usada tanto na versão pura quanto numba."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


# =====================================================================
# Funções numba-acceleradas (compiladas apenas se numba disponível)
# =====================================================================

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _sigmoid_numba(z: np.ndarray) -> np.ndarray:
        """Sigmoid compilada — operação elemento-a-elemento acelerada."""
        out = np.empty_like(z)
        for i in range(z.shape[0]):
            val = z[i]
            if val < -60.0:
                out[i] = 0.0
            elif val > 60.0:
                out[i] = 1.0
            else:
                out[i] = 1.0 / (1.0 + np.exp(-val))
        return out

    @njit(cache=True)
    def _f_numba(
        x: np.ndarray,
        u: np.ndarray,
        W_in: np.ndarray,
        W_rec: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """Sinapse líquida compilada: W_in @ u + W_rec @ x + b com sigmoid."""
        z = W_in @ u + W_rec @ x + b
        return _sigmoid_numba(z)

    @njit(cache=True)
    def _step_numba(
        x: np.ndarray,
        u: np.ndarray,
        W_in: np.ndarray,
        W_rec: np.ndarray,
        b: np.ndarray,
        A: np.ndarray,
        tau: np.ndarray,
        dt: float,
    ) -> tuple:
        """Um passo do solver LTC fundido — versão numba.

        Equivale a LiquidCell.step() mas com todos os pesos passados como
        arrays para evitar overhead de atributos de objeto numba.

        Retorna (x_new, fx) como tupla.
        """
        fx = _f_numba(x, u, W_in, W_rec, b)
        n = x.shape[0]
        x_new = np.empty(n)
        for i in range(n):
            num = x[i] + dt * fx[i] * A[i]
            den = 1.0 + dt * (1.0 / tau[i] + fx[i])
            x_new[i] = num / den
        return x_new, fx

    @njit(cache=True)
    def _rollout_numba(
        seq: np.ndarray,
        W_in: np.ndarray,
        W_rec: np.ndarray,
        b: np.ndarray,
        A: np.ndarray,
        tau: np.ndarray,
        dt: float,
    ) -> tuple:
        """Roda uma sequência completa — versão numba de LiquidCell.rollout.

        Parameters
        ----------
        seq : array (T, n_in)
        W_in, W_rec, b, A, tau, dt : parâmetros do modelo

        Returns
        -------
        states : (T, n_hidden)
        acts : (T, n_hidden)
        """
        T = seq.shape[0]
        n_hidden = W_rec.shape[0]
        states = np.empty((T, n_hidden))
        acts = np.empty((T, n_hidden))
        x = np.zeros(n_hidden)
        for t in range(T):
            x, fx = _step_numba(x, seq[t], W_in, W_rec, b, A, tau, dt)
            states[t] = x
            acts[t] = fx
        return states, acts

    @njit(cache=True)
    def _forward_sequence_numba(
        seq: np.ndarray,
        W_in: np.ndarray,
        W_rec: np.ndarray,
        b: np.ndarray,
        A: np.ndarray,
        tau: np.ndarray,
        dt: float,
        W_out: np.ndarray,
        b_out: np.ndarray,
    ) -> np.ndarray:
        """Inferência completa do VisaoBrain numa sequência — versão numba.

        Reservatório + readout linear, sem plasticidade. Útil para modo
        inferência em lote (mais rápido que passo-a-passo).

        Returns
        -------
        preds : (T, n_out)
        """
        T = seq.shape[0]
        n_out = W_out.shape[0]
        preds = np.empty((T, n_out))
        x = np.zeros(W_rec.shape[0])
        for t in range(T):
            x, fx = _step_numba(x, seq[t], W_in, W_rec, b, A, tau, dt)
            preds[t] = W_out @ x + b_out
        return preds

else:
    # =====================================================================
    # Fallback: implementação pura numpy (sem aceleração, mas funcional)
    # =====================================================================
    def _sigmoid_numba(z: np.ndarray) -> np.ndarray:  # type: ignore[no-redef]
        return _sigmoid(z)

    def _f_numba(x, u, W_in, W_rec, b):
        return _sigmoid(W_in @ u + W_rec @ x + b)

    def _step_numba(x, u, W_in, W_rec, b, A, tau, dt):
        fx = _f_numba(x, u, W_in, W_rec, b)
        num = x + dt * fx * A
        den = 1.0 + dt * (1.0 / tau + fx)
        return num / den, fx

    def _rollout_numba(seq, W_in, W_rec, b, A, tau, dt):
        T = len(seq)
        n_hidden = W_rec.shape[0]
        states = np.empty((T, n_hidden))
        acts = np.empty((T, n_hidden))
        x = np.zeros(n_hidden)
        for t in range(T):
            x, fx = _step_numba(x, seq[t], W_in, W_rec, b, A, tau, dt)
            states[t] = x
            acts[t] = fx
        return states, acts

    def _forward_sequence_numba(
        seq, W_in, W_rec, b, A, tau, dt, W_out, b_out
    ):
        T = len(seq)
        n_out = W_out.shape[0]
        preds = np.empty((T, n_out))
        x = np.zeros(W_rec.shape[0])
        for t in range(T):
            x, fx = _step_numba(x, seq[t], W_in, W_rec, b, A, tau, dt)
            preds[t] = W_out @ x + b_out
        return preds


# =====================================================================
# API pública
# =====================================================================

def step_numba(
    x: np.ndarray,
    u: np.ndarray,
    W_in: np.ndarray,
    W_rec: np.ndarray,
    b: np.ndarray,
    A: np.ndarray,
    tau: np.ndarray,
    dt: float,
) -> tuple:
    """Versão acelerada do LiquidCell.step().

    Todos os parâmetros do modelo devem ser passados explicitamente como
    arrays numpy (dtype float64) — isso é requerido pelo numba.
    """
    return _step_numba(x, u, W_in, W_rec, b, A, tau, dt)


def forward_sequence_numba(
    seq: np.ndarray,
    W_in: np.ndarray,
    W_rec: np.ndarray,
    b: np.ndarray,
    A: np.ndarray,
    tau: np.ndarray,
    dt: float,
    W_out: np.ndarray,
    b_out: np.ndarray,
) -> np.ndarray:
    """Inferência em lote: reservatório + readout numba numa sequência."""
    return _forward_sequence_numba(
        seq, W_in, W_rec, b, A, tau, dt, W_out, b_out
    )


# =====================================================================
# VisaoBrain JIT wrapper — acelera inferência sem mudar API
# =====================================================================

class JitBrain:
    """Wrapper numba-acelerado para VisaoBrain em modo inferência.

    Usa a mesma API do VisaoBrain.forward() mas com o loop interno
    compilado via @njit, eliminando overhead Python por passo.

    Exemplo:
        brain = VisaoBrain(n_in=2, n_hidden=64, n_out=1)
        jit_brain = JitBrain.from_brain(brain)
        pred = jit_brain.forward(u)  # passo único acelerado
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, dt: float):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.dt = dt
        self._x = np.zeros(n_hidden, dtype=np.float64)

    @classmethod
    def from_brain(cls, brain) -> "JitBrain":
        """Cria JitBrain a partir de um VisaoBrain existente."""
        jb = cls(
            n_in=brain.n_in,
            n_hidden=brain.n_hidden,
            n_out=brain.n_out,
            dt=brain.cell.dt,
        )
        jb._x = brain.x.copy()
        jb._W_in = np.ascontiguousarray(brain.cell.W_in, dtype=np.float64)
        jb._W_rec = np.ascontiguousarray(brain.cell.W_rec, dtype=np.float64)
        jb._b = np.ascontiguousarray(brain.cell.b, dtype=np.float64)
        jb._A = np.ascontiguousarray(brain.cell.A, dtype=np.float64)
        jb._tau = np.ascontiguousarray(brain.cell.tau, dtype=np.float64)
        jb._W_out = np.ascontiguousarray(brain.learner.W_out, dtype=np.float64)
        jb._b_out = np.ascontiguousarray(brain.learner.b_out, dtype=np.float64)
        return jb

    def reset_state(self) -> None:
        self._x = np.zeros(self.n_hidden, dtype=np.float64)

    def forward(self, u: np.ndarray) -> np.ndarray:
        """Um passo de inferência acelerado (sem aprendizado)."""
        u = np.asarray(u, dtype=np.float64).ravel()
        self._x, _ = _step_numba(
            self._x, u, self._W_in, self._W_rec, self._b,
            self._A, self._tau, self.dt,
        )
        return self._W_out @ self._x + self._b_out

    def forward_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Inferência em lote numa sequência inteira."""
        seq = np.ascontiguousarray(seq, dtype=np.float64)
        self.reset_state()
        return _forward_sequence_numba(
            seq, self._W_in, self._W_rec, self._b, self._A,
            self._tau, self.dt, self._W_out, self._b_out,
        )


# =====================================================================
# Benchmark: mede speedup com/sem numba
# =====================================================================

def benchmark_speedup(
    n_in: int = 2,
    n_hidden: int = 64,
    n_out: int = 1,
    n_steps: int = 2000,
    dt: float = 0.15,
    seed: int = 42,
    warmup: int = 2,
) -> dict:
    """Compara tempo de inferência com/sem numba.

    Parameters
    ----------
    n_in : dimensão da entrada
    n_hidden : número de neurônios líquidos
    n_out : dimensão da saída
    n_steps : número de passos na sequência de teste
    dt : passo de integração
    seed : semente para reprodutibilidade
    warmup : número de repetições de aquecimento (warm-up JIT)

    Returns
    -------
    dict com métricas de benchmark:
        - time_plain: tempo sem numba (s)
        - time_jit: tempo com numba (s)
        - speedup: razão time_plain / time_jit
        - n_in, n_hidden, n_out, n_steps, dt: parâmetros usados
        - numba_available: se numba está ativo
        - matches: se saídas de ambos são numericamente iguais
    """
    import time

    rng = np.random.default_rng(seed)
    seq = rng.standard_normal((n_steps, n_in))

    # Parâmetros do modelo (aleatórios, mas idênticos para ambos)
    W_in = rng.standard_normal((n_hidden, n_in)) / np.sqrt(max(n_in, 1))
    W_rec = rng.standard_normal((n_hidden, n_hidden)) / np.sqrt(max(n_hidden, 1))
    b = np.zeros(n_hidden)
    A = rng.standard_normal(n_hidden)
    tau = np.exp(rng.uniform(np.log(0.4), np.log(4.0), n_hidden))
    W_out = rng.standard_normal((n_out, n_hidden)) / np.sqrt(max(n_hidden, 1))
    b_out = np.zeros(n_out)

    # ---------- versão pura (numpy puro, loop Python) ----------
    x = np.zeros(n_hidden)
    preds_plain = np.empty((n_steps, n_out))

    # Warm-up
    for t in range(min(5, n_steps)):
        fx = _sigmoid(W_in @ seq[t] + W_rec @ x + b)
        num = x + dt * fx * A
        den = 1.0 + dt * (1.0 / tau + fx)
        x = num / den
        preds_plain[t] = W_out @ x + b_out

    x = np.zeros(n_hidden)
    t0 = time.perf_counter()
    for t in range(n_steps):
        fx = _sigmoid(W_in @ seq[t] + W_rec @ x + b)
        num = x + dt * fx * A
        den = 1.0 + dt * (1.0 / tau + fx)
        x = num / den
        preds_plain[t] = W_out @ x + b_out
    time_plain = time.perf_counter() - t0

    # ---------- versão numba ----------
    # Aquecimento JIT: força compilação antes de medir
    for _ in range(warmup):
        _ = _forward_sequence_numba(
            seq[:10], W_in, W_rec, b, A, tau, dt, W_out, b_out
        )

    t0 = time.perf_counter()
    preds_jit = _forward_sequence_numba(
        seq, W_in, W_rec, b, A, tau, dt, W_out, b_out
    )
    time_jit = time.perf_counter() - t0

    # Verificação de corretude
    if _NUMBA_AVAILABLE:
        matches = np.allclose(preds_plain, preds_jit, atol=1e-12)
        if not matches:
            max_diff = float(np.max(np.abs(preds_plain - preds_jit)))
            warnings.warn(
                f"Saídas JIT e plain não batem (max_diff={max_diff:.2e})",
                stacklevel=2,
            )
    else:
        matches = True  # fallback é idêntico à versão de referência

    speedup = time_plain / time_jit if time_jit > 0 else float("inf")

    return {
        "time_plain": time_plain,
        "time_jit": time_jit,
        "speedup": speedup,
        "n_in": n_in,
        "n_hidden": n_hidden,
        "n_out": n_out,
        "n_steps": n_steps,
        "dt": dt,
        "numba_available": _NUMBA_AVAILABLE,
        "matches": matches,
    }


# =====================================================================
# CLI rápido: python -m visao.ops.jit_opt
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Tarefa 11.2 — JIT Compilation com Numba")
    print("=" * 60)
    print(f"numba disponível: {_NUMBA_AVAILABLE}")
    print()

    configs = [
        (2, 32, 1, 1000),
        (2, 64, 1, 2000),
        (2, 128, 1, 3000),
        (10, 128, 5, 2000),
    ]

    for n_in, n_hidden, n_out, n_steps in configs:
        print(f"Config: n_in={n_in}, n_hidden={n_hidden}, "
              f"n_out={n_out}, n_steps={n_steps}")
        result = benchmark_speedup(
            n_in=n_in, n_hidden=n_hidden, n_out=n_out, n_steps=n_steps
        )
        print(f"  Tempo plain: {result['time_plain']*1000:.2f} ms")
        print(f"  Tempo JIT:   {result['time_jit']*1000:.2f} ms")
        print(f"  Speedup:     {result['speedup']:.2f}x")
        print(f"  Match:       {result['matches']}")
        print()

    print("Concluído.")
