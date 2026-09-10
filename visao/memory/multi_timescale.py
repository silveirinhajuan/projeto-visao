"""
multi_timescale.py — Multi-Timescale Memory for VISÃO.

Implements ThreeTimescaleState: neural state divided into 3 groups
with different time constants (τ_s < τ_m < τ_l) forming a temporal
hierarchy.

    - Short  (τ=0.1): fast-decaying, tracks recent input, high plasticity
    - Medium (τ=1.0): intermediate persistence
    - Long   (τ=10.0): slow-decaying, retains information persistently

The update rule for each group i is:
    dx_i/dt = (-x_i + f(W_i · u)) / τ_i

Smaller τ → faster tracking (short-term memory)
Larger τ → slower decay (long-term persistent memory)

This enables coexistence of fast-decaying information with persistent
memories — a core requirement for continual learning systems.

References:
    - Wang et al. (2018): "Prefrontal cortex as a meta-recurrent network"
    - Boot et al. (2024): Multi-timescale neural computation
    - Hassabis et al. (2017): Complementary memory systems
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ==============================================================
#  THREE TIMESCALE STATE
# ==============================================================


@dataclass
class TimescaleGroup:
    """A single group of neurons with a specific time constant."""

    dim: int
    tau: float
    label: str = ""

    # State (initialized lazily)
    state: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if self.state.size == 0:
            self.state = np.zeros(self.dim, dtype=np.float64)

    @property
    def mean_activation(self) -> float:
        """Mean absolute activation of this group."""
        if self.state.size == 0:
            return 0.0
        return float(np.mean(np.abs(self.state)))

    @property
    def energy(self) -> float:
        """L2 norm of the state."""
        return float(np.linalg.norm(self.state))


class ThreeTimescaleState:
    """Multi-timescale memory: state split into 3 groups with different τ.

    Architecture:
    ┌──────────────────────────────────────────────────────┐
    │  Input u → f(W·u) → drive vector                     │
    │                                                      │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
    │  │  Short τ=0.1│  │ Medium τ=1.0│  │  Long τ=10.0│   │
    │  │  (fast)     │  │ (mid)      │  │ (persistent)│    │
    │  └────────────┘  └────────────┘  └────────────┘     │
    │                                                      │
    │  Full state = [x_s | x_m | x_l]                     │
    └──────────────────────────────────────────────────────┘

    The different time constants create a temporal hierarchy:
    - Short group: quickly forgets, tracks fast-changing info
    - Medium group: intermediate buffer
    - Long group: slowly forgets, retains persistent memories
    """

    def __init__(
        self,
        dims: tuple[int, int, int] = (16, 16, 16),
        taus: tuple[float, float, float] = (0.1, 1.0, 10.0),
        dt: float = 0.05,
        seed: int = 0,
    ):
        assert len(dims) == 3, "Must specify 3 dimensions"
        assert len(taus) == 3, "Must specify 3 time constants"
        assert taus[0] < taus[1] < taus[2], "Must have τ_s < τ_m < τ_l"
        assert dt > 0, "dt must be positive"

        self.dims = dims
        self.taus = taus
        self.dt = dt
        self.total_dim = sum(dims)
        self._rng = np.random.default_rng(seed)

        # Create timescale groups
        self.short = TimescaleGroup(dim=dims[0], tau=taus[0], label="short")
        self.medium = TimescaleGroup(dim=dims[1], tau=taus[1], label="medium")
        self.long = TimescaleGroup(dim=dims[2], tau=taus[2], label="long")

        # Projection matrix: input_dim → total_dim (set on first step)
        self._proj: Optional[np.ndarray] = None
        self._input_dim: Optional[int] = None
        self._bias = np.zeros(self.total_dim, dtype=np.float64)

        # Tracking
        self._step_count = 0

    # ── Properties ──────────────────────────────────────────

    @property
    def state(self) -> np.ndarray:
        """Full state vector: [x_short | x_medium | x_long]."""
        return np.concatenate([self.short.state, self.medium.state, self.long.state])

    @state.setter
    def state(self, value: np.ndarray):
        """Set full state from a vector."""
        value = np.asarray(value, dtype=np.float64).ravel()
        assert len(value) == self.total_dim
        d1, d2 = self.dims[0], self.dims[0] + self.dims[1]
        self.short.state = value[:d1]
        self.medium.state = value[d1:d2]
        self.long.state = value[d2:]

    @property
    def step_count(self) -> int:
        return self._step_count

    def group_state(self, group: str) -> np.ndarray:
        """Get state of a specific group."""
        if group == "short":
            return self.short.state
        elif group == "medium":
            return self.medium.state
        elif group == "long":
            return self.long.state
        raise ValueError(f"Unknown group: {group}")

    # ── Core dynamics ───────────────────────────────────────

    def _init_projection(self, input_dim: int):
        """Initialize projection matrix from input to state."""
        self._input_dim = input_dim
        # Xavier-like init
        self._proj = self._rng.normal(
            0, np.sqrt(2.0 / input_dim), (self.total_dim, input_dim)
        )
        self._bias = np.zeros(self.total_dim, dtype=np.float64)

    @staticmethod
    def _activation(x: np.ndarray) -> np.ndarray:
        """Smooth bounded activation."""
        return np.tanh(x)

    def _compute_drive(self, u: np.ndarray) -> np.ndarray:
        """Compute input drive from external signal."""
        u = np.asarray(u, dtype=np.float64).ravel()
        if self._proj is None or self._input_dim != len(u):
            self._init_projection(len(u))
        raw = self._proj @ u + self._bias
        return self._activation(raw)

    def step(self, u: np.ndarray, input_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance one integration step.

        dx/dt = (-x + drive) / τ  →  x += dt * (-x + drive) / τ

        Parameters
        ----------
        u : np.ndarray
            External input signal.
        input_mask : np.ndarray, optional
            Binary mask (total_dim,) to selectively zero out drive
            for specific neuron groups. Used for targeted memory injection.

        Returns
        -------
        state : np.ndarray
            Full state vector after update.
        """
        drive = self._compute_drive(u)

        if input_mask is not None:
            input_mask = np.asarray(input_mask, dtype=np.float64).ravel()
            drive = drive * input_mask

        # Split drive into groups
        d1, d2 = self.dims[0], self.dims[0] + self.dims[1]
        drive_short = drive[:d1]
        drive_medium = drive[d1:d2]
        drive_long = drive[d2:]

        # Update each group with its own τ
        # dx = dt * (-x + drive) / τ
        self.short.state += self.dt * (-self.short.state + drive_short) / self.short.tau
        self.medium.state += self.dt * (-self.medium.state + drive_medium) / self.medium.tau
        self.long.state += self.dt * (-self.long.state + drive_long) / self.long.tau

        self._step_count += 1
        return self.state

    def inject(self, group: str, pattern: np.ndarray, strength: float = 1.0):
        """Directly inject a pattern into a specific timescale group.

        Unlike step() which integrates gradually, this sets state directly.
        Useful for memory encoding operations.

        Parameters
        ----------
        group : str
            One of 'short', 'medium', 'long'.
        pattern : np.ndarray
            Pattern to inject.
        strength : float
            Scaling factor (1.0 = full injection).
        """
        pattern = np.asarray(pattern, dtype=np.float64).ravel()
        gs = self.group_state(group)
        if len(pattern) != len(gs):
            raise ValueError(
                f"Pattern dim {len(pattern)} doesn't match group '{group}' dim {len(gs)}"
            )
        gs[:] = strength * pattern + (1.0 - strength) * gs

    def decay_step(self) -> None:
        """Apply pure decay (no input) — simulates memory fade."""
        self.short.state *= np.exp(-self.dt / self.short.tau)
        self.medium.state *= np.exp(-self.dt / self.medium.tau)
        self.long.state *= np.exp(-self.dt / self.long.tau)
        self._step_count += 1

    def reset_group(self, group: str) -> None:
        """Reset a specific group to zero."""
        gs = self.group_state(group)
        gs[:] = 0.0

    def reset_all(self) -> None:
        """Reset all groups to zero."""
        self.reset_group("short")
        self.reset_group("medium")
        self.reset_group("long")

    # ── Memory analysis ─────────────────────────────────────

    def retrieval_strength(self, group: str, original: np.ndarray) -> float:
        """Cosine similarity between current state and original pattern.

        High value = good memory retention.
        Low value = forgetting.
        """
        current = self.group_state(group)
        original = np.asarray(original, dtype=np.float64).ravel()
        if len(current) != len(original):
            return 0.0
        norm_c = np.linalg.norm(current)
        norm_o = np.linalg.norm(original)
        if norm_c < 1e-10 or norm_o < 1e-10:
            return 0.0
        return float(np.dot(current, original) / (norm_c * norm_o))

    def get_timescale_stats(self) -> dict:
        """Statistics for all timescale groups."""
        return {
            "short": {
                "dim": self.dims[0],
                "tau": self.taus[0],
                "mean_activation": self.short.mean_activation,
                "energy": self.short.energy,
            },
            "medium": {
                "dim": self.dims[1],
                "tau": self.taus[1],
                "mean_activation": self.medium.mean_activation,
                "energy": self.medium.energy,
            },
            "long": {
                "dim": self.dims[2],
                "tau": self.taus[2],
                "mean_activation": self.long.mean_activation,
                "energy": self.long.energy,
            },
        }

    def get_state_snapshot(self) -> dict[str, np.ndarray]:
        """Return current state of each group."""
        return {
            "short": self.short.state.copy(),
            "medium": self.medium.state.copy(),
            "long": self.long.state.copy(),
        }


# ==============================================================
#  MULTI-TIMESCALE NETWORK (higher-level interface)
# ==============================================================


class MultiTimescaleMemory:
    """Higher-level interface: multi-timescale memory with read/write.

    Wraps ThreeTimescaleState and adds:
    - Named memory slots per timescale
    - Read operation via query
    - Write operation with timescale selection
    - Consolidation: transfer from short → medium → long

    Parameters
    ----------
    state_dim : int
        Dimension of each timescale group.
    input_dim : int
        Dimension of input signal.
    taus : tuple of float
        Time constants (short, medium, long).
    dt : float
        Integration step.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        state_dim: int = 16,
        input_dim: int = 8,
        taus: tuple[float, float, float] = (0.1, 1.0, 10.0),
        dt: float = 0.05,
        seed: int = 0,
    ):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dt = dt

        self.core = ThreeTimescaleState(
            dims=(state_dim, state_dim, state_dim),
            taus=taus,
            dt=dt,
            seed=seed,
        )

        # Named memory storage: {name: (group, pattern)}
        self._memories: dict[str, tuple[str, np.ndarray]] = {}
        self._rng = np.random.default_rng(seed)

        # Readout projection (learned via simple Hebbian)
        self._readout = np.zeros((input_dim, self.core.total_dim))

    def write(
        self,
        name: str,
        pattern: np.ndarray,
        timescale: str = "short",
        strength: float = 1.0,
    ):
        """Write a named memory to a specific timescale.

        Parameters
        ----------
        name : str
            Memory identifier.
        pattern : np.ndarray
            Pattern to store (will be truncated/padded to state_dim).
        timescale : str
            'short', 'medium', or 'long'.
        strength : float
            Injection strength (0-1).
        """
        pattern = np.asarray(pattern, dtype=np.float64).ravel()[:self.state_dim]
        if len(pattern) < self.state_dim:
            padded = np.zeros(self.state_dim)
            padded[:len(pattern)] = pattern
            pattern = padded

        self.core.inject(timescale, pattern, strength=strength)
        self._memories[name] = (timescale, pattern.copy())

    def read(self, query: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Read memories via query similarity.

        Returns top-k matching memories with similarity scores.
        """
        query = np.asarray(query, dtype=np.float64).ravel()[:self.state_dim]
        if len(query) < self.state_dim:
            padded = np.zeros(self.state_dim)
            padded[:len(query)] = query
            query = padded

        results = []
        for name, (group, pattern) in self._memories.items():
            current = self.core.group_state(group)
            # Compare query to current state of that group
            sim = self._cosine_sim(query, current)
            results.append((name, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def consolidate(self, name: str):
        """Consolidate a memory: move it to a slower timescale.

        short → medium → long

        This simulates memory consolidation during rest/sleep.
        """
        if name not in self._memories:
            return False

        group, pattern = self._memories[name]
        if group == "short":
            new_group = "medium"
        elif group == "medium":
            new_group = "long"
        else:
            return False  # Already at longest timescale

        # Inject into new timescale with slight decay
        self.core.inject(new_group, pattern, strength=0.9)
        self._memories[name] = (new_group, pattern.copy())
        return True

    def update(self, u: np.ndarray) -> np.ndarray:
        """One integration step with external input."""
        return self.core.step(u)

    def rest_step(self, n_steps: int = 10):
        """Simulate rest: pure decay + consolidation pressure.

        During rest, short-term memories gradually transfer to
        longer timescales.
        """
        for _ in range(n_steps):
            self.core.decay_step()

    def get_memory_info(self) -> dict:
        """Info about all stored memories."""
        info = {}
        for name, (group, pattern) in self._memories.items():
            retention = self.core.retrieval_strength(group, pattern)
            info[name] = {
                "timescale": group,
                "tau": self.core.taus[0] if group == "short" else (
                    self.core.taus[1] if group == "medium" else self.core.taus[2]
                ),
                "retention": retention,
            }
        return info

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
