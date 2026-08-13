"""
cfc.py — Célula Closed-form Continuous-time (CfC) em JAX.

Porte de prototype/liquid.py (numpy) para JAX com jit + scan.
A assinatura de rollout() é mantida idêntica para permitir o teste de
paridade numérica da tarefa 1.3b — sem ele, isto seria reescrita, não porte.

Referências:
  Hasani et al., "Liquid Time-constant Networks", AAAI 2021 (arXiv:2006.04439)
  Hasani et al., "Closed-form Continuous-time Neural Networks",
                 Nature Machine Intelligence 2022 (arXiv:2106.13898)

Equação:
    dx/dt = -[1/tau + f(x, I)] * x + f(x, I) * A

Solver fundido (explicit/implicit Euler), incondicionalmente estável para
dt > 0, tau > 0 e f >= 0 — o denominador é sempre > 1:

    x_{t+1} = (x_t + dt * f * A) / (1 + dt * (1/tau + f))

O ponto que importa: tau_efetivo = tau / (1 + tau * f(x, I)) depende da
ENTRADA. Cada neurônio reescolhe a própria escala de tempo a cada passo.
Nenhum transformer faz isso.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


class CfCCell:
    """Célula líquida com constantes de tempo dependentes da entrada.

    Os pesos são gerados com numpy.random.Generator (não com jax.random)
    deliberadamente: é a mesma fonte de aleatoriedade do protótipo numpy,
    o que torna a paridade numérica da tarefa 1.3b verificável de verdade.

    Parameters
    ----------
    n_in : dimensão da entrada
    n_hidden : número de neurônios líquidos
    sparsity : fração de sinapses recorrentes zeradas (topologia tipo NCP)
    tau_min, tau_max : faixa log-uniforme das constantes de tempo base
    dt : passo de integração
    seed : semente (mesma convenção do protótipo)
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        sparsity: float = 0.5,
        tau_min: float = 0.4,
        tau_max: float = 4.0,
        dt: float = 0.1,
        seed: int = 0,
        rng: np.random.Generator | None = None,
    ):
        if n_in <= 0:
            raise ValueError(f"n_in deve ser > 0, obtido {n_in}")
        if n_hidden <= 0:
            raise ValueError(f"n_hidden deve ser > 0, obtido {n_hidden}")
        if dt <= 0:
            raise ValueError(f"dt deve ser > 0, obtido {dt}")

        rng = rng if rng is not None else np.random.default_rng(seed)
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.dt = float(dt)

        scale_in = 1.0 / np.sqrt(max(n_in, 1))
        scale_rec = 1.0 / np.sqrt(max(n_hidden, 1))

        W_in = rng.normal(0, scale_in, (n_hidden, n_in))
        W_rec = rng.normal(0, scale_rec, (n_hidden, n_hidden))
        b = np.zeros(n_hidden)

        # Topologia fixa: plasticidade age nos pesos, não cria sinapses.
        mask = rng.random((n_hidden, n_hidden)) > sparsity
        np.fill_diagonal(mask, False)  # sem auto-sinapse: o vazamento já cumpre esse papel
        mask = mask.astype(np.float64)
        W_rec = W_rec * mask

        # tau heterogêneo: neurônios rápidos pegam transientes, lentos guardam
        # contexto. É daí que vem a memória multiescala.
        tau = np.exp(rng.uniform(np.log(tau_min), np.log(tau_max), n_hidden))
        A = rng.normal(0, 1.0, n_hidden)  # potencial de reversão sináptico

        self.W_in = jnp.asarray(W_in)
        self.W_rec = jnp.asarray(W_rec)
        self.b = jnp.asarray(b)
        self.mask = jnp.asarray(mask)
        self.tau = jnp.asarray(tau)
        self.A = jnp.asarray(A)

        # hiperparâmetros guardados para reconstrução do artefato (Fase 6)
        self.sparsity = float(sparsity)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.seed = int(seed)

        self._step_jit = jax.jit(self._step_raw)
        self._rollout_jit = jax.jit(self._rollout_raw)

    # ------------------------------------------------------------- dinâmica
    def _f_raw(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Não-linearidade sináptica; vira a taxa de acoplamento."""
        return jax.nn.sigmoid(self.W_in @ u + self.W_rec @ x + self.b)

    def _step_raw(self, x: jnp.ndarray, u: jnp.ndarray):
        fx = self._f_raw(x, u)
        num = x + self.dt * fx * self.A
        den = 1.0 + self.dt * (1.0 / self.tau + fx)
        return num / den, fx

    def _rollout_raw(self, seq: jnp.ndarray, x0: jnp.ndarray):
        def body(x, u):
            x_next, fx = self._step_raw(x, u)
            return x_next, (x_next, fx)

        _, (states, acts) = jax.lax.scan(body, x0, seq)
        return states, acts

    # --------------------------------------------------------------- API
    def f(self, x, u) -> jnp.ndarray:
        return self._f_raw(jnp.asarray(x), jnp.asarray(u))

    def step(self, x, u):
        """Um passo do solver fundido. Retorna (novo_estado, ativação f)."""
        return self._step_jit(jnp.asarray(x), jnp.asarray(u))

    def tau_effective(self, fx) -> jnp.ndarray:
        """Constante de tempo instantânea — a métrica de liquidez.

        Aceita tanto (n_hidden,) quanto (T, n_hidden): o broadcast de tau
        funciona nos dois casos porque tau é o último eixo.
        """
        return self.tau / (1.0 + self.tau * jnp.asarray(fx))

    def rollout(self, seq, x0=None):
        """Roda uma sequência (T, n_in). Retorna estados e ativações (T, n_hidden)."""
        seq = jnp.asarray(seq)
        x0 = jnp.zeros(self.n_hidden) if x0 is None else jnp.asarray(x0)
        return self._rollout_jit(seq, x0)

    # ---------------------------------------------------------- utilidades
    def params(self) -> dict[str, jnp.ndarray]:
        return {"W_in": self.W_in, "W_rec": self.W_rec, "b": self.b, "A": self.A}

    def flat(self) -> jnp.ndarray:
        return jnp.concatenate([p.ravel() for p in self.params().values()])

    def load_flat(self, vec) -> None:
        vec = jnp.asarray(vec)
        i = 0
        for name, p in self.params().items():
            n = p.size
            new = vec[i : i + n].reshape(p.shape)
            if name == "W_rec":
                new = new * self.mask  # topologia é invariante
            setattr(self, name, new)
            i += n
        self._step_jit = jax.jit(self._step_raw)
        self._rollout_jit = jax.jit(self._rollout_raw)

    # -------------------------------------------------- persistência (Fase 6)
    @staticmethod
    def _brain_path(path: str) -> str:
        path = str(path)
        return path if path.endswith(".npz") else path + ".npz"

    def save_brain(self, path) -> None:
        """Salva o estado completo (pesos + topologia) via jnp.savez.

        `path` recebe a extensão .npz. É a base do agente autocontido (6.2)
        e do arquivo de variantes (3.1).

        Nota: jax.save/load não existem neste build (jax 0.10.2); usa-se a
        serialização nativa jnp.savez/jnp.load, que é fiel a nível de 1e-6.
        """
        state = {
            "W_in": self.W_in,
            "W_rec": self.W_rec,
            "b": self.b,
            "A": self.A,
            "mask": self.mask,
            "tau": self.tau,
        }
        jnp.savez(self._brain_path(path), **state)

    def load_brain(self, path) -> None:
        """Recarrega o estado salvo por save_brain(), sobrescrevendo os pesos."""
        state = jnp.load(self._brain_path(path))
        self.load_brain_state(state)

    def load_brain_state(self, state) -> None:
        """Aplica um dicionário de estado (npz/jnp) aos pesos da célula.

        Usado pelo artefato autocontido (6.2) que decodifica os pesos em
        memória, sem tocar disco. Re-jit dos kernels após a troca.
        """
        self.W_in = jnp.asarray(state["W_in"])
        self.W_rec = jnp.asarray(state["W_rec"])
        self.b = jnp.asarray(state["b"])
        self.A = jnp.asarray(state["A"])
        self.mask = jnp.asarray(state["mask"])
        self.tau = jnp.asarray(state["tau"])
        self._step_jit = jax.jit(self._step_raw)
        self._rollout_jit = jax.jit(self._rollout_raw)
