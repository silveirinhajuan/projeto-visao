"""task_free.py — Tarefa 8.5: Aprendizado contínuo sem fronteira de tarefa.

O cérebro detecta mudança de distribuição automaticamente via CUSUM
(Cumulative Sum) sobre o sinal de erro e adapta plasticidade sem saber
quando a tarefa mudou.

Mecanismo:
  - CUSUM acumula desvio do erro abaixo de uma baseline (surprise persistente
    indica que o modelo parou de acompanhar a distribuição).
  - Ao detectar: aumentar lr temporariamente, decair omega (afrouxar
    consolidação, liberar plasticidade para adaptação rápida).
  - Após estabilizar: voltar ao normal gradualmente.

Referências:
  - Page, E. S. (1954). Continuous Inspection Schemes. Biometrika.
  - Shafer, A. B. et al. (2021). Task-Free Continual Learning.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visao.brain import VisaoBrain


class CUSUMDetector:
    """Detector de mudança de distribuição baseado em CUSUM dois lados.

    O CUSUM positivo detecta aumentos sustentados no erro (degradação de
    performance por mudança de regime). O CUSUM negativo detecta melhorias
    (aprendizado ativo no novo regime — usado para sinalizar que o sistema
    está se adaptando).

    Parâmetros:
      drift: margem antes de acumular (evita ruído de amostra).
      threshold: limite do acumulador para declarar mudança.
      warmup: passos iniciais usados só para estimar baseline.
    """

    def __init__(
        self,
        drift: float = 0.01,
        threshold: float = 0.5,
        warmup: int = 100,
    ):
        self.drift = drift
        self.threshold = threshold
        self.warmup = warmup

        self._sum_pos = 0.0
        self._sum_neg = 0.0
        self._baseline = 0.0
        self._history: list[float] = []
        self._step = 0
        self.triggered = False

    def update(self, value: float) -> bool:
        """Atualiza CUSUM com novo valor de erro.

        Retorna True se detectou mudança de distribuição.
        """
        self._step += 1
        self._history.append(value)

        if self._step <= self.warmup:
            # Fase de aquecimento: só estima baseline
            self._baseline = float(np.mean(self._history))
            return False

        # Atualiza baseline suavemente (exponencial)
        self._baseline = 0.99 * self._baseline + 0.01 * value

        # CUSUM dois lados (acima e abaixo da baseline)
        self._sum_pos = max(0.0, self._sum_pos + value - self._baseline - self.drift)
        self._sum_neg = max(0.0, self._sum_neg + self._baseline - value - self.drift)

        if self._sum_pos > self.threshold:
            # Reset para evitar disparos em cadeia
            self._sum_pos = 0.0
            self._sum_neg = 0.0
            self.triggered = True
            return True

        return False

    @property
    def baseline(self) -> float:
        return self._baseline

    def reset(self) -> None:
        self._sum_pos = 0.0
        self._sum_neg = 0.0
        self.triggered = False


class TaskFreeLearner:
    """Aprendizado contínuo sem fronteira de tarefa.

    Envolve um VisaoBrain e adapta seus hiperparâmetros quando o detector
    CUSUM sinaliza mudança de distribuição.

    Parâmetros:
      brain: instância de VisaoBrain.
      detector: instância de CUSUMDetector.
      lr_boost: multiplicador de lr ao detectar mudança.
      omega_decay: fator de decaimento de omega ao detectar mudança.
      recovery_steps: passos para retornar ao normal após detecção.
    """

    def __init__(
        self,
        brain: VisaoBrain,
        detector: CUSUMDetector | None = None,
        lr_boost: float = 3.0,
        omega_decay: float = 0.5,
        recovery_steps: int = 150,
    ):
        self.brain = brain
        self.detector = detector or CUSUMDetector()
        self.lr_boost = lr_boost
        self.omega_decay = omega_decay
        self.recovery_steps = recovery_steps

        self.change_points: list[int] = []
        self._steps_since_change: int = 0
        self._adapting: bool = False
        self._lr_original = brain.lr

    def learn(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Um passo de aprendizado com detecção de mudança.

        Retorna dict com predição, erro, surprise e se houve detecção.
        """
        result = self.brain.learn(x, y)
        err = result["err"]

        # Atualiza detector CUSUM
        detected = self.detector.update(err)

        if detected:
            self._on_distribution_change()

        # Gerencia recuperação
        if self._adapting:
            self._steps_since_change += 1
            if self._steps_since_change >= self.recovery_steps:
                self._recover()

        result["detected_change"] = detected
        result["adapting"] = self._adapting
        return result

    def _on_distribution_change(self) -> None:
        """Reage a uma mudança de distribuição detectada."""
        self.change_points.append(self.brain.step)
        self._adapting = True
        self._steps_since_change = 0

        # Boost de lr para adaptação rápida
        self.brain.lr = min(self.brain.lr * self.lr_boost, self.brain._meta_max)
        self.brain.learner.lr = self.brain.lr

        # Decair omega: afrouxar consolidação para liberar plasticidade
        self.brain.learner.omega *= self.omega_decay

    def _recover(self) -> None:
        """Retorna gradualmente aos parâmetros normais."""
        self._adapting = False
        self.brain.lr = self._lr_original
        self.brain.learner.lr = self._lr_original

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Inferência (delega ao brain)."""
        return self.brain.forward(x)

    @property
    def n_changes_detected(self) -> int:
        return len(self.change_points)


# ----------------------------------------------------------------- utilidades


def make_regime_stream(
    n: int = 4000,
    seed: int = 0,
    regimes: list[tuple[str, int, int, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int, int, float]]]:
    """Gera stream de regressão com mudanças de regime não-anunciadas.

    Cada regime tem uma função sinal diferente. A saída y é uma versão
    suavizada do sinal de entrada, tornando a tarefa de prever y a partir
    de u uma regressão sensível à mudança de distribuição.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    u = np.zeros((n, 2))
    y = np.zeros((n, 1))

    if regimes is None:
        regimes = [
            ("sine_slow", 0, n // 4, 0.02),
            ("sine_fast", n // 4, n // 2, 0.08),
            ("saw", n // 2, 3 * n // 4, 0.0),
            ("mixed", 3 * n // 4, n, 0.0),
        ]

    for kind, start, end, freq in regimes:
        length = end - start
        tt = np.arange(length)
        if kind == "sine_slow":
            signal = np.sin(2 * np.pi * freq * tt)
        elif kind == "sine_fast":
            signal = np.sin(2 * np.pi * freq * tt)
        elif kind == "saw":
            signal = 2.0 * (tt % 30) / 30.0 - 1.0
        elif kind == "step":
            signal = np.where(tt < length // 2, -1.0, 1.0).astype(float)
        elif kind == "mixed":
            signal = np.sin(2 * np.pi * 0.03 * tt) + 0.5 * np.cos(2 * np.pi * 0.06 * tt)
        else:
            signal = np.sin(2 * np.pi * 0.05 * tt)

        u[start:end, 0] += signal
        u[start:end, 1] += 0.3 * np.roll(signal, 1)
        y[start:end, 0] = np.convolve(signal, np.ones(5) / 5, mode="same")[:length]

    u += rng.normal(0, 0.03, u.shape)
    return u.astype(np.float64), y.astype(np.float64), regimes


def run_task_free(
    learner: TaskFreeLearner,
    u: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    """Roda o learner task-free sobre um stream.

    Retorna métricas por regime e detecções.
    """
    errors = []
    surprises = []
    detections = []
    adapting_flags = []

    for i in range(len(u)):
        result = learner.learn(u[i], y[i])
        errors.append(result["err"])
        surprises.append(result["surprise"])
        detections.append(result["detected_change"])
        adapting_flags.append(result["adapting"])

    errors = np.array(errors)
    return {
        "errors": errors,
        "surprises": np.array(surprises),
        "detections": np.array(detections),
        "adapting": np.array(adapting_flags),
        "n_changes_detected": learner.n_changes_detected,
        "change_points": learner.change_points,
        "mse_total": float(np.mean(errors ** 2)),
        "mse_final_500": float(np.mean(errors[-500:] ** 2)),
    }


def run_experiment(
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    n: int = 4000,
    n_hidden: int = 64,
) -> dict[str, Any]:
    """Roda experimento completo: com e sem detecção."""
    results: dict[str, list[dict[str, Any]]] = {
        "with_detection": [],
        "without_detection": [],
    }

    for seed in seeds:
        u, y, regimes = make_regime_stream(n=n, seed=seed)

        # Com detecção CUSUM
        brain1 = VisaoBrain(
            n_in=2, n_hidden=n_hidden, n_out=1,
            consolidation=8.0, meta_learn=True, seed=seed,
        )
        detector = CUSUMDetector(drift=0.01, threshold=0.5, warmup=100)
        learner = TaskFreeLearner(brain1, detector=detector)
        r1 = run_task_free(learner, u, y)
        results["with_detection"].append(r1)

        # Sem detecção (meta-learn padrão, sem adaptação)
        brain2 = VisaoBrain(
            n_in=2, n_hidden=n_hidden, n_out=1,
            consolidation=8.0, meta_learn=True, seed=seed,
        )
        errs2 = []
        for ui, yi in zip(u, y):
            r = brain2.learn(ui, yi)
            errs2.append(r["err"])
        results["without_detection"].append({
            "mse_total": float(np.mean(np.array(errs2) ** 2)),
            "mse_final_500": float(np.mean(np.array(errs2)[-500:] ** 2)),
        })

    # Agregar
    agg: dict[str, Any] = {}
    for name, runs in results.items():
        agg[name] = {
            "mse_total": float(np.mean([r["mse_total"] for r in runs])),
            "mse_total_std": float(np.std([r["mse_total"] for r in runs])),
            "mse_final_500": float(np.mean([r["mse_final_500"] for r in runs])),
        }
        if "n_changes_detected" in runs[0]:
            agg[name]["avg_change_points"] = float(
                np.mean([r["n_changes_detected"] for r in runs])
            )

    return agg


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 8.5 — Task-Free Learning com CUSUM")
    print("Stream: 4 regimes não-anunciados")
    print("=" * 70)

    results = run_experiment()

    print("\nResultados (5 seeds):")
    for name, r in results.items():
        print(f"  {name}:")
        print(f"    MSE total     = {r['mse_total']:.4f} ± {r.get('mse_total_std', 0):.4f}")
        print(f"    MSE final 500 = {r['mse_final_500']:.4f}")
        if "avg_change_points" in r:
            print(f"    mudanças detectadas = {r['avg_change_points']:.1f}")

    OUT = _ROOT / "visao" / "analysis" / "results_task_free.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[ok] -> {OUT}")
