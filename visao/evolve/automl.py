"""automl.py — Auto-ajuste LOCAL em sandbox (tarefa 5.7, L3 do PLANO.md).

AutoML "comum": o tuner propõe combinações de hiperparâmetros (τ, taxas de
regra local, topologia) e mede cada uma DENTRO do Sandbox existente
(visao.evolve.sandbox), sem rede, sem persistência. O melhor é devolvido e
registrado no Archive. Declaradamente FORA do escopo: replicação em rede e
qualquer mutação de código de governança (R3). Por isso o código de cada
variante é um marcador neutro — nunca toca zona proibida.

O eval script é fornecido por quem chama; ele decide o que medir (ex.:
esquecimento contínuo, RMSE). O tuner só propõe, avalia e seleciona.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Optional

# Marcador neutro: código de variante do tuner NUNCA referencia governança/
# rede/chaves. É só um container de hiperparâmetros para o eval script ler.
CODE_MARKER = "# automl candidate (R3-safe: parametros apenas)"


@dataclass
class AutoTuner:
    """Busca hiperparâmetros em ``search_space`` avaliando cada candidato no sandbox."""

    archive: object  # visao.evolve.archive.Archive
    sandbox: object  # visao.evolve.sandbox.Sandbox
    search_space: dict  # nome -> lista de valores candidatos
    base_config: dict = field(default_factory=dict)
    mode: str = "grid"  # "grid" | "random"
    seed: int = 0
    budget: Optional[int] = None
    code_marker: str = CODE_MARKER

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self.n_evaluated: int = 0

    # ----------------------------------------------------------- combinações
    def _all_combinations(self) -> list[dict]:
        keys = list(self.search_space.keys())
        value_lists = [self.search_space[k] for k in keys]
        combos = []
        for vals in itertools.product(*value_lists):
            combos.append(dict(zip(keys, vals)))
        return combos

    def _combinations(self) -> list[dict]:
        combos = self._all_combinations()
        if self.mode == "random" and self.budget is not None:
            # amostra sem reposição; determinístico pelo seed
            pool = list(combos)
            self._rng.shuffle(pool)
            return pool[: self.budget]
        if self.budget is not None:
            return combos[: self.budget]
        return combos

    # ----------------------------------------------------------------- run
    def run(self, eval_script: str) -> Optional[str]:
        """Avalia candidatos no sandbox; retorna o id da melhor variante (ou None)."""
        self.n_evaluated = 0
        for cfg in self._combinations():
            weights = {**self.base_config, **cfg}
            vid = self.archive.add(code=self.code_marker, weights=weights)
            fitness = self.sandbox.evaluate(self.archive.get(vid), eval_script)
            self.archive.set_fitness(vid, fitness)
            self.n_evaluated += 1
        best_ids = self.archive.best(1)
        return best_ids[0] if best_ids else None

    def best_config(self) -> Optional[dict]:
        """Pesos (hiperparâmetros) da melhor variante avaliada, ou None."""
        best_ids = self.archive.best(1)
        if not best_ids:
            return None
        return dict(self.archive.get(best_ids[0]).weights)
