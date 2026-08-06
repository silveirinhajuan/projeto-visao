"""archive.py — arquivo de variantes (grafo de linhagem), tarefa 3.1.

Cada nó carrega ``(código, pesos, fitness, pai)``. É o "DNA" evolutivo da
VISÃO: um grafo direcionado onde arestas = relação pai→filho (linhagem).

Regra invariável da Fase 3 (FASE3_EVOLUCAO.md §1, reforçada pelo ablation do
Darwin Gödel Machine, arXiv:2505.22954): **NUNCA podar por fitness baixa**.
Ancestrais "ruins" geram descendentes ótimos; hill-climbing puro converte
cedo demais. Por isso ``prune_by_fitness`` existe SÓ para levantar
``PruningForbidden`` — a proibição é estrutural, não uma convenção.

Os pesos são um dict serializável (formato de ``CfCCell.save_brain``:
W_in/W_rec/b/A/mask/tau) e podem ser ``None`` para variantes de só-código.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class PruningForbidden(RuntimeError):
    """Levantado se alguém tentar podar o arquivo por fitness baixa (3.1)."""


@dataclass
class Variant:
    """Um nó do grafo de linhagem."""

    id: str
    code: str
    weights: Optional[dict] = None
    fitness: Optional[float] = None
    parent_id: Optional[str] = None


class Archive:
    """Grafo de linhagem das variantes. Guarda TUDO; nunca poda por fitness."""

    def __init__(self) -> None:
        self._nodes: dict[str, Variant] = {}
        self._seq = 0

    # ------------------------------------------------------------- escrita
    def add(
        self,
        code: str,
        weights: Optional[dict] = None,
        parent_id: Optional[str] = None,
        fitness: Optional[float] = None,
    ) -> str:
        """Insere uma variante e retorna seu id. ``parent_id`` deve existir."""
        if parent_id is not None and parent_id not in self._nodes:
            raise KeyError(f"parent_id {parent_id!r} ausente no arquivo")
        self._seq += 1
        vid = f"v{self._seq}"
        self._nodes[vid] = Variant(
            id=vid, code=code, weights=weights, fitness=fitness, parent_id=parent_id
        )
        return vid

    def set_fitness(self, variant_id: str, fitness: float) -> None:
        self._nodes[variant_id].fitness = float(fitness)

    # --------------------------------------------------------------- leitura
    def get(self, variant_id: str) -> Variant:
        return self._nodes[variant_id]

    def __contains__(self, variant_id: str) -> bool:
        return variant_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def children(self, variant_id: str) -> list[str]:
        return [v.id for v in self._nodes.values() if v.parent_id == variant_id]

    def ancestors(self, variant_id: str) -> list[str]:
        """Caminho raiz→nó (exclusive o próprio nó)."""
        path: list[str] = []
        cur = self._nodes[variant_id].parent_id
        while cur is not None:
            path.append(cur)
            cur = self._nodes[cur].parent_id
        return list(reversed(path))

    def lineage(self, variant_id: str) -> list[str]:
        """Caminho completo raiz→nó (inclusive o próprio nó)."""
        return self.ancestors(variant_id) + [variant_id]

    def best(self, n: Optional[int] = None) -> list[str]:
        """Seleção por fitness (maior primeiro). NÃO remove nada do arquivo."""
        scored = [
            (v.fitness, v.id) for v in self._nodes.values() if v.fitness is not None
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        ids = [vid for _, vid in scored]
        return ids if n is None else ids[:n]

    # ----------------------------------------------------- invariante 3.1
    def prune_by_fitness(self, *args, **kwargs) -> None:
        """Proibido por construção: o arquivo NUNCA poda por fitness baixa."""
        raise PruningForbidden(
            "Arquivo de variantes NUNCA poda por fitness baixa (3.1): "
            "becos sem saida geram os saltos. Use best() para selecionar."
        )
