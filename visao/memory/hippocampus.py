"""
hippocampus.py — Tarefa 13.0: Memory System — Hipocampo do VISÃO.

Implementa memória explícita compatível com o liquid core (Oja + EWC + Surprise):

1. EpisodicBuffer — ring buffer de experiências com consolidação automática.
   Experiências são vetores de estado líquido + metadados. Quando o buffer
   enche, experiências similares são fundidas (consolidação) usando surprise
   como peso de importância — experiências surpreendentes sobrevivem.

2. SemanticGraph — grafo de conhecimento (entidades + relações) com busca por
   similaridade. Entidades são vetores (embeddings); relações são arestas
   tipadas. Busca por similaridade via cosseno. Compatível com EWC: arestas
   importantes (omega alto) resistem a serem sobrescritas.

3. ProceduralMemory — armazenamento de políticas de ação (skills aprendidas).
   Cada skill mapeia um padrão de contexto para uma política de ação, com
   contador de sucesso e timestamp. Consolidação por EWC: skills bem-sucedidas
   ficam rígidas (omega alto); skills fracassadas são podadas.

Referências:
  - ICML 2026: hippocampal explicit memory é essencial para planejamento,
    metacognição e raciocínio simbólico.
  - EWC (Kirkpatrick 2017): consolidação sináptica elástica.
  - Oja (1982): auto-organização Hebbiana normalizada.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Garantir que o prototype/ seja importável
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ==============================================================
#  EPISODIC BUFFER
# ==============================================================

@dataclass
class Experience:
    """Uma experiência individual no buffer episódico.

    Attributes
    ----------
    state : np.ndarray
        Estado do reservatório líquido no momento da experiência.
    action : np.ndarray
        Ação tomada.
    reward : float
        Recompensa recebida.
    next_state : np.ndarray
        Estado resultante após a ação.
    timestamp : int
        Passo temporal da experiência.
    surprise : float
        Fator de surpresa no momento (do gate neuromodulatório).
    importance : float
        Importância acumulada (omega) — cresce com repetição/relevância.
    """
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    timestamp: int = 0
    surprise: float = 1.0
    importance: float = 1.0


class EpisodicBuffer:
    """Ring buffer de experiências com consolidação automática.

    Quando o buffer atinge capacidade, novas experiências substituem as
    menos importantes. Experiências similares (estado próximo) são fundidas
    para evitar redundância — este é o mecanismo de consolidação.

    A consolidação usa:
    - Surpresa como peso: experiências surpreendentes sobrevivem.
    - Similaridade de cosseno: experiências próximas são fundidas (média
      ponderada pela importância).
    - EWC-temporal: importância decai exponencialmente com o tempo
      (experiências antigas e irrelevantes são esquecidas).

    Parameters
    ----------
    capacity : int
        Número máximo de experiências no buffer.
    state_dim : int
        Dimensão do estado do reservatório.
    action_dim : int
        Dimensão da ação.
    consolidation_threshold : float
        Limiar de similaridade para consolidação (0-1). Experiências com
        similaridade acima disso são fundidas.
    lambda_decay : float
        Decaimento temporal de importância (EWC-temporal).
    surprise_gain : float
        Peso da sobrevivência durante consolidação.
    """

    def __init__(
        self,
        capacity: int = 256,
        state_dim: int = 64,
        action_dim: int = 1,
        consolidation_threshold: float = 0.95,
        lambda_decay: float = 0.001,
        surprise_gain: float = 2.0,
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.consolidation_threshold = consolidation_threshold
        self.lambda_decay = lambda_decay
        self.surprise_gain = surprise_gain

        # Ring buffer
        self._buffer: list[Experience] = []
        self._head = 0
        self._size = 0

        # Estatísticas
        self._total_added = 0
        self._total_consolidated = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        timestamp: int = 0,
        surprise: float = 1.0,
    ) -> None:
        """Adiciona uma experiência ao buffer.

        Se o buffer está cheio, aplica consolidação automática:
        1. Se existe experiência similar, funde (média ponderada).
        2. Caso contrário, substitui a menos importante.
        """
        state = np.asarray(state, dtype=np.float64).ravel()
        action = np.asarray(action, dtype=np.float64).ravel()
        next_state = np.asarray(next_state, dtype=np.float64).ravel()

        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            timestamp=timestamp,
            surprise=surprise,
            importance=1.0,
        )

        self._total_added += 1

        if self._size < self.capacity:
            # Buffer não cheio: adiciona direto
            self._buffer.append(exp)
            self._size += 1
        else:
            # Buffer cheio: tenta consolidação
            idx = self._find_similar(state)
            if idx is not None:
                # Funde com experiência similar
                self._consolidate(idx, exp)
            else:
                # Substitui a menos importante
                self._replace_least_important(exp)

    def _find_similar(self, state: np.ndarray) -> Optional[int]:
        """Busca experiência com estado similar (cosseno > threshold)."""
        if self._size == 0:
            return None
        best_idx = None
        best_sim = self.consolidation_threshold
        for i, exp in enumerate(self._buffer):
            sim = self._cosine_sim(state, exp.state)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx

    def _consolidate(self, idx: int, exp: Experience) -> None:
        """Funde experiência existente com nova (média ponderada)."""
        existing = self._buffer[idx]
        w1 = existing.importance
        w2 = exp.importance * (1.0 + self.surprise_gain * max(0, exp.surprise - 1.0))
        total = w1 + w2

        # Média ponderada
        merged = Experience(
            state=(w1 * existing.state + w2 * exp.state) / total,
            action=(w1 * existing.action + w2 * exp.action) / total,
            reward=(w1 * existing.reward + w2 * exp.reward) / total,
            next_state=(w1 * existing.next_state + w2 * exp.next_state) / total,
            timestamp=exp.timestamp,
            surprise=max(existing.surprise, exp.surprise),
            importance=min(total, 10.0),  # cap para evitar explosão
        )
        self._buffer[idx] = merged
        self._total_consolidated += 1

    def _replace_least_important(self, exp: Experience) -> None:
        """Substitui a experiência de menor importância."""
        min_idx = 0
        min_imp = float("inf")
        for i, e in enumerate(self._buffer):
            if e.importance < min_imp:
                min_imp = e.importance
                min_idx = i
        self._buffer[min_idx] = exp

    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> list[Experience]:
        """Amostra n experiências proporcional à importância."""
        rng = rng or np.random.default_rng()
        if self._size == 0:
            return []
        n = min(n, self._size)
        imps = np.array([e.importance for e in self._buffer])
        probs = imps / imps.sum()
        indices = rng.choice(self._size, size=n, replace=False, p=probs)
        return [self._buffer[i] for i in indices]

    def query(self, state: np.ndarray, k: int = 5) -> list[Experience]:
        """Retorna as k experiências mais similares ao estado."""
        state = np.asarray(state, dtype=np.float64).ravel()
        if self._size == 0:
            return []
        sims = [(i, self._cosine_sim(state, e.state)) for i, e in enumerate(self._buffer)]
        sims.sort(key=lambda x: x[1], reverse=True)
        return [self._buffer[i] for i, _ in sims[:k]]

    def decay_importance(self) -> None:
        """Decaimento temporal de importância (EWC-temporal)."""
        for exp in self._buffer:
            exp.importance *= np.exp(-self.lambda_decay)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Similaridade de cosseno entre dois vetores."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity

    def stats(self) -> dict:
        """Estatísticas do buffer."""
        if self._size == 0:
            return {"size": 0, "capacity": self.capacity}
        imps = [e.importance for e in self._buffer]
        surprises = [e.surprise for e in self._buffer]
        return {
            "size": self._size,
            "capacity": self.capacity,
            "total_added": self._total_added,
            "total_consolidated": self._total_consolidated,
            "importance_mean": float(np.mean(imps)),
            "importance_std": float(np.std(imps)),
            "importance_max": float(np.max(imps)),
            "importance_min": float(np.min(imps)),
            "surprise_mean": float(np.mean(surprises)),
        }


# ==============================================================
#  SEMANTIC GRAPH
# ==============================================================

@dataclass
class Entity:
    """Entidade no grafo semântico (nó).

    Attributes
    ----------
    name : str
        Identificador único.
    embedding : np.ndarray
        Vetor de embedding (representação vetorial).
    entity_type : str
        Tipo da entidade (ex: 'concept', 'object', 'relation').
    metadata : dict
        Metadados adicionais.
    importance : float
        Importância acumulada (omega) — EWC.
    """
    name: str
    embedding: np.ndarray
    entity_type: str = "concept"
    metadata: dict = field(default_factory=dict)
    importance: float = 1.0


@dataclass
class Relation:
    """Relação no grafo semântico (aresta).

    Attributes
    ----------
    source : str
        Nome da entidade origem.
    target : str
        Nome da entidade destino.
    relation_type : str
        Tipo da relação (ex: 'is_a', 'part_of', 'causes').
    weight : float
        Peso da relação (força da conexão).
    importance : float
        Importância acumulada (omega) — EWC.
    """
    source: str
    target: str
    relation_type: str = "related"
    weight: float = 1.0
    importance: float = 1.0


class SemanticGraph:
    """Grafo de conhecimento: entidades + relações com busca por similaridade.

    Estrutura:
    - Entidades são nós com embeddings vetoriais.
    - Relações são arestas tipadas e ponderadas.
    - Busca por similaridade via cosseno nos embeddings.
    - EWC protege relações importantes contra sobrescrita.

    Compatível com o liquid core:
    - Embeddings podem vir do estado do reservatório líquido.
    - Importância (omega) cresce com uso e decai com tempo (EWC-temporal).

    Parameters
    ----------
    embedding_dim : int
        Dimensão dos embeddings.
    max_entities : int
        Número máximo de entidades.
    max_relations : int
        Número máximo de relações.
    lambda_decay : float
        Decaimento temporal de importância.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        max_entities: int = 128,
        max_relations: int = 512,
        lambda_decay: float = 0.001,
    ):
        self.embedding_dim = embedding_dim
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.lambda_decay = lambda_decay

        self._entities: OrderedDict[str, Entity] = OrderedDict()
        self._relations: list[Relation] = []
        self._adjacency: dict[str, list[int]] = {}  # entity_name -> relation indices

    def add_entity(
        self,
        name: str,
        embedding: np.ndarray,
        entity_type: str = "concept",
        metadata: dict | None = None,
    ) -> Entity:
        """Adiciona ou atualiza uma entidade.

        Se a entidade já existe, atualiza o embedding (média ponderada
        pela importância — EWC protege embeddings importantes).
        """
        embedding = np.asarray(embedding, dtype=np.float64).ravel()

        if name in self._entities:
            existing = self._entities[name]
            w1 = existing.importance
            w2 = 1.0
            # Média ponderada — EWC: embedding importante resiste
            new_embedding = (w1 * existing.embedding + w2 * embedding) / (w1 + w2)
            existing.embedding = new_embedding
            existing.importance = min(existing.importance + 0.5, 10.0)
            if metadata:
                existing.metadata.update(metadata)
            return existing

        # Capacidade: remove entidade menos importante se cheio
        if len(self._entities) >= self.max_entities:
            self._evict_entity()

        entity = Entity(
            name=name,
            embedding=embedding,
            entity_type=entity_type,
            metadata=metadata or {},
            importance=1.0,
        )
        self._entities[name] = entity
        self._adjacency[name] = []
        return entity

    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str = "related",
        weight: float = 1.0,
    ) -> Relation:
        """Adiciona uma relação entre entidades.

        Se a relação já existe, atualiza o peso (EWC: peso importante resiste).
        """
        # Verifica se entidades existem
        if source not in self._entities:
            raise ValueError(f"Entidade fonte '{source}' não encontrada")
        if target not in self._entities:
            raise ValueError(f"Entidade alvo '{target}' não encontrada")

        # Verifica se relação já existe
        for i, rel in enumerate(self._relations):
            if rel.source == source and rel.target == target and rel.relation_type == relation_type:
                # Atualiza peso (EWC: peso importante resiste)
                w1 = rel.importance
                w2 = 1.0
                rel.weight = (w1 * rel.weight + w2 * weight) / (w1 + w2)
                rel.importance = min(rel.importance + 0.5, 10.0)
                return rel

        # Capacidade: remove relação menos importante se cheio
        if len(self._relations) >= self.max_relations:
            self._evict_relation()

        rel = Relation(
            source=source,
            target=target,
            relation_type=relation_type,
            weight=weight,
            importance=1.0,
        )
        idx = len(self._relations)
        self._relations.append(rel)
        self._adjacency[source].append(idx)
        if target != source:
            self._adjacency[target].append(idx)
        return rel

    def query_similar(
        self,
        embedding: np.ndarray,
        k: int = 5,
        entity_type: str | None = None,
    ) -> list[tuple[str, float]]:
        """Busca as k entidades mais similares ao embedding.

        Returns
        -------
        list of (name, similarity) ordenado por similaridade decrescente.
        """
        embedding = np.asarray(embedding, dtype=np.float64).ravel()
        results = []
        for name, entity in self._entities.items():
            if entity_type and entity.entity_type != entity_type:
                continue
            sim = self._cosine_sim(embedding, entity.embedding)
            results.append((name, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def query_relations(
        self,
        entity_name: str,
        relation_type: str | None = None,
    ) -> list[Relation]:
        """Retorna relações de uma entidade."""
        if entity_name not in self._adjacency:
            return []
        rels = []
        for idx in self._adjacency[entity_name]:
            rel = self._relations[idx]
            if relation_type and rel.relation_type != relation_type:
                continue
            rels.append(rel)
        return rels

    def query_path(
        self,
        source: str,
        target: str,
        max_depth: int = 3,
    ) -> list[list[Relation]]:
        """Busca caminhos entre duas entidades (BFS limitado)."""
        if source not in self._entities or target not in self._entities:
            return []
        paths = []
        self._bfs_paths(source, target, max_depth, [], set(), paths)
        return paths

    def _bfs_paths(
        self,
        current: str,
        target: str,
        depth: int,
        path: list[Relation],
        visited: set,
        results: list[list[Relation]],
    ) -> None:
        """BFS recursivo para encontrar caminhos."""
        if depth <= 0 or current in visited:
            return
        visited.add(current)
        for rel in self.query_relations(current):
            next_node = rel.target if rel.source == current else rel.source
            new_path = path + [rel]
            if next_node == target:
                results.append(new_path)
            else:
                self._bfs_paths(next_node, target, depth - 1, new_path, visited.copy(), results)

    def _evict_entity(self) -> None:
        """Remove a entidade menos importante."""
        if not self._entities:
            return
        min_name = min(self._entities, key=lambda n: self._entities[n].importance)
        # Remove relações associada
        self._relations = [r for r in self._relations if r.source != min_name and r.target != min_name]
        # Reconstrói adjacency
        self._adjacency = {}
        for i, rel in enumerate(self._relations):
            self._adjacency.setdefault(rel.source, []).append(i)
            self._adjacency.setdefault(rel.target, []).append(i)
        del self._entities[min_name]
        self._adjacency.pop(min_name, None)

    def _evict_relation(self) -> None:
        """Remove a relação menos importante."""
        if not self._relations:
            return
        min_idx = min(range(len(self._relations)), key=lambda i: self._relations[i].importance)
        removed = self._relations.pop(min_idx)
        # Reconstrói adjacency
        self._adjacency = {}
        for i, rel in enumerate(self._relations):
            self._adjacency.setdefault(rel.source, []).append(i)
            self._adjacency.setdefault(rel.target, []).append(i)

    def decay_importance(self) -> None:
        """Decaimento temporal de importância (EWC-temporal)."""
        for entity in self._entities.values():
            entity.importance *= np.exp(-self.lambda_decay)
        for rel in self._relations:
            rel.importance *= np.exp(-self.lambda_decay)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Similaridade de cosseno."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @property
    def n_entities(self) -> int:
        return len(self._entities)

    @property
    def n_relations(self) -> int:
        return len(self._relations)

    def stats(self) -> dict:
        """Estatísticas do grafo."""
        if not self._entities:
            return {"n_entities": 0, "n_relations": 0}
        ent_imps = [e.importance for e in self._entities.values()]
        rel_imps = [r.importance for r in self._relations] if self._relations else [0]
        return {
            "n_entities": len(self._entities),
            "n_relations": len(self._relations),
            "entity_importance_mean": float(np.mean(ent_imps)),
            "relation_importance_mean": float(np.mean(rel_imps)),
            "entity_types": {t: sum(1 for e in self._entities.values() if e.entity_type == t)
                           for t in set(e.entity_type for e in self._entities.values())},
        }


# ==============================================================
#  PROCEDURAL MEMORY
# ==============================================================

@dataclass
class Skill:
    """Uma skill (política de ação aprendida).

    Attributes
    ----------
    name : str
        Identificador único da skill.
    context_pattern : np.ndarray
        Padrão de contexto onde a skill se aplica.
    action_policy : np.ndarray
        Política de ação (mapeamento estado -> ação).
    success_count : int
        Número de vezes que a skill teve sucesso.
    total_uses : int
        Total de vezes que a skill foi usada.
    importance : float
        Importância acumulada (omega) — EWC.
    last_used : int
        Timestamp do último uso.
    """
    name: str
    context_pattern: np.ndarray
    action_policy: np.ndarray
    success_count: int = 0
    total_uses: int = 0
    importance: float = 1.0
    last_used: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.success_count / self.total_uses


class ProceduralMemory:
    """Armazenamento de políticas de ação (skills aprendidas).

    Cada skill mapeia um padrão de contexto para uma política de ação.
    Mecanismos:
    - EWC: skills bem-sucedidas (omega alto) resistem a modificação.
    - Consolidação: skills similares são fundidas.
    - Poda: skills com baixa taxa de sucesso são removidas.
    - Surpresa: skills que falham inesperadamente têm importância reduzida.

    Parameters
    ----------
    max_skills : int
        Número máximo de skills.
    context_dim : int
        Dimensão do padrão de contexto.
    action_dim : int
        Dimensão da política de ação.
    consolidation_threshold : float
        Limiar de similaridade para consolidação.
    lambda_decay : float
        Decaimento temporal de importância.
    """

    def __init__(
        self,
        max_skills: int = 64,
        context_dim: int = 64,
        action_dim: int = 1,
        consolidation_threshold: float = 0.92,
        lambda_decay: float = 0.001,
    ):
        self.max_skills = max_skills
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.consolidation_threshold = consolidation_threshold
        self.lambda_decay = lambda_decay

        self._skills: OrderedDict[str, Skill] = OrderedDict()
        self._total_learned = 0

    def learn(
        self,
        name: str,
        context_pattern: np.ndarray,
        action_policy: np.ndarray,
        success: bool = True,
        timestamp: int = 0,
    ) -> Skill:
        """Aprende ou atualiza uma skill.

        Se a skill já existe e o contexto é similar, atualiza a política
        (EWC: política importante resiste). Caso contrário, cria nova.
        """
        context_pattern = np.asarray(context_pattern, dtype=np.float64).ravel()
        action_policy = np.asarray(action_policy, dtype=np.float64).ravel()

        if name in self._skills:
            skill = self._skills[name]
            sim = self._cosine_sim(context_pattern, skill.context_pattern)
            if sim >= self.consolidation_threshold:
                # Atualiza política (EWC: política importante resiste)
                w1 = skill.importance
                w2 = 1.0
                skill.action_policy = (w1 * skill.action_policy + w2 * action_policy) / (w1 + w2)
                skill.context_pattern = (w1 * skill.context_pattern + w2 * context_pattern) / (w1 + w2)
            else:
                # Contexto muito diferente: sobrescreve
                skill.context_pattern = context_pattern
                skill.action_policy = action_policy
            skill.total_uses += 1
            if success:
                skill.success_count += 1
                skill.importance = min(skill.importance + 0.5, 10.0)
            else:
                skill.importance *= 0.9  # Falha reduz importância
            skill.last_used = timestamp
            return skill

        # Capacidade: remove skill menos importante se cheio
        if len(self._skills) >= self.max_skills:
            self._evict_skill()

        skill = Skill(
            name=name,
            context_pattern=context_pattern,
            action_policy=action_policy,
            success_count=1 if success else 0,
            total_uses=1,
            importance=1.0,
            last_used=timestamp,
        )
        self._skills[name] = skill
        self._total_learned += 1
        return skill

    def recall(
        self,
        context_pattern: np.ndarray,
        k: int = 3,
    ) -> list[tuple[str, float, np.ndarray]]:
        """Recupera as k skills mais relevantes para o contexto.

        Returns
        -------
        list of (name, similarity, action_policy) ordenado por similaridade.
        """
        context_pattern = np.asarray(context_pattern, dtype=np.float64).ravel()
        results = []
        for name, skill in self._skills.items():
            sim = self._cosine_sim(context_pattern, skill.context_pattern)
            results.append((name, sim, skill.action_policy))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def execute(self, name: str, timestamp: int = 0) -> np.ndarray | None:
        """Executa uma skill (retorna a política de ação)."""
        if name not in self._skills:
            return None
        skill = self._skills[name]
        skill.total_uses += 1
        skill.last_used = timestamp
        return skill.action_policy.copy()

    def reinforce(self, name: str, success: bool) -> None:
        """Reforça ou punir uma skill baseado no resultado."""
        if name not in self._skills:
            return
        skill = self._skills[name]
        skill.total_uses += 1
        if success:
            skill.success_count += 1
            skill.importance = min(skill.importance + 0.5, 10.0)
        else:
            skill.importance *= 0.9  # Falha reduz importância

    def prune(self, min_success_rate: float = 0.2, min_uses: int = 5) -> int:
        """Remove skills com baixa taxa de sucesso.

        Returns
        -------
        Número de skills removidas.
        """
        to_remove = []
        for name, skill in self._skills.items():
            if skill.total_uses >= min_uses and skill.success_rate < min_success_rate:
                to_remove.append(name)
        for name in to_remove:
            del self._skills[name]
        return len(to_remove)

    def _evict_skill(self) -> None:
        """Remove a skill menos importante."""
        if not self._skills:
            return
        min_name = min(self._skills, key=lambda n: self._skills[n].importance)
        del self._skills[min_name]

    def decay_importance(self) -> None:
        """Decaimento temporal de importância (EWC-temporal)."""
        for skill in self._skills.values():
            skill.importance *= np.exp(-self.lambda_decay)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Similaridade de cosseno."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @property
    def n_skills(self) -> int:
        return len(self._skills)

    def stats(self) -> dict:
        """Estatísticas da memória procedural."""
        if not self._skills:
            return {"n_skills": 0, "total_learned": self._total_learned}
        success_rates = [s.success_rate for s in self._skills.values()]
        importances = [s.importance for s in self._skills.values()]
        return {
            "n_skills": len(self._skills),
            "total_learned": self._total_learned,
            "success_rate_mean": float(np.mean(success_rates)),
            "importance_mean": float(np.mean(importances)),
            "importance_max": float(np.max(importances)),
            "importance_min": float(np.min(importances)),
        }


# ==============================================================
#  HIPPOCAMPUS — Integração
# ==============================================================

class Hippocampus:
    """Hipocampo: integra as três memórias (episódica, semântica, procedural).

    Interface única para o VisaoBrain acessar memória explícita.

    Parameters
    ----------
    state_dim : int
        Dimensão do estado do reservatório líquido.
    action_dim : int
        Dimensão da ação.
    episodic_capacity : int
        Capacidade do buffer episódico.
    max_entities : int
        Máximo de entidades no grafo semântico.
    max_skills : int
        Máximo de skills na memória procedural.
    lambda_decay : float
        Decaimento temporal de importância (EWC-temporal).
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 1,
        episodic_capacity: int = 256,
        max_entities: int = 128,
        max_skills: int = 64,
        lambda_decay: float = 0.001,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.episodic = EpisodicBuffer(
            capacity=episodic_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            lambda_decay=lambda_decay,
        )

        self.semantic = SemanticGraph(
            embedding_dim=state_dim,
            max_entities=max_entities,
            lambda_decay=lambda_decay,
        )

        self.procedural = ProceduralMemory(
            max_skills=max_skills,
            context_dim=state_dim,
            action_dim=action_dim,
            lambda_decay=lambda_decay,
        )

    def encode_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        surprise: float = 1.0,
        timestamp: int = 0,
    ) -> None:
        """Codifica uma experiência no buffer episódico."""
        self.episodic.add(state, action, reward, next_state, timestamp, surprise)

    def encode_knowledge(
        self,
        name: str,
        embedding: np.ndarray,
        entity_type: str = "concept",
        relations: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """Codifica conhecimento no grafo semântico."""
        self.semantic.add_entity(name, embedding, entity_type)
        if relations:
            for source, target, rel_type in relations:
                try:
                    self.semantic.add_relation(source, target, rel_type)
                except ValueError:
                    pass  # Entidade não encontrada

    def encode_skill(
        self,
        name: str,
        context_pattern: np.ndarray,
        action_policy: np.ndarray,
        success: bool = True,
        timestamp: int = 0,
    ) -> None:
        """Codifica uma skill na memória procedural."""
        self.procedural.learn(name, context_pattern, action_policy, success, timestamp)

    def recall_similar_experience(
        self,
        state: np.ndarray,
        k: int = 5,
    ) -> list[Experience]:
        """Recupera experiências similares ao estado."""
        return self.episodic.query(state, k)

    def recall_similar_knowledge(
        self,
        embedding: np.ndarray,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Recupera conhecimento similar ao embedding."""
        return self.semantic.query_similar(embedding, k)

    def recall_skill(
        self,
        context_pattern: np.ndarray,
        k: int = 3,
    ) -> list[tuple[str, float, np.ndarray]]:
        """Recupera skills similares ao contexto."""
        return self.procedural.recall(context_pattern, k)

    def consolidate(self) -> None:
        """Executa consolidação em todas as memórias.

        - EWC-temporal: decaimento de importância.
        - Poda de skills fracas.
        """
        self.episodic.decay_importance()
        self.semantic.decay_importance()
        self.procedural.decay_importance()
        self.procedural.prune()

    def stats(self) -> dict:
        """Estatísticas completas do hipocampo."""
        return {
            "episodic": self.episodic.stats(),
            "semantic": self.semantic.stats(),
            "procedural": self.procedural.stats(),
        }

    def __repr__(self) -> str:
        return (
            f"Hippocampus("
            f"episodic={self.episodic.size}/{self.episodic.capacity}, "
            f"entities={self.semantic.n_entities}, "
            f"skills={self.procedural.n_skills})"
        )
