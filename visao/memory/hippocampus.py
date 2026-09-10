"""
hippocampus.py — Tarefa 13.0: Memory System — Hippocampus.

Implementa três sistemas de memória compatíveis com o liquid core (Oja + EWC + Surprise):

1. EpisodicBuffer — ring buffer de experiências com consolidação automática.
   Experiências são (state, action, reward, next_state, surprise).
   Quando o buffer enche, experiências de baixa importância são consolidadas
   no SemanticGraph (transferência hipocampo → neocórtex).

2. SemanticGraph — grafo de conhecimento (entidades + relações) com busca por
   similaridade. Entidades são embeddings normalizados (Oja-like). Relações
   são arestas tipadas. Busca por similaridade via cosseno.

3. ProceduralMemory — armazenamento de políticas de ação (skills aprendidas).
   Mapeia estados → ações com estimativas de valor. Usa EWC para proteger
   skills importantes e surprise para detectar necessidade de adaptação.

Referências:
    - ICML 2026: hippocampal explicit memory é essencial para planejamento,
      metacognição e raciocínio simbólico.
    - Oja (1982): auto-organização de embeddings.
    - Kirkpatrick et al. (2017): consolidação elástica (EWC).
    - Hassabis et al. (2017): sistemas de memória complementares.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ==============================================================
#  EPISODIC BUFFER
# ==============================================================

@dataclass
class Experience:
    """Uma experiência episódica única."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    surprise: float = 1.0
    timestamp: int = 0
    importance: float = 0.0  # calculado via EWC-like accumulation
    consolidated: bool = False

    def __post_init__(self):
        self.state = np.asarray(self.state, dtype=np.float64).ravel()
        self.next_state = np.asarray(self.next_state, dtype=np.float64).ravel()


class EpisodicBuffer:
    """Ring buffer de experiências com consolidação automática.

    Quando o buffer atinge capacidade máxima, experiências com baixa
    importância (omega) são consolidadas no SemanticGraph e removidas
    do buffer. Experiências com alta importância são retidas por mais
    tempo (consolidação seletiva).

    A importância cresce com a surpresa (experiências inesperadas são
    mais importantes) e decai com o tempo (EWC-temporal).

    Parameters
    ----------
    capacity : int
        Número máximo de experiências no buffer.
    state_dim : int
        Dimensão do vetor de estado.
    importance_decay : float
        Decaimento temporal de importância (EWC-temporal).
    surprise_threshold : float
        Limiar de surpresa para retenção automática.
    consolidate_ratio : float
        Fração de experiências a consolidar quando o buffer enche.
    """

    def __init__(
        self,
        capacity: int = 256,
        state_dim: int = 64,
        importance_decay: float = 0.001,
        surprise_threshold: float = 1.5,
        consolidate_ratio: float = 0.3,
        seed: int = 0,
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.importance_decay = importance_decay
        self.surprise_threshold = surprise_threshold
        self.consolidate_ratio = consolidate_ratio
        self._rng = np.random.default_rng(seed)

        # Ring buffer arrays (evita alocação dinâmica)
        self._states = np.zeros((capacity, state_dim), dtype=np.float64)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float64)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float64)
        self._surprises = np.ones(capacity, dtype=np.float64)
        self._importances = np.zeros(capacity, dtype=np.float64)
        self._consolidated = np.zeros(capacity, dtype=np.bool_)
        self._timestamps = np.zeros(capacity, dtype=np.int64)

        self._size = 0
        self._head = 0
        self._step = 0

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        surprise: float = 1.0,
    ) -> Optional[list[Experience]]:
        """Adiciona experiência ao buffer.

        Se o buffer estiver cheio, consolida experiências antigas
        e retorna a lista de experiências consolidadas (ou None).

        Returns
        -------
        consolidated : list[Experience] | None
            Experiências que foram consolidadas, ou None.
        """
        idx = self._head

        self._states[idx] = np.asarray(state, dtype=np.float64).ravel()[:self.state_dim]
        self._next_states[idx] = np.asarray(next_state, dtype=np.float64).ravel()[:self.state_dim]
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._surprises[idx] = surprise
        self._timestamps[idx] = self._step
        self._consolidated[idx] = False

        # Importância inicial = surpresa (experiências inesperadas são importantes)
        self._importances[idx] = surprise

        self._head = (self._head + 1) % self.capacity
        self._step += 1

        if self._size < self.capacity:
            self._size += 1
            return None

        # Buffer cheio: consolidar
        return self._consolidate()

    def _consolidate(self) -> list[Experience]:
        """Consolida experiências de baixa importância.

        Estratégia:
        1. Aplica decaimento temporal (EWC-temporal).
        2. Seleciona experiências com menor importância.
        3. Retorna as selecionadas para transferência ao SemanticGraph.
        """
        # 1. Decaimento temporal
        self._importances[:self._size] *= np.exp(-self.importance_decay)

        # 2. Selecionar experiências para consolidação
        n_consolidate = max(1, int(self._size * self.consolidate_ratio))
        # Experiências com baixa importância E baixa surpresa
        priority = self._importances[:self._size] + self._surprises[:self._size] * 0.1
        consolidate_indices = np.argsort(priority)[:n_consolidate]

        consolidated = []
        for idx in consolidate_indices:
            exp = Experience(
                state=self._states[idx].copy(),
                action=int(self._actions[idx]),
                reward=float(self._rewards[idx]),
                next_state=self._next_states[idx].copy(),
                surprise=float(self._surprises[idx]),
                timestamp=int(self._timestamps[idx]),
                importance=float(self._importances[idx]),
                consolidated=True,
            )
            consolidated.append(exp)
            self._consolidated[idx] = True

        return consolidated

    def sample(self, n: int = 1) -> list[Experience]:
        """Amostra n experiências do buffer (uniforme ou por importância)."""
        if self._size == 0:
            return []
        n = min(n, self._size)
        # Amostragem por importância (prioritized sampling)
        weights = self._importances[:self._size] + 0.01
        probs = weights / weights.sum()
        indices = self._rng.choice(self._size, size=n, replace=False, p=probs)
        return [
            Experience(
                state=self._states[i].copy(),
                action=int(self._actions[i]),
                reward=float(self._rewards[i]),
                next_state=self._next_states[i].copy(),
                surprise=float(self._surprises[i]),
                timestamp=int(self._timestamps[i]),
                importance=float(self._importances[i]),
            )
            for i in indices
        ]

    def recent(self, n: int = 1) -> list[Experience]:
        """Retorna as n experiências mais recentes."""
        if self._size == 0:
            return []
        n = min(n, self._size)
        indices = [(self._head - 1 - i) % self.capacity for i in range(n)]
        return [
            Experience(
                state=self._states[i].copy(),
                action=int(self._actions[i]),
                reward=float(self._rewards[i]),
                next_state=self._next_states[i].copy(),
                surprise=float(self._surprises[i]),
                timestamp=int(self._timestamps[i]),
                importance=float(self._importances[i]),
            )
            for i in indices
        ]

    def update_importance(self, index: int, delta: float) -> None:
        """Atualiza importância de uma experiência (crescimento EWC-like)."""
        if 0 <= index < self._size:
            self._importances[index] += abs(delta)

    def get_stats(self) -> dict:
        """Estatísticas do buffer."""
        if self._size == 0:
            return {"size": 0, "capacity": self.capacity}
        return {
            "size": self._size,
            "capacity": self.capacity,
            "mean_importance": float(np.mean(self._importances[:self._size])),
            "mean_surprise": float(np.mean(self._surprises[:self._size])),
            "mean_reward": float(np.mean(self._rewards[:self._size])),
            "n_consolidated": int(np.sum(self._consolidated[:self._size])),
        }


# ==============================================================
#  SEMANTIC GRAPH
# ==============================================================

@dataclass
class Entity:
    """Nó do grafo semântico."""
    id: int
    embedding: np.ndarray
    label: str = ""
    count: int = 1  # quantas vezes foi reforçado (Oja)
    importance: float = 0.0  # EWC-like

    def __post_init__(self):
        self.embedding = np.asarray(self.embedding, dtype=np.float64).ravel()


@dataclass
class Relation:
    """Aresta do grafo semântico."""
    source_id: int
    target_id: int
    relation_type: str
    weight: float = 1.0
    count: int = 1


class SemanticGraph:
    """Grafo de conhecimento: entidades + relações com busca por similaridade.

    Entidades são embeddings normalizados (Oja-like: reforço Hebbiano com
    decaimento normalizador). Relações são arestas tipadas com peso.

    A busca por similaridade usa cosseno entre embeddings. A consolidação
    de experiências episódicas cria novas entidades/relações ou reforça
    as existentes.

    Parameters
    ----------
    embedding_dim : int
        Dimensão dos embeddings de entidades.
    max_entities : int
        Número máximo de entidades (apaga menos importantes quando chega ao limite).
    similarity_threshold : float
        Limiar de cosseno para considerar duas entidades similares.
    oja_lr : float
        Taxa de aprendizado Oja para reforço de embeddings.
    ewc_decay : float
        Decaimento temporal de importância (EWC-temporal).
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        max_entities: int = 512,
        similarity_threshold: float = 0.85,
        oja_lr: float = 0.01,
        ewc_decay: float = 0.0005,
        seed: int = 0,
    ):
        self.embedding_dim = embedding_dim
        self.max_entities = max_entities
        self.similarity_threshold = similarity_threshold
        self.oja_lr = oja_lr
        self.ewc_decay = ewc_decay
        self._rng = np.random.default_rng(seed)

        self.entities: dict[int, Entity] = {}
        self.relations: list[Relation] = []
        self._next_id = 0

    @property
    def n_entities(self) -> int:
        return len(self.entities)

    @property
    def n_relations(self) -> int:
        return len(self.relations)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similaridade cosseno entre dois vetores."""
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _normalize_embedding(self, emb: np.ndarray) -> np.ndarray:
        """Normaliza embedding (Oja: mantém norma estável)."""
        emb = np.asarray(emb, dtype=np.float64).ravel()
        norm = np.linalg.norm(emb)
        if norm < 1e-10:
            return emb
        return emb / norm

    def add_entity(
        self,
        embedding: np.ndarray,
        label: str = "",
        importance: float = 1.0,
    ) -> int:
        """Adiciona entidade ao grafo. Retorna o ID."""
        embedding = self._normalize_embedding(embedding)

        # Verificar se já existe entidade similar
        similar_id = self.find_similar(embedding)
        if similar_id is not None:
            # Reforço Oja: atualiza embedding existente
            self._reinforce_entity(similar_id, embedding, importance)
            return similar_id

        # Criar nova entidade
        eid = self._next_id
        self._next_id += 1
        self.entities[eid] = Entity(
            id=eid,
            embedding=embedding.copy(),
            label=label,
            importance=importance,
        )

        # Se excedeu capacidade, remove menos importante
        if len(self.entities) > self.max_entities:
            self._prune_entity()

        return eid

    def _reinforce_entity(self, eid: int, new_emb: np.ndarray, importance: float) -> None:
        """Reforça entidade existente (Oja: Hebb + decaimento normalizador)."""
        entity = self.entities[eid]
        old_emb = entity.embedding

        # Oja update: Δw = η * (x - w * (w·x))
        # Equivale a mover na direção do input, mas puxando de volta pela norma
        dot = np.dot(old_emb, new_emb)
        delta = self.oja_lr * (new_emb - old_emb * dot)
        new_embedding = old_emb + delta
        entity.embedding = self._normalize_embedding(new_embedding)
        entity.count += 1
        entity.importance += importance

    def _prune_entity(self) -> None:
        """Remove entidade menos importante (EWC: mantém as importantes)."""
        if not self.entities:
            return
        min_id = min(self.entities, key=lambda k: self.entities[k].importance)
        del self.entities[min_id]
        # Remove relações associadas
        self.relations = [
            r for r in self.relations
            if r.source_id != min_id and r.target_id != min_id
        ]

    def add_relation(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        """Adiciona relação (aresta) entre entidades."""
        if source_id not in self.entities or target_id not in self.entities:
            return
        # Verificar se já existe relação similar
        for r in self.relations:
            if (r.source_id == source_id and r.target_id == target_id
                    and r.relation_type == relation_type):
                r.weight = min(r.weight + weight, 10.0)  # clamp
                r.count += 1
                return
        self.relations.append(Relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
        ))

    def find_similar(self, embedding: np.ndarray) -> Optional[int]:
        """Busca entidade mais similar (cosseno acima do limiar)."""
        if not self.entities:
            return None
        embedding = np.asarray(embedding, dtype=np.float64).ravel()
        best_id = None
        best_sim = self.similarity_threshold
        for eid, entity in self.entities.items():
            sim = self._cosine_similarity(embedding, entity.embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = eid
        return best_id

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Busca top-k entidades mais similares à query."""
        if not self.entities:
            return []
        query = np.asarray(query, dtype=np.float64).ravel()
        scores = [
            (eid, self._cosine_similarity(query, entity.embedding))
            for eid, entity in self.entities.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def consolidate_experience(self, exp: Experience) -> tuple[int, int]:
        """Consolida experiência episódica no grafo semântico.

        Cria entidades para state e next_state, e uma relação 'action'
        entre elas. Retorna (source_id, target_id).
        """
        source_id = self.add_entity(
            exp.state,
            label=f"state_t{exp.timestamp}",
            importance=exp.importance,
        )
        target_id = self.add_entity(
            exp.next_state,
            label=f"state_t{exp.timestamp+1}",
            importance=exp.importance,
        )
        self.add_relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=f"action_{exp.action}",
            weight=exp.reward,
        )
        return source_id, target_id

    def temporal_decay(self) -> None:
        """Decaimento temporal de importância (EWC-temporal)."""
        for entity in self.entities.values():
            entity.importance *= np.exp(-self.ewc_decay)

    def get_entity(self, eid: int) -> Optional[Entity]:
        """Retorna entidade por ID."""
        return self.entities.get(eid)

    def get_neighbors(self, eid: int) -> list[tuple[int, str, float]]:
        """Retorna vizinhos de uma entidade: (neighbor_id, relation_type, weight)."""
        neighbors = []
        for r in self.relations:
            if r.source_id == eid and r.target_id in self.entities:
                neighbors.append((r.target_id, r.relation_type, r.weight))
            elif r.target_id == eid and r.source_id in self.entities:
                neighbors.append((r.source_id, r.relation_type, r.weight))
        return neighbors

    def get_stats(self) -> dict:
        """Estatísticas do grafo."""
        if not self.entities:
            return {"n_entities": 0, "n_relations": 0}
        importances = [e.importance for e in self.entities.values()]
        return {
            "n_entities": len(self.entities),
            "n_relations": len(self.relations),
            "mean_importance": float(np.mean(importances)),
            "max_importance": float(np.max(importances)),
            "mean_count": float(np.mean([e.count for e in self.entities.values()])),
        }


# ==============================================================
#  PROCEDURAL MEMORY
# ==============================================================

class ProceduralMemory:
    """Memória procedural: armazena políticas de ação (skills aprendidas).

    Mapeia estados → ações com estimativas de valor (Q-learning-like).
    Usa EWC para proteger skills importantes e surprise para detectar
    necessidade de adaptação.

    A política é representada como uma tabela de valores Q(s, a) onde
    s é o índice de uma entidade no SemanticGraph e a é a ação.

    Parameters
    ----------
    n_actions : int
        Número de ações possíveis.
    embedding_dim : int
        Dimensão dos embeddings de estado.
    lr : float
        Taxa de aprendizado base.
    gamma : float
        Fator de desconto.
    consolidation : float
        Força da consolidação (EWC).
    surprise_gain : float
        Sensibilidade do gate de surpresa.
    ewc_decay : float
        Decaimento temporal de importância.
    """

    def __init__(
        self,
        n_actions: int = 4,
        embedding_dim: int = 64,
        lr: float = 0.01,
        gamma: float = 0.95,
        consolidation: float = 2.0,
        surprise_gain: float = 3.0,
        ewc_decay: float = 0.001,
        clip_weight: float = 10.0,
        seed: int = 0,
    ):
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.gamma = gamma
        self.consolidation = consolidation
        self.surprise_gain = surprise_gain
        self.ewc_decay = ewc_decay
        self.clip_weight = clip_weight
        self._rng = np.random.default_rng(seed)

        # Política: rede linear simples (estado → Q-values)
        scale = 1.0 / np.sqrt(embedding_dim)
        self.W = self._rng.normal(0, scale, (n_actions, embedding_dim))
        self.b = np.zeros(n_actions)

        # Importância sináptica (EWC)
        self.omega = np.zeros((n_actions, embedding_dim))

        # Baseline de surpresa
        self.err_ema = 1.0
        self.err_var = 1.0

        # Contador de usos por ação (para stats)
        self.action_counts = np.zeros(n_actions, dtype=np.int64)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Prediz Q-values para um estado."""
        state = np.asarray(state, dtype=np.float64).ravel()
        return self.W @ state + self.b

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Seleciona ação (epsilon-greedy)."""
        if self._rng.random() < epsilon:
            return int(self._rng.integers(self.n_actions))
        q_values = self.predict(state)
        return int(np.argmax(q_values))

    def surprise(self, err_mag: float) -> float:
        """Calcula surpresa (gate neuromodulatório)."""
        z = (err_mag - self.err_ema) / (np.sqrt(self.err_var) + 1e-8)
        return float(1.0 + self.surprise_gain * max(0.0, np.tanh(z)))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> dict:
        """Atualiza política com uma transição.

        Usa Q-learning com:
        - lr modulado por surpresa e consolidação (EWC)
        - Surpresa como decaimento de omega (surprise_decay)
        - Decaimento temporal de importância (EWC-temporal)

        Returns
        -------
        dict com 'td_error', 'surprise', 'lr_effective'
        """
        state = np.asarray(state, dtype=np.float64).ravel()
        next_state = np.asarray(next_state, dtype=np.float64).ravel()

        # Q-values atuais
        q_values = self.predict(state)
        q_current = q_values[action]

        # Target
        if done:
            target = reward
        else:
            q_next = self.predict(next_state)
            target = reward + self.gamma * np.max(q_next)

        # TD error
        td_error = target - q_current
        err_mag = abs(td_error)

        # Surpresa
        s = self.surprise(err_mag)

        # lr efetivo: aberto pela surpresa, fechado pela importância (EWC)
        eff_lr = (self.lr * s) / (1.0 + self.consolidation * self.omega[action])

        # Atualização delta rule
        delta = td_error * state
        self.W[action] += eff_lr * delta
        self.b[action] += self.lr * s * td_error

        # Clip pesos para estabilidade numérica
        np.clip(self.W, -self.clip_weight, self.clip_weight, out=self.W)
        np.clip(self.b, -self.clip_weight, self.clip_weight, out=self.b)

        # Crescimento de importância (EWC)
        self.omega[action] += 0.01 * abs(delta)

        # Decaimento temporal (EWC-temporal)
        if self.ewc_decay > 0:
            self.omega *= np.exp(-self.ewc_decay)

        # Surpresa como decaimento de omega (surprise_decay)
        if s > 1.0:
            decay = np.exp(-(s - 1.0) * 0.5)
            self.omega[action] *= decay

        # Atualização dos baselines de surpresa (Welford)
        d = err_mag - self.err_ema
        self.err_ema += 0.02 * d
        self.err_var += 0.02 * (d * d - self.err_var)

        self.action_counts[action] += 1

        return {
            "td_error": float(td_error),
            "surprise": s,
            "lr_effective": float(np.mean(eff_lr)),
        }

    def get_skill_importance(self) -> np.ndarray:
        """Retorna importância média por ação (EWC omega)."""
        return np.mean(self.omega, axis=1)

    def get_stats(self) -> dict:
        """Estatísticas da memória procedural."""
        return {
            "omega_mean": float(np.mean(self.omega)),
            "omega_max": float(np.max(self.omega)),
            "omega_per_action": np.mean(self.omega, axis=1).tolist(),
            "action_counts": self.action_counts.tolist(),
            "err_ema": self.err_ema,
            "W_norm": float(np.linalg.norm(self.W)),
        }


# ==============================================================
#  HIPPOCAMPUS — integração das 3 memórias
# ==============================================================

class Hippocampus:
    """Integra EpisodicBuffer, SemanticGraph e ProceduralMemory.

    O Hipocampo coordena a transferência de informações entre
    os três sistemas de memória:
    - Experiências novas entram no EpisodicBuffer
    - Quando consolidadas, viram entidades/relações no SemanticGraph
    - Skills aprendidas ficam na ProceduralMemory

    Parameters
    ----------
    state_dim : int
        Dimensão do espaço de estados.
    n_actions : int
        Número de ações possíveis.
    episodic_capacity : int
        Capacidade do buffer episódico.
    max_entities : int
        Máximo de entidades no grafo semântico.
    """

    def __init__(
        self,
        state_dim: int = 64,
        n_actions: int = 4,
        episodic_capacity: int = 256,
        max_entities: int = 512,
        seed: int = 0,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.episodic = EpisodicBuffer(
            capacity=episodic_capacity,
            state_dim=state_dim,
            seed=seed,
        )
        self.semantic = SemanticGraph(
            embedding_dim=state_dim,
            max_entities=max_entities,
            seed=seed,
        )
        self.procedural = ProceduralMemory(
            n_actions=n_actions,
            embedding_dim=state_dim,
            seed=seed,
        )

        self._step = 0

    def encode(self, state: np.ndarray, surprise: float = 1.0) -> int:
        """Codifica estado no grafo semântico (retorna entity ID)."""
        return self.semantic.add_entity(state, importance=surprise)

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        surprise: float = 1.0,
    ) -> dict:
        """Armazena experiência nas 3 memórias.

        1. Adiciona ao EpisodicBuffer
        2. Se buffer consolidou, transfere para SemanticGraph
        3. Atualiza ProceduralMemory (política)

        Returns
        -------
        dict com info sobre o que foi feito.
        """
        result = {"consolidated": False, "n_consolidated": 0}

        # 1. Episodic buffer
        consolidated = self.episodic.add(state, action, reward, next_state, surprise)

        # 2. Consolidação → SemanticGraph
        if consolidated is not None:
            for exp in consolidated:
                self.semantic.consolidate_experience(exp)
            result["consolidated"] = True
            result["n_consolidated"] = len(consolidated)

        # 3. Procedural memory update
        proc_result = self.procedural.update(state, action, reward, next_state)
        result.update(proc_result)

        # 4. Decay temporal no grafo
        self.semantic.temporal_decay()

        self._step += 1
        return result

    def recall(self, query: np.ndarray, k: int = 5) -> dict:
        """Recupera informações relevantes para uma query.

        Busca no SemanticGraph (entidades similares) e no EpisodicBuffer
        (experiências recentes).
        """
        semantic_results = self.semantic.search(query, k=k)
        recent_episodes = self.episodic.recent(n=min(k, 5))

        return {
            "semantic_matches": semantic_results,
            "recent_episodes": recent_episodes,
        }

    def decide(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Decide ação baseada na memória procedural."""
        return self.procedural.select_action(state, epsilon)

    def get_state_embedding(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Retorna embedding semântico de um estado (se existir)."""
        eid = self.semantic.find_similar(state)
        if eid is not None:
            entity = self.semantic.get_entity(eid)
            if entity is not None:
                return entity.embedding
        return None

    def get_stats(self) -> dict:
        """Estatísticas agregadas das 3 memórias."""
        return {
            "step": self._step,
            "episodic": self.episodic.get_stats(),
            "semantic": self.semantic.get_stats(),
            "procedural": self.procedural.get_stats(),
        }
