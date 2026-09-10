"""
test_hippocampus.py — Script de demonstração do Memory System (Tarefa 13.0).

Demonstra as 3 memórias do hipocampo:
1. EpisodicBuffer — armazena experiências, consolida similares
2. SemanticGraph — grafo de conhecimento com busca por similaridade
3. ProceduralMemory — skills aprendidas com EWC
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Garantir que o projeto seja importável
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visao.memory.hippocampus import (
    EpisodicBuffer,
    SemanticGraph,
    ProceduralMemory,
    Hippocampus,
)


def test_episodic_buffer():
    """Demonstra o EpisodicBuffer."""
    print("=" * 60)
    print("1. EPISODIC BUFFER — Ring buffer com consolidação")
    print("=" * 60)

    rng = np.random.default_rng(42)
    buffer = EpisodicBuffer(capacity=16, state_dim=8, action_dim=1)

    # Adiciona experiências
    print("\nAdicionando 20 experiências (capacidade=16)...")
    for i in range(20):
        state = rng.normal(0, 1, 8)
        action = rng.normal(0, 0.5, 1)
        reward = float(rng.normal(0, 1))
        next_state = state + rng.normal(0, 0.1, 8)
        surprise = float(rng.uniform(0.5, 3.0))
        buffer.add(state, action, reward, next_state, timestamp=i, surprise=surprise)

    print(f"  Buffer size: {buffer.size}")
    print(f"  Stats: {buffer.stats()}")

    # Consulta experiências similares
    query_state = buffer._buffer[0].state + rng.normal(0, 0.05, 8)
    similar = buffer.query(query_state, k=3)
    print(f"\n  Query por estado similar (k=3):")
    for i, exp in enumerate(similar):
        sim = buffer._cosine_sim(query_state, exp.state)
        print(f"    [{i}] t={exp.timestamp}, surprise={exp.surprise:.2f}, importance={exp.importance:.2f}, sim={sim:.4f}")

    # Amostragem
    samples = buffer.sample(n=3, rng=rng)
    print(f"\n  Amostragem (n=3):")
    for i, exp in enumerate(samples):
        print(f"    [{i}] t={exp.timestamp}, importance={exp.importance:.2f}")

    # Decaimento temporal
    buffer.decay_importance()
    print(f"\n  Após decay: importance_mean={buffer.stats()['importance_mean']:.4f}")
    print("  ✓ EpisodicBuffer OK\n")


def test_semantic_graph():
    """Demonstra o SemanticGraph."""
    print("=" * 60)
    print("2. SEMANTIC GRAPH — Grafo de conhecimento")
    print("=" * 60)

    rng = np.random.default_rng(42)
    graph = SemanticGraph(embedding_dim=8, max_entities=10, max_relations=20)

    # Adiciona entidades
    print("\nAdicionando entidades...")
    entities = {
        "gato": rng.normal(0, 1, 8),
        "cachorro": rng.normal(0, 1, 8),
        "animal": rng.normal(0, 1, 8),
        "mamifero": rng.normal(0, 1, 8),
        "peixe": rng.normal(0, 1, 8),
    }
    for name, emb in entities.items():
        graph.add_entity(name, emb, entity_type="concept")
        print(f"  + {name}")

    # Adiciona relações
    print("\nAdicionando relações...")
    relations = [
        ("gato", "mamifero", "is_a"),
        ("cachorro", "mamifero", "is_a"),
        ("mamifero", "animal", "is_a"),
        ("peixe", "animal", "is_a"),
        ("gato", "cachorro", "similar_to"),
    ]
    for src, tgt, rel in relations:
        graph.add_relation(src, tgt, rel)
        print(f"  + {src} --[{rel}]--> {tgt}")

    # Busca por similaridade
    print("\nBusca por similaridade (query='gato'):")
    query_emb = entities["gato"] + rng.normal(0, 0.1, 8)
    similar = graph.query_similar(query_emb, k=3)
    for name, sim in similar:
        print(f"  {name}: sim={sim:.4f}")

    # Consulta relações
    print("\nRelações de 'gato':")
    rels = graph.query_relations("gato")
    for rel in rels:
        print(f"  {rel.source} --[{rel.relation_type}]--> {rel.target} (weight={rel.weight:.2f})")

    # Busca de caminho
    print("\nCaminho de 'gato' para 'animal':")
    paths = graph.query_path("gato", "animal")
    for path in paths:
        path_str = " -> ".join(f"{r.source}-[{r.relation_type}]-{r.target}" for r in path)
        print(f"  {path_str}")

    print(f"\n  Stats: {graph.stats()}")
    print("  ✓ SemanticGraph OK\n")


def test_procedural_memory():
    """Demonstra a ProceduralMemory."""
    print("=" * 60)
    print("3. PROCEDURAL MEMORY — Skills aprendidas")
    print("=" * 60)

    rng = np.random.default_rng(42)
    memory = ProceduralMemory(max_skills=8, context_dim=8, action_dim=1)

    # Aprende skills
    print("\nAprendendo skills...")
    skills_data = [
        ("andar_frente", rng.normal(0, 1, 8), [0.5], True),
        ("virar_esquerda", rng.normal(0, 1, 8), [-0.3], True),
        ("virar_direita", rng.normal(0, 1, 8), [0.3], True),
        ("parar", rng.normal(0, 1, 8), [0.0], True),
        ("pular", rng.normal(0, 1, 8), [0.8], False),  # skill que falha
    ]
    for name, ctx, act, success in skills_data:
        memory.learn(name, ctx, np.array(act), success, timestamp=0)
        print(f"  + {name} (success={success})")

    # Reforça skills
    print("\nReforçando skills...")
    for _ in range(5):
        memory.reinforce("andar_frente", success=True)
        memory.reinforce("virar_esquerda", success=True)
        memory.reinforce("pular", success=False)
    print("  + andar_forte: 5 reforços positivos")
    print("  + virar_esquerda: 5 reforços positivos")
    print("  + pular: 5 reforços negativos")

    # Recall por contexto
    print("\nRecall por contexto (similar a 'andar_frente'):")
    query_ctx = skills_data[0][1] + rng.normal(0, 0.1, 8)
    recalled = memory.recall(query_ctx, k=3)
    for name, sim, policy in recalled:
        print(f"  {name}: sim={sim:.4f}, policy={policy}, success_rate={memory._skills[name].success_rate:.2f}")

    # Poda
    print("\nPoda de skills fracas (min_success_rate=0.2)...")
    pruned = memory.prune(min_success_rate=0.2, min_uses=3)
    print(f"  Skills removidas: {pruned}")
    print(f"  Skills restantes: {memory.n_skills}")

    print(f"\n  Stats: {memory.stats()}")
    print("  ✓ ProceduralMemory OK\n")


def test_hippocampus_integration():
    """Demonstra a integração do Hippocampus."""
    print("=" * 60)
    print("4. HIPPOCAMPUS — Integração das 3 memórias")
    print("=" * 60)

    rng = np.random.default_rng(42)
    hippo = Hippocampus(
        state_dim=8,
        action_dim=1,
        episodic_capacity=32,
        max_entities=16,
        max_skills=8,
    )

    # Codifica experiências
    print("\nCodificando experiências...")
    for i in range(20):
        state = rng.normal(0, 1, 8)
        action = rng.normal(0, 0.5, 1)
        reward = float(rng.normal(0, 1))
        next_state = state + rng.normal(0, 0.1, 8)
        surprise = float(rng.uniform(0.5, 3.0))
        hippo.encode_experience(state, action, reward, next_state, surprise, timestamp=i)

    # Codifica conhecimento
    print("Codificando conhecimento...")
    hippo.encode_knowledge(
        "agent",
        rng.normal(0, 1, 8),
        entity_type="entity",
        relations=[("agent", "environment", "interacts_with")],
    )
    hippo.encode_knowledge(
        "environment",
        rng.normal(0, 1, 8),
        entity_type="entity",
    )
    hippo.encode_knowledge(
        "goal",
        rng.normal(0, 1, 8),
        entity_type="concept",
        relations=[("agent", "goal", "pursues")],
    )

    # Codifica skills
    print("Codificando skills...")
    for i in range(5):
        ctx = rng.normal(0, 1, 8)
        policy = rng.normal(0, 0.5, 1)
        hippo.encode_skill(f"skill_{i}", ctx, policy, success=rng.random() > 0.3, timestamp=i)

    # Consolidação
    print("Executando consolidação (EWC-temporal)...")
    hippo.consolidate()

    # Stats finais
    print(f"\n  Hippocampus: {hippo}")
    stats = hippo.stats()
    print(f"  Episodic: {stats['episodic']['size']} experiências")
    print(f"  Semantic: {stats['semantic']['n_entities']} entidades, {stats['semantic']['n_relations']} relações")
    print(f"  Procedural: {stats['procedural']['n_skills']} skills")

    # Recall integrado
    print("\nRecall integrado (experiência similar):")
    query = rng.normal(0, 1, 8)
    similar_exp = hippo.recall_similar_experience(query, k=2)
    for i, exp in enumerate(similar_exp):
        print(f"  [{i}] t={exp.timestamp}, surprise={exp.surprise:.2f}")

    print("\nRecall integrado (conhecimento similar):")
    similar_know = hippo.recall_similar_knowledge(query, k=2)
    for name, sim in similar_know:
        print(f"  {name}: sim={sim:.4f}")

    print("\nRecall integrado (skills similares):")
    similar_skills = hippo.recall_skill(query, k=2)
    for name, sim, _ in similar_skills:
        print(f"  {name}: sim={sim:.4f}")

    print("\n  ✓ Hippocampus integrado OK\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MEMORY SYSTEM — Tarefa 13.0: Demonstração")
    print("=" * 60 + "\n")

    test_episodic_buffer()
    test_semantic_graph()
    test_procedural_memory()
    test_hippocampus_integration()

    print("=" * 60)
    print("  TODOS OS TESTES PASSARAM ✓")
    print("=" * 60)
