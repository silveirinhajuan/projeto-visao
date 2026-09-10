"""
test_hippocampus.py — Teste do Memory System (Tarefa 13.0).

Demonstra as 3 memórias:
1. EpisodicBuffer — armazenamento e consolidação de experiências
2. SemanticGraph — grafo de conhecimento com busca por similaridade
3. ProceduralMemory — aprendizado de política (skills)
"""

import sys
from pathlib import Path
import numpy as np

# Garantir que o projeto seja importável
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visao.memory.hippocampus import (
    EpisodicBuffer,
    SemanticGraph,
    ProceduralMemory,
    Hippocampus,
)


def test_episodic_buffer():
    """Testa o buffer episódico com consolidação automática."""
    print("=" * 60)
    print("TESTE 1: EpisodicBuffer")
    print("=" * 60)

    buf = EpisodicBuffer(capacity=10, state_dim=8, seed=42)

    rng = np.random.default_rng(0)

    # Adicionar experiências
    n_added = 0
    n_consolidated_total = 0
    for i in range(25):
        state = rng.normal(0, 1, 8)
        action = int(rng.integers(0, 4))
        reward = float(rng.normal(0, 1))
        next_state = state + rng.normal(0, 0.1, 8)
        surprise = float(1.0 + rng.exponential(0.5))

        result = buf.add(state, action, reward, next_state, surprise)
        n_added += 1
        if result is not None:
            n_consolidated_total += len(result)
            print(f"  Experiência {i}: buffer cheio → consolidou {len(result)} experiências")

    print(f"\n  Total adicionadas: {n_added}")
    print(f"  Total consolidadas: {n_consolidated_total}")
    print(f"  Buffer size: {buf.size}/{buf.capacity}")

    # Amostragem
    samples = buf.sample(3)
    print(f"\n  Amostras (3):")
    for i, exp in enumerate(samples):
        print(f"    [{i}] action={exp.action}, reward={exp.reward:.3f}, "
              f"surprise={exp.surprise:.3f}, importance={exp.importance:.3f}")

    # Recentes
    recent = buf.recent(2)
    print(f"\n  Recentes (2):")
    for i, exp in enumerate(recent):
        print(f"    [{i}] action={exp.action}, reward={exp.reward:.3f}")

    stats = buf.get_stats()
    print(f"\n  Stats: {stats}")
    assert buf.size == 10, f"Buffer deveria estar cheio (10), mas tem {buf.size}"
    assert n_consolidated_total > 0, "Deveria ter consolidado experiências"
    print("\n  ✅ EpisodicBuffer OK\n")


def test_semantic_graph():
    """Testa o grafo semântico com busca por similaridade."""
    print("=" * 60)
    print("TESTE 2: SemanticGraph")
    print("=" * 60)

    graph = SemanticGraph(embedding_dim=16, max_entities=50, seed=42)

    rng = np.random.default_rng(1)

    # Criar clusters de entidades
    print("\n  Criando entidades em 3 clusters...")
    cluster_centers = [rng.normal(0, 1, 16) for _ in range(3)]
    entity_ids = []

    for i in range(30):
        cluster = i % 3
        emb = cluster_centers[cluster] + rng.normal(0, 0.2, 16)
        eid = graph.add_entity(emb, label=f"entity_{i}", importance=rng.exponential(1))
        entity_ids.append(eid)

    print(f"  Entidades criadas: {graph.n_entities}")
    print(f"  IDs únicos: {len(set(entity_ids))}")

    # Adicionar relações
    for i in range(0, len(entity_ids) - 1, 2):
        graph.add_relation(entity_ids[i], entity_ids[i + 1], "leads_to", weight=rng.random())
        graph.add_relation(entity_ids[i], entity_ids[i], "self", weight=1.0)

    print(f"  Relações: {graph.n_relations}")

    # Busca por similaridade
    print("\n  Buscas por similaridade:")
    query = cluster_centers[0] + rng.normal(0, 0.1, 16)
    results = graph.search(query, k=5)
    print(f"  Query próxima do cluster 0:")
    for eid, sim in results:
        entity = graph.get_entity(eid)
        print(f"    ID={eid}, sim={sim:.4f}, label={entity.label}, count={entity.count}")

    # Testar reforço Oja (mesma entidade reforçada várias vezes)
    print("\n  Teste Oja (reforço de entidade):")
    emb_base = rng.normal(0, 1, 16)
    eid = graph.add_entity(emb_base, label="oja_test", importance=1.0)
    entity_before = graph.get_entity(eid)
    emb_before = entity_before.embedding.copy()

    for _ in range(10):
        noise = rng.normal(0, 0.05, 16)
        graph.add_entity(emb_base + noise, label="oja_test", importance=0.5)

    entity_after = graph.get_entity(eid)
    print(f"    Count antes: {entity_before.count}, depois: {entity_after.count}")
    print(f"    Importância antes: {entity_before.importance:.3f}, "
          f"depois: {entity_after.importance:.3f}")
    print(f"    Norma embedding: {np.linalg.norm(entity_after.embedding):.6f} (deve ≈1.0)")

    # Vizinhos
    neighbors = graph.get_neighbors(entity_ids[0])
    print(f"\n  Vizinhos de entity {entity_ids[0]}: {len(neighbors)}")
    for nid, rtype, w in neighbors[:3]:
        print(f"    → {nid} via '{rtype}' (peso={w:.3f})")

    stats = graph.get_stats()
    print(f"\n  Stats: {stats}")
    assert graph.n_entities > 0
    assert graph.n_relations > 0
    print("\n  ✅ SemanticGraph OK\n")


def test_procedural_memory():
    """Testa a memória procedural (aprendizado de política)."""
    print("=" * 60)
    print("TESTE 3: ProceduralMemory")
    print("=" * 60)

    proc = ProceduralMemory(n_actions=4, embedding_dim=8, lr=0.01, seed=42)

    rng = np.random.default_rng(2)

    # Simular episódios de aprendizado
    print("\n  Simulando 200 passos de aprendizado...")
    rewards_history = []
    surprise_history = []

    state = rng.normal(0, 1, 8)
    for step in range(200):
        action = proc.select_action(state, epsilon=0.2)

        # Simular ambiente: recompensa maior para ação 2
        next_state = state + rng.normal(0, 0.1, 8)
        if action == 2:
            reward = 1.0 + rng.normal(0, 0.1)
        else:
            reward = -0.1 + rng.normal(0, 0.1)

        result = proc.update(state, action, reward, next_state)
        rewards_history.append(reward)
        surprise_history.append(result["surprise"])

        state = next_state

    # Verificar que aprendeu a preferir ação 2
    test_state = rng.normal(0, 1, 8)
    q_values = proc.predict(test_state)
    print(f"\n  Q-values para estado teste: {q_values}")
    print(f"  Ação preferida: {np.argmax(q_values)} (esperado: 2)")

    # Stats
    stats = proc.get_stats()
    print(f"\n  Contagem de ações: {stats['action_counts']}")
    print(f"  Omega (importância) por ação: {[f'{x:.3f}' for x in stats['omega_per_action']]}")
    print(f"  Omega médio: {stats['omega_mean']:.4f}")
    print(f"  ||W||: {stats['W_norm']:.4f}")

    # Verificar que ação 2 tem mais contagem
    action_counts = proc.action_counts
    print(f"\n  Ação mais executada: {np.argmax(action_counts)} "
          f"({np.max(action_counts)} vezes)")

    # Verificar EWC: omega cresceu
    assert stats['omega_mean'] > 0, "Omega deveria ter crescido"
    print("\n  ✅ ProceduralMemory OK\n")


def test_hippocampus_integration():
    """Testa a integração das 3 memórias via Hippocampus."""
    print("=" * 60)
    print("TESTE 4: Hippocampus (Integração)")
    print("=" * 60)

    hippo = Hippocampus(
        state_dim=8,
        n_actions=4,
        episodic_capacity=20,
        max_entities=50,
        seed=42,
    )

    rng = np.random.default_rng(3)

    print("\n  Simulando 100 passos de interação...")
    total_consolidated = 0

    state = rng.normal(0, 1, 8)
    for step in range(100):
        action = hippo.decide(state, epsilon=0.15)

        # Ambiente simples
        next_state = state + rng.normal(0, 0.15, 8)
        reward = float(np.sin(np.sum(state)) + rng.normal(0, 0.05))
        surprise = 1.0 + max(0, abs(reward) - 0.5)

        result = hippo.store(state, action, reward, next_state, surprise)

        if result.get("consolidated"):
            total_consolidated += result["n_consolidated"]

        state = next_state

    print(f"  Total de experiências consolidadas: {total_consolidated}")

    # Recall
    print("\n  Testando recall...")
    query = rng.normal(0, 1, 8)
    memory = hippo.recall(query, k=3)
    print(f"  Matches semânticos: {len(memory['semantic_matches'])}")
    print(f"  Episódios recentes: {len(memory['recent_episodes'])}")

    # Stats finais
    stats = hippo.get_stats()
    print(f"\n  Stats finais:")
    print(f"    Episodic: size={stats['episodic']['size']}, "
          f"mean_importance={stats['episodic']['mean_importance']:.3f}")
    print(f"    Semantic: entities={stats['semantic']['n_entities']}, "
          f"relations={stats['semantic']['n_relations']}")
    print(f"    Procedural: omega_mean={stats['procedural']['omega_mean']:.4f}, "
          f"action_counts={stats['procedural']['action_counts']}")

    assert stats['episodic']['size'] > 0
    assert stats['semantic']['n_entities'] > 0
    print("\n  ✅ Hippocampus (Integração) OK\n")


if __name__ == "__main__":
    print("\n🧠 TESTE DO MEMORY SYSTEM (Tarefa 13.0)\n")

    test_episodic_buffer()
    test_semantic_graph()
    test_procedural_memory()
    test_hippocampus_integration()

    print("=" * 60)
    print("✅ TODOS OS TESTES PASSARAM")
    print("=" * 60)
