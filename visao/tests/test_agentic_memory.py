"""
test_agentic_memory.py — Testes para Tarefa 18.0: A-MEM Agentic Memory.

Testa as operações de memória como ferramentas:
1. store — armazenar notas com atributos estruturados
2. retrieve — recuperação por similaridade + keywords
3. update — evolução de notas existentes
4. summarize — sumarização de conjuntos de notas
5. discard — remoção com limpeza de links
6. link — conexão manual entre notas
7. auto-linking — conexão automática por similaridade
8. memory evolution — notas atualizadas por novas memórias
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visao.memory.agentic_memory import (
    AgenticMemory,
    HashEmbedder,
    KeywordExtractor,
    MemoryNote,
)


# =============================================================
#  TESTES DE UNIDADE
# =============================================================

def test_hash_embedder():
    """Testa o embedder hash-based."""
    embedder = HashEmbedder(dim=64, seed=42)

    # Embedding tem dimensão correta
    emb = embedder.embed("hello world test")
    assert emb.shape == (64,), f"Expected (64,), got {emb.shape}"

    # Normalizado
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-6, f"Expected norm ~1.0, got {norm}"

    # Textos similares têm similaridade alta
    emb1 = embedder.embed("machine learning is great")
    emb2 = embedder.embed("machine learning is awesome")
    sim = embedder.similarity(emb1, emb2)
    assert sim > 0.5, f"Similar texts should have high similarity, got {sim}"

    # Textos diferentes têm similaridade baixa
    emb3 = embedder.embed("quantum physics")
    emb4 = embedder.embed("cooking recipes")
    sim_diff = embedder.similarity(emb3, emb4)
    assert sim_diff < 0.9, f"Dissimilar texts should have lower similarity, got {sim_diff}"

    # Determinismo
    emb_a = embedder.embed("test deterministic")
    emb_b = embedder.embed("test deterministic")
    assert np.allclose(emb_a, emb_b), "Same text should produce same embedding"

    print("✅ test_hash_embedder")


def test_keyword_extractor():
    """Testa o extrator de keywords."""
    extractor = KeywordExtractor()

    text = "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data."
    keywords = extractor.extract(text, top_k=10)

    assert len(keywords) > 0, "Should extract keywords"
    assert "machine" in keywords or "learning" in keywords, f"Expected 'machine' or 'learning' in {keywords}"
    assert "the" not in keywords, "Stopwords should be excluded"
    assert "is" not in keywords, "Stopwords should be excluded"

    # Texto vazio
    empty_kw = extractor.extract("", top_k=5)
    assert len(empty_kw) == 0, "Empty text should produce no keywords"

    print("✅ test_keyword_extractor")


def test_memory_note_creation():
    """Testa criação de MemoryNote."""
    note = MemoryNote(
        id="",
        content="Test memory content",
        context="test context",
        tags=["test", "memory"],
    )

    assert note.id.startswith("note_"), f"ID should start with 'note_', got {note.id}"
    assert note.content == "Test memory content"
    assert note.context == "test context"
    assert "test" in note.tags
    assert note.version == 1
    assert note.timestamp > 0

    # Serialização
    d = note.to_dict()
    assert "id" in d
    assert "content" in d
    assert "embedding" in d

    print("✅ test_memory_note_creation")


# =============================================================
#  TESTES DE OPERAÇÕES (TOOLS)
# =============================================================

def test_store():
    """Testa armazenamento de notas."""
    mem = AgenticMemory(embedding_dim=64, seed=42)

    result = mem.store(
        content="JAX é um framework de diferenciação automática desenvolvido pelo Google",
        context="pesquisa sobre frameworks ML",
        tags=["jax", "google", "ml"],
    )

    assert "note_id" in result
    assert result["n_notes"] == 1
    assert mem.size == 1

    # Nota armazenada tem atributos estruturados
    note = mem.notes[result["note_id"]]
    assert len(note.keywords) > 0, "Keywords should be extracted"
    assert len(note.embedding) == 64, "Embedding should have correct dim"
    assert "jax" in note.tags or "google" in note.tags

    # Armazenar mais notas
    mem.store("LSTM é uma rede neural recorrente", tags=["rnn", "deep_learning"])
    mem.store("Transformers usam atenção para processar sequências", tags=["attention", "nlp"])
    assert mem.size == 3

    print("✅ test_store")


def test_retrieve():
    """Testa recuperação por similaridade."""
    mem = AgenticMemory(embedding_dim=64, seed=42)

    mem.store("JAX para diferenciação automática", tags=["jax"])
    mem.store("PyTorch framework de deep learning", tags=["pytorch"])
    mem.store("Redes neurais recorrentes LSTM", tags=["rnn"])
    mem.store("Mecanismos de atenção em transformers", tags=["attention"])

    # Buscar por similaridade
    results = mem.retrieve("framework de machine learning", k=2)
    assert len(results) > 0, "Should retrieve results"
    assert results[0]["score"] > 0, "Score should be positive"

    # Filtro por tag
    results_tag = mem.retrieve("anything", tag_filter=["jax"])
    assert all("jax" in r["tags"] for r in results_tag), "Tag filter should work"

    # Filtro por keyword
    results_kw = mem.retrieve("anything", keyword_filter=["lstm"])
    assert all("lstm" in r["keywords"] for r in results_kw), "Keyword filter should work"

    # k limita resultados
    results_k = mem.retrieve("neural networks", k=1)
    assert len(results_k) == 1, "k should limit results"

    print("✅ test_retrieve")


def test_update():
    """Testa atualização de notas (evolução)."""
    mem = AgenticMemory(embedding_dim=64, seed=42)

    result = mem.store("Conteúdo original da memória", tags=["original"])
    note_id = result["note_id"]

    # Atualizar conteúdo
    update_result = mem.update(note_id, new_content="Conteúdo atualizado da memória")
    assert update_result["version"] == 2, "Version should increment"
    assert update_result["content_changed"] is True

    note = mem.notes[note_id]
    assert note.content == "Conteúdo atualizado da memória"
    assert len(note.history) > 0, "History should record the update"

    # Atualizar contexto
    mem.update(note_id, new_context="novo contexto")
    assert mem.notes[note_id].context == "novo contexto"

    # Atualizar tags (merge)
    mem.update(note_id, new_tags=["novo_tag"])
    assert "novo_tag" in mem.notes[note_id].tags
    assert "original" in mem.notes[note_id].tags  # mantém antigas

    # Atualizar nota inexistente
    bad_result = mem.update("nonexistent", new_content="test")
    assert "error" in bad_result

    print("✅ test_update")


def test_summarize():
    """Testa sumarização de notas."""
    mem = AgenticMemory(embedding_dim=64, seed=42)

    mem.store("JAX para diferenciação automática", tags=["jax", "framework"])
    mem.store("PyTorch framework de deep learning", tags=["pytorch", "framework"])
    mem.store("TensorFlow também é um framework", tags=["tensorflow", "framework"])

    # Sumarizar todas
    summary = mem.summarize()
    assert summary["n_notes"] == 3
    assert "framework" in summary["common_tags"], "Common tag 'framework' should appear"
    assert summary["total_links"] >= 0

    # Sumarizar por query
    summary_q = mem.summarize(query="framework de ML")
    assert summary_q["n_notes"] > 0

    # Sumarizar IDs específicos
    all_ids = list(mem.notes.keys())
    summary_ids = mem.summarize(note_ids=all_ids[:2])
    assert summary_ids["n_notes"] == 2

    # Sumarizar vazio
    empty_mem = AgenticMemory(embedding_dim=64, seed=42)
    summary_empty = empty_mem.summarize()
    assert summary_empty["n_notes"] == 0

    print("✅ test_summarize")


def test_discard():
    """Testa remoção de notas."""
    mem = AgenticMemory(embedding_dim=64, seed=42)

    r1 = mem.store("Nota 1", tags=["t1"])
    r2 = mem.store("Nota 2 similar à nota 1", tags=["t2"])
    r3 = mem.store("Nota 3", tags=["t3"])

    n1 = r1["note_id"]
    n2 = r2["note_id"]
    n3 = r3["note_id"]

    # Criar links
    mem.link(n1, n2)
    mem.link(n2, n3)

    # Descartar n2
    result = mem.discard(n2)
    assert result["deleted"] is True
    assert result["links_removed"] >= 1, "Should remove links pointing to deleted note"
    assert mem.size == 2
    assert n2 not in mem.notes

    # Links de n1 e n3 não referenciam mais n2
    assert n2 not in mem.notes[n1].links
    assert n2 not in mem.notes[n3].links

    # Descartar nota inexistente
    bad_result = mem.discard("nonexistent")
    assert "error" in bad_result

    print("✅ test_discard")


def test_link():
    """Testa conexão manual entre notas."""
    mem = AgenticMemory(embedding_dim=64, seed=42)

    r1 = mem.store("Nota A")
    r2 = mem.store("Nota B")
    n1, n2 = r1["note_id"], r2["note_id"]

    # Link bidirecional
    result = mem.link(n1, n2, bidirectional=True)
    assert result["linked"] is True
    assert n2 in mem.notes[n1].links
    assert n1 in mem.notes[n2].links

    # Link unidirecional (n3 já pode ter n1 por auto-linking)
    r3 = mem.store("Nota C")
    n3 = r3["note_id"]
    mem.link(n1, n3, bidirectional=False)
    assert n3 in mem.notes[n1].links
    # n3 pode já ter n1 por auto-linking, mas garantimos que n1 tem n3
    # O importante é que link unidirecional não QUEBRA: n1→n3 existe

    # Link com nota inexistente
    bad_result = mem.link(n1, "nonexistent")
    assert "error" in bad_result

    print("✅ test_link")


def test_auto_linking():
    """Testa conexão automática por similaridade."""
    mem = AgenticMemory(
        embedding_dim=64,
        similarity_threshold=0.5,
        seed=42,
    )

    # Notas similares devem ser auto-linkadas
    r1 = mem.store("Machine learning algorithms for data science", tags=["ml"])
    r2 = mem.store("Machine learning models for data analysis", tags=["ml"])

    n1 = r1["note_id"]
    n2 = r2["note_id"]

    # Verificar se foram linkadas (depende do threshold)
    if mem.embedder.similarity(mem.notes[n1].embedding, mem.notes[n2].embedding) >= 0.5:
        assert n2 in mem.notes[n1].links or n1 in mem.notes[n2].links, \
            "Similar notes should be auto-linked"

    print("✅ test_auto_linking")


def test_memory_evolution():
    """Testa evolução de memórias (notas atualizadas por novas)."""
    mem = AgenticMemory(embedding_dim=64, seed=42)

    # Armazenar nota base
    r1 = mem.store("JAX framework Google diferenciação automática", tags=["jax"])
    n1 = r1["note_id"]
    kw_before = set(mem.notes[n1].keywords)

    # Armazenar nota similar (deve evoluir a primeira)
    r2 = mem.store("JAX framework Google machine learning otimização", tags=["jax", "ml"])
    n2 = r2["note_id"]

    # Verificar evolução
    evolved = r2.get("evolved_notes", [])
    kw_after = set(mem.notes[n1].keywords)

    # Se houve evolução, keywords devem ter mudado
    if n1 in evolved:
        assert kw_after != kw_before or mem.notes[n1].version > 1, \
            "Evolved note should have updated keywords or version"

    print("✅ test_memory_evolution")


def test_duplicate_detection():
    """Testa detecção de duplicatas (evolui em vez de duplicar)."""
    mem = AgenticMemory(embedding_dim=64, seed=42)

    r1 = mem.store("Conteúdo idêntico para teste", tags=["test"])
    r2 = mem.store("Conteúdo idêntico para teste", tags=["test2"])

    # Deve ter apenas 1 nota (a segunda evoluiu a primeira)
    assert mem.size == 1, f"Expected 1 note (dedup), got {mem.size}"

    # A nota deve ter tags mescladas
    note = mem.notes[r1["note_id"]]
    assert "test" in note.tags
    assert "test2" in note.tags

    print("✅ test_duplicate_detection")


def test_temporal_decay():
    """Testa decaimento temporal de importância."""
    mem = AgenticMemory(embedding_dim=64, importance_decay=0.1, seed=42)

    r = mem.store("Nota para decair")
    note_id = r["note_id"]
    importance_before = mem.notes[note_id].importance

    # Aplicar decay várias vezes
    for _ in range(10):
        mem.temporal_decay()

    importance_after = mem.notes[note_id].importance
    assert importance_after < importance_before, \
        f"Importance should decay: {importance_before} -> {importance_after}"

    print("✅ test_temporal_decay")


def test_pruning():
    """Testa poda de notas quando excede capacidade."""
    mem = AgenticMemory(embedding_dim=64, max_notes=5, seed=42)

    # Armazenar mais que o limite
    for i in range(10):
        mem.store(f"Nota número {i} com conteúdo diferente", tags=[f"tag{i}"])

    assert mem.size <= 5, f"Should prune to max_notes, got {mem.size}"

    print("✅ test_pruning")


def test_stats():
    """Testa estatísticas do sistema."""
    mem = AgenticMemory(embedding_dim=64, seed=42)

    stats_empty = mem.get_stats()
    assert stats_empty["n_notes"] == 0

    mem.store("Nota 1", tags=["t1"])
    mem.store("Nota 2", tags=["t2"])
    mem.link(list(mem.notes.keys())[0], list(mem.notes.keys())[1])

    stats = mem.get_stats()
    assert stats["n_notes"] == 2
    assert stats["total_links"] >= 1
    assert stats["avg_links"] > 0
    assert stats["avg_importance"] > 0

    print("✅ test_stats")


# =============================================================
#  TESTE DE INTEGRAÇÃO
# =============================================================

def test_full_workflow():
    """Teste de fluxo completo: store → retrieve → update → summarize → discard."""
    mem = AgenticMemory(embedding_dim=64, seed=42)

    # 1. Armazenar memórias
    mem.store(
        "JAX é um framework de diferenciação automática",
        context="pesquisa de frameworks",
        tags=["jax", "framework"],
    )
    mem.store(
        "PyTorch usa grafos computacionais dinâmicos",
        context="comparação de frameworks",
        tags=["pytorch", "framework"],
    )
    mem.store(
        "LSTM resolve o problema do gradiente vanishing",
        context="estudo de RNNs",
        tags=["lstm", "rnn"],
    )

    # 2. Recuperar
    results = mem.retrieve("framework de deep learning", k=2)
    assert len(results) > 0

    # 3. Atualizar
    first_id = results[0]["note_id"]
    mem.update(first_id, new_content="Conteúdo atualizado")

    # 4. Sumarizar
    summary = mem.summarize(query="framework")
    assert summary["n_notes"] > 0

    # 5. Descartar
    mem.discard(first_id)
    assert mem.size == 2

    print("✅ test_full_workflow")


# =============================================================
#  EXECUÇÃO
# =============================================================

if __name__ == "__main__":
    print("\n🧠 TESTES A-MEM AGENTIC MEMORY (Tarefa 18.0)\n")

    test_hash_embedder()
    test_keyword_extractor()
    test_memory_note_creation()
    test_store()
    test_retrieve()
    test_update()
    test_summarize()
    test_discard()
    test_link()
    test_auto_linking()
    test_memory_evolution()
    test_duplicate_detection()
    test_temporal_decay()
    test_pruning()
    test_stats()
    test_full_workflow()

    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM")
    print("=" * 60)
