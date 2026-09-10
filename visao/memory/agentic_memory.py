"""
agentic_memory.py — Tarefa 18.0: A-MEM — Agentic Memory as Tools.

Implements a Zettelkasten-inspired agentic memory system where memories are
structured notes with dynamic linking, evolution, and tool-callable operations.

Key concepts from A-MEM (Xu et al., NeurIPS 2025, arXiv:2502.12110):
1. Structured notes — each memory has content, context, keywords, tags, links
2. Dynamic linking — new memories auto-link to similar historical ones
3. Memory evolution — integrating new memories can update existing representations
4. Agentic operations — store, retrieve, update, summarize, discard as tools

This module integrates with the existing Hippocampus (Tarefa 13.0) but adds
the agentic layer on top: memories become first-class callable tools.

References:
    - Xu et al. (2025): A-MEM: Agentic Memory for LLM Agents. NeurIPS 2025.
    - Zettelkasten method: structured note-taking with interconnections.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =============================================================
#  MEMORY NOTE
# =============================================================

@dataclass
class MemoryNote:
    """A single structured memory note (Zettelkasten-style).

    Attributes
    ----------
    id : str
        Unique identifier (hash-based).
    content : str
        The actual memory content (text).
    context : str
        Situational context (when/where/why this was stored).
    keywords : list[str]
        Extracted keywords for fast retrieval.
    tags : list[str]
        User-defined or auto-generated tags.
    embedding : np.ndarray
        Semantic embedding vector for similarity search.
    timestamp : float
        Creation time (Unix epoch).
    last_accessed : float
        Last access time (for LRU-like decay).
    importance : float
        Importance score (EWC-like, grows with access).
    links : list[str]
        IDs of linked memories (dynamic connections).
    version : int
        Number of times this note has been updated (evolution).
    history : list[dict]
        Evolution history: what changed and when.
    """

    id: str
    content: str
    context: str = ""
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(64))
    timestamp: float = 0.0
    last_accessed: float = 0.0
    importance: float = 1.0
    links: list[str] = field(default_factory=list)
    version: int = 1
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp
        if isinstance(self.embedding, list):
            self.embedding = np.asarray(self.embedding, dtype=np.float64)

    def _generate_id(self) -> str:
        """Generate unique ID from content hash."""
        h = hashlib.sha256(self.content.encode()).hexdigest()[:12]
        return f"note_{h}"

    def touch(self):
        """Update last accessed time."""
        self.last_accessed = time.time()

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": self.id,
            "content": self.content,
            "context": self.context,
            "keywords": self.keywords,
            "tags": self.tags,
            "embedding": self.embedding.tolist(),
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "importance": self.importance,
            "links": self.links,
            "version": self.version,
            "history": self.history,
        }


# =============================================================
#  EMBEDDING ENGINE (simple, no external deps)
# =============================================================

class HashEmbedder:
    """Simple hash-based text embedder.

    Uses character n-gram hashing to produce fixed-dim embeddings.
    No external models needed — fully deterministic and self-contained.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    ngram_range : tuple
        Range of n-gram sizes (min, max).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, dim: int = 64, ngram_range: tuple = (2, 4), seed: int = 0):
        self.dim = dim
        self.ngram_range = ngram_range
        self._rng = np.random.default_rng(seed)
        # Random projection matrix for combining n-gram hashes
        self._proj = self._rng.normal(0, 1.0 / np.sqrt(dim), (dim, 256))

    def _tokenize(self, text: str) -> list[str]:
        """Extract character n-grams from text."""
        text = text.lower().strip()
        # Also extract words as features
        words = re.findall(r'\b\w+\b', text)
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for word in words:
                padded = f"_{word}_"
                for i in range(len(padded) - n + 1):
                    ngrams.append(padded[i:i + n])
        return ngrams

    def embed(self, text: str) -> np.ndarray:
        """Compute embedding for a text string."""
        ngrams = self._tokenize(text)
        if not ngrams:
            return np.zeros(self.dim, dtype=np.float64)

        # Hash each n-gram to a bucket, accumulate
        buckets = np.zeros(256, dtype=np.float64)
        for ng in ngrams:
            h = int(hashlib.md5(ng.encode()).hexdigest(), 16) % 256
            buckets[h] += 1.0

        # Project to embedding dim
        emb = self._proj @ buckets
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-10:
            emb = emb / norm
        return emb

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# =============================================================
#  KEYWORD EXTRACTOR
# =============================================================

class KeywordExtractor:
    """Simple keyword extraction via TF and stopword removal.

    No external dependencies — uses a small built-in stopword list
    and term frequency for scoring.
    """

    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can',
        'this', 'that', 'these', 'those', 'it', 'its', 'not', 'no',
        'as', 'if', 'then', 'than', 'so', 'such', 'what', 'which',
        'who', 'whom', 'how', 'when', 'where', 'why', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'some', 'very',
        'just', 'about', 'also', 'into', 'over', 'after', 'before',
    }

    def extract(self, text: str, top_k: int = 10) -> list[str]:
        """Extract top-k keywords from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        # Count term frequencies (excluding stopwords)
        freq: dict[str, int] = {}
        for w in words:
            if w in self.STOPWORDS or len(w) < 3:
                continue
            freq[w] = freq.get(w, 0) + 1
        # Sort by frequency, then alphabetically
        sorted_kw = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return [kw for kw, _ in sorted_kw[:top_k]]


# =============================================================
#  AGENTIC MEMORY (A-MEM)
# =============================================================

class AgenticMemory:
    """A-MEM: Agentic Memory as Tools.

    Zettelkasten-inspired memory system with:
    - Structured notes (content, context, keywords, tags, links)
    - Dynamic linking (auto-connect similar memories)
    - Memory evolution (new memories update existing ones)
    - Tool-callable operations (store, retrieve, update, summarize, discard)

    Parameters
    ----------
    embedding_dim : int
        Dimension of semantic embeddings.
    similarity_threshold : float
        Cosine similarity threshold for auto-linking.
    max_notes : int
        Maximum number of notes (discards least important when full).
    importance_decay : float
        Temporal decay rate for importance (EWC-temporal).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        similarity_threshold: float = 0.7,
        max_notes: int = 1024,
        importance_decay: float = 0.0001,
        seed: int = 0,
    ):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.max_notes = max_notes
        self.importance_decay = importance_decay
        self._rng = np.random.default_rng(seed)

        self.embedder = HashEmbedder(dim=embedding_dim, seed=seed)
        self.kw_extractor = KeywordExtractor()

        self.notes: dict[str, MemoryNote] = {}
        self._next_seq = 0

    @property
    def size(self) -> int:
        return len(self.notes)

    # ── TOOL: STORE ────────────────────────────────────────────

    def store(
        self,
        content: str,
        context: str = "",
        tags: Optional[list[str]] = None,
        auto_link: bool = True,
    ) -> dict:
        """Store a new memory note.

        Generates structured attributes (embedding, keywords), optionally
        auto-links to similar existing notes, and triggers memory evolution.

        Parameters
        ----------
        content : str
            The memory content.
        context : str
            Situational context.
        tags : list[str], optional
            Tags to attach.
        auto_link : bool
            Whether to auto-link to similar notes.

        Returns
        -------
        dict with note_id, links_created, evolved_notes.
        """
        tags = tags or []

        # Create note
        note = MemoryNote(
            id="",
            content=content,
            context=context,
            tags=tags,
        )
        note.embedding = self.embedder.embed(content + " " + context)
        note.keywords = self.kw_extractor.extract(content + " " + context)

        # Check for near-duplicate (exact content match)
        existing_id = self._find_exact_duplicate(content)
        if existing_id:
            # Evolve existing note instead of creating duplicate
            return self.update(existing_id, content, new_context=context, new_tags=tags)

        # Auto-link to similar notes
        links_created = []
        if auto_link:
            links_created = self._auto_link(note)

        # Memory evolution: update linked notes' representations
        evolved = self._evolve_linked(note, links_created)

        # Store
        self.notes[note.id] = note
        self._next_seq += 1

        # Prune if over capacity
        if len(self.notes) > self.max_notes:
            self._prune()

        return {
            "note_id": note.id,
            "links_created": links_created,
            "evolved_notes": evolved,
            "n_notes": len(self.notes),
        }

    def _find_exact_duplicate(self, content: str) -> Optional[str]:
        """Find note with identical content."""
        content_norm = content.strip().lower()
        for nid, note in self.notes.items():
            if note.content.strip().lower() == content_norm:
                return nid
        return None

    def _auto_link(self, note: MemoryNote) -> list[str]:
        """Auto-link note to similar existing notes."""
        linked = []
        for other_id, other_note in self.notes.items():
            sim = self.embedder.similarity(note.embedding, other_note.embedding)
            if sim >= self.similarity_threshold:
                # Bidirectional link
                if other_id not in note.links:
                    note.links.append(other_id)
                if note.id not in other_note.links:
                    other_note.links.append(note_id := other_id)
                linked.append(other_id)
        return linked

    def _evolve_linked(self, new_note: MemoryNote, linked_ids: list[str]) -> list[str]:
        """Evolve linked notes based on new memory.

        When a new memory is added, linked notes may update their
        context/keywords to reflect the new information.
        """
        evolved = []
        for other_id in linked_ids:
            if other_id not in self.notes:
                continue
            other = self.notes[other_id]

            # Merge keywords from new note (union, keeping top)
            combined_kw = list(set(other.keywords + new_note.keywords))
            if len(combined_kw) > 10:
                # Keep top 10 by frequency in combined content
                combined_text = other.content + " " + new_note.content
                scored = []
                for kw in combined_kw:
                    count = combined_text.lower().count(kw)
                    scored.append((count, kw))
                scored.sort(reverse=True)
                combined_kw = [kw for _, kw in scored[:10]]

            if set(combined_kw) != set(other.keywords):
                other.keywords = combined_kw
                other.version += 1
                other.history.append({
                    "event": "evolve_keywords",
                    "triggered_by": new_note.id,
                    "timestamp": time.time(),
                })
                evolved.append(other_id)

        return evolved

    # ── TOOL: RETRIEVE ─────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 5,
        tag_filter: Optional[list[str]] = None,
        keyword_filter: Optional[list[str]] = None,
    ) -> list[dict]:
        """Retrieve top-k most relevant notes for a query.

        Uses combined scoring: semantic similarity + keyword overlap + importance.

        Parameters
        ----------
        query : str
            Search query.
        k : int
            Number of results.
        tag_filter : list[str], optional
            Only return notes with these tags.
        keyword_filter : list[str], optional
            Only return notes with these keywords.

        Returns
        -------
        list of dicts with note info and relevance scores.
        """
        if not self.notes:
            return []

        query_emb = self.embedder.embed(query)
        query_kws = set(self.kw_extractor.extract(query))

        results = []
        for nid, note in self.notes.items():
            # Tag filter
            if tag_filter and not any(t in note.tags for t in tag_filter):
                continue
            # Keyword filter
            if keyword_filter and not any(kw in note.keywords for kw in keyword_filter):
                continue

            # Semantic similarity
            sem_sim = self.embedder.similarity(query_emb, note.embedding)

            # Keyword overlap (Jaccard-like)
            note_kws = set(note.keywords)
            if query_kws and note_kws:
                kw_overlap = len(query_kws & note_kws) / max(len(query_kws | note_kws), 1)
            else:
                kw_overlap = 0.0

            # Combined score (weighted)
            score = 0.5 * sem_sim + 0.3 * kw_overlap + 0.2 * min(note.importance / 5.0, 1.0)

            results.append({
                "note_id": nid,
                "content": note.content,
                "context": note.context,
                "tags": note.tags,
                "keywords": note.keywords,
                "importance": note.importance,
                "version": note.version,
                "links": note.links,
                "score": float(score),
                "semantic_sim": float(sem_sim),
                "keyword_overlap": float(kw_overlap),
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        # Update access metadata for top-k
        for r in results[:k]:
            if r["note_id"] in self.notes:
                self.notes[r["note_id"]].touch()
                self.notes[r["note_id"]].importance += 0.1

        return results[:k]

    # ── TOOL: UPDATE ───────────────────────────────────────────

    def update(
        self,
        note_id: str,
        new_content: Optional[str] = None,
        new_context: Optional[str] = None,
        new_tags: Optional[list[str]] = None,
    ) -> dict:
        """Update an existing memory note (memory evolution).

        Triggers re-embedding, keyword re-extraction, and re-linking.

        Parameters
        ----------
        note_id : str
            ID of note to update.
        new_content : str, optional
            New content (if None, keeps old).
        new_context : str, optional
            New context.
        new_tags : list[str], optional
            New tags (merged with old if merge_tags=True).

        Returns
        -------
        dict with update info.
        """
        if note_id not in self.notes:
            return {"error": f"Note {note_id} not found"}

        note = self.notes[note_id]
        old_content = note.content

        # Update fields
        if new_content is not None:
            note.content = new_content
        if new_context is not None:
            note.context = new_context
        if new_tags is not None:
            note.tags = list(set(note.tags + new_tags))

        # Re-embed
        note.embedding = self.embedder.embed(note.content + " " + note.context)
        note.keywords = self.kw_extractor.extract(note.content + " " + note.context)

        # Version + history
        note.version += 1
        note.history.append({
            "event": "update",
            "old_content": old_content,
            "timestamp": time.time(),
        })

        # Re-link (find new similar notes)
        new_links = self._auto_link(note)

        note.touch()

        return {
            "note_id": note_id,
            "version": note.version,
            "new_links": new_links,
            "content_changed": new_content is not None and new_content != old_content,
        }

    # ── TOOL: SUMMARIZE ────────────────────────────────────────

    def summarize(self, note_ids: Optional[list[str]] = None, query: Optional[str] = None) -> dict:
        """Summarize a set of notes or notes matching a query.

        Parameters
        ----------
        note_ids : list[str], optional
            Specific notes to summarize. If None, uses query.
        query : str, optional
            Query to find notes to summarize.

        Returns
        -------
        dict with summary info.
        """
        if note_ids:
            notes = [self.notes[nid] for nid in note_ids if nid in self.notes]
        elif query:
            retrieved = self.retrieve(query, k=10)
            notes = [self.notes[r["note_id"]] for r in retrieved if r["note_id"] in self.notes]
        else:
            # Summarize all
            notes = list(self.notes.values())

        if not notes:
            return {"summary": "", "n_notes": 0, "common_keywords": [], "common_tags": []}

        # Aggregate keywords and tags
        all_kw: dict[str, int] = {}
        all_tags: dict[str, int] = {}
        for note in notes:
            for kw in note.keywords:
                all_kw[kw] = all_kw.get(kw, 0) + 1
            for tag in note.tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1

        # Top common keywords/tags
        common_kw = sorted(all_kw.items(), key=lambda x: -x[1])[:10]
        common_tags = sorted(all_tags.items(), key=lambda x: -x[1])[:10]

        # Build summary
        summary_parts = [f"Summary of {len(notes)} notes:"]
        if common_kw:
            summary_parts.append(f"  Keywords: {', '.join(f'{k}({v})' for k, v in common_kw)}")
        if common_tags:
            summary_parts.append(f"  Tags: {', '.join(f'{t}({v})' for t, v in common_tags)}")
        summary_parts.append(f"  Total links: {sum(len(n.links) for n in notes)}")
        summary_parts.append(f"  Avg importance: {np.mean([n.importance for n in notes]):.3f}")

        return {
            "summary": "\n".join(summary_parts),
            "n_notes": len(notes),
            "common_keywords": [kw for kw, _ in common_kw],
            "common_tags": [tag for tag, _ in common_tags],
            "total_links": sum(len(n.links) for n in notes),
            "avg_importance": float(np.mean([n.importance for n in notes])),
        }

    # ── TOOL: DISCARD ──────────────────────────────────────────

    def discard(self, note_id: str) -> dict:
        """Discard (delete) a memory note.

        Also removes all links pointing to it from other notes.

        Parameters
        ----------
        note_id : str
            ID of note to discard.

        Returns
        -------
        dict with deletion info.
        """
        if note_id not in self.notes:
            return {"error": f"Note {note_id} not found"}

        # Remove links from other notes pointing to this one
        links_removed = 0
        for other in self.notes.values():
            if note_id in other.links:
                other.links.remove(note_id)
                links_removed += 1

        # Delete
        del self.notes[note_id]

        return {
            "note_id": note_id,
            "deleted": True,
            "links_removed": links_removed,
            "n_notes": len(self.notes),
        }

    # ── TOOL: LINK ─────────────────────────────────────────────

    def link(self, note_id_a: str, note_id_b: str, bidirectional: bool = True) -> dict:
        """Create a manual link between two notes.

        Parameters
        ----------
        note_id_a : str
            First note ID.
        note_id_b : str
            Second note ID.
        bidirectional : bool
            If True, link both directions.

        Returns
        -------
        dict with link info.
        """
        if note_id_a not in self.notes or note_id_b not in self.notes:
            return {"error": "One or both notes not found"}

        a = self.notes[note_id_a]
        b = self.notes[note_id_b]

        if note_id_b not in a.links:
            a.links.append(note_id_b)
        if bidirectional and note_id_a not in b.links:
            b.links.append(note_id_a)

        return {
            "linked": True,
            "a": note_id_a,
            "b": note_id_b,
            "bidirectional": bidirectional,
        }

    # ── UTILITY ────────────────────────────────────────────────

    def get_note(self, note_id: str) -> Optional[dict]:
        """Get a note by ID."""
        if note_id in self.notes:
            return self.notes[note_id].to_dict()
        return None

    def get_all_notes(self) -> list[dict]:
        """Get all notes as dicts."""
        return [note.to_dict() for note in self.notes.values()]

    def get_stats(self) -> dict:
        """Get system statistics."""
        if not self.notes:
            return {
                "n_notes": 0,
                "total_links": 0,
                "avg_links": 0.0,
                "avg_importance": 0.0,
                "avg_version": 1.0,
            }
        n = len(self.notes)
        total_links = sum(len(note.links) for note in self.notes.values())
        return {
            "n_notes": n,
            "total_links": total_links,
            "avg_links": total_links / n,
            "avg_importance": float(np.mean([note.importance for note in self.notes.values()])),
            "avg_version": float(np.mean([note.version for note in self.notes.values()])),
            "max_version": max(note.version for note in self.notes.values()),
        }

    def temporal_decay(self) -> None:
        """Apply temporal decay to all notes' importance."""
        for note in self.notes.values():
            note.importance *= np.exp(-self.importance_decay)

    def _prune(self) -> None:
        """Remove least important notes when over capacity."""
        if len(self.notes) <= self.max_notes:
            return
        # Sort by importance, remove bottom 10%
        n_remove = len(self.notes) - self.max_notes
        sorted_notes = sorted(self.notes.items(), key=lambda x: x[1].importance)
        for nid, _ in sorted_notes[:n_remove]:
            self.discard(nid)
