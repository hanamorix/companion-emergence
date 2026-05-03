"""Combined memory search — semantic + emotional + temporal + spreading.

Each sub-query returns a ranked list. combined_search blends them with
simple weighted sum when multiple filters are provided. Domain filter
is applied as a pre-scoring restriction on the candidate pool.

Design per spec Section 4.1 (brain/memory/search.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from brain.memory.embeddings import EmbeddingCache, cosine_similarity
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


@dataclass
class MemorySearch:
    """Composed search over a MemoryStore + HebbianMatrix + EmbeddingCache."""

    store: MemoryStore
    hebbian: HebbianMatrix
    embeddings: EmbeddingCache

    def semantic_search(
        self, query: str, limit: int = 10, domain: str | None = None
    ) -> list[tuple[Memory, float]]:
        """Return (memory, similarity) ordered desc by cosine similarity of
        query vs each memory's content embedding.
        """
        query_vec = self.embeddings.get_or_compute(query)
        candidates = self._candidates(domain)
        scored: list[tuple[Memory, float]] = []
        for mem in candidates:
            mem_vec = self.embeddings.get_or_compute(mem.content)
            sim = cosine_similarity(query_vec, mem_vec)
            scored.append((mem, sim))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def emotional_search(
        self,
        emotions: dict[str, float],
        limit: int = 10,
        domain: str | None = None,
    ) -> list[Memory]:
        """Return memories ordered desc by dot-product overlap with `emotions`."""
        candidates = self._candidates(domain)
        scored: list[tuple[Memory, float]] = []
        for mem in candidates:
            score = sum(
                mem.emotions.get(name, 0.0) * query_val for name, query_val in emotions.items()
            )
            if score > 0:
                scored.append((mem, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [mem for mem, _ in scored[:limit]]

    def temporal_search(
        self,
        after: datetime | None = None,
        before: datetime | None = None,
        limit: int | None = None,
        domain: str | None = None,
    ) -> list[Memory]:
        """Return memories with created_at in (after, before] bounds."""
        candidates = self._candidates(domain)
        filtered = [
            mem
            for mem in candidates
            if (after is None or mem.created_at > after)
            and (before is None or mem.created_at <= before)
        ]
        filtered.sort(key=lambda m: m.created_at, reverse=True)
        return filtered[:limit] if limit is not None else filtered

    def spreading_search(
        self,
        seed_id: str,
        depth: int = 2,
        decay_per_hop: float = 0.5,
        limit: int = 20,
    ) -> list[tuple[Memory, float]]:
        """Return (memory, activation) for memories reached via spreading
        activation from `seed_id`. Seed itself excluded.
        """
        activation = self.hebbian.spreading_activation(
            [seed_id], depth=depth, decay_per_hop=decay_per_hop
        )
        activation.pop(seed_id, None)
        results: list[tuple[Memory, float]] = []
        for mid, act in sorted(activation.items(), key=lambda p: p[1], reverse=True):
            mem = self.store.get(mid)
            if mem is not None:
                results.append((mem, act))
                if len(results) >= limit:
                    break
        return results

    def combined_search(
        self,
        query: str | None = None,
        emotions: dict[str, float] | None = None,
        domain: str | None = None,
        seed_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[Memory, float]]:
        """Blend sub-queries with equal weight (1.0 each). Returns
        (memory, combined_score) ordered desc.

        Emotion scores are divided by 100 as a rough scale heuristic so
        they don't dwarf cosine similarity (0..1). Not principled —
        tuning expected in v1.1.

        Domain filter: applied pre-scoring for query/emotions via
        _candidates(); applied post-scoring for seed_id since the
        Hebbian graph is domain-agnostic.

        Returns [] if no filters are specified.
        """
        if not any((query, emotions, seed_id)):
            return []

        scores: dict[str, float] = {}
        ref_memory: dict[str, Memory] = {}

        if query is not None:
            for mem, sim in self.semantic_search(query, limit=limit * 2, domain=domain):
                scores[mem.id] = scores.get(mem.id, 0.0) + sim
                ref_memory[mem.id] = mem

        if emotions:
            for mem in self.emotional_search(emotions, limit=limit * 2, domain=domain):
                emo_score = sum(mem.emotions.get(name, 0.0) * v for name, v in emotions.items())
                scores[mem.id] = scores.get(mem.id, 0.0) + emo_score / 100.0  # normalised
                ref_memory[mem.id] = mem

        if seed_id is not None:
            for mem, act in self.spreading_search(
                seed_id, depth=2, decay_per_hop=0.5, limit=limit * 2
            ):
                # spreading_search has no domain parameter — the graph is
                # domain-agnostic — so post-filter here to honour the
                # domain contract documented at the module level.
                if domain is not None and mem.domain != domain:
                    continue
                scores[mem.id] = scores.get(mem.id, 0.0) + act
                ref_memory[mem.id] = mem

        ranked = sorted(
            ((ref_memory[mid], score) for mid, score in scores.items()),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:limit]

    def _candidates(self, domain: str | None) -> list[Memory]:
        """Candidate memories to score — filtered by domain if given."""
        if domain is not None:
            return self.store.list_by_domain(domain)
        # All active memories (unbounded; caller's limit governs output).
        # Note: for large stores this is O(N); later optimisation via
        # ANN index is scoped for v1.1.
        return self.store.list_active()
