"""recall.py — graveyard-augmented search per spec §5.

search_with_loss returns active/fading/lost partitioned into a
SearchResult so brain/chat/prompt._build_recall_block can render all
three buckets distinctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from brain.forgetting import graveyard
from brain.memory.store import Memory, MemoryStore


@dataclass(frozen=True)
class SearchResult:
    """Partitioned search results: active, fading, and lost memories."""

    active: list[Memory] = field(default_factory=list)
    fading: list[Memory] = field(default_factory=list)
    lost: list[dict] = field(default_factory=list)


def search_with_loss(
    persona_dir: Path,
    store: MemoryStore,
    query: str,
    *,
    limit: int = 5,
) -> SearchResult:
    """Partitioned search: active + fading via MemoryStore, lost via graveyard.

    Args:
        persona_dir: Path to the persona directory (for graveyard access).
        store: MemoryStore instance for active/fading memory queries.
        query: Search query string.
        limit: Maximum results per bucket.

    Returns:
        SearchResult with active, fading, and lost lists partitioned by state.
    """
    if not query:
        return SearchResult()

    rows = store.search_text(query, include_fading=True, limit=limit)
    active = [r for r in rows if r.state == "active"]
    fading = [r for r in rows if r.state == "fading"]
    lost = graveyard.search(persona_dir, query, limit=limit)

    return SearchResult(active=active, fading=fading, lost=lost)
