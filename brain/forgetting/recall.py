"""recall.py — graveyard-augmented search per spec §5.

search_with_loss returns active/fading/lost partitioned into a
SearchResult so brain/chat/prompt._build_recall_block can render all
three buckets distinctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from brain.forgetting import graveyard
from brain.memory.relevance import rank_memories
from brain.memory.store import Memory, MemoryStore

if TYPE_CHECKING:
    from brain.memory.hebbian import HebbianMatrix


@dataclass(frozen=True)
class SearchResult:
    """Partitioned search results: active, fading, and lost memories.

    ``scores`` maps a surfaced memory id → its blended relevance score (P2).
    Unranked ids (ranking disabled) are omitted so the recall-block merge can
    fall back to ``(-importance, -ts)`` for them. ``active``/``fading`` stay
    ``list[Memory]`` — the field defaults empty, so existing shape holds.
    """

    active: list[Memory] = field(default_factory=list)
    fading: list[Memory] = field(default_factory=list)
    lost: list[dict] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


def search_with_loss(
    persona_dir: Path,
    store: MemoryStore,
    query: str,
    *,
    limit: int = 5,
    hebbian: HebbianMatrix | None = None,
) -> SearchResult:
    """Partitioned search: active + fading via ranked retrieval, lost via graveyard.

    Args:
        persona_dir: Path to the persona directory (for graveyard access).
        store: MemoryStore instance for active/fading memory queries.
        query: Search query string.
        limit: Maximum results per bucket.
        hebbian: Optional HebbianMatrix — **forwarded** to rank_memories (this
            function never opens one; the caller owns its lifecycle). None →
            the hebbian ranking term is 0.

    Returns:
        SearchResult with active, fading, and lost lists partitioned by state,
        plus a ``scores`` map of relevance scores (bump-free retrieval).
    """
    if not query:
        return SearchResult()

    ranked = rank_memories(store, hebbian, query, limit=limit, include_fading=True)
    active = [m for m, _ in ranked if m.state == "active"]
    fading = [m for m, _ in ranked if m.state == "fading"]
    scores = {m.id: s for m, s in ranked if s is not None}
    lost = graveyard.search(persona_dir, query, limit=limit)

    return SearchResult(active=active, fading=fading, lost=lost, scores=scores)
