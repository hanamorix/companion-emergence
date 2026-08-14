"""relevance.py — cross-DB relevance ranking over the committed memory pool.

Blends four signals into a single 0..1-normalized score:

    score = W_MATCH·bm25 + W_IMP·importance + W_HEB·hebbian-activation + W_REC·recency

The blend spans two databases — ``memories.db`` (BM25 text-match, importance,
recency) and ``hebbian.db`` (spreading activation) — so it lives here rather
than in ``store.py`` (the pure ``memories.db`` layer, which has no handle on
``HebbianMatrix``). ``store.py`` owns only the FTS5 primitives.

P2 (memory relevance overhaul). The weight/decay consts are documented defaults;
tuning is deferred to #129. Also the single home for the snippet-then-read
render consts (imported by ``brain.chat.prompt`` and the search tools) so the
values live in one place.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

# --- ranking weights (applied to 0..1-normalized signals; tuning → #129) -----
# Reasoned ordering: BM25 text-match is the primary relevance signal; importance
# a strong secondary; hebbian "related-to-now" and recency are tie-breakers.
W_MATCH = 1.0
W_IMP = 0.5
W_HEB = 0.3
W_REC = 0.2

# recency decay: recency = 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)
RECENCY_HALFLIFE_DAYS = 30

# hebbian spreading-activation seed/traversal (spec §6.2). Seeded from this
# turn's top BM25 ids — the memories the current input is textually about.
HEB_SEED_COUNT = 5
HEB_DEPTH = 2
HEB_DECAY_PER_HOP = 0.5

# Candidate pool pulled from FTS before ranking — wider than the render limit so
# the ranker has something to reorder. A named const, sibling to
# `_RECALL_TOKEN_LIMIT`.
CANDIDATE_POOL = 50

# Flag: when False, falls back to recency-only `search_text` (score None).
RELEVANCE_RANKING_ENABLED = True

# --- snippet-then-read render consts (one home; imported by prompt + tools) ---
SNIPPET_COUNT = 8  # up from 5 — snippets are cheap
SNIPPET_MAX_CHARS = 140  # unchanged from the current recall truncation
FULL_INJECT_IMPORTANCE = 9.0  # a genuinely important memory is never gated behind a read-call
FULL_INJECT_MAX = 3  # at most this many full-injects (bounds volatile-tail growth)
SNIPPET_MODE_ENABLED = True  # when False, falls back to the current full-body render

# The recall-path hebbian open degrades to None (w_heb=0) on any open/query error.
HEB_OPEN_FAILSOFT = True


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _created_ts(mem: Memory) -> float:
    try:
        return mem.created_at.timestamp()
    except Exception:  # noqa: BLE001 — defensive; tie-break only
        return 0.0


def rank_memories(
    store: MemoryStore,
    hebbian: HebbianMatrix | None,
    query: str,
    *,
    limit: int,
    exclude_ids: Iterable[str] = frozenset(),
    active_only: bool = True,
    include_fading: bool = True,
) -> list[tuple[Memory, float | None]]:
    """Rank committed memories by blended relevance — bump-free, best→worst.

    Returns ``(memory, blended_score)`` pairs. The score is ``None`` when
    ranking is disabled (an *unranked* sentinel — callers then fall back to
    ``(-importance, -ts)``; distinct from a real 0.0 blended score).

    ``rank_memories`` NEVER opens a ``HebbianMatrix``: ``hebbian`` is supplied by
    the caller (or ``None``, which zeroes the ``w_heb`` term). This makes the
    "opens N× per turn" defect structurally impossible.
    """
    exclude = frozenset(exclude_ids)

    if not RELEVANCE_RANKING_ENABLED:
        # Recency-only fallback. None signals "unranked".
        fallback = store.search_text(
            query,
            active_only=active_only,
            include_fading=include_fading,
            bump=False,
            limit=limit,
        )
        return [(m, None) for m in fallback if m.id not in exclude]

    scored = store.search_fts_scored(
        query,
        active_only=active_only,
        include_fading=include_fading,
        bump=False,
        limit=CANDIDATE_POOL,
    )
    if not scored:
        return []

    # Hebbian spreading activation seeded from this turn's top BM25 ids.
    activation: dict[str, float] = {}
    if hebbian is not None:
        seeds = [m.id for m, _ in scored[:HEB_SEED_COUNT]]
        try:
            activation = hebbian.spreading_activation(seeds, HEB_DEPTH, HEB_DECAY_PER_HOP)
        except Exception:  # noqa: BLE001 — hebbian is a tie-breaker, never fatal
            activation = {}

    # bm25 min-max over the pool, inverted so higher = better match.
    bm_values = [bm for _, bm in scored]
    bm_min, bm_max = min(bm_values), max(bm_values)
    bm_span = bm_max - bm_min

    now = datetime.now(UTC)
    ranked: list[tuple[Memory, float]] = []
    for mem, bm in scored:
        if mem.id in exclude:
            continue
        # Single-candidate set (bm_span == 0) → 1.0, no divide-by-zero.
        m_norm = 1.0 if bm_span == 0 else (bm_max - bm) / bm_span
        i_norm = _clamp01(mem.importance / 10.0)
        h_norm = _clamp01(activation.get(mem.id, 0.0))
        age_days = max(0.0, (now - mem.created_at).total_seconds() / 86400.0)
        r_norm = 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)
        score = W_MATCH * m_norm + W_IMP * i_norm + W_HEB * h_norm + W_REC * r_norm
        ranked.append((mem, score))

    # P3: filter superseded here  (no-op today — forward-compat seam, spec §6.8)

    ranked.sort(key=lambda pair: (-pair[1], -_created_ts(pair[0])))
    return [(mem, score) for mem, score in ranked[:limit]]
