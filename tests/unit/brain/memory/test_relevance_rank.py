"""rank_memories — relevance ranking, normalization, exclusion (P2 — C3/C4/C5).

Oracles are written to FAIL against the named known-bad:
  - C3 fails against RELEVANCE_RANKING_ENABLED=False (recency order → newer weak
    match first). Both orders are asserted so the discrimination is explicit.
  - C4 pins the exact weighted sum + single-candidate no-divide-by-zero.
  - C5 pins exclusion.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import brain.memory.relevance as rel
from brain.memory.relevance import (
    RECENCY_HALFLIFE_DAYS,
    W_IMP,
    W_MATCH,
    W_REC,
    rank_memories,
)
from brain.memory.store import Memory, MemoryStore


def _mk(store: MemoryStore, content: str, *, importance: float, age_days: float) -> Memory:
    m = Memory(
        id=str(uuid.uuid4()),
        content=content,
        memory_type="event",
        domain="d",
        created_at=datetime.now(UTC) - timedelta(days=age_days),
        importance=importance,
    )
    store.create(m)
    return m


def test_c3_ranking_is_relevance_not_recency(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    # Older, strong text-match + high importance.
    old_strong = _mk(store, "henryk henryk henryk", importance=9.0, age_days=60)
    # Newer, weak text-match (one mention buried in filler) + low importance.
    new_weak = _mk(
        store,
        "henryk mentioned only once amid a great many other unrelated filler words here",
        importance=2.0,
        age_days=0,
    )

    ranked = rank_memories(store, None, "henryk", limit=5)
    order = [m.id for m, _ in ranked]
    assert order[0] == old_strong.id, "blended relevance should rank the strong old match first"

    # Known-bad: with ranking disabled the fallback is recency-only → newer first.
    monkeypatch.setattr(rel, "RELEVANCE_RANKING_ENABLED", False)
    ranked_off = rank_memories(store, None, "henryk", limit=5)
    assert [m.id for m, _ in ranked_off][0] == new_weak.id


def test_c4_normalization_exact_and_single_candidate_no_div0() -> None:
    store = MemoryStore(":memory:")
    _mk(store, "henryk", importance=7.0, age_days=0)

    ranked = rank_memories(store, None, "henryk", limit=5)
    assert len(ranked) == 1  # single candidate — bm25 min==max, must not raise
    mem, score = ranked[0]
    assert score is not None

    # Single candidate → m_norm = 1.0; hebbian None → h = 0.
    age_days = (datetime.now(UTC) - mem.created_at).total_seconds() / 86400.0
    r_norm = 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)
    expected = W_MATCH * 1.0 + W_IMP * (7.0 / 10.0) + W_REC * r_norm
    assert abs(score - expected) < 1e-3


def test_c5_exclusion_set_is_honored() -> None:
    store = MemoryStore(":memory:")
    m1 = _mk(store, "henryk one", importance=5.0, age_days=0)
    m2 = _mk(store, "henryk two", importance=5.0, age_days=0)

    ranked = rank_memories(store, None, "henryk", limit=5, exclude_ids={m1.id})
    ids = {m.id for m, _ in ranked}
    assert m1.id not in ids
    assert m2.id in ids


def test_disabled_ranking_returns_none_sentinel(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    _mk(store, "henryk here", importance=5.0, age_days=0)
    monkeypatch.setattr(rel, "RELEVANCE_RANKING_ENABLED", False)
    ranked = rank_memories(store, None, "henryk", limit=5)
    assert ranked and all(score is None for _, score in ranked)


def test_disabled_ranking_excludes_before_limit_backfills(monkeypatch) -> None:
    """Flag-off fallback applies exclude_ids BEFORE the limit slice, so an
    excluded id inside the top-`limit` backfills from the next match instead of
    shrinking the result (stage-6 minor). Known-bad: excluding then slicing
    would return `limit - 1` items.
    """
    store = MemoryStore(":memory:")
    # 3 matches; search_text orders by created_at DESC → newest first.
    m_new = _mk(store, "henryk newest", importance=5.0, age_days=0)
    m_mid = _mk(store, "henryk middle", importance=5.0, age_days=1)
    m_old = _mk(store, "henryk oldest", importance=5.0, age_days=2)
    monkeypatch.setattr(rel, "RELEVANCE_RANKING_ENABLED", False)

    # limit=2, exclude the newest (which would occupy slot 1 of the top-2).
    ranked = rank_memories(store, None, "henryk", limit=2, exclude_ids={m_new.id})
    ids = [m.id for m, _ in ranked]
    assert m_new.id not in ids
    # Backfill: still 2 results (m_mid, m_old), not 1.
    assert ids == [m_mid.id, m_old.id]
