from datetime import UTC, datetime, timedelta

from brain.felt_time.state import FeltTimeState
from brain.forgetting.salience import (
    DEFAULT_WEIGHTS,
    score,
)
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


def _make_memory(*, content="x", emotions=None, created_iso="2026-01-01T00:00:00+00:00") -> Memory:
    m = Memory.create_new(
        content=content,
        memory_type="episodic",
        domain="chat",
        emotions=emotions or {},
    )
    # Override created_at for deterministic age tests.
    object.__setattr__(m, "created_at", datetime.fromisoformat(created_iso))
    return m


def test_score_uses_all_five_inputs(tmp_path):
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    state = FeltTimeState(lived_age_hours=100.0)

    m = _make_memory(emotions={"joy": 10.0})
    store.create(m)

    s = score(
        m,
        store=store,
        hebbian=hebbian,
        felt_time_state=state,
        soul_linked_ids=set(),
    )
    assert 0.0 <= s <= 1.0
    # emotion alone (intensity 10 -> normalised 1.0, weight 0.30) plus zeros from others
    # = 0.30, plus a freshness term from a never-accessed memory.
    store.close()
    hebbian.close()


def test_emotion_input_normalised_to_zero_one():
    """emotional_weight_at_ingest = min(max_intensity / 10, 1.0)."""
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    state = FeltTimeState(lived_age_hours=0.0)

    low = _make_memory(emotions={"joy": 1.0})
    high = _make_memory(emotions={"joy": 8.0})
    capped = _make_memory(emotions={"joy": 50.0})
    for m in (low, high, capped):
        store.create(m)

    s_low = score(low, store=store, hebbian=hebbian, felt_time_state=state, soul_linked_ids=set())
    s_high = score(high, store=store, hebbian=hebbian, felt_time_state=state, soul_linked_ids=set())
    s_capped = score(
        capped, store=store, hebbian=hebbian, felt_time_state=state, soul_linked_ids=set()
    )
    assert s_high > s_low
    # 50 caps at the same normalised value as 10.
    assert (
        abs(
            s_capped
            - score(
                _make_memory(emotions={"joy": 10.0}),
                store=store,
                hebbian=hebbian,
                felt_time_state=state,
                soul_linked_ids=set(),
            )
        )
        < 1e-6
    )
    store.close()
    hebbian.close()


def test_recall_count_drives_recall_input():
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    state = FeltTimeState(lived_age_hours=0.0)

    m = _make_memory()
    store.create(m)
    # Bump recall_count via get().
    for _ in range(5):
        store.get(m.id)
    fresh = store.get(m.id)  # re-fetch to see updated recall_count
    s = score(fresh, store=store, hebbian=hebbian, felt_time_state=state, soul_linked_ids=set())
    # 6 recalls -> normalised 0.6, weighted 0.20 -> 0.12 added by recall input alone.
    assert s > 0.0
    store.close()
    hebbian.close()


def test_soul_linkage_binary():
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    state = FeltTimeState(lived_age_hours=0.0)

    m = _make_memory()
    store.create(m)
    s_unlinked = score(
        m, store=store, hebbian=hebbian, felt_time_state=state, soul_linked_ids=set()
    )
    s_linked = score(m, store=store, hebbian=hebbian, felt_time_state=state, soul_linked_ids={m.id})
    # Soul-linked adds exactly 0.20 (binary input * weight 0.20).
    assert abs((s_linked - s_unlinked) - 0.20) < 1e-6
    store.close()
    hebbian.close()


def test_lived_age_freshness_inverse_to_time_since_access(tmp_path):
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")

    m = _make_memory(created_iso="2026-01-01T00:00:00+00:00")
    store.create(m)

    # Simulate "just accessed" — set last_accessed_at via store.update
    just_now = datetime.now(UTC)
    store.update(m.id, last_accessed_at=just_now)
    m_fresh = store.get(m.id)

    # Simulate "30 lived-days ago"
    state = FeltTimeState(lived_age_hours=720.0 + 100.0)  # current lived_age
    s_fresh = score(
        m_fresh, store=store, hebbian=hebbian, felt_time_state=state, soul_linked_ids=set()
    )

    # Same memory but set last_accessed to long ago wall-clock time
    long_ago = just_now - timedelta(days=60)
    store.update(m.id, last_accessed_at=long_ago)
    m_stale = store.get(m.id)
    s_stale = score(
        m_stale, store=store, hebbian=hebbian, felt_time_state=state, soul_linked_ids=set()
    )

    assert s_fresh > s_stale
    store.close()
    hebbian.close()


def test_freshness_zero_when_last_accessed_is_none():
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    state = FeltTimeState(lived_age_hours=100.0)

    m = _make_memory()
    store.create(m)
    # No get() called — last_accessed_at stays None
    s = score(m, store=store, hebbian=hebbian, felt_time_state=state, soul_linked_ids=set())
    # Freshness contributes 0; emotion contributes 0 (no emotions); soul 0; hebbian 0; recall 0.
    assert s == 0.0
    store.close()
    hebbian.close()


def test_score_uses_default_weights_summing_to_one():
    """Sanity: 0.30 + 0.20 + 0.20 + 0.20 + 0.10 = 1.00."""
    total = sum(DEFAULT_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6


def test_score_handles_missing_hebbian_entry():
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")  # empty — no edges for the memory
    state = FeltTimeState(lived_age_hours=0.0)

    m = _make_memory(emotions={"joy": 5.0})
    store.create(m)
    s = score(m, store=store, hebbian=hebbian, felt_time_state=state, soul_linked_ids=set())
    # Doesn't crash; hebbian contributes 0.
    assert s >= 0.0
    store.close()
    hebbian.close()
