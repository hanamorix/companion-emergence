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


def test_freshness_falls_back_to_created_at_when_never_accessed():
    """Part B: a never-accessed memory derives freshness from created_at, not 0.0.

    Before the fix _freshness_input returned 0.0 whenever last_accessed_at was
    None — which is every freshly-created memory. That zeroed the one signal
    meant to protect recent memories. Now creation recency is the fallback
    anchor: a just-created memory is fresh; an old never-accessed one is less so.
    """
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    # last_tick_ts=None → lived hours == wall hours (deterministic).
    state = FeltTimeState(lived_age_hours=100.0)

    recent = _make_memory(created_iso=datetime.now(UTC).isoformat())
    old = _make_memory(created_iso=(datetime.now(UTC) - timedelta(days=60)).isoformat())
    store.create(recent)
    store.create(old)
    # Neither is ever get()-accessed → last_accessed_at stays None for both.

    s_recent = score(
        recent, store=store, hebbian=hebbian, felt_time_state=state, soul_linked_ids=set()
    )
    s_old = score(old, store=store, hebbian=hebbian, felt_time_state=state, soul_linked_ids=set())

    # Was exactly 0.0 before the fix — the structural defect.
    assert s_recent > 0.0
    # Creation recency is the only differing signal → recent scores higher.
    assert s_recent > s_old
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


def test_freshness_survives_recent_tick_with_seeded_first_tick():
    """#3 real fix (replaces the interim clamp canary): production shape — a
    recent last_tick + large lived_age — no longer collapses freshness, because
    the rate is computed against first_tick_ts (felt-time start), not last_tick."""
    from brain.forgetting.salience import _freshness_input

    now = datetime.now(UTC)
    state = FeltTimeState(
        lived_age_hours=500.0,
        last_tick_ts=(now - timedelta(minutes=5)).isoformat(),   # recent (the trap)
        first_tick_ts=(now - timedelta(hours=500.0)).isoformat(),  # felt-time began ~500h ago → rate ≈ 1
    )
    mem = _make_memory()
    object.__setattr__(mem, "created_at", now - timedelta(hours=48))
    object.__setattr__(mem, "last_accessed_at", None)

    fresh = _freshness_input(mem, state)
    assert fresh > 0.9, f"recent-tick no longer collapses freshness: {fresh}"


def test_lived_rate_clamped_to_max_when_first_tick_too_recent():
    """A first_tick_ts far more recent than lived_age implies (skew / bad seed)
    → rate clamps to MAX_LIVED_RATE, not an inflated value."""
    from brain.felt_time.lived_age import MAX_LIVED_RATE
    from brain.forgetting.salience import _lived_hours_since

    now = datetime.now(UTC)
    state = FeltTimeState(
        lived_age_hours=1000.0,
        last_tick_ts=now.isoformat(),
        first_tick_ts=(now - timedelta(hours=1)).isoformat(),  # 1000/1 → clamp
    )
    lived = _lived_hours_since(now - timedelta(hours=10), state)
    # 10 wall-hours × clamped rate MAX_LIVED_RATE.
    assert abs(lived - 10.0 * MAX_LIVED_RATE) < 0.01


def test_lived_rate_falls_back_to_one_when_first_tick_missing():
    from brain.forgetting.salience import _lived_hours_since

    now = datetime.now(UTC)
    state = FeltTimeState(lived_age_hours=500.0, last_tick_ts=now.isoformat(), first_tick_ts=None)
    lived = _lived_hours_since(now - timedelta(hours=10), state)
    assert abs(lived - 10.0) < 0.01  # rate 1.0


# ── B2 phase B: the emotion input fades through the engine ───────────────────


def test_emotion_input_uses_the_faded_value_and_peak_input_is_not_faded(tmp_path, emotion_engine_stub):
    from brain.forgetting.salience import _emotion_input, _peak_input

    m = _make_memory(emotions={"joy": 8.0, "love": 4.0}, created_iso=datetime.now(UTC).isoformat())
    emotion_engine_stub.faded = {(m.id, "joy"): 2.0, (m.id, "love"): 5.0}
    assert _emotion_input(m, emotion_engine_stub) == 0.5  # the faded 5.0 wins, over the stored 8.0
    assert _emotion_input(m, None) == 0.8  # no engine: the stored values
    object.__setattr__(m, "_stored_peak", 9.0)
    state = FeltTimeState(lived_age_hours=100.0)
    before = _peak_input(m, state)
    assert before > 0.0
    # The peak input reads the monotone peak field and never the engine: changing the faded values moves nothing.
    emotion_engine_stub.faded = {(m.id, "joy"): 0.0, (m.id, "love"): 0.0}
    assert _peak_input(m, state) == before


def test_emotion_input_keeps_its_empty_emotions_guard_with_an_engine(emotion_engine_stub):
    from brain.forgetting.salience import _emotion_input

    assert _emotion_input(_make_memory(emotions={}), emotion_engine_stub) == 0.0


def test_score_resolves_the_engine_from_a_file_store_and_an_explicit_engine_wins(tmp_path, emotion_engine_stub):
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    state = FeltTimeState(lived_age_hours=100.0)
    try:
        m = _make_memory(emotions={"joy": 10.0})
        store.create(m)
        emotion_engine_stub.faded = {(m.id, "joy"): 0.0}
        kwargs = {"store": store, "hebbian": hebbian, "felt_time_state": state, "soul_linked_ids": set()}
        faded = score(m, **kwargs)
        stored = score(m, emotion_engine=_StoredEngine(), **kwargs)
        assert faded < stored
    finally:
        store.close()
        hebbian.close()


class _StoredEngine:
    """An engine whose feelings are the stored ones (an explicit engine passed to `score`)."""

    def memory_feelings(self, memory):
        from brain.emotion.engine_utils import memory_feelings

        return memory_feelings(memory, None, None, None)


def test_compute_inputs_fades_the_emotion_input_through_the_engine(tmp_path, emotion_engine_stub):
    """`salience_inputs_at_drop.emotion` is the faded value, taken once at the write."""
    from brain.forgetting.salience import compute_inputs

    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    try:
        m = _make_memory(emotions={"joy": 10.0})
        store.create(m)
        emotion_engine_stub.faded = {(m.id, "joy"): 4.0}
        inputs = compute_inputs(
            m,
            store=store,
            hebbian=hebbian,
            felt_time_state=FeltTimeState(lived_age_hours=100.0),
            soul_linked_ids=set(),
        )
        assert inputs.emotion == 0.4
    finally:
        store.close()
        hebbian.close()
