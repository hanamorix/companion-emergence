"""Tests for the one-time historical emotion backfill.

TDD — one test at a time per tdd-guard.
"""

from __future__ import annotations

from pathlib import Path


def _make_store(tmp_path: Path):
    from brain.memory.store import MemoryStore
    return MemoryStore(str(tmp_path / "memories.db"), integrity_check=False)


def _create_memory(store, *, has_emotions: bool = False):
    from brain.memory.store import Memory

    emotions = {"loneliness": 5.0} if has_emotions else {}
    m = Memory.create_new(
        content="some content",
        memory_type="conversation",
        domain="us",
        emotions=emotions,
    )
    store.create(m)
    return m


# ---------------------------------------------------------------------------
# Step 1.1 — should_run returns True when emotion-less memories + no state
# ---------------------------------------------------------------------------

def test_should_run_returns_true_when_emotion_less_memories_and_no_state(tmp_path):
    """With emotion-less memories and no state file → should return True."""
    from brain.ingest.emotion_backfill import should_run_emotion_backfill

    store = _make_store(tmp_path)
    _create_memory(store, has_emotions=False)
    store.close()

    assert should_run_emotion_backfill(tmp_path) is True


def test_should_run_returns_false_when_state_complete(tmp_path):
    """With a complete state file → False regardless of memories."""
    import json

    from brain.ingest.emotion_backfill import should_run_emotion_backfill

    store = _make_store(tmp_path)
    _create_memory(store, has_emotions=False)
    store.close()

    state_file = tmp_path / "emotion_backfill_state.json"
    state_file.write_text(json.dumps({
        "status": "complete",
        "schema_version": "v1",
        "started_at": "2026-01-01T00:00:00Z",
        "total_memories": 1,
        "tagged_memories": 1,
        "last_cursor": "",
    }))

    assert should_run_emotion_backfill(tmp_path) is False


def test_should_run_returns_false_when_all_memories_have_emotions(tmp_path):
    """All active memories already have emotion vectors → False."""
    from brain.ingest.emotion_backfill import should_run_emotion_backfill

    store = _make_store(tmp_path)
    _create_memory(store, has_emotions=True)
    store.close()

    assert should_run_emotion_backfill(tmp_path) is False


def test_should_run_returns_false_when_no_memories(tmp_path):
    """Empty store → nothing to backfill → False."""
    from brain.ingest.emotion_backfill import should_run_emotion_backfill

    _make_store(tmp_path).close()

    assert should_run_emotion_backfill(tmp_path) is False


# ---------------------------------------------------------------------------
# Step 2 — run_emotion_backfill
# ---------------------------------------------------------------------------

def _stub_tagger(memory) -> dict:  # noqa: ANN001
    return {"loneliness": 7.0}


def test_run_tags_emotion_less_memories(tmp_path):
    """Stub tagger should apply emotions to emotion-less memories."""
    from brain.ingest.emotion_backfill import run_emotion_backfill
    from brain.memory.store import MemoryStore

    store = _make_store(tmp_path)
    m = _create_memory(store, has_emotions=False)
    store.close()

    run_emotion_backfill(tmp_path, tagger_fn=_stub_tagger, cap=50)

    store2 = MemoryStore(str(tmp_path / "memories.db"), integrity_check=False)
    updated = store2.get(m.id)
    store2.close()

    assert updated is not None
    assert updated.emotions == {"loneliness": 7.0}


def test_run_does_not_retag_emotion_bearing_memories(tmp_path):
    """Memories that already have emotions must not be overwritten."""
    from brain.ingest.emotion_backfill import run_emotion_backfill
    from brain.memory.store import MemoryStore

    store = _make_store(tmp_path)
    m = _create_memory(store, has_emotions=True)
    store.close()

    call_count = {"n": 0}

    def counting_tagger(memory):  # noqa: ANN001
        call_count["n"] += 1
        return {"loneliness": 7.0}

    run_emotion_backfill(tmp_path, tagger_fn=counting_tagger, cap=50)

    assert call_count["n"] == 0

    store2 = MemoryStore(str(tmp_path / "memories.db"), integrity_check=False)
    updated = store2.get(m.id)
    store2.close()
    # original emotions preserved
    assert updated.emotions == {"loneliness": 5.0}


def test_run_drops_unregistered_names_from_tagger(tmp_path):
    """Names not in the registered vocabulary must be dropped from the result."""
    from brain.ingest.emotion_backfill import run_emotion_backfill
    from brain.memory.store import MemoryStore

    store = _make_store(tmp_path)
    m = _create_memory(store, has_emotions=False)
    store.close()

    def bad_tagger(memory):  # noqa: ANN001
        return {"loneliness": 5.0, "FAKE_UNREGISTERED_XYZ": 9.0}

    run_emotion_backfill(tmp_path, tagger_fn=bad_tagger, cap=50)

    store2 = MemoryStore(str(tmp_path / "memories.db"), integrity_check=False)
    updated = store2.get(m.id)
    store2.close()

    assert "FAKE_UNREGISTERED_XYZ" not in updated.emotions
    assert updated.emotions.get("loneliness") == 5.0


def test_run_budget_cap_halts_and_cursor_resumes(tmp_path):
    """Cap=1 on 3 emotion-less memories → halts after 1; re-run with new day tags rest."""
    from datetime import UTC, datetime, timedelta

    from brain.ingest.emotion_backfill import run_emotion_backfill
    from brain.memory.store import MemoryStore

    store = _make_store(tmp_path)
    ids = [_create_memory(store, has_emotions=False).id for _ in range(3)]
    store.close()

    # First run — cap 1
    state1 = run_emotion_backfill(tmp_path, tagger_fn=_stub_tagger, cap=1)
    assert state1.status == "deferred_to_next_day"

    store2 = MemoryStore(str(tmp_path / "memories.db"), integrity_check=False)
    tagged_after_1 = sum(1 for mid in ids if store2.get(mid).emotions)
    store2.close()
    assert tagged_after_1 < 3

    # Second run on tomorrow — cap resets, processes remaining
    tomorrow = datetime.now(UTC) + timedelta(days=1)
    state2 = run_emotion_backfill(
        tmp_path, tagger_fn=_stub_tagger, cap=10, now_dt=tomorrow
    )

    store3 = MemoryStore(str(tmp_path / "memories.db"), integrity_check=False)
    tagged_after_2 = sum(1 for mid in ids if store3.get(mid).emotions)
    store3.close()

    assert tagged_after_2 == 3
    assert state2.status == "complete"


def test_run_rerun_after_complete_is_noop(tmp_path):
    """Re-running after status=complete should be a no-op (returns early)."""
    from brain.ingest.emotion_backfill import run_emotion_backfill

    store = _make_store(tmp_path)
    _create_memory(store, has_emotions=False)
    store.close()

    call_count = {"n": 0}

    def counting_tagger(memory):  # noqa: ANN001
        call_count["n"] += 1
        return {"loneliness": 7.0}

    run_emotion_backfill(tmp_path, tagger_fn=counting_tagger, cap=50)
    first_calls = call_count["n"]

    # Second call — should be a no-op
    run_emotion_backfill(tmp_path, tagger_fn=counting_tagger, cap=50)

    assert call_count["n"] == first_calls  # no additional calls


# ---------------------------------------------------------------------------
# Step 3 — supervisor wiring
# ---------------------------------------------------------------------------

def test_supervisor_calls_emotion_backfill_when_should_run_true(tmp_path):
    """Supervisor startup must call run_emotion_backfill when should_run returns True."""
    import threading
    from unittest.mock import patch

    from brain.bridge.events import EventBus
    from brain.bridge.provider import FakeProvider
    from brain.bridge.supervisor import run_folded

    persona_dir = tmp_path / "test-persona"
    persona_dir.mkdir()
    (persona_dir / "active_conversations").mkdir()
    (persona_dir / "persona_config.json").write_text(
        '{"provider": "fake", "searcher": "noop"}'
    )

    stop = threading.Event()
    stop.set()  # exit immediately after startup

    with patch("brain.bridge.supervisor._attunement_should_run_backfill", return_value=False), \
         patch("brain.bridge.supervisor._attunement_run_backfill"), \
         patch("brain.bridge.supervisor._attunement_should_run_supplementary_backfill", return_value=False), \
         patch("brain.bridge.supervisor._attunement_run_supplementary_backfill"), \
         patch("brain.bridge.supervisor._emotion_backfill_should_run") as mock_should, \
         patch("brain.bridge.supervisor._emotion_backfill_run") as mock_run:

        mock_should.return_value = True

        run_folded(
            stop,
            persona_dir=persona_dir,
            provider=FakeProvider(),
            event_bus=EventBus(),
            tick_interval_s=0.0,
            heartbeat_interval_s=None,
            soul_review_interval_s=None,
            finalize_interval_s=None,
            log_rotation_interval_s=None,
            initiate_review_interval_s=None,
            voice_reflection_interval_s=None,
        )

        mock_should.assert_called_once_with(persona_dir)
        mock_run.assert_called_once_with(persona_dir)


def test_supervisor_skips_emotion_backfill_when_should_run_false(tmp_path):
    """Supervisor must not call run_emotion_backfill when should_run returns False."""
    import threading
    from unittest.mock import patch

    from brain.bridge.events import EventBus
    from brain.bridge.provider import FakeProvider
    from brain.bridge.supervisor import run_folded

    persona_dir = tmp_path / "test-persona"
    persona_dir.mkdir()
    (persona_dir / "active_conversations").mkdir()
    (persona_dir / "persona_config.json").write_text(
        '{"provider": "fake", "searcher": "noop"}'
    )

    stop = threading.Event()
    stop.set()

    with patch("brain.bridge.supervisor._attunement_should_run_backfill", return_value=False), \
         patch("brain.bridge.supervisor._attunement_run_backfill"), \
         patch("brain.bridge.supervisor._attunement_should_run_supplementary_backfill", return_value=False), \
         patch("brain.bridge.supervisor._attunement_run_supplementary_backfill"), \
         patch("brain.bridge.supervisor._emotion_backfill_should_run") as mock_should, \
         patch("brain.bridge.supervisor._emotion_backfill_run") as mock_run:

        mock_should.return_value = False

        run_folded(
            stop,
            persona_dir=persona_dir,
            provider=FakeProvider(),
            event_bus=EventBus(),
            tick_interval_s=0.0,
            heartbeat_interval_s=None,
            soul_review_interval_s=None,
            finalize_interval_s=None,
            log_rotation_interval_s=None,
            initiate_review_interval_s=None,
            voice_reflection_interval_s=None,
        )

        mock_should.assert_called_once_with(persona_dir)
        mock_run.assert_not_called()
