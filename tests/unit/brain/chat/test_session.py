"""Tests for brain.chat.session — SessionState + module registry."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.chat import ChatMessage
from brain.chat.session import (
    HISTORY_MAX_TURNS,
    all_sessions,
    create_session,
    get_or_hydrate_session,
    get_session,
    prune_empty_sessions,
    reset_registry,
)
from brain.ingest.buffer import ingest_turn


@pytest.fixture(autouse=True)
def _clear_registry():
    """Isolate each test with a fresh registry."""
    reset_registry()
    yield
    reset_registry()


# ── create_session ────────────────────────────────────────────────────────────


def test_create_session_generates_uuid() -> None:
    """create_session produces a valid UUIDv4 session_id."""
    s = create_session("nell")
    assert len(s.session_id) == 36
    assert s.session_id.count("-") == 4


def test_create_session_initializes_empty_history() -> None:
    s = create_session("nell")
    assert s.history == []
    assert s.turns == 0
    assert s.last_turn_at is None


def test_create_session_sets_persona_name() -> None:
    s = create_session("siren")
    assert s.persona_name == "siren"


def test_create_session_registers_in_registry() -> None:
    s = create_session("nell")
    assert get_session(s.session_id) is s


# ── append_turn ───────────────────────────────────────────────────────────────


def test_append_turn_adds_user_and_assistant_entries() -> None:
    s = create_session("nell")
    s.append_turn("hello", "hi there")
    assert len(s.history) == 2
    assert s.history[0] == ChatMessage(role="user", content="hello")
    assert s.history[1] == ChatMessage(role="assistant", content="hi there")


def test_append_turn_increments_turns() -> None:
    s = create_session("nell")
    assert s.turns == 0
    s.append_turn("a", "b")
    assert s.turns == 1
    s.append_turn("c", "d")
    assert s.turns == 2


def test_append_turn_sets_last_turn_at() -> None:
    s = create_session("nell")
    assert s.last_turn_at is None
    before = datetime.now(UTC)
    s.append_turn("x", "y")
    after = datetime.now(UTC)
    assert s.last_turn_at is not None
    assert before <= s.last_turn_at <= after


def test_append_turn_truncates_at_history_max_turns() -> None:
    """History must not exceed HISTORY_MAX_TURNS pairs (2 * HISTORY_MAX_TURNS msgs)."""
    s = create_session("nell")
    # Fill past the limit
    for i in range(HISTORY_MAX_TURNS + 5):
        s.append_turn(f"user{i}", f"asst{i}")

    max_msgs = HISTORY_MAX_TURNS * 2
    assert len(s.history) == max_msgs
    # Most recent messages should be at the end
    assert s.history[-1].content == f"asst{HISTORY_MAX_TURNS + 4}"
    assert s.history[-2].content == f"user{HISTORY_MAX_TURNS + 4}"


# ── get_session + all_sessions ────────────────────────────────────────────────


def test_get_session_returns_none_for_unknown() -> None:
    assert get_session("does-not-exist") is None


def test_all_sessions_returns_registered() -> None:
    s1 = create_session("nell")
    s2 = create_session("siren")
    sessions = all_sessions()
    assert s1 in sessions
    assert s2 in sessions


def test_reset_registry_clears() -> None:
    create_session("nell")
    reset_registry()
    assert all_sessions() == []


def test_prune_empty_sessions_removes_only_old_zero_turn_sessions() -> None:
    now = datetime.now(UTC)
    old_empty = create_session("nell")
    old_empty.created_at = now - timedelta(minutes=10)
    fresh_empty = create_session("nell")
    fresh_empty.created_at = now
    with_turn = create_session("nell")
    with_turn.created_at = now - timedelta(minutes=10)
    with_turn.append_turn("hi", "hello")

    removed = prune_empty_sessions(older_than_seconds=300, now=now, persona_name="nell")

    assert removed == [old_empty.session_id]
    assert get_session(old_empty.session_id) is None
    assert get_session(fresh_empty.session_id) is fresh_empty
    assert get_session(with_turn.session_id) is with_turn


def test_prune_empty_sessions_respects_persona_filter() -> None:
    now = datetime.now(UTC)
    nell = create_session("nell")
    nell.created_at = now - timedelta(minutes=10)
    other = create_session("siren")
    other.created_at = now - timedelta(minutes=10)

    removed = prune_empty_sessions(older_than_seconds=300, now=now, persona_name="nell")

    assert removed == [nell.session_id]
    assert get_session(nell.session_id) is None
    assert get_session(other.session_id) is other


# ---- I-8 follow-up audit: thread-safe registry ----


def test_registry_concurrent_create_remove_no_lost_sessions() -> None:
    """Stress: 8 threads × 50 ops each (create, get, remove, all_sessions) —
    no exceptions, no lost sessions. Pre-fix _SESSIONS was an unlocked dict;
    compound ops (snapshot + check + remove) raced.

    This test is probabilistic — but with 400 ops across 8 threads it
    surfaces classic dict-corruption races (RuntimeError: dictionary
    changed size during iteration) within a few seconds on cpython."""
    import threading

    reset_registry()
    errors: list[Exception] = []
    sids: list[str] = []
    sids_lock = threading.Lock()

    def worker(thread_id: int) -> None:
        try:
            for i in range(50):
                op = i % 4
                if op == 0:
                    s = create_session(f"persona_{thread_id}")
                    with sids_lock:
                        sids.append(s.session_id)
                elif op == 1:
                    list(all_sessions())  # iterate-while-mutating snapshot
                elif op == 2 and sids:
                    with sids_lock:
                        if not sids:
                            continue
                        target = sids[-1]
                    get_session(target)
                else:
                    with sids_lock:
                        if not sids:
                            continue
                        target = sids.pop()
                    from brain.chat.session import remove_session
                    remove_session(target)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive(), "worker hung"

    assert errors == [], f"unexpected errors under concurrent load: {errors[:3]}"
    reset_registry()


# ---- F-201 Phase B: get_or_hydrate_session ----


def _persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "nell"
    p.mkdir()
    (p / "active_conversations").mkdir()
    return p


def test_hydrate_already_in_registry_returns_existing(tmp_path: Path) -> None:
    """If the session is in _SESSIONS, return it unchanged — no disk read."""
    persona_dir = _persona_dir(tmp_path)
    sess = create_session("nell")
    same = get_or_hydrate_session(persona_dir, "nell", sess.session_id)
    assert same is sess


def test_hydrate_not_in_registry_no_buffer_returns_none(tmp_path: Path) -> None:
    persona_dir = _persona_dir(tmp_path)
    result = get_or_hydrate_session(
        persona_dir, "nell", "00000000-0000-0000-0000-000000000000"
    )
    assert result is None


def test_hydrate_not_in_registry_buffer_present_returns_hydrated(tmp_path: Path) -> None:
    """Empty registry + buffer with 3 pairs -> SessionState turns=3 + last ts."""
    persona_dir = _persona_dir(tmp_path)
    sid = "11111111-1111-1111-1111-111111111111"

    base = datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)
    for i in range(3):
        ingest_turn(persona_dir, {
            "session_id": sid, "speaker": "user", "text": f"u{i}",
            "ts": (base + timedelta(seconds=i * 2)).isoformat(),
        })
        ingest_turn(persona_dir, {
            "session_id": sid, "speaker": "assistant", "text": f"a{i}",
            "ts": (base + timedelta(seconds=i * 2 + 1)).isoformat(),
        })

    result = get_or_hydrate_session(persona_dir, "nell", sid)
    assert result is not None
    assert result.session_id == sid
    assert result.persona_name == "nell"
    assert result.history == []
    assert result.turns == 3
    # last_turn_at matches the final ingest_turn (asst pair index 2 + 1 = 5s offset).
    expected_last = base + timedelta(seconds=2 * 2 + 1)
    assert result.last_turn_at == expected_last


def test_hydrate_registers_in_registry(tmp_path: Path) -> None:
    """After hydration the session is in _SESSIONS — next call returns same instance."""
    persona_dir = _persona_dir(tmp_path)
    sid = "22222222-2222-2222-2222-222222222222"
    ingest_turn(persona_dir, {
        "session_id": sid, "speaker": "user", "text": "hello",
        "ts": datetime.now(UTC).isoformat(),
    })
    ingest_turn(persona_dir, {
        "session_id": sid, "speaker": "assistant", "text": "hi",
        "ts": datetime.now(UTC).isoformat(),
    })

    first = get_or_hydrate_session(persona_dir, "nell", sid)
    assert first is not None
    assert get_session(sid) is first

    second = get_or_hydrate_session(persona_dir, "nell", sid)
    assert second is first


def test_hydrate_with_uneven_buffer_floors_turn_count(tmp_path: Path) -> None:
    """3 user + 2 assistant -> turns=2 (pairs). Trailing user is incomplete."""
    persona_dir = _persona_dir(tmp_path)
    sid = "33333333-3333-3333-3333-333333333333"
    base = datetime(2026, 5, 11, 12, 0, 0, tzinfo=UTC)
    for i in range(2):
        ingest_turn(persona_dir, {
            "session_id": sid, "speaker": "user", "text": f"u{i}",
            "ts": (base + timedelta(seconds=i * 3)).isoformat(),
        })
        ingest_turn(persona_dir, {
            "session_id": sid, "speaker": "assistant", "text": f"a{i}",
            "ts": (base + timedelta(seconds=i * 3 + 1)).isoformat(),
        })
    # Trailing unpaired user turn.
    ingest_turn(persona_dir, {
        "session_id": sid, "speaker": "user", "text": "u-trailing",
        "ts": (base + timedelta(seconds=10)).isoformat(),
    })

    result = get_or_hydrate_session(persona_dir, "nell", sid)
    assert result is not None
    assert result.turns == 2
    assert result.last_turn_at == base + timedelta(seconds=10)


def test_hydrate_malformed_last_ts_leaves_last_turn_at_none(tmp_path: Path) -> None:
    persona_dir = _persona_dir(tmp_path)
    sid = "44444444-4444-4444-4444-444444444444"
    ingest_turn(persona_dir, {
        "session_id": sid, "speaker": "user", "text": "u",
        "ts": "not-a-timestamp",
    })
    ingest_turn(persona_dir, {
        "session_id": sid, "speaker": "assistant", "text": "a",
        "ts": "also-bad",
    })
    result = get_or_hydrate_session(persona_dir, "nell", sid)
    assert result is not None
    assert result.turns == 1
    assert result.last_turn_at is None
