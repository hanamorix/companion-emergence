"""Tests for brain.chat.session — SessionState + module registry."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from brain.bridge.chat import ChatMessage
from brain.chat.session import (
    HISTORY_MAX_TURNS,
    all_sessions,
    create_session,
    get_session,
    reset_registry,
)


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
