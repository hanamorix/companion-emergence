"""Tests for brain.initiate.memory — episodic memory writes on send + transitions.

The plan's MagicMock-style test referenced a hypothetical `save` /
`get_by_external_id` / `update_by_id` API; the real `MemoryStore`
(brain/memory/store.py) exposes `create(Memory)`, `update(id, **fields)`,
and `list_by_type(...)`. These tests assert against the *real* API.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from brain.initiate.memory import (
    render_memory_for_state,
    update_initiate_memory_for_state,
    write_initiate_memory,
)
from brain.memory.store import Memory, MemoryStore


def test_render_memory_for_state_pending() -> None:
    text = render_memory_for_state(
        subject="the dream from this morning",
        message="the dream from this morning landed somewhere",
        state="pending",
    )
    assert "haven't sent it yet" in text or "pending" in text


def test_render_memory_for_state_delivered_not_read() -> None:
    text = render_memory_for_state(
        subject="the dream",
        message="the dream from this morning landed somewhere",
        state="delivered",
    )
    assert "hasn't seen it yet" in text or "not seen" in text


def test_render_memory_for_state_read() -> None:
    text = render_memory_for_state(subject="the dream", message="x", state="read")
    assert "seen it" in text


def test_render_memory_for_state_replied_explicit() -> None:
    text = render_memory_for_state(subject="the dream", message="x", state="replied_explicit")
    assert "answered" in text or "replied" in text


def test_render_memory_for_state_acknowledged_unclear() -> None:
    text = render_memory_for_state(subject="the dream", message="x", state="acknowledged_unclear")
    assert "can't tell" in text or "unclear" in text or "not sure" in text


def test_render_memory_for_state_unanswered() -> None:
    text = render_memory_for_state(subject="the dream", message="x", state="unanswered")
    assert "hasn't said anything" in text or "no answer" in text


def test_render_memory_for_state_dismissed() -> None:
    text = render_memory_for_state(subject="the dream", message="x", state="dismissed")
    assert "dismissed" in text or "closed" in text


def test_render_memory_truncates_long_message() -> None:
    """Long messages truncated to 240 chars in the quoted block."""
    long_msg = "a" * 500
    text = render_memory_for_state(subject="x", message=long_msg, state="delivered")
    # original message length 500 must not appear; truncation marker present
    assert "..." in text
    assert "a" * 500 not in text


def test_write_initiate_memory_uses_user_name_in_stored_content(tmp_path) -> None:
    """write_initiate_memory must store content containing user_name, not 'Hana'."""
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        write_initiate_memory(
            store,
            audit_id="ia_user",
            subject="the morning light",
            message="it felt like something worth sharing",
            state="delivered",
            ts="2026-05-28T10:00:00+00:00",
            user_name="Henryk",
        )
        rows = store.list_by_type("initiate_outbound")
        assert len(rows) == 1
        assert "Henryk" in rows[0].content
        assert "Hana" not in rows[0].content
    finally:
        store.close()


def test_write_initiate_memory_calls_store_create() -> None:
    """A mock store records a Memory dataclass on send."""
    mock_store = MagicMock()
    write_initiate_memory(
        mock_store,
        audit_id="ia_001",
        subject="the dream",
        message="x",
        state="delivered",
        ts="2026-05-11T14:47:09+00:00",
    )
    mock_store.create.assert_called_once()
    args, _kwargs = mock_store.create.call_args
    memory = args[0]
    assert isinstance(memory, Memory)
    assert memory.metadata["initiate_audit_id"] == "ia_001"
    assert "the dream" in memory.content
    assert memory.memory_type == "initiate_outbound"


def test_write_initiate_memory_swallows_store_failure(caplog) -> None:
    """A store exception is logged at warning, not raised."""
    mock_store = MagicMock()
    mock_store.create.side_effect = RuntimeError("disk on fire")
    # Must not raise.
    write_initiate_memory(
        mock_store,
        audit_id="ia_002",
        subject="x",
        message="y",
        state="delivered",
        ts="2026-05-11T14:47:09+00:00",
    )


def test_write_then_update_against_real_store(tmp_path) -> None:
    """End-to-end: write_initiate_memory + update_initiate_memory_for_state
    against a real in-memory MemoryStore re-renders the same row."""
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        write_initiate_memory(
            store,
            audit_id="ia_real",
            subject="the dream",
            message="the dream from this morning",
            state="delivered",
            ts="2026-05-11T14:47:09+00:00",
        )
        rows = store.list_by_type("initiate_outbound")
        assert len(rows) == 1
        original_id = rows[0].id
        assert "hasn't seen it yet" in rows[0].content

        update_initiate_memory_for_state(
            store,
            audit_id="ia_real",
            subject="the dream",
            message="the dream from this morning",
            new_state="read",
            ts="2026-05-11T18:00:00+00:00",
        )
        rows = store.list_by_type("initiate_outbound")
        assert len(rows) == 1, "transition must update in place, not duplicate"
        assert rows[0].id == original_id
        assert "seen it" in rows[0].content
        assert rows[0].metadata["initiate_state"] == "read"
    finally:
        store.close()


def test_render_memory_for_state_uses_user_name_not_hana() -> None:
    """render_memory_for_state must use user_name instead of the hardcoded 'Hana'.

    Regression: all seven _TEMPLATES hardcoded 'Hana', so every initiate
    memory written to the store contained the wrong user name for non-Hana
    users. When recalled via ambient search or search_memories, these entries
    actively misled the LLM about who it was talking to.
    """
    text = render_memory_for_state(
        subject="the dream",
        message="the dream landed somewhere",
        state="delivered",
        user_name="Henryk",
    )
    assert "Hana" not in text
    assert "Henryk" in text


def test_render_memory_for_state_falls_back_gracefully_without_user_name() -> None:
    """When user_name is omitted, render_memory_for_state must still work
    (default is 'my user', not 'Hana')."""
    text = render_memory_for_state(subject="the dream", message="x", state="delivered")
    assert "Hana" not in text
    assert isinstance(text, str)
    assert len(text) > 0


def test_update_with_no_prior_memory_writes_fresh(tmp_path) -> None:
    """If the audit_id has no prior memory row, update falls back to a write."""
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        update_initiate_memory_for_state(
            store,
            audit_id="ia_missing",
            subject="x",
            message="y",
            new_state="read",
            ts="2026-05-11T18:00:00+00:00",
        )
        rows = store.list_by_type("initiate_outbound")
        assert len(rows) == 1
        assert rows[0].metadata["initiate_audit_id"] == "ia_missing"
    finally:
        store.close()


# ── Task 4: pronoun slots in non-pending templates ──────────────────────────


def test_render_memory_they_them_agreement():
    from brain.pronouns import PRESETS

    text = render_memory_for_state(
        subject="the garden",
        message="hi",
        state="unanswered",
        user_name="Alex",
        pronouns=PRESETS["they/them"],
    )
    assert "They've seen it. They haven't said anything about it." in text


def test_render_memory_defaults_she_her():
    text = render_memory_for_state(subject="s", message="m", state="delivered", user_name="Alex")
    assert "She hasn't seen it yet." in text


def test_render_memory_he_him():
    from brain.pronouns import PRESETS

    text = render_memory_for_state(
        subject="s",
        message="m",
        state="replied_explicit",
        user_name="Alex",
        pronouns=PRESETS["he/him"],
    )
    assert "He answered." in text
