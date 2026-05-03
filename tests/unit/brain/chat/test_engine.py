"""Tests for brain.chat.engine — respond() + ChatResult."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider
from brain.chat.engine import ChatResult, respond
from brain.chat.session import create_session, reset_registry
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


@pytest.fixture(autouse=True)
def _reset_sessions():
    reset_registry()
    yield
    reset_registry()


@pytest.fixture()
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    # Write a minimal persona_config so _resolve_routing doesn't fail
    import json

    (d / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "noop"}),
        encoding="utf-8",
    )
    return d


@pytest.fixture()
def store() -> MemoryStore:
    s = MemoryStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture()
def hebbian() -> HebbianMatrix:
    h = HebbianMatrix(db_path=":memory:")
    yield h
    h.close()


@pytest.fixture()
def provider() -> FakeProvider:
    return FakeProvider()


# ── Basic respond() ───────────────────────────────────────────────────────────


def test_respond_with_no_session_creates_new_session(
    persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix, provider: FakeProvider
) -> None:
    result = respond(
        persona_dir,
        "hello",
        store=store,
        hebbian=hebbian,
        provider=provider,
        voice_md_override="# Nell\n\nHello.",
    )
    assert isinstance(result, ChatResult)
    assert result.session_id  # has a UUID


def test_respond_returns_content_and_session_id(
    persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix, provider: FakeProvider
) -> None:
    result = respond(
        persona_dir,
        "hello",
        store=store,
        hebbian=hebbian,
        provider=provider,
        voice_md_override="# Nell",
    )
    assert result.content.startswith("FAKE_CHAT")
    assert len(result.session_id) == 36


def test_respond_returns_turn_count(
    persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix, provider: FakeProvider
) -> None:
    result = respond(
        persona_dir,
        "hello",
        store=store,
        hebbian=hebbian,
        provider=provider,
        voice_md_override="# Nell",
    )
    assert result.turn == 1


def test_respond_returns_duration_ms(
    persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix, provider: FakeProvider
) -> None:
    result = respond(
        persona_dir,
        "hello",
        store=store,
        hebbian=hebbian,
        provider=provider,
        voice_md_override="# Nell",
    )
    assert result.duration_ms >= 0


# ── Session continuity ────────────────────────────────────────────────────────


def test_respond_appends_to_existing_session_history(
    persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix, provider: FakeProvider
) -> None:
    session = create_session("nell")
    respond(
        persona_dir,
        "first",
        store=store,
        hebbian=hebbian,
        provider=provider,
        session=session,
        voice_md_override="# Nell",
    )
    respond(
        persona_dir,
        "second",
        store=store,
        hebbian=hebbian,
        provider=provider,
        session=session,
        voice_md_override="# Nell",
    )
    assert session.turns == 2
    assert len(session.history) == 4  # 2 pairs


# ── Persistence ───────────────────────────────────────────────────────────────


def test_respond_persists_turn_via_ingest_turn(
    persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix, provider: FakeProvider
) -> None:
    """After respond(), the active_conversations buffer file should exist."""
    respond(
        persona_dir,
        "persist me",
        store=store,
        hebbian=hebbian,
        provider=provider,
        voice_md_override="# Nell",
    )
    active_dir = persona_dir / "active_conversations"
    assert active_dir.exists()
    buffer_files = list(active_dir.glob("*.jsonl"))
    assert len(buffer_files) == 1


def test_respond_catches_and_logs_ingest_persistence_error(
    persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix, provider: FakeProvider, caplog
) -> None:
    """Persistence failure must not break the response."""
    from unittest.mock import patch

    with (
        patch("brain.chat.engine.ingest_turn", side_effect=OSError("disk full")),
        caplog.at_level(logging.WARNING, logger="brain.chat.engine"),
    ):
        result = respond(
            persona_dir,
            "should still respond",
            store=store,
            hebbian=hebbian,
            provider=provider,
            voice_md_override="# Nell",
        )
    # Response still delivered, but persistence failure is visible to callers.
    assert result.content.startswith("FAKE_CHAT")
    assert result.metadata["persistence_ok"] is False
    assert result.metadata["persistence_error"] == "disk full"
    # Warning logged
    assert any(
        "buffer" in r.message.lower() or "failed" in r.message.lower() for r in caplog.records
    )


# ── Tool calls (passthrough) ──────────────────────────────────────────────────


def test_respond_returns_empty_tool_invocations_with_fake_provider(
    persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix, provider: FakeProvider
) -> None:
    """FakeProvider never synthesises tool calls."""
    result = respond(
        persona_dir,
        "hello",
        store=store,
        hebbian=hebbian,
        provider=provider,
        voice_md_override="# Nell",
    )
    assert result.tool_invocations == []
