"""Tests for brain.chat.engine — respond() + ChatResult."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pytest

from brain.bridge.chat import (
    ChatMessage as _ChatMessage,
)
from brain.bridge.chat import (
    ChatResponse as _ChatResponse,
)
from brain.bridge.provider import FakeProvider
from brain.bridge.provider import LLMProvider as _LLMProvider
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


# ---------------------------------------------------------------------------
# Buffer-driven prompt construction (Phase B sticky sessions)
# ---------------------------------------------------------------------------


class _RecordingProvider(_LLMProvider):
    """Like FakeProvider but records the messages list it was last sent."""

    def __init__(self) -> None:
        self.last_messages: list[_ChatMessage] = []

    def name(self) -> str:
        return "recording"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "GEN: ok"

    def chat(
        self,
        messages: list[_ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> _ChatResponse:
        self.last_messages = list(messages)
        h = hashlib.sha256(repr(messages).encode()).hexdigest()[:16]
        return _ChatResponse(content=f"RECORDED: {h}", tool_calls=())


@pytest.fixture()
def recording_provider() -> _RecordingProvider:
    return _RecordingProvider()


def test_respond_reads_prior_turns_from_buffer_not_history(
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    recording_provider: _RecordingProvider,
) -> None:
    """The prompt sent to the provider must contain prior turns read from
    the buffer file — NOT from session.history."""
    from brain.ingest.buffer import ingest_turn

    sess = create_session(persona_dir.name)
    # Pre-seed buffer with prior turns that are NOT in session.history.
    ingest_turn(
        persona_dir,
        {
            "session_id": sess.session_id,
            "speaker": "user",
            "text": "I love watercolour",
        },
    )
    ingest_turn(
        persona_dir,
        {
            "session_id": sess.session_id,
            "speaker": "assistant",
            "text": "tell me about the brushes",
        },
    )
    # session.history is empty for this session — proves the buffer is the source.

    respond(
        persona_dir,
        "the kolinsky sable",
        store=store,
        hebbian=hebbian,
        provider=recording_provider,
        session=sess,
        voice_md_override="# Nell",
    )

    sent = recording_provider.last_messages
    user_texts = [
        m.content for m in sent if m.role == "user" and isinstance(m.content, str)
    ]
    assistant_texts = [
        m.content
        for m in sent
        if m.role == "assistant" and isinstance(m.content, str)
    ]
    assert "I love watercolour" in user_texts
    assert "tell me about the brushes" in assistant_texts
    assert "the kolinsky sable" in user_texts


def test_respond_falls_back_to_history_when_buffer_read_fails(
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    recording_provider: _RecordingProvider,
    monkeypatch,
) -> None:
    sess = create_session(persona_dir.name)
    sess.append_turn("hi from history", "hi back")

    def boom(*a, **kw):
        raise OSError("disk gone")

    monkeypatch.setattr("brain.chat.engine.read_session", boom)

    respond(
        persona_dir,
        "next turn",
        store=store,
        hebbian=hebbian,
        provider=recording_provider,
        session=sess,
        voice_md_override="# Nell",
    )

    sent = recording_provider.last_messages
    contents = [m.content for m in sent if isinstance(m.content, str)]
    assert "hi from history" in contents
    assert "hi back" in contents


def test_respond_system_message_includes_outbound_recall_block(
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    recording_provider: _RecordingProvider,
) -> None:
    """Phase 7.2 — the always-on verify slice is injected into the system message.

    Seeds an audit row dated within the 24h ambient window, then drives a
    full chat turn through a recording provider and asserts the system
    message contains "Recent outbound" plus the seeded subject.
    """
    from datetime import UTC, datetime, timedelta

    from brain.initiate.audit import append_audit_row
    from brain.initiate.schemas import AuditRow

    # Use a recent ts so read_recent_audit's 24h window (anchored at
    # datetime.now(UTC)) includes it. One hour ago is comfortably inside.
    recent_ts = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    row = AuditRow(
        audit_id="ia_engine_001",
        candidate_id="ic_engine_001",
        ts=recent_ts,
        kind="message",
        subject="the kolinsky sable brushes",
        tone_rendered="the kolinsky sable brushes landed",
        decision="send_quiet",
        decision_reasoning="x",
        gate_check={"allowed": True, "reason": None},
        delivery=None,
    )
    row.record_transition("delivered", recent_ts)
    append_audit_row(persona_dir, row)

    respond(
        persona_dir,
        "hello",
        store=store,
        hebbian=hebbian,
        provider=recording_provider,
        voice_md_override="# Nell",
    )

    sent = recording_provider.last_messages
    system_msgs = [m.content for m in sent if m.role == "system"]
    assert system_msgs, "expected a system message"
    system_text = system_msgs[0]
    assert isinstance(system_text, str)
    assert "Recent outbound" in system_text
    assert "the kolinsky sable brushes" in system_text


def test_respond_replays_image_turn_from_buffer(
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    recording_provider: _RecordingProvider,
) -> None:
    """A prior user turn with image_shas is reconstructed as a content tuple."""
    from brain.bridge.chat import ImageBlock, TextBlock
    from brain.images import save_image_bytes
    from brain.ingest.buffer import ingest_turn

    # Store a real 1x1 PNG so media_type_for_sha can resolve it.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00"
        b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    record = save_image_bytes(persona_dir, png_bytes, "image/png")
    sha = record.sha

    sess = create_session(persona_dir.name)
    ingest_turn(
        persona_dir,
        {
            "session_id": sess.session_id,
            "speaker": "user",
            "text": "look at this",
            "image_shas": [sha],
        },
    )

    respond(
        persona_dir,
        "what do you think?",
        store=store,
        hebbian=hebbian,
        provider=recording_provider,
        session=sess,
        voice_md_override="# Nell",
    )

    sent = recording_provider.last_messages
    image_msgs = [
        m for m in sent if m.role == "user" and not isinstance(m.content, str)
    ]
    assert image_msgs, "expected at least one user msg with tuple content"
    # The replayed prior turn should be among them. The live turn ("what do you
    # think?") is text-only so it stays a string-content message.
    found = False
    for msg in image_msgs:
        blocks = list(msg.content)
        text_blocks = [b for b in blocks if isinstance(b, TextBlock)]
        image_blocks = [b for b in blocks if isinstance(b, ImageBlock)]
        if any(b.text == "look at this" for b in text_blocks) and any(
            b.image_sha == sha for b in image_blocks
        ):
            found = True
            break
    assert found, "image-bearing user turn was not replayed from buffer"
