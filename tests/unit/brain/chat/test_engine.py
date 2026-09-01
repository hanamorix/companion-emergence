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
        patch(
            "brain.chat.engine.persist_turns_following_successor",
            side_effect=OSError("disk full"),
        ),
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
        self.last_options: dict[str, Any] | None = None

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
        self.last_options = dict(options) if options else None
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
    user_texts = [m.content for m in sent if m.role == "user" and isinstance(m.content, str)]
    assistant_texts = [
        m.content for m in sent if m.role == "assistant" and isinstance(m.content, str)
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


def test_respond_outbound_recall_block_rides_volatile_tail(
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    recording_provider: _RecordingProvider,
) -> None:
    """Phase 7.2 — the always-on verify slice still reaches the model.

    Post prompt-caching split (Option A+), per-turn volatile blocks no longer
    ride in the frozen system message — they are threaded to the provider as
    the stdin ``volatile_suffix`` (options) appended after history. Seeds an
    audit row inside the 24h ambient window, drives a full chat turn through a
    recording provider, and asserts the outbound-recall block ("Recent
    outbound" + the seeded subject) is present in the volatile suffix and is
    NOT in the (now frozen) system message.
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
    # The block now rides the stdin volatile tail, NOT the frozen system prompt.
    assert "Recent outbound" not in system_text
    options = recording_provider.last_options
    assert options is not None, "expected per-call options carrying the volatile suffix"
    suffix = options.get("volatile_suffix")
    assert isinstance(suffix, str) and suffix, "expected a volatile_suffix in options"
    assert "Recent outbound" in suffix
    assert "the kolinsky sable brushes" in suffix
    # And the clock relocation signal travels with it.
    assert options.get("include_block_clock") is False


# ---------------------------------------------------------------------------
# P0 image-tool-route — shared-file surfacing + volatile-drop fix
# ---------------------------------------------------------------------------


def test_respond_file_send_surfaces_path_in_user_text(
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    recording_provider: _RecordingProvider,
) -> None:
    """C7a — a shared file's resolved path (+ filename) appears in the assembled
    user turn text, alongside the raw typed text."""
    sha = "a" * 64
    respond(
        persona_dir,
        "look at this",
        store=store,
        hebbian=hebbian,
        provider=recording_provider,
        shared_files=[{"kind": "file", "sha": sha, "filename": "notes.txt"}],
        voice_md_override="# Nell",
    )
    sent = recording_provider.last_messages
    user_msgs = [m for m in sent if m.role == "user"]
    assert user_msgs, "expected a user message"
    last_user = user_msgs[-1].content_text()
    assert "the user shared a file" in last_user
    assert sha in last_user
    assert "notes.txt" in last_user
    assert "look at this" in last_user
    # Shown-able-to-fail: with no shared file, the path line is absent.
    respond(
        persona_dir,
        "plain text",
        store=store,
        hebbian=hebbian,
        provider=recording_provider,
        voice_md_override="# Nell",
    )
    plain = [m for m in recording_provider.last_messages if m.role == "user"][-1].content_text()
    assert "the user shared a file" not in plain


def test_respond_file_send_turn_then_text_turn_carries_volatile(
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    recording_provider: _RecordingProvider,
) -> None:
    """C5 (post-change) — after a file-send turn, a following text-only turn still
    carries its volatile_suffix.

    The volatile-drop bug: pre-change a turn carrying an image took the image
    fork (``build_system_message`` inline, ``volatile_suffix=None``) and the
    provider picked its transport from *replayed history*, so a later text turn
    silently dropped its volatile context. That fork is gone — every turn takes
    the static-system + volatile-suffix split. The pre-change reproduce ran the
    image transport, which no longer exists in-tree (removed by this change), so
    per the plan we assert the post-change invariant here; the pre-change repro
    is the deleted transport itself.
    """
    from brain.chat.session import create_session

    sess = create_session(persona_dir.name)
    # turn k — file-send
    respond(
        persona_dir,
        "here is a file",
        store=store,
        hebbian=hebbian,
        provider=recording_provider,
        session=sess,
        shared_files=[{"kind": "file", "sha": "b" * 64, "filename": "n.txt"}],
        voice_md_override="# Nell",
    )
    # turn k+1 — text only
    respond(
        persona_dir,
        "what did you think?",
        store=store,
        hebbian=hebbian,
        provider=recording_provider,
        session=sess,
        voice_md_override="# Nell",
    )
    options = recording_provider.last_options
    assert options is not None, "text turn after a file turn must carry per-call options"
    assert options.get("volatile_suffix") is not None, "volatile_suffix dropped after a file turn"
    assert options.get("include_block_clock") is False


def test_respond_text_only_static_system_is_byte_preserving(
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    recording_provider: _RecordingProvider,
) -> None:
    """C9 — a text-only turn's system message is byte-identical to
    ``build_static_system_message`` and volatile rides the stdin suffix."""
    from brain.chat.prompt import build_static_system_message

    respond(
        persona_dir,
        "hello there",
        store=store,
        hebbian=hebbian,
        provider=recording_provider,
        voice_md_override="# Nell",
    )
    sent = recording_provider.last_messages
    system_msgs = [m.content for m in sent if m.role == "system"]
    assert system_msgs, "expected a system message"
    sent_system = system_msgs[0]
    expected_static = build_static_system_message(persona_dir, voice_md="# Nell")
    assert sent_system == expected_static, "text-only static system message drifted from pre-change"
    options = recording_provider.last_options
    assert options is not None
    assert isinstance(options.get("volatile_suffix"), str)
    assert options.get("include_block_clock") is False


def test_respond_path_line_not_fed_to_salience_or_volatile(
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    recording_provider: _RecordingProvider,
    monkeypatch,
) -> None:
    """C14 — the surfaced ``[the user shared a file: ...]`` line does NOT reach
    ``assess_salience`` / ``build_volatile_context``; both see only the raw
    typed text (so a file-send turn's volatile content equals the same text
    without a file)."""
    import brain.chat.engine as engine_mod

    seen: dict[str, str] = {}
    real_salience = engine_mod.assess_salience
    real_volatile = engine_mod.build_volatile_context

    def spy_salience(user_input, **kw):
        seen["salience"] = user_input
        return real_salience(user_input, **kw)

    def spy_volatile(persona_dir_, *, user_input, **kw):
        seen["volatile"] = user_input
        return real_volatile(persona_dir_, user_input=user_input, **kw)

    monkeypatch.setattr(engine_mod, "assess_salience", spy_salience)
    monkeypatch.setattr(engine_mod, "build_volatile_context", spy_volatile)

    raw = "please look at it"
    respond(
        persona_dir,
        raw,
        store=store,
        hebbian=hebbian,
        provider=recording_provider,
        shared_files=[{"kind": "file", "sha": "c" * 64, "filename": "x.txt"}],
        voice_md_override="# Nell",
    )
    assert seen["salience"] == raw
    assert "the user shared a file" not in seen["salience"]
    assert seen["volatile"] == raw
    assert "the user shared a file" not in seen["volatile"]


def test_respond_replays_file_send_turn_from_buffer_as_text(
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    recording_provider: _RecordingProvider,
) -> None:
    """A prior file-send turn (persisted as plain text with its path line)
    replays as a plain-string user message — no ImageBlock reconstruction."""
    from brain.chat.session import create_session
    from brain.ingest.buffer import ingest_turn

    sess = create_session(persona_dir.name)
    ingest_turn(
        persona_dir,
        {
            "session_id": sess.session_id,
            "speaker": "user",
            "text": "look at this\n[the user shared a file: /p/files/abc]",
        },
    )
    respond(
        persona_dir,
        "and?",
        store=store,
        hebbian=hebbian,
        provider=recording_provider,
        session=sess,
        voice_md_override="# Nell",
    )
    sent = recording_provider.last_messages
    for m in sent:
        if m.role in ("user", "assistant"):
            assert isinstance(m.content, str), "every replayed turn must be plain string content"
    replayed = [m.content for m in sent if m.role == "user" and "look at this" in m.content]
    assert replayed, "the prior file-send turn was not replayed"
    assert "the user shared a file" in replayed[0]
