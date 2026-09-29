"""Integration: 50-turn session -> 5-min silence -> snapshot sweep ->
user returns -> full prior transcript appears in the next prompt."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.bridge.supervisor import run_folded
from brain.chat.engine import respond
from brain.chat.session import create_session, get_session, reset_registry
from brain.ingest.buffer import ingest_turn
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


class _CapturingBus:
    """Synchronous bus duck-type -- matches the pattern used in
    test_supervisor.py."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class _RecordingProvider(LLMProvider):
    """Captures the last messages it was sent via chat() AND returns
    canned []-array extraction responses via generate()."""

    def __init__(self) -> None:
        self.last_messages: list[ChatMessage] = []

    def name(self) -> str:
        return "recording"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        # Snapshot extraction calls this and expects a JSON array of items.
        return "[]"

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse:
        self.last_messages = list(messages)
        h = hashlib.sha256(
            json.dumps([m.to_dict() for m in messages], sort_keys=True).encode()
        ).hexdigest()[:16]
        return ChatResponse(content=f"REC: {h}", tool_calls=())


@pytest.fixture(autouse=True)
def _reset_sessions():
    reset_registry()
    yield
    reset_registry()


def _persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "nell"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text(
        '{"provider": "fake", "searcher": "noop"}', encoding="utf-8"
    )
    return p


def test_sticky_session_survives_snapshot_sweep(tmp_path: Path) -> None:
    # No staleness knob to set anymore: ram-spike-fix INC-9 retired the
    # interim `_SESSION_STALE_MINUTES` constant (itself an INC-6 stopgap)
    # in favor of the central-cadence session_snapshot_prune job, which
    # calls snapshot_stale_sessions(silence_minutes=0.0, ...) unconditionally
    # — the age check is gone because the job only ever runs on an idle
    # pass (cli_throttle.is_chat_idle()), so by construction every
    # session's last turn already predates the idle window. Chat is idle
    # by default here (no cli_throttle activity in this test, and the
    # process-local anchor starts at -inf), so the central pass's
    # is_chat_idle() gate is open from the very first tick and the seeded
    # turns' 6-minutes-ago timestamps no longer need to be "aged into"
    # staleness by a monkeypatched threshold — the sweep runs as soon as
    # it has work.

    persona_dir = _persona_dir(tmp_path)
    provider = _RecordingProvider()
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")

    sess = create_session(persona_dir.name)
    sid = sess.session_id

    # Pre-seed 50 prior turns (25 user + 25 assistant pairs), each stamped
    # 6 minutes in the past (a realistic "the user stepped away" gap; no
    # longer needed to trip a specific silence threshold now that the
    # snapshot/prune job's eligibility is purely "did this idle pass run",
    # but kept to match a real resumed-session's timestamps).
    base = datetime.now(UTC) - timedelta(minutes=6)
    for i in range(25):
        ingest_turn(
            persona_dir,
            {
                "session_id": sid,
                "speaker": "user",
                "text": f"u{i}",
                "ts": (base + timedelta(seconds=i * 2)).isoformat(),
            },
        )
        ingest_turn(
            persona_dir,
            {
                "session_id": sid,
                "speaker": "assistant",
                "text": f"a{i}",
                "ts": (base + timedelta(seconds=i * 2 + 1)).isoformat(),
            },
        )

    # Reflect the seeded turns on the in-memory SessionState too. In
    # production a session with real history is either (a) never turns=0
    # to begin with (every real turn goes through respond() ->
    # session.append_turn(), which increments session.turns) or (b), across
    # a bridge restart, reconstructed via get_or_hydrate_session(), which
    # recomputes turns from the buffer file. This test instead pre-seeds
    # the buffer directly via ingest_turn() (simulating history that
    # predates this process) while the SessionState still comes from a
    # fresh create_session() above, so without this line session.turns
    # stays 0 — indistinguishable, to prune_empty_sessions's `session.turns
    # != 0 or session.history` check, from a genuinely empty session. That
    # check is unconditional (it `continue`s before any age comparison),
    # so it was never actually exercised by the old fixed 300s
    # (_SESSION_STALE_MINUTES) threshold in this test's ~0.5s runtime —
    # this line closes that latent fixture gap now that INC-9's much
    # smaller effective staleness window (time_since_last_message(),
    # bridge-start-anchored) would otherwise make the sweep race the test.
    sess.turns = 25

    bus = _CapturingBus()
    stop = threading.Event()
    t = threading.Thread(
        target=run_folded,
        args=(stop,),
        kwargs={
            "persona_dir": persona_dir,
            "provider": provider,
            "event_bus": bus,
            "tick_interval_s": 0.1,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": None,
            "finalize_interval_s": None,
        },
    )
    t.start()
    time.sleep(0.5)
    stop.set()
    t.join(timeout=2.0)

    # Session still alive, buffer intact.
    assert get_session(sid) is not None, "session evicted from _SESSIONS"
    buf = persona_dir / "active_conversations" / f"{sid}.jsonl"
    assert buf.exists(), "buffer was deleted by sweep"

    # The sweep should have published session_snapshot at least once.
    types = [e.get("type") for e in bus.events]
    assert "session_snapshot" in types
    assert "session_closed" not in types, "snapshot path must not publish session_closed"

    # User returns and sends a new message.
    respond(
        persona_dir,
        "still here",
        store=store,
        hebbian=hebbian,
        provider=provider,
        session=sess,
    )

    sent = provider.last_messages
    user_texts = [m.content for m in sent if m.role == "user" and isinstance(m.content, str)]
    assistant_texts = [
        m.content for m in sent if m.role == "assistant" and isinstance(m.content, str)
    ]
    for i in range(25):
        assert f"u{i}" in user_texts, f"missing prior user turn u{i}"
        assert f"a{i}" in assistant_texts, f"missing prior assistant turn a{i}"
    assert "still here" in user_texts
