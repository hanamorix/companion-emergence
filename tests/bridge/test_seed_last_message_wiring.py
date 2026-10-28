"""ram-spike-fix S82 (C41): the REAL wiring — server.py's lifespan must call
cli_throttle.seed_last_message_from_active_conversations(persona_dir) against
the actual persona directory's active_conversations/ buffers, before the
supervisor thread starts, so is_chat_idle reflects on-disk history from the
very first check after a restart — not merely "process just started"."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from brain.bridge import cli_throttle
from brain.bridge.server import build_app


def _write_turn(persona_dir: Path, session_id: str, ts: str) -> None:
    d = persona_dir / "active_conversations"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{session_id}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"session_id": session_id, "speaker": "user", "text": "hi", "ts": ts}) + "\n")


def test_lifespan_seeds_not_idle_right_after_a_recent_saved_message(persona_dir: Path) -> None:
    cli_throttle.reset()
    recent_ts = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
    _write_turn(persona_dir, "sess-1", recent_ts)

    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        # The lifespan runs the seed synchronously on __enter__, before the
        # supervisor thread starts — a message saved 5s ago is well inside
        # the default 600s lull, so the bridge must NOT be idle yet.
        assert cli_throttle.is_chat_idle() is False

    cli_throttle.reset()


def test_lifespan_seeds_idle_when_saved_message_predates_the_lull(persona_dir: Path) -> None:
    cli_throttle.reset()
    old_ts = (datetime.now(UTC) - timedelta(seconds=700)).isoformat()  # > default 600s lull
    _write_turn(persona_dir, "sess-1", old_ts)

    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        assert cli_throttle.is_chat_idle() is True

    cli_throttle.reset()


def test_lifespan_with_no_saved_messages_is_idle(persona_dir: Path) -> None:
    cli_throttle.reset()
    # persona_dir fixture already creates an empty active_conversations/ dir.

    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        assert cli_throttle.is_chat_idle() is True

    cli_throttle.reset()
