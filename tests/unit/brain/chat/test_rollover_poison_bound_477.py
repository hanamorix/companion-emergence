"""#477 (confirm review S1): a live session whose extraction keeps failing must not block its rollover forever.

Since the rollover reaps only after a COMPLETED extraction, a session with a persistent extraction failure (a transcript the model
cannot answer, a store fault) that is in daily use never got past `finalize_stale_sessions`' poison move (silent for 24 h, 3 recorded
failures), so its buffer grew without bound and every rollover attempt cost a model call. The same 3-failure threshold now applies at
the rollover: the unextracted span is set aside in `active_conversations/poison/` (kept, never silently lost), the cursor moves past it,
and the rollover proceeds. A deferral (login outage) or a busy pass never poisons.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.cli_throttle import ThrottleDeferred
from brain.chat.rollover import perform_rollover
from brain.ingest.buffer import (
    ingest_turn,
    list_active_sessions,
    read_backoff,
    read_cursor,
    read_session,
    write_backoff,
    write_cursor,
)
from brain.ingest.pipeline import extract_session_snapshot
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

NAME = "persona"
BASE = datetime.now(UTC) - timedelta(hours=3)


def _ts(i: int) -> str:
    return (BASE + timedelta(minutes=i)).isoformat(timespec="seconds")


def _persona(tmp_path: Path) -> Path:
    p = tmp_path / "persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text(json.dumps({"provider": "fake", "searcher": "fake"}), encoding="utf-8")
    return p


def _seed(persona: Path, sid: str, start: int, stop: int) -> None:
    for i in range(start, stop):
        ingest_turn(
            persona,
            {"session_id": sid, "speaker": "user" if i % 2 == 0 else "assistant", "text": f"turn {i}", "ts": _ts(i)},
        )


class _Provider:
    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls = 0

    def name(self):
        return "stub"

    def healthy(self):
        return True

    def complete(self, prompt):
        return ""

    def generate(self, prompt=None, *, system=None, **kw):
        prompt = prompt if prompt is not None else kw.get("prompt", "")
        if "extracting durable memories" in prompt:
            self.calls += 1
            raise self.exc
        return "I remember this."


@pytest.fixture
def world(tmp_path):
    persona = _persona(tmp_path)
    store = MemoryStore(persona / "memories.db")
    hebbian = HebbianMatrix(persona / "hebbian.db")
    yield persona, store, hebbian
    store.close()
    hebbian.close()


def _roll(world, provider):
    persona, store, hebbian = world
    return perform_rollover(persona, "sess_old", NAME, seed_mode="summary_only", provider=provider, store=store, hebbian=hebbian)


def _poisoned_texts(persona: Path) -> list[str]:
    path = persona / "active_conversations" / "poison" / "sess_old.span.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_a_session_in_daily_use_with_a_persistent_extraction_failure_ends_with_a_bounded_buffer(world):
    persona, store, hebbian = world
    _seed(persona, "sess_old", 0, 10)
    write_cursor(persona, "sess_old", _ts(3))  # turns 0..3 were extracted earlier
    provider = _Provider(RuntimeError("the model cannot answer this transcript"))
    for _ in range(3):  # the 5-minute sweep fails three times: the poison threshold
        extract_session_snapshot(persona, "sess_old", store=store, hebbian=hebbian, provider=provider)
    assert read_backoff(persona, "sess_old")["failures"] == 3
    _seed(persona, "sess_old", 10, 14)  # and the conversation goes on

    new_sid = _roll(world, provider)

    assert new_sid is not None
    assert "sess_old" not in list_active_sessions(persona)  # the old buffer is gone: bounded
    assert len(read_session(persona, new_sid)) <= 2  # the successor holds the summary, not the unextracted turns
    # the unextracted span (after the cursor, including the turns that arrived after the failures) is kept aside, not lost
    assert _poisoned_texts(persona) == [f"turn {i}" for i in range(4, 14)]
    assert read_backoff(persona, "sess_old") is None
    assert read_cursor(persona, "sess_old") is None


def test_below_the_threshold_the_rollover_still_waits_and_keeps_everything(world):
    persona, store, hebbian = world
    _seed(persona, "sess_old", 0, 10)
    write_cursor(persona, "sess_old", _ts(3))
    # one recorded failure; the rollover's own failing pass records the second: still below the threshold of 3
    write_backoff(persona, "sess_old", failures=1, first_failure_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat())

    assert _roll(world, _Provider(RuntimeError("down"))) is None
    assert read_backoff(persona, "sess_old")["failures"] == 2

    assert "sess_old" in list_active_sessions(persona)
    assert len(read_session(persona, "sess_old")) == 10
    assert _poisoned_texts(persona) == []


def test_a_login_outage_never_poisons_even_after_earlier_failures(world):
    persona, store, hebbian = world
    _seed(persona, "sess_old", 0, 10)
    write_cursor(persona, "sess_old", _ts(3))
    # three old failures, outside the 10-minute window, so the pass really calls the provider, which now defers
    write_backoff(persona, "sess_old", failures=3, first_failure_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat())
    provider = _Provider(ThrottleDeferred("login expired"))

    assert _roll(world, provider) is None

    assert provider.calls == 1
    assert "sess_old" in list_active_sessions(persona)
    assert _poisoned_texts(persona) == []


def test_a_busy_pass_never_poisons(world):
    from brain.ingest.buffer import extract_lock_path
    from brain.utils.file_lock import file_lock

    persona, store, hebbian = world
    _seed(persona, "sess_old", 0, 10)
    write_cursor(persona, "sess_old", _ts(3))
    write_backoff(persona, "sess_old", failures=3, first_failure_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat())
    with file_lock(extract_lock_path(persona, "sess_old")):
        assert _roll(world, _Provider(RuntimeError("down"))) is None
    assert _poisoned_texts(persona) == []
    assert "sess_old" in list_active_sessions(persona)


def test_if_the_span_cannot_be_set_aside_nothing_is_reaped(world, monkeypatch):
    persona, store, hebbian = world
    _seed(persona, "sess_old", 0, 10)
    write_cursor(persona, "sess_old", _ts(3))
    write_backoff(persona, "sess_old", failures=3, first_failure_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat())

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("brain.ingest.pipeline.append_jsonl", boom)
    assert _roll(world, _Provider(RuntimeError("down"))) is None
    assert "sess_old" in list_active_sessions(persona)
    assert len(read_session(persona, "sess_old")) == 10
