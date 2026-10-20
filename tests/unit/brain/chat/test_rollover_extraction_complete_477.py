"""#477 (a) / #473 wiring (code review 3a, NB2 and NB4): a rollover reaps the old buffer only when its extraction COMPLETED.

`perform_rollover` ran a best-effort extraction, ignored what it returned, then folded the old session into its summary and deleted the
old buffer and cursor. Since #477 a pass that finds another extraction of the session running returns a "busy" report, and since #473
a refused call returns a "deferred" one; both read as "nothing to do" to the rollover, so turns appended after the running pass read
the buffer (or never extracted during a login outage) were reaped unextracted. The rollover now skips (returns None, buffer kept) unless
the report says the extraction completed, which also covers a failed or raising extraction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.cli_throttle import ThrottleDeferred
from brain.chat.rollover import perform_rollover
from brain.ingest.buffer import (
    extract_lock_path,
    ingest_turn,
    list_active_sessions,
    read_cursor,
    read_session,
    write_cursor,
)
from brain.ingest.types import IngestReport
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.utils.file_lock import file_lock

NAME = "persona"


def _persona(tmp_path: Path) -> Path:
    p = tmp_path / "persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text(json.dumps({"provider": "fake", "searcher": "fake"}), encoding="utf-8")
    return p


def _seed(persona: Path, sid: str) -> None:
    """Six turns; the extraction cursor sits at turn 2, as after an earlier snapshot: the fold works on the turns BEFORE the cursor,
    so a rollover that ignores a skipped extraction still folds, rolls and reaps (without a cursor the fold folds nothing, the
    rollover aborts on its own, and the check under test is invisible)."""
    base = datetime.now(UTC) - timedelta(hours=3)
    for i in range(6):
        ingest_turn(
            persona,
            {"session_id": sid, "speaker": "user" if i % 2 == 0 else "assistant", "text": f"turn {i}",
             "ts": (base + timedelta(minutes=i)).isoformat(timespec="seconds")},
        )
    write_cursor(persona, sid, (base + timedelta(minutes=2)).isoformat(timespec="seconds"))


class _Provider:
    def __init__(self, exc: Exception | None = None):
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
            if self.exc is not None:
                raise self.exc
            return "[]"
        return "I remember this."


@pytest.fixture
def world(tmp_path):
    persona = _persona(tmp_path)
    _seed(persona, "sess_old")
    store = MemoryStore(persona / "memories.db")
    hebbian = HebbianMatrix(persona / "hebbian.db")
    yield persona, store, hebbian
    store.close()
    hebbian.close()


def _roll(world, provider):
    persona, store, hebbian = world
    return perform_rollover(persona, "sess_old", NAME, seed_mode="summary_only", provider=provider, store=store, hebbian=hebbian)


def test_a_completed_extraction_rolls_over_as_before(world):
    persona, _, _ = world
    new_sid = _roll(world, _Provider())
    assert new_sid is not None
    assert "sess_old" not in list_active_sessions(persona)


def test_a_deferred_extraction_keeps_the_old_buffer(world):
    persona, _, _ = world
    provider = _Provider(ThrottleDeferred("login expired"))
    assert _roll(world, provider) is None
    assert provider.calls == 1
    assert "sess_old" in list_active_sessions(persona)
    assert len(read_session(persona, "sess_old")) == 6
    assert read_cursor(persona, "sess_old") is not None  # untouched by the refused pass


def test_a_busy_extraction_keeps_the_old_buffer_and_does_not_extract(world):
    persona, _, _ = world
    provider = _Provider()
    with file_lock(extract_lock_path(persona, "sess_old")):  # a sweep is extracting this session right now
        assert _roll(world, provider) is None
    assert provider.calls == 0
    assert "sess_old" in list_active_sessions(persona)
    assert len(read_session(persona, "sess_old")) == 6
    # once the sweep is done the rollover goes through
    assert _roll(world, provider) is not None


def test_a_raising_extraction_keeps_the_old_buffer(world, monkeypatch):
    persona, _, _ = world

    def boom(*a, **k):
        raise RuntimeError("extractor exploded")

    monkeypatch.setattr("brain.ingest.pipeline.extract_session_snapshot", boom)
    assert _roll(world, _Provider()) is None
    assert "sess_old" in list_active_sessions(persona)


def test_an_extraction_that_errored_keeps_the_old_buffer(world):
    persona, _, _ = world
    assert _roll(world, _Provider(RuntimeError("provider down"))) is None
    assert "sess_old" in list_active_sessions(persona)


def test_without_a_store_no_extraction_is_attempted_and_the_rollover_proceeds(world):
    persona, _, _ = world
    provider = _Provider(ThrottleDeferred("login expired"))
    new_sid = perform_rollover(persona, "sess_old", NAME, seed_mode="summary_only", provider=provider)
    assert provider.calls == 0
    assert new_sid is not None


def test_the_report_says_whether_the_extraction_completed():
    assert IngestReport(session_id="s").completed is True
    assert IngestReport(session_id="s", busy=True).completed is False
    assert IngestReport(session_id="s", deferred=True).completed is False
    assert IngestReport(session_id="s", errors=1).completed is False
