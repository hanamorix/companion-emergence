"""Integration smoke for conversation compaction — exercises all THREE callers
through their REAL entry points (not the core directly), with a FakeProvider so
it runs in CI with zero network/CLI calls. This is the automated pre-flight for
the live (real `claude` CLI) C1b verification: it proves the wiring of

  1. the apply_budget BACKSTOP   — via a real engine.respond() turn over the cap,
  2. the daily CADENCE           — via the supervisor's _run_compaction_tick,
  3. the command-driven TOOL     — via the real tools.dispatch() path,

is intact and that a real chat turn still assembles + responds with a summary
block present. Heavier than the unit tests in tests/unit/.../test_compaction.py
(which call compact_conversation directly); this drives the framework wiring.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider
from brain.chat.engine import respond
from brain.chat.session import create_session, reset_registry
from brain.ingest.buffer import ingest_turn, read_archive, read_session, write_cursor
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.dispatch import dispatch


@pytest.fixture(autouse=True)
def _reset_sessions():
    reset_registry()
    yield
    reset_registry()


@pytest.fixture()
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    (d / "persona_config.json").write_text(
        json.dumps({"provider": "fake", "searcher": "noop"}), encoding="utf-8"
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


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _summaries(persona_dir: Path, sid: str) -> list[dict]:
    return [t for t in read_session(persona_dir, sid) if t.get("speaker") == "summary"]


def _seed_aged(persona_dir: Path, sid: str, n: int, *, chars: int = 100,
               hours_ago: float = 10) -> list[str]:
    base = datetime.now(UTC) - timedelta(hours=hours_ago)
    tss = []
    for i in range(n):
        ts = _iso(base + timedelta(minutes=i))
        tss.append(ts)
        ingest_turn(persona_dir, {"session_id": sid,
                                  "speaker": "user" if i % 2 == 0 else "assistant",
                                  "text": "x" * chars, "ts": ts})
    return tss


# ---------------------------------------------------------------- caller 3: TOOL
def test_tool_path_compacts_via_real_dispatch(persona_dir, store, hebbian) -> None:
    """The Kindled `compact_history` tool through the REAL dispatch() — proves the
    provider + session_id threading (F-2) works end-to-end."""
    sid = "sess_tool"
    tss = _seed_aged(persona_dir, sid, 100)
    write_cursor(persona_dir, sid, tss[-1])  # all extracted
    result = dispatch(
        "compact_history",
        {"age_hours": 1},
        store=store,
        hebbian=hebbian,
        persona_dir=persona_dir,
        provider=FakeProvider(),
        session_id=sid,
    )
    assert result["compacted"] is True
    assert result["compacted_n"] > 0
    assert len(_summaries(persona_dir, sid)) == 1
    assert read_archive(persona_dir, sid)  # originals archived


def test_tool_path_requires_provider_and_session(persona_dir, store, hebbian) -> None:
    """dispatch must reject the provider-needing tool when wiring is absent
    (guards the F-2 threading rather than silently no-op'ing)."""
    from brain.tools.dispatch import ToolDispatchError

    with pytest.raises(ToolDispatchError):
        dispatch("compact_history", {"age_hours": 1}, store=store, hebbian=hebbian,
                 persona_dir=persona_dir)  # no provider/session_id


# ------------------------------------------------------------- caller 2: CADENCE
def test_cadence_path_compacts_via_supervisor_tick(persona_dir) -> None:
    """The daily cadence through the supervisor's real _run_compaction_tick —
    iterates active sessions and folds each."""
    from brain.bridge.supervisor import _run_compaction_tick

    sid = "sess_cadence"
    # The cadence folds turns older than 24h — seed them >24h old (this also
    # confirms that 24h cutoff is actually wired).
    tss = _seed_aged(persona_dir, sid, 120, hours_ago=30)
    write_cursor(persona_dir, sid, tss[-1])
    _run_compaction_tick(persona_dir, FakeProvider())
    assert len(_summaries(persona_dir, sid)) == 1
    assert read_archive(persona_dir, sid)


def test_cadence_is_fault_isolated(persona_dir) -> None:
    """A failing session must not stop the sweep (the cadence swallows per-session
    errors). With no cursor, compaction no-ops cleanly — the tick must not raise."""
    from brain.bridge.supervisor import _run_compaction_tick

    _seed_aged(persona_dir, "sess_nocursor", 60)  # no cursor -> core no-ops
    _run_compaction_tick(persona_dir, FakeProvider())  # must not raise
    assert _summaries(persona_dir, "sess_nocursor") == []  # nothing compacted, no crash


# ------------------------------------------------------------ caller 1: BACKSTOP
def test_backstop_fires_in_real_respond_turn(persona_dir, store, hebbian) -> None:
    """A real engine.respond() turn whose buffer is over the 80K cap: apply_budget
    fires the persisted compaction on the hot path, the turn still returns content,
    and the buffer ends up with a summary block at the head."""
    session = create_session(persona_dir.name)
    sid = session.session_id
    # Seed a buffer that exceeds the 80K-token estimate (~320K chars): 100 turns
    # x ~3500 chars ≈ 350K chars ≈ 87K tokens.
    tss = _seed_aged(persona_dir, sid, 100, chars=3500)
    write_cursor(persona_dir, sid, tss[-1])  # all extracted -> compaction can fold

    result = respond(
        persona_dir,
        "what were we talking about?",
        store=store,
        hebbian=hebbian,
        provider=FakeProvider(),
        session=session,
        voice_md_override="# Nell\n\nYou are Nell.",
    )
    # The turn completed without error and returned content.
    assert result.content
    # The backstop persisted a summary block into the buffer (compaction fired).
    assert len(_summaries(persona_dir, sid)) == 1
    assert read_archive(persona_dir, sid)
    # The buffer now ends with this turn's freshly-appended user/assistant pair.
    after = read_session(persona_dir, sid)
    assert after[0]["speaker"] == "summary"      # summary hoisted to head
    assert after[-1]["speaker"] == "assistant"   # latest turn persisted last


def test_followup_turn_after_compaction_still_responds(persona_dir, store, hebbian) -> None:
    """After a compaction has left [summary, *recent] in the buffer, a subsequent
    real turn assembles the summary at the head and responds without error."""
    session = create_session(persona_dir.name)
    sid = session.session_id
    tss = _seed_aged(persona_dir, sid, 100, chars=3500)
    write_cursor(persona_dir, sid, tss[-1])
    respond(persona_dir, "first", store=store, hebbian=hebbian, provider=FakeProvider(),
            session=session, voice_md_override="# Nell")
    # Second turn — buffer already holds a summary block.
    result2 = respond(persona_dir, "second", store=store, hebbian=hebbian,
                      provider=FakeProvider(), session=session, voice_md_override="# Nell")
    assert result2.content
    assert len(_summaries(persona_dir, sid)) == 1  # still exactly one summary
