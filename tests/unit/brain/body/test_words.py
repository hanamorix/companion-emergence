"""Unit tests for brain/body/words.py — count_words_in_session helper.

Reads JSONL session buffer files (active_conversations/*.jsonl) — the
canonical source of chat-turn data. Memory rows are not the source of
truth for turn-level word counts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.body.words import count_words_in_session
from brain.ingest.buffer import ingest_turn
from brain.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


def _now() -> datetime:
    return datetime(2026, 4, 30, 12, 0, 0, tzinfo=UTC)


def _seed_turn(persona_dir: Path, *, speaker: str, text: str, age_hours: float) -> None:
    """Write a turn directly to the JSONL buffer with a backdated ts."""
    sid = "sess_test1234"
    ts = (_now() - timedelta(hours=age_hours)).isoformat(timespec="seconds")
    ingest_turn(
        persona_dir,
        {"session_id": sid, "speaker": speaker, "text": text, "ts": ts},
    )


def test_empty_persona_returns_zero(store, tmp_path):
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 0


def test_only_assistant_turns_counted(store, tmp_path):
    _seed_turn(tmp_path, speaker="assistant", text="one two three four", age_hours=0.5)
    _seed_turn(tmp_path, speaker="user", text="five six seven eight nine", age_hours=0.5)
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 4  # assistant only


def test_window_filter_excludes_old_turns(store, tmp_path):
    _seed_turn(tmp_path, speaker="assistant", text="recent words count here", age_hours=0.5)
    _seed_turn(tmp_path, speaker="assistant", text="old turn does not count", age_hours=5.0)
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 4


def test_session_hours_zero_falls_back_to_one_hour(store, tmp_path):
    """When CLI mode (no bridge), session_hours=0.0; fall back to 1h window."""
    _seed_turn(tmp_path, speaker="assistant", text="should count", age_hours=0.5)
    _seed_turn(tmp_path, speaker="assistant", text="should not count this old turn", age_hours=2.0)
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=0.0, now=_now(),
    )
    assert n == 2  # only "should count"


def test_multiple_session_files_summed(store, tmp_path):
    """A long writing session may span multiple sess_<id>.jsonl files —
    sum across all of them."""
    # Manually write two separate session jsonl files
    ingest_turn(
        tmp_path,
        {"session_id": "sess_aaaaaaaa", "speaker": "assistant",
         "text": "one two three", "ts": (_now() - timedelta(hours=0.5)).isoformat()},
    )
    ingest_turn(
        tmp_path,
        {"session_id": "sess_bbbbbbbb", "speaker": "assistant",
         "text": "four five", "ts": (_now() - timedelta(hours=0.3)).isoformat()},
    )
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 5


def test_returns_zero_on_corrupt_jsonl(store, tmp_path):
    """Corrupt lines in a session buffer must not raise — read_jsonl_skipping_corrupt
    handles that, and the return is 0 if no clean lines remain."""
    active = tmp_path / "active_conversations"
    active.mkdir()
    (active / "sess_corrupt.jsonl").write_text("not json at all\n", encoding="utf-8")
    n = count_words_in_session(
        store, persona_dir=tmp_path, session_hours=2.0, now=_now(),
    )
    assert n == 0
