"""#477: the ingest cursor and archive-marker writers.

`write_cursor` and `write_archive_marker` wrote through a temp file whose name was the same for every writer of a session, and neither
retried a transient Windows PermissionError on the replace. Two writers at once could publish an empty cursor, and `read_cursor` reads
an empty cursor as "extract from the beginning". They now go through the shared atomic text writer (unique temp per save, bounded retry,
fsync, cleanup), and one extraction of a session runs at a time (a second one skips: the first one advances the cursor).
"""

from __future__ import annotations

import json
import os

import pytest

import brain.health.attempt_heal as heal
from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.ingest.buffer import (
    extract_lock_path,
    ingest_turn,
    read_archive_marker,
    read_cursor,
    write_archive_marker,
    write_cursor,
)
from brain.ingest.pipeline import extract_session_snapshot
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.utils.file_lock import file_lock

TS = "2026-06-15T10:00:00+00:00"


def _record_replaces(monkeypatch):
    seen: list[tuple[str, str]] = []
    real = os.replace

    def spy(src, dst):
        seen.append((str(src), str(dst)))
        return real(src, dst)

    monkeypatch.setattr(heal.os, "replace", spy)
    return seen


@pytest.mark.parametrize("which", ["cursor", "marker"])
def test_two_saves_never_share_a_temp_name(tmp_path, monkeypatch, which):
    seen = _record_replaces(monkeypatch)
    (tmp_path / "active_conversations").mkdir()
    for i in range(2):
        if which == "cursor":
            write_cursor(tmp_path, "s1", TS)
        else:
            write_archive_marker(tmp_path, "s1", f"key{i}")
    temps = [src for src, _ in seen if src.endswith(".new")]
    assert len(temps) == 2 and temps[0] != temps[1]


@pytest.mark.parametrize("which", ["cursor", "marker"])
def test_a_transient_permission_error_on_the_replace_is_retried(tmp_path, monkeypatch, which):
    (tmp_path / "active_conversations").mkdir()
    real = os.replace
    failures = {"n": 2}

    def flaky(src, dst):
        if failures["n"] > 0 and str(dst).endswith((".cursor", ".archived")):
            failures["n"] -= 1
            raise PermissionError(32, "sharing violation")
        return real(src, dst)

    monkeypatch.setattr(heal.os, "replace", flaky)
    monkeypatch.setattr(heal.time, "sleep", lambda _s: None)
    if which == "cursor":
        write_cursor(tmp_path, "s1", TS)
        assert read_cursor(tmp_path, "s1") == TS
    else:
        write_archive_marker(tmp_path, "s1", "k1")
        assert read_archive_marker(tmp_path, "s1") == "k1"
    assert failures["n"] == 0


@pytest.mark.parametrize("which", ["cursor", "marker"])
def test_a_persistent_failure_raises_and_leaves_no_temp_behind(tmp_path, monkeypatch, which):
    (tmp_path / "active_conversations").mkdir()

    def always(src, dst):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(heal.os, "replace", always)
    monkeypatch.setattr(heal.time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError):
        if which == "cursor":
            write_cursor(tmp_path, "s1", TS)
        else:
            write_archive_marker(tmp_path, "s1", "k1")
    folder = "active_conversations" if which == "cursor" else "archived_conversations"
    assert sorted(p.name for p in (tmp_path / folder).iterdir()) == []


def test_a_cursor_write_never_publishes_an_empty_file(tmp_path):
    (tmp_path / "active_conversations").mkdir()
    write_cursor(tmp_path, "s1", TS)
    with pytest.raises(ValueError):
        write_cursor(tmp_path, "s1", "")
    assert read_cursor(tmp_path, "s1") == TS


class _MustNotRun(LLMProvider):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, *, system=None):
        self.calls += 1
        return "[]"

    def name(self):
        return "must-not-run"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:
        self.calls += 1
        return ChatResponse(content="[]", tool_calls=())


def test_a_second_extraction_of_a_session_in_flight_skips_without_error(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    ingest_turn(tmp_path, {"session_id": "s1", "speaker": "Hana", "text": "hello", "ts": TS})
    provider = _MustNotRun()
    with file_lock(extract_lock_path(tmp_path, "s1")):
        report = extract_session_snapshot(
            tmp_path, "s1", store=store, hebbian=HebbianMatrix(":memory:"), provider=provider
        )
    assert provider.calls == 0
    assert report.errors == 0 and report.committed == 0
    assert report.busy is True and report.completed is False  # explicit, not "nothing to do"
    assert read_cursor(tmp_path, "s1") is None


def test_an_extraction_runs_when_no_other_one_holds_the_session(tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    ingest_turn(tmp_path, {"session_id": "s1", "speaker": "Hana", "text": "hello", "ts": TS})
    provider = _MustNotRun()
    extract_session_snapshot(
        tmp_path, "s1", store=store, hebbian=HebbianMatrix(":memory:"), provider=provider
    )
    assert provider.calls == 1
    assert read_cursor(tmp_path, "s1") == TS


def test_the_lock_sidecar_is_not_taken_for_a_session_buffer(tmp_path):
    ingest_turn(tmp_path, {"session_id": "s1", "speaker": "Hana", "text": "hello", "ts": TS})
    with file_lock(extract_lock_path(tmp_path, "s1")):
        pass
    from brain.ingest.buffer import list_active_sessions

    assert list_active_sessions(tmp_path) == ["s1"]
    assert json.loads((tmp_path / "active_conversations" / "s1.jsonl").read_text().splitlines()[0])["text"] == "hello"
